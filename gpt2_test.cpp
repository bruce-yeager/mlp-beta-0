// GPT-2 C++ Training Script - No PyTorch Dependencies
// Based on trainpy.py architecture with cgadimpl autodiff library
// WORKING VERSION - Using random initialized embeddings without custom backward hooks

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ad/ag_all.hpp"
#include "optim.hpp"

using namespace ag;
using namespace ag::nn;
using OwnTensor::Tensor;
using OwnTensor::Shape;
using OwnTensor::Dtype;
using OwnTensor::Device;
using OwnTensor::DeviceIndex;
using OwnTensor::TensorOptions;

namespace fs = std::filesystem;

// ============================================================================
// DataLoader Implementation
// ============================================================================

static std::vector<std::string> list_shards(const std::string& root,
                                            const std::string& split,
                                            const std::string& ext = ".bin") {
    std::vector<std::string> shards;
    for (const auto& e : fs::directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        std::string name = p.filename().string();
        if (p.extension() == ext && name.find(split) != std::string::npos) {
            shards.push_back(p.string());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

class UInt16ShardView {
public:
    UInt16ShardView() = default;
    ~UInt16ShardView() { close(); }

    UInt16ShardView(const UInt16ShardView&) = delete;
    UInt16ShardView& operator=(const UInt16ShardView&) = delete;

    void open(const std::string& path, size_t max_tokens) {
        close();
        path_ = path;

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("failed to open: " + path);

        struct stat st {};
        if (fstat(fd_, &st) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("failed to stat: " + path);
        }

        file_bytes_ = static_cast<size_t>(st.st_size);
        if (file_bytes_ % sizeof(u_int16_t) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("file size not divisible by 2 (uint16): " + path);
        }

        size_t total_tokens = file_bytes_ / 2;
        tokens_ = std::min(total_tokens, max_tokens);

        data_ = ::mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            ::close(fd_); fd_ = -1; data_ = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }
    }

    void close() {
        if (data_) {
            ::munmap(data_, file_bytes_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        file_bytes_ = 0;
        tokens_ = 0;
        path_.clear();
    }

    size_t size_tokens() const { return tokens_; }
    const std::string& path() const { return path_; }

    void read_block(size_t start, size_t count, std::vector<u_int16_t>& out) const {
        if (start + count > tokens_) throw std::out_of_range("read_block out of range");
        out.resize(count);
        const u_int16_t* p = reinterpret_cast<const u_int16_t*>(data_);
        for (size_t i = 0; i < count; ++i) out[i] = p[start + i];
    }

private:
    std::string path_;
    int fd_ = -1;
    void* data_ = nullptr;
    size_t file_bytes_ = 0;
    size_t tokens_ = 0;
};

struct Batch {
    int B = 0, T = 0;
    std::vector<u_int16_t> x;
    std::vector<u_int16_t> y;
    Tensor input;
    Tensor target;
};

class DataLoaderLite {
public:
    DataLoaderLite(int B, int T,
                   int rank, int world_size,
                   const std::string& split,
                   const std::string& data_root,
                   bool master_process = true,
                   size_t max_tokens_per_shard = 100000000)
        : B_(B), T_(T),
          rank_(rank), world_(world_size),
          split_(split), root_(data_root),
          master_(master_process),
          max_tokens_(max_tokens_per_shard) {

        if (!(split_ == "train" || split_ == "val"))
            throw std::runtime_error("split must be 'train' or 'val'");
        if (B_ <= 0 || T_ <= 0)
            throw std::runtime_error("B and T must be > 0");
        if (world_ <= 0 || rank_ < 0 || rank_ >= world_)
            throw std::runtime_error("invalid rank/world_size");

        shards_ = list_shards(root_, split_, ".bin");
        if (shards_.empty())
            throw std::runtime_error("no .bin shards found for split " + split_);

        if (master_) {
            std::cout << "found " << shards_.size() << " shards for split " << split_ << "\n";
        }

        reset();
    }

    void reset() {
        current_shard_ = 0;
        shard_.open(shards_[current_shard_], max_tokens_);
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
    }

    Batch next_batch() {
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        const size_t need = BT + 1;

        if (shard_.size_tokens() < need) {
            throw std::runtime_error("shard too small for one batch: " + shard_.path());
        }

        if (pos_ + need > shard_.size_tokens()) {
            advance_shard();
        }

        std::vector<u_int16_t> buf;
        shard_.read_block(pos_, need, buf);

        Batch b;
        b.B = B_; b.T = T_;
        b.x.resize(BT);
        b.y.resize(BT);

        for (size_t i = 0; i < BT; ++i) {
            b.x[i] = buf[i];
            b.y[i] = buf[i + 1];
        }

        b.input = Tensor(Shape{{static_cast<int64_t>(B_), static_cast<int64_t>(T_)}}, Dtype::UInt16, Device::CPU);
        std::copy(b.x.begin(), b.x.end(), b.input.data<uint16_t>());
        
        b.target = Tensor(Shape{{static_cast<int64_t>(B_), static_cast<int64_t>(T_)}}, Dtype::UInt16, Device::CPU);
        std::copy(b.y.begin(), b.y.end(), b.target.data<uint16_t>());

        pos_ += BT * static_cast<size_t>(world_);

        if (pos_ + (BT * static_cast<size_t>(world_) + 1) > shard_.size_tokens()) {
            advance_shard();
        }

        return b;
    }

private:
    void advance_shard() {
        current_shard_ = (current_shard_ + 1) % shards_.size();
        shard_.open(shards_[current_shard_], max_tokens_);
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        pos_ = BT * static_cast<size_t>(rank_);
    }

    int B_, T_;
    int rank_, world_;
    std::string split_, root_;
    bool master_;
    size_t max_tokens_;

    std::vector<std::string> shards_;
    size_t current_shard_ = 0;
    size_t pos_ = 0;

    UInt16ShardView shard_;
};

// ============================================================================
// Model Configuration
// ============================================================================

struct GPTConfig {
    int context_length = 256;  // Reduced for testing
    int vocab_size = 50304;
    int n_embd = 384;          // Reduced for testing
    int n_layers = 6;

    float max_lr = 5e-4f;
    float min_lr = 5e-5f;
    int warmup_steps = 100;
    int max_steps = 100;
};

// ============================================================================
// Module Implementations
// ============================================================================

class GELU : public Module {
public:
    Value operator()(Value x) override {
        return ag::gelu(x);
    }
};

class MLP : public Module {
public:
    MLP(GPTConfig config) {
        l_up = new Linear(config.n_embd, 4 * config.n_embd, Device::CPU);
        l_down = new Linear(4 * config.n_embd, config.n_embd, Device::CPU);
        gelu = new GELU();

        for(auto& p : l_up->parameters()) params_.push_back(p);
        for(auto& p : l_down->parameters()) params_.push_back(p);
    }

    ~MLP() {
        delete l_up;
        delete l_down;
        delete gelu;
    }

    Value operator()(Value x) override {
        x = (*l_up)(x);
        x = (*gelu)(x);
        x = (*l_down)(x);
        return x;
    }

    Linear* l_up;
    Linear* l_down;
    GELU* gelu;
};

class GPT : public Module {
public:
    GPT(GPTConfig config) : config(config) {
        // Initialize embedding weights on CPU - keep on CPU for lookup
        std::vector<float> wte_data(config.vocab_size * config.n_embd);
        std::vector<float> wpe_data(config.context_length * config.n_embd);
        
        std::mt19937 rng(1337);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for(auto& w : wte_data) w = dist(rng);
        for(auto& w : wpe_data) w = dist(rng);
        
        // Token embedding - keep on CPU (require_grad=false since we won't train embeddings for now)
        Tensor wte_t = Tensor(Shape{{static_cast<int64_t>(config.vocab_size), static_cast<int64_t>(config.n_embd)}}, 
                              Dtype::Float32, Device::CPU, false);  // no grad for simplicity
        std::copy(wte_data.begin(), wte_data.end(), wte_t.data<float>());
        wte = make_tensor(wte_t, "wte");
        // Don't add to params_ - embeddings stay on CPU
        
        // Positional embedding - keep on CPU
        Tensor wpe_t = Tensor(Shape{{static_cast<int64_t>(config.context_length), static_cast<int64_t>(config.n_embd)}}, 
                              Dtype::Float32, Device::CPU, false);  // no grad for simplicity
        std::copy(wpe_data.begin(), wpe_data.end(), wpe_t.data<float>());
        wpe = make_tensor(wpe_t, "wpe");
        // Don't add to params_ - embeddings stay on CPU
        
        mlp = new MLP(config);
        finall = new Linear(config.n_embd, config.vocab_size, Device::CPU);

        for(auto& p : mlp->parameters()) params_.push_back(p);
        for(auto& p : finall->parameters()) params_.push_back(p);
    }

    ~GPT() {
        delete mlp;
        delete finall;
    }

    Value operator()(Value x) override {
        // Not used
        return Value();
    }

    // CPU embedding lookup - returns (B, T, C) tensor on GPU
    Tensor embed_tokens(const Tensor& ids, const Value& embedding) {
        Tensor ids_cpu = (ids.device().device != Device::CPU) ? ids.to(Device::CPU) : ids;
        Tensor emb_cpu = (embedding.val().device().device != Device::CPU) ? embedding.val().to(Device::CPU) : embedding.val();
        
        int64_t B = ids.shape().dims[0];
        int64_t T = ids.shape().dims[1];
        int64_t C = emb_cpu.shape().dims[1];
        int64_t N = B * T;
        
        Tensor out_cpu = Tensor(Shape{{B, T, C}}, Dtype::Float32, Device::CPU);
        float* out_ptr = out_cpu.data<float>();
        const uint16_t* ids_ptr = ids_cpu.data<uint16_t>();
        const float* emb_ptr = emb_cpu.data<float>();
        int V = emb_cpu.shape().dims[0];
        
        for (int64_t i = 0; i < N; ++i) {
            int idx = static_cast<int>(ids_ptr[i]);
            if (idx < 0 || idx >= V) idx = 0;
            const float* src = emb_ptr + idx * C;
            float* dst = out_ptr + i * C;
            std::copy(src, src + C, dst);
        }
        
        return out_cpu.to(Device::CUDA);
    }

    std::pair<Value, Value> forward(const Tensor& input_ids, const Tensor& target_ids) {
        int64_t B = input_ids.shape().dims[0];
        int64_t T = input_ids.shape().dims[1];

        // Get embeddings (on CPU, then move to GPU)
        Tensor tok_emb = embed_tokens(input_ids, wte);
        
        // Positional embeddings
        std::vector<uint16_t> pos_data(B * T);
        for(int b = 0; b < B; ++b) {
            for(int t = 0; t < T; ++t) {
                pos_data[b * T + t] = t;
            }
        }
        Tensor pos_ids = Tensor(Shape{{B, T}}, Dtype::UInt16, Device::CPU);
        std::copy(pos_data.begin(), pos_data.end(), pos_ids.data<uint16_t>());
        Tensor pos_emb = embed_tokens(pos_ids, wpe);
        
        // Add embeddings
        Value x = make_tensor(tok_emb + pos_emb, "emb_sum");

        // Apply MLP blocks with residual connections
        for(int i = 0; i < config.n_layers; ++i) {
            Value residual = x;
            Value m = (*mlp)(x);
            x = ag::add(residual, m);
        }

        Value logits = (*finall)(x);

        // Compute loss
        Value loss;
        if (target_ids.numel() > 0) {
            Tensor t = (target_ids.device().device != Device::CPU) ? target_ids.to(Device::CPU) : target_ids;

            int64_t N = B * T;
            
            // Create one-hot encoding
            Tensor onehot = Tensor(Shape{{B, T, static_cast<int64_t>(config.vocab_size)}}, Dtype::Float32, Device::CPU);
            float* oh_ptr = onehot.data<float>();
            std::fill(oh_ptr, oh_ptr + N * config.vocab_size, 0.0f);

            const uint16_t* t_ptr = t.data<uint16_t>();
            for(int64_t i = 0; i < N; ++i) {
                int label = static_cast<int>(t_ptr[i]);
                if (label >= 0 && label < config.vocab_size) {
                    oh_ptr[i * config.vocab_size + label] = 1.0f;
                }
            }

            Tensor onehot_gpu = onehot.to(Device::CUDA);
            Value onehot_val = make_tensor(onehot_gpu);
            loss = ag::cross_entropy_with_logits(logits, onehot_val);
        }

        return {logits, loss};
    }

    int64_t count_parameters() const {
        int64_t total = 0;
        for (const auto& p : params_) {
            total += p.val().numel();
        }
        return total;
    }

    GPTConfig config;
    Value wte, wpe;  // Embedding weights
    MLP* mlp;
    Linear* finall;
};

// ============================================================================
// Learning Rate Schedule
// ============================================================================

float get_lr(int step, int warmup_steps, int max_steps, float max_lr, float min_lr) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }
    if (step > max_steps) {
        return min_lr;
    }
    float decay_ratio = static_cast<float>(step - warmup_steps) / static_cast<float>(max_steps - warmup_steps);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return min_lr + coeff * (max_lr - min_lr);
}

// ============================================================================
// Main Training Loop
// ============================================================================

int main() {
    try {
        std::cout << "===== GPT-2 C++ Training (No PyTorch) =====\n";
        
        GPTConfig config;
        const int B = 4;
        const int T = 256;
        const std::string data_root = "/home/blubridge-035/Desktop/Backup/parallelism/script";
        
        cudaSetDevice(0);
        
        std::cout << "Initializing model..." << std::endl;
        GPT model(config);
        model.to(DeviceIndex(Device::CUDA));
        
        int64_t num_params = model.count_parameters();
        std::cout << "Number of parameters: " << num_params << std::endl;
        
        std::cout << "Initializing dataloaders..." << std::endl;
        DataLoaderLite train_loader(B, T, 0, 1, "train", data_root, true);
        DataLoaderLite val_loader(B, T, 0, 1, "val", data_root, true);
        
        std::vector<Value> all_params = model.parameters();
        Adam optimizer(all_params, config.max_lr, 0.9f, 0.95f, 1e-8f);
        
        std::cout << "Starting training...\n" << std::endl;
        
        for (int step = 0; step < config.max_steps; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bool last_step = (step == config.max_steps - 1);
            
            // Validation
            if (step % 50 == 0 || last_step) {
                val_loader.reset();
                float val_loss_accum = 0.0f;
                const int val_loss_steps = 5;
                
                for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                    Batch batch = val_loader.next_batch();
                    auto result = model.forward(batch.input, batch.target);
                    Value loss = result.second;
                    
                    Tensor l = loss.val().to_cpu();
                    val_loss_accum += l.data<float>()[0] / val_loss_steps;
                }
                
                std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
            }
            
            // Training step
            Batch batch = train_loader.next_batch();
            auto result = model.forward(batch.input, batch.target);
            Value logits = result.first;
            Value loss = result.second;
            
            optimizer.zero_grad();
            ag::backward(loss);
            
            float norm = ag::clip_grad_norm_(all_params, 1.0f);
            
            float lr = get_lr(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr);
            optimizer.set_alpha(lr);
            
            optimizer.step();
            
            cudaDeviceSynchronize();
            
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            
            int tokens_processed = B * T;
            double tokens_per_sec = tokens_processed / dt;
            
            Tensor l = loss.val().to_cpu();
            float loss_val = l.data<float>()[0];
            
            std::cout << "step " << std::setw(5) << step 
                      << " | loss: " << std::fixed << std::setprecision(6) << loss_val
                      << " | lr " << std::scientific << std::setprecision(4) << lr
                      << " | norm: " << std::fixed << std::setprecision(4) << norm
                      << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000) << "ms"
                      << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec
                      << std::endl;
        }
        
        std::cout << "\nTraining complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
