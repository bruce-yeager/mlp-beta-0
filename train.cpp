#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
// #include <mpi.h>


// #include "TensorLib.h"
// #include "ad/core/graph.hpp"
#include "ad/ag_all.hpp"
// #include "ad/ops/ops.hpp"
// #include "nn/nn.hpp"
// #include "ad/autodiff/autodiff.hpp"
#include "optim.hpp"

using namespace ag;
using namespace ag::nn;
namespace fs = std::filesystem;

// --- DataLoaderLite Implementation ---

static int getenv_int(const char* key, int def) {
    const char* v = std::getenv(key);
    return v ? std::atoi(v) : def;
}

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

        size_t total_tokens = file_bytes_ /2;
        std::cout<<"Tokens: "<< total_tokens<<std::endl;
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
    OwnTensor::Tensor input;
    OwnTensor::Tensor target;
};

class DataLoaderLite {
public:
    DataLoaderLite(int B, int T,
                     int rank, int world_size,
                     const std::string& split,
                     const std::string& data_root,
                     bool master_process = true,
                     size_t max_tokens_per_shard = 400000000)
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
        
        std::cout << "Creating input tensor on CPU" << std::endl;
        b.input = OwnTensor::Tensor(OwnTensor::Shape{{static_cast<int64_t>(B_), static_cast<int64_t>(T_)}}, OwnTensor::Dtype::UInt16, OwnTensor::DeviceIndex(OwnTensor::Device::CPU));
        std::cout << "Setting input data" << std::endl;
        b.input.set_data(b.x);

        std::cout << "Creating target tensor on CPU" << std::endl;
        b.target = OwnTensor::Tensor(OwnTensor::Shape{{static_cast<int64_t>(B_), static_cast<int64_t>(T_)}}, OwnTensor::Dtype::UInt16, OwnTensor::DeviceIndex(OwnTensor::Device::CPU));
        std::cout << "Setting target data" << std::endl;
        b.target.set_data(b.y);

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

// --- End DataLoaderLite ---

// Configuration
struct GPTConfig {
    int context_length = 1024;
    int vocab_size = 50304;
    int n_embd = 768;
    int n_layers = 12;
    int n_head = 12; 
    
    // Optimization
    float max_lr = 6e-4;
    float min_lr = 6e-5;
    int warmup_steps = 10;
    int max_steps = 100; 
};

// Modules

class Embedding : public Module {
public:
    Embedding(int num_embeddings, int embedding_dim, int padding_idx = -1) 
        : V(num_embeddings), C(embedding_dim), padding_idx(padding_idx) {
        std::cout << "Embedding ctor start" << std::endl;
        W_data.resize(V * C);
        std::mt19937 rng(1337);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for(auto& w : W_data) w = dist(rng);
        
        if (padding_idx >= 0 && padding_idx < V) {
            for(int j=0; j<C; ++j) W_data[padding_idx * C + j] = 0.0f;
        }

        std::cout << "Creating Tensor" << std::endl;
        Tensor t = Tensor(Shape{{static_cast<int64_t>(V), static_cast<int64_t>(C)}}, Dtype::Float32, Device::CPU, true);
        std::cout << "Copying data" << std::endl;
        std::copy(W_data.begin(), W_data.end(), t.data<float>());
        
        std::cout << "Making Value" << std::endl;
        W = make_tensor(t, "embedding_weights");
        params_.push_back(W);
        
        grad_W_data.resize(V * C, 0.0f);
        std::cout << "Embedding ctor done" << std::endl;
    }

    Value operator()(Value input) override {
        Tensor ids = input.val();
        Tensor ids_cpu = ids;
        if (ids.device().device != Device::CPU) {
            ids_cpu = ids.to(Device::CPU);
        }
        
        int64_t B = ids.shape().dims[0];
        int64_t T = ids.shape().dims[1];
        int64_t N = B * T;
        
        Tensor out_cpu = Tensor(Shape{{B, T, static_cast<int64_t>(C)}}, Dtype::Float32, Device::CPU, true);
        float* out_ptr = out_cpu.data<float>();
        
        const uint16_t* ids_ptr = ids_cpu.data<uint16_t>();
        
        // Lookup on CPU
        Tensor w_cpu = W.val();
        if (w_cpu.device().device != Device::CPU) {
            w_cpu = w_cpu.to(Device::CPU);
        }
        const float* w_val_ptr = w_cpu.data<float>();
        
        for(int64_t i=0; i<N; ++i) {
            int idx = static_cast<int>(ids_ptr[i]);
            if (idx < 0 || idx >= V) idx = 0; 
            
            const float* src = w_val_ptr + idx * C;
            float* dst = out_ptr + i * C;
            std::copy(src, src + C, dst);
        }
        
        // Move result to weight's device
        DeviceIndex target_device = W.val().device();
        Tensor out_tensor = out_cpu;
        if (target_device.device != Device::CPU) {
            out_tensor = out_cpu.to(target_device);
        }
        
        Value out = make_tensor(out_tensor, "embedding_out");
        
        out.register_hook([this, ids_cpu](Node* n) {
            Tensor grad = n->grad;
            if (grad.device().device != Device::CPU) {
                grad = grad.to(Device::CPU);
            }
            
            const float* g_ptr = grad.data<float>();
            const uint16_t* ids_ptr = ids_cpu.data<uint16_t>();
            int64_t N = ids_cpu.numel();
            
            if (W.node->grad.numel() == 0) {
                 W.node->grad = Tensor(W.val().shape(), Dtype::Float32, Device::CPU);
                 std::fill(W.node->grad.data<float>(), W.node->grad.data<float>() + W.node->grad.numel(), 0.0f);
            }
            
            float* w_g_ptr = W.node->grad.data<float>();
            
            for(int64_t i=0; i<N; ++i) {
                int idx = static_cast<int>(ids_ptr[i]);
                if (idx < 0 || idx >= V) continue;
                
                float* dst = w_g_ptr + idx * C;
                const float* src = g_ptr + i * C;
                for(int j=0; j<C; ++j) {
                    dst[j] += src[j];
                }
            }
        });
        
        return out;
    }

    Value W;
    int V, C, padding_idx;
    std::vector<float> W_data;
    std::vector<float> grad_W_data;
};

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
    
    Value operator()(Value x) override {
        std::cout << "MLP forward start" << std::endl;
        x = (*l_up)(x);
        std::cout << "MLP l_up done" << std::endl;
        x = (*gelu)(x);
        std::cout << "MLP gelu done" << std::endl;
        x = (*l_down)(x);
        std::cout << "MLP l_down done" << std::endl;
        return x;
    }
    
    Linear* l_up;
    Linear* l_down;
    GELU* gelu;
};

class GPT: public nn::Module {
public:
    GPT(GPTConfig config) : config(config) {
        std::cout << "GPT start" << std::endl;
        wte = new Embedding(config.vocab_size, config.n_embd);
        std::cout << "Token Embedding created" << std::endl;
        wpe = new Embedding(config.context_length, config.n_embd);
        std::cout << "Positional Encoding created" << std::endl;
        mlp = new MLP(config);
        std::cout << "MLP created" << std::endl;
        finall = new Linear(config.n_embd, config.vocab_size, Device::CPU);
        std::cout << "Final Linear created" << std::endl;

        for(auto& p : wte->parameters()) params_.push_back(p);
        for(auto& p : wpe->parameters()) params_.push_back(p);
        for(auto& p : mlp->parameters()) params_.push_back(p);
        for(auto& p : finall->parameters()) params_.push_back(p);
    }

    using nn::Module::operator();
    Value operator()(Value x) override {
        auto result = forward(x, Value());
        return result.first;
    }
    
    std::pair<Value, Value> forward(Value idx, Value targets) {
        int64_t B = idx.val().shape().dims[0];
        int64_t T = idx.val().shape().dims[1];
        
        std::vector<uint16_t> pos_data(B * T);
        for(int b=0; b<B; ++b) {
            for(int i=0; i<T; ++i) {
                pos_data[b*T + i] = i;
            }
        }
        Tensor pos_t = Tensor(Shape{{B, T}}, Dtype::UInt16, Device::CPU); 
        std::copy(pos_data.begin(), pos_data.end(), pos_t.data<uint16_t>());
        
        // Keep pos on CPU for embedding lookup
        Value pos = make_tensor(pos_t, "pos");
        
        Value pos_emb = (*wpe)(pos); 
        Value tok_emb = (*wte)(idx); 
        
        Value x = ag::add(tok_emb, pos_emb); 
        
        for(int i=0; i<6; ++i) {
            Value residual = x;
            Value m = (*mlp)(x);
            x = ag::add(residual, m);
        }
        
        Value logits = (*finall)(x); 
        
        DeviceIndex dev = logits.val().device();
        // Value v5 = make_tensor(Tensor::full(Shape{{1}}, ag::options(logits.val()), 5.0f));
        // Value v7_5 = make_tensor(Tensor::full(Shape{{1}}, ag::options(logits.val()), 7.5f));
        // Value v23 = make_tensor(Tensor::full(Shape{{1}}, ag::options(logits.val()), 23.0f));

        // Value scaled = ag::div(ag::add(logits, v5), v7_5);
        // Value sig = ag::sigmoid(scaled);
        // logits = ag::mul(sig, v23);
        
        Value loss;
        if (targets.val().numel() > 0) {
            Tensor t = targets.val();
            // targets are on CPU from loader, but let's be sure
            if (t.device().device != Device::CPU) t = t.to(Device::CPU);
            
            int64_t N = B * T;
            Tensor onehot_cpu = Tensor(Shape{{N, static_cast<int64_t>(config.vocab_size)}}, Dtype::Float32, Device::CPU);
            float* oh_ptr = onehot_cpu.data<float>();
            std::fill(oh_ptr, oh_ptr + N * config.vocab_size, 0.0f);
            
            const uint16_t* t_ptr = t.data<uint16_t>();
            for(int64_t i=0; i<N; ++i) {
                int label = static_cast<int>(t_ptr[i]);
                if (label >= 0 && label < config.vocab_size) {
                    oh_ptr[i * config.vocab_size + label] = 1.0f;
                }
            }
            
            Tensor onehot_3d_tensor = Tensor(Shape{{B, T, static_cast<int64_t>(config.vocab_size)}}, Dtype::Float32, Device::CPU);
            std::copy(oh_ptr, oh_ptr + N * config.vocab_size, onehot_3d_tensor.data<float>());
            
            if (dev.device != Device::CPU) {
                onehot_3d_tensor = onehot_3d_tensor.to(dev);
            }
            
            Value onehot_3d = make_tensor(onehot_3d_tensor);
            
            loss = ag::cross_entropy_with_logits(logits, onehot_3d);
        }
        
        return {logits, loss};
    }
    
    void zero_grad() {
        wte->zero_grad();
        wpe->zero_grad();
        mlp->zero_grad();
        finall->zero_grad();
    }
    
    GPTConfig config;
    Embedding *wte, *wpe;
    MLP *mlp;
    Linear *finall;
};

int main() {
    try {
        std::cout << "Init model" << std::endl;
        GPTConfig config;
        GPT model(config);
        model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA));
        
        std::cout << "Init loader" << std::endl;
        int B = 8;
        int T = 1024;
        DataLoaderLite loader(B, T, 0, 1, "train", "/home/blubridge-035/Desktop/Backup/parallelism/DataLoaderLite", true);
        
        for (int step = 0; step < config.max_steps; ++step) {
            std::cout << "Step " << step << " start" << std::endl;
            Batch batch = loader.next_batch();
            std::cout << "Batch loaded" << std::endl;
            
            std::cout << "Making input/target values" << std::endl;
            Value x = make_tensor(batch.input, "input");
            Value y = make_tensor(batch.target, "target"); 
            
            std::cout << "Calling model.forward" << std::endl;
            auto result = model.forward(x, y);
            std::cout << "Model forward done" << std::endl;
            Value logits = result.first;
            Value loss = result.second;
            
            model.zero_grad();
            
            if (loss.node) {
                std::cout << "Calling backward" << std::endl;
                ag::backward(loss);
                std::cout << "Calling SGD" << std::endl;
                ag::SGD(loss, nullptr, config.max_lr);
            }
            
            if (step % 1 == 0) {
                Tensor l = loss.val().to(Device::CPU);
                std::cout << "Step " << step << " Loss: " << l.data<float>()[0] << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}