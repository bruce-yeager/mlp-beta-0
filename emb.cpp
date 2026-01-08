// embedding_cpu.cpp
// g++ -O2 -std=c++17 embedding_cpu.cpp -o embedding

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>

#include "dl_test.cpp"

// struct Batch {
//     int B = 0, T = 0;
//     std::vector<int32_t> x; // (B*T) token ids
//     std::vector<int32_t> y; // (B*T) targets (not used here)
// };

// A minimal nn.Embedding-like layer (CPU).
// - weights: float32 [V, C]
// - forward: ids [B*T] -> out [B*T*C]
// - backward: grad_out [B*T*C] -> gradW [V*C] (scatter-add by token id)
class Embedding {
public:
    Embedding() = default;
    Embedding(int vocab_size, int embed_dim, int padding_idx = -1, uint64_t seed = 1234, int rank=0)
        : V(vocab_size), C(embed_dim), padding_idx(padding_idx), rank(rank)
    {
        if (V <= 0 || C <= 0) throw std::runtime_error("Embedding: V and C must be > 0");

        W.resize(static_cast<size_t>(V) * static_cast<size_t>(C));
        // gradW.resize(W.size(), 0.0f);

        // Init similar-ish to common practice (small normal)
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd(0.0f, 0.02f);
        for (auto &w : W) w = nd(rng);

        // Optionally zero padding row if requested
        if (padding_idx >= 0 && padding_idx < V) {
            for (int j = 0; j < C; ++j) W[static_cast<size_t>(padding_idx) * C + j] = 0.0f;
        }
    }

    // Forward: ids is flat length N = B*T, returns out flat length N*C.
    // out[n*C + j] = W[ ids[n] * C + j]
    OwnTensor::Tensor forward(const OwnTensor::Tensor& ids) const {
    const int64_t N = ids.numel();

    // OwnTensor::Tensor out(N * static_cast<size_t>(C));
    OwnTensor::Tensor out(OwnTensor::Shape{{N, static_cast<int64_t>(C) }}, OwnTensor::Dtype::Float32, OwnTensor::DeviceIndex(OwnTensor::Device::CPU));

        for (size_t n = 0; n < N; ++n) {
            uint16_t tok = ids.data<uint16_t>()[n];

            // Padding: output zeros (like PyTorch if you set padding_idx and keep that row fixed)
            if (tok == padding_idx) {
                float* o = static_cast<float*> (out.data()) + n * static_cast<size_t>(C);
                std::fill(o, o + C, 0.0f);
                continue;
            }

            if (tok < 0 || tok >= V) {
                throw std::runtime_error("Embedding::forward: token id out of range: " + std::to_string(tok));
            }

            const float* row = W.data() + static_cast<size_t>(tok) * static_cast<size_t>(C);
            float* o = static_cast<float*> (out.data()) + n * static_cast<size_t>(C);
            std::copy(row, row + C, o);
        }
        OwnTensor::Tensor out_cuda = out.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));
        return out_cuda;
    }

    // Backward: accumulates gradW from grad_out (same shape as forward output).
    // ids: length N (B*T), grad_out: length N*C
    // void backward(const std::vector<u_int16_t>& ids, const std::vector<float>& grad_out) {
    //     const size_t N = ids.size();
    //     if (grad_out.size() != N * static_cast<size_t>(C)) {
    //         throw std::runtime_error("Embedding::backward: grad_out has wrong size");
    //     }

    //     // zero gradW
    //     std::fill(gradW.begin(), gradW.end(), 0.0f);

    //     for (size_t n = 0; n < N; ++n) {
    //         u_int16_t tok = ids[n];
    //         if (tok == padding_idx) continue; // no grad for padding row

    //         if (tok < 0 || tok >= V) {
    //             throw std::runtime_error("Embedding::backward: token id out of range: " + std::to_string(tok));
    //         }

    //         float* gRow = gradW.data() + static_cast<size_t>(tok) * static_cast<size_t>(C);
    //         const float* go  = grad_out.data() + n * static_cast<size_t>(C);

    //         // scatter-add
    //         for (int j = 0; j < C; ++j) gRow[j] += go[j];
    //     }
    // }

    // Simple SGC update: W -= lr * gradW
    // Keeps padding row fixed at zero if padding_idx is set.
    // void sgd_step(float lr) {
    //     if (lr <= 0) throw std::runtime_error("sgd_step: lr must be > 0");

    //     for (size_t i = 0; i < W.size(); ++i) {
    //         W[i] -= lr * gradW[i];
    //     }
    //     if (padding_idx >= 0 && padding_idx < V) {
    //         for (int j = 0; j < C; ++j) W[static_cast<size_t>(padding_idx) * C + j] = 0.0f;
    //     }
    // }

    const std::vector<float>& weights() const { return W; }
    // const std::vector<float>& grads() const { return gradW; }

private:
    int V; // vocab
    int C; // embedding dim
    int padding_idx;
    int rank = 0;
    std::vector<float> W;      // [V*C]
    // std::vector<float> gradW;  // [V*C]
};

int main() {
    try {
        // Suppose you have a batch (B,T) tokens from your loader:
        DataLoaderLite loader(2, 3, 0, 2, "train","/home/blubridge-035/Desktop/Backup/parallelism/DataLoaderLite" , 0);
        Batch batch = loader.next_batch();


        int vocab_size = 50304; // GPT-2 vocab size
        int embed_dim  = 5;   // example
        Embedding emb(vocab_size, embed_dim, /*padding_idx=*/-1);

        // Forward: (B*T) -> (B*T*C)
        OwnTensor::Tensor out = emb.forward(batch.input);
        std::cout << "out size = " << "< " << out.shape().dims[0] << ","<< out.shape().dims[1] << " >" << " = (B*T*C)\n";
        // std::cout << "first token embedding first 5 dims: ";
        out.to_cpu().display();
        // for (int j = 0; j < 5; ++j) std::cout << out_cpu.data<float>()[j] << " ";
        // std::cout << "\n";

        // Fake grad_out (normally comes from later layers)
        // std::vector<float> grad_out(out.size(), 1.0f);

        // Backward accumulates grads into embedding table
        // emb.backward(batch.x, grad_out);

        // SGC update
        // emb.sgd_step(1e-3f);

        std::cout << "done\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}

