#include <iostream>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include "/home/blubridge-035/Desktop/Backup/parallelism/DataLoaderLite/Tensor-Implementations/include/TensorLib.h"
#include "dl_test.cpp"

// Helper to get env var with default
std::string get_env(const char* name, const char* default_val = "") {
    const char* val = std::getenv(name);
    return val ? std::string(val) : std::string(default_val);
}

int main(int argc, char* argv[]) {
    // 1. Read Environment Variables injected by launcher.py
    int rank = std::stoi(get_env("RANK", "0"));
    int local_rank = std::stoi(get_env("LOCAL_RANK", "0"));
    int world_size = std::stoi(get_env("WORLD_SIZE", "1"));
    std::string master_addr = get_env("MASTER_ADDR", "127.0.0.1");
    int master_port = std::stoi(get_env("MASTER_PORT", "29500"));

    std::cout << "[C++ Worker] Initialized Rank: " << rank 
              << " Local Rank: " << local_rank 
              << " World Size: " << world_size << std::endl;

    // 2. Set Device
    // Thanks to the launcher setting CUDA_VISIBLE_DEVICES, 
    // every process thinks it is using Device 0.
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device!" << std::endl;
        return -1;
    }

    // 3. Your Training Loop Here...
    // Initialize your sockets/NCCL using master_addr and master_port + rank
    try {
        const std::string data_root =
            "/home/blubridge-035/Desktop/Backup/parallelism/DataLoaderLite";
        const std::string split = "train";
        const int B = 32;
        const int T = 524288;  // Reduced from 524288 to prevent GPU OOM

        const bool master = (rank == 0);
        DataLoaderLite loader(B, T, rank, world_size, split, data_root, master);
        
        // for(int i=0; i<50; i++)
        // {
        
        // Get one batch (per rank)
        Batch batch = loader.next_batch();
        
        
        // Verify data is on GPU
        std::cout << "\n[Rank " << rank << "] Batch loaded successfully!\n";
        std::cout << "  Batch shape: B=" << batch.B << ", T=" << batch.T << "\n";
        std::cout << "  Total tokens per batch: " << (batch.B * batch.T) << "\n";
        
        // Print first few CPU values
        std::cout << "  CPU x[0:5]: ";
        for (int i = 0; i < std::min(5, (int)batch.x.size()); i++) {
            std::cout << batch.x[i] << " ";
        }
        std::cout << "\n";
        
        std::cout << "  CPU y[0:5]: ";
        for (int i = 0; i < std::min(5, (int)batch.y.size()); i++) {
            std::cout << batch.y[i] << " ";
        }
        std::cout << "\n";
        
        // Verify tensors are on CUDA
        std::cout << "  Input tensor device: " 
                  << (batch.input.device().is_cuda() ? "CUDA " : "CPU ") << "\n";
        std::cout << "  Target tensor device: " 
                  << (batch.target.device().is_cuda() ? "CUDA " : "CPU ") << "\n";
        
        // Verify different ranks get different data
        std::cout << "  First token value: x[0]=" << batch.x[0] 
                  << " (should differ across ranks)\n";
        
        std::cout << "[Rank " << rank << "] Verification complete!\n\n";
    // }
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
     
        
    }
    
    std::cout << "[C++ Worker " << rank << "] Training finished." << std::endl;
    return 0;
}