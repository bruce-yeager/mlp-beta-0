# GPT-2 C++ Compilation and Usage Commands

## Prerequisites

- CUDA Toolkit installed (tested with CUDA 11.x/12.x)
- cgadimpl library built
- OwnTensor library built

## Building cgadimpl Library

```bash
cd /home/blubridge-035/Desktop/Backup/parallelism/script/cgadimpl/cgadimpl/build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)
```

## Building GPT-2 Training Script

### Option 1: Using g++ directly

```bash
cd /home/blubridge-035/Desktop/Backup/parallelism/script

g++ -O3 -std=c++20 gpt2_test.cpp -o gpt2_test \
    -I/home/blu-bridge020/Code_Repos/MLP-Custom-Framework/cgadimpl/cgadimpl/include \
    -I/home/blu-bridge020/Code_Repos/MLP-Custom-Framework/cgadimpl/tensor/include \
    -I/usr/local/cuda/include \
    -L/home/blu-bridge020/Code_Repos/MLP-Custom-Framework/cgadimpl/cgadimpl/build \
    -L/home/blu-bridge020/Code_Repos/MLP-Custom-Framework/cgadimpl/tensor/lib \
    -L/usr/local/cuda/lib64 \
    -lcgadimpl -ltensor -lcudart -lpthread -ltbb -fopenmp

g++ -O3 -std=c++20 gpt2_test.cpp -o gpt2_test \
    -I/home/blubridge-035/Desktop/Backup/parallelism/script/cgadimpl/cgadimpl/include \
    -I/home/blubridge-035/Desktop/Backup/parallelism/script/cgadimpl/tensor/include \
    -I/usr/local/cuda/include \
    -L/home/blubridge-035/Desktop/Backup/parallelism/script/cgadimpl/cgadimpl/build \
    -L/home/blubridge-035/Desktop/Backup/parallelism/script/cgadimpl/tensor/lib \
    -L/usr/local/cuda/lib64 \
    -lcgadimpl -ltensor -lcudart -lpthread -ltbb -fopenmp

```

### Option 2: Using CMake (recommended)

Add to CMakeLists.txt in cgadimpl directory:

```cmake
add_executable(gpt2_test /home/blubridge-035/Desktop/Backup/parallelism/script/gpt2_test.cpp)
target_link_libraries(gpt2_test PRIVATE cgadimpl::cgadimpl CUDA::cudart OpenMP::OpenMP_CXX)
```

Then build:

```bash
cd /home/blubridge-035/Desktop/Backup/parallelism/script/cgadimpl/cgadimpl/build
cmake ..
make gpt2_test -j$(nproc)
```

## Running the Training Script

```bash
# Set library paths
export LD_LIBRARY_PATH=/home/blubridge-035/Desktop/Backup/parallelism/script/cgadimpl/tensor/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run training
./gpt2_test
```

## Expected Output Format

```
===== GPT-2 C++ Training (No PyTorch) =====
Initializing model...
Number of parameters: XXXXXXXX
Initializing dataloaders...
found X shards for split train
found X shards for split val
Starting training...

validation loss: 11.XXXX
step     0 | loss: 11.XXXXXX | lr 5.0000e-06 | norm: X.XXXX | dt: XXX.XXms | tok/sec: XXXXX.XX
step     1 | loss: 11.XXXXXX | lr 1.0000e-05 | norm: X.XXXX | dt: XXX.XXms | tok/sec: XXXXX.XX
...
```

## Troubleshooting

### Library not found errors

Make sure `LD_LIBRARY_PATH` includes both the tensor library and CUDA library paths:

```bash
export LD_LIBRARY_PATH=/home/blubridge-035/Desktop/Backup/parallelism/script/cgadimpl/tensor/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CUDA errors

Verify CUDA device is available:

```bash
nvidia-smi
```

### Missing symbols

Rebuild cgadimpl with latest changes:

```bash
cd /home/blubridge-035/Desktop/Backup/parallelism/script/cgadimpl/cgadimpl/build
make clean
make -j$(nproc)
```
