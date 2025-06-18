# Build Instructions for HDR Tone Mapping CUDA Webcam Filter

## Prerequisites

### System Requirements
- **OS**: Windows 10/11 or Ubuntu 20.04+
- **GPU**: CUDA-capable GPU (Compute Capability 6.0+)
- **RAM**: Minimum 8GB, recommended 16GB

### Required Software

#### 1. CUDA Toolkit
```bash
# Check if CUDA is installed
nvcc --version

# If not installed, download from:
# https://developer.nvidia.com/cuda-downloads

# For Ubuntu:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

#### 2. CMake (3.27+)
```bash
# Check current version
cmake --version

# Install/upgrade CMake
# For Ubuntu:
sudo apt-get update
sudo apt-get install cmake

# If version is too old, install from Kitware:
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update
sudo apt-get install cmake
```

#### 3. GCC Compiler
```bash
# Install GCC-12 (recommended for CUDA compatibility)
sudo apt-get install gcc-12 g++-12

# Or use existing GCC version
gcc --version
```

#### 4. Dependencies
```bash
# Ubuntu dependencies
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install libgtk-3-dev pkg-config
sudo apt-get install git

# Optional: for better performance
sudo apt-get install libtbb-dev
```

## Build Process

### Step 1: Configure GPU Architecture
**IMPORTANT**: Edit `CMakeLists.txt` to match your GPU:

```cmake
# Line 167-171: Update for your GPU
set(CMAKE_CUDA_ARCHITECTURES "89")  # RTX 4060/4070/4080/4090
set(CUDA_ARCH_BIN "8.9" CACHE STRING "CUDA architectures" FORCE)

# Common architectures:
# RTX 30 series (Ampere): "86" / "8.6"
# RTX 40 series (Ada): "89" / "8.9" 
# GTX 16 series: "75" / "7.5"
# RTX 20 series: "75" / "7.5"
```

Check your GPU architecture:
```bash
# Method 1: nvidia-smi
nvidia-smi

# Method 2: deviceQuery (if CUDA samples installed)
/usr/local/cuda/extras/demo_suite/deviceQuery

# Method 3: Online lookup
# Search your GPU model at: https://developer.nvidia.com/cuda-gpus
```

### Step 2: Create Build Directory
```bash
cd cuda-webcam-filter
mkdir build
cd build
```

### Step 3: Configure with CMake
```bash
# Basic configuration
cmake ..

# OR with specific compiler (if needed)
cmake -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 ..

# OR disable CUDA (CPU-only build)
cmake -DCUDA_SUPPORT=OFF ..
```

### Step 4: Build
```bash
# Build with all available cores
cmake --build . -j $(nproc)

# OR for specific number of cores
cmake --build . -j 4

# OR traditional make
make -j $(nproc)
```

## Windows Build

### Prerequisites (Windows)
1. **Visual Studio 2019/2022** with C++ tools
2. **CUDA Toolkit 12.x** from NVIDIA
3. **CMake 3.27+** from cmake.org
4. **Git for Windows**

### Build Steps (Windows)
```powershell
# Open Visual Studio Developer Command Prompt
cd cuda-webcam-filter
mkdir build
cd build

# Configure
cmake .. -G "Visual Studio 17 2022" -A x64

# Build
cmake --build . --config Release -j 8
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found
```bash
# Set CUDA path manually
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or in CMake
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
```

#### 2. OpenCV Build Fails
```bash
# Clean and rebuild
rm -rf build/*
cmake ..
cmake --build . -j 1  # Build with single core for better error visibility
```

#### 3. GCC Version Issues
```bash
# Force specific compiler
cmake -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 ..
```

#### 4. GPU Architecture Mismatch
```bash
# Find your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits

# Update CMakeLists.txt accordingly
```

### Build Options

#### Enable Debug
```bash
cmake -DCUDA_DEBUG=ON ..
```

#### Disable CUDA (CPU-only)
```bash
cmake -DCUDA_SUPPORT=OFF ..
```

#### Enable Tests
```bash
cmake -DRUN_UNIT_TESTS=ON ..
```

## Testing the Build

### Basic Test
```bash
# Run with webcam (default)
./cuda-webcam-filter

# Test HDR filter
./cuda-webcam-filter -f hdr -e 1.5 -g 2.2

# Test different algorithms
./cuda-webcam-filter -f hdr --tone-algorithm drago -e 2.0
./cuda-webcam-filter -f hdr --tone-algorithm mantiuk --saturation 1.2

# With preview mode
./cuda-webcam-filter -f hdr --preview
```

### Help and Options
```bash
# Show all options
./cuda-webcam-filter --help

# Version info
./cuda-webcam-filter --version
```

## Performance Optimization

### GPU Memory
- Ensure sufficient GPU memory (2GB+ for HD video)
- Monitor GPU usage: `nvidia-smi -l 1`

### CPU Usage
- Use Release build for better performance
- Monitor CPU usage during real-time processing

### Frame Rate
- Adjust exposure and parameters for optimal performance
- Use preview mode to compare CPU vs GPU performance

## Expected Results

### Performance Targets
- **1080p video**: 30+ FPS on RTX 3060+
- **720p video**: 60+ FPS on RTX 3060+
- **GPU Speedup**: 5-15x faster than CPU implementation

### Visual Quality
- Real-time HDR tone mapping effects
- Preserved color accuracy
- Smooth parameter adjustments
- Professional-quality results

## Next Steps

After successful build:
1. Test with different webcam resolutions
2. Experiment with HDR parameters
3. Compare different tone mapping algorithms
4. Monitor performance metrics
5. Consider advanced optimizations (shared memory, texture memory)

## Support

If you encounter issues:
1. Check GPU drivers are up to date
2. Verify CUDA installation with `nvcc --version`
3. Ensure CMake finds all dependencies
4. Check CMakeLists.txt GPU architecture settings
5. Try CPU-only build first to isolate CUDA issues