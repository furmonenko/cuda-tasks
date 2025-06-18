# Build Instructions for Advanced Filter Pipeline CUDA Webcam Filter

## ⚠️ Prerequisites: External Dependencies

**IMPORTANT: You must set up external dependencies before building!**

1. **Read [SETUP_DEPENDENCIES.md](../SETUP_DEPENDENCIES.md)** in the project root
2. **Download required libraries**: OpenCV, Google Test, cxxopts, plog (~475MB total)
3. **Follow the setup instructions** completely before proceeding with build

## System Requirements
- **OS**: Windows 10/11 or Ubuntu 20.04+
- **GPU**: CUDA-capable GPU (Compute Capability 6.0+)
- **RAM**: Minimum 8GB, recommended 16GB for multi-stream processing

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

### Step 1: Set Up External Dependencies
**CRITICAL**: Complete dependency setup before building:
```bash
# Follow instructions in SETUP_DEPENDENCIES.md
# Ensure external/ directory contains:
# - opencv/
# - opencv_contrib/
# - gtest/
# - cxxopts/
# - plog/
```

### Step 2: Configure GPU Architecture
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

### Step 3: Create Build Directory
```bash
cd cuda-webcam-filter
mkdir build
cd build
```

### Step 4: Configure with CMake
```bash
# Basic configuration
cmake ..

# OR with specific compiler (if needed)
cmake -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 ..

# OR disable CUDA (CPU-only build)
cmake -DCUDA_SUPPORT=OFF ..
```

### Step 5: Build
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
# Set up external dependencies first (see SETUP_DEPENDENCIES.md)

# Open Visual Studio Developer Command Prompt
cd cuda-webcam-filter
mkdir build
cd build

# Configure
cmake .. -G "Visual Studio 17 2022" -A x64

# Build
cmake --build . --config Release -j 8
```

## Testing the Build

### Basic Pipeline Tests
```bash
# Run with single filter
./cuda-webcam-filter -f blur

# Test basic pipeline
./cuda-webcam-filter --pipeline "blur,sharpen"

# Test multi-stream performance
./cuda-webcam-filter --pipeline "blur,sharpen,hdr" --multi-stream

# Test transitions
./cuda-webcam-filter --pipeline "blur,edge" --transitions --wipe-speed 2.0
```

### Advanced Features
```bash
# Performance analysis
./cuda-webcam-filter --pipeline "blur,sharpen,hdr,edge,emboss" --benchmark

# Timing visualization
./cuda-webcam-filter --pipeline "blur,sharpen,hdr" --timing-visualization

# Dynamic pipeline modification
./cuda-webcam-filter --pipeline "blur" --dynamic-pipeline
```

### Help and Options
```bash
# Show all options
./cuda-webcam-filter --help

# Version info
./cuda-webcam-filter --version
```

## Performance Targets

### Pipeline Performance
- **Single filter**: 60+ FPS on RTX 3060+
- **3-filter pipeline**: 45+ FPS on RTX 3060+
- **5-filter pipeline**: 30+ FPS on RTX 3060+
- **Multi-stream improvement**: 25-40% performance gain

### Memory Requirements
- **Single filter**: 2GB GPU memory
- **5-filter pipeline**: 4GB GPU memory
- **Multi-stream processing**: 6GB GPU memory

## Troubleshooting

### Common Issues

#### 1. External Dependencies Not Found
```bash
# Verify dependencies are set up
ls external/
# Should show: opencv, opencv_contrib, gtest, cxxopts, plog

# If missing, follow SETUP_DEPENDENCIES.md
```

#### 2. OpenCV Build Fails
```bash
# Clean and rebuild
rm -rf build/*
cmake ..
cmake --build . -j 1  # Build with single core for better error visibility
```

#### 3. CUDA Streams Not Working
```bash
# Check GPU supports concurrent kernels
nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
# Should be 6.0 or higher for full stream support
```

#### 4. Performance Issues
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check CPU usage
htop

# Enable debug timing
./cuda-webcam-filter --pipeline "blur,sharpen" --debug-timing
```

## Build Options

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

#### Enable Performance Profiling
```bash
cmake -DENABLE_PROFILING=ON ..
```

## Next Steps

After successful build:
1. Test pipeline performance with different filter combinations
2. Experiment with transition effects
3. Compare single-stream vs multi-stream performance
4. Monitor timing visualization for optimization opportunities
5. Test dynamic pipeline modification features

## Support

If you encounter issues:
1. **First**: Ensure external dependencies are properly set up
2. Check GPU drivers are up to date
3. Verify CUDA installation with `nvcc --version`
4. Ensure CMake finds all dependencies
5. Check CMakeLists.txt GPU architecture settings
6. Try CPU-only build first to isolate CUDA issues