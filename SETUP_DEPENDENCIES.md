# Setup Dependencies

This project uses several external libraries that are excluded from the repository to reduce size. Follow these steps to set up the dependencies:

## Required Dependencies

### For Tasks 3 & 4 (CUDA Webcam Filter)

The following external libraries are required:

1. **OpenCV** (208-367MB) - Computer vision library
2. **OpenCV Contrib** (101MB) - Additional OpenCV modules  
3. **Google Test** (4.3MB) - Testing framework
4. **cxxopts** (816KB) - Command line parsing
5. **plog** (692KB) - Logging library

## Setup Instructions

### Option 1: Git Submodules (Recommended)

```bash
# Navigate to Task 3 or 4 directory
cd "cuda-tasks/Task 3/cuda-webcam-filter" 
# or
cd "cuda-tasks/Task 4/cuda-webcam-filter"

# Initialize and update submodules
git submodule update --init --recursive external/opencv
git submodule update --init --recursive external/opencv_contrib  
git submodule update --init --recursive external/gtest
git submodule update --init --recursive external/cxxopts
git submodule update --init --recursive external/plog
```

### Option 2: Manual Download

If submodules are not configured, download manually:

```bash
cd external/

# OpenCV
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Testing and utilities
git clone https://github.com/google/googletest.git gtest
git clone https://github.com/jarro2783/cxxopts.git
git clone https://github.com/SergiusTheBest/plog.git
```

### Option 3: Package Manager (Linux/macOS)

```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev libgtest-dev

# macOS with Homebrew  
brew install opencv googletest

# Windows with vcpkg
vcpkg install opencv gtest cxxopts
```

## Build Instructions

After setting up dependencies:

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release
```

## Notes

- Total download size: ~315-475MB per task
- Build artifacts are also ignored and will be regenerated
- Make sure you have CUDA toolkit installed for GPU acceleration
- OpenCV build may take significant time (20-60 minutes)

## Troubleshooting

- If CMake can't find OpenCV, set `OpenCV_DIR` environment variable
- For CUDA support, ensure CUDA toolkit matches your GPU driver version
- On Windows, you may need to set proper paths in CMake GUI