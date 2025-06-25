# CUDA Programming Assignment: Real-Time HDR Tone Mapping

## 🚀 Setup Instructions

**Use the complete template repository that includes all dependencies:**

```bash
# Clone the foundation template with all dependencies included
git clone https://github.com/lsawicki-cdv/course-accelerating-apps-nvidia-cuda.git
cd course-accelerating-apps-nvidia-cuda/templates/cuda-webcam-filter

# Copy our HDR implementation over the template
# (Our implementation files are in this Task 3 folder)
```

## Overview

This assignment extends the [CUDA webcam filter template](https://github.com/lsawicki-cdv/course-accelerating-apps-nvidia-cuda/tree/main/templates/cuda-webcam-filter) to implement real-time High Dynamic Range (HDR) tone mapping. While traditional cameras have limited dynamic range, HDR tone mapping algorithms simulate higher dynamic range by intelligently compressing the luminance range of an image to preserve details in both dark and bright regions.

The implementation transforms a standard webcam feed into one with enhanced dynamic range through GPU acceleration, allowing real-time video stream processing.

## Requirements Completed

### 1. Extended Filter Framework ✅
* Added HDR_TONEMAPPING filter type to existing framework
* Implemented parameter controls for exposure, gamma, saturation, and algorithm selection

### 2. Core Implementation ✅
Extended the CUDA webcam filter template with:
1. New filter type in FilterType enum (`filter_utils.h`)
2. Filter mapping implementation in `stringToFilterType()` (`filter_utils.cpp`)
3. CUDA kernels for:
   * Color space conversion (RGB to luminance/chrominance)
   * Global tone mapping operator (Reinhard, Drago, Mantiuk algorithms)
   * Local tone mapping operator (advanced)
   * Color space conversion back to RGB
4. Command-line parameters for tone mapping behavior control

### 3. Performance Optimization ✅
* Optimized memory transfers for HDR data
* Implemented shared memory usage where appropriate
* Provided performance comparison between GPU and CPU implementations

## Build Instructions

```bash
# 1. Clone the template with all dependencies
git clone https://github.com/lsawicki-cdv/course-accelerating-apps-nvidia-cuda.git
cd course-accelerating-apps-nvidia-cuda/templates/cuda-webcam-filter

# 2. Replace files with our HDR implementation
cp -r /path/to/Task\ 3/cuda-webcam-filter/src/* ./src/
# (Copy our enhanced files: convolution_kernels.cu, filter_utils.*, etc.)

# 3. Build the project
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
```

### Prerequisites
```bash
# CUDA Toolkit
nvidia-smi
nvcc --version

# All other dependencies included in the template repository
```

### Usage Examples
```bash
# Basic HDR tone mapping
./cuda-webcam-filter --filter hdr --exposure 1.5 --gamma 2.2

# Advanced tone mapping with algorithm selection
./cuda-webcam-filter --filter hdr --algorithm reinhard --exposure 2.0 --saturation 1.2

# Performance benchmarking
./cuda-webcam-filter --filter hdr --benchmark
```

## Features Implemented

- **Real-time HDR processing**: 30-60 FPS on modern GPUs
- **Multiple algorithms**: Reinhard, Drago, and Mantiuk tone mapping
- **Interactive controls**: Live parameter adjustment
- **Performance monitoring**: Real-time FPS and timing statistics
- **Webcam integration**: Direct camera feed processing

## Documentation

- [RESULTS.md](./RESULTS.md): Performance analysis and technical findings
- [README_HDR_IMPLEMENTATION.md](./README_HDR_IMPLEMENTATION.md): Technical implementation details
- [Template Repository](https://github.com/lsawicki-cdv/course-accelerating-apps-nvidia-cuda/tree/main/templates/cuda-webcam-filter): Foundation with all dependencies