# CUDA Programming Assignment: Advanced Filter Pipeline with Streams

## 🚀 Setup Instructions

**Use the complete template repository that includes all dependencies:**

```bash
# Clone the foundation template with all dependencies included
git clone https://github.com/lsawicki-cdv/course-accelerating-apps-nvidia-cuda.git
cd course-accelerating-apps-nvidia-cuda/templates/cuda-webcam-filter

# Copy our advanced pipeline implementation over the template
# (Our implementation files are in this Task 4 folder)
```

## Overview

This assignment extends the [CUDA webcam filter template](https://github.com/lsawicki-cdv/course-accelerating-apps-nvidia-cuda/tree/main/templates/cuda-webcam-filter) to support a sophisticated filter pipeline that applies multiple filters sequentially while maintaining real-time performance. The implementation uses CUDA streams for concurrent processing and advanced filter chaining capabilities.

## Tasks Completed

### Part 1: Filter Pipeline Architecture ✅
1. **Designed and implemented** a filter pipeline architecture for sequential filter application
2. **Created dynamic mechanisms** to add/remove filters from the pipeline at runtime
3. **Implemented efficient memory management** for intermediate results using ping-pong buffers
4. **Used CUDA streams** to execute different pipeline stages concurrently

### Part 2: Custom Filter Transitions ✅
1. **Implemented Wipe Transition** with gradual left-to-right filter replacement
2. **Added UI controls** for transition parameters and timing adjustment
3. **Ensured transition compatibility** with all input sources (webcam, video files, synthetic patterns)

### Part 3: Performance Analysis and Optimization ✅
1. **Implemented comprehensive instrumentation** for pipeline performance measurement
2. **Identified and resolved bottlenecks** to maintain real-time performance (>30 FPS)
3. **Compared single-stream vs multi-stream** performance (25-40% improvement with streams)
4. **Created real-time visualization** of filter pipeline timings with live updates
5. **Documented findings** and optimization strategies in RESULTS.md

## Build Instructions

```bash
# 1. Clone the template with all dependencies
git clone https://github.com/lsawicki-cdv/course-accelerating-apps-nvidia-cuda.git
cd course-accelerating-apps-nvidia-cuda/templates/cuda-webcam-filter

# 2. Replace files with our advanced pipeline implementation
cp -r /path/to/Task\ 4/cuda-webcam-filter/src/* ./src/
# (Copy our enhanced files: filter_pipeline.*, convolution_kernels.cu, etc.)

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
# Basic filter pipeline
./cuda-webcam-filter --pipeline "blur,sharpen,hdr" --multi-stream

# Pipeline with transitions
./cuda-webcam-filter --pipeline "blur,edge" --transitions --wipe-speed 2.0

# Performance analysis mode
./cuda-webcam-filter --pipeline "blur,sharpen,hdr,edge,emboss" --benchmark --timing-visualization
```

## Features Implemented

- **Multi-filter pipeline**: Chain up to 10 filters sequentially
- **CUDA streams**: Concurrent execution with 25-40% performance improvement
- **Dynamic pipeline**: Runtime filter addition/removal
- **Transition effects**: Wipe, fade, and blend transitions
- **Performance monitoring**: Real-time timing visualization
- **Memory optimization**: Ping-pong buffers and efficient memory management

## Deliverables Completed

1. ✅ **Modified source code** with complete pipeline architecture and transitions
2. ✅ **Performance analysis** with comprehensive charts and graphs in RESULTS.md
3. ✅ **Documentation** of optimization strategies and findings

## Documentation

- [RESULTS.md](./RESULTS.md): Performance analysis, benchmarks, and optimization findings
- [SETUP_DEPENDENCIES.md](../SETUP_DEPENDENCIES.md): Template repository setup guide
- [Template Repository](https://github.com/lsawicki-cdv/course-accelerating-apps-nvidia-cuda/tree/main/templates/cuda-webcam-filter): Foundation with all dependencies
- Source code with comprehensive comments and documentation