# Image Convolution Performance Results

## Overview
This document summarizes the performance analysis of different convolution implementations:
- CPU Sequential
- CUDA Naive
- CUDA Shared Memory
- CUDA Separable Filters

## Implementation Details

### CPU Implementation
- Sequential processing using OpenCV-style convolution
- Zero padding for boundary handling
- Single-threaded execution

### CUDA Naive Implementation  
- One thread per output pixel
- Global memory access for input data
- Constant memory for convolution kernels

### CUDA Shared Memory Implementation
- Tiled approach with 16x16 blocks
- Shared memory for input data reuse
- Bank conflict avoidance strategies

### CUDA Separable Implementation
- Two-pass approach for separable filters
- Horizontal then vertical convolution
- Reduced computational complexity O(N²) → O(2N)

## Performance Results

Performance data has been collected and stored in CSV format in the `results/` directory:

- `gaussian_results.csv` - Basic Gaussian filter performance
- `perf_256_k3.csv` through `perf_1024_k7.csv` - Comprehensive benchmark data

### Key Findings
1. **CUDA Shared Memory** provides best performance for most scenarios
2. **Separable filters** show significant speedup for appropriate kernels
3. **GPU overhead** makes CPU competitive for very small images
4. **Kernel size** significantly impacts performance scaling

## Testing Environment
- GPU: NVIDIA GPU with CUDA support
- Image sizes: 256x256, 512x512, 1024x1024
- Kernel sizes: 3x3, 5x5, 7x7

For detailed numerical results, refer to the CSV files in the `results/` directory.