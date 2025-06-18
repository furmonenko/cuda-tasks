#include "matrix_utils.h"
#include <cuda_runtime.h>
#include <iomanip>

// Naive CUDA kernel for matrix transpose
__global__ void transposeKernelNaive(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

// Optimized CUDA kernel using shared memory - configurable tile size
#define TILE_SIZE 32

template<int TILE_DIM>
__global__ void transposeKernelOptimizedTemplate(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load data into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Write transposed data to global memory
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Original kernel with 32x32 tiles
__global__ void transposeKernelOptimized(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load data into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Write transposed data to global memory
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void cudaTransposeNaive(const float* input, float* output, int rows, int cols) {
    float *d_input, *d_output;
    size_t size = rows * cols * sizeof(float);
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    transposeKernelNaive<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Wait for completion
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

void cudaTransposeOptimized(float* matrix, int rows, int cols) {
    float *d_matrix, *d_transposed;
    size_t size = rows * cols * sizeof(float);
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_transposed, size));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters for tiled approach
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cols + TILE_SIZE - 1) / TILE_SIZE, 
                  (rows + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch optimized kernel
    transposeKernelOptimized<<<gridSize, blockSize>>>(d_matrix, d_transposed, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(matrix, d_transposed, size, cudaMemcpyDeviceToHost));
    
    // Wait for completion
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Cleanup
    cudaFree(d_matrix);
    cudaFree(d_transposed);
}

// Unified Memory version
void cudaTransposeUnified(float* matrix, int rows, int cols) {
    float *unified_matrix, *unified_transposed;
    size_t size = rows * cols * sizeof(float);
    
    // Allocate unified memory
    CHECK_CUDA_ERROR(cudaMallocManaged(&unified_matrix, size));
    CHECK_CUDA_ERROR(cudaMallocManaged(&unified_transposed, size));
    
    // Copy original data
    memcpy(unified_matrix, matrix, size);
    
    // Configure kernel launch parameters
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cols + TILE_SIZE - 1) / TILE_SIZE, 
                  (rows + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel with unified memory
    transposeKernelOptimized<<<gridSize, blockSize>>>(unified_matrix, unified_transposed, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for completion
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy result back
    memcpy(matrix, unified_transposed, size);
    
    // Cleanup
    cudaFree(unified_matrix);
    cudaFree(unified_transposed);
}

// Block size analysis functions
void cudaTransposeBlockSize8(const float* input, float* output, int rows, int cols) {
    float *d_input, *d_output;
    size_t size = rows * cols * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice));
    
    dim3 blockSize(8, 8);
    dim3 gridSize((cols + 8 - 1) / 8, (rows + 8 - 1) / 8);
    
    transposeKernelOptimizedTemplate<8><<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void cudaTransposeBlockSize16(const float* input, float* output, int rows, int cols) {
    float *d_input, *d_output;
    size_t size = rows * cols * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + 16 - 1) / 16, (rows + 16 - 1) / 16);
    
    transposeKernelOptimizedTemplate<16><<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    cudaFree(d_input);
    cudaFree(d_output);
}