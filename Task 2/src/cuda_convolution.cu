#include "convolution.h"
#include <cuda_runtime.h>
#include <iostream>

// Constant memory for small kernels
__constant__ float d_kernel[49]; // Max 7x7 kernel

// Naive CUDA kernel - one thread per output pixel
__global__ void convolutionNaiveKernel(const float* input, float* output,
                                      int width, int height, int channels,
                                      int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int kernelRadius = kernelSize / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                int inputX = x + kx;
                int inputY = y + ky;
                
                // Zero padding for boundaries
                if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
                    int kernelIdx = (ky + kernelRadius) * kernelSize + (kx + kernelRadius);
                    int inputIdx = (inputY * width + inputX) * channels + c;
                    sum += input[inputIdx] * d_kernel[kernelIdx];
                }
            }
        }
        
        int outputIdx = (y * width + x) * channels + c;
        output[outputIdx] = sum;
    }
}

// Shared memory kernel with tiling
#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 7

__global__ void convolutionSharedMemoryKernel(const float* input, float* output,
                                             int width, int height, int channels,
                                             int kernelSize) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int x = bx * TILE_SIZE + tx;
    int y = by * TILE_SIZE + ty;
    
    int kernelRadius = kernelSize / 2;
    int sharedSize = TILE_SIZE + 2 * kernelRadius;
    
    // Shared memory for tile + halo
    extern __shared__ float sharedInput[];
    
    // Load data into shared memory
    for (int c = 0; c < channels; c++) {
        int sharedOffset = c * sharedSize * sharedSize;
        
        // Each thread loads multiple elements if needed
        for (int sy = ty; sy < sharedSize; sy += blockDim.y) {
            for (int sx = tx; sx < sharedSize; sx += blockDim.x) {
                int inputX = bx * TILE_SIZE + sx - kernelRadius;
                int inputY = by * TILE_SIZE + sy - kernelRadius;
                
                float value = 0.0f;
                if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
                    int inputIdx = (inputY * width + inputX) * channels + c;
                    value = input[inputIdx];
                }
                
                sharedInput[sharedOffset + sy * sharedSize + sx] = value;
            }
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            int sharedOffset = c * sharedSize * sharedSize;
            
            for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
                for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                    int sharedX = tx + kernelRadius + kx;
                    int sharedY = ty + kernelRadius + ky;
                    int kernelIdx = (ky + kernelRadius) * kernelSize + (kx + kernelRadius);
                    
                    sum += sharedInput[sharedOffset + sharedY * sharedSize + sharedX] * d_kernel[kernelIdx];
                }
            }
            
            int outputIdx = (y * width + x) * channels + c;
            output[outputIdx] = sum;
        }
    }
}

// Separable convolution kernels
__global__ void convolutionSeparableHorizontal(const float* input, float* output,
                                              int width, int height, int channels,
                                              const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int kernelRadius = kernelSize / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int k = -kernelRadius; k <= kernelRadius; k++) {
            int inputX = x + k;
            if (inputX >= 0 && inputX < width) {
                int inputIdx = (y * width + inputX) * channels + c;
                sum += input[inputIdx] * kernel[k + kernelRadius];
            }
        }
        
        int outputIdx = (y * width + x) * channels + c;
        output[outputIdx] = sum;
    }
}

__global__ void convolutionSeparableVertical(const float* input, float* output,
                                            int width, int height, int channels,
                                            const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int kernelRadius = kernelSize / 2;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int k = -kernelRadius; k <= kernelRadius; k++) {
            int inputY = y + k;
            if (inputY >= 0 && inputY < height) {
                int inputIdx = (inputY * width + x) * channels + c;
                sum += input[inputIdx] * kernel[k + kernelRadius];
            }
        }
        
        int outputIdx = (y * width + x) * channels + c;
        output[outputIdx] = sum;
    }
}

// Implementation functions
void cudaConvolutionNaive(const Image& input, Image& output, const ConvolutionKernel& kernel) {
    // Copy kernel to constant memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_kernel, kernel.data.data(), 
                                        kernel.data.size() * sizeof(float)));
    
    // Allocate device memory
    float *d_input, *d_output;
    size_t imageSize = input.sizeInBytes();
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data.data(), imageSize, cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    convolutionNaiveKernel<<<gridSize, blockSize>>>(d_input, d_output,
                                                    input.width, input.height, input.channels,
                                                    kernel.size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(output.data.data(), d_output, imageSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

void cudaConvolutionSharedMemory(const Image& input, Image& output, const ConvolutionKernel& kernel) {
    // Copy kernel to constant memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_kernel, kernel.data.data(), 
                                        kernel.data.size() * sizeof(float)));
    
    // Allocate device memory
    float *d_input, *d_output;
    size_t imageSize = input.sizeInBytes();
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data.data(), imageSize, cudaMemcpyHostToDevice));
    
    // Calculate shared memory size
    int kernelRadius = kernel.size / 2;
    int sharedSize = TILE_SIZE + 2 * kernelRadius;
    size_t sharedMemSize = input.channels * sharedSize * sharedSize * sizeof(float);
    
    // Launch kernel
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((input.width + TILE_SIZE - 1) / TILE_SIZE,
                  (input.height + TILE_SIZE - 1) / TILE_SIZE);
    
    convolutionSharedMemoryKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_input, d_output, input.width, input.height, input.channels, kernel.size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(output.data.data(), d_output, imageSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

void cudaConvolutionConstantMemory(const Image& input, Image& output, const ConvolutionKernel& kernel) {
    // This is the same as naive but uses constant memory (already implemented above)
    cudaConvolutionNaive(input, output, kernel);
}

void cudaConvolutionSeparable(const Image& input, Image& output, 
                            const std::vector<float>& kernelX, 
                            const std::vector<float>& kernelY) {
    // Allocate device memory
    float *d_input, *d_temp, *d_output;
    float *d_kernelX, *d_kernelY;
    size_t imageSize = input.sizeInBytes();
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernelX, kernelX.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernelY, kernelY.size() * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data.data(), imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernelX, kernelX.data(), kernelX.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernelY, kernelY.data(), kernelY.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernels
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    // Horizontal pass
    convolutionSeparableHorizontal<<<gridSize, blockSize>>>(
        d_input, d_temp, input.width, input.height, input.channels, d_kernelX, kernelX.size());
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Vertical pass
    convolutionSeparableVertical<<<gridSize, blockSize>>>(
        d_temp, d_output, input.width, input.height, input.channels, d_kernelY, kernelY.size());
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(output.data.data(), d_output, imageSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
    cudaFree(d_kernelX);
    cudaFree(d_kernelY);
}