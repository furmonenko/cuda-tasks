#include "kernels.h"
#include <cuda_runtime.h>
#include <plog/Log.h>
#include <cmath>

namespace cuda_filter
{

// CUDA error checking
#define CHECK_CUDA_ERROR(call)                                                          \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err); \
            return;                                                                     \
        }                                                                               \
    }

    // CUDA kernel for 2D convolution
    __global__ void convolutionKernel(const unsigned char *input, unsigned char *output,
                                      const float *kernel, int width, int height,
                                      int channels, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int radius = kernelSize / 2;

        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);

                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    float pixelValue = input[(iy * width + ix) * channels + c];

                    sum += pixelValue * kernelValue;
                }
            }

            // Clamp the result to [0, 255]
            output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;

        // Allocate device memory
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        float *d_kernel = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);
        size_t kernelSize_bytes = kernelSize * kernelSize * sizeof(float);

        // Copy kernel to CPU float array
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelSize_bytes));

        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernelSize_bytes, cudaMemcpyHostToDevice));

        // Define block and grid dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        // Launch kernel
        convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, width, height, channels, kernelSize);

        // Check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Synchronize to ensure kernel execution is complete
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);

        // Free host memory
        delete[] h_kernel;
    }

    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;
        int radius = kernelSize / 2;

        // Convert kernel to float array for faster access
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Process each pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0.0f;

                    // Apply kernel
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ix = std::min(std::max(x + kx, 0), width - 1);
                            int iy = std::min(std::max(y + ky, 0), height - 1);

                            float kernelValue = h_kernel[(ky + radius) * kernelSize + (kx + radius)];
                            float pixelValue = input.at<cv::Vec3b>(iy, ix)[c];

                            sum += pixelValue * kernelValue;
                        }
                    }

                    // Clamp the result to [0, 255]
                    output.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
                }
            }
        }

        delete[] h_kernel;
    }

    void applyFilterGPU_Stream(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel, cudaStream_t stream)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;

        // Allocate device memory
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        float *d_kernel = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);
        size_t kernelSize_bytes = kernelSize * kernelSize * sizeof(float);

        // Copy kernel to CPU float array
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelSize_bytes));

        // Copy data to device using the provided stream
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, input.data, imageSize, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_kernel, h_kernel, kernelSize_bytes, cudaMemcpyHostToDevice, stream));

        // Define block and grid dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        // Launch kernel on the provided stream
        convolutionKernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, d_kernel, width, height, channels, kernelSize);

        // Check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Copy result back to host using the provided stream
        CHECK_CUDA_ERROR(cudaMemcpyAsync(output.data, d_output, imageSize, cudaMemcpyDeviceToHost, stream));

        // Note: Synchronization is handled by the caller
        // The caller should call cudaStreamSynchronize(stream) when needed

        // Free device memory (this is safe even with pending async operations)
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);

        // Free host memory
        delete[] h_kernel;
    }

    // HDR Tone Mapping CUDA Kernels

    // RGB to Luminance conversion
    __device__ float rgbToLuminance(float r, float g, float b)
    {
        return 0.299f * r + 0.587f * g + 0.114f * b;
    }

    // Reinhard tone mapping operator
    __device__ float reinhardToneMapping(float luminance, float exposure)
    {
        float L = luminance * exposure;
        return L / (1.0f + L);
    }

    // Drago tone mapping operator
    __device__ float dragoToneMapping(float luminance, float exposure, float bias = 0.85f)
    {
        float L = luminance * exposure;
        float Lmax = 1.0f; // Maximum luminance
        float Ldmax = 100.0f; // Display maximum luminance
        
        float logL = logf(L + 1e-6f);
        float logLmax = logf(Lmax + 1e-6f);
        float logLdmax = logf(Ldmax);
        
        float c1 = logLdmax / logLmax;
        float c2 = (bias * logLdmax) / logLmax;
        
        return c1 * (logL / (logL + c2));
    }

    // Mantiuk tone mapping operator (simplified)
    __device__ float mantiukToneMapping(float luminance, float exposure)
    {
        float L = luminance * exposure;
        return powf(L / (L + 1.0f), 0.6f);
    }

    // HDR Tone Mapping Kernel
    __global__ void hdrToneMappingKernel(const unsigned char *input, unsigned char *output,
                                        int width, int height, int channels,
                                        float exposure, float gamma, float saturation, int algorithm)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int idx = (y * width + x) * channels;

        // Convert input to float [0, 1]
        float r = input[idx] / 255.0f;
        float g = input[idx + 1] / 255.0f;
        float b = input[idx + 2] / 255.0f;

        // Calculate luminance
        float luminance = rgbToLuminance(r, g, b);

        // Apply tone mapping based on algorithm
        float toneMappedLuminance;
        switch (algorithm)
        {
        case 0: // Reinhard
            toneMappedLuminance = reinhardToneMapping(luminance, exposure);
            break;
        case 1: // Drago
            toneMappedLuminance = dragoToneMapping(luminance, exposure);
            break;
        case 2: // Mantiuk
            toneMappedLuminance = mantiukToneMapping(luminance, exposure);
            break;
        default:
            toneMappedLuminance = reinhardToneMapping(luminance, exposure);
            break;
        }

        // Preserve color ratios
        float ratio = (luminance > 1e-6f) ? (toneMappedLuminance / luminance) : 1.0f;
        
        // Apply ratio to RGB channels
        r *= ratio;
        g *= ratio;
        b *= ratio;

        // Apply saturation adjustment
        float gray = rgbToLuminance(r, g, b);
        r = gray + saturation * (r - gray);
        g = gray + saturation * (g - gray);
        b = gray + saturation * (b - gray);

        // Apply gamma correction
        r = powf(fmaxf(r, 0.0f), 1.0f / gamma);
        g = powf(fmaxf(g, 0.0f), 1.0f / gamma);
        b = powf(fmaxf(b, 0.0f), 1.0f / gamma);

        // Clamp and convert back to [0, 255]
        output[idx] = static_cast<unsigned char>(fminf(fmaxf(r * 255.0f, 0.0f), 255.0f));
        output[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(g * 255.0f, 0.0f), 255.0f));
        output[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(b * 255.0f, 0.0f), 255.0f));
    }

    void applyHDRToneMappingGPU(const cv::Mat &input, cv::Mat &output,
                               float exposure, float gamma, float saturation, int algorithm)
    {
        if (input.empty())
        {
            PLOG_ERROR << "Input image is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();

        // Allocate device memory
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));

        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));

        // Define block and grid dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        // Launch HDR tone mapping kernel
        hdrToneMappingKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels,
                                                   exposure, gamma, saturation, algorithm);

        // Check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Synchronize to ensure kernel execution is complete
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
    }

    void applyHDRToneMappingCPU(const cv::Mat &input, cv::Mat &output,
                               float exposure, float gamma, float saturation, int algorithm)
    {
        if (input.empty())
        {
            PLOG_ERROR << "Input image is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();

        // Process each pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
                
                // Convert to float [0, 1]
                float r = pixel[2] / 255.0f; // OpenCV uses BGR
                float g = pixel[1] / 255.0f;
                float b = pixel[0] / 255.0f;

                // Calculate luminance
                float luminance = 0.299f * r + 0.587f * g + 0.114f * b;

                // Apply tone mapping
                float toneMappedLuminance;
                switch (algorithm)
                {
                case 0: // Reinhard
                    {
                        float L = luminance * exposure;
                        toneMappedLuminance = L / (1.0f + L);
                    }
                    break;
                case 1: // Drago
                    {
                        float L = luminance * exposure;
                        float bias = 0.85f;
                        float Lmax = 1.0f;
                        float Ldmax = 100.0f;
                        
                        float logL = std::log(L + 1e-6f);
                        float logLmax = std::log(Lmax + 1e-6f);
                        float logLdmax = std::log(Ldmax);
                        
                        float c1 = logLdmax / logLmax;
                        float c2 = (bias * logLdmax) / logLmax;
                        
                        toneMappedLuminance = c1 * (logL / (logL + c2));
                    }
                    break;
                case 2: // Mantiuk
                    {
                        float L = luminance * exposure;
                        toneMappedLuminance = std::pow(L / (L + 1.0f), 0.6f);
                    }
                    break;
                default:
                    {
                        float L = luminance * exposure;
                        toneMappedLuminance = L / (1.0f + L);
                    }
                    break;
                }

                // Preserve color ratios
                float ratio = (luminance > 1e-6f) ? (toneMappedLuminance / luminance) : 1.0f;
                
                r *= ratio;
                g *= ratio;
                b *= ratio;

                // Apply saturation adjustment
                float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                r = gray + saturation * (r - gray);
                g = gray + saturation * (g - gray);
                b = gray + saturation * (b - gray);

                // Apply gamma correction
                r = std::pow(std::max(r, 0.0f), 1.0f / gamma);
                g = std::pow(std::max(g, 0.0f), 1.0f / gamma);
                b = std::pow(std::max(b, 0.0f), 1.0f / gamma);

                // Clamp and convert back to [0, 255]
                cv::Vec3b outputPixel;
                outputPixel[0] = static_cast<unsigned char>(std::min(std::max(b * 255.0f, 0.0f), 255.0f)); // B
                outputPixel[1] = static_cast<unsigned char>(std::min(std::max(g * 255.0f, 0.0f), 255.0f)); // G
                outputPixel[2] = static_cast<unsigned char>(std::min(std::max(r * 255.0f, 0.0f), 255.0f)); // R

                output.at<cv::Vec3b>(y, x) = outputPixel;
            }
        }
    }

    void applyHDRToneMappingGPU_Stream(const cv::Mat &input, cv::Mat &output,
                                      float exposure, float gamma, float saturation, int algorithm, cudaStream_t stream)
    {
        if (input.empty())
        {
            PLOG_ERROR << "Input image is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();

        // Allocate device memory
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));

        // Copy data to device using the provided stream
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, input.data, imageSize, cudaMemcpyHostToDevice, stream));

        // Define block and grid dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        // Launch HDR tone mapping kernel on the provided stream
        hdrToneMappingKernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, width, height, channels,
                                                              exposure, gamma, saturation, algorithm);

        // Check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Copy result back to host using the provided stream
        CHECK_CUDA_ERROR(cudaMemcpyAsync(output.data, d_output, imageSize, cudaMemcpyDeviceToHost, stream));

        // Note: Synchronization is handled by the caller
        // The caller should call cudaStreamSynchronize(stream) when needed

        // Free device memory (this is safe even with pending async operations)
        cudaFree(d_input);
        cudaFree(d_output);
    }

} // namespace cuda_filter
