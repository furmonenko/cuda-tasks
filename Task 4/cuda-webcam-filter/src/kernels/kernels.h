#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace cuda_filter
{

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    
    // Stream versions for multi-stream processing
    void applyFilterGPU_Stream(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel, cudaStream_t stream);
    
    // HDR tone mapping functions
    void applyHDRToneMappingGPU(const cv::Mat &input, cv::Mat &output, 
                               float exposure, float gamma, float saturation, int algorithm);
    void applyHDRToneMappingCPU(const cv::Mat &input, cv::Mat &output, 
                               float exposure, float gamma, float saturation, int algorithm);
                               
    // Stream version for HDR tone mapping
    void applyHDRToneMappingGPU_Stream(const cv::Mat &input, cv::Mat &output, 
                                      float exposure, float gamma, float saturation, int algorithm, cudaStream_t stream);

    namespace cuda
    {
// CUDA-specific type declarations and helper functions
#ifdef __CUDACC__
        // These will only be visible to CUDA compiler
        __host__ __device__ inline int divUp(int a, int b)
        {
            return (a + b - 1) / b;
        }
#endif
    }

} // namespace cuda_filter