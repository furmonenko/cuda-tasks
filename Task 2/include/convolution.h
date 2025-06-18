#pragma once

#include <vector>
#include <chrono>
#include <string>
#include <memory>

// Timer utility class
class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }
    
    double getElapsedMilliseconds() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time, end_time;
};

// Image structure
struct Image {
    int width;
    int height;
    int channels;
    std::vector<float> data;
    
    Image(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c) {}
    
    float& operator()(int x, int y, int c = 0) {
        return data[y * width * channels + x * channels + c];
    }
    
    const float& operator()(int x, int y, int c = 0) const {
        return data[y * width * channels + x * channels + c];
    }
    
    size_t size() const { return data.size(); }
    size_t sizeInBytes() const { return data.size() * sizeof(float); }
};

// Convolution kernel structure
struct ConvolutionKernel {
    int size;
    std::vector<float> data;
    
    ConvolutionKernel(int s) : size(s), data(s * s) {}
    
    float& operator()(int x, int y) {
        return data[y * size + x];
    }
    
    const float& operator()(int x, int y) const {
        return data[y * size + x];
    }
};

// Image utilities
class ImageUtils {
public:
    static std::unique_ptr<Image> createTestImage(int width, int height, int channels = 1);
    static std::unique_ptr<Image> loadImage(const std::string& filename);
    static void saveImage(const Image& img, const std::string& filename);
    static bool compareImages(const Image& img1, const Image& img2, float tolerance = 1e-5f);
    static void printImageStats(const Image& img, const std::string& name);
    
    // Predefined kernels
    static ConvolutionKernel createGaussianKernel(int size, float sigma);
    static ConvolutionKernel createSobelXKernel();
    static ConvolutionKernel createSobelYKernel();
    static ConvolutionKernel createLaplacianKernel();
    static ConvolutionKernel createBoxBlurKernel(int size);
    static ConvolutionKernel createSharpenKernel();
};

// CPU implementations
void cpuConvolution2D(const Image& input, Image& output, const ConvolutionKernel& kernel);
void cpuConvolutionSeparable(const Image& input, Image& output, 
                           const std::vector<float>& kernelX, 
                           const std::vector<float>& kernelY);

// CUDA implementations
void cudaConvolutionNaive(const Image& input, Image& output, const ConvolutionKernel& kernel);
void cudaConvolutionSharedMemory(const Image& input, Image& output, const ConvolutionKernel& kernel);
void cudaConvolutionConstantMemory(const Image& input, Image& output, const ConvolutionKernel& kernel);
void cudaConvolutionSeparable(const Image& input, Image& output, 
                            const std::vector<float>& kernelX, 
                            const std::vector<float>& kernelY);

// Performance testing
struct ConvolutionResult {
    std::string method;
    double timeMs;
    double throughputMPixelPerSec;
    double bandwidthGBps;
    bool isCorrect;
};

class ConvolutionBenchmark {
public:
    static std::vector<ConvolutionResult> runBenchmarks(const Image& input, 
                                                       const ConvolutionKernel& kernel,
                                                       int iterations = 10);
    static void saveResults(const std::vector<ConvolutionResult>& results, 
                           const std::string& filename);
    static void printResults(const std::vector<ConvolutionResult>& results);
};

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)