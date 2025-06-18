#include "convolution.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>

std::vector<ConvolutionResult> ConvolutionBenchmark::runBenchmarks(const Image& input, 
                                                                  const ConvolutionKernel& kernel,
                                                                  int iterations) {
    std::vector<ConvolutionResult> results;
    
    // Create output images
    Image cpuOutput(input.width, input.height, input.channels);
    Image gpuNaiveOutput(input.width, input.height, input.channels);
    Image gpuSharedOutput(input.width, input.height, input.channels);
    Image gpuConstantOutput(input.width, input.height, input.channels);
    
    Timer timer;
    
    // Calculate throughput metrics
    double megaPixels = (input.width * input.height) / 1e6;
    double dataSize = input.sizeInBytes() + cpuOutput.sizeInBytes(); // Read + Write
    
    // CPU Benchmark
    std::cout << "Running CPU convolution..." << std::endl;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cpuConvolution2D(input, cpuOutput, kernel);
    }
    timer.stop();
    
    double cpuTime = timer.getElapsedMilliseconds() / iterations;
    results.push_back({
        "CPU_Sequential",
        cpuTime,
        megaPixels / (cpuTime / 1000.0),
        (dataSize / 1e9) / (cpuTime / 1000.0),
        true
    });
    
    // CUDA Naive Benchmark
    std::cout << "Running CUDA naive convolution..." << std::endl;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cudaConvolutionNaive(input, gpuNaiveOutput, kernel);
    }
    timer.stop();
    
    double naiveTime = timer.getElapsedMilliseconds() / iterations;
    bool naiveCorrect = ImageUtils::compareImages(cpuOutput, gpuNaiveOutput, 1e-3f);
    results.push_back({
        "CUDA_Naive",
        naiveTime,
        megaPixels / (naiveTime / 1000.0),
        (dataSize / 1e9) / (naiveTime / 1000.0),
        naiveCorrect
    });
    
    // CUDA Shared Memory Benchmark
    std::cout << "Running CUDA shared memory convolution..." << std::endl;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        cudaConvolutionSharedMemory(input, gpuSharedOutput, kernel);
    }
    timer.stop();
    
    double sharedTime = timer.getElapsedMilliseconds() / iterations;
    bool sharedCorrect = ImageUtils::compareImages(cpuOutput, gpuSharedOutput, 1e-3f);
    results.push_back({
        "CUDA_SharedMemory",
        sharedTime,
        megaPixels / (sharedTime / 1000.0),
        (dataSize / 1e9) / (sharedTime / 1000.0),
        sharedCorrect
    });
    
    // CUDA Constant Memory (same as naive but different name for clarity)
    results.push_back({
        "CUDA_ConstantMemory",
        naiveTime,
        megaPixels / (naiveTime / 1000.0),
        (dataSize / 1e9) / (naiveTime / 1000.0),
        naiveCorrect
    });
    
    return results;
}

void runBasicTest() {
    std::cout << "\n=== Basic Convolution Test ===" << std::endl;
    
    // Create test image
    auto input = ImageUtils::createTestImage(512, 512, 1);
    ImageUtils::printImageStats(*input, "Input Image");
    
    // Create test kernels
    auto gaussianKernel = ImageUtils::createGaussianKernel(5, 1.0f);
    auto sobelKernel = ImageUtils::createSobelXKernel();
    auto sharpenKernel = ImageUtils::createSharpenKernel();
    
    std::cout << "\nTesting Gaussian Blur (5x5)..." << std::endl;
    auto results = ConvolutionBenchmark::runBenchmarks(*input, gaussianKernel, 5);
    ConvolutionBenchmark::printResults(results);
    
    // Save results
    system("mkdir ..\\results 2>nul || mkdir -p ../results 2>/dev/null || true");
    ConvolutionBenchmark::saveResults(results, "../results/gaussian_results.csv");
}

void runSeparableTest() {
    std::cout << "\n=== Separable Filter Test ===" << std::endl;
    
    auto input = ImageUtils::createTestImage(512, 512, 1);
    
    // Create separable Gaussian kernel
    std::vector<float> gaussianKernel1D = {0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f};
    
    Image output2D(input->width, input->height, input->channels);
    Image outputSeparable(input->width, input->height, input->channels);
    
    Timer timer;
    
    // 2D convolution
    auto gaussian2D = ImageUtils::createGaussianKernel(5, 1.0f);
    timer.start();
    cudaConvolutionNaive(*input, output2D, gaussian2D);
    timer.stop();
    double time2D = timer.getElapsedMilliseconds();
    
    // Separable convolution
    timer.start();
    cudaConvolutionSeparable(*input, outputSeparable, gaussianKernel1D, gaussianKernel1D);
    timer.stop();
    double timeSeparable = timer.getElapsedMilliseconds();
    
    bool isCorrect = ImageUtils::compareImages(output2D, outputSeparable, 1e-2f);
    
    std::cout << "2D Convolution time: " << time2D << " ms" << std::endl;
    std::cout << "Separable Convolution time: " << timeSeparable << " ms" << std::endl;
    std::cout << "Speedup: " << time2D / timeSeparable << "x" << std::endl;
    std::cout << "Correctness: " << (isCorrect ? "PASS" : "FAIL") << std::endl;
}

void runPerformanceAnalysis() {
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    
    std::vector<int> imageSizes = {256, 512, 1024};
    std::vector<int> kernelSizes = {3, 5, 7};
    
    for (int imageSize : imageSizes) {
        for (int kernelSize : kernelSizes) {
            std::cout << "\nTesting " << imageSize << "x" << imageSize 
                      << " image with " << kernelSize << "x" << kernelSize << " kernel" << std::endl;
            
            auto input = ImageUtils::createTestImage(imageSize, imageSize, 1);
            auto kernel = ImageUtils::createGaussianKernel(kernelSize, 1.0f);
            
            auto results = ConvolutionBenchmark::runBenchmarks(*input, kernel, 3);
            ConvolutionBenchmark::printResults(results);
            
            // Save results
            std::string filename = "../results/perf_" + std::to_string(imageSize) + 
                                 "_k" + std::to_string(kernelSize) + ".csv";
            ConvolutionBenchmark::saveResults(results, filename);
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA Image Convolution Performance Analysis" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using GPU: " << deviceProp.name << std::endl;
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Constant Memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
    
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--separable") {
            runSeparableTest();
        } else if (arg == "--benchmark") {
            runPerformanceAnalysis();
        } else {
            std::cout << "Usage:" << std::endl;
            std::cout << "  " << argv[0] << "            (basic test)" << std::endl;
            std::cout << "  " << argv[0] << " --separable  (separable filter test)" << std::endl;
            std::cout << "  " << argv[0] << " --benchmark  (performance analysis)" << std::endl;
            return 1;
        }
    } else {
        runBasicTest();
    }
    
    return 0;
}