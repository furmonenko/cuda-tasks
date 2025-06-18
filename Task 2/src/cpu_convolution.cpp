#include "convolution.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>

void cpuConvolution2D(const Image& input, Image& output, const ConvolutionKernel& kernel) {
    int kernelRadius = kernel.size / 2;
    
    for (int y = 0; y < output.height; y++) {
        for (int x = 0; x < output.width; x++) {
            for (int c = 0; c < output.channels; c++) {
                float sum = 0.0f;
                
                for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
                    for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                        int inputX = x + kx;
                        int inputY = y + ky;
                        
                        // Handle boundaries with zero padding
                        if (inputX >= 0 && inputX < input.width && 
                            inputY >= 0 && inputY < input.height) {
                            float kernelValue = kernel(kx + kernelRadius, ky + kernelRadius);
                            sum += input(inputX, inputY, c) * kernelValue;
                        }
                    }
                }
                
                output(x, y, c) = sum;
            }
        }
    }
}

void cpuConvolutionSeparable(const Image& input, Image& output, 
                           const std::vector<float>& kernelX, 
                           const std::vector<float>& kernelY) {
    
    // Create temporary image for intermediate result
    Image temp(input.width, input.height, input.channels);
    
    int radiusX = kernelX.size() / 2;
    int radiusY = kernelY.size() / 2;
    
    // First pass: horizontal convolution
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            for (int c = 0; c < input.channels; c++) {
                float sum = 0.0f;
                
                for (int k = -radiusX; k <= radiusX; k++) {
                    int inputX = x + k;
                    if (inputX >= 0 && inputX < input.width) {
                        sum += input(inputX, y, c) * kernelX[k + radiusX];
                    }
                }
                
                temp(x, y, c) = sum;
            }
        }
    }
    
    // Second pass: vertical convolution
    for (int y = 0; y < output.height; y++) {
        for (int x = 0; x < output.width; x++) {
            for (int c = 0; c < output.channels; c++) {
                float sum = 0.0f;
                
                for (int k = -radiusY; k <= radiusY; k++) {
                    int inputY = y + k;
                    if (inputY >= 0 && inputY < temp.height) {
                        sum += temp(x, inputY, c) * kernelY[k + radiusY];
                    }
                }
                
                output(x, y, c) = sum;
            }
        }
    }
}

void ConvolutionBenchmark::printResults(const std::vector<ConvolutionResult>& results) {
    std::cout << "\n=== Convolution Performance Results ===" << std::endl;
    std::cout << "Method                    | Time (ms) | Throughput (MP/s) | Bandwidth (GB/s) | Correct" << std::endl;
    std::cout << "--------------------------|-----------|-------------------|------------------|--------" << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(25) << result.method << " | "
                  << std::right << std::setw(9) << std::fixed << std::setprecision(3) << result.timeMs << " | "
                  << std::setw(17) << std::setprecision(2) << result.throughputMPixelPerSec << " | "
                  << std::setw(16) << std::setprecision(2) << result.bandwidthGBps << " | "
                  << (result.isCorrect ? "  YES" : "   NO") << std::endl;
    }
}

void ConvolutionBenchmark::saveResults(const std::vector<ConvolutionResult>& results, 
                                      const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "Method,Time_ms,Throughput_MP_s,Bandwidth_GB_s,Correct\n";
    for (const auto& result : results) {
        file << result.method << "," 
             << std::fixed << std::setprecision(6) << result.timeMs << ","
             << result.throughputMPixelPerSec << ","
             << result.bandwidthGBps << ","
             << (result.isCorrect ? "true" : "false") << "\n";
    }
}