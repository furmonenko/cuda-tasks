#pragma once

#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <string>

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

class MatrixUtils {
public:
    static void generateRandomMatrix(float* matrix, int rows, int cols);
    static bool verifyTranspose(const float* original, const float* transposed, int rows, int cols);
    static void printMatrix(const float* matrix, int rows, int cols, int max_display = 8);
    static void printPerformanceResults(const std::string& method, double time_ms, int rows, int cols);
    static void logPerformanceResults(const std::string& method, double time_ms, int rows, int cols, 
                                    std::ofstream& csvFile, std::ofstream& logFile);
    static void initializeResultFiles(std::ofstream& csvFile, std::ofstream& logFile);
};

// CPU implementations
void cpuTransposeNaive(const float* input, float* output, int rows, int cols);

// CUDA implementations
void cudaTransposeNaive(const float* input, float* output, int rows, int cols);
void cudaTransposeOptimized(float* matrix, int rows, int cols);
void cudaTransposeUnified(float* matrix, int rows, int cols);

// Block size analysis implementations
void cudaTransposeBlockSize8(const float* input, float* output, int rows, int cols);
void cudaTransposeBlockSize16(const float* input, float* output, int rows, int cols);

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)