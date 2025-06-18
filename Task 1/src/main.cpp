#include "matrix_utils.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cuda_runtime.h>

void runPerformanceTests(int rows, int cols, std::ofstream* csvFile = nullptr, std::ofstream* logFile = nullptr) {
    std::cout << "\n=== Matrix Transpose Performance Test ===" << std::endl;
    std::cout << "Matrix size: " << rows << " x " << cols 
              << " (" << (rows * cols * sizeof(float) / 1024.0 / 1024.0) << " MB)" << std::endl;
    
    // Allocate host memory
    size_t size = rows * cols * sizeof(float);
    std::unique_ptr<float[]> original(new float[rows * cols]);
    std::unique_ptr<float[]> cpu_result(new float[rows * cols]);
    std::unique_ptr<float[]> gpu_naive_result(new float[rows * cols]);
    std::unique_ptr<float[]> gpu_optimized_result(new float[rows * cols]);
    std::unique_ptr<float[]> gpu_unified_result(new float[rows * cols]);
    
    // Generate random test data
    MatrixUtils::generateRandomMatrix(original.get(), rows, cols);
    
    Timer timer;
    
    // Test CPU implementation
    std::cout << "\nRunning CPU transpose..." << std::endl;
    timer.start();
    cpuTransposeNaive(original.get(), cpu_result.get(), rows, cols);
    timer.stop();
    MatrixUtils::printPerformanceResults("CPU Naive", timer.getElapsedMilliseconds(), rows, cols);
    if (csvFile && logFile) {
        MatrixUtils::logPerformanceResults("CPU_Naive", timer.getElapsedMilliseconds(), rows, cols, *csvFile, *logFile);
    }
    
    // Test naive CUDA implementation
    std::cout << "\nRunning naive CUDA transpose..." << std::endl;
    timer.start();
    cudaTransposeNaive(original.get(), gpu_naive_result.get(), rows, cols);
    timer.stop();
    MatrixUtils::printPerformanceResults("CUDA Naive", timer.getElapsedMilliseconds(), rows, cols);
    if (csvFile && logFile) {
        MatrixUtils::logPerformanceResults("CUDA_Naive", timer.getElapsedMilliseconds(), rows, cols, *csvFile, *logFile);
    }
    
    // Verify correctness
    if (MatrixUtils::verifyTranspose(original.get(), gpu_naive_result.get(), rows, cols)) {
        std::cout << "PASS - CUDA Naive result is correct" << std::endl;
    } else {
        std::cout << "FAIL - CUDA Naive result is incorrect!" << std::endl;
    }
    
    // Test optimized CUDA implementation
    std::cout << "\nRunning optimized CUDA transpose..." << std::endl;
    std::copy(original.get(), original.get() + rows * cols, gpu_optimized_result.get());
    timer.start();
    cudaTransposeOptimized(gpu_optimized_result.get(), rows, cols);
    timer.stop();
    MatrixUtils::printPerformanceResults("CUDA Optimized", timer.getElapsedMilliseconds(), rows, cols);
    if (csvFile && logFile) {
        MatrixUtils::logPerformanceResults("CUDA_Optimized", timer.getElapsedMilliseconds(), rows, cols, *csvFile, *logFile);
    }
    
    // Verify correctness
    if (MatrixUtils::verifyTranspose(original.get(), gpu_optimized_result.get(), rows, cols)) {
        std::cout << "PASS - CUDA Optimized result is correct" << std::endl;
    } else {
        std::cout << "FAIL - CUDA Optimized result is incorrect!" << std::endl;
    }
    
    // Test unified memory CUDA implementation
    std::cout << "\nRunning unified memory CUDA transpose..." << std::endl;
    std::copy(original.get(), original.get() + rows * cols, gpu_unified_result.get());
    timer.start();
    cudaTransposeUnified(gpu_unified_result.get(), rows, cols);
    timer.stop();
    MatrixUtils::printPerformanceResults("CUDA Unified Memory", timer.getElapsedMilliseconds(), rows, cols);
    if (csvFile && logFile) {
        MatrixUtils::logPerformanceResults("CUDA_Unified", timer.getElapsedMilliseconds(), rows, cols, *csvFile, *logFile);
    }
    
    // Verify correctness
    if (MatrixUtils::verifyTranspose(original.get(), gpu_unified_result.get(), rows, cols)) {
        std::cout << "PASS - CUDA Unified Memory result is correct" << std::endl;
    } else {
        std::cout << "FAIL - CUDA Unified Memory result is incorrect!" << std::endl;
    }
    
    // Print sample of original and transposed matrices (for small matrices)
    if (rows <= 8 && cols <= 8) {
        std::cout << "\nOriginal matrix:" << std::endl;
        MatrixUtils::printMatrix(original.get(), rows, cols);
        std::cout << "\nTransposed matrix (CPU):" << std::endl;
        MatrixUtils::printMatrix(cpu_result.get(), cols, rows);
    }
}

void runScalabilityTest() {
    std::cout << "\n=== Scalability Analysis ===" << std::endl;
    
    // Create results directory if it doesn't exist
    #ifdef _WIN32
        system("if not exist ..\\results mkdir ..\\results");
    #else
        system("mkdir -p ../results");
    #endif
    
    // Open result files
    std::ofstream csvFile("../results/scalability_results.csv");
    std::ofstream logFile("../results/scalability_results.log");
    
    if (!csvFile.is_open() || !logFile.is_open()) {
        std::cerr << "Error: Could not open result files for writing!" << std::endl;
        return;
    }
    
    MatrixUtils::initializeResultFiles(csvFile, logFile);
    
    struct TestSize {
        int rows, cols;
        std::string description;
    };
    
    std::vector<TestSize> test_sizes = {
        {512, 512, "Small (512x512)"},
        {1024, 1024, "Medium (1024x1024)"},
        {2048, 1024, "Assignment size (2048x1024)"},
        {2048, 2048, "Large (2048x2048)"},
        {4096, 2048, "Very Large (4096x2048)"}
    };
    
    for (const auto& test : test_sizes) {
        std::cout << "\n--- " << test.description << " ---" << std::endl;
        logFile << "\n--- " << test.description << " ---" << std::endl;
        runPerformanceTests(test.rows, test.cols, &csvFile, &logFile);
    }
    
    csvFile.close();
    logFile.close();
    
    std::cout << "\n=== Results saved to files ===" << std::endl;
    std::cout << "CSV data: results/scalability_results.csv" << std::endl;
    std::cout << "Detailed log: results/scalability_results.log" << std::endl;
}

void runBlockSizeAnalysis() {
    std::cout << "\n=== Block Size Analysis ===" << std::endl;
    
    // Test on assignment size matrix
    const int rows = 2048, cols = 1024;
    
    // Create results directory if it doesn't exist
    #ifdef _WIN32
        system("if not exist ..\\results mkdir ..\\results");
    #else
        system("mkdir -p ../results");
    #endif
    
    // Open result files
    std::ofstream csvFile("../results/block_size_results.csv");
    std::ofstream logFile("../results/block_size_results.log");
    
    if (!csvFile.is_open() || !logFile.is_open()) {
        std::cerr << "Error: Could not open block size result files!" << std::endl;
        return;
    }
    
    MatrixUtils::initializeResultFiles(csvFile, logFile);
    
    std::cout << "Testing matrix size: " << rows << "x" << cols 
              << " (" << (rows * cols * sizeof(float) / 1024.0 / 1024.0) << " MB)" << std::endl;
    logFile << "Block Size Analysis - Matrix: " << rows << "x" << cols << std::endl;
    
    // Allocate host memory
    size_t size = rows * cols * sizeof(float);
    std::unique_ptr<float[]> original(new float[rows * cols]);
    std::unique_ptr<float[]> result_8x8(new float[rows * cols]);
    std::unique_ptr<float[]> result_16x16(new float[rows * cols]);
    std::unique_ptr<float[]> result_32x32(new float[rows * cols]);
    
    // Generate test data
    MatrixUtils::generateRandomMatrix(original.get(), rows, cols);
    
    Timer timer;
    
    // Test 8x8 blocks
    std::cout << "\nTesting 8x8 block size..." << std::endl;
    timer.start();
    cudaTransposeBlockSize8(original.get(), result_8x8.get(), rows, cols);
    timer.stop();
    MatrixUtils::printPerformanceResults("CUDA 8x8 Blocks", timer.getElapsedMilliseconds(), rows, cols);
    MatrixUtils::logPerformanceResults("CUDA_8x8", timer.getElapsedMilliseconds(), rows, cols, csvFile, logFile);
    
    // Verify correctness
    if (MatrixUtils::verifyTranspose(original.get(), result_8x8.get(), rows, cols)) {
        std::cout << "PASS - 8x8 blocks result is correct" << std::endl;
    } else {
        std::cout << "FAIL - 8x8 blocks result is incorrect!" << std::endl;
    }
    
    // Test 16x16 blocks
    std::cout << "\nTesting 16x16 block size..." << std::endl;
    timer.start();
    cudaTransposeBlockSize16(original.get(), result_16x16.get(), rows, cols);
    timer.stop();
    MatrixUtils::printPerformanceResults("CUDA 16x16 Blocks", timer.getElapsedMilliseconds(), rows, cols);
    MatrixUtils::logPerformanceResults("CUDA_16x16", timer.getElapsedMilliseconds(), rows, cols, csvFile, logFile);
    
    // Verify correctness
    if (MatrixUtils::verifyTranspose(original.get(), result_16x16.get(), rows, cols)) {
        std::cout << "PASS - 16x16 blocks result is correct" << std::endl;
    } else {
        std::cout << "FAIL - 16x16 blocks result is incorrect!" << std::endl;
    }
    
    // Test 32x32 blocks (using existing optimized function)
    std::cout << "\nTesting 32x32 block size..." << std::endl;
    std::copy(original.get(), original.get() + rows * cols, result_32x32.get());
    timer.start();
    cudaTransposeOptimized(result_32x32.get(), rows, cols);
    timer.stop();
    MatrixUtils::printPerformanceResults("CUDA 32x32 Blocks", timer.getElapsedMilliseconds(), rows, cols);
    MatrixUtils::logPerformanceResults("CUDA_32x32", timer.getElapsedMilliseconds(), rows, cols, csvFile, logFile);
    
    // Verify correctness
    if (MatrixUtils::verifyTranspose(original.get(), result_32x32.get(), rows, cols)) {
        std::cout << "PASS - 32x32 blocks result is correct" << std::endl;
    } else {
        std::cout << "FAIL - 32x32 blocks result is incorrect!" << std::endl;
    }
    
    csvFile.close();
    logFile.close();
    
    std::cout << "\n=== Block Size Results saved to files ===" << std::endl;
    std::cout << "CSV data: results/block_size_results.csv" << std::endl;
    std::cout << "Detailed log: results/block_size_results.log" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA Matrix Transpose Performance Analysis" << std::endl;
    std::cout << "===========================================" << std::endl;
    
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
    
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--scale") {
            runScalabilityTest();
        } else if (arg == "--blocks") {
            runBlockSizeAnalysis();
        } else {
            std::cout << "Usage:" << std::endl;
            std::cout << "  " << argv[0] << "          (default 2048x1024 test)" << std::endl;
            std::cout << "  " << argv[0] << " --scale  (scalability analysis)" << std::endl;
            std::cout << "  " << argv[0] << " --blocks (block size analysis)" << std::endl;
            return 1;
        }
    } else {
        // Default: run with assignment specified size
        runPerformanceTests(2048, 1024);
    }
    
    return 0;
}