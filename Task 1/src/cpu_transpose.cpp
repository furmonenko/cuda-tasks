#include "matrix_utils.h"
#include <iomanip>

void cpuTransposeNaive(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

void MatrixUtils::generateRandomMatrix(float* matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

bool MatrixUtils::verifyTranspose(const float* original, const float* transposed, int rows, int cols) {
    const float epsilon = 1e-6f;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float original_val = original[i * cols + j];
            float transposed_val = transposed[j * rows + i];
            
            if (std::abs(original_val - transposed_val) > epsilon) {
                std::cout << "Mismatch at (" << i << "," << j << "): "
                          << "original=" << original_val 
                          << ", transposed=" << transposed_val << std::endl;
                return false;
            }
        }
    }
    return true;
}

void MatrixUtils::printMatrix(const float* matrix, int rows, int cols, int max_display) {
    int display_rows = std::min(rows, max_display);
    int display_cols = std::min(cols, max_display);
    
    for (int i = 0; i < display_rows; i++) {
        for (int j = 0; j < display_cols; j++) {
            std::cout << std::fixed << std::setprecision(2) << matrix[i * cols + j] << "\t";
        }
        if (cols > max_display) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > max_display) {
        std::cout << "..." << std::endl;
    }
}

void MatrixUtils::printPerformanceResults(const std::string& method, double time_ms, int rows, int cols) {
    double elements = static_cast<double>(rows) * cols;
    double bandwidth_gb_s = (elements * sizeof(float) * 2) / (time_ms * 1e-3) / 1e9; // Read + Write
    
    std::cout << method << ":\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << time_ms << " ms\n";
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s\n";
    std::cout << "  Elements/sec: " << std::scientific << std::setprecision(2) 
              << elements / (time_ms * 1e-3) << std::endl;
}

void MatrixUtils::logPerformanceResults(const std::string& method, double time_ms, int rows, int cols, 
                                       std::ofstream& csvFile, std::ofstream& logFile) {
    double elements = static_cast<double>(rows) * cols;
    double bandwidth_gb_s = (elements * sizeof(float) * 2) / (time_ms * 1e-3) / 1e9;
    double elements_per_sec = elements / (time_ms * 1e-3);
    double size_mb = (elements * sizeof(float)) / (1024.0 * 1024.0);
    
    // Write to CSV file
    csvFile << method << "," << rows << "," << cols << "," << std::fixed << std::setprecision(6) 
            << size_mb << "," << time_ms << "," << bandwidth_gb_s << "," 
            << std::scientific << elements_per_sec << std::endl;
    
    // Write to detailed log file
    logFile << "=== " << method << " ===" << std::endl;
    logFile << "Matrix Size: " << rows << "x" << cols << " (" << std::fixed << std::setprecision(2) 
            << size_mb << " MB)" << std::endl;
    logFile << "Time: " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
    logFile << "Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gb_s << " GB/s" << std::endl;
    logFile << "Elements/sec: " << std::scientific << std::setprecision(2) << elements_per_sec << std::endl;
    logFile << std::endl;
}

void MatrixUtils::initializeResultFiles(std::ofstream& csvFile, std::ofstream& logFile) {
    // Initialize CSV file with headers
    csvFile << "Method,Rows,Cols,Size_MB,Time_ms,Bandwidth_GB_s,Elements_per_sec" << std::endl;
    
    // Initialize log file with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    logFile << "Matrix Transpose Performance Results" << std::endl;
    logFile << "Generated on: " << std::ctime(&time_t) << std::endl;
}