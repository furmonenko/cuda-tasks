#include "convolution.h"
#include <iostream>
#include <random>
#include <fstream>
#include <cmath>
#include <algorithm>

std::unique_ptr<Image> ImageUtils::createTestImage(int width, int height, int channels) {
    auto img = std::make_unique<Image>(width, height, channels);
    
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0f, 255.0f);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                // Create a pattern with some structure for testing
                float value = 128.0f + 64.0f * std::sin(x * 0.1f) * std::cos(y * 0.1f);
                value += dis(gen) * 0.1f; // Add small amount of noise
                (*img)(x, y, c) = std::clamp(value, 0.0f, 255.0f);
            }
        }
    }
    
    return img;
}

std::unique_ptr<Image> ImageUtils::loadImage(const std::string& filename) {
    // Simple implementation - for now just create a test image
    // In a real implementation, you'd use a library like STB or OpenCV
    std::cout << "Loading test image (512x512) instead of " << filename << std::endl;
    return createTestImage(512, 512, 1);
}

void ImageUtils::saveImage(const Image& img, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    // Save as simple text format for debugging
    file << "P2\n" << img.width << " " << img.height << "\n255\n";
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            file << static_cast<int>(std::clamp(img(x, y, 0), 0.0f, 255.0f)) << " ";
        }
        file << "\n";
    }
}

bool ImageUtils::compareImages(const Image& img1, const Image& img2, float tolerance) {
    if (img1.width != img2.width || img1.height != img2.height || img1.channels != img2.channels) {
        return false;
    }
    
    for (size_t i = 0; i < img1.data.size(); i++) {
        if (std::abs(img1.data[i] - img2.data[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void ImageUtils::printImageStats(const Image& img, const std::string& name) {
    if (img.data.empty()) return;
    
    float minVal = *std::min_element(img.data.begin(), img.data.end());
    float maxVal = *std::max_element(img.data.begin(), img.data.end());
    float sum = 0.0f;
    for (float val : img.data) sum += val;
    float mean = sum / img.data.size();
    
    std::cout << name << " stats: " << img.width << "x" << img.height << "x" << img.channels
              << ", range [" << minVal << ", " << maxVal << "], mean=" << mean << std::endl;
}

ConvolutionKernel ImageUtils::createGaussianKernel(int size, float sigma) {
    ConvolutionKernel kernel(size);
    int center = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int dx = x - center;
            int dy = y - center;
            float value = std::exp(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            kernel(x, y) = value;
            sum += value;
        }
    }
    
    // Normalize
    for (int i = 0; i < size * size; i++) {
        kernel.data[i] /= sum;
    }
    
    return kernel;
}

ConvolutionKernel ImageUtils::createSobelXKernel() {
    ConvolutionKernel kernel(3);
    float data[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    std::copy(data, data + 9, kernel.data.begin());
    return kernel;
}

ConvolutionKernel ImageUtils::createSobelYKernel() {
    ConvolutionKernel kernel(3);
    float data[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    std::copy(data, data + 9, kernel.data.begin());
    return kernel;
}

ConvolutionKernel ImageUtils::createLaplacianKernel() {
    ConvolutionKernel kernel(3);
    float data[] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    std::copy(data, data + 9, kernel.data.begin());
    return kernel;
}

ConvolutionKernel ImageUtils::createBoxBlurKernel(int size) {
    ConvolutionKernel kernel(size);
    float value = 1.0f / (size * size);
    std::fill(kernel.data.begin(), kernel.data.end(), value);
    return kernel;
}

ConvolutionKernel ImageUtils::createSharpenKernel() {
    ConvolutionKernel kernel(3);
    float data[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    std::copy(data, data + 9, kernel.data.begin());
    return kernel;
}