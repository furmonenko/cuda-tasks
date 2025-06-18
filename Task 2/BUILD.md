# Build Instructions for CUDA Image Convolution

## Prerequisites
- Windows 10/11
- NVIDIA CUDA Toolkit 12.x
- Visual Studio 2022 Community (with C++ support)
- CMake 3.18 or higher

## Building the Project

### Command Line Build
```bash
# Navigate to project directory
cd "C:\Users\zfurm\Desktop\CUDA\Task 2"

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -G "Visual Studio 17 2022" -A x64

# Build the project
cmake --build . --config Release

# Run the executable
.\Release\image_convolution.exe
```

## Run Options
- Basic test: `.\Release\image_convolution.exe`
- Different kernels: `.\Release\image_convolution.exe --kernel gaussian`
- Performance benchmark: `.\Release\image_convolution.exe --benchmark`
- Separable filters: `.\Release\image_convolution.exe --separable`

## Results
All test results are automatically saved to the `results/` directory:
- `results/convolution_results.csv` - Performance data
- `results/convolution_results.log` - Detailed analysis logs
- `results/output_*.pgm` - Processed images

## Supported Kernels
- Gaussian Blur (3x3, 5x5, 7x7)
- Sobel Edge Detection (X and Y)
- Laplacian Edge Detection
- Box Blur
- Sharpen Filter