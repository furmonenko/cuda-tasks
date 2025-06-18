# Build Instructions

## Prerequisites
- Windows 10/11
- NVIDIA CUDA Toolkit 12.x
- Visual Studio 2022 Community (with C++ support)
- CMake 3.18+

## Build Steps

```powershell
# Navigate to project
cd "C:\Users\zfurm\Desktop\CUDA\Task 1"

# Create and enter build directory
mkdir build
cd build

# Configure
cmake .. -G "Visual Studio 17 2022" -A x64

# Build
cmake --build . --config Release

# Run
.\Release\matrix_transpose.exe
```

## Run Options
- Basic test: `.\Release\matrix_transpose.exe`
- Scalability test: `.\Release\matrix_transpose.exe --scale`
- Block size analysis: `.\Release\matrix_transpose.exe --blocks`

## Results
All test results are automatically saved to the `results/` directory:
- `results/scalability_results.csv` - Performance data across matrix sizes
- `results/block_size_results.csv` - Block size comparison data
- `results/*.log` - Detailed analysis logs

## Troubleshooting
- Restart PowerShell after installing CMake
- Check: `cmake --version` and `nvcc --version`
- Ensure Visual Studio has C++ workload installed