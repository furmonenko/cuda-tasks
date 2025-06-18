cmake_minimum_required(VERSION 3.18)
project(MatrixTranspose CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/cpu_transpose.cpp
    src/matrix_transpose.cu
)

# Create executable
add_executable(matrix_transpose ${SOURCES})

# Set properties for CUDA compilation
set_target_properties(matrix_transpose PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(matrix_transpose ${CUDA_LIBRARIES})

# Compiler-specific options
if(MSVC)
    target_compile_options(matrix_transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>)
else()
    target_compile_options(matrix_transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
endif()