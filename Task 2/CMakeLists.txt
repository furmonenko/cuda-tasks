cmake_minimum_required(VERSION 3.18)
project(ImageConvolution CUDA CXX)

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
    src/cpu_convolution.cpp
    src/cuda_convolution.cu
    src/image_utils.cpp
)

# Create executable
add_executable(image_convolution ${SOURCES})

# Set properties for CUDA compilation
set_target_properties(image_convolution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(image_convolution ${CUDA_LIBRARIES})

# Compiler-specific options
if(MSVC)
    target_compile_options(image_convolution PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>)
else()
    target_compile_options(image_convolution PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
endif()