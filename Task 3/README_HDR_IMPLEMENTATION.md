# HDR Tone Mapping Implementation for CUDA Webcam Filter

## Overview
This document describes the implementation of real-time HDR tone mapping functionality added to the CUDA webcam filter template. The implementation extends the existing filter framework to support advanced image processing techniques that simulate High Dynamic Range effects.

## Implementation Details

### 1. Filter Framework Extension

#### Added HDR Filter Type
- **FilterType enum**: Added `HDR_TONEMAPPING` to the existing filter types
- **String mapping**: Added support for "hdr" and "tonemapping" filter names
- **Filter creation**: HDR uses identity kernel as processing is done in specialized kernels

### 2. HDR Parameters Structure

#### New Parameters Added
```cpp
enum class ToneMappingAlgorithm {
    REINHARD,  // 0 - Reinhard tone mapping
    DRAGO,     // 1 - Drago tone mapping  
    MANTIUK    // 2 - Mantiuk tone mapping
};

struct FilterOptions {
    // ... existing parameters ...
    
    // HDR tone mapping parameters
    float exposure;                      // Exposure adjustment (default: 1.0)
    float gamma;                        // Gamma correction (default: 2.2)
    float saturation;                   // Saturation adjustment (default: 1.0)
    ToneMappingAlgorithm toneMappingAlgorithm; // Algorithm selection
};
```

### 3. Command Line Interface

#### New Command Line Arguments
```bash
-e, --exposure <value>         # HDR exposure value (default: 1.0)
-g, --gamma <value>           # HDR gamma correction (default: 2.2) 
--saturation <value>          # HDR saturation adjustment (default: 1.0)
--tone-algorithm <algorithm>  # Tone mapping algorithm: reinhard, drago, mantiuk
```

#### Updated Filter Selection
```bash
-f, --filter <type>           # Filter type: blur, sharpen, edge, emboss, hdr
```

### 4. CUDA Kernels Implementation

#### Color Space Conversion
```cpp
__device__ float rgbToLuminance(float r, float g, float b)
{
    return 0.299f * r + 0.587f * g + 0.114f * b;
}
```

#### Tone Mapping Operators

**Reinhard Tone Mapping**
```cpp
__device__ float reinhardToneMapping(float luminance, float exposure)
{
    float L = luminance * exposure;
    return L / (1.0f + L);
}
```

**Drago Tone Mapping**
```cpp
__device__ float dragoToneMapping(float luminance, float exposure, float bias = 0.85f)
{
    float L = luminance * exposure;
    // Logarithmic compression with bias parameter
    // ... implementation details ...
}
```

**Mantiuk Tone Mapping**
```cpp
__device__ float mantiukToneMapping(float luminance, float exposure)
{
    float L = luminance * exposure;
    return powf(L / (L + 1.0f), 0.6f);
}
```

#### Main HDR Processing Kernel
```cpp
__global__ void hdrToneMappingKernel(const unsigned char *input, unsigned char *output,
                                    int width, int height, int channels,
                                    float exposure, float gamma, float saturation, int algorithm)
```

The kernel performs the following steps:
1. **RGB to Luminance**: Convert input RGB to luminance value
2. **Tone Mapping**: Apply selected tone mapping operator
3. **Color Preservation**: Maintain color ratios while adjusting luminance
4. **Saturation Adjustment**: Apply saturation enhancement/reduction
5. **Gamma Correction**: Apply gamma correction for display
6. **Output Conversion**: Convert back to [0, 255] range

### 5. Performance Optimization

#### Memory Access Patterns
- **Coalesced Memory Access**: Threads access consecutive memory locations
- **Minimal Global Memory**: Single pass processing reduces memory bandwidth
- **Efficient Block Size**: 16x16 thread blocks for optimal occupancy

#### Algorithm Selection
- **Runtime Selection**: Algorithm chosen at kernel launch time
- **Branch Optimization**: Minimal branching within kernels
- **Mathematical Optimization**: Efficient implementations of logarithmic and power functions

### 6. CPU Implementation

Parallel CPU implementation provided for comparison:
- Same algorithms as GPU version
- Pixel-by-pixel processing using OpenCV Mat structures
- Performance baseline for GPU acceleration measurement

## Usage Examples

### Basic HDR Tone Mapping
```bash
./cuda-webcam-filter -f hdr -e 1.5 -g 2.2 --saturation 1.2
```

### Different Algorithms
```bash
# Reinhard (default)
./cuda-webcam-filter -f hdr --tone-algorithm reinhard

# Drago tone mapping
./cuda-webcam-filter -f hdr --tone-algorithm drago -e 2.0

# Mantiuk tone mapping  
./cuda-webcam-filter -f hdr --tone-algorithm mantiuk --saturation 0.8
```

### With Preview Mode
```bash
./cuda-webcam-filter -f hdr -e 1.8 -g 2.0 --preview
```

## Technical Features

### Real-time Processing
- **GPU Acceleration**: CUDA kernels process frames in real-time
- **Minimal Latency**: Single-pass processing with optimized memory transfers
- **FPS Monitoring**: Real-time performance metrics display

### Algorithm Comparison
- **Side-by-side Display**: CPU vs GPU processing comparison
- **Performance Metrics**: Frame rate and processing time measurement
- **Visual Quality**: Real-time visual comparison of different algorithms

### Memory Management
- **Optimized Transfers**: Efficient host-device memory copying
- **Error Handling**: Comprehensive CUDA error checking
- **Resource Cleanup**: Proper memory deallocation

## Expected Performance

### GPU Acceleration Benefits
- **Processing Speed**: 5-15x faster than CPU implementation
- **Real-time Capability**: Maintains 30+ FPS on modern hardware
- **Power Efficiency**: Lower CPU usage allows for other concurrent tasks

### Hardware Requirements
- **CUDA-enabled GPU**: Compute Capability 6.0+
- **Memory**: Minimum 2GB GPU memory for HD video processing
- **Drivers**: CUDA Toolkit 12.0+

## Algorithm Characteristics

### Reinhard Tone Mapping
- **Purpose**: Simple, fast tone mapping
- **Characteristics**: Preserves local contrast, good for real-time
- **Best for**: General purpose HDR simulation

### Drago Tone Mapping  
- **Purpose**: Logarithmic tone compression
- **Characteristics**: Better detail preservation in highlights
- **Best for**: High dynamic range scenes with bright areas

### Mantiuk Tone Mapping
- **Purpose**: Perceptually uniform compression
- **Characteristics**: Maintains visual quality across luminance range
- **Best for**: Artistic effects and enhanced realism

## Future Enhancements

### Advanced Features
- **Local Tone Mapping**: Spatially-varying tone mapping operators
- **Adaptive Parameters**: Automatic parameter adjustment based on scene analysis
- **Multiple Exposure**: Combining multiple exposures for true HDR

### Performance Optimizations
- **Shared Memory**: Tile-based processing for improved memory efficiency
- **Texture Memory**: Utilizing texture cache for input images
- **Multi-GPU**: Scaling across multiple GPUs for higher resolutions

### Additional Algorithms
- **Photographic Tone Reproduction**: More sophisticated algorithms
- **Adaptive Histogram**: Dynamic range analysis and adjustment
- **Temporal Stability**: Frame-to-frame consistency for video processing

## Conclusion

The HDR tone mapping implementation successfully extends the CUDA webcam filter framework with advanced image processing capabilities. The implementation provides:

- **Real-time Performance**: GPU-accelerated processing for interactive use
- **Multiple Algorithms**: Choice of tone mapping operators for different effects
- **Professional Quality**: High-quality results suitable for content creation
- **Ease of Use**: Simple command-line interface for parameter control

This implementation demonstrates the power of CUDA for real-time image processing and provides a foundation for further advanced computer vision applications.