# Setup Dependencies for Tasks 3 & 4

## Use Template Repository

**For Tasks 3 & 4, use the official course template that includes all dependencies:**

```bash
git clone https://github.com/lsawicki-cdv/course-accelerating-apps-nvidia-cuda.git
cd course-accelerating-apps-nvidia-cuda/templates/cuda-webcam-filter
# Copy your implementation files over the template
# See TEMPLATE_SETUP_GUIDE.md for detailed instructions
```

**What's included:**
- ✅ OpenCV (208-367MB) - Computer vision library
- ✅ OpenCV Contrib (101MB) - Additional OpenCV modules  
- ✅ Google Test (4.3MB) - Testing framework
- ✅ cxxopts (816KB) - Command line parsing
- ✅ plog (692KB) - Logging library
- ✅ Pre-configured CMake build system
- ✅ Cross-platform compatibility

**Advantages:**
- ✅ 5-10 minutes setup time
- ✅ ~50MB total download
- ✅ Tested configuration
- ✅ No dependency conflicts

## Build Instructions

After cloning the template repository:

```bash
# Navigate to the template directory
cd course-accelerating-apps-nvidia-cuda/templates/cuda-webcam-filter

# Copy your implementation files
# Task 3:
cp -r /path/to/Task\ 3/cuda-webcam-filter/src/* ./src/

# Task 4:
cp -r /path/to/Task\ 4/cuda-webcam-filter/src/* ./src/

# Build the project
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j $(nproc)
```

## Prerequisites

```bash
# CUDA Toolkit (required)
nvidia-smi
nvcc --version

# All other dependencies included in template
```

## Usage

```bash
# Task 3 - HDR Tone Mapping
./cuda-webcam-filter --filter hdr --exposure 1.5 --gamma 2.2 --algorithm reinhard

# Task 4 - Filter Pipeline
./cuda-webcam-filter --pipeline "blur,sharpen,hdr" --multi-stream --transitions
```