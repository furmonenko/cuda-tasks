**CUDA Programming Assignment: Real-Time HDR Tone Mapping**

**Overview**
In this assignment, you will extend the CUDA webcam filter template (cuda-webcam-filter) to implement real-time High Dynamic Range (HDR) tone mapping. While traditional cameras have limited dynamic range, HDR tone mapping algorithms simulate higher dynamic range by intelligently compressing the luminance range of an image to preserve details in both dark and bright regions. This creates more visually appealing images that better represent what the human eye perceives.
Your implementation will transform a standard webcam feed into one with enhanced dynamic range through GPU acceleration, allowing you to process video streams in real-time.

**Requirements**

**1. Extend the Filter Framework**
* Add a new HDR_TONEMAPPING filter type to the existing ones in the template
* Implement parameter controls for exposure, gamma, saturation, and tone mapping algorithm selection

**2. Requirements - Core Implementation**
Extend the CUDA webcam filter template to support HDR tone mapping by:
1. Adding a new filter type to the FilterType enum in filter_utils.h
2. Implementing the filter mapping in stringToFilterType() in filter_utils.cpp
3. Creating CUDA kernels for:
* Color space conversion (RGB to luminance/chrominance)
* Global tone mapping operator
* Local tone mapping operator (advanced)
* Color space conversion back to RGB
        4. Adding command-line parameters to control the tone mapping behavior

**3. Performance Optimization**
* Optimize memory transfers for HDR data
* Implement shared memory usage where appropriate
* Provide a performance comparison between GPU and CPU implementations

**Please share the code using GitHub.**