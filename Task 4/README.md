**Overview**
In this assignment, you will extend the CUDA webcam filter application (cuda-webcam-filter) to support a filter pipeline that allows multiple filters to be applied sequentially while maintaining real-time performance. You will use CUDA streams to implement concurrent processing and filter chaining capabilities.

**Tasks**

**Part 1: Filter Pipeline Architecture**
1. Design and implement a filter pipeline architecture that allows multiple filters to be applied in sequence
2. Create a mechanism to add/remove filters from the pipeline at runtime dynamically
3. Implement appropriate memory management to efficiently handle intermediate results
4. Use CUDA streams to execute different pipeline stages concurrently, where appropriate

**Part 2: Custom Filter Transitions**
1. Implement **Wipe Transition **to gradually replace one filter with another (left-to-right)
2. Add UI controls to adjust transition parameters and timing
3. Ensure transitions work correctly regardless of the input source

**Part 3: Performance Analysis and Optimization**
1. Implement instrumentation to measure the performance of the filter pipeline
2. Identify and resolve any bottlenecks in the pipeline to maintain real-time performance
3. Compare single-stream vs multi-stream performance for your pipeline implementation
4. Create a visualization of filter pipeline timings that updates in real-time
5. Document your findings and optimization strategies

**Deliverables**
1. Modified source code with pipeline architecture and transitions implemented
2. Performance analysis with charts/graphs comparing different pipeline configurations
3. A short demonstration video showing your pipeline in action with transitions