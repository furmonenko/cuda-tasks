**Overview**
In this assignment, you will implement and optimize matrix transposition using NVIDIA CUDA. Matrix transposition is a fundamental operation in linear algebra where the rows and columns of a matrix are interchanged. While conceptually simple, efficient implementation on GPUs requires an understanding of memory access patterns and optimization techniques.

**Learning Objectives**
* Understand basic CUDA programming concepts
* Implement a parallel algorithm for matrix transposition (start from a 2048x1024 matrix with random data and increase the size gradually)
* Analyze and optimize memory access patterns
* Utilize unified memory to improve performance
* Measure and compare performance between different implementations

**Requirements**

**Part 1: Basic CUDA Matrix Transposition**
1. Implement a naive CPU-based matrix transposition function
2. Implement a basic CUDA kernel for matrix transposition
3. Compare the performance between CPU and GPU implementations
4. Verify the correctness of your implementation

**Part 2: Optimized CUDA Matrix Transposition**
1. Implement an optimized version using unified memory
2. Address coalesced memory access patterns
3. Analyze the impact of different block sizes on performance
4. Compare the performance between the naive and optimized GPU implementations

**Part 3: Analysis and Documentation**
1. Measure and report execution times for all implementations
2. Analyze performance differences and explain the reasons
3. Discuss the impact of matrix size on relative performance
4. Identify potential further optimizations
5. Preferrably share the code using GitHub