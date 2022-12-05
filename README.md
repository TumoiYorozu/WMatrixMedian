# Constant Time Median Filter using 2D Wavelet Matrix - Supplemental Codes

These codes are based on the supplemental codes by [Andrew Adams "Fast median filters using separable sorting networks" SIGGRAPH 2021], with the addition of our two-dimensional wavelet matrix methods.

We downloaded Adams' code from the ACM DIGITAL LIBRARY. We can read Adams' readme at `README_Adams.txt`.
https://dl.acm.org/doi/10.1145/3450626.3459773

The code for our method can be found in the `2d_wavelet_matrix_median` directory. It includes code for CPUs in three easy-to-read implementations and highly optimized Cuda code for time measurement.

## Environment
We have tested the following environments.
OS: Ubuntu 20.04 LTS
CPU: Intel core i9-9900X
RAM: 32GB (8GB x4, DDR4 2666MHz)
GPU: NVidia RTX 3090

- Halide v14.0.0
- OpenCV v4.5.5
- ArrayFire v3.8.1
- CUDA 11.6
- Clang 12.0.0
- OpenMP: libomp-12-dev


## Building
First edit the Makefile to adjust variables at the top to set paths to opencv, arrayfire, and ipp, and your C++ compiler.

To build the benchmark:
% make bin/run -j8

This takes about two hours to build all the variants on my machine. The long compile time is due to the time required to generate Adams' static code specific to the particular kernel radius. We want to ignore compile time and compare the fastest median filters. Parallel build of make is also possible, but since memory is required for compilation, if the number of parallelism is increased too much, make will terminate with an error.

There are many dependencies in making this program in order to make comparisons with various methods. Most of these can be disabled by setting TEST_* in run.cpp to zero.


## Running
Once it has built, the following line will benchmark median filters
within the specified radii:

% ./bin/run {min_radius} {max_radius}

To run it, you may need to add some of the library install paths from the packages above to your LD_LIBRARY_PATH, depending on your linker.

"wm_cuda" is the CUDA implementation of our method. Includes both construction and runtime time. This value is used in the measurement results.
"wm_runtimeonly_cuda" is the runtime-only time of our method, excluding construction. This value will be helpful when changing the filter radius for the same image, for example.
"wm_parallel2_cpu" is the result of a relatively easy-to-read CPU run. This value is for reference only, as we are focusing on readability rather than speed. All runs are inspected to ensure that the output matches the other methods.
Outputs are also dumped as .csv files, which is what we used to generate the performance figures in
the paper.


## Description of Our Code

### WaveletMatrix_cpu_1_simple.h
This is the simplest implementation of our method.
It corresponds to the pseudo code of Algorithm 1 and 3-5 in the paper.


### WaveletMatrix_cpu_2_parallel.h
The code is designed for parallelization.
The construction part corresponds to Algorithm 7.
OpenMP is used for parallelization. However, it is implemented inefficiently for readability.


### WaveletMatrix_cpu_3_parallel2.h
The code enables the method of reducing atomic add mentioned at the end of Appendix A and the method of storing B mentioned in the construction of Chapter 4, which increases memory consumption but allows for faster computation.


### WaveletMatrixMultiCu4G.cuh & WaveletMatrix2dCu5B.cuh
Based on the CPU version of parallel2, various trivial speedups were added for Cuda.
"Multi" constructs multiple 1D wavelet matrices simultaneously. "2d" realizes a 2D wavelet matrix using multi inside. Both are highly optimized and have various options added for testing, making them difficult to read. This was used for the time measurement.
We are considering refactoring this code and making it publicly available in the future.


### WaveletMatrix_cpu_bitvector.h
It is a data structure that, by performing precomputation, O(1) quickly finds the rank operation (number of bits with 0) of an interval, in a bit array. It should be read in conjunction with Chapter 3.5.
It is used in the CPU version of wavelet matrix. The Cuda version does not exist because the procedure for manipulating bitvectors is built into the wavelet matrix itself.


### float_supporter
Support functions to adapt our method to float images. Provides sorting and a table for converting the resulting image to float.

### standalone_samples directory
Compiling a program in the root directory is a difficult task that requires the preparation of a very large number of dependent packages.
We have prepared a simple program that can be compiled if OpenCV and OpenMP are available.
A CPU version of wavelet matrix is used, not for time measurement purposes, but for analysis to understand the behavior of the method.
