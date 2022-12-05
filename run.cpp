#include <HalideBuffer.h>
#include <HalideRuntimeCuda.h>
#include <halide_benchmark.h>
#include <halide_image_io.h>
#include <immintrin.h>
#include <random>
#include <stdint.h>

#ifndef MAX_RADIUS
#define MAX_RADIUS 18
#endif

#ifndef MIN_RADIUS
#define MIN_RADIUS 1
#endif

// CPU methods

// Naive. Gather a window and then call std::nth_element. Too slow to even benchmark.
#define TEST_NAIVE 0

// Perreault's constant-time-median filter, extended to 16-bit by the
// EBImage folks. Parallelized and AVX2'd by me.
#define TEST_CTMF 0

// OpenCV's median filter. Uses sorting networks for 3x3 and 5x5.
// Uses CTMF above that, but parallelizing it is harder than just
// parallelizing Perreault's CTMF code directly, so it's a little
// slower.
#define TEST_OPENCV 0

// My own implementation of Kim et al. 2015
#define TEST_KIM 0

// Intel performance primitives. Very good sorting networks for small
// sizes. O(r) algorithm (Huang?) for larger sizes. Not internally
// parallel, but supports a boundary condition that makes
// parallelizing it in tiles easy.
#define TEST_IPP 0

// Adams precompiled version. Works for sizes up to MAX_RADIUS
#define TEST_STATIC 0

// Adams interpreted version
#define TEST_DYNAMIC_V2 0


#define TEST_WAVELET_MATRIX_NAIVE_CPU 0
#define TEST_WAVELET_MATRIX_PARALLEL_CPU 0
#define TEST_WAVELET_MATRIX_PARALLEL2_CPU 1


// GPU methods

#define TEST_WAVELET_MATRIX_OPTIMIZED_CUDA 1
#define TEST_WAVELET_MATRIX_RUNTIMEONLY_CUDA 1


// Green's method from OpenCV contrib
#define TEST_OPENCV_CUDA 1

// Nvidia's answer to IPP.
#define TEST_NPP 1

// Arrayfire. Works for sizes up to 15x15
#define TEST_ARRAYFIRE 1

// Adams precompiled GPU version
#define TEST_STATIC_CUDA 1

// Adams interpreted GPU version
#define TEST_DYNAMIC_CUDA 1

#if TEST_CTMF
#include "ctmf_u16.h"
#include "ctmf_u8.h"
#endif

// clang-format off
#if TEST_OPENCV || TEST_OPENCV_CUDA
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#endif
// clang-format on

#if TEST_KIM
#include "kim_median_filter_f32_1.h"
#include "kim_median_filter_f32_2.h"
#include "kim_median_filter_u16_1.h"
#include "kim_median_filter_u16_2.h"
#include "kim_median_filter_u8_1.h"
#include "kim_median_filter_u8_2.h"
#endif

#if TEST_WAVELET_MATRIX_OPTIMIZED_CUDA || TEST_WAVELET_MATRIX_RUNTIMEONLY_CUDA
#include "2d_wavelet_matrix_median/WaveletMatrix_Cuda_main.h"
#endif

#if TEST_WAVELET_MATRIX_NAIVE_CPU || TEST_WAVELET_MATRIX_PARALLEL_CPU || TEST_WAVELET_MATRIX_PARALLEL2_CPU
#include "2d_wavelet_matrix_median/WaveletMatrix_cpu.h"
#endif

#if TEST_IPP
#include <ippcore.h>
#include <ippi.h>
#endif

#if TEST_NPP
// To avoid making multiple cuda contexts, we'll use the Halide cuda
// context and allocation cache for the NPP call (none of the Halide
// runtime stuff is inside the NPP benchmarking loop).
#include <nppcore.h>
#include <nppi.h>
#endif

#if TEST_ARRAYFIRE
#include <arrayfire.h>
#endif

#if TEST_DYNAMIC_V2 || TEST_DYNAMIC_CUDA
#include "dynamic_median_filter.h"
#endif

#if TEST_STATIC || TEST_STATIC_CUDA
#include "median_filter_cuda_f32_1.h"
#include "median_filter_cuda_u16_1.h"
#include "median_filter_cuda_u8_1.h"
#include "median_filter_f32_1.h"
#include "median_filter_u16_1.h"
#include "median_filter_u8_1.h"

#if MAX_RADIUS >= 2
#include "median_filter_cuda_f32_2.h"
#include "median_filter_cuda_u16_2.h"
#include "median_filter_cuda_u8_2.h"
#include "median_filter_f32_2.h"
#include "median_filter_u16_2.h"
#include "median_filter_u8_2.h"
#else
#define median_filter_u8_2 median_filter_u8_1
#define median_filter_u16_2 median_filter_u16_1
#define median_filter_f32_2 median_filter_f32_1
#define median_filter_cuda_u8_2 median_filter_u8_1
#define median_filter_cuda_u16_2 median_filter_u16_1
#define median_filter_cuda_f32_2 median_filter_f32_1
#endif

#if MAX_RADIUS >= 3
#include "median_filter_cuda_f32_3.h"
#include "median_filter_cuda_u16_3.h"
#include "median_filter_cuda_u8_3.h"
#include "median_filter_f32_3.h"
#include "median_filter_u16_3.h"
#include "median_filter_u8_3.h"
#else
#define median_filter_u8_3 median_filter_u8_1
#define median_filter_u16_3 median_filter_u16_1
#define median_filter_f32_3 median_filter_f32_1
#define median_filter_cuda_u8_3 median_filter_u8_1
#define median_filter_cuda_u16_3 median_filter_u16_1
#define median_filter_cuda_f32_3 median_filter_f32_1
#endif

#if MAX_RADIUS >= 4
#include "median_filter_cuda_f32_4.h"
#include "median_filter_cuda_u16_4.h"
#include "median_filter_cuda_u8_4.h"
#include "median_filter_f32_4.h"
#include "median_filter_u16_4.h"
#include "median_filter_u8_4.h"
#else
#define median_filter_u8_4 median_filter_u8_1
#define median_filter_u16_4 median_filter_u16_1
#define median_filter_f32_4 median_filter_f32_1
#define median_filter_cuda_u8_4 median_filter_u8_1
#define median_filter_cuda_u16_4 median_filter_u16_1
#define median_filter_cuda_f32_4 median_filter_f32_1
#endif

#if MAX_RADIUS >= 5
#include "median_filter_cuda_f32_5.h"
#include "median_filter_cuda_u16_5.h"
#include "median_filter_cuda_u8_5.h"
#include "median_filter_f32_5.h"
#include "median_filter_u16_5.h"
#include "median_filter_u8_5.h"
#else
#define median_filter_u8_5 median_filter_u8_1
#define median_filter_u16_5 median_filter_u16_1
#define median_filter_f32_5 median_filter_f32_1
#define median_filter_cuda_u8_5 median_filter_u8_1
#define median_filter_cuda_u16_5 median_filter_u16_1
#define median_filter_cuda_f32_5 median_filter_f32_1
#endif

#if MAX_RADIUS >= 6
#include "median_filter_cuda_f32_6.h"
#include "median_filter_cuda_u16_6.h"
#include "median_filter_cuda_u8_6.h"
#include "median_filter_f32_6.h"
#include "median_filter_u16_6.h"
#include "median_filter_u8_6.h"
#else
#define median_filter_u8_6 median_filter_u8_1
#define median_filter_u16_6 median_filter_u16_1
#define median_filter_f32_6 median_filter_f32_1
#define median_filter_cuda_u8_6 median_filter_u8_1
#define median_filter_cuda_u16_6 median_filter_u16_1
#define median_filter_cuda_f32_6 median_filter_f32_1
#endif

#if MAX_RADIUS >= 7
#include "median_filter_cuda_f32_7.h"
#include "median_filter_cuda_u16_7.h"
#include "median_filter_cuda_u8_7.h"
#include "median_filter_f32_7.h"
#include "median_filter_u16_7.h"
#include "median_filter_u8_7.h"
#else
#define median_filter_u8_7 median_filter_u8_1
#define median_filter_u16_7 median_filter_u16_1
#define median_filter_f32_7 median_filter_f32_1
#define median_filter_cuda_u8_7 median_filter_u8_1
#define median_filter_cuda_u16_7 median_filter_u16_1
#define median_filter_cuda_f32_7 median_filter_f32_1
#endif

#if MAX_RADIUS >= 8
#include "median_filter_cuda_f32_8.h"
#include "median_filter_cuda_u16_8.h"
#include "median_filter_cuda_u8_8.h"
#include "median_filter_f32_8.h"
#include "median_filter_u16_8.h"
#include "median_filter_u8_8.h"
#else
#define median_filter_u8_8 median_filter_u8_1
#define median_filter_u16_8 median_filter_u16_1
#define median_filter_f32_8 median_filter_f32_1
#define median_filter_cuda_u8_8 median_filter_u8_1
#define median_filter_cuda_u16_8 median_filter_u16_1
#define median_filter_cuda_f32_8 median_filter_f32_1
#endif

#if MAX_RADIUS >= 9
#include "median_filter_cuda_f32_9.h"
#include "median_filter_cuda_u16_9.h"
#include "median_filter_cuda_u8_9.h"
#include "median_filter_f32_9.h"
#include "median_filter_u16_9.h"
#include "median_filter_u8_9.h"
#else
#define median_filter_u8_9 median_filter_u8_1
#define median_filter_u16_9 median_filter_u16_1
#define median_filter_f32_9 median_filter_f32_1
#define median_filter_cuda_u8_9 median_filter_u8_1
#define median_filter_cuda_u16_9 median_filter_u16_1
#define median_filter_cuda_f32_9 median_filter_f32_1
#endif

#if MAX_RADIUS >= 10
#include "median_filter_cuda_f32_10.h"
#include "median_filter_cuda_u16_10.h"
#include "median_filter_cuda_u8_10.h"
#include "median_filter_f32_10.h"
#include "median_filter_u16_10.h"
#include "median_filter_u8_10.h"
#else
#define median_filter_u8_10 median_filter_u8_1
#define median_filter_u16_10 median_filter_u16_1
#define median_filter_f32_10 median_filter_f32_1
#define median_filter_cuda_u8_10 median_filter_u8_1
#define median_filter_cuda_u16_10 median_filter_u16_1
#define median_filter_cuda_f32_10 median_filter_f32_1
#endif

#if MAX_RADIUS >= 11
#include "median_filter_cuda_f32_11.h"
#include "median_filter_cuda_u16_11.h"
#include "median_filter_cuda_u8_11.h"
#include "median_filter_f32_11.h"
#include "median_filter_u16_11.h"
#include "median_filter_u8_11.h"
#else
#define median_filter_u8_11 median_filter_u8_1
#define median_filter_u16_11 median_filter_u16_1
#define median_filter_f32_11 median_filter_f32_1
#define median_filter_cuda_u8_11 median_filter_u8_1
#define median_filter_cuda_u16_11 median_filter_u16_1
#define median_filter_cuda_f32_11 median_filter_f32_1
#endif

#if MAX_RADIUS >= 12
#include "median_filter_cuda_f32_12.h"
#include "median_filter_cuda_u16_12.h"
#include "median_filter_cuda_u8_12.h"
#include "median_filter_f32_12.h"
#include "median_filter_u16_12.h"
#include "median_filter_u8_12.h"
#else
#define median_filter_u8_12 median_filter_u8_1
#define median_filter_u16_12 median_filter_u16_1
#define median_filter_f32_12 median_filter_f32_1
#define median_filter_cuda_u8_12 median_filter_u8_1
#define median_filter_cuda_u16_12 median_filter_u16_1
#define median_filter_cuda_f32_12 median_filter_f32_1
#endif

#if MAX_RADIUS >= 13
#include "median_filter_cuda_f32_13.h"
#include "median_filter_cuda_u16_13.h"
#include "median_filter_cuda_u8_13.h"
#include "median_filter_f32_13.h"
#include "median_filter_u16_13.h"
#include "median_filter_u8_13.h"
#else
#define median_filter_u8_13 median_filter_u8_1
#define median_filter_u16_13 median_filter_u16_1
#define median_filter_f32_13 median_filter_f32_1
#define median_filter_cuda_u8_13 median_filter_u8_1
#define median_filter_cuda_u16_13 median_filter_u16_1
#define median_filter_cuda_f32_13 median_filter_f32_1
#endif

#if MAX_RADIUS >= 14
#include "median_filter_cuda_f32_14.h"
#include "median_filter_cuda_u16_14.h"
#include "median_filter_cuda_u8_14.h"
#include "median_filter_f32_14.h"
#include "median_filter_u16_14.h"
#include "median_filter_u8_14.h"
#else
#define median_filter_u8_14 median_filter_u8_1
#define median_filter_u16_14 median_filter_u16_1
#define median_filter_f32_14 median_filter_f32_1
#define median_filter_cuda_u8_14 median_filter_u8_1
#define median_filter_cuda_u16_14 median_filter_u16_1
#define median_filter_cuda_f32_14 median_filter_f32_1
#endif


#if MAX_RADIUS >= 15
#include "median_filter_cuda_f32_15.h"
#include "median_filter_cuda_u16_15.h"
#include "median_filter_cuda_u8_15.h"
#include "median_filter_f32_15.h"
#include "median_filter_u16_15.h"
#include "median_filter_u8_15.h"
#else
#define median_filter_u8_15 median_filter_u8_1
#define median_filter_u16_15 median_filter_u16_1
#define median_filter_f32_15 median_filter_f32_1
#define median_filter_cuda_u8_15 median_filter_u8_1
#define median_filter_cuda_u16_15 median_filter_u16_1
#define median_filter_cuda_f32_15 median_filter_f32_1
#endif


#if MAX_RADIUS >= 16
#include "median_filter_cuda_f32_16.h"
#include "median_filter_cuda_u16_16.h"
#include "median_filter_cuda_u8_16.h"
#include "median_filter_f32_16.h"
#include "median_filter_u16_16.h"
#include "median_filter_u8_16.h"
#else
#define median_filter_u8_16 median_filter_u8_1
#define median_filter_u16_16 median_filter_u16_1
#define median_filter_f32_16 median_filter_f32_1
#define median_filter_cuda_u8_16 median_filter_u8_1
#define median_filter_cuda_u16_16 median_filter_u16_1
#define median_filter_cuda_f32_16 median_filter_f32_1
#endif



#if MAX_RADIUS >= 17
#include "median_filter_cuda_f32_17.h"
#include "median_filter_cuda_u16_17.h"
#include "median_filter_cuda_u8_17.h"
#include "median_filter_f32_17.h"
#include "median_filter_u16_17.h"
#include "median_filter_u8_17.h"
#else
#define median_filter_u8_17 median_filter_u8_1
#define median_filter_u16_17 median_filter_u16_1
#define median_filter_f32_17 median_filter_f32_1
#define median_filter_cuda_u8_17 median_filter_u8_1
#define median_filter_cuda_u16_17 median_filter_u16_1
#define median_filter_cuda_f32_17 median_filter_f32_1
#endif



#if MAX_RADIUS >= 18
#include "median_filter_cuda_f32_18.h"
#include "median_filter_cuda_u16_18.h"
#include "median_filter_cuda_u8_18.h"
#include "median_filter_f32_18.h"
#include "median_filter_u16_18.h"
#include "median_filter_u8_18.h"
#else
#define median_filter_u8_18 median_filter_u8_1
#define median_filter_u16_18 median_filter_u16_1
#define median_filter_f32_18 median_filter_f32_1
#define median_filter_cuda_u8_18 median_filter_u8_1
#define median_filter_cuda_u16_18 median_filter_u16_1
#define median_filter_cuda_f32_18 median_filter_f32_1
#endif



#if MAX_RADIUS >= 19
#include "median_filter_cuda_f32_19.h"
#include "median_filter_cuda_u16_19.h"
#include "median_filter_cuda_u8_19.h"
#include "median_filter_f32_19.h"
#include "median_filter_u16_19.h"
#include "median_filter_u8_19.h"
#else
#define median_filter_u8_19 median_filter_u8_1
#define median_filter_u16_19 median_filter_u16_1
#define median_filter_f32_19 median_filter_f32_1
#define median_filter_cuda_u8_19 median_filter_u8_1
#define median_filter_cuda_u16_19 median_filter_u16_1
#define median_filter_cuda_f32_19 median_filter_f32_1
#endif


#if MAX_RADIUS >= 20
#include "median_filter_cuda_f32_20.h"
#include "median_filter_cuda_u20_20.h"
#include "median_filter_cuda_u8_20.h"
#include "median_filter_f32_20.h"
#include "median_filter_u20_20.h"
#include "median_filter_u8_20.h"
#else
#define median_filter_u8_20 median_filter_u8_1
#define median_filter_u20_20 median_filter_u20_1
#define median_filter_f32_20 median_filter_f32_1
#define median_filter_cuda_u8_20 median_filter_u8_1
#define median_filter_cuda_u20_20 median_filter_u20_1
#define median_filter_cuda_f32_20 median_filter_f32_1
#endif

#endif

#include <map>
using std::map;
using std::pair;
using std::vector;

#if TEST_NAIVE

template<typename T>
void naive_median(const Halide::Runtime::Buffer<T> &src, int radius, Halide::Runtime::Buffer<T> dst) {
    int diameter = 2 * radius + 1;
    int footprint_size = diameter * diameter;
    int median_idx = footprint_size / 2;
    vector<T> scratch(footprint_size);
    for (int y = 0; y < dst.height(); y++) {
        for (int x = 0; x < dst.width(); x++) {
            const T *src_ptr = &src(x - radius, y - radius);
            T *scratch_ptr = &scratch[0];
            int row_step = src.dim(1).stride() - diameter;
            for (int dy = y - radius; dy <= y + radius; dy++) {
                for (int dx = x - radius; dx <= x + radius; dx++) {
                    *scratch_ptr++ = *src_ptr++;
                }
                src_ptr += row_step;
            }
            std::nth_element(scratch.begin(), scratch.begin() + median_idx, scratch.end());
            dst(x, y) = scratch[median_idx];
        }
    }
}
#endif





// Benchmark and return time in milliseconds
template<typename Callable>
double bench(Callable c) {
    Halide::Tools::BenchmarkConfig config;
    config.accuracy = 0.0001f;
    config.min_time = 1.0f;
    config.max_time = 10.0f;
    return Halide::Tools::benchmark(c, config) * 1e3;
}

template<typename T>
int test(int min_radius, int max_radius, const char *results_filename) {
    FILE *results = fopen(results_filename, "w");

    assert(min_radius <= max_radius);

    for (int radius = min_radius; radius <= max_radius; radius++) {
        const int diameter = 2 * radius + 1;

        // Pad out the input in a way that's friendly to lots of
        // different possible Halide schedules and tilings. We're not
        // interested in measuring differences in boundary condition
        // handling.
        int padding = ((radius + 7) / 8) * 8;

        /*
        int W = 2560, H = 1600;

        Halide::Runtime::Buffer<T> src(W + padding * 2, H + padding * 2);
        src.set_min(-padding, -padding);

        std::mt19937 rng(1);
        assert(!src.device_dirty());
        for (int y = -padding; y < H + padding; y++) {
            for (int x = -padding; x < W + padding; x++) {
                src(x, y) = rng();
            }
        }
        */

        Halide::Runtime::Buffer<uint8_t> src8 =
            Halide::Tools::load_and_convert_image("median_filters_before_srgb_flowers.png");
        // Just use the red channel for benchmarking
        src8.slice(2, 0);
        int W = src8.width() - padding * 2, H = src8.height() - padding * 2;
        W &= ~31;
        H &= ~31;
        src8.crop(0, 0, W + padding * 2);
        src8.crop(1, 0, H + padding * 2);
        src8.set_min(-padding, -padding);

        Halide::Runtime::Buffer<T> src(src8.width(), src8.height());
        src.set_min(-padding, -padding);

        // Convert from 8-bit to a plausible higher-bit-width value by
        // using noise for the low bits.
        std::mt19937 rng(1);
        if (std::is_same<T, uint16_t>::value) {
            src8.for_each_value([&](const uint8_t &src, T &dst) {
                dst = src * 256 + (rng() & 0xff);
            },
                                src);
        } else if (std::is_same<T, float>::value) {
            std::uniform_real_distribution<> dis(0.0, 1.0f / 256);
            src8.for_each_value([&](const uint8_t &src, T &dst) {
                dst = (T)(src / 256.f + dis(rng));
            },
                                src);
        } else {
            src8.for_each_value([&](const uint8_t &src, T &dst) {
                dst = src;
            },
                                src);
        }
        src.set_host_dirty();

#if TEST_NAIVE
        Halide::Runtime::Buffer<T> dst_naive(W, H);
        double t_naive =
            bench([&]() {
                naive_median(src, radius, dst_naive);
            });
#endif

#if TEST_WAVELET_MATRIX_NAIVE_CPU
        Halide::Runtime::Buffer<T> dst_wm_naive_cpu(W, H);
        double t_wm_naive_cpu =
            bench([&]() {
                wavelet_matrix_median::wm_naive_cpu_median<T>(
                    &src(-radius, -radius),
                    radius,
                    H,
                    W,
                    src.dim(1).stride(),
                    &dst_wm_naive_cpu(0, 0),
                    dst_wm_naive_cpu.dim(1).stride());
            });
#endif

#if TEST_WAVELET_MATRIX_PARALLEL_CPU
        Halide::Runtime::Buffer<T> dst_wm_parallel_cpu(W, H);
        double t_wm_parallel_cpu =
            bench([&]() {
                wavelet_matrix_median::wm_parallel_cpu_median<T>(
                    &src(-radius, -radius),
                    radius,
                    H,
                    W,
                    src.dim(1).stride(),
                    &dst_wm_parallel_cpu(0, 0),
                    dst_wm_parallel_cpu.dim(1).stride());
            });
#endif

#if TEST_WAVELET_MATRIX_PARALLEL2_CPU
        Halide::Runtime::Buffer<T> dst_wm_parallel2_cpu(W, H);
        double t_wm_parallel2_cpu =
            bench([&]() {
                wavelet_matrix_median::wm_parallel2_cpu_median<T>(
                    &src(-radius, -radius),
                    radius,
                    H,
                    W,
                    src.dim(1).stride(),
                    &dst_wm_parallel2_cpu(0, 0),
                    dst_wm_parallel2_cpu.dim(1).stride());
            });
#endif

        double t_opencv = 0;
#if TEST_OPENCV
        Halide::Runtime::Buffer<T> dst_opencv(W, H);
        {
            // OpenCV can't handle outputs that are smaller than the
            // input, so we have to pad the output buffer and then ignore
            // edge pixels.
            auto opencv_type = std::is_same<T, uint8_t>::value  ? CV_8UC1 :
                               std::is_same<T, uint16_t>::value ? CV_16UC1 :
                                                                  CV_32FC1;
            cv::Mat cv_src(H + padding * 2, W + padding * 2, opencv_type, src.data());
            cv::Mat cv_dst(H + padding * 2, W + padding * 2, opencv_type);

            if (opencv_type == CV_8UC1 ||
                radius <= 2) {
                // OpenCV only has an arbitrary-radius implementation
                // for 8-bit. It also has a compulsory boundary
                // condition (you can't just feed it a padded input
                // without it also computing a padded output), which
                // makes parallelizing it quite annoying. The warm-up
                // portion of this work is not wasted, it's necessary
                // for the algorithm used (the constant time median
                // filter). We'll go as coarse-grained as
                // possible. The size below was find via
                // trial-and-error to be the one that's fastest on the
                // benchmarking machine. It corresponds to one task
                // per physical core.

                int strip_height = (H + 15) / 16;
                int inset = padding - radius;

                struct Closure {
                    cv::Mat &cv_src, &cv_dst;
                    int W, H, radius, padding, strip_height, inset, diameter, opencv_type;
                } closure{cv_src, cv_dst,
                          W, H, radius, padding, strip_height, inset, diameter, opencv_type};

                auto one_strip = [](void *ucon, int y, uint8_t *closure) {
                    Closure *c = (Closure *)closure;
                    y *= c->strip_height;
                    int y_max = std::min(c->H, y + c->strip_height);
                    int y_extent = y_max - y;
                    cv::Mat dst_window(c->strip_height + c->radius * 2, c->W + c->radius * 2, c->opencv_type);
                    cv::Rect2d src_roi(c->inset, y + c->inset,
                                       c->W + c->radius * 2,
                                       y_extent + c->radius * 2);
                    cv::Mat src_window = c->cv_src(src_roi);

                    cv::medianBlur(src_window, dst_window, c->diameter);

                    // Copy out the stuff not affected by the boundary condition
                    cv::Mat dst_slice = c->cv_dst(cv::Rect2d(c->padding, y + c->padding, c->W, y_extent));

                    dst_window(cv::Rect2d(c->radius, c->radius, c->W, y_extent)).copyTo(dst_slice);
                    return 0;
                };

                t_opencv =
                    bench([&]() {
                        halide_do_par_for(nullptr, one_strip,
                                          0,
                                          (H + strip_height - 1) / strip_height,
                                          (uint8_t *)&closure);
                    });
            }

            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    dst_opencv(x, y) = cv_dst.at<T>(y + padding, x + padding);
                }
            }
        }
#endif

        double t_wm_cuda = 0;
#if TEST_WAVELET_MATRIX_OPTIMIZED_CUDA
        Halide::Runtime::Buffer<T> dst_wm_cuda(W, H);
        {
            namespace wm = wavelet_matrix_median;
            wm::cuda_error_check(__LINE__, __FILE__);
            const int exH= H + 2 * radius;
            const int exW= W + 2 * radius;
            T* src_cu = reinterpret_cast<T*>(wm::alloc_and_transfer_cuda(
                &src(-radius, -radius),exH, exW, src.dim(1).stride(), sizeof(T)));
            T *res_cu;
            cudaMalloc(&res_cu, H * W * sizeof(T));
            wm::wm_median_cuda_alloc(radius, H, W, typeid(T).hash_code());
            
            if constexpr(std::is_same<T, float>()){
                float* val_in_cu = wm::get_float_supporter_val_in_cu();
                cudaMemcpy(val_in_cu, src_cu, sizeof(T) * exH * exW, cudaMemcpyDeviceToDevice);
            }
            cudaStreamSynchronize(0);

            t_wm_cuda = bench([&]() {
                wm::wm_median_cuda(src_cu, radius, H, W, exW, res_cu, W);
                cudaStreamSynchronize(0);
            });

            cudaStreamSynchronize(0);
            wm::transfer_mem_cuda_to_host(res_cu, H, W, &dst_wm_cuda(0, 0), dst_wm_cuda.dim(1).stride(), sizeof(T));
            cudaFree(src_cu);
            cudaFree(res_cu);
            wm::wm_median_cuda_delete(typeid(T).hash_code());
            wm::cuda_error_check(__LINE__, __FILE__);
        }
#endif


#if TEST_WAVELET_MATRIX_RUNTIMEONLY_CUDA
        double t_wm_runtime_cuda = 0;
        Halide::Runtime::Buffer<T> dst_wm_runtime_cuda(W, H);
        {
            namespace wm = wavelet_matrix_median;
            wm::cuda_error_check(__LINE__, __FILE__);
            const int exH= H + 2 * radius;
            const int exW= W + 2 * radius;
            T* src_cu = reinterpret_cast<T*>(wm::alloc_and_transfer_cuda(
                &src(-radius, -radius),exH, exW, src.dim(1).stride(), sizeof(T)));
            T *res_cu;
            cudaMalloc(&res_cu, H * W * sizeof(T));
            wm::wm_median_cuda_alloc(radius, H, W, typeid(T).hash_code());
            
            if constexpr(std::is_same<T, float>()){
                float* val_in_cu = wm::get_float_supporter_val_in_cu();
                cudaMemcpy(val_in_cu, src_cu, sizeof(T) * exH * exW, cudaMemcpyDeviceToDevice);
            }
            wm::wm_median_cuda(src_cu, radius, H, W, exW, res_cu, W);
            cudaStreamSynchronize(0);

            t_wm_runtime_cuda = bench([&]() {
                wm::wm_median_cuda_runtime_only(radius, H, W, res_cu, W);
                cudaStreamSynchronize(0);
            });

            cudaStreamSynchronize(0);
            wm::transfer_mem_cuda_to_host(res_cu, H, W, &dst_wm_runtime_cuda(0, 0), dst_wm_runtime_cuda.dim(1).stride(), sizeof(T));
            cudaFree(src_cu);
            cudaFree(res_cu);
            wm::wm_median_cuda_delete(typeid(T).hash_code());
            wm::cuda_error_check(__LINE__, __FILE__);
        }
#endif


        double t_opencv_cuda = 0;
#if TEST_OPENCV_CUDA
        Halide::Runtime::Buffer<T> dst_opencv_cuda(W, H);
        if (std::is_same<T, uint8_t>::value) {
            auto opencv_type = CV_8UC1;
            cv::Mat cv_src(H + padding * 2, W + padding * 2, opencv_type, src.data());
            cv::Mat cv_dst(H + padding * 2, W + padding * 2, opencv_type);

            cv::cuda::GpuMat dst_gpu, src_gpu;
            src_gpu.upload(cv_src);

            auto filt = cv::cuda::createMedianFilter(opencv_type, radius * 2 + 1);
            t_opencv_cuda =
                bench([&]() {
                    filt->apply(src_gpu, dst_gpu);
                    cv::cuda::Stream::Null().waitForCompletion();
                });

            dst_gpu.download(cv_dst);

            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    dst_opencv_cuda(x, y) = cv_dst.at<T>(y + padding, x + padding);
                }
            }
        }
#endif

        double t_arrayfire = 0;
        // Arrayfire throws a not-supported error above 15 x 15
#if TEST_ARRAYFIRE
        const int arrayfire_max_radius = 7;
        Halide::Runtime::Buffer<T> dst_arrayfire(W, H);
        if (radius <= arrayfire_max_radius) {

            af::array af_src(W + padding * 2, H + padding * 2, src.data());

            // Arrayfire has no option to not have a boundary
            // condition and just compute an inset region instead.
            af::array af_dst = af::medfilt2(af_src, radius * 2 + 1, radius * 2 + 1, AF_PAD_ZERO);

            t_arrayfire = bench(
                [&]() {
                    af::medfilt2(af_src, radius * 2 + 1, radius * 2 + 1, AF_PAD_ZERO);
                    af::sync();
                });

            const T *ptr = af_dst.host<T>();
            const int stride = W + padding * 2;
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    dst_arrayfire(x, y) = ptr[(y + padding) * stride + (x + padding)];
                }
            }
        }
#endif

        double t_kim = 0;
#if TEST_KIM
        Halide::Runtime::Buffer<T> dst_kim(W, H);
        {
            decltype(&kim_median_filter_u8_1) kim_fns_u8[] = {kim_median_filter_u8_1, kim_median_filter_u8_2};
            decltype(&kim_median_filter_u16_1) kim_fns_u16[] = {kim_median_filter_u16_1, kim_median_filter_u16_2};
            decltype(&kim_median_filter_f32_1) kim_fns_f32[] = {kim_median_filter_f32_1, kim_median_filter_f32_2};

            if (radius <= 2) {
                decltype(&kim_median_filter_u8_1) kim_fn;
                if (std::is_same<T, uint8_t>::value) {
                    kim_fn = kim_fns_u8[radius - 1];
                } else if (std::is_same<T, uint16_t>::value) {
                    kim_fn = kim_fns_u16[radius - 1];
                } else {
                    kim_fn = kim_fns_f32[radius - 1];
                }

                t_kim =
                    bench(
                        [&]() {
                            kim_fn(src, dst_kim);
                        });
            }
        }
#endif

        double t_static = 0;
#if TEST_STATIC
        Halide::Runtime::Buffer<T> dst_static(W, H);
        {
            decltype(&median_filter_u8_1) halide_fns_u8[] = {median_filter_u8_1, median_filter_u8_2, median_filter_u8_3, median_filter_u8_4, median_filter_u8_5, median_filter_u8_6, median_filter_u8_7, median_filter_u8_8, median_filter_u8_9, median_filter_u8_10, median_filter_u8_11, median_filter_u8_12, median_filter_u8_13, median_filter_u8_14                , median_filter_u8_15, median_filter_u8_16, median_filter_u8_17, median_filter_u8_18};
            decltype(&median_filter_u16_1) halide_fns_u16[] = {median_filter_u16_1, median_filter_u16_2, median_filter_u16_3, median_filter_u16_4, median_filter_u16_5, median_filter_u16_6, median_filter_u16_7, median_filter_u16_8, median_filter_u16_9, median_filter_u16_10, median_filter_u16_11, median_filter_u16_12, median_filter_u16_13, median_filter_u16_14, median_filter_u8_15, median_filter_u8_16, median_filter_u8_17, median_filter_u8_18};
            decltype(&median_filter_f32_1) halide_fns_f32[] = {median_filter_f32_1, median_filter_f32_2, median_filter_f32_3, median_filter_f32_4, median_filter_f32_5, median_filter_f32_6, median_filter_f32_7, median_filter_f32_8, median_filter_f32_9, median_filter_f32_10, median_filter_f32_11, median_filter_f32_12, median_filter_f32_13, median_filter_f32_14, median_filter_u8_15, median_filter_u8_16, median_filter_u8_17, median_filter_u8_18};

            if (radius >= MIN_RADIUS && radius <= MAX_RADIUS) {
                decltype(&median_filter_u8_1) halide_fn;
                if (std::is_same<T, uint8_t>::value) {
                    halide_fn = halide_fns_u8[radius - 1];
                } else if (std::is_same<T, uint16_t>::value) {
                    halide_fn = halide_fns_u16[radius - 1];
                } else {
                    halide_fn = halide_fns_f32[radius - 1];
                }

                t_static =
                    bench(
                        [&]() {
                            halide_fn(src, dst_static);
                        });
            }
        }
#endif

        double t_static_cuda = 0;
#if TEST_STATIC_CUDA
        Halide::Runtime::Buffer<T> dst_static_cuda(W, H);
        {
            decltype(&median_filter_cuda_u8_1) halide_fns_u8[] = {median_filter_cuda_u8_1, median_filter_cuda_u8_2, median_filter_cuda_u8_3, median_filter_cuda_u8_4, median_filter_cuda_u8_5, median_filter_cuda_u8_6, median_filter_cuda_u8_7, median_filter_cuda_u8_8, median_filter_cuda_u8_9, median_filter_cuda_u8_10, median_filter_cuda_u8_11, median_filter_cuda_u8_12, median_filter_cuda_u8_13, median_filter_cuda_u8_14,                  median_filter_cuda_u8_15,  median_filter_cuda_u8_16,  median_filter_cuda_u8_17,  median_filter_cuda_u8_18};
            decltype(&median_filter_cuda_u16_1) halide_fns_u16[] = {median_filter_cuda_u16_1, median_filter_cuda_u16_2, median_filter_cuda_u16_3, median_filter_cuda_u16_4, median_filter_cuda_u16_5, median_filter_cuda_u16_6, median_filter_cuda_u16_7, median_filter_cuda_u16_8, median_filter_cuda_u16_9, median_filter_cuda_u16_10, median_filter_cuda_u16_11, median_filter_cuda_u16_12, median_filter_cuda_u16_13, median_filter_cuda_u16_14, median_filter_cuda_u16_15, median_filter_cuda_u16_16, median_filter_cuda_u16_17, median_filter_cuda_u16_18};
            decltype(&median_filter_cuda_f32_1) halide_fns_f32[] = {median_filter_cuda_f32_1, median_filter_cuda_f32_2, median_filter_cuda_f32_3, median_filter_cuda_f32_4, median_filter_cuda_f32_5, median_filter_cuda_f32_6, median_filter_cuda_f32_7, median_filter_cuda_f32_8, median_filter_cuda_f32_9, median_filter_cuda_f32_10, median_filter_cuda_f32_11, median_filter_cuda_f32_12, median_filter_cuda_f32_13, median_filter_cuda_f32_14, median_filter_cuda_f32_15, median_filter_cuda_f32_16, median_filter_cuda_f32_17, median_filter_cuda_f32_18};

            if (radius >= MIN_RADIUS && radius <= MAX_RADIUS) {
                decltype(&median_filter_cuda_u8_1) halide_fn;
                if (std::is_same<T, uint8_t>::value) {
                    halide_fn = halide_fns_u8[radius - 1];
                } else if (std::is_same<T, uint16_t>::value) {
                    halide_fn = halide_fns_u16[radius - 1];
                } else {
                    halide_fn = halide_fns_f32[radius - 1];
                }

                src.copy_to_device(halide_cuda_device_interface());
                t_static_cuda =
                    bench(
                        [&]() {
                            halide_fn(src, dst_static_cuda);
                            dst_static_cuda.device_sync();
                        });
                dst_static_cuda.copy_to_host();
            }
        }
#endif

        double t_ipp = 0;
#if TEST_IPP
        // IPP's implementation

        Halide::Runtime::Buffer<T> dst_ipp(W, H);
        {
            // IPP wants to use an externally-allocated tmp buffer of
            // modest size. We'll just allocate a giant one and not
            // include the allocation time in the benchmarking loop.
            Halide::Runtime::Buffer<T> scratch(W, H);

            struct Closure {
                Halide::Runtime::Buffer<T> *src, *dst, *scratch;
                int W, H, radius;
            } closure{&src, &dst_ipp, &scratch, W, H, radius};

            auto one_strip = [](void *ucon, int y, uint8_t *closure) {
                Closure *c = (Closure *)closure;
                y *= 32;
                if (std::is_same<T, uint8_t>::value) {
                    ippiFilterMedianBorder_8u_C1R(
                        (const uint8_t *)&((*(c->src))(0, y)), c->src->dim(1).stride(),
                        (uint8_t *)&((*(c->dst))(0, y)), c->dst->dim(1).stride(),
                        IppiSize{c->W, std::min(32, c->H - y)},
                        IppiSize{c->radius * 2 + 1, c->radius * 2 + 1},
                        ippBorderInMem /* no border, just read off the edge */, 0,
                        (uint8_t *)&((*(c->scratch))(0, y)));
                } else if (std::is_same<T, uint16_t>::value) {
                    ippiFilterMedianBorder_16u_C1R(
                        (const uint16_t *)&((*(c->src))(0, y)), c->src->dim(1).stride() * 2,
                        (uint16_t *)&((*(c->dst))(0, y)), c->dst->dim(1).stride() * 2,
                        IppiSize{c->W, std::min(32, c->H - y)},
                        IppiSize{c->radius * 2 + 1, c->radius * 2 + 1},
                        ippBorderInMem /* no border, just read off the edge */, 0,
                        (uint8_t *)&((*(c->scratch))(0, y)));
                } else {
                    ippiFilterMedianBorder_32f_C1R(
                        (const float *)&((*(c->src))(0, y)), c->src->dim(1).stride() * 4,
                        (float *)&((*(c->dst))(0, y)), c->dst->dim(1).stride() * 4,
                        IppiSize{c->W, std::min(32, c->H - y)},
                        IppiSize{c->radius * 2 + 1, c->radius * 2 + 1},
                        ippBorderInMem /* no border, just read off the edge */, 0,
                        (uint8_t *)&((*(c->scratch))(0, y)));
                }

                return 0;
            };
            t_ipp =
                bench([&]() {
                    halide_do_par_for(nullptr, one_strip, 0, (H + 31) / 32, (uint8_t *)&closure);
                });
        }
#endif

        double t_npp = 0;
#if TEST_NPP
        Halide::Runtime::Buffer<T> dst_npp(W, H);
        {
            dst_npp.device_malloc(halide_cuda_device_interface());
            src.set_host_dirty();
            src.copy_to_device(halide_cuda_device_interface());

            ptrdiff_t offset = (intptr_t)(&src(0, 0)) - (intptr_t)(src.raw_buffer()->host);
            int ret = 0;

            // npp requires some absurdly large scratch buffers to compute median filters.
            halide_cuda_release_unused_device_allocations(nullptr);
            uint32_t size = 0;

            //if (std::is_same<T, uint8_t>::value && radius <= 7) {
            if (std::is_same<T, uint8_t>::value) {
                nppiFilterMedianGetBufferSize_8u_C1R(NppiSize{W, H},
                                                     NppiSize{2 * radius + 1, 2 * radius + 1},
                                                     &size);

                Halide::Runtime::Buffer<uint8_t> scratch(nullptr, size);
                if (size > 0) {
                    scratch.device_malloc(halide_cuda_device_interface());
                }

                t_npp =
                    bench([&]() {
                        ret = nppiFilterMedian_8u_C1R((const uint8_t *)(src.raw_buffer()->device + offset),
                                                      src.dim(1).stride(),
                                                      (uint8_t *)dst_npp.raw_buffer()->device,
                                                      dst_npp.dim(1).stride(),
                                                      NppiSize{W, H},
                                                      NppiSize{2 * radius + 1, 2 * radius + 1},
                                                      NppiPoint{radius, radius},  // Anchor
                                                      (uint8_t *)scratch.raw_buffer()->device);
                        dst_npp.device_sync();
                    });
            } else if (std::is_same<T, uint16_t>::value) {
                nppiFilterMedianGetBufferSize_16u_C1R(NppiSize{W, H},
                                                      NppiSize{2 * radius + 1, 2 * radius + 1},
                                                      &size);

                Halide::Runtime::Buffer<uint8_t> scratch(nullptr, size);
                if (size > 0) {
                    scratch.device_malloc(halide_cuda_device_interface());
                }

                t_npp =
                    bench([&]() {
                        ret = nppiFilterMedian_16u_C1R((const uint16_t *)(src.raw_buffer()->device + offset),
                                                       src.dim(1).stride() * sizeof(uint16_t),
                                                       (uint16_t *)dst_npp.raw_buffer()->device,
                                                       dst_npp.dim(1).stride() * sizeof(uint16_t),
                                                       NppiSize{W, H},
                                                       NppiSize{2 * radius + 1, 2 * radius + 1},
                                                       NppiPoint{radius, radius},  // Anchor
                                                       (uint8_t *)scratch.raw_buffer()->device);
                        dst_npp.device_sync();
                    });
            } else if (std::is_same<T, float>::value) {
                nppiFilterMedianGetBufferSize_32f_C1R(NppiSize{W, H},
                                                      NppiSize{2 * radius + 1, 2 * radius + 1},
                                                      &size);

                Halide::Runtime::Buffer<uint8_t> scratch(nullptr, size);
                if (size > 0) {
                    scratch.device_malloc(halide_cuda_device_interface());
                }

                t_npp =
                    bench([&]() {
                        ret = nppiFilterMedian_32f_C1R((const float *)(src.raw_buffer()->device + offset),
                                                       src.dim(1).stride() * sizeof(float),
                                                       (float *)dst_npp.raw_buffer()->device,
                                                       dst_npp.dim(1).stride() * sizeof(float),
                                                       NppiSize{W, H},
                                                       NppiSize{2 * radius + 1, 2 * radius + 1},
                                                       NppiPoint{radius, radius},  // Anchor
                                                       (uint8_t *)scratch.raw_buffer()->device);
                        dst_npp.device_sync();
                    });
            }
            assert(ret == 0);

            dst_npp.set_device_dirty();
            dst_npp.copy_to_host();
        }
#endif

        double t_dynamic_v2 = 0;
#if TEST_DYNAMIC_V2
        Halide::Runtime::Buffer<T> dst_dynamic_v2(W, H);
        t_dynamic_v2 = bench(
            [&]() {
                dynamic_median_filter(src, radius, dst_dynamic_v2);
            });
#endif

        double t_dynamic_cuda = 0;
#if TEST_DYNAMIC_CUDA
        Halide::Runtime::Buffer<T> dst_dynamic_cuda(W, H);
        t_dynamic_cuda = bench(
            [&]() {
                dynamic_median_filter(src, radius, dst_dynamic_cuda, true);
                dst_dynamic_cuda.device_sync();
            });
        assert(dst_dynamic_cuda.device_dirty());
        dst_dynamic_cuda.copy_to_host();
#endif

        double t_ctmf = 0;
#if TEST_CTMF
        Halide::Runtime::Buffer<T> dst_ctmf(W + radius * 2, H + radius * 2);
        dst_ctmf.set_min(-radius, -radius);
        if (std::is_same<T, uint8_t>::value) {
            t_ctmf =
                bench(
                    [&]() {
                        ctmf_u8((const uint8_t *)&src(-radius, -radius),
                                (uint8_t *)&dst_ctmf(-radius, -radius),
                                W + radius * 2,
                                H + radius * 2,
                                src.dim(1).stride(),
                                dst_ctmf.dim(1).stride(),
                                radius, 1, 32);
                        // 32 stripes gives the best runtime on the test image
                    });
        } else if (std::is_same<T, uint16_t>::value) {
            t_ctmf =
                bench(
                    [&]() {
                        ctmf_u16((const uint16_t *)&src(-radius, -radius),
                                 (uint16_t *)&dst_ctmf(-radius, -radius),
                                 W + radius * 2,
                                 H + radius * 2,
                                 src.dim(1).stride(),
                                 dst_ctmf.dim(1).stride(),
                                 radius, 1, 32);
                        // 32 stripes gives the best runtime on the test image
                    });
        }
#endif

        // Dump a tile of input and outputs for debugging
#if 0
        int x_min = 0, x_max = x_min + 16;
        int y_min = 0, y_max = y_min + 16;

        // Dump the top left of the correct and Halide-produced outputs
        printf("Input region:\n");
        for (int y = y_min - radius; y < y_max + radius; y++) {
            for (int x = x_min - radius; x < x_max + radius; x++) {
                printf("%6.1f ", (float)src(x, y));
            }
            printf("\n");
        }

        printf("Reference output:\n");
        for (int y = y_min; y < y_max; y++) {
            for (int x = x_min; x < x_max; x++) {
                printf("%6.1f ", (float)dst_kim(x, y));
            }
            printf("\n");
        }

        printf("Test output:\n");
        for (int y = y_min; y < y_max; y++) {
            for (int x = x_min; x < x_max; x++) {
                printf("%6.1f ", (float)dst_ctmf(x, y));
            }
            printf("\n");
        }

        printf("Wrong sites:\n");
        for (int y = y_min; y < y_max; y++) {
            for (int x = x_min; x < x_max; x++) {
                printf("   %c ", dst_ctmf(x, y) < dst_kim(x, y) ? '-' :
                                 dst_ctmf(x, y) > dst_kim(x, y) ? '+' :
                                                                  '=');
            }
            printf("\n");
        }
#endif

        // Check the outputs agree with Adams Dynamic, which works at all sizes
#if TEST_DYNAMIC_CUDA
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {

#if TEST_OPENCV
                if (t_opencv && dst_opencv(x, y) != dst_dynamic_cuda(x, y)) {
                    printf("dst_opencv vs dst_dynamic_cuda %d %d: %f vs %f\n", x, y, (float)dst_opencv(x, y), (float)dst_dynamic_cuda(x, y));
                    return -1;
                }
#endif

#if TEST_IPP
                if (t_ipp && dst_ipp(x, y) != dst_dynamic_cuda(x, y)) {
                    printf("dst_ipp vs dst_dynamic_cuda %d %d: %f vs %f\n", x, y, (float)dst_ipp(x, y), (float)dst_dynamic_cuda(x, y));
                    return -1;
                }
#endif

#if TEST_OPENCV_CUDA
                if (t_opencv_cuda && dst_opencv_cuda(x, y) != dst_dynamic_cuda(x, y)) {
                    printf("dst_opencv_cuda vs dst_dynamic_cuda %d %d: %f vs %f\n", x, y, (float)dst_opencv_cuda(x, y), (float)dst_dynamic_cuda(x, y));
                    return -1;
                }
#endif

#if TEST_WAVELET_MATRIX_OPTIMIZED_CUDA
                if (t_wm_cuda && dst_wm_cuda(x, y) != dst_dynamic_cuda(x, y)) {
                    printf("dst_wm_cuda vs dst_dynamic_cuda %d %d: %f vs %f\n", x, y, (float)dst_wm_cuda(x, y), (float)dst_dynamic_cuda(x, y));
                    return -1;
                }
#endif

#if TEST_WAVELET_MATRIX_RUNTIMEONLY_CUDA
                if (t_wm_runtime_cuda && dst_wm_runtime_cuda(x, y) != dst_dynamic_cuda(x, y)) {
                    printf("dst_wm_runtime_cuda vs dst_dynamic_cuda %d %d: %f vs %f\n", x, y, (float)dst_wm_runtime_cuda(x, y), (float)dst_dynamic_cuda(x, y));
                    return -1;
                }
#endif

#if TEST_KIM
                if (t_kim && dst_dynamic_cuda(x, y) != dst_kim(x, y)) {
                    printf("dst_dynamic_cuda vs dst_kim %d %d: %f vs %f\n", x, y, (float)dst_dynamic_cuda(x, y), (float)dst_kim(x, y));
                    return -1;
                }
#endif

#if TEST_STATIC
                if (t_static && dst_dynamic_cuda(x, y) != dst_static(x, y)) {
                    printf("dst_dynamic_cuda vs dst_static %d %d: %f vs %f\n", x, y, (float)dst_dynamic_cuda(x, y), (float)dst_static(x, y));
                    return -1;
                }
#endif

#if TEST_CTMF
                if (t_ctmf && dst_dynamic_cuda(x, y) != dst_ctmf(x, y)) {
                    printf("dst_dynamic_cuda vs dst_ctmf %d %d: %f vs %f\n", x, y, (float)dst_dynamic_cuda(x, y), (float)dst_ctmf(x, y));
                    return -1;
                }
#endif

#if TEST_STATIC_CUDA
                if (t_static_cuda && dst_dynamic_cuda(x, y) != dst_static_cuda(x, y)) {
                    printf("dst_dynamic_cuda vs dst_static_cuda %d %d: %f vs %f\n", x, y, (float)dst_dynamic_cuda(x, y), (float)dst_static_cuda(x, y));
                    return -1;
                }
#endif

#if TEST_NPP
                if (t_npp && dst_dynamic_cuda(x, y) != dst_npp(x, y)) {
                    printf("dst_dynamic_cuda vs dst_npp %d %d: %f vs %f\n", x, y, (float)dst_dynamic_cuda(x, y), (float)dst_npp(x, y));
                    return -1;
                }
#endif

#if TEST_ARRAYFIRE
                if (t_arrayfire && dst_dynamic_cuda(x, y) != dst_arrayfire(x, y)) {
                    printf("dst_dynamic_cuda vs dst_arrayfire %d %d: %f vs %f\n",
                           x, y, (float)dst_dynamic_cuda(x, y), (float)dst_arrayfire(x, y));
                    return -1;
                }
#endif

#if TEST_NAIVE
                if (t_naive && dst_dynamic_cuda(x, y) != dst_naive(x, y)) {
                    printf("dst_dynamic_cuda vs dst_naive %d %d: %f vs %f\n",
                           x, y, (float)dst_dynamic_cuda(x, y), (float)dst_naive(x, y));
                    return -1;
                }
#endif


#if TEST_WAVELET_MATRIX_NAIVE_CPU
                if (t_wm_naive_cpu && dst_dynamic_cuda(x, y) != dst_wm_naive_cpu(x, y)) {
                    printf("dst_dynamic_cuda vs dst_wm_naive_cpu %d %d: %f vs %f\n",
                           x, y, (float)dst_dynamic_cuda(x, y), (float)dst_wm_naive_cpu(x, y));
                    return -1;
                }
#endif

#if TEST_WAVELET_MATRIX_PARALLEL_CPU
                if (t_wm_parallel_cpu && dst_dynamic_cuda(x, y) != dst_wm_parallel_cpu(x, y)) {
                    printf("dst_dynamic_cuda vs dst_wm_parallel_cpu %d %d: %f vs %f\n",
                           x, y, (float)dst_dynamic_cuda(x, y), (float)dst_wm_parallel_cpu(x, y));
                    return -1;
                }
#endif

#if TEST_WAVELET_MATRIX_PARALLEL2_CPU
                if (t_wm_parallel2_cpu && dst_dynamic_cuda(x, y) != dst_wm_parallel2_cpu(x, y)) {
                    printf("dst_dynamic_cuda vs dst_wm_parallel2_cpu %d %d: %f vs %f\n",
                           x, y, (float)dst_dynamic_cuda(x, y), (float)dst_wm_parallel2_cpu(x, y));
                    return -1;
                }
#endif

#if TEST_DYNAMIC_V2
                if (t_dynamic_v2 && dst_dynamic_cuda(x, y) != dst_dynamic_v2(x, y)) {
                    printf("dst_dynamic_cuda vs dst_dynamic_v2 %d %d: %f vs %f\n",
                           x, y, (float)dst_dynamic_cuda(x, y), (float)dst_dynamic_v2(x, y));
                    return -1;
                }
#endif

#if TEST_DYNAMIC_CUDA
                if (t_dynamic_cuda && dst_dynamic_cuda(x, y) != dst_dynamic_cuda(x, y)) {
                    printf("dst_dynamic_cuda vs dst_dynamic_cuda %d %d: %f vs %f\n",
                           x, y, (float)dst_dynamic_cuda(x, y), (float)dst_dynamic_cuda(x, y));
                    return -1;
                }
#endif
            }
        }
#endif

        // Convert a time in milliseconds to a throughput in megapixels per second
        auto throughput = [=](float ms) {
            return (W * H) / (ms * 1000);
        };

        if (radius == min_radius) {
            // Print CSV header row
            fprintf(results, "Kernel radius");

#if TEST_NAIVE
            fprintf(results, ",naive");
#endif

#if TEST_WAVELET_MATRIX_NAIVE_CPU
            fprintf(results, ",wm_naive_cpu");
#endif

#if TEST_WAVELET_MATRIX_PARALLEL_CPU
            fprintf(results, ",wm_parallel_cpu");
#endif

#if TEST_WAVELET_MATRIX_PARALLEL2_CPU
            fprintf(results, ",wm_parallel2_cpu");
#endif

#if TEST_STATIC
            fprintf(results, ",cpu_static");
#endif

#if TEST_DYNAMIC_V2
            fprintf(results, ",cpu_dynamic");
#endif

#if TEST_IPP
            fprintf(results, ",ipp");
#endif

#if TEST_OPENCV
            fprintf(results, ",opencv");
#endif

#if TEST_OPENCV
            fprintf(results, ",kim");
#endif

#if TEST_CTMF
            fprintf(results, ",ctmf");
#endif

            // Start of GPU results. Repeat the kernel size.
            fprintf(results, ",Kernel radius");

#if TEST_STATIC_CUDA
            fprintf(results, ",adams_gpu_static");
#endif

#if TEST_DYNAMIC_CUDA
            fprintf(results, ",adams_gpu_dynamic");
#endif

#if TEST_NPP
            fprintf(results, ",npp");
#endif

#if TEST_ARRAYFIRE
            fprintf(results, ",arrayfire");
#endif

#if TEST_OPENCV_CUDA
            fprintf(results, ",opencv_cuda");
#endif

#if TEST_WAVELET_MATRIX_OPTIMIZED_CUDA
            fprintf(results, ",wavelet_matrix_cuda");
#endif

#if TEST_WAVELET_MATRIX_RUNTIMEONLY_CUDA
            fprintf(results, ",wm_runtimeonly_cuda");
#endif

            fprintf(results, "\n");
        }

        printf("Radius %2d: ", radius);
        fprintf(results, "%d", radius);

#if TEST_NAIVE
        printf("naive: %7.3f ", t_naive);
        fprintf(results, ",%f", throughput(t_naive));
#endif

#if TEST_WAVELET_MATRIX_NAIVE_CPU
        printf("wm_naive_cpu: %7.3f ", t_wm_naive_cpu);
        fprintf(results, ",%f", throughput(t_wm_naive_cpu));
#endif

#if TEST_WAVELET_MATRIX_PARALLEL_CPU
        printf("wm_parallel_cpu: %7.3f ", t_wm_parallel_cpu);
        fprintf(results, ",%f", throughput(t_wm_parallel_cpu));
#endif

#if TEST_WAVELET_MATRIX_PARALLEL2_CPU
        printf("wm_parallel2_cpu: %7.3f ", t_wm_parallel2_cpu);
        fprintf(results, ",%f", throughput(t_wm_parallel2_cpu));
#endif

#if TEST_STATIC
        printf("cpu: %7.3f ", t_static);
        fprintf(results, ",%f", throughput(t_static));
#endif

#if TEST_DYNAMIC_V2
        printf("dynamic_v2: %7.3f (%7.2f x) ", t_dynamic_v2, t_dynamic_v2 / t_static);
        fprintf(results, ",%f", throughput(t_dynamic_v2));
#endif

#if TEST_IPP
        printf("ipp: %7.3f (%7.2f x) ", t_ipp, t_ipp / t_static);
        fprintf(results, ",%f", throughput(t_ipp));
#endif

#if TEST_OPENCV
        printf("opencv: %7.3f (%7.2f x) ", t_opencv, t_opencv / t_static);
        fprintf(results, ",%f", throughput(t_opencv));
#endif

#if TEST_KIM
        printf("kim: %7.3f ", t_kim);
        fprintf(results, ",%f", throughput(t_kim));
#endif

#if TEST_CTMF
        printf("ctmf: %7.3f (%7.2f x) ", t_ctmf, t_ctmf / t_static);
        fprintf(results, ",%f", throughput(t_ctmf));
#endif

        // Separate CPU results from GPU results
        printf("\n           ");
        fprintf(results, ",%d", radius);

#if TEST_STATIC_CUDA
        printf("adams_cuda: %7.3f  (%7.2f x) ", t_static_cuda, t_static_cuda / t_wm_cuda);
        fprintf(results, ",%f", throughput(t_static_cuda));
#endif

#if TEST_DYNAMIC_CUDA
        printf("adams_dynamic_cuda: %7.3f (%7.2f x) ", t_dynamic_cuda, t_dynamic_cuda / t_wm_cuda);
        fprintf(results, ",%f", throughput(t_dynamic_cuda));
#endif

#if TEST_NPP
        printf("npp: %7.3f (%7.2f x) ", t_npp, t_npp / t_wm_cuda);
        fprintf(results, ",%f", throughput(t_npp));
#endif

#if TEST_ARRAYFIRE
        printf("arrayfire: %7.3f (%7.2f x) ", t_arrayfire, t_arrayfire / t_wm_cuda);
        fprintf(results, ",%f", throughput(t_arrayfire));
#endif

#if TEST_OPENCV_CUDA
        printf("opencv_cuda: %7.3f (%7.2f x) ", t_opencv_cuda, t_opencv_cuda / t_wm_cuda);
        fprintf(results, ",%f", throughput(t_opencv_cuda));
#endif

#if TEST_WAVELET_MATRIX_OPTIMIZED_CUDA
        printf("wm_cuda: %7.3f (%7.2f x) ", t_wm_cuda, t_wm_cuda / t_wm_cuda);
        fprintf(results, ",%f", throughput(t_wm_cuda));
#endif

#if TEST_WAVELET_MATRIX_RUNTIMEONLY_CUDA
        printf("wm_runtimeonly_cuda: %7.3f (%7.2f x) ", t_wm_runtime_cuda, t_wm_runtime_cuda / t_wm_cuda);
        fprintf(results, ",%f", throughput(t_wm_runtime_cuda));
#endif

        fprintf(results, "\n");

        printf("\n");
    }

    fclose(results);

    return 0;
}

int main(int argc, char **argv) {
    // The cuda implementations use lots of registers, which lowers
    // occupancy. That's fine. Better that than spilling to local.
    setenv("HL_CUDA_JIT_MAX_REGISTERS", "256", 0);

    int min_radius = MIN_RADIUS, max_radius = 50;

    if (argc > 1) {
        min_radius = max_radius = std::atoi(argv[1]);
    }
    if (argc > 2) {
        max_radius = std::atoi(argv[2]);
    }

    test<uint8_t>(min_radius, max_radius,  "results_uint8.csv");
    test<uint16_t>(min_radius, max_radius, "results_uint16.csv");
    test<float>(min_radius, max_radius,    "results_float.csv");
    
    return 0;
}
