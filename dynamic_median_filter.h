#ifndef DYNAMIC_MEDIAN_FILTER_H
#define DYNAMIC_MEDIAN_FILTER_H

#include <HalideBuffer.h>

void dynamic_median_filter(Halide::Runtime::Buffer<const uint8_t> src,
                           int radius,
                           Halide::Runtime::Buffer<uint8_t> &dst,
                           bool use_cuda = false,
                           int max_leaf = 0,
                           bool verbose = false);

void dynamic_median_filter(Halide::Runtime::Buffer<const uint16_t> src,
                           int radius,
                           Halide::Runtime::Buffer<uint16_t> &dst,
                           bool use_cuda = false,
                           int max_leaf = 0,
                           bool verbose = false);

void dynamic_median_filter(Halide::Runtime::Buffer<const float> src,
                           int radius,
                           Halide::Runtime::Buffer<float> &dst,
                           bool use_cuda = false,
                           int max_leaf = 0,
                           bool verbose = false);

#endif
