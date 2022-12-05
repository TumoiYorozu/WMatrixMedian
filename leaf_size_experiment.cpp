#include <HalideRuntimeCuda.h>
#include <halide_benchmark.h>

#include "dynamic_median_filter.h"

template<typename Callable>
double bench(Callable c) {
    Halide::Tools::BenchmarkConfig config;
    config.accuracy = 0.0001f;
    return Halide::Tools::benchmark(c, config) * 1e3;
}

int main(int argc, char **argv) {
    // For a single sample task, measure the effect of leaf size on
    // swaps and performance

    const int W = 2560, H = 1600, pad = 64;

    const int radius = 50;

    Halide::Runtime::Buffer<uint8_t> src(W + pad * 2, H + pad * 2);
    src.set_min(-pad, -pad);
    src.fill(0);

    Halide::Runtime::Buffer<uint8_t> dst(W, H);

    // In the makefile, set max_leaf = 24 for the target
    // bin/dynamic_median_filter_u8_8x8_v2.a to get correct output.
    for (int i = 1; i < 13; i++) {
        double t = bench([&]() {
            dynamic_median_filter(src, radius, dst, true, i * 2, true);
            dst.device_sync();
        });
        printf("op limit: %d runtime: %f\n", i, t);
    }

    return 0;
}
