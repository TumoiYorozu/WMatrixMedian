#include <HalideRuntimeCuda.h>

#include "dynamic_median_filter.h"

#include "dynamic_median_filter_f32_2x2_v2.h"
#include "dynamic_median_filter_f32_4x4_v2.h"
#include "dynamic_median_filter_f32_8x8_v2.h"
#include "dynamic_median_filter_u16_2x2_v2.h"
#include "dynamic_median_filter_u16_4x4_v2.h"
#include "dynamic_median_filter_u16_8x8_v2.h"
#include "dynamic_median_filter_u8_2x2_v2.h"
#include "dynamic_median_filter_u8_4x4_v2.h"
#include "dynamic_median_filter_u8_8x8_v2.h"

#include "dynamic_median_filter_cuda_f32_2x2_v2.h"
#include "dynamic_median_filter_cuda_f32_4x4_v2.h"
#include "dynamic_median_filter_cuda_f32_8x8_v2.h"
#include "dynamic_median_filter_cuda_u16_2x2_v2.h"
#include "dynamic_median_filter_cuda_u16_4x4_v2.h"
#include "dynamic_median_filter_cuda_u16_8x8_v2.h"
#include "dynamic_median_filter_cuda_u8_2x2_v2.h"
#include "dynamic_median_filter_cuda_u8_4x4_v2.h"
#include "dynamic_median_filter_cuda_u8_8x8_v2.h"
#include "sorting_bytecode.h"

#include <map>
using std::map;
using std::tuple;
using std::vector;

using Halide::Runtime::Buffer;

struct SortingProgram {
    vector<Instruction> col_sort_instructions, col_merge_instructions;
    Buffer<int> col_sort_instructions_buf, col_merge_instructions_buf;
    int working_mem = 0;
    int tw = 0, th = 0;
    bool initialized = false;
};

SortingProgram &prepare_program(int radius, bool use_cuda, int max_leaf, bool verbose) {
    static map<tuple<int, int, bool>, SortingProgram> instruction_cache;

    auto &e = instruction_cache[{radius, max_leaf, use_cuda}];
    if (!e.initialized) {
        e.initialized = true;

        // The tile size.
        const int tw = (radius >= 30 && !use_cuda) ? 8 :
                       radius >= 7                 ? 4 :
                                                     2;
        const int th = tw;

        const int diameter = radius * 2 + 1;

        // First we must construct the sorting networks we're going to
        // use, and then encode that into instructions for a sorting
        // network interpreter. Constructing this program is outside
        // the loop over pixels, and we can assume this gets cached
        // per-radius, so we leave it out of the benchmark
        vector<Op> col_sort_network = sort_1d(diameter - th + 1, 0, diameter - th);

        // The network that extracts one tile's worth of medians from the sorted column pairs
        vector<Op> col_merge_network = tiled_median_2d(diameter, diameter, tw, th);

        if (verbose) {
            print_op_histogram(col_merge_network);
        }

        // The table of instructions we're going to use. Must be kept in sync with the generator.
        auto table = get_instruction_table(max_leaf);

        // Trim the program down to instructions that fit within the limit
        col_merge_network = limit_max_op_length(col_merge_network, max_leaf);
        col_sort_network = limit_max_op_length(col_sort_network, max_leaf);

        // Compute how much working memory the interpreter will need
        int working_mem = compute_working_memory(col_merge_network);
        // printf("Working memory: %d\n", working_mem);

        // Encode the instructions into bytecode
        for (Op op : col_sort_network) {
            if (op.op == AssertSorted ||
                op.op == Print) {
                continue;
            }
            e.col_sort_instructions.push_back(encode(op, table));
        }
        for (Op op : col_merge_network) {
            if (op.op == AssertSorted ||
                op.op == Print) {
                continue;
            }
            e.col_merge_instructions.push_back(encode(op, table));
        }

        if (verbose) {
            PerformanceCounters col_sort_perf = estimate_performance(col_sort_network);
            PerformanceCounters col_merge_perf = estimate_performance(col_merge_network);

            printf("Col sort counters: instructions: %d loads: %d stores: %d swaps: %d\n",
                   (int)e.col_sort_instructions.size(),
                   col_sort_perf.loads, col_sort_perf.stores, col_sort_perf.swaps);
            printf("Col merge counters: instructions: %d loads: %d stores: %d swaps: %d\n",
                   (int)e.col_merge_instructions.size(),
                   col_merge_perf.loads, col_merge_perf.stores, col_merge_perf.swaps);

            double instructions_per_pixel = (double)e.col_sort_instructions.size() / th + (double)e.col_merge_instructions.size() / (tw * th);
            double loads_per_pixel = (double)col_sort_perf.loads / th + (double)col_merge_perf.loads / (tw * th);
            double stores_per_pixel = (double)col_sort_perf.stores / th + (double)col_merge_perf.stores / (tw * th);
            double swaps_per_pixel = (double)col_sort_perf.swaps / th + (double)col_merge_perf.swaps / (tw * th);
            printf("Total per-pixel instructions: %.2f loads: %.2f stores: %.2f swaps: %.2f\n",
                   instructions_per_pixel,
                   loads_per_pixel,
                   stores_per_pixel,
                   swaps_per_pixel);

            vector<Op> combined = col_sort_network;
            combined.insert(combined.end(), col_merge_network.begin(), col_merge_network.end());
            print_op_histogram(combined);
        }

        e.col_merge_instructions_buf =
            Buffer<int>((int *)e.col_merge_instructions.data(),
                        sizeof(e.col_merge_instructions[0]) / sizeof(int),
                        (int)e.col_merge_instructions.size());

        e.col_sort_instructions_buf =
            Buffer<int>((int *)e.col_sort_instructions.data(),
                        sizeof(e.col_sort_instructions[0]) / sizeof(int),
                        (int)e.col_sort_instructions.size());

        e.col_sort_instructions_buf.set_host_dirty();
        e.col_merge_instructions_buf.set_host_dirty();

        e.tw = tw;
        e.th = th;
        e.working_mem = working_mem;
    }
    return e;
}

int leaf_size(int max_leaf, bool use_cuda) {
    if (max_leaf > 0) {
        return max_leaf;
    } else if (use_cuda) {
        return 24;
    } else {
        return 14;
    }
}

void dynamic_median_filter(Halide::Runtime::Buffer<const uint8_t> src,
                           int radius,
                           Halide::Runtime::Buffer<uint8_t> &dst,
                           bool use_cuda,
                           int max_leaf,
                           bool verbose) {

    auto &p = prepare_program(radius, use_cuda, leaf_size(max_leaf, use_cuda), verbose);
    if (use_cuda) {
        if (p.tw == 2 && p.th == 2) {
            dynamic_median_filter_cuda_u8_2x2_v2(radius,
                                                 p.col_sort_instructions_buf,
                                                 p.col_merge_instructions_buf,
                                                 p.working_mem,
                                                 src, dst);
        } else if (p.tw == 4 && p.th == 4) {
            dynamic_median_filter_cuda_u8_4x4_v2(radius,
                                                 p.col_sort_instructions_buf,
                                                 p.col_merge_instructions_buf,
                                                 p.working_mem,
                                                 src, dst);
        } else {
            assert(p.tw == 8 && p.th == 8);
            dynamic_median_filter_cuda_u8_8x8_v2(radius,
                                                 p.col_sort_instructions_buf,
                                                 p.col_merge_instructions_buf,
                                                 p.working_mem,
                                                 src, dst);
        }
    } else {
        if (p.tw == 2 && p.th == 2) {
            dynamic_median_filter_u8_2x2_v2(radius,
                                            p.col_sort_instructions_buf,
                                            p.col_merge_instructions_buf,
                                            p.working_mem,
                                            src, dst);
        } else if (p.tw == 4 && p.th == 4) {
            dynamic_median_filter_u8_4x4_v2(radius,
                                            p.col_sort_instructions_buf,
                                            p.col_merge_instructions_buf,
                                            p.working_mem,
                                            src, dst);
        } else {
            assert(p.tw == 8 && p.th == 8);
            dynamic_median_filter_u8_8x8_v2(radius,
                                            p.col_sort_instructions_buf,
                                            p.col_merge_instructions_buf,
                                            p.working_mem,
                                            src, dst);
        }
    }
}

void dynamic_median_filter(Halide::Runtime::Buffer<const uint16_t> src,
                           int radius,
                           Halide::Runtime::Buffer<uint16_t> &dst,
                           bool use_cuda,
                           int max_leaf,
                           bool verbose) {
    auto &p = prepare_program(radius, use_cuda, leaf_size(max_leaf, use_cuda), verbose);
    if (use_cuda) {
        if (p.tw == 2 && p.th == 2) {
            dynamic_median_filter_cuda_u16_2x2_v2(radius,
                                                  p.col_sort_instructions_buf,
                                                  p.col_merge_instructions_buf,
                                                  p.working_mem,
                                                  src, dst);
        } else if (p.tw == 4 && p.th == 4) {
            dynamic_median_filter_cuda_u16_4x4_v2(radius,
                                                  p.col_sort_instructions_buf,
                                                  p.col_merge_instructions_buf,
                                                  p.working_mem,
                                                  src, dst);
        } else {
            assert(p.tw == 8 && p.th == 8);
            dynamic_median_filter_cuda_u16_8x8_v2(radius,
                                                  p.col_sort_instructions_buf,
                                                  p.col_merge_instructions_buf,
                                                  p.working_mem,
                                                  src, dst);
        }
    } else {
        if (p.tw == 2 && p.th == 2) {
            dynamic_median_filter_u16_2x2_v2(radius,
                                             p.col_sort_instructions_buf,
                                             p.col_merge_instructions_buf,
                                             p.working_mem,
                                             src, dst);
        } else if (p.tw == 4 && p.th == 4) {
            dynamic_median_filter_u16_4x4_v2(radius,
                                             p.col_sort_instructions_buf,
                                             p.col_merge_instructions_buf,
                                             p.working_mem,
                                             src, dst);
        } else {
            assert(p.tw == 8 && p.th == 8);
            dynamic_median_filter_u16_8x8_v2(radius,
                                             p.col_sort_instructions_buf,
                                             p.col_merge_instructions_buf,
                                             p.working_mem,
                                             src, dst);
        }
    }
}

void dynamic_median_filter(Halide::Runtime::Buffer<const float> src,
                           int radius,
                           Halide::Runtime::Buffer<float> &dst,
                           bool use_cuda,
                           int max_leaf,
                           bool verbose) {
    auto &p = prepare_program(radius, use_cuda, leaf_size(max_leaf, use_cuda), verbose);
    if (use_cuda) {
        if (p.tw == 2 && p.th == 2) {
            dynamic_median_filter_cuda_f32_2x2_v2(radius,
                                                  p.col_sort_instructions_buf,
                                                  p.col_merge_instructions_buf,
                                                  p.working_mem,
                                                  src, dst);
        } else if (p.tw == 4 && p.th == 4) {
            dynamic_median_filter_cuda_f32_4x4_v2(radius,
                                                  p.col_sort_instructions_buf,
                                                  p.col_merge_instructions_buf,
                                                  p.working_mem,
                                                  src, dst);
        } else {
            assert(p.tw == 8 && p.th == 8);
            dynamic_median_filter_cuda_f32_8x8_v2(radius,
                                                  p.col_sort_instructions_buf,
                                                  p.col_merge_instructions_buf,
                                                  p.working_mem,
                                                  src, dst);
        }
    } else {
        if (p.tw == 2 && p.th == 2) {
            dynamic_median_filter_f32_2x2_v2(radius,
                                             p.col_sort_instructions_buf,
                                             p.col_merge_instructions_buf,
                                             p.working_mem,
                                             src, dst);
        } else if (p.tw == 4 && p.th == 4) {
            dynamic_median_filter_f32_4x4_v2(radius,
                                             p.col_sort_instructions_buf,
                                             p.col_merge_instructions_buf,
                                             p.working_mem,
                                             src, dst);
        } else {
            assert(p.tw == 8 && p.th == 8);
            dynamic_median_filter_f32_8x8_v2(radius,
                                             p.col_sort_instructions_buf,
                                             p.col_merge_instructions_buf,
                                             p.working_mem,
                                             src, dst);
        }
    }
}
