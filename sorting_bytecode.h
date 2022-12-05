#ifndef SORTING_BYTECODE_H
#define SORTING_BYTECODE_H

#include <vector>

// A bytecode that describe in-place sorting operations performed on a
// 2D grid.
enum OpCode {
    // Sort a sequence
    Sort = 0,

    // Assert that a sequence is sorted. Some interpreters may ignore
    // this.
    AssertSorted = 1,

    // Merge two sorted sequences in place
    Merge = 2,

    // Just copy from s1 to s2, which must be the same length
    Copy = 3,

    // Print a sequence to stdout. Some interpreters may ignore this.
    Print = 4,
};

// A linear sequence
struct Seq {
    Seq() = default;
    Seq(const Seq &other) = default;
    Seq(int x, int dx, int len)
        : x(x), dx(dx), len(len) {
    }

    // The start
    int x = 0;

    // The step
    int dx = 0;

    // The length;
    int len = 0;
};

// Sort a sequence in-place, or merge two subsequences.
struct Op {
    // Sort, merge, or pairwise swap. If sort, any field ending in '2'
    // below gets ignored.
    OpCode op = Sort;

    // Output range of interest. Inclusive. The sort is permitted to
    // be incorrect outside of this range.
    int min_idx = 0, max_idx = 0;

    // The first sequence
    Seq s1;

    // The second sequence.
    Seq s2;

    // The reason the op is being performed. Used for
    // debugging. Should be a string constant.
    const char *reason = "";

    void print() const;
};

// Sort a 2D rectangle into scanline order, using the 1D primitives
// above. Sorts the rows then uses a simple merge across them. Ignores
// min_idx/max_idx.
std::vector<Op> sort_2d_naive(int w, int h, int min_idx, int max_idx);

// Sort a 2D rectangle into scanline order, using the 1D primitives
// above. Aggressively eliminates things outside [min_idx, max_idx] as
// it goes.
std::vector<Op> sort_2d(int w, int h, int min_idx, int max_idx);

// Just sort in 1d along the x dimension
std::vector<Op> sort_1d(int w, int min_idx, int max_idx);

// Take a pair of arrays stored back-to-back and sort both arrays and
// then also write the merged list of the same total length after
// them. We'll use it to sort pairs of columns while still having the
// original sorted columns available.
std::vector<Op> sort_pair(int w);

// Sort a 2d rectangle into scanline order. Equivalent to sort_2d but
// assumes each columns is already in sorted order. Uses addresses
// starting at scratch_idx for scratch space. The final range
// min_idx...max_idx is placed at position zero.
std::vector<Op> merge_cols_2d(int w, int h, int min_idx, int max_idx, int scratch_idx);

// Sort a 2d rectangle into scanline order. Equivalent to sort_2d but
// assumes each row is already in sorted order. Does not use scratch
// space. The final range from min_idx to max_idx is placed starting
// at position zero.
std::vector<Op> merge_rows_2d(int w, int h, int min_idx, int max_idx);

// In a sequence of ops, count the number of instructions due to each
// reason and print a table. Used to figure out which tasks are taking
// up all the ops.
void print_op_histogram(const std::vector<Op> &ops);

// Compute the high water mark (plus one) of locations touched during
// a sequence of ops, so an interpreter can allocation enough working
// space.
int compute_working_memory(const std::vector<Op> &ops);

// Computes a tile of medians of overlapping rectangles. The medians
// are over a w x h window, and the tile size is tw x th. Requires
// input in a particular layout. The core values shared by all outputs
// (the intersection of the footprints) is a sub-rectangle of size (w
// - tw + 1) x (h - th + 1). This comes first in scanline order, with
// the columns already sorted. Then we have (tw - 1)*2 extra central
// columns of height (h - th + 1), also sorted. The columns appear in
// order from left to right. Then we have the (th - 1)*2 extra central
// rows of width (w - tw + 1). These are unsorted. Finally we have the
// corner elements in scanline order. Given a 2x2 tiling of a 4x4
// window, the order of the elements expected in the input is thus:

/*
21 | 15 16 17 | 22
---+----------+---
 9 |  0  1  2 | 12
10 |  3  4  5 | 13
11 |  6  7  8 | 14
---+----------+---
23 | 18 19 20 | 24
*/

// The resulting tw x th medians are placed in scanline order starting
// at position (w + tw - 1) * (h + th - 1) (i.e. right after the all
// the input data)

// If the last arg is true, then instead of the cols being sorted
// within a core, the rows are sorted (and the shape of the core is
// transposed). In this case we use merge_rows_2d instead of
// merge_cols_2d for sorting the core.
std::vector<Op> tiled_median_2d(int w, int h, int tw, int th, bool col_major_core = false);

// Take the median of a rectangle.
inline std::vector<Op> median_2d(int w, int h) {
    return sort_2d(w, h, (w * h) / 2, (w * h) / 2);
}

// Limit the maximum length of the ops by decomposing merges and sorts
// into combinations of smaller primitives. Pass '2' to lower all the
// way to individual swaps.
std::vector<Op> limit_max_op_length(const std::vector<Op> &ops, int max_len);

// Run the sorting network over some ints for testing
void run(const int *src, int *dst, int W, int H, const std::vector<Op> &ops);

// A more compact instruction encoding for a subset of op
// types. Useful to drive an interpreter.
struct Instruction {
    int op, x1, x2;
};

// Generate all instructions within some size
// limit. limit_max_op_length with the same size limit will only
// generate ops encodable with these instructions.
std::vector<std::pair<Op, Instruction>> get_instruction_table(int max_len);

// Encode an instruction using a given instruction table
Instruction encode(Op op, const std::vector<std::pair<Op, Instruction>> &table);

// Decode an instruction using an instruction table, and the source
// stride to use for any strided copy instructions.
Op decode(Instruction inst, int stride, const std::vector<std::pair<Op, Instruction>> &table);

struct PerformanceCounters {
    int loads = 0, stores = 0, swaps = 0;
};
PerformanceCounters estimate_performance(const std::vector<Op> &ops);

#endif
