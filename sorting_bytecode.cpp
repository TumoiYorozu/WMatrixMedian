#include "sorting_bytecode.h"
#include "sorting_network.h"

#include <algorithm>
#include <assert.h>
#include <map>
#include <numeric>
#include <set>
#include <stdio.h>

using std::pair;
using std::set;
using std::vector;

struct Program {
    vector<Op> ops;

    void copy(const Seq &src, const Seq &dst, const char *reason) {
        assert(src.len == dst.len);
        ops.emplace_back();
        auto &op = ops.back();
        op.op = Copy;
        op.s1 = src;
        op.s2 = dst;
        op.min_idx = 0;
        op.max_idx = src.len - 1;
        op.reason = reason;
    }

    void sort_in_place(const Seq &s, const char *reason) {
        sort_in_place(s, 0, s.len - 1, reason);
    }

    void sort_in_place(const Seq &s, int min_idx, int max_idx, const char *reason) {
        assert(min_idx >= 0 && min_idx < s.len);
        assert(max_idx >= 0 && max_idx < s.len);
        assert(min_idx <= max_idx);
        ops.emplace_back();
        auto &op = ops.back();
        op.op = Sort;
        op.s1 = s;
        op.min_idx = min_idx;
        op.max_idx = max_idx;
        op.reason = reason;
    }

    void merge_in_place(const Seq &s1, const Seq &s2, const char *reason) {
        merge_in_place(s1, s2, 0, s1.len + s2.len - 1, reason);
    }

    void merge_in_place(const Seq &s1, const Seq &s2, int min_idx, int max_idx, const char *reason) {
        ops.emplace_back();
        auto &op = ops.back();
        op.op = Merge;
        op.s1 = s1;
        op.s2 = s2;
        op.min_idx = min_idx;
        op.max_idx = max_idx;
        op.reason = reason;
    }

    void print(const Seq &s, const char *reason) {
        ops.emplace_back();
        auto &op = ops.back();
        op.op = Print;
        op.s1 = s;
        op.min_idx = 0;
        op.max_idx = op.s1.len - 1;
        op.reason = reason;
    }

    void assert_sorted(const Seq &s, const char *reason) {
        assert_sorted(s, 0, s.len - 1, reason);
    }

    void assert_sorted(const Seq &s, int min_idx, int max_idx, const char *reason) {
        assert(min_idx >= 0 && min_idx < s.len);
        assert(max_idx >= 0 && max_idx < s.len);
        assert(min_idx <= max_idx);
        ops.emplace_back();
        auto &op = ops.back();
        op.op = AssertSorted;
        op.s1 = s;
        op.min_idx = min_idx;
        op.max_idx = max_idx;
        op.reason = reason;
    }
};

void Op::print() const {
    if (op == Sort) {
        printf("sort(reason = %s\n"
               "      min_idx = %d, max_idx = %d, x = %d, dx = %d, len = %d)\n",
               reason, min_idx, max_idx, s1.x, s1.dx, s1.len);
    } else if (op == AssertSorted) {
        printf("assert_sorted(reason = %s\n"
               "      min_idx = %d, max_idx = %d, x = %d, dx = %d, len = %d)\n",
               reason, min_idx, max_idx, s1.x, s1.dx, s1.len);
    } else if (op == Print) {
        printf("print(reason = %s\n"
               "      min_idx = %d, max_idx = %d, x = %d, dx = %d, len = %d)\n",
               reason, min_idx, max_idx, s1.x, s1.dx, s1.len);
    } else if (op == Merge) {
        printf("merge(reason = %s\n"
               "      min_idx = %d, max_idx = %d, \n"
               "      s1.x = %d, s1.dx = %d, s1.len = %d, \n"
               "      s2.x = %d, s2.dx = %d, s2.len = %d)\n",
               reason,
               min_idx, max_idx,
               s1.x, s1.dx, s1.len,
               s2.x, s2.dx, s2.len);
    } else if (op == Copy) {
        printf("copy(reason = %s\n"
               "      min_idx = %d, max_idx = %d, \n"
               "      s1.x = %d, s1.dx = %d, s1.len = %d, \n"
               "      s2.x = %d, s2.dx = %d, s2.len = %d)\n",
               reason,
               min_idx, max_idx,
               s1.x, s1.dx, s1.len,
               s2.x, s2.dx, s2.len);
    } else {
        assert(false && "Bad op");
    }
}

vector<Op> sort_2d_naive(int w, int h, int min_idx, int max_idx) {
    // Just sort each row, then merge the rows using a Batcher
    // even-odd merge sort across y, where each link is interpreted as
    // a batcher merge of two rows.
    Program result;

    for (int y = 0; y < h; y++) {
        result.sort_in_place(Seq{y * w, 1, w}, "Sort row");
    }

    for (const auto &p : pairwise_sort(h)) {
        result.merge_in_place(Seq{p.first * w, 1, w},
                              Seq{p.second * w, 1, w},
                              "Merge rows");
    }

    return result.ops;
}

vector<Op> sort_1d(int w, int min_idx, int max_idx) {
    Program result;
    result.sort_in_place(Seq{0, 1, w}, min_idx, max_idx, "Sort");
    return result.ops;
}

vector<Op> sort_pair(int w) {
    Program result;
    Seq a{0, 1, w}, b{w, 1, w}, c{2 * w, 1, w}, d{3 * w, 1, w};
    result.sort_in_place(a, "sort_pair: Sort first half");
    result.copy(a, c, "sort_pair: Copying first half");
    result.sort_in_place(b, "sort_pair: Sort second half");
    result.copy(b, d, "sort_pair: Copying second half");
    result.merge_in_place(c, d, "sort_pair: Merge sorted halves");
    return result.ops;
}

vector<Op> tiled_median_2d(int w, int h, int tw, int th, bool col_major_core) {
    // The size of the intersection of the supports. We call this the
    // "core".
    int core_w = w - tw + 1;
    int core_h = h - th + 1;

    // Each output is going to use some number of values from outside
    // the core unique(ish) to that output.
    int values_inside_core = core_w * core_h;
    int values_outside_core = w * h - values_inside_core;

    // To compute the median, we'll sort the values outside the core
    // unique to this output, then sort the values inside the core
    // (this work is shared). Then we merge those two lists and return
    // the median. The final merge is between values_outside_core and
    // values_inside_core.

    // When taking a median of two sorted lists of size n and m, where
    // n < m, you only need the middle-most n + 1 values from the list
    // of size m. Everything outside of that is too small or too large
    // to possibly be the median of the sorted combination.

    // For example: If we have n == 5 and m == 20, then the combined
    // list is of size 25, so the median of the combined list is
    // greater than 12 things, less than 12 things, and equal to one
    // thing. Anything in the list of size m that's already greater
    // than 13 things can be excluded from consideration. This lops
    // off the last 7 elements (indices 13, 14, 15, 16, 17, 18, 19). A
    // similar argument lops off the first 7 elements, leaving us with
    // 6 elements remaining (7, 8, 9, 10, 11, 12).

    int min_core_idx = 0;
    int max_core_idx = core_w * core_h - 1;
    while (max_core_idx - min_core_idx > values_outside_core) {
        min_core_idx++;
        max_core_idx--;
    }

    max_core_idx = std::min(max_core_idx, values_inside_core - 1);
    min_core_idx = std::max(min_core_idx, 0);

    assert(min_core_idx <= max_core_idx);

    int total_w = w + (tw - 1);
    int total_h = h + (th - 1);

    int output_start = total_w * total_h;

    // First do the shared work of sorting the values inside the
    // core.

    // In theory if there are more values outside the core than
    // inside, we should just do a conventional mergesort. However, in
    // that case the total amount of work to do inside the core is
    // negligible compared to total runtime, so it doesn't matter.
    Program result;
    if (!col_major_core) {
        //printf("Calling merge_cols_2d(%d, %d, %d, %d, %d)\n",
        //core_w, core_h, min_core_idx, max_core_idx, output_start);
        result.ops = merge_cols_2d(core_w, core_h, min_core_idx, max_core_idx, output_start);
    } else {
        //printf("Calling merge_rows_2d(%d, %d, %d, %d)\n",
        //core_h, core_w, min_core_idx, max_core_idx);
        result.ops = merge_rows_2d(core_h, core_w, min_core_idx, max_core_idx);
    }

    // int peak_while_sorting_core = compute_working_memory(result.ops);
    // printf("Peak memory while sorting core: %d\n", peak_while_sorting_core);

    Seq sorted_core;
    sorted_core.x = 0;
    sorted_core.dx = 1;
    sorted_core.len = max_core_idx - min_core_idx + 1;

    result.print(sorted_core, "Sorted core");

    if (tw == 1 && th == 1) {
        // Untiled.

        // Copy the median to the expected output location
        result.copy(Seq{sorted_core.x, 1, 1},
                    Seq{output_start, 1, 1},
                    "Copying the result to the expected location");
        return result.ops;
    }

    // Unpack the rest of the memory layout (see the comment in the header file)
    int extra_cols_start = core_w * core_h;
    int num_extra_cols = 2 * (tw - 1);
    int extra_rows_start = extra_cols_start + core_h * num_extra_cols;
    int num_extra_rows = 2 * (th - 1);
    int corners_start = extra_rows_start + core_w * num_extra_rows;

    assert(corners_start + num_extra_cols * num_extra_rows == output_start);

    // Sort all the extra rows in-place
    for (int r = 0; r < num_extra_rows; r++) {
        Seq row;
        row.x = extra_rows_start + r * core_w;
        row.dx = 1;
        row.len = core_w;
        result.sort_in_place(row, "Sorting extra row");
    }

    // TODO: We could find the optimal order in which to merge things
    // using a directed Steiner tree algorithm.

    int row_groups_start = output_start + tw * th;
    int row_group_size = (th - 1) * core_w;
    int num_row_groups = th;

    auto merge_rows_net = pairwise_sort(th - 1);
    auto merge_cols_net = pairwise_sort(tw - 1);
    // Gather and merge each window of th - 1 consecutive sorted rows
    if (th > 1) {
        for (int y = 0; y < th; y++) {
            // TODO: the windows overlap, so there's shared work here we
            // can still exploit (e.g. consider aligned pairs starting at
            // idx 1).

            // This is an im2col-like operation where we just gather
            // the values under the stencil footprint.

            Seq row_group{row_groups_start + y * row_group_size, 1, row_group_size};
            result.copy(Seq{extra_rows_start + y * core_w, 1, row_group_size},
                        row_group,
                        "Copy group of extra rows");

            // Now use a sorting network of size th - 1 to drive a series
            // of merge operations to sort the row group.
            for (const auto &p : merge_rows_net) {
                int x1 = row_groups_start + y * row_group_size + p.first * core_w;
                int x2 = row_groups_start + y * row_group_size + p.second * core_w;
                result.merge_in_place(Seq{x1, 1, core_w},
                                      Seq{x2, 1, core_w},
                                      "Merge extra row group");
            }

            // Assert sorted. Not just for debugging - it also lets the op limiter knows this is now sorted
            result.assert_sorted(row_group, "Assert row group sorted");
            result.print(row_group, "Sorted row group");

            if (tw == 1) {
                // merge the in sorted core and finish
                Seq sorted_core_copy{row_group.x + row_group.len, 1, sorted_core.len};
                result.copy(sorted_core, sorted_core_copy,
                            "Copy sorted core");

                int median_idx = (row_group.len + sorted_core.len) / 2;
                result.merge_in_place(row_group, sorted_core_copy,
                                      median_idx, median_idx,
                                      "Merge row group with sorted core");

                row_group.len += sorted_core.len;
                result.print(row_group, "Median taken with sorted core");

                // Copy the median back to the expected output location
                result.copy(Seq{row_group.x + median_idx, 1, 1},
                            Seq{output_start + y, 1, 1},
                            "Copying the result to the expected location");
            }
        }
    }

    if (tw == 1) {
        return result.ops;
    }

    int col_groups_start = row_groups_start + num_row_groups * row_group_size;

    int col_group_size = (tw - 1) * core_h;

    /*
    printf("sorted core ends at %d\n"
           "extra_cols_start: %d\n"
           "extra_rows_start: %d\n"
           "corners_start: %d\n"
           "row_groups_start: %d\n"
           "col_groups_start %d\n",
           sorted_core.len,
           extra_cols_start,
           extra_rows_start,
           corners_start,
           row_groups_start,
           col_groups_start);
    */

    result.print(Seq{row_groups_start, 1, num_row_groups * row_group_size},
                 "All row groups before sorting col groups");

    col_group_size += sorted_core.len;

    // When taking a median by merging a bunch of sorted lists, at
    // each merge step you can ignore everything except for the
    // middle-most n + 1, where n is all the number of elements
    // still left to handle after this merge.

    // The elements not yet merged here are the missing rows/corners
    int elements_remaining_after_merge = w * (h - core_h);
    int elements_to_discard_after_this_merge = col_group_size - (elements_remaining_after_merge + 1);
    int col_merge_min_idx = elements_to_discard_after_this_merge / 2;
    int col_merge_max_idx = col_group_size - 1 - col_merge_min_idx;

    int scratch_start = col_groups_start + col_merge_max_idx + 1;

    //printf("Scratch start: %d\n", scratch_start);

    // Then we enter the loop over the tile
    for (int x = 0; x < tw; x++) {
        // Merge the sorted core with the appropriate additional columns for this x coord within the tile
        {
            // This is an im2col-like operation where we just gather
            // the values under the stencil footprint.
            int cols_only_size = (tw - 1) * core_h;
            Seq col_group{col_groups_start, 1, cols_only_size};
            result.copy(Seq{extra_cols_start + x * core_h, 1, cols_only_size},
                        col_group,
                        "Copy group of extra cols");

            // Now use a sorting network of size th - 1 to drive a series
            // of merge operations to sort the col group.
            for (const auto &p : merge_cols_net) {
                int x1 = col_groups_start + p.first * core_h;
                int x2 = col_groups_start + p.second * core_h;
                result.merge_in_place(Seq{x1, 1, core_h},
                                      Seq{x2, 1, core_h},
                                      "Merge extra col group");
            }

            // Merge in the sorted core too
            Seq sorted_core_dst{col_groups_start + (tw - 1) * core_h, 1, sorted_core.len};
            result.copy(sorted_core, sorted_core_dst,
                        "Copying sorted core onto the end of the sorted col group");

            result.merge_in_place(col_group, sorted_core_dst,
                                  col_merge_min_idx, col_merge_max_idx,
                                  "Merging col group with sorted core");

            assert(col_group.x + col_group.len == sorted_core_dst.x);

            col_group.len += sorted_core.len;

            if (th == 1) {
                // We're actually done with this x
                assert(col_merge_min_idx == col_merge_max_idx);
                // Copy the median back to the expected output location
                result.copy(Seq{col_group.x + col_merge_min_idx, 1, 1},
                            Seq{output_start + x, 1, 1},
                            "Copying the result to the expected location");
            } else {
                // Make sure the op limiter knows this is now sorted
                result.assert_sorted(col_group, col_merge_min_idx, col_merge_max_idx,
                                     "Assert col group sorted");
                result.print(col_group, "Sorted col group");
            }
        }

        if (tw == 1) {
            continue;
        }

        for (int y = 0; y < th; y++) {
            Seq extra_cols, extra_rows;
            vector<Seq> corners;

            extra_rows.x = row_groups_start + y * row_group_size;
            extra_rows.dx = 1;
            extra_rows.len = row_group_size;

            // Grab the sorted columns we need for this tile element
            // (TODO: there is shared work here to exploit, at the
            // cost of more memory).
            extra_cols.x = col_groups_start + col_merge_min_idx;
            extra_cols.dx = 1;
            extra_cols.len = col_merge_max_idx - col_merge_min_idx + 1;

            for (int r = y; r < y + th - 1; r++) {
                // Also grab the corner sequences. They form a little
                // square num_extra_cols wide and num_extra_rows
                // high. We want a (tw - 1) x (th - 1) crop from inside of it
                // starting at position x, y.
                Seq corner{corners_start + r * num_extra_cols + x, 1, tw - 1};
                corners.push_back(corner);
                result.print(corner, "Row of corners");
            }

            // We can't mutate these things in-place, because
            // different pixels in the tile need different subsets of
            // them. The scratch space is currently entirely free
            // though, so we can start by copying stuff over there.

            // TODO: for y == th - 1 we can clobber the
            // extra_rows. For x == tw - 1 we can clobber the extra
            // cols.

            vector<Seq *> to_copy;

            if (y < th - 1) {
                // For the final value of y we can just clobber the
                // extra_cols in place, as this is their last use.
                to_copy.push_back(&extra_cols);
            }
            to_copy.push_back(&extra_rows);
            for (auto &c : corners) {
                to_copy.push_back(&c);
            }
            int idx = scratch_start;
            for (Seq *s : to_copy) {
                Seq dst{idx, 1, s->len};
                result.copy(*s, dst,
                            "Assembling pieces unique to this output in scratch space");
                *s = dst;
                idx += s->len;
            }

            {
                // Consider a graph where the nodes are all the
                // different intersections of subsets of the
                // input. There are edges that go from an
                // intersection of many things to an intersection
                // of a subset of them. These edges can be
                // traversed by sorting in the newly-included
                // elements. Ultimately we want to reach the nodes
                // with one thing in the intersection. We start at
                // the intersection of everything. We want a
                // spanning tree rooted at the intersection of
                // everything that hits all our leaves. This is
                // the "Directed Steiner Tree Problem". It's
                // NP-hard. We can probably take the minimum
                // spanning tree and delete the less useful nodes,
                // and repeat? Or we could iterate over all
                // possible spanning trees? Or maybe a structure
                // will jump out at us.

                // Sort the corners
                Seq all_corners = corners[0];
                all_corners.len = idx - all_corners.x;

                assert(all_corners.len == (tw - 1) * (th - 1));

                if (all_corners.len > 1) {
                    result.sort_in_place(all_corners, "Sorting the corner pieces unique to this output");
                    result.print(all_corners, "Sorted corners");
                }

                // We now have three groups of things to take the
                // median of. Surprisingly, it's best to merge in
                // the smallest thing (for us, probably the
                // corners) last.

                // First ensure no one thing is larger than the
                // other two combined. If that's the case it
                // should have been pruned down when it was created.

                /*
                printf("extra_rows.x = %d, extra_cols.x = %d, all_corners.x = %d\n",
                       extra_rows.x, extra_cols.x, all_corners.x);
                */

                assert(all_corners.len <= extra_rows.len + extra_cols.len + 1);
                assert(extra_rows.len <= all_corners.len + extra_cols.len + 1);
                assert(extra_cols.len <= all_corners.len + extra_rows.len + 1);

                // Merge rows with cols
                int rows_and_cols_min_idx = 0;
                int rows_and_cols_max_idx = extra_rows.len + extra_cols.len - 1;
                // Prune it down to the middle all_corners.len + 1 values
                while (rows_and_cols_max_idx - rows_and_cols_min_idx + 1 > all_corners.len + 1) {
                    rows_and_cols_min_idx++;
                    rows_and_cols_max_idx--;
                }

                result.merge_in_place(extra_cols, extra_rows,
                                      rows_and_cols_min_idx,
                                      rows_and_cols_max_idx,
                                      "Merging the sorted rows into the sorted cols and core");

                // Grow the extra_cols seq to include the merged-in rows
                assert(extra_cols.x + extra_cols.len == extra_rows.x);
                extra_cols.x += rows_and_cols_min_idx;
                extra_cols.len = all_corners.len + 1;

                // Now take the median with the corners, if there are any.
                result.merge_in_place(extra_cols, all_corners,
                                      all_corners.len, all_corners.len,
                                      "Merging the sorted corners into the sorted extra rows");

                // Copy the median back to the expected output location
                result.copy(Seq{extra_cols.x + all_corners.len, 1, 1},
                            Seq{output_start + y * tw + x, 1, 1},
                            "Copying the result to the expected location");
            }
        }
    }

    return result.ops;
}

vector<Op> merge_rows_2d(int w, int h, int min_idx, int max_idx) {
    Program result;

    // Which rows of the output do we want?
    int min_row_idx = min_idx / w;
    int max_row_idx = max_idx / w;

    auto merge_network = pairwise_sort(h, min_row_idx, max_row_idx, false);
    if (w % 2 == 0) {
        int hw = w / 2;
        assert(merge_network.size() <= pairwise_sort(2 * h, min_idx / hw, max_idx / hw, true).size() &&
               "Should have used a pairwise network on half-rows");
    }

    for (auto p : merge_network) {
        result.merge_in_place(Seq{p.first * w, 1, w},
                              Seq{p.second * w, 1, w},
                              "Merge rows");
    }
    if (min_idx != 0) {
        result.copy(Seq{min_idx, 1, max_idx - min_idx + 1},
                    Seq{0, 1, max_idx - min_idx + 1},
                    "Copying merge_rows_2d output to expected location");
    }
    return result.ops;
}

vector<Op> merge_cols_2d(int w, int h, int min_idx, int max_idx, int scratch_start) {
    assert(min_idx <= max_idx);
    Program result;

    // After sorting horizontally, a bunch of values are no longer
    // going to be in min_idx / max_idx and so may as well be negative
    // or positive infinity. We'll maintain a mask that specifies the
    // state of each site.
    enum State {
        Valid = 0,
        NegInf = 1,
        PosInf = 2,
    };

    // We'll also compute two thresholds for detecting when a value
    // falls into one of these states.

    // In the final list, the value stored at max_idx is greater than
    // or equal to max_idx + 1 total values in the list (including
    // itself). Therefore, if something is greater than or equal to
    // more than that many total values in the input, it may as well
    // be positive infinity, because it's going to land in the final
    // last after max_idx.

    // If something is >= more than this many values, it may as well
    // be positive infinity.
    int max_threshold = max_idx + 1;

    // If a value is <= more than this many other values, it may as
    // well be negative infinity.
    int min_threshold = w * h - min_idx;

    // Initially everything is valid (we assume the vertical sort
    // alone isn't enough to render anything +/- infinity).
    vector<State> state(w * h, Valid);

    // Sort the rows. Note that after doing this, the square is still
    // sorted along the columns!
    for (int y = 0; y < h; y++) {
        int num_pos_inf = 0, num_neg_inf = 0;
        for (int x = 0; x < w; x++) {
            // After sorting, how many values in the square is this
            // value definitely greater than or equal to (including itself).

            // Answer: Everything above and to the left.
            int gt = (x + 1) * (y + 1);
            if (gt > max_threshold) {
                state[x + y * w] = PosInf;
                num_pos_inf++;
            }

            // We're less than everything below and to the right.
            int lt = (w - x) * (h - y);
            if (lt > min_threshold) {
                state[x + y * w] = NegInf;
                num_neg_inf++;
            }
        }
        result.sort_in_place(Seq{y * w, 1, w},
                             num_neg_inf,
                             w - 1 - num_pos_inf,
                             "Sort along rows");
    }

    if (0) {
        printf("State after horizontal sort:\n");
        int valids = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                switch (state[y * w + x]) {
                case Valid:
                    valids++;
                    printf(".");
                    break;
                case PosInf:
                    printf("^");
                    break;
                case NegInf:
                    printf("_");
                    break;
                }
            }
            printf("\n");
        }
        printf("%d valid sites\n", valids);
    }

    // Sort diagonally up and to the right. It will still be sorted
    // horizontally and vertically after doing this, though the proof
    // only works if the square is taller than it is wide.

    bool transposed = h < w;
    int x_stride = 1, y_stride = w;
    if (transposed) {
        std::swap(w, h);
        std::swap(x_stride, y_stride);
    }

    struct Diagonal {
        // Current location in memory
        Seq seq;
        // Whether or not it's currently sorted
        bool sorted = false;
        // The 2d coordinates it represents
        vector<pair<int, int>> sites;
    };

    vector<Diagonal> diagonals;
    vector<State> state_after = state;
    int scratch_cursor = scratch_start;
    int num_diagonals = 0, num_sorted_diagonals = 0;
    int neg_infs_in_final_sort = 0, pos_infs_in_final_sort = 0;

    // Figure out the order in which to visit the x coordinates
    vector<int> x_coords;
    for (int x = -h + 1; x < w; x++) {
        x_coords.emplace_back(x);
    }
    std::sort(x_coords.begin(), x_coords.end(), [=](int x1, int x2) {
        int y1_start = std::min(x1, 0) + h - 1;
        int y1_end = std::max(x1 + h - w, 0);
        int len1 = y1_start - y1_end;
        int y2_start = std::min(x2, 0) + h - 1;
        int y2_end = std::max(x2 + h - w, 0);
        int len2 = y2_start - y2_end;
        return (len1 > len2);
    });

    for (int x_start : x_coords) {
        //for (int x_start = -h + 1; x_start < w; x_start++) {
        int y_start = std::min(x_start, 0) + h - 1;
        int y_end = std::max(x_start + h - w, 0);
        assert(y_end <= y_start);
        int idx = 0;
        int min_input_idx = std::max(w, h);
        int max_input_idx = -1;
        int min_output_idx = std::max(w, h);
        int max_output_idx = -1;
        for (int y = y_start; y >= y_end; y--) {
            int x = x_start + (h - 1 - y);
            assert(x >= 0 && x < w && y >= 0 && y < h);

            // Track the min and max valid index along this diagonal
            // to see what we'll need to sort.
            if (state[x * x_stride + y * y_stride] == Valid) {
                min_input_idx = std::min(min_input_idx, idx);
                max_input_idx = std::max(max_input_idx, idx);
            }

            // After sorting, this value may be out-of-range. It's
            // less than everything in a wide fan to the right, and
            // greater than everything in a wide fan to the left. For
            // for the center value in a 7x7 grid the relations look
            // like:

            // >>>>..<
            // >>>>.<<
            // >>>><<<
            // >>>=<<<
            // >>><<<<
            // >>.<<<<
            // >..<<<<
            int ge = 0, le = 0;
            for (int y2 = 0; y2 < h; y2++) {
                for (int x2 = 0; x2 < w; x2++) {
                    if (x <= x2 && x + y <= x2 + y2) {
                        le++;
                    }
                    if (x2 <= x && x2 + y2 <= x + y) {
                        ge++;
                    }
                    // Not an else-if because there's a single value where both are true (x == x2, y == y2)
                }
            }

            if (le > min_threshold) {
                state_after[x * x_stride + y * y_stride] = NegInf;
            } else if (ge > max_threshold) {
                state_after[x * x_stride + y * y_stride] = PosInf;
            } else {
                min_output_idx = std::min(min_output_idx, idx);
                max_output_idx = std::max(max_output_idx, idx);
            }

            idx++;
        }

        if (min_input_idx <= max_input_idx) {
            int x = x_start + (h - 1 - y_start);
            diagonals.emplace_back();
            auto &diag = diagonals.back();
            diag.seq.x = (x + min_input_idx) * x_stride + (y_start - min_input_idx) * y_stride;
            diag.seq.dx = x_stride - y_stride;
            diag.seq.len = max_input_idx - min_input_idx + 1;

            for (int i = min_input_idx; i <= max_input_idx; i++) {
                diag.sites.emplace_back(x + i, y_start - i);
            }

            // Output indices are relative to the starting input index
            min_output_idx -= min_input_idx;
            max_output_idx -= min_input_idx;

            // Because of the hyperbolic slice we've cut out of the
            // valid region, some diagonals have a segment of pos_inf
            // or neg_inf in the middle that we can excise before
            // copying.
            int neg_inf_start = diag.seq.len;
            int neg_inf_end = -1;
            int pos_inf_start = diag.seq.len;
            int pos_inf_end = -1;
            assert(state[diag.seq.x] == Valid);
            assert(state[diag.seq.x + (diag.seq.len - 1) * diag.seq.dx] == Valid);
            for (int i = 0; i < diag.seq.len; i++) {
                auto v = state[diag.seq.x + i * diag.seq.dx];
                if (v == NegInf) {
                    neg_inf_start = std::min(neg_inf_start, i);
                    neg_inf_end = std::max(neg_inf_end, i);
                }
                if (v == PosInf) {
                    pos_inf_start = std::min(pos_inf_start, i);
                    pos_inf_end = std::max(pos_inf_end, i);
                }
            }

            // Copy the diagonal into scratch space
            if (pos_inf_start > pos_inf_end && neg_inf_start > neg_inf_end) {
                // One long diagonal
                Seq diag_dst{scratch_cursor, 1, diag.seq.len};
                result.copy(diag.seq, diag_dst, "Copying diagonal to scratch space");
                diag.seq = diag_dst;

                num_diagonals++;
            } else if (pos_inf_start <= pos_inf_end ||
                       neg_inf_start <= neg_inf_end) {
                int inf_start, inf_end;
                if (pos_inf_start <= pos_inf_end) {
                    inf_start = pos_inf_start;
                    inf_end = pos_inf_end;
                } else {
                    inf_start = neg_inf_start;
                    inf_end = neg_inf_end;
                    // If we only wanted some region of this diagonal,
                    // we should adjust it according to the negative
                    // infinities we just discarded on the left.
                    int num_neg_inf = neg_inf_end - neg_inf_start + 1;
                    min_output_idx -= num_neg_inf;
                    max_output_idx -= num_neg_inf;
                }
                // There's a blob of positive or negative infinities
                // in the middle. Copy the regions around it.
                Seq diag1_src = diag.seq;
                Seq diag2_src = diag.seq;
                diag1_src.len = inf_start;

                diag2_src.x += (inf_end + 1) * diag2_src.dx;
                diag2_src.len = diag.seq.len - inf_end - 1;

                Seq diag1_dst{scratch_cursor, 1, diag1_src.len};
                Seq diag2_dst{scratch_cursor + inf_start, 1, diag2_src.len};

                result.copy(diag1_src, diag1_dst, "Copying start of diagonal to scratch space");
                result.copy(diag2_src, diag2_dst, "Copying end of diagonal to scratch space");

                // Update diag to refer to its new location
                diag.seq = diag1_dst;
                diag.seq.len = diag1_dst.len + diag2_dst.len;

                vector<pair<int, int>> new_coords;
                for (int i = 0; i < diag1_dst.len; i++) {
                    new_coords.push_back(diag.sites[i]);
                }
                for (int i = 0; i < diag2_dst.len; i++) {
                    new_coords.push_back(diag.sites[inf_end + 1 + i]);
                }
                diag.sites.swap(new_coords);

                num_diagonals++;
            }

            // 2) Sort it in-place. There's no point doing this if we
            // don't get pruning from it, because we're just going to
            // re-sort the whole thing anyway.
            int pruned_len = max_output_idx - min_output_idx + 1;
            if (pruned_len < diag.seq.len) {
                result.sort_in_place(diag.seq, min_output_idx, max_output_idx, "Sorting diagonal");

                num_sorted_diagonals++;

                // Copy the unpruned region leftwards if necessary
                diag.seq.len = pruned_len;

                if (min_output_idx) {
                    result.copy(Seq{diag.seq.x + min_output_idx, 1, pruned_len},
                                diag.seq,
                                "Compacting sorted diagonal");
                }

                diag.sorted = true;

            } else {
                // We may have willingly included things that are
                // known to not be the median in order to do fewer
                // sorting ops total.
                neg_infs_in_final_sort += min_output_idx;
                pos_infs_in_final_sort += diag.seq.len - 1 - max_output_idx;

                diag.sorted = false;
            }
            scratch_cursor += diag.seq.len;
        }
    }

    if (0) {
        printf("State after diagonal sort:\n");
        int valids = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                switch (state_after[y * w + x]) {
                case Valid:
                    valids++;
                    printf(".");
                    break;
                case PosInf:
                    printf("^");
                    break;
                case NegInf:
                    printf("_");
                    break;
                }
            }
            printf("\n");
        }
        printf("%d valid sites\n", valids);
    }

    // Count how many Infs vs Valids we ended up with, and assert that
    // every value is accounted for
    int num_pos_inf = 0, num_neg_inf = 0, num_valid = 0;
    for (auto s : state_after) {
        if (s == PosInf) {
            num_pos_inf++;
        } else if (s == NegInf) {
            num_neg_inf++;
        } else if (s == Valid) {
            num_valid++;
        } else {
            assert(false && "bad state");
        }
    }

    assert(scratch_cursor - scratch_start == neg_infs_in_final_sort + pos_infs_in_final_sort + num_valid);

    if ((num_diagonals > 1 ||                  // There are multiple diagonals to merge
         num_sorted_diagonals == 0) &&         // There's only one diagonal, but it's not sorted
        scratch_cursor > scratch_start + 1) {  // We have more than one value to worry about

        for (auto &d : diagonals) {
            if (d.sorted) {
                // Make sure the limiter knows this span is already sorted
                result.assert_sorted(d.seq, "Assert diagonal sub-component is already sorted");

                // TODO: This final sort is somewhat wasteful because
                // there are already large sorted sequences within
                // it. Try a merge sort instead?
            }
        }

        int final_min_idx = min_idx - num_neg_inf + neg_infs_in_final_sort;
        int final_max_idx = max_idx - num_neg_inf + neg_infs_in_final_sort;
        // Ignore the fact that (some of) the diagonals are already
        // sorted, and just do a full sort of their concatenation

        Seq merged{scratch_start, 1, scratch_cursor - scratch_start};
        result.sort_in_place(merged,
                             final_min_idx,
                             final_max_idx,
                             "Final sort of concatenated diagonals");

        /*
        // First sort every contiguous span of unsorted ones
        Seq pending{0, 1, 0};
        vector<Seq> sorted_diagonals;
        for (size_t i = 0; i < diagonals.size(); i++) {
            if (diagonals[i].sorted && diagonals[i].seq.len > 8) {
                if (pending.len) {
                    // TODO: min_idx / max_idx
                    result.sort_in_place(pending, "Sorting section of diagonals");
                    sorted_diagonals.push_back(pending);
                    pending.len = 0;
                }
                sorted_diagonals.push_back(diagonals[i].seq);
            } else {
                if (!pending.len) {
                    pending.x = diagonals[i].seq.x;
                }
                pending.len += diagonals[i].seq.len;
            }
        }
        if (pending.len) {
            result.sort_in_place(pending, "Sorting section of diagonals");
            sorted_diagonals.push_back(pending);
            pending.len = 0;
        }

        // Then iteratively merge the smallest pair
        while (sorted_diagonals.size() > 1) {
            if (sorted_diagonals.size() == 2) {
                result.merge_in_place(sorted_diagonals[0],
                                      sorted_diagonals[1],
                                      final_min_idx, final_max_idx,
                                      "Merging two sorted diagonals");
                break;
            } else {
                int smallest_pair = 0;
                int smallest_pair_len = sorted_diagonals[0].len + sorted_diagonals[1].len;
                for (int i = 1; i < (int)sorted_diagonals.size() - 1; i++) {
                    int len = sorted_diagonals[i].len + sorted_diagonals[i + 1].len;
                    if (len < smallest_pair_len) {
                        smallest_pair = i;
                        smallest_pair_len = len;
                    }
                }

                result.merge_in_place(sorted_diagonals[smallest_pair],
                                      sorted_diagonals[smallest_pair + 1],
                                      "Merging two sorted diagonals");
                sorted_diagonals[smallest_pair].len = smallest_pair_len;
                sorted_diagonals.erase(sorted_diagonals.begin() + smallest_pair + 1);
            }
        }
        */
        // HACK: for benchmarking things
        //result.ops.pop_back();
    }

    // Now copy the result back into the expected place
    Seq final_dst{0, 1, max_idx - min_idx + 1};

    Seq final_src = final_dst;
    final_src.x = scratch_start + min_idx - num_neg_inf;

    result.copy(final_src, final_dst, "Copying core to output location");

    return result.ops;
}

int compute_working_memory(const vector<Op> &ops) {
    int max = 0;
    auto include_seq = [&](const Seq &seq) {
        if (seq.len) {
            int end = seq.x + seq.dx * (seq.len - 1);
            max = std::max(std::max(seq.x, end), max);
        }
    };

    for (const auto &op : ops) {
        include_seq(op.s1);
        include_seq(op.s2);
    }
    return max + 1;
}

vector<Op> limit_max_op_length(const vector<Op> &ops, int max_len) {
    // Keep track of which subsequences are sorted, to skip redundant
    // sorting ops. The way this works is to just assign a unique
    // integer per memory location. If two sites have the same
    // integer, then they are in sorted order. The order is increasing
    // when the integer is positive, decreasing when the integer
    // is negative, and unsorted when the integer is zero.
    int next_id = 1;
    vector<int> already_sorted(compute_working_memory(ops), 0);

    auto mark = [&](int x, int id) {
        assert(x >= 0 && x < already_sorted.size());
        already_sorted[x] = id;
    };

    auto is_already_sorted = [&](const Seq &seq, int dir) {
        int id = already_sorted[seq.x];
        if (id == 0) {
            return false;
        }
        if ((id < 0) != (dir < 0)) {
            return false;
        }
        for (int i = 1; i < seq.len; i++) {
            if (already_sorted[seq.x + i * seq.dx] != id) {
                return false;
            }
        }
        return true;
    };

    auto is_already_merged = [&](const Seq &s1, const Seq &s2, int dir) {
        int id = already_sorted[s1.x];
        if (id == 0) {
            return false;
        }
        if ((id < 0) != (dir < 0)) {
            return false;
        }
        for (int i = 1; i < s1.len; i++) {
            if (already_sorted[s1.x + i * s1.dx] != id) {
                return false;
            }
        }
        for (int i = 0; i < s2.len; i++) {
            if (already_sorted[s2.x + i * s2.dx] != id) {
                return false;
            }
        }
        return true;
    };

    auto update_already_sorted = [&](const Op &op) {
        if (op.op == Sort || op.op == AssertSorted) {
            int id = next_id++;
            if (op.s1.dx < 0) {
                id = -id;
            }
            for (int i = 0; i < op.s1.len; i++) {
                int x = op.s1.x + op.s1.dx * i;
                if (i < op.min_idx || i > op.max_idx) {
                    mark(x, 0);
                } else {
                    mark(x, id);
                }
            }
        } else if (op.op == Merge) {
            int id = next_id++;
            if (op.s1.dx < 0) {
                id = -id;
            }
            assert(((op.s1.dx < 0) == (op.s2.dx < 0)) && "mixed sign case unimplemented");
            for (int i = 0; i < op.s1.len + op.s2.len; i++) {
                int x1 = op.s1.x + op.s1.dx * i;
                int x2 = op.s2.x + op.s2.dx * (i - op.s1.len);
                int x = i < op.s1.len ? x1 : x2;
                if (i < op.min_idx || i > op.max_idx) {
                    mark(x, 0);
                } else {
                    mark(x, id);
                }
            }
        } else if (op.op == Copy) {
            int src_id = already_sorted[op.s1.x];
            bool src_is_sorted = src_id != 0;
            for (int i = 1; i < op.s1.len; i++) {
                int x = op.s1.x + op.s1.dx * i;
                src_is_sorted &= (already_sorted[x] == src_id);
            }
            if (src_is_sorted) {
                int id = next_id++;
                if (src_id < 0) {
                    id = -id;
                }
                for (int i = 0; i < op.s2.len; i++) {
                    int x = op.s2.x + op.s2.dx * i;
                    mark(x, id);
                }
            } else {
                for (int i = 0; i < op.s2.len; i++) {
                    int x = op.s2.x + op.s2.dx * i;
                    mark(x, 0);
                }
            }
        }
    };

    vector<Op> result;
    for (const auto &op : ops) {
        auto already_sorted_before = already_sorted;

        if ((op.op == Sort && op.s1.len <= max_len) ||
            (op.op == Merge && op.s1.len <= max_len / 2 && op.s2.len <= max_len / 2) ||
            (op.op == Copy && op.s1.len <= max_len) ||
            op.op == AssertSorted ||
            op.op == Print) {
            result.push_back(op);
        } else if (op.op == Sort) {
            // Break into subsequences of at most leaf_size, and sort
            // each pair of subsequences.
            int leaf_size = max_len / 2;
            int num_leaves = (op.s1.len + leaf_size - 1) / leaf_size;
            leaf_size = (op.s1.len + num_leaves - 1) / num_leaves;

            vector<Seq> leaves;

            // Start by sorting each pair of leaves
            for (int i = 0; i < num_leaves; i += 2) {
                int start = i * leaf_size;
                int end = std::min(op.s1.len, start + 2 * leaf_size);
                int len = end - start;

                leaves.emplace_back(op.s1);
                auto &leaf = leaves.back();
                leaf.x += leaf.dx * start;
                leaf.len = len;

                if (len > 1) {
                    result.emplace_back(op);
                    auto &sort_leaf = result.back();

                    sort_leaf.s1 = leaf;

                    // TODO: Can get a little tighter here, but by
                    // default assume we need everything. Right now
                    // leaf sorts use the same generated code
                    // regardless of min_idx/max_idx to keep the
                    // interpreter small, so it doesn't matter.
                    sort_leaf.min_idx = 0;
                    sort_leaf.max_idx = len - 1;

                    if (is_already_sorted(leaf, leaf.dx)) {
                        result.pop_back();
                    } else {
                        update_already_sorted(result.back());
                    }
                }

                // If this was a pair of leaves, push the other member
                // of the pair on the back for the merge.
                if (leaves.back().len > leaf_size) {
                    leaves.back().len = leaf_size;
                    leaves.push_back(leaves.back());
                    auto &leaf_pair = leaves.back();
                    leaf_pair.x += leaf_pair.dx * leaf_size;
                    leaf_pair.len = len - leaf_size;
                }
            }

            // TODO: By rounding down we're implicitly padding the
            // last leaf with pos_inf. For a given min_idx vs max_idx
            // it might make it better to pad with some number of
            // neg_infs as well or instead, but then we need to
            // compile more instructions.

            vector<pair<int, int>> swaps =
                pairwise_sort(num_leaves,
                              op.min_idx / leaf_size,  // Which leaf does min_idx fall into?
                              op.max_idx / leaf_size,  // Which leaf does max_idx fall into?
                              true);                   // Already sorted in pairs

            for (const auto &p : swaps) {
                result.emplace_back();
                auto &merge = result.back();
                merge.reason = op.reason;
                merge.op = Merge;
                assert(p.first >= 0 && p.first < (int)leaves.size());
                merge.s1 = leaves[p.first];
                assert(p.second >= 0 && p.second < (int)leaves.size());
                merge.s2 = leaves[p.second];
                assert(merge.s1.len);
                assert(merge.s2.len);
                // TODO: Can possibly do better here?
                merge.min_idx = 0;
                merge.max_idx = merge.s1.len + merge.s2.len - 1;
                if (is_already_merged(merge.s1, merge.s2, merge.s1.dx)) {
                    result.pop_back();
                } else {
                    update_already_sorted(result.back());
                }
            }
        } else if (op.op == Merge) {

            // The standard decomposition of a batcher even-odd merge
            // step is to merge the even elements, then merge the odd
            // elements, then do n/2 - 1 fixup swaps. We don't have a
            // suitable single op for the fixup step in our
            // instruction set (intentionally, because it would have
            // horrible arithmetic intensity). So here's an
            // alternative decomposition that does fewer loads/stores
            // and more swaps

            // Special case merge decomposition for taking the median
            // of two sorted lists.
            bool extract_median = (op.s1.len == op.s2.len + 1 &&
                                   op.min_idx == op.max_idx &&
                                   op.min_idx == op.s2.len);
            if (extract_median && op.s1.len <= max_len) {
                // It's fine as-is. The check above was too strict.
                result.push_back(op);

            } else if (extract_median) {

                // Let the length of list 1 be n + 1, and let the
                // length of list 2 be n.

                // Consider the first element in list 1. Call it A. It's less
                // than n things, all from list 1.  Consider the second
                // element in list 1. Call it B. It's less than n - 1 things and
                // greater than one thing, all from list 1.

                // Consider the last element in list 2. It's greater
                // than n - 1 things in list 2. Call this element C

                // Now let's break their possible orderings down into cases:

                // A < B < C

                // If this is true, then A is actually less than n+1
                // things, because it's also less than C which was not
                // in list 1, so it can't possibly be the median.
                // Similarly, C is actually greater than n + 1 things,
                // because it's known to be greater than A and B, so
                // it can't be the median either.  We can thus discard
                // A and C and continue with two lists than are each
                // one shorter.

                // A < C < B

                // As before, we can eliminate A as a candidate for
                // median. B is greater than A, and it's also greater
                // than C and everything that C is in turn greater
                // than. This adds up to n + 1 things total because C
                // was greater than n - 1 things. Thus B is not the
                // median. C is still a contender. Because it's less
                // than B, it's smaller than the third element of list
                // 1, so after disposing of C and B we can attach A to
                // the start of list 1 without breaking the
                // ordering.

                // C < A < B

                // We can eliminate B as we did above. Consider now
                // C. C is less than A and transitively everything
                // that A is less than, this adds up to n + 1 things,
                // so we can eliminate C. A is still a contender for
                // median. It's smaller than B, so after dumping C and
                // A we can attach it to the start of list one without
                // breaking the ordering of list 1.

                // This argument gives us a nibble-and-conquer way to
                // work our way through the lists to finally get the
                // median. We can go k elements at a time instead of
                // one with no additional trouble.

                Op remaining = op;

                while (remaining.s1.len > max_len) {
                    // Nibble off the start of list 1 and the end of
                    // list 2, compute the median, and leave it at the
                    // start of list 1.
                    int nibble_size = max_len - 1;
                    Op nibble = remaining;
                    nibble.s1.len = nibble_size + 1;
                    nibble.s2.x += nibble.s2.dx * nibble.s2.len;
                    nibble.s2.len = nibble_size;
                    nibble.s2.x -= nibble.s2.dx * nibble.s2.len;
                    nibble.min_idx = nibble.max_idx = (nibble.s1.len + nibble.s2.len) / 2;

                    result.push_back(nibble);

                    // We bit off the start of s1 and the end of s2
                    remaining.s1.x += nibble_size * remaining.s1.dx;
                    remaining.s1.len -= nibble_size;
                    remaining.s2.len -= nibble_size;
                    remaining.min_idx = remaining.max_idx = (remaining.s1.len + remaining.s2.len) / 2;
                }

                if (remaining.s1.len) {
                    result.push_back(remaining);
                }

            } else {
                // Split sequences a and b into a0, a1, b0, b1
                // merge(a0, b0)
                // merge(a1, b1)
                // merge(a1, b0)

                // More generally, slice a and b into pieces of size at
                // most max_len/2, then use an even odd merge network of
                // over the pieces to combine. The smaller the pieces the
                // closer this is to a regular even-odd merge network.

                int a_leaves = (op.s1.len + (max_len / 2) - 1) / (max_len / 2);
                int b_leaves = (op.s2.len + (max_len / 2) - 1) / (max_len / 2);
                int a_leaf_size = (op.s1.len + a_leaves - 1) / a_leaves;
                int b_leaf_size = (op.s2.len + b_leaves - 1) / b_leaves;
                a_leaf_size = std::max(a_leaf_size, b_leaf_size);
                b_leaf_size = a_leaf_size;

                assert(a_leaf_size >= 1 && a_leaf_size <= max_len / 2);

                assert(a_leaf_size == b_leaf_size);

                // We'll implicitly pad a on the left with negative
                // infinities, and pad b on the right with positive
                // infinities, so that a and b are each a whole number
                // of leaves.

                // TODO: There's a simple closed form for this.
                int a_pad = 0;
                while ((op.s1.len + a_pad) % a_leaf_size) {
                    a_pad++;
                }

                // Figure out which leaves min_idx and max_idx will fall into post-sorting
                int min_leaf_idx, max_leaf_idx;
                bool min_idx_is_in_a = op.min_idx < op.s1.len;
                if (min_idx_is_in_a) {
                    min_leaf_idx = (op.min_idx + a_pad) / a_leaf_size;
                } else {
                    min_leaf_idx = (op.min_idx - op.s1.len) / b_leaf_size + a_leaves;
                }

                bool max_idx_is_in_a = op.max_idx < op.s1.len;
                if (max_idx_is_in_a) {
                    max_leaf_idx = (op.max_idx + a_pad) / a_leaf_size;
                } else {
                    max_leaf_idx = (op.max_idx - op.s1.len) / b_leaf_size + a_leaves;
                }

                vector<pair<int, int>> merge =
                    odd_even_merge(0, a_leaves, a_leaves, b_leaves, min_leaf_idx, max_leaf_idx);

                vector<Seq> leaves;
                for (int i = 0; i < a_leaves; i++) {
                    leaves.emplace_back(op.s1);
                    auto &leaf = leaves.back();
                    int leaf_start = i * a_leaf_size - a_pad;
                    int leaf_end = leaf_start + a_leaf_size;
                    leaf_start = std::max(leaf_start, 0);
                    leaf.x += leaf_start * leaf.dx;
                    leaf.len = leaf_end - leaf_start;
                    assert(leaf.len <= max_len);
                }

                for (int i = 0; i < b_leaves; i++) {
                    leaves.emplace_back(op.s2);
                    auto &leaf = leaves.back();
                    int leaf_start = i * b_leaf_size;
                    int leaf_end = std::min(leaf_start + b_leaf_size, op.s2.len);
                    leaf.x += leaf_start * leaf.dx;
                    leaf.len = leaf_end - leaf_start;
                    assert(leaf.len <= max_len);
                }

                for (auto p : merge) {
                    result.emplace_back(op);
                    Op &merge_pair = result.back();
                    merge_pair.s1 = leaves[p.first];
                    merge_pair.s2 = leaves[p.second];
                    // TODO: pruning
                    merge_pair.min_idx = 0;
                    merge_pair.max_idx = merge_pair.s1.len + merge_pair.s2.len - 1;

                    assert(merge_pair.s1.len <= max_len / 2 &&
                           merge_pair.s2.len <= max_len / 2);

                    if (((op.s1.dx < 0) == (op.s2.dx < 0)) &&
                        is_already_merged(merge_pair.s1, merge_pair.s2, op.s1.dx)) {
                        result.pop_back();
                    } else {
                        update_already_sorted(result.back());
                    }
                }
            }
        } else if (op.op == Copy) {
            assert(op.s1.len == op.s2.len);
            int num_leaves = (op.s1.len + max_len - 1) / max_len;
            int leaf_size = (op.s1.len + num_leaves - 1) / num_leaves;
            for (int i = 0; i < num_leaves; i++) {
                int leaf_start = i * leaf_size;
                int leaf_end = std::min(leaf_start + leaf_size, op.s1.len);
                result.emplace_back();
                Op &subcopy = result.back();
                subcopy = op;
                subcopy.s1.x += leaf_start * subcopy.s1.dx;
                subcopy.s1.len = leaf_end - leaf_start;
                subcopy.s2.x += leaf_start * subcopy.s2.dx;
                subcopy.s2.len = leaf_end - leaf_start;
                subcopy.min_idx = 0;
                subcopy.max_idx = subcopy.s1.len - 1;
            }
        } else {
            assert(false && "Unknown op");
        }

        // Whatever just happened was equivalent to the original op
        already_sorted.swap(already_sorted_before);
        update_already_sorted(op);
    }

    // Reorder the ops to minimize branch mispredictions in an
    // interpreter. This did not turn out to matter, so it's disabled.

#if 0
    auto overlaps = [](const Seq &a, const Seq &b) {
        // TODO: This is clearly egregiously slow
        for (int i = 0; i < a.len; i++) {
            for (int j = 0; j < b.len; j++) {
                if (a.x + i * a.dx == b.x + j * b.dx) {
                    return true;
                }
            }
        }
        return false;
    };

    auto can_reorder = [&](const Op &a, const Op &b) {
        return !overlaps(a.s1, b.s1) &&
               !overlaps(a.s1, b.s2) &&
               !overlaps(a.s2, b.s1) &&
               !overlaps(a.s2, b.s2);
    };

    auto same_code = [&](const Op &a, const Op &b) {
        return (a.op == b.op &&
                a.s1.len == b.s1.len &&
                a.s2.len == b.s2.len &&
                a.min_idx == b.min_idx &&
                a.max_idx == b.max_idx);
    };

    // Edges in the dependency graph. For each pair, the second int
    // is an op that must run after the first op.
    set<pair<int, int>> edges;
    std::map<int, vector<int>> consumers;
    std::map<int, int> unscheduled_producers;
    for (int i = 0; i < (int)result.size(); i++) {
        for (int j = i + 1; j < (int)result.size(); j++) {
            if (!can_reorder(result[i], result[j])) {
                edges.emplace(i, j);
                unscheduled_producers[j]++;
                consumers[i].push_back(j);
            }
        }
    }

    // Now schedule the ops
    vector<Op> optimized_result;
    set<int> pending;
    for (int i = 0; i < (int)result.size(); i++) {
        pending.insert(i);
    }

    Op last;
    auto schedule = [&](int op_idx) {
        printf("Scheduling %d\n", op_idx);
        if (result[op_idx].op != AssertSorted &&
            result[op_idx].op != Print) {
            last = result[op_idx];
        }
        optimized_result.push_back(result[op_idx]);
        pending.erase(op_idx);
        for (int i : consumers[op_idx]) {
            unscheduled_producers[i]--;
        }
    };

    // Start at the start
    schedule(0);

    while (!pending.empty()) {
        // Look for a schedulable op of the same type as the last
        // scheduled op
        int any_op = -1, matching_op = -1;
        for (int i : pending) {
            bool schedulable = (unscheduled_producers[i] == 0);
            if (schedulable) {
                if (same_code(last, result[i])) {
                    matching_op = i;
                    if (!can_reorder(last, result[i])) {
                        // Favor direct consumers for
                        // locality. Otherwise keep looking.
                        break;
                    }
                } else if (any_op == -1) {
                    any_op = i;
                }
            }
        }
        if (matching_op != -1) {
            any_op = matching_op;
        }
        assert(any_op != -1);
        schedule(any_op);
    }

    return optimized_result;
#endif

    return result;
}

void run(const int *src, int *dst, int W, int H, const vector<Op> &ops) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            dst[y * W + x] = src[y * W + x];
        }
    }

    int working_memory = compute_working_memory(ops);

    for (const auto &op : ops) {
        //op.print();
        if (op.op == Sort) {
            vector<int> seq;
            for (int i = 0; i < op.s1.len; i++) {
                int x = op.s1.x + i * op.s1.dx;
                seq.push_back(dst[x]);
            }
            std::sort(seq.begin(), seq.end());
            for (int i = 0; i < op.s1.len; i++) {
                int x = op.s1.x + i * op.s1.dx;
                dst[x] = seq[i];
            }
        } else if (op.op == AssertSorted) {
            bool ok = true;
            for (int i = op.min_idx + 1; i <= op.max_idx; i++) {
                int x = op.s1.x + i * op.s1.dx;
                int prev = x - op.s1.dx;
                ok &= dst[x] >= dst[prev];
            }
            if (!ok) {
                printf("AssertSorted failure:\n");
                printf("%s (%d): ", op.reason, op.s1.len);
                for (int i = 0; i < op.s1.len; i++) {
                    printf("%d ", dst[op.s1.x + i * op.s1.dx]);
                }
                printf("\n");
                abort();
            }
        } else if (op.op == Merge) {
            // Be grossly inefficient and make a copy of the whole
            // thing before doing the merge.
            vector<int> tmp(working_memory);
            for (int i = 0; i < working_memory; i++) {
                tmp[i] = dst[i];
            }

            int s1_remaining = op.s1.len;
            int s2_remaining = op.s2.len;
            int s1_x = op.s1.x;
            int s2_x = op.s2.x;
            for (int i = 0; i < op.s1.len + op.s2.len; i++) {
                int dst_x;
                if (i < op.s1.len) {
                    dst_x = op.s1.x + i * op.s1.dx;
                } else {
                    dst_x = op.s2.x + (i - op.s1.len) * op.s2.dx;
                }

                int src1 = s1_remaining ? tmp[s1_x] : 0x7fffffff;
                int src2 = s2_remaining ? tmp[s2_x] : 0x7fffffff;
                if (src1 < src2) {
                    dst[dst_x] = src1;
                    s1_x += op.s1.dx;
                    s1_remaining--;
                } else {
                    dst[dst_x] = src2;
                    s2_x += op.s2.dx;
                    s2_remaining--;
                }
            }
            assert(s1_remaining == 0 && s2_remaining == 0);
        } else if (op.op == Copy) {
            assert(op.s1.len == op.s2.len);
            for (int i = 0; i < op.s1.len; i++) {
                int src_x = op.s1.x + i * op.s1.dx;
                int dst_x = op.s2.x + i * op.s2.dx;
                dst[dst_x] = dst[src_x];
            }
        } else if (op.op == Print) {
            printf("%s (%d): ", op.reason, op.s1.len);
            for (int i = 0; i < op.s1.len; i++) {
                printf("%d ", dst[op.s1.x + i * op.s1.dx]);
            }
            printf("\n");
        } else {
            assert(false && "Unknown op");
        }
    }
}

vector<pair<Op, Instruction>> get_instruction_table(int max_len) {
    vector<pair<Op, Instruction>> result;
    int op_id = 0;
    for (int i = 1; i <= max_len; i++) {
        if (i >= 2) {
            // Dense sort of size i
            Op op;
            op.op = Sort;
            op.s1.len = i;
            op.s2.len = 0;
            op.min_idx = 0;
            op.max_idx = i - 1;
            op.s1.dx = 1;
            result.emplace_back(op, Instruction{op_id++, 0, 0});
        }

        // Median of two sorted lists where one s1 is one longer than
        // s2. Each input is only used once so we can afford to load a
        // lot of data.
        if (i >= 2) {
            Op op;
            op.op = Merge;
            op.s1.dx = op.s2.dx = 1;
            op.s1.len = i;
            op.s2.len = i - 1;
            op.min_idx = i - 1;
            op.max_idx = i - 1;
            result.emplace_back(op, Instruction{op_id++, 0, 0});
        }

        if ((i & 1) == 0) {

            // Generic Merge
            for (int j = 1; j <= i / 2; j++) {
                Op op;
                op.op = Merge;
                op.s1.len = i / 2;
                op.s2.len = j;
                op.min_idx = 0;
                op.max_idx = i / 2 + j - 1;

                op.s1.dx = op.s2.dx = 1;

                result.emplace_back(op, Instruction{op_id++, 0, 0});

                if (op.s1.len != op.s2.len) {
                    std::swap(op.s1, op.s2);
                    result.emplace_back(op, Instruction{op_id++, 0, 0});
                }
            }
        }
        // Copy of size i
        if (i > 0) {
            Op op;
            op.op = Copy;
            op.s1.len = i;
            op.s2.len = i;
            op.min_idx = 0;
            op.max_idx = i - 1;

            // Dense to dense
            op.s1.dx = 1;
            op.s2.dx = 1;
            result.emplace_back(op, Instruction{op_id++, 0, 0});

            // Diagonal to dense
            op.s1.dx = 0;
            op.s2.dx = 1;
            result.emplace_back(op, Instruction{op_id++, 0, 0});
        }
    }
    return result;
}

Instruction encode(Op op, const vector<pair<Op, Instruction>> &table) {
    for (const auto &p : table) {
        bool s1_dense = (op.s1.dx == 1);
        bool s2_dense = (op.s2.dx == 1);
        if (p.first.op == op.op &&
            p.first.s1.len == op.s1.len &&
            p.first.s2.len == op.s2.len &&
            (p.first.s1.dx == 1) == s1_dense &&
            (p.first.s2.dx == 1) == s2_dense &&
            op.min_idx >= p.first.min_idx &&
            op.max_idx <= p.first.max_idx) {
            Instruction inst = p.second;
            inst.x1 = op.s1.x;
            inst.x2 = op.s2.x;
            return inst;
        }
    }
    op.print();
    assert(false && "Could not encode op");
}

Op decode(Instruction inst, int stride, const vector<pair<Op, Instruction>> &table) {
    for (const auto &p : table) {
        if (p.second.op == inst.op) {
            Op op = p.first;
            op.s1.x = inst.x1;
            op.s2.x = inst.x2;
            if (op.s1.dx == 0) {
                op.s1.dx = stride;
            }
            return op;
        }
    }
    assert(false && "Could not decode instruction");
}

void print_op_histogram(const vector<Op> &ops) {
    std::map<std::string, int> inst_hist;
    std::map<std::string, int> swaps_hist;
    vector<std::string> reasons;

    std::map<std::tuple<int, int, int, int, int>, int> op_type;
    int total_inst = 0, total_swaps = 0;
    for (const auto &op : ops) {
        std::string r{op.reason};
        if (!inst_hist.count(r)) {
            reasons.push_back(r);
        }
        auto counters = estimate_performance({op});
        inst_hist[r]++;
        swaps_hist[r] += counters.swaps;
        total_inst++;
        total_swaps += counters.swaps;
        op_type[{op.op, op.s1.len, op.s2.len, op.min_idx, op.max_idx}]++;
    }

    printf("Instruction / swap count by reason:\n");
    for (const auto &r : reasons) {
        printf("% 5d % 5d ( %5.2f %% %5.2f %%)  %s\n",
               inst_hist[r], swaps_hist[r],
               (100.0 * inst_hist[r]) / total_inst,
               (100.0 * swaps_hist[r]) / total_swaps,
               r.c_str());
    }

    printf("Instruction count by instruction type:\n");
    for (const auto &p : op_type) {
        int type = std::get<0>(p.first);
        int l1 = std::get<1>(p.first);
        int l2 = std::get<2>(p.first);
        int min_idx = std::get<3>(p.first);
        int max_idx = std::get<4>(p.first);
        int count = p.second;
        if (type == Sort) {
            printf("% 5d ( %5.2f %%)  sort %d\n",
                   count, (100.0 * count) / total_inst, l1);
        } else if (type == Merge) {
            printf("% 5d ( %5.2f %%)  merge %d %d (%d %d)\n",
                   count, (100.0 * count) / total_inst, l1, l2, min_idx, max_idx);
        } else if (type == Copy) {
            printf("% 5d ( %5.2f %%)  copy %d\n",
                   count, (100.0 * count) / total_inst, l1);
        }
    }
}

PerformanceCounters estimate_performance(const vector<Op> &ops) {
    PerformanceCounters result;
    for (const auto &op : ops) {
        if (op.op == Sort) {
            // Load the offset
            result.loads++;
            // Load the values
            result.loads += op.s1.len;
            // Store the values
            result.stores += op.s1.len;
            result.swaps += (int)pairwise_sort(op.s1.len, op.min_idx, op.max_idx, false).size();
        } else if (op.op == Merge) {
            // Load the offsets
            result.loads += 2;
            // Load the values
            result.loads += op.s1.len + op.s2.len;
            if (op.min_idx == op.max_idx &&
                op.s1.len == op.s2.len + 1 &&
                op.min_idx == op.s2.len) {
                // Median of two sorted lists
                result.stores++;
                result.swaps += op.s2.len * 2;
            } else {

                // Store the values
                result.stores += op.s1.len + op.s2.len;

                result.swaps += (int)odd_even_merge(0, op.s1.len,
                                                    op.s1.len, op.s2.len,
                                                    op.min_idx, op.max_idx)
                                    .size();
            }
        } else if (op.op == Copy) {
            // Load the offsets
            result.loads += 2;

            // Load the values
            result.loads += op.s1.len;
            // Store the values
            result.stores += op.s1.len;
        }
    }
    return result;
}
