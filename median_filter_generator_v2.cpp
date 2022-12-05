#include "Halide.h"

#include "sorting_bytecode.h"
#include "sorting_network.h"

using namespace Halide;
using std::pair;
using std::string;
using std::vector;

class MedianFilter : public Generator<MedianFilter> {

public:
    GeneratorParam<int> radius{"radius", 1};
    GeneratorParam<int> tw_{"tw", -1};
    GeneratorParam<int> th_{"th", -1};

    int tw, th;

    Input<Buffer<>> src{"src", 2};
    Output<Buffer<>> dst{"dst", 2};

    Var x{"x"}, y{"y"}, u{"u"}, v{"v"};
    Expr diameter;

    Expr return_second(const Expr &a, const Expr &b) {
        return Internal::Call::make(a.type(),
                                    Internal::Call::return_second,
                                    {a, b},
                                    Internal::Call::PureIntrinsic);
    }

    // Equivalent to Halide's mux operator, but uses a binary tree and
    // accepts lists of size 1.
    Expr bmux(Expr idx, const vector<Expr> &values) {
        assert(!values.empty());
        if (values.size() == 1) {
            return values[0];
        } else if (values.size() == 2) {
            return select(idx == 0, values[0], values[1]);
        } else if (values.size() == 3) {
            return select(idx == 0, values[0],
                          idx == 1, values[1],
                          values[2]);
        } else {
            auto middle = values.begin() + values.size() / 2;
            vector<Expr> a(values.begin(), middle), b(middle, values.end());
            return select(idx < (int)a.size(), bmux(idx, a), bmux(idx - (int)a.size(), b));
        }
    }

    vector<Expr> apply_network(const vector<pair<int, int>> &network, vector<Expr> values) {
        for (const auto &p : network) {
            Expr a = values[p.first];
            Expr b = values[p.second];
            // We need some way to force the two values to get
            // evaluated at the same time, because then both
            // inputs can be retired. Otherwise we're at the mercy
            // of Halide's instruction scheduler. We'll
            // make both values depend on the min *and* the max
            // using Halide's return_second intrinsic, which
            // evaluates both args and then returns the second
            // one.
            values[p.first] = return_second(max(a, b), min(a, b));
            values[p.second] = return_second(min(a, b), max(a, b));
            //values[p.first] = min(a, b);
            //values[p.second] = max(a, b);
        }
        return values;
    }

    int sort_in_place(Func f, vector<Op> instructions,
                      int starting_pc = 0,
                      int guard_region_start = -1,
                      int guard_region_end = -1) {
        //RDom r_debug(0, diameter * diameter);
        //f(x, y, r_debug) = print_when(x == 0 && y == 0, f(x, y, r_debug), r_debug, f.name() + " before");

        vector<Expr> args;
        for (Var v : f.args()) {
            args.push_back(v);
        }

        auto f_at = [&](Expr e) {
            // We're always sorting along dimension 2
            args[2] = e;
            return f(args);
        };

        int swaps = 0;
        int pc;  // program counter in the instruction stream
        for (pc = starting_pc; pc < (int)instructions.size(); pc++) {
            const auto &op = instructions[pc];

            // If the op breaches the guard region, stop and return current pc
            int x1 = op.s1.x, x2 = op.s1.x + op.s1.dx * (op.s1.len - 1);
            int x3 = op.s2.x, x4 = op.s2.x + op.s2.dx * (op.s2.len - 1);
            if (guard_region_start >= 0 &&
                ((x1 >= guard_region_start && x1 < guard_region_end) ||
                 (x2 >= guard_region_start && x2 < guard_region_end) ||
                 (x3 >= guard_region_start && x3 < guard_region_end) ||
                 (x4 >= guard_region_start && x4 < guard_region_end))) {
                return pc;
            }

            op.print();
            if (op.op == Sort || op.op == Merge) {
                vector<Expr> values, coords;
                for (int i = 0; i < op.s1.len; i++) {
                    values.push_back(f_at(op.s1.x + i * op.s1.dx));
                    coords.emplace_back(op.s1.x + i * op.s1.dx);
                }
                for (int i = 0; i < op.s2.len; i++) {
                    values.push_back(f_at(op.s2.x + i * op.s2.dx));
                    coords.emplace_back(op.s2.x + i * op.s2.dx);
                }
                vector<pair<int, int>> net;
                if (op.op == Sort) {
                    net = pairwise_sort(op.s1.len, op.min_idx, op.max_idx, false);
                } else {
                    net = odd_even_merge(0, op.s1.len, op.s1.len, op.s2.len, op.min_idx, op.max_idx);
                }

                // Reorder sorting network links to maximize reuse of loaded values.
                auto may_reorder = [](const pair<int, int> &a,
                                      const pair<int, int> &b) {
                    return a.first != b.first &&
                           a.second != b.first &&
                           a.first != b.second &&
                           a.second != b.second;
                };
                // Push everything as far back as possible
                for (int i = (int)net.size() - 1; i >= 0; i--) {
                    for (size_t j = i + 1; j < net.size(); j++) {
                        if (may_reorder(net[j - 1], net[j])) {
                            std::swap(net[j - 1], net[j]);
                        } else {
                            break;
                        }
                    }
                }
                // Pull everything as far forwards as possible
                for (size_t i = 0; i < net.size(); i++) {
                    for (size_t j = i; j > 0; j--) {
                        if (may_reorder(net[j - 1], net[j])) {
                            std::swap(net[j - 1], net[j]);
                        } else {
                            break;
                        }
                    }
                }

                swaps += (int)net.size();

                if (get_target().has_gpu_feature()) {
                    values = apply_network(net, values);
                    f_at(scatter(coords)) = gather(values);
                } else {

                    vector<Expr> dst_coords, dst_values;
                    for (size_t i = 0; i < net.size();) {
                        vector<bool> used(values.size(), false);
                        int num_used = 0;
                        // The maximum number of unique values we load
                        // before doing some sorting operations on them
                        // and storing again.  Too small and we send lots
                        // of data back and forth from memory. Too large
                        // and we just spill and send lots of data back
                        // and forth to the stack. There's a nice wide
                        // plateau of good values from 4 - 12. We just
                        // need to keep the number of memory ops under the
                        // number of min or max ops.
                        const int limit = 8;

                        size_t j = i;
                        for (; j < net.size(); j++) {
                            const auto &p = net[j];
                            int new_uses = !used[p.first] + !used[p.second];
                            if (new_uses + num_used > limit) {
                                break;
                            } else {
                                used[p.first] = used[p.second] = true;
                                num_used += new_uses;
                            }
                        }
                        vector<pair<int, int>> subnet(net.begin() + i, net.begin() + j);
                        i = j;
                        vector<Expr> subcoords, subvalues;
                        vector<Expr> sorted = apply_network(subnet, values);
                        for (size_t k = 0; k < sorted.size(); k++) {
                            if (used[k]) {
                                subcoords.push_back(coords[k]);
                                subvalues.push_back(sorted[k]);
                            }
                        }
                        while (subcoords.size() < limit) {
                            subcoords.push_back(subcoords.back());
                            subvalues.push_back(subvalues.back());
                        }
                        dst_coords.push_back(scatter(subcoords));
                        dst_values.push_back(gather(subvalues));
                    }
                    if (dst_coords.size() == 1) {
                        f_at(dst_coords[0]) = dst_values[0];
                    } else {
                        RDom r(0, (int)dst_values.size());
                        (f_at(bmux(r, dst_coords)) = bmux(r, dst_values)).unroll(r);
                    }
                }
            } else if (op.op == Copy) {
                RDom r(0, op.s1.len);
                Stage s{f_at(op.s2.x + r * op.s2.dx) = f_at(op.s1.x + r * op.s1.dx)};
                s.unroll(r);
            }
            printf("Swaps: %d\n", swaps);
        }
        return pc;
    }

    void generate() {
        tw = tw_;
        th = th_;
        if (tw == -1) {
            // Select a good tile size. This decision tree was
            // empirically determined.
            if (radius < 2) {
                tw = 1;
            } else if (radius < 9 ||
                       (get_target().has_gpu_feature() &&
                        ((src.type().bits() > 8 &&
                          radius < 15) ||
                         radius < 12))) {
                tw = 2;
            } else {
                tw = 4;
            }
        }
        if (th == -1) {
            if (radius < 9 ||
                (get_target().has_gpu_feature() &&
                 ((src.type().bits() > 8 &&
                   radius < 15) ||
                  radius < 12))) {
                th = 2;
            } else {
                th = 4;
            }
        }

        // Load a square footprint, then apply a sorting network to it
        // in-place and take the median output.
        int diameter = radius * 2 + 1;

        Func sorted_cols{"sorted_cols"}, merged_cols{"merged_cols"};

        // Make some scratch space for the 1d sort over columns
        sorted_cols(x, y, u, v) = undef(src.type());

        int core_width = diameter - (tw - 1);
        int core_height = diameter - (th - 1);

        // The top left corner of our region loaded from src
        Expr x_base = x * tw - radius, y_base = y * th - radius;

        // Make three separate Funcs for the three different uses of
        // the source image so we can schedule them all
        // separately. They'll all be stored interleaved so that we
        // don't end up doing lots of strided loads when tw > 1.
        Func src_i_cols{"src_i_cols"}, src_i_top{"src_i_top"}, src_i_bottom{"src_i_bottom"};
        src_i_cols(x, y, v) = src(x * tw + v, y);
        src_i_top(x, y, v) = src(x * tw + v, y);
        src_i_bottom(x, y, v) = src(x * tw + v, y);

        // Load a core column into it
        RDom r_load_1d(0, core_height);
        Stage loading_col_input_1d{
            sorted_cols(x, y, r_load_1d, v) = src_i_cols(x, y_base + r_load_1d + th - 1, v)};
        loading_col_input_1d.unroll(r_load_1d.x);

        auto sort_cols_program = sort_1d(core_height, 0, core_height - 1);

        // Do the sort of each column
        sort_in_place(sorted_cols, sort_cols_program);

        Func sorted_cols_i{"sorted_cols_i"};
        sorted_cols_i(x, y, u, v) = sorted_cols(x, y, u, v);

        // Reinterleave so that I don't have to change the indexing math below
        Func src_r_top{"src_r_top"}, src_r_bottom{"src_r_bottom"}, sorted_cols_r{"sorted_cols_r"};
        src_r_top(x, y) = src_i_top(x / tw, y, x % tw);
        src_r_bottom(x, y) = src_i_bottom(x / tw, y, x % tw);
        sorted_cols_r(x, y, u) = sorted_cols_i(x / tw, y, u, x % tw);

        // Make some scratch space in which to merge the sorted cols
        merged_cols(x, y, u) = undef(src.type());

        int core_start = 0;

        // For very small radius, don't use diagonal sort as the core
        // is too small for it to make sense.
        const bool col_major_core = (radius < 3);

        // Prepare the column merge program
        auto merge_cols_program = tiled_median_2d(diameter, diameter, tw, th, col_major_core);

        if (1) {
            // Print some stats about the generated program
            PerformanceCounters col_sort_perf = estimate_performance(sort_cols_program);
            PerformanceCounters col_merge_perf = estimate_performance(merge_cols_program);

            printf("Col sort counters: loads: %d stores: %d swaps: %d\n",
                   col_sort_perf.loads, col_sort_perf.stores, col_sort_perf.swaps);
            printf("Col merge counters: loads: %d stores: %d swaps: %d\n",
                   col_merge_perf.loads, col_merge_perf.stores, col_merge_perf.swaps);

            double loads_per_pixel = (double)col_sort_perf.loads / th + (double)col_merge_perf.loads / (tw * th);
            double stores_per_pixel = (double)col_sort_perf.stores / th + (double)col_merge_perf.stores / (tw * th);
            double swaps_per_pixel = (double)col_sort_perf.swaps / th + (double)col_merge_perf.swaps / (tw * th);
            printf("Total per-pixel loads: %.2f stores: %.2f swaps: %.2f\n",
                   loads_per_pixel,
                   stores_per_pixel,
                   swaps_per_pixel);
        }

        // Program counter in the instruction stream
        int pc = 0;

        if (col_major_core) {
            // Load the core into it first in col-major order.
            RDom r_load_core(0, core_width, 0, core_height);
            Stage loading_core{
                merged_cols(x, y, core_start + r_load_core.y + r_load_core.x * core_height) =
                    sorted_cols_r(x_base + r_load_core.x + tw - 1, y, r_load_core.y)};

            loading_core.unroll(r_load_core.x).unroll(r_load_core.y);
        } else if (radius <= 5 || get_target().has_gpu_feature()) {
            // Load the core into it first in row-major order.
            RDom r_load_core(0, core_width, 0, core_height);
            Stage loading_core{
                merged_cols(x, y, core_start + r_load_core.y * core_width + r_load_core.x) =
                    sorted_cols_r(x_base + r_load_core.x + tw - 1, y, r_load_core.y)};

            loading_core.unroll(r_load_core.x).unroll(r_load_core.y);
        } else {
            // Code size gets out of hand if we follow the
            // instructions in the progam to the letter. Do a full
            // sort of each row instead, inside a loop. Does more
            // swaps because the row sorts can't prune their outputs,
            // but it's a net win because we generate many fewer
            // icache misses.
            RDom r_y(0, core_height);
            vector<Expr> values;
            vector<Expr> coords;
            for (int i = 0; i < core_width; i++) {
                values.push_back(sorted_cols_r(x_base + i + tw - 1, y, r_y));
                coords.push_back(core_start + r_y * core_width + i);
            }
            auto net = pairwise_sort(core_width);
            values = apply_network(net, values);
            merged_cols(x, y, scatter(coords)) = gather(values);

            // Skip the sort of the rows. We just did it.
            const char *sort_along_rows = merge_cols_program[0].reason;
            while (merge_cols_program[pc].reason == sort_along_rows) {
                pc++;
            }
        }

        // Load the input into the appropriate locations as dictated
        // by the comment in sorting_bytecode.h
        int extra_cols_start = core_width * core_height;

        printf("Starting at pc %d\n", pc);

        int scratch_start = (diameter + tw - 1) * (diameter + th - 1);
        printf("Scratch space starts at %d\n", scratch_start);

        // Run as much of the program as we can on the core alone. We
        // will eagerly try to run more of the program after each
        // loading phase.
        pc = sort_in_place(merged_cols, merge_cols_program, pc, extra_cols_start, scratch_start);

        RDom r_load_extra_cols(0, tw - 1, 0, core_height);
        Stage loading_extra_cols_on_left{
            merged_cols(x, y, extra_cols_start + r_load_extra_cols.x * core_height + r_load_extra_cols.y) =
                sorted_cols_r(x_base + r_load_extra_cols.x, y, r_load_extra_cols.y)};
        loading_extra_cols_on_left.unroll(r_load_extra_cols.x).unroll(r_load_extra_cols.y);

        extra_cols_start += (tw - 1) * core_height;

        printf("Starting at pc %d\n", pc);
        pc = sort_in_place(merged_cols, merge_cols_program, pc, extra_cols_start, scratch_start);

        Stage loading_extra_cols_on_right{
            merged_cols(x, y, extra_cols_start + r_load_extra_cols.x * core_height + r_load_extra_cols.y) =
                sorted_cols_r(x_base + r_load_extra_cols.x + diameter, y, r_load_extra_cols.y)};
        loading_extra_cols_on_right.unroll(r_load_extra_cols.x).unroll(r_load_extra_cols.y);

        int extra_rows_start = extra_cols_start + (tw - 1) * core_height;

        printf("Starting at pc %d\n", pc);
        pc = sort_in_place(merged_cols, merge_cols_program, pc, extra_rows_start, scratch_start);

        RDom r_load_extra_rows(0, th - 1, 0, core_width);
        Stage loading_extra_rows_at_top{
            merged_cols(x, y, extra_rows_start + r_load_extra_rows.y + r_load_extra_rows.x * core_width) =
                src_r_top(x_base + r_load_extra_rows.y + tw - 1, y_base + r_load_extra_rows.x)};
        loading_extra_rows_at_top.unroll(r_load_extra_rows.x).unroll(r_load_extra_rows.y);

        extra_rows_start += (th - 1) * core_width;

        printf("Starting at pc %d\n", pc);
        pc = sort_in_place(merged_cols, merge_cols_program, pc, extra_rows_start, scratch_start);

        Stage loading_extra_rows_at_bottom{
            merged_cols(x, y, extra_rows_start + r_load_extra_rows.y + r_load_extra_rows.x * core_width) =
                src_r_bottom(x_base + r_load_extra_rows.y + tw - 1, y_base + diameter + r_load_extra_rows.x)};
        int corners_start = extra_rows_start + (th - 1) * core_width;
        loading_extra_rows_at_bottom.unroll(r_load_extra_rows.x).unroll(r_load_extra_rows.y);

        printf("Starting at pc %d\n", pc);
        pc = sort_in_place(merged_cols, merge_cols_program, pc, corners_start, scratch_start);

        RDom r_load_corners(0, tw - 1, 0, th - 1);

        Stage load_top_left_corners{
            merged_cols(x, y, corners_start + r_load_corners.y * 2 * (tw - 1) + r_load_corners.x) =
                src_r_top(x_base + r_load_corners.x, y_base + r_load_corners.y)};
        load_top_left_corners.unroll(r_load_corners.x).unroll(r_load_corners.y);

        Stage load_top_right_corners{
            merged_cols(x, y, corners_start + r_load_corners.y * 2 * (tw - 1) + r_load_corners.x + tw - 1) =

                src_r_top(x_base + diameter + r_load_corners.x, y_base + r_load_corners.y)};
        load_top_right_corners.unroll(r_load_corners.x).unroll(r_load_corners.y);

        corners_start += 2 * (tw - 1) * (th - 1);

        Stage load_bottom_left_corners{
            merged_cols(x, y, corners_start + r_load_corners.y * 2 * (tw - 1) + r_load_corners.x) =
                src_r_bottom(x_base + r_load_corners.x, y_base + diameter + r_load_corners.y)};

        load_bottom_left_corners.unroll(r_load_corners.x).unroll(r_load_corners.y);

        Stage load_bottom_right_corners{
            merged_cols(x, y, corners_start + r_load_corners.y * 2 * (tw - 1) + r_load_corners.x + tw - 1) =
                src_r_bottom(x_base + diameter + r_load_corners.x, y_base + diameter + r_load_corners.y)};

        load_bottom_right_corners.unroll(r_load_corners.x).unroll(r_load_corners.y);

        int results_start = corners_start + 2 * (tw - 1) * (th - 1);

        /*
        Stage dump_loaded_data{
            merged_cols(x, y, u) = print_when(x == 0 && y == 0, merged_cols(x, y, u), u, "before")};
        */

        // Now that everything is loaded, run the rest of the program
        printf("Starting at pc %d\n", pc);
        sort_in_place(merged_cols, merge_cols_program, pc);

        /*
        RDom r_dump(0, merge_cols_working_buffer_size);
        Stage dump_state_after_sort{
        merged_cols(x, y, r_dump) = print_when(x == 0 && y == 0, merged_cols(x, y, r_dump), r_dump, "after")};
        */

        // Extract the results

        Func median{"median"};
        median(x, y) = merged_cols(x / tw, y / th, results_start + tw * (y % th) + (x % tw));

        dst(x, y) = median(x, y);

        // Schedule
        Var xii, yii, xi, yi, xo, yo, xio;
        if (get_target().has_gpu_feature()) {
            const int warp = 32;

            if (radius > 3) {
                // compute_root the sorted cols and stage loads of it
                // into shared.
                sorted_cols.in()
                    .compute_root()
                    .split(x, x, xi, warp)
                    .reorder(u, xi, v, x, y)
                    .gpu_threads(xi, v)
                    .gpu_blocks(x, y)
                    .unroll(u);

                sorted_cols.in()
                    .in()
                    .compute_at(dst, x)
                    .split(x, x, xi, warp * 2)
                    .vectorize(xi, 2)
                    .gpu_lanes(xi)
                    .reorder(x, y, u, v, xi);

                // Stage the other rows of src into shared
                src.in(src_i_top)
                    .compute_at(dst, x)
                    .split(_0, x, xi, warp)
                    .gpu_lanes(xi);

                src.in(src_i_bottom)
                    .compute_at(dst, x)
                    .split(_0, x, xi, warp)
                    .gpu_lanes(xi);

                dst
                    .align_bounds(x, tw)
                    .align_bounds(y, th)
                    .tile(x, y, xii, yii, tw, th)
                    .unroll(xii)
                    .unroll(yii)
                    .split(x, x, xi, warp)
                    .gpu_lanes(xi)
                    .gpu_blocks(x, y);

            } else {
                // Compute the sorted cols directly into the register
                // file, accessing it later using warp shuffles. This
                // incurs some redundant work at tile boundaries.

                const int vec = radius == 1 ? 2 : 1;

                sorted_cols.in()
                    .store_in(MemoryType::Register)
                    .align_storage(x, warp)
                    .compute_at(dst, x)
                    .reorder(u, x, v, y)
                    .split(x, x, xi, vec * warp, TailStrategy::GuardWithIf)
                    .unroll(x)
                    .vectorize(xi, vec)
                    .gpu_lanes(xi)
                    .unroll(u);

                // Stage the extra rows of src into registers and
                // access using warp shuffles
                src.in(src_i_top)
                    .store_in(MemoryType::Register)
                    .compute_at(dst, x)
                    .split(_0, x, xi, vec * warp, TailStrategy::GuardWithIf)
                    .unroll(x)
                    .vectorize(xi, vec)
                    .gpu_lanes(xi);

                src.in(src_i_bottom)
                    .store_in(MemoryType::Register)
                    .compute_at(dst, x)
                    .split(_0, x, xi, vec * warp, TailStrategy::GuardWithIf)
                    .unroll(x)
                    .vectorize(xi, vec)
                    .gpu_lanes(xi);

                dst
                    .align_bounds(x, tw)
                    .align_bounds(y, th)
                    .tile(x, y, xii, yii, tw, th)
                    .unroll(xii)
                    .unroll(yii)
                    .split(x, x, xi, vec * warp - 2 * radius)
                    .vectorize(xi, vec)
                    .gpu_lanes(xi)
                    .gpu_blocks(x, y);

                if (vec > 1) {
                    for (int i = 0; i < sorted_cols.num_update_definitions(); i++) {
                        sorted_cols.update(i).vectorize(x);
                    }

                    for (int i = 0; i < merged_cols.num_update_definitions(); i++) {
                        merged_cols.update(i).vectorize(x);
                    }
                }
            }

            sorted_cols.compute_at(sorted_cols.in(), xi);
            merged_cols.compute_at(dst, xi);

        } else {
            const int vec = natural_vector_size(src.type());

            dst
                .align_bounds(x, tw)
                .align_bounds(y, th)
                .tile(x, y, xii, yii, tw, th)
                .unroll(xii)
                .unroll(yii)
                .split(y, y, yi, 32 / th)
                .split(x, x, xi, vec)
                .vectorize(xi)
                .parallel(y);

            // Vectorize wider and use additional registers if we can
            // afford them, to cover memory latencies.
            int col_unroll = std::max(1, 4 / (radius * tw));

            sorted_cols_i.compute_at(dst, yi)
                .store_in(MemoryType::Stack)
                .align_storage(x, vec)
                .split(x, xo, xi, vec * col_unroll)
                .reorder(v, xi, u, xo, y)
                .vectorize(xi, vec)
                .unroll(xi)
                .unroll(u)
                .unroll(v);

            sorted_cols.compute_at(sorted_cols_i, xo)
                .store_in(MemoryType::Stack)
                .align_storage(x, vec)
                .vectorize(x, vec)
                .unroll(x);

            loading_col_input_1d.reorder(v, x, r_load_1d, y);

            if (tw > 1) {
                // Tile width > 1, so schedule the
                // interleaving/deinterleaving stages.
                src.in(src_i_cols)
                    .compute_at(sorted_cols, r_load_1d)
                    .vectorize(_0)
                    .unroll(_1);

                src_i_top.compute_at(dst, x)
                    .vectorize(x, vec, TailStrategy::RoundUp)
                    .reorder(v, y)
                    .unroll(x)
                    .unroll(y)
                    .unroll(v);

                src.in(src_i_top)
                    .compute_at(src_i_top, y)
                    .vectorize(_0)
                    .unroll(_1);

                src_i_bottom.compute_at(dst, x)
                    .vectorize(x, vec, TailStrategy::RoundUp)
                    .reorder(v, y)
                    .unroll(x)
                    .unroll(y)
                    .unroll(v);

                src.in(src_i_bottom)
                    .compute_at(src_i_bottom, y)
                    .vectorize(_0)
                    .unroll(_1);
            }
            for (int i = 0; i < sorted_cols.num_update_definitions(); i++) {
                sorted_cols.update(i)
                    .unroll(v)
                    .vectorize(x, vec)
                    .unroll(x);
            }

            merged_cols.compute_at(dst, x)
                .store_in(MemoryType::Stack)
                .align_storage(x, vec)
                .vectorize(x);

            for (int i = 0; i < merged_cols.num_update_definitions(); i++) {
                merged_cols.update(i).vectorize(x);
            }
        }

        // Require that the output starts at a multiple of the tile
        // size. Zero works.
        dst.dim(0).set_min(0);
        dst.dim(1).set_min(0);
    }
};

HALIDE_REGISTER_GENERATOR(MedianFilter, median_filter);
