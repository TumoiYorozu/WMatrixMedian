#include "Halide.h"

#include "sorting_bytecode.h"
#include "sorting_network.h"

using namespace Halide;
using std::pair;
using std::string;
using std::vector;

class DynamicMedianFilter : public Generator<DynamicMedianFilter> {

public:
    Input<int> radius_{"radius"};

    // The sorting network to run for the initial preprocessing of a
    // 1d vertical footprint around each pixel.
    Input<Buffer<int>> sort_cols_program{"sort_cols_program", 2};

    // The sorting network to run over the 2d footprints.
    Input<Buffer<int>> merge_cols_program{"merge_cols_program", 2};
    Input<int> merge_cols_working_buffer_size{"merge_cols_working_buffer_size"};

    Input<Buffer<>> src{"src", 2};
    Output<Buffer<>> dst{"dst", 2};

    GeneratorParam<int> tw{"tw", 2};
    GeneratorParam<int> th{"th", 2};

    // Maximum instruction size supported by the interpreter. 14 seems
    // to be a sweet spot for x86. Larger is better on GPU.
    GeneratorParam<int> max_leaf{"max_leaf", 14};

    Var x{"x"}, y{"y"}, u{"u"}, v{"v"};
    Expr diameter;

    Stage sort_in_place(Func f, const Input<Buffer<int>> &instructions, Expr working_buffer_size) {
        // Apply all the actions in the input sorting network using an
        // interpreter. The way we'll phrase it is that we'll apply
        // all possible ops in the inner loop, and then use
        // RDom::where to mask off all but the correct one. Should
        // compile to something like a switch statement.

        //RDom r_debug(0, diameter * diameter);
        //f_at(r_debug) = print_when(x == 0 && y == 0, f_at(r_debug), r_debug, f.name() + " before");

        vector<Expr> args;
        for (Var v : f.args()) {
            args.push_back(v);
        }

        auto f_at = [&](Expr e) {
            // We're always sorting along dimension 2
            args[2] = e;
            return f(args);
        };

        auto table = get_instruction_table(max_leaf);
        int num_possible_actions = (int)table.size();

        RDom r(0, num_possible_actions, 0, instructions.dim(1).extent());
        Expr opcode = instructions(0, r.y);
        Expr x1 = instructions(1, r.y);
        Expr x2 = instructions(2, r.y);

        auto return_second = [](const Expr &a, const Expr &b) {
            return Internal::Call::make(a.type(),
                                        Internal::Call::return_second,
                                        {a, b},
                                        Internal::Call::PureIntrinsic);
        };

        auto apply_network = [&](const vector<pair<int, int>> &network, vector<Expr> values) {
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
            }
            return values;
        };

        // For every possible instruction we have values to store, and
        // coords to store them to.
        struct Action {
            vector<Expr> values, coords;
        };
        vector<Action> actions;

        // We know bounds on the computed coords that Halide can't
        // infer, so just make some promises so it knows how much to
        // allocate for the footprint.
        auto c = [=](const Expr &e) {
            return unsafe_promise_clamped(e, 0, working_buffer_size - 1);
        };

        // For each action, we also have the condition under which we
        // should perform that action
        vector<Expr>
            conditions;

        for (const auto &p : table) {
            const auto &op = p.first;
            const auto &inst = p.second;

            conditions.push_back(opcode == inst.op);

            // TODO: It would be cleaner if the stride was an
            // additional input rather than computing it from the
            // diameter here.
            Expr stride = 1 - (diameter - tw + 1);
            stride = select((int)tw < th, -stride, stride);

            Expr dx1 = (op.s1.dx == 1) ? 1 : stride;
            Expr dx2 = (op.s2.dx == 1) ? 1 : stride;
            op.print();
            std::cout << inst.op << ": " << dx1 << ", " << dx2 << "\n";

            Action action;
            if (op.op == Sort) {
                for (int i = 0; i < op.s1.len; i++) {
                    action.values.emplace_back(f_at(c(x1 + i * dx1)));
                    action.coords.emplace_back(c(x1 + i * dx1));
                }
                auto net = pairwise_sort(op.s1.len);
                action.values = apply_network(net, action.values);
            } else if (op.op == Merge) {
                if (op.min_idx == op.max_idx && op.min_idx == op.s2.len && op.s1.len == op.s2.len + 1) {
                    // Median of two sorted lists where the first is
                    // one larger than the second. We only need to
                    // store one value here, and there's a nice closed
                    // form for it with no repeated subexpressions, so
                    // just make an Expr.

                    // Start with the first element of the longer list
                    Expr med_idx = c(x1);
                    Expr med = f_at(med_idx);
                    for (int i = 1; i < op.s1.len; i++) {
                        // Pair up two elements of s1 and s2 and take the min.
                        Expr a = f_at(c(x1 + i * dx1));
                        Expr b = f_at(c(x2 + (op.s2.len - i) * dx2));
                        // The median is the max of all those pairs
                        med = max(med, min(a, b));
                    }
                    action.values.emplace_back(med);
                    action.coords.emplace_back(c(x1 + op.s2.len * dx1));
                } else {
                    for (int i = 0; i < op.s1.len; i++) {
                        action.values.emplace_back(f_at(c(x1 + i * dx1)));
                        action.coords.emplace_back(c(x1 + i * dx1));
                    }
                    for (int i = 0; i < op.s2.len; i++) {
                        action.values.emplace_back(f_at(c(x2 + i * dx2)));
                        action.coords.emplace_back(c(x2 + i * dx2));
                    }

                    auto net = odd_even_merge(0, op.s1.len, op.s1.len, op.s2.len, op.min_idx, op.max_idx);
                    action.values = apply_network(net, action.values);
                }
            } else if (op.op == Copy) {
                Expr idx1 = x1, idx2 = x2;
                for (int i = 0; i < op.s1.len; i++) {
                    action.values.emplace_back(f_at(c(idx1)));
                    action.coords.emplace_back(c(idx2));
                    idx1 += dx1;
                    idx2 += dx2;
                }
            } else {
                op.print();
                assert(false && "No implementation for op\n");
            }

            actions.emplace_back(std::move(action));
        }

        // Due to Halide restrictions, all our actions need to compute
        // and store the same number of values due to Halide
        // restrictions on scatter/gather, so just pad out the lists
        // by redundantly re-storing the last value to the last coord
        // and assuming it'll be dead-code eliminated.
        for (Action &action : actions) {
            while (action.values.size() < max_leaf * 2) {
                action.values.push_back(action.values.back());
                action.coords.push_back(action.coords.back());
            }
        }

        // Gather the lists into one value and coord expr per action,
        // using Halide's scatter/gather instrinsics.
        std::cout << "Actions: " << actions.size() << "\n";
        vector<Expr> coords, values;
        for (const Action &action : actions) {
            values.emplace_back(gather(action.values));
            coords.emplace_back(scatter(action.coords));
        }

        r.where(mux(r.x, conditions));
        Stage s = f_at(mux(r.x, coords)) = mux(r.x, values);

        //f_at(r_debug) = print_when(x == 0 && y == 0, f_at(r_debug), r_debug, f.name() + " after");

        // x should really be inside the loop over instructions (r.y)
        s.reorder(x, r.x, r.y, y).unroll(r.x);

        if (get_target().has_gpu_feature()) {
            if (f.dimensions() == 3) {
                s.reorder(x, y, r.x, r.y);
            } else {
                s.reorder(v, x, y, r.x, r.y)
                    .unroll(v);
            }
        }

        return s;
    }

    void generate() {
        // Load a square footprint, then apply a sorting network to it
        // in-place and take the median output (could easily be
        // adapted to take a specific output).
        Expr radius = max(1, radius_);
        diameter = radius * 2 + 1;

        Func src_i{"src_i"};
        src_i(x, y, v) = src(x * tw + v, y);

        // Make some scratch space for the 1d sort over columns
        Func sorted_cols{"sorted_cols"};
        sorted_cols(x, y, u, v) = undef(src.type());

        Expr core_width = diameter - (tw - 1);
        Expr core_height = diameter - (th - 1);

        // The top left corner of our region loaded from src
        Expr x_base = x * tw - radius, y_base = y * th - radius;

        // Load a core column into it
        RDom r_load_1d(0, core_height);
        Stage loading_col_input_1d{
            sorted_cols(x, y, r_load_1d, v) = src_i(x, y_base + r_load_1d + th - 1, v)};

        Stage sort_cols = sort_in_place(sorted_cols, sort_cols_program, core_height);

        Func sorted_cols_i{"sorted_cols_i"};
        sorted_cols_i(x, y, u, v) = sorted_cols(x, y, u, v);

        // Reinterleave so that I don't have to change the indexing math below
        Func src_r{"src_r"}, sorted_cols_r{"sorted_cols_r"};
        src_r(x, y) = src_i(x / tw, y, x % tw);
        sorted_cols_r(x, y, u) = sorted_cols_i(x / tw, y, u, x % tw);

        // Make some scratch space in which to merge the sorted cols
        Func merged_cols{"merged_cols"};
        merged_cols(x, y, u) = undef(src.type());

        Expr core_start = 0;

        // Load the core into it first in scanline order.
        RDom r_load_core(0, core_width, 0, core_height);
        Stage loading_core{
            merged_cols(x, y, core_start + r_load_core.y * core_width + r_load_core.x) =
                sorted_cols_r(x_base + r_load_core.x + tw - 1, y, r_load_core.y)};

        Expr extra_cols_start = core_width * core_height;

        RDom r_load_extra_cols(0, tw - 1, 0, core_height);
        Stage loading_extra_cols_on_left{
            merged_cols(x, y, extra_cols_start + r_load_extra_cols.x * core_height + r_load_extra_cols.y) =
                sorted_cols_r(x_base + r_load_extra_cols.x, y, r_load_extra_cols.y)};
        loading_extra_cols_on_left.unroll(r_load_extra_cols.x);

        extra_cols_start += (tw - 1) * core_height;

        Stage loading_extra_cols_on_right{
            merged_cols(x, y, extra_cols_start + r_load_extra_cols.x * core_height + r_load_extra_cols.y) =
                sorted_cols_r(x_base + r_load_extra_cols.x + diameter, y, r_load_extra_cols.y)};
        loading_extra_cols_on_right.unroll(r_load_extra_cols.x);

        Expr extra_rows_start = extra_cols_start + (tw - 1) * core_height;

        RDom r_load_extra_rows(0, th - 1, 0, core_width);
        Stage loading_extra_rows_at_top{
            merged_cols(x, y, extra_rows_start + r_load_extra_rows.y + r_load_extra_rows.x * core_width) =
                src_r(x_base + r_load_extra_rows.y + tw - 1, y_base + r_load_extra_rows.x)};
        loading_extra_rows_at_top.unroll(r_load_extra_rows.x);

        extra_rows_start += (th - 1) * core_width;

        // scratch[scratch_idx + (y + th - 1) * core_w + x] = src[(y + h) * W + x + tw - 1];
        Stage loading_extra_rows_at_bottom{
            merged_cols(x, y, extra_rows_start + r_load_extra_rows.y + r_load_extra_rows.x * core_width) =
                src_r(x_base + r_load_extra_rows.y + tw - 1, y_base + diameter + r_load_extra_rows.x)};
        Expr corners_start = extra_rows_start + (th - 1) * core_width;
        loading_extra_rows_at_bottom.unroll(r_load_extra_rows.x);

        RDom r_load_corners(0, tw - 1, 0, th - 1);

        // scratch[scratch_idx + y * 2 * (tw - 1) + x] = src[y * W + x];
        Stage load_top_left_corners{
            merged_cols(x, y, corners_start + r_load_corners.y * 2 * (tw - 1) + r_load_corners.x) =
                src_r(x_base + r_load_corners.x, y_base + r_load_corners.y)};
        load_top_left_corners.unroll(r_load_corners.x).unroll(r_load_corners.y);

        // scratch[scratch_idx + y * 2 * (tw - 1) + x + tw - 1] = src[y * W + (x + w)];
        Stage load_top_right_corners{
            merged_cols(x, y, corners_start + r_load_corners.y * 2 * (tw - 1) + r_load_corners.x + tw - 1) =

                src_r(x_base + diameter + r_load_corners.x, y_base + r_load_corners.y)};
        load_top_right_corners.unroll(r_load_corners.x).unroll(r_load_corners.y);

        corners_start += 2 * (tw - 1) * (th - 1);

        // scratch[scratch_idx + y * 2 * (tw - 1) + x] = src[y * W + x];
        Stage load_bottom_left_corners{
            merged_cols(x, y, corners_start + r_load_corners.y * 2 * (tw - 1) + r_load_corners.x) =
                src_r(x_base + r_load_corners.x, y_base + diameter + r_load_corners.y)};

        load_bottom_left_corners.unroll(r_load_corners.x).unroll(r_load_corners.y);

        // scratch[scratch_idx + y * 2 * (tw - 1) + x + tw - 1] = src[y * W + (x + w)];
        Stage load_bottom_right_corners{
            merged_cols(x, y, corners_start + r_load_corners.y * 2 * (tw - 1) + r_load_corners.x + tw - 1) =
                src_r(x_base + diameter + r_load_corners.x, y_base + diameter + r_load_corners.y)};

        load_bottom_right_corners.unroll(r_load_corners.x).unroll(r_load_corners.y);

        Expr results_start = corners_start + 2 * (tw - 1) * (th - 1);

        /*
        Stage dump_loaded_data{
            merged_cols(x, y, u) = print_when(x == 0 && y == 0, merged_cols(x, y, u), u, "before")};
        */

        // Run the program
        Stage merge_cols = sort_in_place(merged_cols, merge_cols_program, merge_cols_working_buffer_size);

        /*
        RDom r_dump(0, merge_cols_working_buffer_size);
        Stage dump_state_after_sort{
        merged_cols(x, y, r_dump) = print_when(x == 0 && y == 0, merged_cols(x, y, r_dump), r_dump, "after")};
        */

        // Extract the results

        dst(x, y) = merged_cols(x / tw, y / th, results_start + tw * (y % th) + (x % tw));

        Var xii, yii, xi{"xi"}, yi, xo, yo;

        if (get_target().has_gpu_feature()) {
            // Keep the working memory allocation below 2GB
            Expr memory_per_scanline =
                (dst.dim(0).extent() *
                 merge_cols_working_buffer_size *
                 src.type().bytes());
            Expr strip_height = Expr(2000000000.0) / memory_per_scanline;
            Expr num_strips = cast<int>(ceil(dst.dim(1).extent() / strip_height));
            strip_height = clamp((dst.dim(1).extent() + num_strips - 1) / num_strips,
                                 1, dst.dim(1).extent());

            const int vec = std::min(4, 64 / src.type().bits());

            dst
                .split(y, yo, y, strip_height, TailStrategy::GuardWithIf)
                .gpu_tile(x, y, xi, yi, 32 * vec, 1)
                .vectorize(xi, vec);

            sorted_cols
                .align_storage(x, vec)
                .align_bounds(x, vec)
                .compute_at(dst, yo)
                .gpu_tile(x, y, xi, yi, 32 * vec, 1)
                .vectorize(xi, vec);

            merged_cols
                .align_storage(x, vec)
                .align_bounds(x, vec)
                .compute_at(dst, yo)
                .gpu_tile(x, y, xi, yi, 32 * vec, 1)
                .vectorize(xi, vec);

            for (int i = 0; i < merged_cols.num_update_definitions(); i++) {
                merged_cols.update(i)
                    .gpu_tile(x, y, xi, yi, 32 * vec, 1, TailStrategy::RoundUp)
                    .vectorize(xi, vec);
            }

            for (int i = 0; i < sorted_cols.num_update_definitions(); i++) {
                sorted_cols.update(i)
                    .gpu_tile(x, y, xi, yi, 32 * vec, 1, TailStrategy::RoundUp)
                    .vectorize(xi, vec);
            }
        } else {
            const int vec = natural_vector_size(src.type());

            // We can amortize interpreter overhead by doing multiple
            // vectors of work per instruction. It inflates code size
            // and working memory though. Doesn't seem to be a solid
            // win.
            const int vecs_per_instruction = 1;

            dst
                .align_bounds(x, tw)
                .align_bounds(y, th)
                .split(y, y, yi, 32)
                .split(yi, yi, yii, th)
                .split(x, x, xi, tw * vec * vecs_per_instruction)
                .split(xi, xi, xii, tw)
                .reorder(xii, yii, xi, x, yi, y)
                .unroll(xii)
                .unroll(yii)
                .vectorize(xi, vec)
                .parallel(y);

            sorted_cols_i.compute_at(dst, yi)
                .store_in(MemoryType::Stack)
                .align_storage(x, vec)
                .split(x, xo, xi, vec)
                .reorder(v, xi, u, xo, y)
                .vectorize(xi, vec * vecs_per_instruction)
                .unroll(xi)
                .unroll(v);

            sorted_cols.compute_at(sorted_cols_i, xo)
                .store_in(MemoryType::Stack)
                .align_storage(x, vec)
                .vectorize(x, vec * vecs_per_instruction)
                .unroll(x);

            if (tw > 1) {
                src.in(src_i)
                    .compute_at(src_i, x)
                    .vectorize(_0);

                src_i.store_at(dst, y)
                    .compute_at(dst, yi)
                    .reorder(v, x, y)
                    .vectorize(x, vec)
                    .unroll(v);
            }

            loading_col_input_1d
                .reorder(v, x, r_load_1d, y)
                .unroll(v)
                .vectorize(x, vec, TailStrategy::GuardWithIf);

            sort_cols
                .vectorize(x, vec, TailStrategy::RoundUp)
                .unroll(x);

            merged_cols.compute_at(dst, x)
                .store_in(MemoryType::Stack)
                .vectorize(x, vec);

            for (int i = 0; i < merged_cols.num_update_definitions(); i++) {
                merged_cols
                    .update(i)
                    .vectorize(x, vec)
                    .unroll(x);
            }
        }

        // Require that the output is aligned to the tile size and at least tw vectors wide.
        dst
            .dim(0)
            .set_bounds(0, (dst.dim(0).extent() / tw) * tw)
            .dim(1)
            .set_bounds(0, (dst.dim(1).extent() / th) * th);

        const int fields_per_instruction = sizeof(Instruction) / sizeof(Instruction::op);

        sort_cols_program
            .dim(0)
            .set_bounds(0, fields_per_instruction)
            .dim(1)
            .set_stride(fields_per_instruction);

        merge_cols_program
            .dim(0)
            .set_bounds(0, fields_per_instruction)
            .dim(1)
            .set_stride(fields_per_instruction);
    }
};

HALIDE_REGISTER_GENERATOR(DynamicMedianFilter, dynamic_median_filter);
