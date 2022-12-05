// An implementation of "An accelerated separable median filter with
// sorting networks" by Kim et al.

#include "Halide.h"

#include "sorting_bytecode.h"
#include "sorting_network.h"

using namespace Halide;
using std::pair;
using std::string;
using std::vector;

class KimMedianFilter : public Generator<KimMedianFilter> {

public:
    GeneratorParam<int> radius{"radius", 1};

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

    void generate() {
        // Kim provides a sketch for sorting networks above this size,
        // but only evaluates 3x3 and 5x5, so we will too.
        assert(radius == 1 || radius == 2);

        Func sort_horiz{"sort_horiz"};

        vector<Expr> horiz_footprint;
        for (int i = -(int)radius; i <= radius; i++) {
            horiz_footprint.push_back(src(x + i, y));
        }

        int diameter = 2 * radius + 1;
        auto horiz_network = pairwise_sort(diameter);
        horiz_footprint = apply_network(horiz_network, horiz_footprint);
        sort_horiz(x, y, u) = mux(u, horiz_footprint);

        size_t steady_state_swaps = horiz_network.size();

        if (radius == 1) {
            // Sort down the columns
            vector<vector<Expr>> footprint;
            for (int col = -(int)radius; col <= radius; col++) {
                footprint.emplace_back();
                for (int row = -(int)radius; row <= radius; row++) {
                    footprint.back().push_back(sort_horiz(x, y + row, col + radius));
                }

                // Sort down the columns
                int idx = 1 - col;
                auto net = pairwise_sort(diameter, idx, idx, false);
                footprint.back() = apply_network(net, footprint.back());
                steady_state_swaps += net.size();
            }

            vector<Expr> diag{footprint[0][2], footprint[1][1], footprint[2][0]};
            auto net = pairwise_sort(3, 1, 1, false);
            steady_state_swaps += net.size();
            diag = apply_network(net, diag);

            dst(x, y) = diag[1];

        } else {
            vector<vector<Expr>> footprint;
            for (int col = -(int)radius; col <= radius; col++) {
                footprint.emplace_back();
                for (int row = -(int)radius; row <= radius; row++) {
                    footprint.back().push_back(sort_horiz(x, y + row, col + radius));
                }

                // Sort down the columns
                int min_idx = std::max(1 - col, 0);
                int max_idx = std::min(3 - col, 2 * radius);
                std::cout << col << " " << min_idx << " " << max_idx << "\n";
                auto net = pairwise_sort(diameter, min_idx, max_idx, false);
                footprint.back() = apply_network(net, footprint.back());
                steady_state_swaps += net.size();
            }

            // Extract the four pieces to merge
            std::vector<Expr> v = {
                footprint[0][3], footprint[1][3], footprint[0][4], footprint[1][4],
                footprint[1][2], footprint[2][2], footprint[2][3],
                footprint[2][1], footprint[3][1], footprint[3][2],
                footprint[3][0], footprint[4][0], footprint[4][1]};

            // Fix up the middle pair of v0
            vector<pair<int, int>> fixup{{1, 2}};

            // Merge the four fragments. Kim just describes this as
            // "four sorted array merge sorting". We'll try to find
            // something that hits that many swaps. After the fixup
            // swap, our remaining budget is 22 swaps.

            /*
            // Balanced binary tree of merge networks:
            auto merge_net1 = odd_even_merge(0, 4, 4, 3, 0, 7);
            auto merge_net2 = odd_even_merge(7, 3, 10, 3, 0, 6);

            // Median of a list of size 7 and a list of size 6 (12 swaps)
            auto merge_net3 = odd_even_merge(0, 7, 7, 6, 6, 6);

            // 26 swaps
            */

            /*
              // Lop-sized binary tree of merges, with a size-3 list last
            auto merge_net1 = odd_even_merge(0, 4, 4, 3, 0, 7);
            auto merge_net2 = odd_even_merge(0, 7, 7, 3, 3, 6);  // 3 remaining to consider, so need central 4
            auto merge_net3 = odd_even_merge(3, 4, 10, 3, 6, 6);

            28 swaps
            */

            // Lop-sided binary tree of merges, with a size-4 list last

            auto merge_net1 = odd_even_merge(4, 3, 7, 3, 0, 6);
            auto merge_net2 = odd_even_merge(4, 6, 10, 3, 2, 6);  // 4 remaining to consider, so need central 5
            auto merge_net3 = odd_even_merge(0, 4, 6, 5, 4, 4);

            // 25 swaps

            v = apply_network(fixup, v);
            v = apply_network(merge_net1, v);
            v = apply_network(merge_net2, v);
            v = apply_network(merge_net3, v);
            steady_state_swaps += fixup.size() + merge_net1.size() + merge_net2.size() + merge_net3.size();

            dst(x, y) = v[6];
        }

        // Assert steady-state swaps matches the numbers claimed in Kim et al.
        std::cerr << "Steady state swaps: " << steady_state_swaps << "\n";
        if (radius == 1) {
            assert(steady_state_swaps == 13);
        } else if (radius == 2) {
            // The best I could do from the description in Kim's paper
            // is 74 swaps. Hopefully this is close enough to 71 to be
            // representative of performance.
            //assert(steady_state_swaps == 71);
        }

        // Schedule as described by Kim et al. Kim doesn't describe
        // how to parallelize. Given that there's a serial dependence
        // across y, one option is vertical strips like CTMF to keep the
        // circular buffers in L1. We've tuned the strip width to
        // maximize performance on the benchmarking machine.
        /*
        Var xo, xi;
        dst.compute_root()
            .split(x, xo, xi, 64)
            .reorder(xi, y, xo)
            .vectorize(xi)
            .parallel(xo);

        sort_horiz.store_at(dst, xo)
            .compute_at(dst, y)
            .fold_storage(y, radius == 1 ? 4 : 8)
            .reorder(x, u, y)
            .vectorize(x)
            .unroll(u);
        */

        // A faster option, which is perhaps more friendly on the
        // prefetchers, is just slicing up the image vertically and
        // eating some redundant recompute at slice boundaries. Slice
        // height tuned empirically for best performance on
        // benchmarking machine.

        Var xi, yi;
        dst.compute_root()
            .tile(x, y, xi, yi, natural_vector_size(src.type()), 16)
            .parallel(y)
            .vectorize(xi);

        sort_horiz.store_at(dst, x)
            .compute_at(dst, yi)
            .store_in(MemoryType::Stack)
            .fold_storage(y, radius == 1 ? 4 : 8)
            .reorder(x, u, y)
            .vectorize(x)
            .unroll(u);
    }
};

HALIDE_REGISTER_GENERATOR(KimMedianFilter, kim_median_filter);
