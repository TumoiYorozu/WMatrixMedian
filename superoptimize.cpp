#include "integer_set.h"
#include "sorting_network.h"
#include <algorithm>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using std::pair;
using std::vector;

// A fairly simple pruning method that just tries to cut or fuse
// existing links
template<int max_n>
void prune_network(int num_inputs,
                   vector<pair<int, int>> &network,
                   IntegerSet<max_n> set,
                   const IntegerSet<max_n> &target) {

    vector<pair<int, int>> pruned;
    for (size_t i = 0; i < network.size(); i++) {
        // Is link i really necessary? Skip it and run the rest of the network.
        {
            IntegerSet<max_n> set2 = set;
            for (size_t j = i + 1; j < network.size(); j++) {
                set2.sort(network[j].first, network[j].second);
            }
            if (set2.is_subset_of(target)) {
                // Don't need this one!
                continue;
            }
        }

        // Ok, we can't prune it, but maybe we can fuse it with a later link
        {
            size_t j;
            for (j = i + 1; j < network.size(); j++) {
                if (network[j].first == network[i].second ||
                    network[j].second == network[i].first) {
                    break;
                }
            }
            if (j < network.size()) {
                // We have a fusion candidate. We'll try both doing
                // the fused pair at position i, and also at position
                // j.
                pair<int, int> p{network[i].first, network[j].second};
                if (p.first == p.second) {
                    p.first = network[j].first;
                    p.second = network[i].second;
                }
                IntegerSet<max_n> set2 = set;
                IntegerSet<max_n> set3 = set;
                set2.sort(p.first, p.second);
                for (size_t k = i + 1; k < j; k++) {
                    set2.sort(network[k].first, network[k].second);
                    set3.sort(network[k].first, network[k].second);
                }
                set3.sort(p.first, p.second);
                for (size_t k = j + 1; k < network.size(); k++) {
                    set2.sort(network[k].first, network[k].second);
                    set3.sort(network[k].first, network[k].second);
                }
                if (set2.is_subset_of(target)) {
                    // Run the fused pair now
                    set.sort(p.first, p.second);
                    pruned.push_back(p);
                    network.erase(network.begin() + j);
                    continue;
                }
                if (set3.is_subset_of(target)) {
                    // Skip this link, run the fused pair later
                    network[j] = p;
                    continue;
                }
            }
        }

        // We need this one
        set.sort(network[i].first, network[i].second);
        pruned.push_back(network[i]);
    }

    network.swap(pruned);
}

template<int max_n>
struct State {
    int num_inputs = 0;

    // Which inputs have been used (a bitmask)
    uint32_t used_inputs = 0;

    pair<int, int> link;

    State<max_n> *parent = nullptr;

    int num_links = 0;

    uint64_t cost = 0;

    int num_children = 0;

    // Possible wire states after running these links
    IntegerSet<max_n> current;

    int current_size;

    void enqueue_children(vector<State *> &pending, IntegerSet<max_n> &target) {
        for (int i = num_inputs - 1; i >= 0; i--) {
            for (int j = num_inputs - 1; j > i; j--) {
                if (current.is_sorted(i, j)) {
                    // This link would do nothing
                    continue;
                }
                num_children++;
                State *child = new State(*this);
                child->current.sort(i, j);
                child->current_size = (int)child->current.size();
                child->link.first = i;
                child->link.second = j;
                child->used_inputs |= (1 << i) | (1 << j);
                child->parent = this;
                child->num_links = num_links + 1;

                // Count the number of possible states we have that
                // aren't in the target.
                uint64_t penalty = (int)child->current.count_elements_in_exclusion(target);

                child->cost = penalty + child->num_links;
                pending.emplace_back(child);
                std::push_heap(pending.begin(), pending.end(),
                               [](const State<max_n> *a, const State<max_n> *b) {
                                   return a->cost < b->cost;
                               });
            }
        }
    }

    int min_additional_links_required_to_reach(const IntegerSet<max_n> &target) const {
        // Each additional link can at most reduce the set size by a
        // factor of 2, so we need one more link for each additional
        // leading zero the target size has compared to the current
        // size.
        size_t target_size = target.size();
        size_t current_size = current.size();
        return __builtin_clzll(target_size) - __builtin_clzll(current_size);
    }

    bool done(IntegerSet<max_n> &target) {
        return current.is_subset_of(target);
    }

    State(int num_inputs)
        : num_inputs(num_inputs), current(num_inputs) {
    }

    void dump() const {
        if (parent) {
            parent->dump();
            printf("%d %d\n", link.first, link.second);
        }
    }

    ~State() {
        assert(num_children == 0);
        if (parent) {
            parent->num_children--;
            if (parent->num_children == 0) {
                delete parent;
            }
        }
    }
};

template<int max_n>
vector<pair<int, int>> find_path(int n, IntegerSet<max_n> src, IntegerSet<max_n> dst, int max_depth) {

    // Instead of searching until we find a subset of the single
    // target set above, also enumerate backwards from the goal to
    // make a larger target to hit. This is a sort of poor man's
    // bidirectional search. Note that this speeds up finding a
    // solution at a given size, but doesn't speed-up proving there
    // are no solutions at a given size, which is the slow part if
    // we're trying to find the optimal solution.

    struct GoalState {
        // What set of activations will get you to the end from here
        IntegerSet<max_n> set;

        // What the links are that get there, in reverse order
        vector<pair<int, int>> links;
    };

    std::unordered_map<uint64_t, GoalState> goal_states;

    GoalState final_target{dst, {}};
    goal_states.emplace(dst.hash(), final_target);

    std::unordered_map<uint64_t, GoalState> prev = goal_states;
    for (int i = 1; i < 5; i++) {
        if (goal_states.size() > 10000) {
            // That's probably as much memory as we want to use
            break;
        }
        std::unordered_map<uint64_t, GoalState> next;
        for (auto &p : prev) {
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (p.second.set.is_sorted(i, j)) {
                        GoalState child{p.second};
                        child.set.unsort(i, j);
                        child.links.emplace_back(i, j);
                        uint64_t hash = child.set.hash();
                        if (goal_states.count(hash) == 0) {
                            // Don't clobber a goal state with the
                            // same hash that's closer to the target.
                            next.emplace(hash, std::move(child));
                        }
                    }
                }
            }
        }
        goal_states.insert(next.begin(), next.end());
        prev.swap(next);
    }

    printf("Target area has %d states\n", (int)goal_states.size());

    // Set up our initial state
    State<max_n> *next = new State<max_n>{n};

    next->current = src;

    struct StateHash {
        size_t operator()(const State<max_n> *s) const {
            return s->current.hash() ^ s->num_links;
        }
    };

    struct StateEq {
        size_t operator()(const State<max_n> *s1, const State<max_n> *s2) const {
            return s1->current == s2->current &&
                   s1->num_links == s2->num_links;
        }
    };
    std::unordered_set<State<max_n> *, StateHash, StateEq> visited;

    // Our priority queue of states to visit next
    vector<State<max_n> *> pending;
    State<max_n> *best = nullptr;
    vector<pair<int, int>> goal_state_links;

    constexpr int bloom_bits = 20;
    IntegerSet<bloom_bits> bloom1(bloom_bits), bloom2(bloom_bits), bloom3(bloom_bits), bloom4(bloom_bits);

    auto bloom_check = [&](const State<max_n> *s) {
        uint64_t h = StateHash()(s);
        uint64_t mask = (1 << bloom_bits) - 1;
        return (bloom1.contains(h & mask) &&
                bloom2.contains((h >> 16) & mask) &&
                bloom3.contains((h >> 32) & mask) &&
                bloom4.contains((h >> 48) & mask));
    };

    auto bloom_set = [&](const State<max_n> *s) {
        uint64_t h = StateHash()(s);
        uint64_t mask = (1 << bloom_bits) - 1;
        bloom1.set(h & mask);
        bloom2.set((h >> 16) & mask);
        bloom3.set((h >> 32) & mask);
        bloom4.set((h >> 48) & mask);
    };

    const bool approximate = (n > 7);
    const int beam_size = 1024;

    while (1) {
        //printf("%d %d %d\n", next->num_links, max_depth, (int)visited.size());
        if (next->num_links < max_depth &&
            (approximate || !visited.count(next)) &&
            (!approximate || !bloom_check(next))) {
            // I've never seen this state before.

            int extra_links = 0;

            // Check if it's a subset of the target state, which would
            // mean that the activations this sorting network produces
            // sort the input at least as well as our requirement.
            bool done = next->done(dst);

            vector<pair<int, int>> candidate_goal_state_links;
            if (!done) {
                // Maybe it hashes to one of our precomputed states
                // for which we already know the shortest pathway to
                // the goal state.
                auto it = goal_states.find(next->current.hash());
                if (it != goal_states.end() && next->done(it->second.set)) {
                    done = true;
                    extra_links = (int)it->second.links.size();
                    candidate_goal_state_links = it->second.links;
                }
            }

            int depth = next->num_links + extra_links - 1;
            if (done && depth < max_depth) {
                printf("Found a solution at depth %d + %d (max = %d)\n", next->num_links, extra_links, max_depth);
                // Update the depth limit to see if we can find a
                // better solution.
                max_depth = depth;
                best = next;
                goal_state_links.swap(candidate_goal_state_links);
            } else if (next->num_links + next->min_additional_links_required_to_reach(dst) < max_depth) {
                // If adding another link would not exceed the depth
                // limit, add states representing sorting networks
                // consisting of this network plus one more swap
                // operation.
                next->enqueue_children(pending, dst);
            }

            if (!approximate) {
                visited.insert(next);
            } else {
                bloom_set(next);
            }
        }

        if (next->num_children == 0 && next != best) {
            // We're done with this node and no child states want to
            // keep it as a parent state.
            delete next;
        }

        if (pending.empty()) {
            printf("No more solutions\n");
            break;
        }

        // Get the top priority state and repeat.
        next = pending.back();
        pending.pop_back();

        if (approximate && pending.size() > beam_size) {
            decltype(pending) tmp(pending.end() - beam_size, pending.end());
            tmp.swap(pending);
        }
    }

    if (!best) {
        printf("No solutions\n");
        return {};
    }

    vector<pair<int, int>> links = goal_state_links;
    while (best->parent) {
        links.push_back(best->link);
        best = best->parent;
    }
    std::reverse(links.begin(), links.end());

    return links;
}

template<int max_n>
int search(int n, int min_idx, int max_idx, int known_orderings) {

    // This is going to be a textbook branch and bound search over
    // sorting networks, with some pruning and a little bit of
    // bidirectionalily.

    assert(n <= max_n || max_n == -1);

    // Define our goal state, which is the set of possible outputs to
    // the network that we would consider correct.
    IntegerSet<max_n> target(n);
    target.set_all();

    // We care about the result within [min_idx, max_idx]
    for (int i = 0; i < min_idx; i++) {
        target.remove_all_where_not_sorted(i, min_idx);
    }
    for (int i = min_idx; i < max_idx; i++) {
        target.remove_all_where_not_sorted(i, i + 1);
    }
    for (int i = max_idx + 1; i < n; i++) {
        target.remove_all_where_not_sorted(max_idx, i);
    }

    printf("Acceptable outputs:\n");
    target.dump_binary();

    // Assume the inputs could be anything
    IntegerSet<max_n> initial(n);
    initial.set_all();

    // A hash set of states already visited
    if (known_orderings < 0) {
        // We know we have sorted pairs
        for (int i = 0; i < n - 1; i += 2) {
            initial.remove_all_where_not_sorted(i, i + 1);
        }
        if (known_orderings == -2) {
            // And the odds and evens are sorted too
            for (int i = 0; i < n - 2; i += 2) {
                initial.remove_all_where_not_sorted(i, i + 2);
            }
            for (int i = 1; i < n - 2; i += 2) {
                initial.remove_all_where_not_sorted(i, i + 2);
            }
        }
    } else if (known_orderings == 0) {
        // We know nothing
    } else {
        // It's a merge of two sorted lists
        for (int i = 0; i < n - 1; i++) {
            if (i + 1 == known_orderings) continue;
            initial.remove_all_where_not_sorted(i, i + 1);
        }
    }

    // The depth limit. We'll reduce it each time we find a
    // solution.
    vector<pair<int, int>> reference_solution;
    if (known_orderings == -1) {
        // Try to do better than a pairwise selection network
        reference_solution = pairwise_sort(n, min_idx, max_idx, true, false);
    } else if (known_orderings == -2) {
        reference_solution = pairwise_merge(n, min_idx, max_idx);
    } else if (known_orderings == 0) {
        reference_solution = pairwise_sort(n, min_idx, max_idx, false, false);
    } else {
        // Try to do better than an even odd merge network
        reference_solution = odd_even_merge(0, known_orderings, known_orderings, n - known_orderings, min_idx, max_idx);
    }

    size_t original_size = reference_solution.size();

    // Start by pruning the reference solution
    prune_network(n, reference_solution, initial, target);

    // TODO: sometimes flipping the reference solution gets a better
    // end result after pruning. Not sure how to work that in.

    // Make sure we believe the reference solution works
    {
        IntegerSet<max_n> s = initial;
        for (auto l : reference_solution) {
            s.sort(l.first, l.second);
        }
        assert(s.is_subset_of(target) && "Reference solution doesn't seem to work!");
    }

    int max_depth = (int)original_size;
    printf("Current solution has size %d (%d after pruning)\n", max_depth, (int)reference_solution.size());
    const int window_size = 7;
    vector<pair<int, int>> solution;
    if (n <= 7) {
        solution = find_path(n, initial, target, max_depth);
    } else {
        // Super-optimization of the entire thing would be too
        // slow. Super-optimize overlapping windows instead (giving up on optimality).
        printf("Super-optimizing overlapping windows. Solution will not be optimal.\n");

        auto may_swap = [&](int i, int j) {
            const auto &pi = reference_solution[i];
            const auto &pj = reference_solution[j];
            return (pi.first != pj.first &&
                    pi.second != pj.first &&
                    pi.first != pj.second &&
                    pi.second != pj.second);
        };

        // Topologically sort the network to make overlapping windows more meaningful
        for (int i = (int)reference_solution.size() - 1; i >= 0; i--) {
            for (int j = i + 1; j < (int)reference_solution.size() && may_swap(j - 1, j); j++) {
                std::swap(reference_solution[j - 1], reference_solution[j]);
            }
        }
        for (int i = 0; i < (int)reference_solution.size(); i++) {
            for (int j = i - 1; j >= 0 && may_swap(j + 1, j); j--) {
                std::swap(reference_solution[j + 1], reference_solution[j]);
            }
        }

        for (auto p : reference_solution) {
            printf(" %d %d\n", p.first, p.second);
        }

        bool any_success;
        do {
            IntegerSet<max_n> current = initial;
            any_success = false;
            for (int i = 0; i + window_size <= (int)reference_solution.size(); i++) {
                printf("Superoptimizing range %d ... %d (inclusive) \n", i, i + window_size - 1);

                // Run the tail end of the network backwards
                IntegerSet<max_n> window_end_state = target;
                for (int j = (int)reference_solution.size() - 1; j >= i + window_size; j--) {
                    window_end_state.unsort(reference_solution[j].first, reference_solution[j].second);
                }

                // TODO: trim off all wires that haven't been touched
                // yet in the network. They can't possibly be
                // relevant.

                // Find a better set of links just for this stretch
                auto links = find_path(n, current, window_end_state, window_size);

                if (!links.empty() && links.size() < window_size) {
                    printf("Found a simplification in this window %d -> %d\n", window_size, (int)links.size());

                    printf("Before:\n");
                    for (int j = 0; j < window_size; j++) {
                        printf(" %d %d\n",
                               reference_solution[i + j].first,
                               reference_solution[i + j].second);
                    }
                    printf("After:\n");
                    for (auto l : links) {
                        printf(" %d %d\n", l.first, l.second);
                    }

                    // Double check the old and new network segments
                    // worked (At one point there was a bug in unsort
                    // which meant they didn't).
                    if (1) {
                        IntegerSet<max_n> test1 = current, test2 = current;
                        for (int j = 0; j < window_size; j++) {
                            test1.sort(reference_solution[i + j].first, reference_solution[i + j].second);
                        }
                        for (auto l : links) {
                            test2.sort(l.first, l.second);
                        }
                        assert(test1.is_subset_of(window_end_state));
                        assert(test2.is_subset_of(window_end_state));
                    }

                    // Huzzah. Graft it in
                    vector<pair<int, int>> new_network(reference_solution.begin(), reference_solution.begin() + i);
                    new_network.insert(new_network.end(),
                                       links.begin(),
                                       links.end());
                    new_network.insert(new_network.end(),
                                       reference_solution.begin() + i + window_size,
                                       reference_solution.end());
                    reference_solution.swap(new_network);
                    any_success = true;
                    break;
                } else {
                    printf("Found no simplification in this window\n");
                }

                // Run the network forwards up to state i to prepare for the next iteration
                current.sort(reference_solution[i].first, reference_solution[i].second);
            }
        } while (any_success);
        if ((int)reference_solution.size() < max_depth) {
            solution = reference_solution;
        }
    }

    if (solution.empty()) {
        return 0;
    }

    if (solution.size() < original_size) {
        printf("Solution found with %d links:\n", (int)solution.size());
        const char *prefix = "";
        printf("optimal[{%d, %d, %d, %d}] = {",
               n, min_idx, max_idx, known_orderings);
        for (auto p : solution) {
            printf("%s{%d, %d}", prefix, p.first, p.second);
            prefix = ", ";
        }
        printf("};\n");
    } else {
        printf("Reference solution already optimal:\n"
               "already_optimal.insert({%d, %d, %d, %d});  // size %d\n",
               n, min_idx, max_idx, known_orderings, (int)original_size);
    }

    return 0;
}

int main(int argc, char **argv) {

    if (argc != 5) {
        printf("Usage: superoptimize num_inputs min_idx max_idx known_orderings\n"
               "\n"
               "If the input is entirely unsorted, set known_ordering to 0\n"
               "If the each pair in the input is sorted, set known_ordering to -1\n"
               "If the each pair in the input is sorted and the even and odd \n"
               "elements are also sorted, set known_ordering to -2\n"
               "If the first n elements are sorted internally, and the remaining\n"
               "elements are also sorted internally (i.e. this is a merge network),\n"
               "set known_ordering to n\n"
               "\n"
               "Examples:\n"
               "--------\n"
               "\n"
               "Sort 7 elements entirely:\n"
               "\n"
               "superoptimize 7 0 6 0\n"
               "\n"
               "Get the median of 9 elements:\n"
               "\n"
               "superoptimize 9 4 4 0\n"
               "\n"
               "Get the median of 9 elements where we know the first four pairs of\n"
               "elements are already in order:\n"
               "\n"
               "superoptimize 9 4 4 -1\n"
               "\n"
               "Get the median of 9 elements where the first 6 elements are sorted\n"
               "and so are the last 3:\n"
               "\n"
               "superoptimize 9 4 4 6\n"
               "\n"
               "Get the top-3 of 12 elements where each pair of elements is already\n"
               "in order:\n"
               "\n"
               "superoptimize 12 9 11 -1\n"
               "\n");
        return 0;
    }

    const int n = atoi(argv[1]);
    const int min_idx = atoi(argv[2]);
    const int max_idx = atoi(argv[3]);
    const int known_orderings = atoi(argv[4]);

    // Early out in degenerate cases
    if (min_idx == 1 || max_idx == n - 2) {
        printf("Degenerate case\n");
        return 0;
    }

    if (n <= 6) {
        return search<6>(n, min_idx, max_idx, known_orderings);
    } else if (n <= 7) {
        return search<7>(n, min_idx, max_idx, known_orderings);
    } else if (n <= 8) {
        return search<8>(n, min_idx, max_idx, known_orderings);
    } else if (n <= 9) {
        return search<9>(n, min_idx, max_idx, known_orderings);
    } else if (n <= 10) {
        return search<10>(n, min_idx, max_idx, known_orderings);
    } else if (n <= 11) {
        return search<11>(n, min_idx, max_idx, known_orderings);
    } else if (n <= 12) {
        return search<12>(n, min_idx, max_idx, known_orderings);
    } else {
        return search<-1>(n, min_idx, max_idx, known_orderings);
    }
}
