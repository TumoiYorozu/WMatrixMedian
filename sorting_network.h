#ifndef SORTING_NETWORK_H
#define SORTING_NETWORK_H

// Generate primitive sorting networks in terms of single swaps

#include <vector>

// Batcher's odd-even merge network, for merging two sorted
// lists. Pruned to only be correct from min_idx to max_idx inclusive
// (counting from 0 to a_size + b_size - 1 inclusive)
std::vector<std::pair<int, int>> odd_even_merge(int a_start, int a_size,
                                                int b_start, int b_size,
                                                int min_idx, int max_idx);

// Parberry's pairwise sort, pruned to be correct from min_idx to max_idx inclusive.
std::vector<std::pair<int, int>> pairwise_sort(int size, int min_idx, int max_idx,
                                               bool sorted_pairwise_already,
                                               bool use_superoptimized_leaves = true);

// A full pairwise sort.
std::vector<std::pair<int, int>> pairwise_sort(int size);

// Just the merge step from a pairwise sorting network.
std::vector<std::pair<int, int>> pairwise_merge(int size, int min_idx, int max_idx);

#endif
