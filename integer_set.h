#ifndef INTEGER_SET_H
#define INTEGER_SET_H

#include <array>
#include <assert.h>
#include <iostream>
#include <stdint.h>
#include <vector>

template<int max_n>
struct IntegerSetStorage {
    uint64_t storage[1ULL << std::max(0, (max_n - 6))];
    size_t sz = 0;
    IntegerSetStorage(int n, uint64_t init)
        : sz(n) {
        for (size_t i = 0; i < sz; i++) {
            storage[i] = init;
        }
    }
    size_t size() const {
        return sz;
    }
    uint64_t &operator[](int idx) {
        return storage[idx];
    }
    uint64_t operator[](int idx) const {
        return storage[idx];
    }
};

template<>
struct IntegerSetStorage<-1> : public std::vector<uint64_t> {
    vector<uint64_t> storage;
    IntegerSetStorage(int n, uint64_t init)
        : storage(n, init) {
    }
    size_t size() const {
        return storage.size();
    }
    uint64_t &operator[](int idx) {
        return storage[idx];
    }
    uint64_t operator[](int idx) const {
        return storage[idx];
    }
};

// A set of integers (uint64_ts)
template<int max_n = -1>  // -1 means unbounded. Uses a std::vector for storage in this case
class IntegerSet {
public:
    int n;
    IntegerSetStorage<max_n> storage;

    // Make an integer set capable of storing the integers 0 through 2^n - 1
    IntegerSet(int n)
        : n(n), storage((1ULL << std::max(0, (n - 6))), 0) {
        assert(n <= 64);
    }

    // Add an integer to the set
    void set(uint64_t id) {
        storage[id >> 6] |= ((uint64_t)1 << (id & 63));
    }

    // Add an bunch of integers to the set
    void set(const IntegerSet<max_n> &other) {
        for (size_t i = 0; i < storage.size(); i++) {
            storage[i] |= other.storage[i];
        }
    }

    // Remove an integer from the set
    void reset(uint64_t id) {
        storage[id >> 6] &= ~((uint64_t)1 << (id & 63));
    }

    // Check if an integer is in the set
    bool contains(uint64_t id) const {
        return storage[id >> 6] & ((uint64_t)1 << (id & 63));
    }

    // Make the set contain all possible integers up to the size limit.
    void set_all() {
        if (n < 6) {
            storage[0] = (1ULL << (1 << n)) - 1;
        } else {
            for (size_t i = 0; i < storage.size(); i++) {
                storage[i] = (uint64_t)(-1);
            }
        }
    }

    // Call a function on every integer in the set
    template<typename Callable>
    void for_each(Callable c) {
        if (n >= 6) {
            for (size_t i = 0; i < storage.size(); i++) {
                uint64_t bin = storage[i];
                if (!bin) {
                    continue;
                }
                for (int b = 0; b < 64; b++) {
                    if (bin & (1ULL << b)) {
                        uint64_t id = (i << 6) + b;
                        c(id);
                    }
                }
            }
        } else {
            // The set uses less than 64 bits
            uint64_t bin = storage[0];
            for (uint64_t b = 0; b < (1ULL << n); b++) {
                if (bin & (1ULL << b)) {
                    c(b);
                }
            }
        }
    }

    template<typename Callable>
    void for_each_in_exclusion(Callable c, const IntegerSet &other) {
        if (n >= 6) {
            for (size_t i = 0; i < storage.size(); i++) {
                uint64_t bin = storage[i] & ~other.storage[i];
                if (!bin) {
                    continue;
                }
                for (int b = 0; b < 64; b++) {
                    if (bin & (1ULL << b)) {
                        uint64_t id = (i << 6) + b;
                        c(id);
                    }
                }
            }
        } else {
            // The set uses less than 64 bits
            uint64_t bin = storage[0] & other.storage[0];
            for (uint64_t b = 0; b < (1ULL << n); b++) {
                if (bin & (1ULL << b)) {
                    c(b);
                }
            }
        }
    }

    template<typename Callable>
    void for_each_in_union(Callable c, const IntegerSet &other) {
        if (n >= 6) {
            for (size_t i = 0; i < storage.size(); i++) {
                uint64_t bin = storage[i] | other.storage[i];
                if (!bin) {
                    continue;
                }
                for (int b = 0; b < 64; b++) {
                    if (bin & (1ULL << b)) {
                        uint64_t id = (i << 6) + b;
                        c(id);
                    }
                }
            }
        } else {
            // The set uses less than 64 bits
            uint64_t bin = storage[0] & other.storage[0];
            for (uint64_t b = 0; b < (1ULL << n); b++) {
                if (bin & (1ULL << b)) {
                    c(b);
                }
            }
        }
    }

    template<typename Callable>
    void for_each_in_intersection(Callable c, const IntegerSet &other) {
        if (n >= 6) {
            for (size_t i = 0; i < storage.size(); i++) {
                uint64_t bin = storage[i] & other.storage[i];
                if (!bin) {
                    continue;
                }
                for (int b = 0; b < 64; b++) {
                    if (bin & (1ULL << b)) {
                        uint64_t id = (i << 6) + b;
                        c(id);
                    }
                }
            }
        } else {
            // The set uses less than 64 bits
            uint64_t bin = storage[0] & other.storage[0];
            for (uint64_t b = 0; b < (1ULL << n); b++) {
                if (bin & (1ULL << b)) {
                    c(b);
                }
            }
        }
    }

    // For every integer in the set where bit1 is set and bit2 is
    // unset, remove it, and insert the same integer with bit1 unset
    // and bit2 set
    void sort(int bit1, int bit2) {
        assert(bit2 > bit1);
        int dbit = bit2 - bit1;
        uint64_t mask1 = (1ULL << bit1);
        uint64_t mask2 = (1ULL << bit2);
        uint64_t mask12 = ~(mask1 | mask2);
        for_each([=](uint64_t i) {
            reset(i);
            uint64_t b1 = i & mask1;
            uint64_t b2 = i & mask2;
            // bit1 and bit2 in position 1
            uint64_t and12 = (b1 & (b2 >> dbit));
            // bit1 or bit2 in position 2
            uint64_t or12 = ((b1 << dbit) | b2);
            uint64_t j = (i & mask12) | and12 | or12;
            set(j);
        });
    }

    // Find the largest set such that sorting bit1 and bit2 produces a
    // subset of the original set. unsorting then sorting again
    // produces a subset of the original.
    void unsort(int bit1, int bit2) {

        // We're never going to be able to hit these states by sorting
        // bit1 and bit2, so remove them.
        remove_all_where_not_sorted(bit1, bit2);

        // In the remaining states, wherever bit1 and bit2 are sorted,
        // we can add bit1 and bit2 reverse-sorted to the set as well.
        assert(bit2 > bit1);
        int dbit = bit2 - bit1;
        uint64_t mask1 = (1ULL << bit1);
        uint64_t mask2 = (1ULL << bit2);
        uint64_t mask12 = ~(mask1 | mask2);
        for_each([=](uint64_t i) {
            uint64_t b1 = i & mask1;
            uint64_t b2 = i & mask2;
            // bit1 or bit2 in position 1
            uint64_t or12 = (b1 | (b2 >> dbit));
            // bit1 and bit2 in position 2
            uint64_t and12 = ((b1 << dbit) & b2);
            uint64_t j = (i & mask12) | or12 | and12;
            set(j);
        });
    }

    // Is every integer in the set a bunch of zeros followed by a
    // bunch of ones from LSB to MSB.
    bool is_sorted() {
        bool check = true;
        for_each([&](uint64_t i) {
            // Bit pattern must be a bunch of zeros followed by a bunch of ones
            // First fill all unused high bits with ones.
            i |= ~((1ULL << n) - 1);
            // Then check the trailing zero bits plus the leading one bits adds up to 64
            check = check && (__builtin_ctzll(i) + __builtin_clzll(~i) == 64);
        });
        return check;
    }

    // Remove all the integers where the following two bits aren't in order
    void remove_all_where_not_sorted(int bit1, int bit2) {
        assert(bit2 > bit1);
        const uint64_t mask1 = (1ULL << bit1);
        const uint64_t mask2 = (1ULL << bit2);
        for_each([=](uint64_t i) {
            // Check if bit1 is set and bit2 is not set
            if ((i & mask1) && ((i & mask2) == 0)) {
                reset(i);
            }
        });
    }

    // For every integer in the set, check if bit1 <= bit2
    bool is_sorted(int bit1, int bit2) {
        assert(bit2 > bit1);
        const int dbit = bit2 - bit1;
        uint64_t bad = 0;
        for_each([&](uint64_t i) {
            bad |= (i & ~(i >> dbit));
        });
        const uint64_t mask1 = (1ULL << bit1);
        return (bad & mask1) == 0;
    }

    // The number of integers in the set
    size_t size() const {
        size_t count = 0;
        for (size_t i = 0; i < storage.size(); i++) {
            count += __builtin_popcountll(storage[i]);
        }
        return count;
    }

    // List the binary representation of the integers in the set
    void dump_binary() {
        for_each([=](uint64_t id) {
            for (int j = 0; j < n; j++) {
                std::cout << "01"[((id >> j) & 1)];
            }
            std::cout << "\n";
        });
    }

    // List the integers in the set
    void dump() {
        for_each([=](uint64_t id) {
            std::cout << id << "\n";
        });
    }

    // Is this a subset of another set
    bool is_subset_of(const IntegerSet &other) const {
        for (size_t i = 0; i < storage.size(); i++) {
            uint64_t this_bucket = storage[i];
            uint64_t other_bucket = other.storage[i];
            if (this_bucket & ~other_bucket) {
                // There is at least one integer in this set that isn't in the other set
                return false;
            }
        }
        return true;
    }

    // Count the number of elements that aren't contained in the given arg
    size_t count_elements_in_exclusion(const IntegerSet &other) const {
        size_t count = 0;
        for (size_t i = 0; i < storage.size(); i++) {
            count += __builtin_popcountll(storage[i] & (~other.storage[i]));
        }
        return count;
    }

    // Count the number of elements that are also contained in some other set
    size_t count_elements_in_intersection(const IntegerSet &other) const {
        size_t count = 0;
        for (size_t i = 0; i < storage.size(); i++) {
            count += __builtin_popcountll(storage[i] & other.storage[i]);
        }
        return count;
    }

    // Count the number of elements that are contained in this set or the other set
    size_t count_elements_in_union(const IntegerSet &other) const {
        size_t count = 0;
        for (size_t i = 0; i < storage.size(); i++) {
            count += __builtin_popcountll(storage[i] | other.storage[i]);
        }
        return count;
    }

    // Is this equal to another set
    bool operator==(const IntegerSet &other) const {
        for (size_t i = 0; i < storage.size(); i++) {
            uint64_t this_bucket = storage[i];
            uint64_t other_bucket = other.storage[i];
            if (this_bucket != other_bucket) {
                return false;
            }
        }
        return true;
    }

    uint64_t hash() const {
        uint64_t h = 0;
        for (size_t i = 0; i < storage.size(); i++) {
            h ^= (storage[i] + 0x9e3779b9 + (h << 6) + (h >> 2));
        }
        return h;
    }
};

#endif
