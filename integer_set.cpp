#include "integer_set.h"

#include <iostream>

void IntegerSet::sort(int bit1, int bit2) {
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

void IntegerSet::unsort(int bit1, int bit2) {
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

bool IntegerSet::is_sorted() {
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
// TODO: Could remove entire buckets at a time instead of single bits if bit1, bit2 > 6
void IntegerSet::remove_all_where_not_sorted(int bit1, int bit2) {
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

bool IntegerSet::is_sorted(int bit1, int bit2) {
    assert(bit2 > bit1);
    const int dbit = bit2 - bit1;
    uint64_t bad = 0;
    for_each([&](uint64_t i) {
        bad |= (i & ~(i >> dbit));
    });
    const uint64_t mask1 = (1ULL << bit1);
    return (bad & mask1) == 0;
}

size_t IntegerSet::size() const {
    size_t count = 0;
    for (uint64_t i : storage) {
        count += __builtin_popcountll(i);
    }
    return count;
}

void IntegerSet::dump_binary() {
    for_each([=](uint64_t id) {
        for (int j = 0; j < n; j++) {
            std::cout << "01"[((id >> j) & 1)];
        }
        std::cout << "\n";
    });
}

void IntegerSet::dump() {
    for_each([=](uint64_t id) {
        std::cout << id << "\n";
    });
}

bool IntegerSet::is_subset_of(const IntegerSet &other) const {
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

bool IntegerSet::operator==(const IntegerSet &other) const {
    for (size_t i = 0; i < storage.size(); i++) {
        uint64_t this_bucket = storage[i];
        uint64_t other_bucket = other.storage[i];
        if (this_bucket != other_bucket) {
            return false;
        }
    }
    return true;
}

uint64_t IntegerSet::hash() const {
    uint64_t h = 0;
    for (size_t i = 0; i < storage.size(); i++) {
        h ^= (storage[i] + 0x9e3779b9 + (h << 6) + (h >> 2));
    }
    return h;
}
