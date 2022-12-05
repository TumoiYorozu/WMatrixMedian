// Taken from https://github.com/aoles/EBImage/blob/master/src/medianFilter.c
// With R-specific code removed

// This one file continues to be under the GPL, as Simon Perreault's
// original was released that way.

// Modifications:

// Removed sse2 specialization and instead ensured that gcc
// autovectorizes this fine (this made it about 10-20% faster due to
// avx2 usage)

// Hardwired channel count to 1 (5% faster)

// Run stripes in parallel using Halide's parallel runtime.

#include <HalideRuntime.h>

// For posix_memalign
#include <malloc.h>

/* medianFilter.c - Constant-time median filtering of 16-bit images
 *  (for inclusion in the Bioconductor Package EBImage)
 *
 *  The original ctmf algorithm here has minor modifications,
 *  which continue to be covered by the GNU General Public License.
 *
 * Contact:
 *  Joseph Barry
 *  Huber Group
 *  EMBL Heidelberg
 *  Meyerhofstr. 1
 *  69115 Germany
 *
 *  joseph.barry@embl.de
*/

/* R/Bioconductor includes */

/*
 * ctmf.c - Constant-time median filtering
 * Copyright (C) 2006  Simon Perreault
 *
 * Reference: S. Perreault and P. Hébert, "Median Filtering in Constant Time",
 * IEEE Transactions on Image Processing, September 2007.
 *
 * This program has been obtained from http://nomis80.org/ctmf.html. No patent
 * covers this program, although it is subject to the following license:
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contact:
 *  Laboratoire de vision et systèmes numériques
 *  Pavillon Adrien-Pouliot
 *  Université Laval
 *  Sainte-Foy, Québec, Canada
   G1K 7P4
 *
 *  perreaul@gel.ulaval.ca
 */

/* Standard C includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Type declarations */
#ifdef _MSC_VER
#include <basetsd.h>
typedef UINT8 uint8_t;
typedef UINT16 uint16_t;
typedef UINT32 uint32_t;
#pragma warning(disable : 4799)
#else
#include <stdint.h>
#endif

/* Compiler peculiarities */
#if defined(__GNUC__)
#include <stdint.h>
#define inline __inline__
#define align(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
#define inline __inline
#define align(x) __declspec(align(x))
#else
#define inline
#define align(x)
#endif

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

/**
 * This structure represents a two-tier histogram. The first tier (known as the
 * "coarse" level) is 8 bit wide and the second tier (known as the "fine" level)
 * is 16 bit wide. Pixels inserted in the fine level also get inserted into the
 * coarse bucket designated by the MSBs of the fine bucket value.
 *
 * The structure is aligned on 16 bytes, which is a prerequisite for SIMD
 * instructions. Each bucket is 16 bit wide, which means that extra care must be
 * taken to prevent overflow.
 */
typedef struct align(16) {
    uint16_t coarse[SQRT_BUCKET_SIZE];
    uint16_t fine[SQRT_BUCKET_SIZE][SQRT_BUCKET_SIZE];
}
Histogram;

/**
 * HOP is short for Histogram OPeration. This macro makes an operation \a op on
 * histogram \a h for pixel value \a x. It takes care of handling both levels.
 */

#define HOP(h, x, op)      \
    h.coarse[x >> MSB] op; \
    *((uint16_t *)h.fine + x) op;

#define COP(c, j, x, op)                                      \
    h_coarse[SQRT_BUCKET_SIZE * (n * c + j) + (x >> MSB)] op; \
    h_fine[SQRT_BUCKET_SIZE * (n * (SQRT_BUCKET_SIZE * c + (x >> MSB)) + j) + (x & (SQRT_BUCKET_SIZE - 1))] op;

static inline void histogram_add(const uint16_t x[SQRT_BUCKET_SIZE], uint16_t y[SQRT_BUCKET_SIZE]) {
    int i;
    for (i = 0; i < SQRT_BUCKET_SIZE; ++i) {
        y[i] += x[i];
    }
}

static inline void histogram_sub(const uint16_t x[SQRT_BUCKET_SIZE], uint16_t y[SQRT_BUCKET_SIZE]) {
    int i;
    for (i = 0; i < SQRT_BUCKET_SIZE; ++i) {
        y[i] -= x[i];
    }
}

static inline void histogram_muladd(const uint16_t a, const uint16_t x[SQRT_BUCKET_SIZE],
                                    uint16_t y[SQRT_BUCKET_SIZE]) {
    int i;
    for (i = 0; i < SQRT_BUCKET_SIZE; ++i) {
        y[i] += a * x[i];
    }
}

static void ctmf_helper(
    const pixel_t *const src, pixel_t *const dst,
    const int width, const int height,
    const int src_step, const int dst_step,
    const int r, const int cn_,
    const int pad_left, const int pad_right) {
    const int m = height, n = width;
    int i, j, k, c;
    const pixel_t *p, *q;

    // Modified to always use a channel count of one
    assert(cn_ == 1);
    const int cn = 1;

    Histogram H[4];
    uint16_t *h_coarse, *h_fine, luc[4][SQRT_BUCKET_SIZE];
    assert(src);
    assert(dst);
    assert(r >= 0);
    assert(width >= 2 * r + 1);
    assert(height >= 2 * r + 1);
    assert(src_step != 0);
    assert(dst_step != 0);

    h_coarse = (uint16_t *)calloc(1 * SQRT_BUCKET_SIZE * n * cn, sizeof(uint16_t));
    h_fine = (uint16_t *)calloc(SQRT_BUCKET_SIZE * SQRT_BUCKET_SIZE * n * cn, sizeof(uint16_t));

    /* First row initialization */
    for (j = 0; j < n; ++j) {
        for (c = 0; c < cn; ++c) {
            COP(c, j, src[cn * j + c], += r + 1);
        }
    }
    for (i = 0; i < r; ++i) {
        for (j = 0; j < n; ++j) {
            for (c = 0; c < cn; ++c) {
                COP(c, j, src[src_step * i + cn * j + c], ++);
            }
        }
    }

    for (i = 0; i < m; ++i) {

        /* Update column histograms for entire row. */
        p = src + src_step * MAX(0, i - r - 1);
        q = p + cn * n;
        for (j = 0; p != q; ++j) {
            for (c = 0; c < cn; ++c, ++p) {
                COP(c, j, *p, --);
            }
        }

        p = src + src_step * MIN(m - 1, i + r);
        q = p + cn * n;
        for (j = 0; p != q; ++j) {
            for (c = 0; c < cn; ++c, ++p) {
                COP(c, j, *p, ++);
            }
        }

        /* First column initialization */
        memset(H, 0, cn * sizeof(H[0]));
        memset(luc, 0, cn * sizeof(luc[0]));
        if (pad_left) {
            for (c = 0; c < cn; ++c) {
                histogram_muladd(r, &h_coarse[SQRT_BUCKET_SIZE * n * c], H[c].coarse);
            }
        }
        for (j = 0; j < (pad_left ? r : 2 * r); ++j) {
            for (c = 0; c < cn; ++c) {
                histogram_add(&h_coarse[SQRT_BUCKET_SIZE * (n * c + j)], H[c].coarse);
            }
        }
        for (c = 0; c < cn; ++c) {
            for (k = 0; k < SQRT_BUCKET_SIZE; ++k) {
                histogram_muladd(2 * r + 1, &h_fine[SQRT_BUCKET_SIZE * n * (SQRT_BUCKET_SIZE * c + k)], &H[c].fine[k][0]);
            }
        }

        for (j = pad_left ? 0 : r; j < (pad_right ? n : n - r); ++j) {
            for (c = 0; c < cn; ++c) {
                const uint16_t t = 2 * r * r + 2 * r;
                uint16_t sum = 0, *segment;
                int b;

                histogram_add(&h_coarse[SQRT_BUCKET_SIZE * (n * c + MIN(j + r, n - 1))], H[c].coarse);

                /* Find median at coarse level */
                for (k = 0; k < SQRT_BUCKET_SIZE; ++k) {
                    sum += H[c].coarse[k];
                    if (sum > t) {
                        sum -= H[c].coarse[k];
                        break;
                    }
                }
                assert(k < (uint16_t)SQRT_BUCKET_SIZE);

                /* Update corresponding histogram segment */
                if (luc[c][k] <= j - r) {
                    memset(&H[c].fine[k], 0, SQRT_BUCKET_SIZE * sizeof(uint16_t));
                    for (luc[c][k] = j - r; luc[c][k] < MIN(j + r + 1, n); ++luc[c][k]) {
                        histogram_add(&h_fine[SQRT_BUCKET_SIZE * (n * (SQRT_BUCKET_SIZE * c + k) + luc[c][k])], H[c].fine[k]);
                    }
                    if (luc[c][k] < j + r + 1) {
                        histogram_muladd(j + r + 1 - n, &h_fine[SQRT_BUCKET_SIZE * (n * (SQRT_BUCKET_SIZE * c + k) + (n - 1))], &H[c].fine[k][0]);
                        luc[c][k] = j + r + 1;
                    }
                } else {
                    for (; luc[c][k] < j + r + 1; ++luc[c][k]) {
                        histogram_sub(&h_fine[SQRT_BUCKET_SIZE * (n * (SQRT_BUCKET_SIZE * c + k) + MAX(luc[c][k] - 2 * r - 1, 0))], H[c].fine[k]);
                        histogram_add(&h_fine[SQRT_BUCKET_SIZE * (n * (SQRT_BUCKET_SIZE * c + k) + MIN(luc[c][k], n - 1))], H[c].fine[k]);
                    }
                }

                histogram_sub(&h_coarse[SQRT_BUCKET_SIZE * (n * c + MAX(j - r, 0))], H[c].coarse);

                /* Find median in segment */
                segment = H[c].fine[k];
                for (b = 0; b < SQRT_BUCKET_SIZE; ++b) {
                    sum += segment[b];
                    if (sum > t) {
                        dst[dst_step * i + cn * j + c] = SQRT_BUCKET_SIZE * k + b;
                        break;
                    }
                }
                assert(b < (uint16_t)SQRT_BUCKET_SIZE);
            }
        }
    }

    free(h_coarse);
    free(h_fine);
}

/**
 * \brief Constant-time median filtering
 *
 * This function does a median filtering of an 16-bit image. The source image is
 * processed as if it was padded with zeros. The median kernel is square with
 * odd dimensions. Images of arbitrary size may be processed.
 *
 * To process multi-channel images, you must call this function multiple times,
 * changing the source and destination adresses and steps such that each channel
 * is processed as an independent single-channel image.
 *
 * Processing images of arbitrary bit depth is not supported.
 *
 * The computing time is O(1) per pixel, independent of the radius of the
 * filter. The algorithm's initialization is O(r*width), but it is negligible.
 * Memory usage is simple: it will be as big as the cache size, or smaller if
 * the image is small. For efficiency, the histograms' bins are 16-bit wide.
 * This may become too small and lead to overflow as \a r increases.
 *
 * \param src           Source image data.
 * \param dst           Destination image data. Must be preallocated.
 * \param width         Image width, in pixels.
 * \param height        Image height, in pixels.
 * \param src_step      Distance between adjacent pixels on the same column in
 *                      the source image, in bytes.
 * \param dst_step      Distance between adjacent pixels on the same column in
 *                      the destination image, in bytes.
 * \param r             Median filter radius. The kernel will be a 2*r+1 by
 *                      2*r+1 square.
 * \param cn            Number of channels. For example, a grayscale image would
 *                      have cn=1 while an RGB image would have cn=3.
 * \param num_stripes   The number of stripes to use. The algorithm is
 *                      parallel over stripes, so set this to some multiple
 *                      of the number of cores you want to use.
 */
void CTMF_FN(
    const pixel_t *const src, pixel_t *const dst,
    const int width, const int height,
    const int src_step, const int dst_step,
    const int r, const int cn, int num_stripes) {
    /*
     * Processing the image in vertical stripes is an optimization made
     * necessary by the limited size of the CPU cache. Each histogram is 544
     * bytes big and therefore I can fit a limited number of them in the cache.
     * That number may sometimes be smaller than the image width, which would be
     * the number of histograms I would need without stripes.
     *
     * I need to keep histograms in the cache so that they are available
     * quickly when processing a new row. Each row needs access to the previous
     * row's histograms. If there are too many histograms to fit in the cache,
     * thrashing to RAM happens.
     *
     * To solve this problem, I figure out the maximum number of histograms
     * that can fit in cache. From this is determined the number of stripes in
     * an image. The formulas below make the stripes all the same size and use
     * as few stripes as possible.
     *
     * Note that each stripe causes an overlap on the neighboring stripes, as
     * when mowing the lawn. That overlap is proportional to r. When the overlap
     * is a significant size in comparison with the stripe size, then we are not
     * O(1) anymore, but O(r). In fact, we have been O(r) all along, but the
     * initialization term was neglected, as it has been (and rightly so) in B.
     * Weiss, "Fast Median and Bilateral Filtering", SIGGRAPH, 2006. Processing
     * by stripes only makes that initialization term bigger.
     *
     * Also, note that the leftmost and rightmost stripes don't need overlap.
     * A flag is passed to ctmf_helper() so that it treats these cases as if the
     * image was zero-padded.
     */

    // Modification: The whole memory size thing didn't seem to have
    // any performance effect for the sizes tested in this
    // paper. Instead we always use num_cores stripes and parallelize
    // over stripes.

    // Force the stripes to be at least 2*r + 1 wide.
    int stripe_size, last_stripe;
    num_stripes *= 2;
    do {
        num_stripes /= 2;
        stripe_size = (width + num_stripes - 1) / num_stripes;
        last_stripe = width - stripe_size * (num_stripes - 1);
    } while (last_stripe < 2 * r + 1);

    struct Closure {
        int stripe_size, r, width, height, cn, src_step, dst_step;
        const pixel_t *src;
        pixel_t *dst;
    } closure{stripe_size, r, width, height, cn, src_step, dst_step,
              src, dst};

    auto do_one_stripe = [](void *ucon, int idx, uint8_t *closure) {
        Closure *c = (Closure *)closure;

        bool first_stripe = false, last_stripe = false;
        int start = idx * c->stripe_size;
        int end = start + c->stripe_size + 2 * c->r;
        if (end >= c->width) {
            end = c->width;
            last_stripe = true;
        }
        if (idx == 0) {
            start = 0;
            first_stripe = true;
        }
        int extent = end - start;

        ctmf_helper(c->src + c->cn * start, c->dst + c->cn * start, extent,
                    c->height, c->src_step, c->dst_step, c->r, c->cn,
                    first_stripe, last_stripe);
        return 0;
    };

    halide_do_par_for(nullptr, do_one_stripe, 0, num_stripes, (uint8_t *)(&closure));
}
