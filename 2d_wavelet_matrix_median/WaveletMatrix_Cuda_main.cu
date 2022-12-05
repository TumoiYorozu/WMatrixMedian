#include "WaveletMatrixMultiCu4G.cuh"
#include "WaveletMatrix2dCu5B.cuh"
#include "WaveletMatrix_Cuda_float_supporter.cuh"

#include <cuda_runtime_api.h>
#include <type_traits>

#include "WaveletMatrix_Cuda_main.h"

namespace wavelet_matrix_median {

void cuda_error_check(const int line, const char* file){
    cudaError_t cuda_error;
    if ((cuda_error = cudaGetLastError()) != cudaSuccess) {
        printf("CUDA ERROR! %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_error)); 
    }
}

void* alloc_and_transfer_cuda(const void* src_cpu, const int H, const int W, const int src_col_step, const int pix_byte) {
    const uint8_t* src8 = reinterpret_cast<const uint8_t*>(src_cpu);
    uint8_t* res_cu;

    const size_t W_byte = W * pix_byte;
    const size_t src_line_byte = src_col_step * pix_byte;
    cudaMalloc(&res_cu, W_byte * H);

    for(int y = 0; y < H; ++y) {
        cudaMemcpy(res_cu + y * W_byte, src8 + y * src_line_byte, W_byte, cudaMemcpyHostToDevice);
    }
    return reinterpret_cast<void*>(res_cu);
}

void transfer_mem_cuda_to_host(const void* src_cu, const int H, const int W, void* dst_cpu, int dst_col_step, const int pix_byte) {
    const uint8_t* src8 = reinterpret_cast<const uint8_t*>(src_cu);
    uint8_t* dst8 = reinterpret_cast<uint8_t*>(dst_cpu);

    const size_t W_byte = W * pix_byte;
    const size_t dst_line_byte = dst_col_step * pix_byte;
    for(int y = 0; y < H; ++y) {
        cudaMemcpy(dst8 + y * dst_line_byte, src8 + y * W_byte, W_byte, cudaMemcpyDeviceToHost);
    }
}

namespace VARS {
    constexpr static int CH_NUM = 1;
    constexpr static int WORD_SIZE = 32;
    using XYIdxT = uint32_t;
    using XIdxT = uint16_t;
    using WM2D_IMPL08u = WaveletMatrix2dCu5B<uint8_t,  CH_NUM, WaveletMatrixMultiCu4G<XIdxT, 512>, 512, WORD_SIZE>;
    using WM2D_IMPL16u = WaveletMatrix2dCu5B<uint16_t, CH_NUM, WaveletMatrixMultiCu4G<XIdxT, 512>, 512, WORD_SIZE>;
    using WM2D_IMPL32f = WaveletMatrix2dCu5B<uint32_t, CH_NUM, WaveletMatrixMultiCu4G<XIdxT, 512>, 512, WORD_SIZE>;
    WM2D_IMPL08u *wm2d_gpu_08u;
    WM2D_IMPL16u *wm2d_gpu_16u;
    WM2D_IMPL32f *wm2d_gpu_32f;
    WMMedianFloatSupporter<float, XYIdxT> float_supporter;
}


void wm_median_cuda_alloc(const int radius, const int H, const int W, const size_t type_hash) {
    const int exH= H + 2 * radius;
    const int exW= W + 2 * radius;
    const bool is_float = (type_hash == typeid(float).hash_code());
    if (type_hash == typeid(uint8_t).hash_code()) {
        VARS::wm2d_gpu_08u = new VARS::WM2D_IMPL08u(exH, exW, is_float, false);
        return;
    }
    if (type_hash == typeid(uint16_t).hash_code()) {
        VARS::wm2d_gpu_16u = new VARS::WM2D_IMPL16u(exH, exW, is_float, false);
        return;
    }
    if (type_hash == typeid(float).hash_code()) {
        VARS::wm2d_gpu_32f = new VARS::WM2D_IMPL32f(exH, exW, is_float, false);
        VARS::float_supporter.reset(exH, exW);
        VARS::float_supporter.alloc();
        return;
    }
}
void wm_median_cuda_delete(const size_t type_hash) {
    if (type_hash == typeid(uint8_t).hash_code()) {
        delete VARS::wm2d_gpu_08u;
        return;
    }
    if (type_hash == typeid(uint16_t).hash_code()) {
        delete VARS::wm2d_gpu_16u;
        return;
    }
    if (type_hash == typeid(float).hash_code()) {
        delete VARS::wm2d_gpu_32f;
        return;
    }
}

float* get_float_supporter_val_in_cu(){
    return VARS::float_supporter.val_in_cu;
}

template<class WM2D_IMPL, typename T>
void wm_median_cuda_runtime_only_core(WM2D_IMPL *wm2d_gpu, const int radius, const int H, const int W, T *dst_cu, const int dst_step) {
    using WM_T = std::conditional_t<std::is_same_v<T, float>, uint32_t, T>;
    constexpr bool is_float = std::is_same_v<T, float>;
    constexpr static int ThW = (std::is_same_v<T, uint8_t> ?  8 : 4);
    constexpr static int ThH = (std::is_same_v<T, uint8_t> ? 64 : 256);

    using MedianResT = std::conditional_t<is_float, T, std::nullptr_t>;
    const MedianResT* res_table = is_float ? (MedianResT*)VARS::float_supporter.get_res_table() : nullptr;

    if (wm2d_gpu->res_cu != nullptr) { puts("wm2d_gpu->res_cu panic!"); exit(1);}
    if (W != dst_step) { puts("this support only W == dst_step."); exit(1);}
    wm2d_gpu->res_cu =  reinterpret_cast<WM_T*>(dst_cu);
    wm2d_gpu->template median2d<ThW, ThH, MedianResT, true>(radius, res_table);
    wm2d_gpu->res_cu = nullptr;
}

template<class WM2D_IMPL, typename T>
void wm_median_cuda_core(WM2D_IMPL *wm2d_gpu, const T *src_cu, const int radius, const int H, const int W, const int src_step, T *dst_cu, const int dst_step) {
    using WM_T = std::conditional_t<std::is_same_v<T, float>, uint32_t, T>;
    constexpr bool is_float = std::is_same_v<T, float>;
    const int exW= W + 2 * radius;

    if (exW != src_step) { puts("this support only exW == src_step."); exit(1);}
    if constexpr (is_float) {
        VARS::float_supporter.sort_and_set(wm2d_gpu->src_cu);
        wm2d_gpu->construct();
    } else {
        wm2d_gpu->construct(src_cu);
    }
    wm_median_cuda_runtime_only_core(wm2d_gpu, radius, H, W, dst_cu, dst_step);
}

template<typename T>
void wm_median_cuda(const T *src_cu, const int radius, const int H, const int W, const int src_step, T *dst_cu, const int dst_step) {
    if constexpr(is_same<T, uint8_t>()) {
        wm_median_cuda_core(VARS::wm2d_gpu_08u, src_cu, radius, H, W, src_step, dst_cu, dst_step);
    }
    if constexpr(is_same<T, uint16_t>()) {
        wm_median_cuda_core(VARS::wm2d_gpu_16u, src_cu, radius, H, W, src_step, dst_cu, dst_step);
    }
    if constexpr(is_same<T, float>()) {
        wm_median_cuda_core(VARS::wm2d_gpu_32f, src_cu, radius, H, W, src_step, dst_cu, dst_step);
    }
}

template<typename T>
void wm_median_cuda_runtime_only(const int radius, const int H, const int W, T *dst_cu, const int dst_step) {
    if constexpr(is_same<T, uint8_t>()) {
        wm_median_cuda_runtime_only_core(VARS::wm2d_gpu_08u, radius, H, W, dst_cu, dst_step);
    }
    if constexpr(is_same<T, uint16_t>()) {
        wm_median_cuda_runtime_only_core(VARS::wm2d_gpu_16u, radius, H, W, dst_cu, dst_step);
    }
    if constexpr(is_same<T, float>()) {
        wm_median_cuda_runtime_only_core(VARS::wm2d_gpu_32f, radius, H, W, dst_cu, dst_step);
    }
}

template void wm_median_cuda(const uint8_t  *src_cu, const int radius, const int H, const int W, const int src_step, uint8_t  *dst_cu, const int dst_step);
template void wm_median_cuda(const uint16_t *src_cu, const int radius, const int H, const int W, const int src_step, uint16_t *dst_cu, const int dst_step);
template void wm_median_cuda(const float    *src_cu, const int radius, const int H, const int W, const int src_step, float    *dst_cu, const int dst_step);

template void wm_median_cuda_runtime_only(const int radius, const int H, const int W, uint8_t  *dst_cu, const int dst_step);
template void wm_median_cuda_runtime_only(const int radius, const int H, const int W, uint16_t *dst_cu, const int dst_step);
template void wm_median_cuda_runtime_only(const int radius, const int H, const int W, float    *dst_cu, const int dst_step);


} // end namespace wavelet_median
