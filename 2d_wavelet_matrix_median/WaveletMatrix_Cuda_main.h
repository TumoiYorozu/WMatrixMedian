#include <cuda_runtime.h>
#include <typeinfo>

namespace wavelet_matrix_median {

float* get_float_supporter_val_in_cu();

void cuda_error_check(const int line, const char* file);
void* alloc_and_transfer_cuda(const void* src_cpu, const int H, const int W, const int src_col_step, const int pix_byte);
void transfer_mem_cuda_to_host(const void* src_cu, const int H, const int W, void* dst_cpu, int dst_col_step, const int pix_byte);

void wm_median_cuda_alloc(const int radius, const int H, const int W, const size_t type_hash);
void wm_median_cuda_delete(const size_t type_hash);

template<typename T>
void wm_median_cuda(const T *src_cu, const int radius, const int H, const int W, const int src_step, T *dst_cu, const int dst_step);

template<typename T>
void wm_median_cuda_runtime_only(const int radius, const int H, const int W, T *dst_cu, const int dst_step);

} // end namespace wavelet_median


