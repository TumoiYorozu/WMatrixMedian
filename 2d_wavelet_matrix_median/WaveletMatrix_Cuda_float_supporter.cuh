namespace wavelet_matrix_median {

template<int blockDim, typename IdxT>
__global__ void iota_idx(IdxT *idx_in_cu, const IdxT hw) {
    const IdxT i = blockIdx.x * blockDim + threadIdx.x;
    if (i >= hw) return;
    idx_in_cu[i] = i;
}

template<int blockDim, typename IdxT>
__global__ void set_wm_val(IdxT *wm_src_p, const IdxT *idx_out_cu, const IdxT hw) {
    // TODO: Should we make the division of the block two-dimensional?
    const IdxT i = blockIdx.x * blockDim + threadIdx.x;
    if (i >= hw) return;
    const IdxT j = idx_out_cu[i];
    wm_src_p[j] = i;
}

template<int blockDim, typename IdxT, typename ValT>
__global__ void conv_res_cu(ValT *dst, const ValT *val_out_cu, const IdxT *res_cu, const IdxT hw) {
    const IdxT i = blockIdx.x * blockDim + threadIdx.x;
    if (i >= hw) return;
    
    const IdxT r = res_cu[i];
    dst[i] = val_out_cu[r];
}

template<typename ValT, typename IdxT>
struct WMMedianFloatSupporter {
    constexpr static int blockDim = 512;
    int h = 0, w = 0;
    int hw_bit_len = -1;
    WMMedianFloatSupporter(){};
    WMMedianFloatSupporter(int h, int w) { reset(h, w); }
    ~WMMedianFloatSupporter(){
        free();
    }
    ValT *val_in_cu = nullptr;
    IdxT *idx_in_cu;
    ValT *val_out_cu;
    IdxT *idx_out_cu;
    void *cub_temp_storage = nullptr;
    size_t cub_temp_storage_bytes;
    void reset(const int H, const int W) {
        h = H; w = W;
        free();
    }
    void alloc(){
        const int end_bit = sizeof(ValT) * 8;
        cub::DeviceRadixSort::SortPairs(
            nullptr, cub_temp_storage_bytes, val_in_cu, val_out_cu, idx_in_cu, idx_out_cu, 
            h * w, 0, end_bit);

        cudaMalloc(&val_in_cu, 2 * h * w * (sizeof(ValT) + sizeof(IdxT)) + cub_temp_storage_bytes);
        idx_in_cu = (IdxT*)(val_in_cu + h * w);
        val_out_cu = (ValT*)(idx_in_cu + h * w);
        idx_out_cu = (IdxT*)(val_out_cu + h * w);
        cub_temp_storage = idx_out_cu + h * w;
    }
    void free() {
        if (val_in_cu != nullptr) {
            cudaFree(val_in_cu);
        }
    }
    void sort_and_set(IdxT *wm_src_p){
        const IdxT hw = h * w;
        const int gridDim((h * w + blockDim - 1) / blockDim);
        iota_idx<blockDim><<<gridDim, blockDim>>>(idx_in_cu, hw);
        const int end_bit = sizeof(ValT) * 8;
        cub::DeviceRadixSort::SortPairs(
            cub_temp_storage, cub_temp_storage_bytes, val_in_cu, val_out_cu, idx_in_cu, idx_out_cu, 
            hw, 0, end_bit);
        set_wm_val<blockDim><<<gridDim, blockDim>>>(wm_src_p, idx_out_cu, hw);
        for(hw_bit_len = 1; ; ++hw_bit_len) {
            if ((1ull << hw_bit_len) >= hw) {
                break;
            }
        }
    }
    const ValT* get_res_table() const {
        return val_out_cu;
    }
};
} // end namespace wavelet_median
