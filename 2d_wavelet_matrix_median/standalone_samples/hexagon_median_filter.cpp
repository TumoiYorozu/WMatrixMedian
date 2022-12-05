#include "../WaveletMatrix_cpu_1_simple.h"
#include <opencv2/opencv.hpp>
// OpenCV is used only for image input/output


namespace wavelet_matrix_median {
using std::pair;

template<typename ValT, class WaveletMatrixImpl>
struct WM_2DMedianHexagon {
    WM_2DMedianHexagon() {}
    using XIdxT = uint16_t;
    using XYIdxT = int32_t;
    static_assert(is_same<ValT, uint32_t>() || is_same<ValT, uint16_t>() || is_same<ValT, uint8_t>());
    static_assert(is_same<XIdxT, typename WaveletMatrixImpl::T_Type>());
    static constexpr int MAX_VAL_BIT_LEN = 8 *sizeof(ValT);
    WaveletMatrixImpl X_wm_A[MAX_VAL_BIT_LEN];
    WaveletMatrixImpl X_wm_B[MAX_VAL_BIT_LEN];
    int val_bit_len = -1;
    int H, W;

    inline int get_a(int x, int y) const {
        return x * 2 + y;
    }
    inline int get_b(int x, int y) const {
        return x * 2 + H - 1 - y;
    }

    WM_2DMedianHexagon* construct(const ValT* src, const int _H, const int _W, const int src_step, const int _val_bit_len = MAX_VAL_BIT_LEN) {
        val_bit_len = _val_bit_len;
        H = _H;
        W = _W;
        const int HW = H * W;
        const XYIdxT Inf = get_a(W - 1, H - 1) + 1;
        const int w_bit_len = WaveletMatrixImpl::get_bit_len(Inf);
        assert(W < 65535); // That is, less than 65534.

        vector<std::pair<ValT, pair<XIdxT, XIdxT>>> I; // mathcal{I}_i. buffer
        vector<std::pair<ValT, pair<XIdxT, XIdxT>>> first, last;
        I.reserve(HW);
        first.reserve(HW);
        last.reserve(HW);

        for(int y = 0; y < H; ++y) {
            for(int x = 0; x < W; ++x) {
                I.emplace_back(src[y * src_step + x], pair<XIdxT, XIdxT>{get_a(x, y), get_b(x, y)});
            }
        }
        
        for (int i = val_bit_len - 1; i >= 0; --i) {
            first.clear();
            last.clear();
            X_wm_A[i] = WaveletMatrixImpl(HW);
            X_wm_B[i] = WaveletMatrixImpl(HW);
            for(int j = 0; j < HW; ++j) {
                const auto v = I[j].first;
                const auto x = I[j].second;
                if ((v >> i & 1) == 0) { // 0
                    X_wm_A[i].set_preconstruct(j, x.first);
                    X_wm_B[i].set_preconstruct(j, x.second);
                    first.emplace_back(v, x);
                }
                else {                   // 1
                    X_wm_A[i].set_preconstruct(j, Inf);
                    X_wm_B[i].set_preconstruct(j, Inf);
                    last.emplace_back(v, x);
                }
            }
            I.swap(first);
            I.insert(I.end(), last.begin(), last.end());
            X_wm_A[i].construct(w_bit_len);
            X_wm_B[i].construct(w_bit_len);
        }
        return this;
    }
    
    inline ValT quantile2d_hexagon(const XIdxT cx, const int cy, const int radius, const double position_ratio = 0.5) const {
        const XYIdxT Inf = get_a(W - 1, H - 1) + 1;

        // Number of pixels in a hexagon with radius
        const XYIdxT area = (3 * radius * radius + 2 * radius + 1 - (radius % 2));
        XYIdxT k = area * position_ratio;

        XIdxT x0a = get_a(cx - radius, cy);
        XIdxT x0b = get_b(cx - radius, cy);
        XIdxT x1a = get_a(cx + radius + 1, cy) - 1;
        XIdxT x1b = get_b(cx + radius + 1, cy) - 1;

        XYIdxT y0 = cy - radius;
        XYIdxT y1 = cy;
        XYIdxT y2 = cy + radius + 1;
        y0 *= W;
        y1 *= W;
        y2 *= W;

        ValT res = 0;
        for (int i = val_bit_len - 1; i >= 0; --i) {
            const XYIdxT y0_num = X_wm_A[i].range_freq(0, y0, 0, Inf); // same: X_wm_B[i].range_freq(0, y0, 0, Inf)
            const XYIdxT y1_num = X_wm_A[i].range_freq(0, y1, 0, Inf);
            const XYIdxT y2_num = X_wm_A[i].range_freq(0, y2, 0, Inf);
            
            const int num_bot = X_wm_B[i].less_freq(y0, y1, x1b) - X_wm_A[i].less_freq(y0, y1, x0a);
            const int num_top = X_wm_A[i].less_freq(y1, y2, x1a) - X_wm_B[i].less_freq(y1, y2, x0b);
            const int num = num_bot + num_top;
            if (k < num) {
                y0 = y0_num;
                y1 = y1_num;
                y2 = y2_num;
            }
            else {
                k -= num;
                const XYIdxT zeros =  X_wm_A[i].range_freq(0, H * W, 0, Inf); // same: X_wm_B[i].range_freq(0, H * W, 0, Inf)
                y0 = y0 - y0_num + zeros;
                y1 = y1 - y1_num + zeros;
                y2 = y2 - y2_num + zeros;
                res |= (ValT)1 << i;
            }
        }
        return res;
    }
    
    void median_cut_border(const int r, ValT *dst, const int dst_step, const double position_ratio = 0.5) const {
        const int diameter = 2 * r + 1;
        for(int y = 0; y <= H - diameter; ++y) {
            for(int x = 0; x <= W - diameter; ++x) {
                const ValT res = quantile2d_hexagon(x + r, y + r, r, position_ratio);
                dst[y * dst_step + x] = res;
            }
        }
    }
    //*/
};

} // end namespace wavelet_matrix_median




namespace wm = wavelet_matrix_median;
using T = uint8_t;

void hexagonal_interval_minimum_filter(const int radius){
    const int H = 3 + radius * 4;
    const int W = 3 + radius * 4;

    cv::Mat img = cv::Mat::ones(H, W, CV_8UC1);
    img.at<T>(H/2, W/2) = 0;

    // std::cout << "src: " << std::endl << img << std::endl;

    wm::WM_2DMedianHexagon<T, wm::WaveletMatrix<uint16_t>> WM;

    WM.construct(img.data, img.rows, img.cols, img.step);
    cv::Mat dst(img.rows - radius*2, img.cols-radius*2, img.type());
    WM.median_cut_border(radius, dst.data, dst.step, 0.0);
    
    std::cout << "radius: " << radius << std::endl;
    std::cout << dst << std::endl;
}




void logo_melt() {
    std::string file_name = "logo_melt0.png";
    int radius = 200;
    
    cv::Mat src = cv::imread(file_name, cv::IMREAD_UNCHANGED);

    copyMakeBorder(src, src, radius, radius, radius, radius, cv::BORDER_REPLICATE);

    std::vector<cv::Mat> planes;
    split(src, planes);

    wm::WM_2DMedianHexagon<T, wm::WaveletMatrix<uint16_t>> WM;
    for(auto& cmat : planes) {
        assert(cmat.type() == CV_8UC1);
        cv::Mat cmat_dst(src.rows - radius*2, src.cols - radius*2, cmat.type());

        WM.construct(cmat.data, cmat.rows, cmat.cols, cmat.step);
        WM.median_cut_border(radius, cmat_dst.data, cmat_dst.step);
        cmat = cmat_dst;
    }

    cv::Mat dst;
    merge(planes, dst);
    imwrite("logo_melt_hex"+ std::to_string(radius) + ".png", dst);
}


int main() {

    for(int r = 0; r <= 5; ++r) {
        hexagonal_interval_minimum_filter(r);
    }
    
    logo_melt();

    return 0;
}
