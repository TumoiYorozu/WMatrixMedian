#include "../WaveletMatrix_cpu_3_parallel2.h"
#include <opencv2/opencv.hpp>
// OpenCV is used only for image input/output

namespace wm = wavelet_matrix_median;

int main() {
    std::string file_name = "../../median_filters_before_srgb_flowers.png";
    int radius = 20;


    cv::Mat src = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
    cv::Mat dst(src.rows, src.cols, src.type());
    assert(src.type() == CV_8UC1);

    copyMakeBorder(src, src, radius, radius, radius, radius, cv::BORDER_REPLICATE);

    using T = uint8_t;
    
    // wm::WM_2DMedian<T, wm::WaveletMatrix<uint16_t>> WM; // More simple implementation
    wm::WM_2DMedianParallel2<T, wm::WaveletMatrixParallel2<uint16_t>> WM;

    WM.construct(src.data, src.rows, src.cols, src.step);

    WM.median_cut_border(radius, dst.data, dst.step);

    imwrite("output.jpg", dst);
}
