These are our two-dimensional wavelet watrix median filter samples.
We can compile and test them if prepared with OpenMP and OpenCV (for image input/output).
Please use them to check and analyze our method.


# wm_with_opencv.cpp
This is a simple sample of applying a median filter to an image.
By changing the file_name and radius variables and compiling, we can actually try median filtering on various images.


# hexagon_median_filter.cpp
This is an example of adaptation to the polygonal window introduced in Appendix B. This is a median filter with wavelet matrix for a hexagonal window.
It uses two WaveletMatrix inside. Compare the implementation with the "WM_2DMedian" class in "WaveletMatrix_cpu_1_simple.h".

The basic idea to deal with hexagonal windows is to divide the interval into two quadrilaterals. Then, by using a wavelet matrix with two different indexes, we can handle skewed interval. Since two wavelet matrices are used, the memory usage and computation time will increase by a factor of two.
Note that the hexagons we will be dealing with are not regular hexagons. However, the effect is not a problem because it is close enough to a circle for practical use.

This program contains two functions.

"hexagonal_interval_minimum_filter" is a modification of the median filter to obtain the minimum value of an interval instead of the median.
The input is filled with 1's, but only one pixel in the center is 0.
By applying a hexagonal interval minimum filter to this, a hexagonally arranged image of zeros can be obtained.
The following is a part of the output. We can see that the zeros are arranged in a hexagonal shape.

```
radius: 3
[  1,   1,   1,   1,   1,   1,   1,   1,   1;
   1,   1,   1,   0,   0,   0,   1,   1,   1;
   1,   1,   0,   0,   0,   0,   0,   1,   1;
   1,   1,   0,   0,   0,   0,   0,   1,   1;
   1,   0,   0,   0,   0,   0,   0,   0,   1;
   1,   1,   0,   0,   0,   0,   0,   1,   1;
   1,   1,   0,   0,   0,   0,   0,   1,   1;
   1,   1,   1,   0,   0,   0,   1,   1,   1;
   1,   1,   1,   1,   1,   1,   1,   1,   1]
```

The "logo_melt()" function is a sample that takes a SIGGRAPH Logo image in the directory as input and performs a median filter with a large radius. By changing the input and radius, the hexagonal median filter can be tried on a variety of images.
