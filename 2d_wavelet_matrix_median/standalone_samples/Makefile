OPENCV_DIR ?= $(shell echo /usr/local)

OPENCV_FLAGS ?= -I $(OPENCV_DIR)/include/opencv4
OPENCV_LIBS ?= -L $(OPENCV_DIR)/lib -lpng -lopencv_core -ltbb -lopencv_imgcodecs -lopencv_imgproc -lopencv_cudafilters -Wl,-rpath,$(OPENCV_DIR)/lib

OPENMP_FLAGS ?= -fopenmp=libomp

CXX_FLAGS=-O3 -Wall -std=c++17 -march=skylake
CXX=clang++-12

wm_with_opencv.bin: wm_with_opencv.cpp
	$(CXX) $< $(CXX_FLAGS) $(OPENCV_FLAGS) $(OPENCV_LIBS) $(OPENMP_FLAGS) -o $@

hexagon_median_filter.bin: hexagon_median_filter.cpp
	$(CXX) $< $(CXX_FLAGS) $(OPENCV_FLAGS) $(OPENCV_LIBS) $(OPENMP_FLAGS) -o $@

