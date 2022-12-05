# ulimit -s unlimited
HALIDE_DIR ?= $(shell echo ~/projects/Halide/distrib)
OPENCV_DIR ?= $(shell echo ~/projects/opencv/install)
ARRAYFIRE_DIR ?= $(shell echo ~/projects/arrayfire/install)
IPP_DIR ?= $(shell echo ~/intel/compilers_and_libraries/linux/ipp)

OPENCV_FLAGS ?= -I $(OPENCV_DIR)/include/opencv4
OPENCV_LIBS ?= -L $(OPENCV_DIR)/lib -lpng -lopencv_core -ltbb -lopencv_imgcodecs -lopencv_imgproc -lopencv_cudafilters -Wl,-rpath,$(OPENCV_DIR)/lib

ARRAYFIRE_FLAGS ?= -I $(ARRAYFIRE_DIR)/include
ARRAYFIRE_LIBS ?= -L $(ARRAYFIRE_DIR)/lib -lafcuda -Wl,-rpath,$(ARRAYFIRE_DIR)/lib # -lafcpu to use CPU backend

OPENMP_FLAGS ?= -fopenmp=libomp -I /llvm-12/lib/clang/12.0.0/include/

HL_TARGET ?= x86-64-linux-avx2-no_asserts

CXX_FLAGS=-O3 -Wall -std=c++17
CXX=clang++-12
NVCC=nvcc
NVCC_FLAGS=-std=c++17 -arch=sm_86 -O3 -DNDEBUG --use_fast_math --cudart static --maxrregcount 40 --diag-suppress 1675 -ccbin $(CXX) --compiler-options -march=native,-O3


# CXX_CUDA_FLAGS=--cuda-gpu-arch=sm_86 -L/usr/local/cuda/lib -L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs -L/usr/local/cuda/bin/../targets/x86_64-linux/lib -lcudart_static -ldl -lrt -pthread -lcudart
CXX_CUDA_FLAGS=--cuda-gpu-arch=sm_86 -lcudart -I /usr/local/cuda-11.6/targets/x86_64-linux/include/ -L /usr/local/cuda-11.6/targets/x86_64-linux/lib/


TIME_LIMIT=-t 3600

# We can link different ippis to select different instruction sets:
# Use -lippi for platform native
# Use -lippil9 to force avx2
# We use avx2 for this work. avx512 is faster for the Halide and IPP
# versions, but less representative of what people actually have
# available.

IPP_FLAGS ?= -I $(IPP_DIR)/include
IPP_LIBS ?= -L $(IPP_DIR)/lib/intel64 -lippil9 -lippcore -Wl,-rpath,$(IPP_DIR)/lib/intel64

NPP_LIBS ?= -lnppif

bin/WaveletMatrix_Cuda_main.o: 2d_wavelet_matrix_median/WaveletMatrix_Cuda_main.cu 2d_wavelet_matrix_median/*.h 2d_wavelet_matrix_median/*.cuh
	mkdir -p $(@D)
	$(NVCC) $< -c -o $@ $(NVCC_FLAGS)

bin/median_filter_generator_v2: median_filter_generator_v2.cpp bin/sorting_network.o bin/sorting_bytecode.o
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -march=native $(HALIDE_DIR)/tools/GenGen.cpp $^ -I $(HALIDE_DIR)/include -L $(HALIDE_DIR)/lib -lHalide -Wl,-rpath,$(HALIDE_DIR)/lib -o bin/median_filter_generator_v2
	# $(CXX) $(CXX_FLAGS) -march=native $(HALIDE_DIR)/tools/GenGen.cpp $^ -I $(HALIDE_DIR)/include -L $(HALIDE_DIR)/lib -lHalide -Wl,-rpath,$(HALIDE_DIR)/lib  -Wl,-z,stack-size=1610612736 -o bin/median_filter_generator_v2

bin/median_filter_u8_%.a: bin/median_filter_generator_v2 
	./bin/median_filter_generator_v2 -g median_filter -f median_filter_u8_$* -o bin ${TIME_LIMIT} -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-no_runtime src.type=uint8 dst.type=uint8 radius=$* 2>&1 
bin/median_filter_u16_%.a: bin/median_filter_generator_v2 
	./bin/median_filter_generator_v2 -g median_filter -f median_filter_u16_$* -o bin ${TIME_LIMIT} -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-no_runtime src.type=uint16 dst.type=uint16 radius=$* 2>&1 
bin/median_filter_f32_%.a: bin/median_filter_generator_v2 
	./bin/median_filter_generator_v2 -g median_filter -f median_filter_f32_$* -o bin ${TIME_LIMIT} -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-no_runtime src.type=float32 dst.type=float32 radius=$* 2>&1

bin/median_filter_cuda_u8_%.a: bin/median_filter_generator_v2 
	./bin/median_filter_generator_v2 -g median_filter -f median_filter_cuda_u8_$* ${TIME_LIMIT} -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_runtime src.type=uint8 dst.type=uint8 radius=$* 2>&1 
bin/median_filter_cuda_u16_%.a: bin/median_filter_generator_v2 
	./bin/median_filter_generator_v2 -g median_filter -f median_filter_cuda_u16_$* ${TIME_LIMIT} -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_runtime src.type=uint16 dst.type=uint16 radius=$* 2>&1 
bin/median_filter_cuda_f32_%.a: bin/median_filter_generator_v2 
	./bin/median_filter_generator_v2 -g median_filter -f median_filter_cuda_f32_$* ${TIME_LIMIT} -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_runtime src.type=float32 dst.type=float32 radius=$* 2>&1

bin/dynamic_median_filter_generator_v2: dynamic_median_filter_generator_v2.cpp bin/sorting_bytecode.o bin/sorting_network.o
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -march=native $(HALIDE_DIR)/tools/GenGen.cpp $^ -I $(HALIDE_DIR)/include -L $(HALIDE_DIR)/lib -lHalide -Wl,-rpath,$(HALIDE_DIR)/lib -o bin/dynamic_median_filter_generator_v2

bin/dynamic_median_filter_u8_2x2_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_u8_2x2_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-disable_llvm_loop_opt-no_runtime src.type=uint8 dst.type=uint8 tw=2 th=2 

bin/dynamic_median_filter_u8_4x4_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_u8_4x4_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-disable_llvm_loop_opt-no_runtime src.type=uint8 dst.type=uint8 tw=4 th=4 

bin/dynamic_median_filter_u8_8x8_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_u8_8x8_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-disable_llvm_loop_opt-no_runtime src.type=uint8 dst.type=uint8 tw=8 th=8 #max_leaf=24 # For max leaf size experiment

bin/dynamic_median_filter_u16_2x2_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_u16_2x2_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-disable_llvm_loop_opt-no_runtime src.type=uint16 dst.type=uint16 tw=2 th=2

bin/dynamic_median_filter_u16_4x4_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_u16_4x4_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-disable_llvm_loop_opt-no_runtime src.type=uint16 dst.type=uint16 tw=4 th=4

bin/dynamic_median_filter_u16_8x8_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_u16_8x8_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-disable_llvm_loop_opt-no_runtime src.type=uint16 dst.type=uint16 tw=8 th=8

bin/dynamic_median_filter_f32_2x2_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_f32_2x2_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-disable_llvm_loop_opt-no_runtime src.type=float32 dst.type=float32 tw=2 th=2

bin/dynamic_median_filter_f32_4x4_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_f32_4x4_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-disable_llvm_loop_opt-no_runtime src.type=float32 dst.type=float32 tw=4 th=4

bin/dynamic_median_filter_f32_8x8_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_f32_8x8_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-disable_llvm_loop_opt-no_runtime src.type=float32 dst.type=float32 tw=8 th=8 

bin/dynamic_median_filter_cuda_u8_2x2_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_cuda_u8_2x2_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_asserts-disable_llvm_loop_opt-no_runtime src.type=uint8 dst.type=uint8 tw=2 th=2 max_leaf=24

bin/dynamic_median_filter_cuda_u8_4x4_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_cuda_u8_4x4_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_asserts-disable_llvm_loop_opt-no_runtime src.type=uint8 dst.type=uint8 tw=4 th=4 max_leaf=24

bin/dynamic_median_filter_cuda_u8_8x8_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_cuda_u8_8x8_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_asserts-disable_llvm_loop_opt-no_runtime src.type=uint8 dst.type=uint8 tw=8 th=8 max_leaf=24

bin/dynamic_median_filter_cuda_u16_2x2_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_cuda_u16_2x2_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_asserts-disable_llvm_loop_opt-no_runtime src.type=uint16 dst.type=uint16 tw=2 th=2 max_leaf=24

bin/dynamic_median_filter_cuda_u16_4x4_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_cuda_u16_4x4_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_asserts-disable_llvm_loop_opt-no_runtime src.type=uint16 dst.type=uint16 tw=4 th=4 max_leaf=24

bin/dynamic_median_filter_cuda_u16_8x8_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_cuda_u16_8x8_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_asserts-disable_llvm_loop_opt-no_runtime src.type=uint16 dst.type=uint16 tw=8 th=8 max_leaf=24

bin/dynamic_median_filter_cuda_f32_2x2_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_cuda_f32_2x2_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_asserts-disable_llvm_loop_opt-no_runtime src.type=float32 dst.type=float32 tw=2 th=2 max_leaf=24

bin/dynamic_median_filter_cuda_f32_4x4_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_cuda_f32_4x4_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_asserts-disable_llvm_loop_opt-no_runtime src.type=float32 dst.type=float32 tw=4 th=4 max_leaf=24

bin/dynamic_median_filter_cuda_f32_8x8_v2.a: bin/dynamic_median_filter_generator_v2
	mkdir -p $(@D)
	./bin/dynamic_median_filter_generator_v2 -g dynamic_median_filter -f dynamic_median_filter_cuda_f32_8x8_v2 -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-cuda-cuda_capability_86-no_asserts-disable_llvm_loop_opt-no_runtime src.type=float32 dst.type=float32 tw=8 th=8 max_leaf=24

bin/kim_generator: kim_generator.cpp bin/sorting_network.o bin/sorting_bytecode.o
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -march=native $(HALIDE_DIR)/tools/GenGen.cpp $^ -I $(HALIDE_DIR)/include -L $(HALIDE_DIR)/lib -lHalide -Wl,-rpath,$(HALIDE_DIR)/lib -o bin/kim_generator

bin/kim_median_filter_u8_%.a: bin/kim_generator
	./bin/kim_generator -g kim_median_filter -f kim_median_filter_u8_$* -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-no_runtime src.type=uint8 dst.type=uint8 radius=$* 2>&1 
bin/kim_median_filter_u16_%.a: bin/kim_generator
	./bin/kim_generator -g kim_median_filter -f kim_median_filter_u16_$* -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-no_runtime src.type=uint16 dst.type=uint16 radius=$* 2>&1 
bin/kim_median_filter_f32_%.a: bin/kim_generator
	./bin/kim_generator -g kim_median_filter -f kim_median_filter_f32_$* -o bin -e static_library,h,registration,stmt,assembly,llvm_assembly,h target=$(HL_TARGET)-no_runtime src.type=float32 dst.type=float32 radius=$* 2>&1

bin/runtime.a: bin/median_filter_generator_v2
	./bin/median_filter_generator_v2 -r runtime -o bin target=$(HL_TARGET)-cuda-cuda_capability_86

bin/sorting_network.o: sorting_network.cpp  sorting_network.h
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

bin/sorting_bytecode.o: sorting_bytecode.cpp sorting_bytecode.h
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

bin/ctmf_u16.o: ctmf_u16.cpp ctmf_u16.h ctmf_common.cpp
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -march=skylake -I $(HALIDE_DIR)/include -c $< -o $@

bin/ctmf_u8.o: ctmf_u8.cpp ctmf_u16.h ctmf_common.cpp
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -march=skylake -I $(HALIDE_DIR)/include -c $< -o $@	

bin/dynamic_median_filter.o: dynamic_median_filter.cpp dynamic_median_filter.h \
bin/dynamic_median_filter_u8_2x2_v2.a bin/dynamic_median_filter_u8_4x4_v2.a bin/dynamic_median_filter_u8_8x8_v2.a \
bin/dynamic_median_filter_u16_2x2_v2.a bin/dynamic_median_filter_u16_4x4_v2.a bin/dynamic_median_filter_u16_8x8_v2.a \
bin/dynamic_median_filter_f32_2x2_v2.a bin/dynamic_median_filter_f32_4x4_v2.a bin/dynamic_median_filter_f32_8x8_v2.a \
bin/dynamic_median_filter_cuda_u8_2x2_v2.a bin/dynamic_median_filter_cuda_u8_4x4_v2.a bin/dynamic_median_filter_cuda_u8_8x8_v2.a \
bin/dynamic_median_filter_cuda_u16_2x2_v2.a bin/dynamic_median_filter_cuda_u16_4x4_v2.a bin/dynamic_median_filter_cuda_u16_8x8_v2.a \
bin/dynamic_median_filter_cuda_f32_2x2_v2.a bin/dynamic_median_filter_cuda_f32_4x4_v2.a bin/dynamic_median_filter_cuda_f32_8x8_v2.a
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -I $(HALIDE_DIR)/include -I bin -c $< -o $@

bin/run: run.cpp bin/WaveletMatrix_Cuda_main.o\
bin/sorting_bytecode.o bin/sorting_network.o bin/dynamic_median_filter.o \
bin/ctmf_u8.o bin/ctmf_u16.o \
bin/dynamic_median_filter_u8_2x2_v2.a bin/dynamic_median_filter_u8_4x4_v2.a bin/dynamic_median_filter_u8_8x8_v2.a \
bin/dynamic_median_filter_cuda_u8_2x2_v2.a bin/dynamic_median_filter_cuda_u8_4x4_v2.a bin/dynamic_median_filter_cuda_u8_8x8_v2.a \
bin/dynamic_median_filter_u16_2x2_v2.a bin/dynamic_median_filter_u16_4x4_v2.a bin/dynamic_median_filter_u16_8x8_v2.a \
bin/dynamic_median_filter_cuda_u16_2x2_v2.a bin/dynamic_median_filter_cuda_u16_4x4_v2.a bin/dynamic_median_filter_cuda_u16_8x8_v2.a \
bin/dynamic_median_filter_f32_2x2_v2.a bin/dynamic_median_filter_f32_4x4_v2.a bin/dynamic_median_filter_f32_8x8_v2.a \
bin/dynamic_median_filter_cuda_f32_2x2_v2.a bin/dynamic_median_filter_cuda_f32_4x4_v2.a bin/dynamic_median_filter_cuda_f32_8x8_v2.a \
bin/runtime.a \
bin/kim_median_filter_u8_1.a bin/kim_median_filter_u16_1.a bin/kim_median_filter_f32_1.a \
bin/kim_median_filter_u8_2.a bin/kim_median_filter_u16_2.a bin/kim_median_filter_f32_2.a \
bin/median_filter_u8_1.a bin/median_filter_u16_1.a bin/median_filter_f32_1.a \
bin/median_filter_cuda_u8_1.a bin/median_filter_cuda_u16_1.a bin/median_filter_cuda_f32_1.a \
bin/median_filter_u8_2.a bin/median_filter_u16_2.a bin/median_filter_f32_2.a \
bin/median_filter_cuda_u8_2.a bin/median_filter_cuda_u16_2.a bin/median_filter_cuda_f32_2.a \
bin/median_filter_u8_3.a bin/median_filter_u16_3.a bin/median_filter_f32_3.a \
bin/median_filter_cuda_u8_3.a bin/median_filter_cuda_u16_3.a bin/median_filter_cuda_f32_3.a \
bin/median_filter_u8_4.a bin/median_filter_u16_4.a bin/median_filter_f32_4.a \
bin/median_filter_cuda_u8_4.a bin/median_filter_cuda_u16_4.a bin/median_filter_cuda_f32_4.a \
bin/median_filter_u8_5.a bin/median_filter_u16_5.a bin/median_filter_f32_5.a \
bin/median_filter_cuda_u8_5.a bin/median_filter_cuda_u16_5.a bin/median_filter_cuda_f32_5.a \
bin/median_filter_u8_6.a bin/median_filter_u16_6.a bin/median_filter_f32_6.a \
bin/median_filter_cuda_u8_6.a bin/median_filter_cuda_u16_6.a bin/median_filter_cuda_f32_6.a \
bin/median_filter_u8_7.a bin/median_filter_u16_7.a bin/median_filter_f32_7.a \
bin/median_filter_cuda_u8_7.a bin/median_filter_cuda_u16_7.a bin/median_filter_cuda_f32_7.a \
bin/median_filter_u8_8.a bin/median_filter_u16_8.a bin/median_filter_f32_8.a \
bin/median_filter_cuda_u8_8.a bin/median_filter_cuda_u16_8.a bin/median_filter_cuda_f32_8.a \
bin/median_filter_u8_9.a bin/median_filter_u16_9.a bin/median_filter_f32_9.a  \
bin/median_filter_cuda_u8_9.a bin/median_filter_cuda_u16_9.a bin/median_filter_cuda_f32_9.a  \
bin/median_filter_u8_10.a bin/median_filter_u16_10.a bin/median_filter_f32_10.a \
bin/median_filter_cuda_u8_10.a bin/median_filter_cuda_u16_10.a bin/median_filter_cuda_f32_10.a \
bin/median_filter_u8_11.a bin/median_filter_u16_11.a bin/median_filter_f32_11.a \
bin/median_filter_cuda_u8_11.a bin/median_filter_cuda_u16_11.a bin/median_filter_cuda_f32_11.a \
bin/median_filter_u8_12.a bin/median_filter_u16_12.a bin/median_filter_f32_12.a \
bin/median_filter_cuda_u8_12.a bin/median_filter_cuda_u16_12.a bin/median_filter_cuda_f32_12.a \
bin/median_filter_u8_13.a bin/median_filter_u16_13.a bin/median_filter_f32_13.a \
bin/median_filter_cuda_u8_13.a bin/median_filter_cuda_u16_13.a bin/median_filter_cuda_f32_13.a \
bin/median_filter_u8_14.a bin/median_filter_u16_14.a bin/median_filter_f32_14.a \
bin/median_filter_cuda_u8_14.a bin/median_filter_cuda_u16_14.a bin/median_filter_cuda_f32_14.a  \
bin/median_filter_u8_15.a bin/median_filter_u16_15.a bin/median_filter_f32_15.a \
bin/median_filter_cuda_u8_15.a bin/median_filter_cuda_u16_15.a bin/median_filter_cuda_f32_15.a  \
bin/median_filter_u8_16.a bin/median_filter_u16_16.a bin/median_filter_f32_16.a \
bin/median_filter_cuda_u8_16.a bin/median_filter_cuda_u16_16.a bin/median_filter_cuda_f32_16.a  \
bin/median_filter_u8_17.a bin/median_filter_u16_17.a bin/median_filter_f32_17.a \
bin/median_filter_cuda_u8_17.a bin/median_filter_cuda_u16_17.a bin/median_filter_cuda_f32_17.a  \
bin/median_filter_u8_18.a bin/median_filter_u16_18.a bin/median_filter_f32_18.a \
bin/median_filter_cuda_u8_18.a bin/median_filter_cuda_u16_18.a bin/median_filter_cuda_f32_18.a
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -march=skylake $(CXX_CUDA_FLAGS) $(OPENCV_FLAGS) $(IPP_FLAGS) $(ARRAYFIRE_FLAGS) $(OPENMP_FLAGS) -DMIN_RADIUS=1 -DMAX_RADIUS=18 -mavx2 -I bin -I $(HALIDE_DIR)/tools -I $(HALIDE_DIR)/include $^ -lpthread -ldl -ljpeg $(OPENCV_LIBS) $(IPP_LIBS) $(NPP_LIBS) $(ARRAYFIRE_LIBS) -o $@

bin/superoptimize: superoptimize.cpp bin/sorting_network.o
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -march=native $^ -o $@

bin/leaf_size_experiment: leaf_size_experiment.cpp bin/sorting_network.o bin/sorting_bytecode.o bin/dynamic_median_filter.o \
bin/dynamic_median_filter_u8_2x2_v2.a bin/dynamic_median_filter_u8_4x4_v2.a bin/dynamic_median_filter_u8_8x8_v2.a \
bin/dynamic_median_filter_cuda_u8_2x2_v2.a bin/dynamic_median_filter_cuda_u8_4x4_v2.a bin/dynamic_median_filter_cuda_u8_8x8_v2.a \
bin/dynamic_median_filter_u16_2x2_v2.a bin/dynamic_median_filter_u16_4x4_v2.a bin/dynamic_median_filter_u16_8x8_v2.a \
bin/dynamic_median_filter_cuda_u16_2x2_v2.a bin/dynamic_median_filter_cuda_u16_4x4_v2.a bin/dynamic_median_filter_cuda_u16_8x8_v2.a \
bin/dynamic_median_filter_f32_2x2_v2.a bin/dynamic_median_filter_f32_4x4_v2.a bin/dynamic_median_filter_f32_8x8_v2.a \
bin/dynamic_median_filter_cuda_f32_2x2_v2.a bin/dynamic_median_filter_cuda_f32_4x4_v2.a bin/dynamic_median_filter_cuda_f32_8x8_v2.a \
bin/runtime.a 
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -I $(HALIDE_DIR)/include -I $(HALIDE_DIR)/tools -march=native $^ -o $@ -lpthread -ldl -ljpeg -lpng 

test: bin/run
	./bin/run

clean:
	rm -f bin/*


