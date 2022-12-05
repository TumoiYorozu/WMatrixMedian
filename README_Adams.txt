* Requirements

Building this program unfortunately has a lot of dependencies, because
we compare to a lot of things in the paper. Most of these can be
disabled by setting TEST_* in run.cpp to zero.

The dependencies are:

- Halide master commit d4c27ca30a9ce9181d837c4cb33b53c9e17916b3 

- Intel IPP version 2020.4.304

- OpenCV commit 941a9792f7c7d60640c82926091d710e6ecbfb55
(Make sure to include the contrib modules to get Green's O(1) CUDA median filter.)

- ArrayFire commit a004f5352e71d5b4b540684e0b3f6149e548079e

- NPP from cuda 10.1.243-3 (default on ubuntu 20.04)

* Building

First edit the Makefile to adjust variables at the top to set paths to
opencv, arrayfire, and ipp, and your C++ compiler.

To build the benchmark:
% make bin/run -j16

This takes about 30 mins to build all the variants on my machine.

* Running

Once it has built, the following line will benchmark median filters
within the specified radii:

% ./bin/run min_radius max_radius

To run it, you may need to add some of the library install paths from
the packages above to your LD_LIBRARY_PATH, depending on your linker.

In the output, "cpu" or "cuda" means our CPU and GPU AOT-compiled
version. "dynamic_v2" and "dynamic_cuda" are our CPU and GPU sorting
network interpreters respectively. Outputs are also dumped as .csv
files, which is what we used to generate the performance figures in
the paper.

* Other experiments

leaf_size_experiment.cpp measures the effect of instruction size limit
on performance.

mcguire_counterexample.py uses a SAT solver to find a counterexample
to McGuire's 5x5 median filter.

superoptimize.cpp is our tool for generating superoptimized small
sorting networks. Note that it's only optimal at sizes 7 and below.

sorting_network_to_svg.cpp is a tool for dumping sorting networks to
svg file, used in the preparation of the paper figure illustrating
inflating a network.
