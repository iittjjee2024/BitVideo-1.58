# ===========================================================================
#  BitVideo-1.58  --  top level CMake entry point
#
#  Thin wrapper that forwards to the kernel project in bitvideo/cuda so that
#  `cmake -S . -B build` works from the repository root.
#
#     cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#                         -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120"
#     cmake --build build -j
#     ctest --test-dir build --output-on-failure
# ===========================================================================
cmake_minimum_required(VERSION 3.24)
project(bitvideo LANGUAGES NONE)

add_subdirectory(bitvideo/cuda)
