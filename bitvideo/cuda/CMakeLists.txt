# ===========================================================================
#  BitVideo-1.58  --  W1.58A8 ternary GEMM/GEMV kernels
#
#  This CMake project builds the kernels WITHOUT any PyTorch dependency, which
#  makes them usable from plain C++/CUDA applications (TensorRT plugins,
#  llama.cpp style runtimes, ...) and makes the GoogleTest suite cheap to run.
#
#  The PyTorch bridge (bit_gemm.cpp / kernels.cpp) is only compiled when a
#  libtorch installation is provided via -DTorch_DIR=/path/to/share/cmake/Torch
#
#  Layering
#  --------
#     cuda_helpers.h      arch macros, CUDA_CHECK, cp.async, warp primitives
#     bit_utils.h         2-bit ternary codec, layouts, PRMT decode   (host+dev)
#     bit_gemm_kernel.h   public C ABI  (no CUDA types in the signatures)
#     bit_gemm_kernel.cu  GEMV + DP4A GEMM + MMA (tensor core) GEMM
#     ternary_gemm.cu     ternarisation / packing / layout conversion kernels
#     int8_gemm.cu        INT8xINT8 GEMM + activation quantisation kernels
#     bit_gemm_launch.cu  heuristics, autotuner, dispatch
#     bit_packing.cpp     host reference packer (pure C++, no CUDA)
#
#  Build
#  -----
#     cmake -S bitvideo/cuda -B build -DCMAKE_BUILD_TYPE=Release \
#           -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120"
#     cmake --build build -j
#     ctest --test-dir build --output-on-failure
# ===========================================================================
cmake_minimum_required(VERSION 3.24)

project(bit_gemm
        VERSION 1.58.0
        DESCRIPTION "Ternary W1.58A8 GEMM kernels for BitNet-style linear layers"
        LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
option(BITVIDEO_BUILD_TESTS    "Build the GoogleTest suite"            ON)
option(BITVIDEO_BUILD_TORCH    "Build the PyTorch bridge"              OFF)
option(BITVIDEO_BUILD_BENCH    "Build the standalone C++ benchmark"    ON)
option(BITVIDEO_USE_FAST_MATH  "Enable --use_fast_math"                ON)
option(BITVIDEO_VERBOSE_PTXAS  "Print register/smem usage per kernel"  ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Default fat binary: Ampere -> Ada -> Hopper, plus consumer Blackwell when
# the compiler understands SM120 (CUDA 12.8+).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(_bitvideo_default_arches 80 86 89 90)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
        list(APPEND _bitvideo_default_arches 100 120)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_bitvideo_default_arches}" CACHE STRING
        "CUDA architectures for BitVideo kernels" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS 12.0)
    message(FATAL_ERROR "BitVideo requires CUDA 12.0 or newer (found ${CUDAToolkit_VERSION})")
endif()
if(("100" IN_LIST CMAKE_CUDA_ARCHITECTURES OR "120" IN_LIST CMAKE_CUDA_ARCHITECTURES)
   AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR "SM100/SM120 require CUDA 12.8+ (compiler is ${CMAKE_CUDA_COMPILER_VERSION})")
endif()
message(STATUS "[bitvideo] CUDA ${CUDAToolkit_VERSION}  arch=${CMAKE_CUDA_ARCHITECTURES}")

# ---------------------------------------------------------------------------
# core kernel library (no torch)
# ---------------------------------------------------------------------------
add_library(bit_gemm_core STATIC
        bit_gemm_kernel.cu
        bit_gemm_launch.cu
        ternary_gemm.cu
        int8_gemm.cu
        bit_packing.cpp)

target_include_directories(bit_gemm_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(bit_gemm_core PUBLIC CUDA::cudart)

set_target_properties(bit_gemm_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF     # single-TU kernels: better inlining
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(bit_gemm_core PUBLIC BITVIDEO_WITH_CUDA=1)

target_compile_options(bit_gemm_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
        $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
        $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF2_OPERATORS__>
        $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>)

if(BITVIDEO_USE_FAST_MATH)
    target_compile_options(bit_gemm_core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

if(BITVIDEO_VERBOSE_PTXAS)
    target_compile_options(bit_gemm_core PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()

if(MSVC)
    target_compile_options(bit_gemm_core PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
            $<$<COMPILE_LANGUAGE:CUDA>:-DNOMINMAX>)
endif()

# Convenience alias mirroring the python extension name.
add_library(bit_gemm::core ALIAS bit_gemm_core)

# ---------------------------------------------------------------------------
# optional PyTorch bridge
# ---------------------------------------------------------------------------
if(BITVIDEO_BUILD_TORCH)
    find_package(Torch REQUIRED)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    add_library(bitvideo_C SHARED bit_gemm.cpp kernels.cpp)
    target_link_libraries(bitvideo_C PRIVATE bit_gemm_core "${TORCH_LIBRARIES}" Python3::Python)
    target_compile_definitions(bitvideo_C PRIVATE TORCH_EXTENSION_NAME=_C)
    set_target_properties(bitvideo_C PROPERTIES PREFIX "" OUTPUT_NAME "_C")
    message(STATUS "[bitvideo] PyTorch bridge enabled (${TORCH_INSTALL_PREFIX})")
endif()

# ---------------------------------------------------------------------------
# standalone benchmark
# ---------------------------------------------------------------------------
if(BITVIDEO_BUILD_BENCH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_main.cpp)
    add_executable(bit_gemm_bench tests/bench_main.cpp)
    target_link_libraries(bit_gemm_bench PRIVATE bit_gemm_core)
endif()

# ---------------------------------------------------------------------------
# GoogleTest suite
# ---------------------------------------------------------------------------
if(BITVIDEO_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
            GIT_SHALLOW TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)

    add_executable(test_bit_gemm tests/test_bit_gemm.cpp)
    target_link_libraries(test_bit_gemm PRIVATE bit_gemm_core GTest::gtest_main)
    set_target_properties(test_bit_gemm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    gtest_discover_tests(test_bit_gemm
            PROPERTIES TIMEOUT 300
            DISCOVERY_MODE PRE_TEST)
endif()

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS bit_gemm_core
        EXPORT bit_gemmTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
        bit_gemm_kernel.h
        bit_packing.h
        bit_utils.h
        cuda_helpers.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bit_gemm)
install(EXPORT bit_gemmTargets
        NAMESPACE bit_gemm::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bit_gemm)
