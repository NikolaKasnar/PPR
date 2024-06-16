cmake_minimum_required(VERSION 3.24)
project(cuda)

enable_language(C CXX CUDA)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Ovo je nužno na laptopu leny-2 gdje je default gcc 13
set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

set(CMAKE_CUDA_ARCHITECTURES native)
set (CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}; "-lineinfo")

add_executable(test_matrix test_matrix.cpp matrix.h)
add_executable(test_csrmat test_csrmat.cpp csr_mat.h matrix.h)

add_executable(mat-vec-csr            mat-vec-csr.cu csr_mat.h)
add_executable(mat-vec-csr-lap        mat-vec-csr-lap.cu csr_mat.h lapmat.h)
add_executable(mat-vec-csr-lap-um     mat-vec-csr-lap-um.cu csr_mat_um.h lapmat.h)
add_executable(mat-vec-csr-lap-um-pre mat-vec-csr-lap-um-pre.cu csr_mat_um_pre.h lapmat.h) 

add_executable(um_example_1 um_example_1.cu)
add_executable(um_example_2 um_example_2.cu)
add_executable(um_example_3 um_example_3.cu)
