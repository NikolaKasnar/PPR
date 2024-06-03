cmake_minimum_required(VERSION 3.24)
project(cuda-matmu)

enable_language(C CXX CUDA)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

# Ovo je nužno na laptopu leny-2 gdje je default gcc 13
#set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(main  main.cu)
