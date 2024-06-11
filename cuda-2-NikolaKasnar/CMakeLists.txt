cmake_minimum_required(VERSION 3.8)
project(cuda-2)

enable_language(C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(main main.cu)
