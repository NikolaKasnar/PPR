cmake_minimum_required(VERSION 3.1)

project(cuda-bfs CXX CUDA)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)


set(CMAKE_CUDA_ARCHITECTURES native)

add_executable("main" main.cu labirint_io.cpp bfs.cpp sparse.cpp 
                      aux.cpp sparse.h bfs.h labirint_io.h)

