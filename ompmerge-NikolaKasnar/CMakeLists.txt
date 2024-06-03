cmake_minimum_required(VERSION 3.13)
project(merge)


find_package(OpenMP REQUIRED)
#add_compile_options("-fopenmp")
add_compile_options(${OpenMP_CXX_FLAGS})
add_link_options(${OpenMP_CXX_FLAGS})

add_executable(main main.cpp)
