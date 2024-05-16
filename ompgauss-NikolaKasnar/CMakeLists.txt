cmake_minimum_required(VERSION 3.15)
project(Gauss)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
#add_compile_options("-fopenmp")

# Jedan izvorni kod za paralelnu i serijsku verziju.
# paralelna verzija
add_executable(main_par gauss.cpp clock.h)
target_compile_options(main_par PRIVATE ${OpenMP_CXX_FLAGS})
target_link_options(main_par PRIVATE ${OpenMP_CXX_FLAGS})

# serijska verzija
add_executable(main_seq gauss.cpp clock.h)
