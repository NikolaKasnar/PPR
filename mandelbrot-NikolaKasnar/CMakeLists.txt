cmake_minimum_required(VERSION 3.20)
project(maldenbrot)

add_executable(mandelbrot mandelbrot.cpp)
set_property(TARGET mandelbrot PROPERTY CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
  PNGwriter
  GIT_REPOSITORY https://github.com/pngwriter/pngwriter
  GIT_TAG 0.7.0
)
FetchContent_MakeAvailable(PNGwriter)

target_link_libraries(mandelbrot PRIVATE PNGwriter)
