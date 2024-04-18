cmake_minimum_required(VERSION 3.19)
project(ts_list)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(main main.cpp ts_list.h)
target_link_libraries(main  Threads::Threads)

add_executable(tst test.cpp ts_list.h)
target_link_libraries(tst  Threads::Threads)
