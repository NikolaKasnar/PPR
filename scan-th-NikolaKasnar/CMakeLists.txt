cmake_minimum_required(VERSION 3.20)
project(prefixsum)

find_package(Threads REQUIRED)

add_executable(main main.cpp parallel_scan.cpp aux.h random_int.h clock.h)
target_link_libraries(main Threads::Threads)
set_target_properties(main PROPERTIES CXX_STANDARD 20)
