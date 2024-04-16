cmake_minimum_required(VERSION 3.20)
project(floyd)

find_package(Threads REQUIRED)
add_executable(main main.cpp floyd.cpp)
target_link_libraries(main Threads::Threads)
set_target_properties(main PROPERTIES CXX_STANDARD 20)
