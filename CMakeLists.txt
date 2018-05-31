cmake_minimum_required(VERSION 3.10)
project(hi)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_subdirectory(lib/dlib-19.13 dlib-build)

add_executable(hi src/main.cpp src/Hi.cpp src/Hi.h src/HiCamera.cpp src/HiCamera.h)

target_link_libraries(hi ${OpenCV_LIBS})
target_link_libraries(hi dlib::dlib)
target_link_libraries(hi stdc++fs)
