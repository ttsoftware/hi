cmake_minimum_required(VERSION 3.10)
project(hi)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/dlib-19.13 dlib_build)

add_executable(hi src/main.cpp)
#add_executable(hi src/dnn_face_recognition_ex.cpp)

target_link_libraries(hi dlib::dlib)