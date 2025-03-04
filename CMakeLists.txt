cmake_minimum_required(VERSION 3.10)
project(ImageProcessor)

find_package(OpenCV REQUIRED)

add_executable(images-processor images-processor.cpp)
target_link_libraries(images-processor ${OpenCV_LIBS})
