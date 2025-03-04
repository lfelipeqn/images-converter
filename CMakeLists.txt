cmake_minimum_required(VERSION 3.10)
project(images-processor)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Let CMake find OpenCV automatically (most reliable).  Consider specifying version if needed
find_package(OpenCV 4 REQUIRED) # Or find_package(OpenCV REQUIRED) if you don't care about version

add_executable(images-processor images-processor.cpp)

# Link against the OpenCV library (CMake will resolve the correct library names).
target_link_libraries(images-processor ${OpenCV_LIBS})
