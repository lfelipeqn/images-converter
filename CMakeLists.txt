cmake_minimum_required(VERSION 3.10)
project(images-processor)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV (version 4 or newer)
find_package(OpenCV 4 REQUIRED)

# Find the Google Cloud Storage component (installed via vcpkg)
find_package(google_cloud_cpp_storage CONFIG REQUIRED)

add_executable(images-processor images-processor.cpp)

# Link against OpenCV, libcurl, and the google-cloud-cpp storage component
target_link_libraries(images-processor PRIVATE ${OpenCV_LIBS} CURL::libcurl google-cloud-cpp::storage)
