cmake_minimum_required(VERSION 3.10)
project(images-processor)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Packages
find_package(OpenCV 4 REQUIRED)
find_package(CURL REQUIRED)
find_package(google_cloud_cpp_storage CONFIG REQUIRED)

# Enable gnu++23 for __int128 support (required by abseil-cpp)
add_compile_options(-std=gnu++23)

# Add additional linker flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")

# Define executable
add_executable(images-processor images-processor.cpp)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Link Libraries (order matters!)
target_link_libraries(images-processor PRIVATE
    ${OpenCV_LIBS}
    CURL::libcurl
    google-cloud-cpp::storage
    -lssl
    -lcrypto
)
