cmake_minimum_required(VERSION 3.10)
project(BA)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${OpenCV_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS}
                    ${Eigen3_INCLUDE_DIRS})
link_directories(${PROJECT_SOURCE_DIR}/lib)
add_executable(BA main.cpp)
target_link_libraries(BA ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES})
