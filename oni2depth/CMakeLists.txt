cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(oni2depth)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBS})

message(${OpenCV_LIBS})


add_executable (oni2depth oni2depth.cpp)
target_link_libraries(oni2depth ${PCL_LIBRARIES} ${OpenCV_LIBS})
