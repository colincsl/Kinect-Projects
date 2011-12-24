cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})


project(recorder)

find_package(OpenCV REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
#link_directories(${OpenCV_LIBS})


add_executable (main main.cpp)

target_link_libraries (main ${OpenCV_LIBS})

