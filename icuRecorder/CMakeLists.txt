cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})


project(icuRecorder)

find_package(OpenCV REQUIRED)
find_package(OpenNI REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
#link_directories(${OpenCV_LIBS})

include_directories(${OPENNI_INCLUDE_DIRS})
#link_directories(${OPENNI_LIBRARIES})
add_definitions(${OPENNI_DEFINITIONS})

message(${OPENNI_LIBRARIES})
message(${OpenCV_INCLUDE_DIRS})


add_executable (icuRecorder icuRecorder.cpp kinectstorage.h kinectstorage.cpp)

target_link_libraries (icuRecorder ${OpenCV_LIBS} ${OPENNI_LIBRARY})


