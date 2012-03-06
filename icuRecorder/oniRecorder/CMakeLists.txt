cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})


project(icuRecorder)

find_package(OpenCV REQUIRED)
find_package(OpenNI REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${OPENNI_INCLUDE_DIRS})
include_directories(${GLUT_LIBRARY})
#link_directories(${OPENNI_LIBRARIES})
add_definitions(${OPENNI_DEFINITIONS})



#message(${OPENNI_LIBRARIES})
#message(${OpenCV_INCLUDE_DIRS})
message(${CMAKE_SOURCE_DIR})

add_executable (icuRecorder main.cpp opengles.cpp opengles.h SceneDrawer.cpp SceneDrawer.h)

target_link_libraries (icuRecorder ${OpenCV_LIBS} ${OPENNI_LIBRARY} ${GLUT_LIBRARY} ${OPENGL_LIBRARIES})


