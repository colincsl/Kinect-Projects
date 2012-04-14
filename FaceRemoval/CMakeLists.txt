cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(recorder)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBS})

add_executable (main main.cpp)
add_executable (reader CloudSequenceReader.cpp)
add_executable (remover FaceRemovalSequenceReader)

target_link_libraries (main ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries (reader ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries (remover ${PCL_LIBRARIES} ${OpenCV_LIBS})


