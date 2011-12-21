cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(recorder)

find_package(PCL REQUIRED)


#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

add_executable (main main.cpp)

  ##target_link_libraries (oni2depth ${PCL_RANGE_IMAGE_LIBRARIES})
  #target_link_libraries (main ${PCL_VISUALIZATION_LIBRARIES})

#target_link_libraries(main.cpp pcl_common pcl_io pcl_kdtree pcl_range_image pcl_visualization)

