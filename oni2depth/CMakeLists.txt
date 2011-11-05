cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(oni2depth)

find_package(PCL 1.2 REQUIRED)

if(WXWIDGETS_FOUND)
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})

  add_executable (oni2depth oni2depth.cpp)
  ##target_link_libraries (oni2depth ${PCL_RANGE_IMAGE_LIBRARIES})
  ##target_link_libraries (oni2depth ${PCL_VISUALIZATION_LIBRARIES})
  target_link_libraries(oni2depth pcl_common pcl_io pcl_kdtree
        pcl_range_image pcl_visualization)
else(WXWIDGETS_FOUND)
  message("WxWidgets is required to build ${PROJECT_NAME}.")
endif(WXWIDGETS_FOUND)

