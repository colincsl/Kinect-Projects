cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


project(gpu_tests)

find_package(CUDA REQUIRED)
find_package(PCL REQUIRED)

set(CUDA_ARCH_BIN 1.3)
set(CUDA_ARCH_PTX 1.3)


include_directories(${PCL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories("~/NVIDIA_GPU_Computing_SDK/C/common/inc")

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

CUDA_ADD_EXECUTABLE(MLS_test MLS_test MLS.cu)
target_link_libraries (MLS_test  ${PCL_LIBRARIES} ${PCL_CUDA_COMMON_LIBRARIES} )

