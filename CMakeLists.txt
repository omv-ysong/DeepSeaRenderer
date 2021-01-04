cmake_minimum_required(VERSION 3.0)

set(CAMKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

project(DeepSeaRenderer)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package( yaml-cpp REQUIRED )

find_package( OpenCV REQUIRED )

find_package( Eigen3 REQUIRED )
if (EIGEN3_FOUND)
    message(STATUS "**Found Eigen")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    message(STATUS "**Found OpenMP")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(THIRD_LIBS
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
    Eigen3::Eigen)

#include_directories(${PROJECT_SOURCE_DIR}/src)
message(${CMAKE_CXX_FLAGS})

add_subdirectory(src)

