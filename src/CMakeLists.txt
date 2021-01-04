cmake_minimum_required(VERSION 3.0)

set(headers
    depth.h
    rendering.h)

set(sources
    depth.cpp
    rendering.cpp)

#include_directories(${PROJECT_SOURCE_DIR}/src)

add_library(renderer SHARED depth.cpp rendering.cpp)
target_link_libraries(renderer ${THIRD_LIBS} )

add_subdirectory(test)
