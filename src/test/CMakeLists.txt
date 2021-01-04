cmake_minimum_required(VERSION 3.0)

add_executable(testRenderer testRenderer.cpp)
target_link_libraries(testRenderer renderer ${THIRD_LIBS})
