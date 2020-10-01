cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0076 NEW)

project(sssp)

find_package(TAPA REQUIRED)
find_package(FRT REQUIRED)
find_package(OpenMP)

add_executable(sssp)
target_sources(sssp PRIVATE src/sssp-host.cpp src/util.h)
target_compile_features(sssp PRIVATE cxx_std_17)
target_link_libraries(sssp PRIVATE tapa::tapa)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sssp PUBLIC OpenMP::OpenMP_CXX)
endif()
