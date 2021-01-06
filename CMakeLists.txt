cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0076 NEW)

project(sssp)

find_package(gflags REQUIRED)
find_package(TAPA REQUIRED)
find_package(FRT REQUIRED)
find_package(OpenMP)

add_library(sssp_kernel)
target_sources(sssp_kernel PRIVATE src/sssp.cpp)
target_compile_features(sssp_kernel PUBLIC cxx_std_11)
target_link_libraries(sssp_kernel PUBLIC tapa::tapa)

add_executable(sssp)
target_sources(sssp PRIVATE src/sssp-host.cpp)
target_compile_features(sssp PRIVATE cxx_std_17)
target_link_libraries(sssp PRIVATE sssp_kernel tapa::tapa gflags)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sssp PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(sssp_kernel PRIVATE OpenMP::OpenMP_CXX)
endif()

add_tapa_target(
  sssp-hw-xo
  INPUT src/sssp.cpp
  TOP SSSP
  CONSTRAINT ${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
  PLATFORM xilinx_u280_xdma_201920_3)

include(ProcessorCount)
processorcount(NPROC)
if(NPROC EQUAL 0)
  set(NPROC 8)
endif()

add_xocc_hw_link_targets(
  ${CMAKE_CURRENT_BINARY_DIR}
  --jobs=${NPROC}
  --remote_ip_cache=$ENV{HOME}/.remote_ip_cache
  --config=${CMAKE_CURRENT_SOURCE_DIR}/link_config.ini
  --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
  INPUT sssp-hw-xo
  HW_EMU_XCLBIN hw_emu_xclbin_target
  HW_XCLBIN hw_xclbin_target)

add_executable(sssp-frt)
target_compile_features(sssp-frt PRIVATE cxx_std_17)
target_sources(sssp-frt PRIVATE src/sssp-frt.cpp src/sssp-host.cpp)
target_link_libraries(sssp-frt PRIVATE frt::frt tapa::tapa gflags)

add_custom_target(
  sssp-cosim
  COMMAND
    BITSTREAM=$<TARGET_PROPERTY:${hw_emu_xclbin_target},FILE_NAME>
    $<TARGET_FILE:sssp-frt> ${CMAKE_CURRENT_SOURCE_DIR}/data/graph500-scale-5
  DEPENDS sssp-frt ${hw_emu_xclbin_target}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
