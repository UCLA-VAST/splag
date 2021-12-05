cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0076 NEW)

project(sssp)

find_package(gflags REQUIRED)
find_package(TAPA REQUIRED)

include(FetchContent)

FetchContent_Declare(
  ap_int
  GIT_REPOSITORY https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git
  GIT_TAG 200a9aecaadf471592558540dc5a88256cbf880f # 2019/03/18
)

FetchContent_MakeAvailable(ap_int)

add_library(sssp_kernel)
target_sources(sssp_kernel PRIVATE src/sssp.cpp)
target_compile_options(sssp_kernel PUBLIC -I${ap_int_SOURCE_DIR}/include)
target_compile_features(sssp_kernel PUBLIC cxx_std_17)
target_link_libraries(sssp_kernel PUBLIC tapa::tapa)

add_executable(sssp)
target_sources(sssp PRIVATE src/sssp-host.cpp)
target_compile_features(sssp PRIVATE cxx_std_17)
target_link_libraries(sssp PRIVATE sssp_kernel tapa::tapa gflags cnpy)

find_package(SDx)
if(SDx_FOUND)
  add_tapa_target(
    sssp-hw-xo
    INPUT src/sssp.cpp
    TOP SSSP
    DIRECTIVE ${CMAKE_CURRENT_SOURCE_DIR}/directive.json
    CONSTRAINT ${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
    CLOCK_PERIOD 6.250
    PLATFORM xilinx_u250_xdma_201830_2)

  include(ProcessorCount)
  ProcessorCount(NPROC)
  if(NPROC EQUAL 0)
    set(NPROC 8)
  endif()

  add_xocc_hw_link_targets(
    ${CMAKE_CURRENT_BINARY_DIR}
    --jobs=${NPROC}
    --remote_ip_cache=$ENV{HOME}/.remote_ip_cache
    --config=${CMAKE_CURRENT_SOURCE_DIR}/link_config.ini
    --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
    --kernel_frequency=160
    INPUT sssp-hw-xo
    HW_EMU_XCLBIN hw_emu_xclbin_target
    HW_XCLBIN hw_xclbin_target)

  add_custom_target(
    sssp-cosim
    COMMAND
      $<TARGET_FILE:sssp> ${CMAKE_CURRENT_SOURCE_DIR}/data/graph500-scale-5
      --bitstream=$<TARGET_PROPERTY:${hw_emu_xclbin_target},FILE_NAME>
    DEPENDS sssp ${hw_emu_xclbin_target}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
