cmake_minimum_required(VERSION 3.16)
project(topk_profile_one LANGUAGES CXX)

set(RUN_MODE "sim" CACHE STRING "cpu/sim/npu")
# ascendc_library 使用编译器 SoC 标识；msprof simulator 型号由外层脚本单独传递。
set(SOC_VERSION "ascend910b1" CACHE STRING "compiler SoC type")
set(PROFILE_VARIANT "baseline" CACHE STRING "baseline/pivot")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE STRING "CANN install root")

if(NOT ASCEND_CANN_PACKAGE_PATH)
    message(FATAL_ERROR "ASCEND_HOME_PATH is empty; source CANN set_env.sh first")
endif()

find_package(ASC REQUIRED)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found under ${ASCEND_CANN_PACKAGE_PATH}")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# CANN 9.0 legacy ascendc precompile 只会把常规 C/C++ 源传给
# extract_src_template.py；临时 device 源因此生成成 .cpp，而不是仓库里的 .asc 后缀。
ascendc_library(ascendc_kernels_${RUN_MODE} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_profile.cpp
)

# production kernel 顶部仍包含 tiling/tiling_api.h；它会继续引用
# register/tilingdata_base.h。普通 host 工程通过 find_package(ASC) 能找到它，
# 但 ascendc_library 的 device precompile target 不自动继承该搜索路径。
# 使用官方 ascendc_include_directories 为 kernel target 显式补齐。
find_path(ASCEND_REGISTER_INCLUDE_DIR
    NAMES register/tilingdata_base.h
    PATHS ${ASCEND_CANN_PACKAGE_PATH}
    PATH_SUFFIXES
        x86_64-linux/include
        aarch64-linux/include
        include
    NO_DEFAULT_PATH
)
if(NOT ASCEND_REGISTER_INCLUDE_DIR)
    message(FATAL_ERROR
        "register/tilingdata_base.h not found under ${ASCEND_CANN_PACKAGE_PATH}; "
        "check CANN installation")
endif()
message(STATUS "Ascend register include: ${ASCEND_REGISTER_INCLUDE_DIR}")
ascendc_include_directories(ascendc_kernels_${RUN_MODE} PRIVATE
    ${ASCEND_REGISTER_INCLUDE_DIR}
)

ascendc_compile_options(ascendc_kernels_${RUN_MODE} PRIVATE
    -g
    -O2
)
ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    -DASCENDC_TRACE_ON
)

add_executable(topk_profile_one
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

set_target_properties(topk_profile_one PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_compile_options(topk_profile_one PRIVATE
    -O2
    -D_GLIBCXX_USE_CXX11_ABI=0
)

if(PROFILE_VARIANT STREQUAL "pivot")
    target_compile_definitions(topk_profile_one PRIVATE PROFILE_PIVOT=1)
elseif(NOT PROFILE_VARIANT STREQUAL "baseline")
    message(FATAL_ERROR "PROFILE_VARIANT must be baseline or pivot")
endif()

target_link_libraries(topk_profile_one PRIVATE
    host_intf_pub
    ascendc_kernels_${RUN_MODE}
    graph_base
    tiling_api
    register
    platform
    ascendalog
    dl
    m
)
