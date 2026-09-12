cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_RUN_MODE "sim" CACHE STRING "Run mode for profiling harness")
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "Target NPU architecture")

find_package(ASC REQUIRED)

project(topk_profile_one LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(topk_profile_one
    topk_profile_one.asc
)

target_link_libraries(topk_profile_one PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_compile_options(topk_profile_one PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    $<$<COMPILE_LANGUAGE:ASC>:-g>
    $<$<COMPILE_LANGUAGE:ASC>:-O2>
)
