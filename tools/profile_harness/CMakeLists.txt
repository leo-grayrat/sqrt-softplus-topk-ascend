cmake_minimum_required(VERSION 3.16)

# profiling 使用 CANN 9.0 原生 ASC 语言构建，而不是 legacy ascendc_library。
# 后者会在 AIV auto-gen 阶段把 __global__ 重定义为 inline，与 production
# 的 __global__ __vector__ kernel 入口冲突。
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU architecture")

find_package(ASC REQUIRED)
project(topk_profile_one LANGUAGES ASC CXX)

add_executable(topk_profile_one
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_profile.asc
)

target_compile_options(topk_profile_one PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES} -g -O2>
)

target_compile_definitions(topk_profile_one PRIVATE
    ASCENDC_TRACE_ON
)

target_link_libraries(topk_profile_one PRIVATE
    dl
    m
)
