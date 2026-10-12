cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_RUN_MODE "cpu" CACHE STRING "Run mode for the diagnostic probe")
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "Target NPU architecture")

find_package(ASC REQUIRED)

project(topk_hierarchical_probe LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(topk_hierarchical_probe
    topk_hierarchical_probe.asc
)

add_executable(topk_prefix_merge_probe
    topk_prefix_merge_probe.asc
)

add_executable(topk_prefix_merge_padding_probe
    topk_prefix_merge_padding_probe.asc
)

add_executable(topk_prefix_production_probe
    topk_prefix_production_probe.asc
)

add_executable(topk_dense_repeat_probe
    topk_dense_repeat_probe.asc
)

add_executable(topk_fp16_vendor_probe
    topk_fp16_vendor_probe.asc
)

add_executable(topk_fp16_vendor_production_probe
    topk_fp16_vendor_production_probe.asc
)

add_executable(topk_fp32_vendor_production_probe
    topk_fp32_vendor_production_probe.asc
)

add_executable(topk_fp16_vendor_medium_probe
    topk_fp16_vendor_medium_probe.asc
)

add_executable(topk_fp32_vendor_large_probe
    topk_fp32_vendor_large_probe.asc
)

add_executable(topk_fp16_vendor_large_probe
    topk_fp16_vendor_large_probe.asc
)

add_executable(topk_fp16_large_production_probe topk_fp16_large_production_probe.asc)

add_executable(topk_fp32_large_production_probe topk_fp32_large_production_probe.asc)
add_executable(topk_fp16_compact_production_probe topk_fp16_compact_production_probe.asc)

add_executable(topk_threshold_compact_probe topk_threshold_compact_probe.asc)

add_executable(topk_threshold_full_probe topk_threshold_full_probe.asc)
add_executable(topk_threshold_baseline_probe topk_threshold_full_probe.asc)
target_compile_definitions(topk_threshold_baseline_probe PRIVATE THRESHOLD_BASELINE=1)

add_executable(topk_fp32_medium_production_probe topk_fp32_medium_production_probe.asc)

add_executable(topk_values_only_probe topk_values_only_probe.asc)
add_executable(topk_values_only_baseline_probe topk_values_only_probe.asc)
target_compile_definitions(topk_values_only_baseline_probe PRIVATE VALUES_ONLY_BASELINE=1)

add_executable(topk_values_only_raw_probe topk_values_only_probe.asc)
target_compile_definitions(topk_values_only_raw_probe PRIVATE VALUES_ONLY_RAW=1)

add_executable(topk_fp32_scratch_production_probe topk_fp32_scratch_production_probe.asc)

foreach(probe_target IN ITEMS
        topk_fp16_large_production_probe
        topk_fp16_compact_production_probe
        topk_fp32_scratch_production_probe
        topk_values_only_raw_probe
        topk_values_only_probe
        topk_values_only_baseline_probe
        topk_fp32_medium_production_probe
        topk_threshold_full_probe
        topk_threshold_baseline_probe
        topk_threshold_compact_probe
        topk_fp32_large_production_probe
        topk_hierarchical_probe
        topk_prefix_merge_probe
        topk_prefix_merge_padding_probe
        topk_prefix_production_probe
        topk_dense_repeat_probe
        topk_fp16_vendor_probe
        topk_fp16_vendor_production_probe
        topk_fp32_vendor_production_probe
        topk_fp16_vendor_medium_probe
        topk_fp32_vendor_large_probe
        topk_fp16_vendor_large_probe)
    target_link_libraries(${probe_target} PRIVATE
        tiling_api
        register
        platform
        unified_dlog
        dl
        m
        graph_base
    )

    target_compile_options(${probe_target} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    )
endforeach()

foreach(kind IN ITEMS full raw baseline)
 add_executable(topk_reduction_${kind} topk_reduction_probe.asc)
 target_link_libraries(topk_reduction_${kind} PRIVATE tiling_api register platform unified_dlog dl m graph_base)
 target_compile_options(topk_reduction_${kind} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
endforeach()
target_compile_definitions(topk_reduction_raw PRIVATE VALUES_ONLY_RAW=1)
target_compile_definitions(topk_reduction_baseline PRIVATE VALUES_ONLY_BASELINE=1)

foreach(kind IN ITEMS full raw baseline)
 add_executable(topk_block_reduce_${kind} topk_block_reduce_probe.asc)
 target_link_libraries(topk_block_reduce_${kind} PRIVATE tiling_api register platform unified_dlog dl m graph_base)
 target_compile_options(topk_block_reduce_${kind} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
endforeach()
target_compile_definitions(topk_block_reduce_raw PRIVATE VALUES_ONLY_RAW=1)
target_compile_definitions(topk_block_reduce_baseline PRIVATE VALUES_ONLY_BASELINE=1)

add_executable(topk_c12_block_production_probe topk_c12_block_production_probe.asc)
target_link_libraries(topk_c12_block_production_probe PRIVATE tiling_api register platform unified_dlog dl m graph_base)
target_compile_options(topk_c12_block_production_probe PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

foreach(kind IN ITEMS full raw baseline)
 add_executable(topk_c14_block_${kind} topk_c14_block_reduce_probe.asc)
 target_link_libraries(topk_c14_block_${kind} PRIVATE tiling_api register platform unified_dlog dl m graph_base)
 target_compile_options(topk_c14_block_${kind} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
endforeach()
target_compile_definitions(topk_c14_block_raw PRIVATE VALUES_ONLY_RAW=1)
target_compile_definitions(topk_c14_block_baseline PRIVATE VALUES_ONLY_BASELINE=1)

add_executable(topk_c14_block_production_probe topk_c14_block_production_probe.asc)
target_link_libraries(topk_c14_block_production_probe PRIVATE tiling_api register platform unified_dlog dl m graph_base)
target_compile_options(topk_c14_block_production_probe PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
