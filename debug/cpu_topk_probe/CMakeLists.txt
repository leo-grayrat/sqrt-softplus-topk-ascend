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

foreach(kind IN ITEMS full raw baseline baseline_raw copy)
 add_executable(topk_c15_group_${kind} topk_c15_group_research_probe.asc)
 target_link_libraries(topk_c15_group_${kind} PRIVATE tiling_api register platform unified_dlog dl m graph_base)
 target_compile_options(topk_c15_group_${kind} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
endforeach()
target_compile_definitions(topk_c15_group_raw PRIVATE C15_RAW=1)
target_compile_definitions(topk_c15_group_baseline PRIVATE C15_BASELINE=1)
target_compile_definitions(topk_c15_group_baseline_raw PRIVATE C15_BASELINE_RAW=1)
target_compile_definitions(topk_c15_group_copy PRIVATE C15_COPY=1)

add_executable(topk_c15_group_production_probe topk_c15_group_production_probe.asc)
target_link_libraries(topk_c15_group_production_probe PRIVATE tiling_api register platform unified_dlog dl m graph_base)
target_compile_options(topk_c15_group_production_probe PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

foreach(kind IN ITEMS full raw baseline)
 add_executable(topk_fp32_k8_block_${kind} topk_fp32_k8_block_research_probe.asc)
 target_link_libraries(topk_fp32_k8_block_${kind} PRIVATE tiling_api register platform unified_dlog dl m graph_base)
 target_compile_options(topk_fp32_k8_block_${kind} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
endforeach()
target_compile_definitions(topk_fp32_k8_block_raw PRIVATE FP32_K8_RAW=1)
target_compile_definitions(topk_fp32_k8_block_baseline PRIVATE FP32_K8_BASELINE=1)
add_executable(topk_fp32_k8_block_production_probe topk_fp32_k8_block_production_probe.asc)
target_link_libraries(topk_fp32_k8_block_production_probe PRIVATE tiling_api register platform unified_dlog dl m graph_base)
target_compile_options(topk_fp32_k8_block_production_probe PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

foreach(kind IN ITEMS full raw block block_raw baseline)
 add_executable(topk_c13_selection_${kind} topk_c13_selection_research_probe.asc)
 target_link_libraries(topk_c13_selection_${kind} PRIVATE tiling_api register platform unified_dlog dl m graph_base)
 target_compile_options(topk_c13_selection_${kind} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
endforeach()
target_compile_definitions(topk_c13_selection_raw PRIVATE C13_RAW=1)
target_compile_definitions(topk_c13_selection_block PRIVATE C13_BLOCK8=1)
target_compile_definitions(topk_c13_selection_block_raw PRIVATE C13_BLOCK8=1 C13_RAW=1)
target_compile_definitions(topk_c13_selection_baseline PRIVATE C13_BASELINE=1)

add_executable(topk_c13_selection_production_probe topk_c13_selection_production_probe.asc)
target_link_libraries(topk_c13_selection_production_probe PRIVATE tiling_api register platform unified_dlog dl m graph_base)
target_compile_options(topk_c13_selection_production_probe PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

foreach(kind IN ITEMS full raw baseline baseline_raw copy)
 add_executable(topk_c14_ten_${kind} topk_c14_exact_ten_research_probe.asc)
 target_link_libraries(topk_c14_ten_${kind} PRIVATE tiling_api register platform unified_dlog dl m graph_base)
 target_compile_options(topk_c14_ten_${kind} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
endforeach()
target_compile_definitions(topk_c14_ten_raw PRIVATE VALUES_ONLY_RAW=1)
target_compile_definitions(topk_c14_ten_baseline PRIVATE C14_BASELINE=1)
target_compile_definitions(topk_c14_ten_baseline_raw PRIVATE C14_BASELINE=1 VALUES_ONLY_RAW=1)
target_compile_definitions(topk_c14_ten_copy PRIVATE C14_BASELINE=1 C14_COPY=1)

foreach(kind IN ITEMS half half_raw)
 add_executable(topk_c14_ten_${kind} topk_c14_exact_ten_research_probe.asc)
 target_link_libraries(topk_c14_ten_${kind} PRIVATE tiling_api register platform unified_dlog dl m graph_base)
 target_compile_options(topk_c14_ten_${kind} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
 target_compile_definitions(topk_c14_ten_${kind} PRIVATE C14_HALF_SCRATCH=1)
endforeach()
target_compile_definitions(topk_c14_ten_half_raw PRIVATE VALUES_ONLY_RAW=1)

add_executable(topk_multi_half_production_probe topk_multi_half_production_probe.asc)
target_link_libraries(topk_multi_half_production_probe PRIVATE tiling_api register platform unified_dlog dl m graph_base)
target_compile_options(topk_multi_half_production_probe PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

foreach(kind IN ITEMS full raw baseline)
 add_executable(topk_c15_batch_filter_${kind} topk_c15_batch_filter_probe.asc)
 target_link_libraries(topk_c15_batch_filter_${kind} PRIVATE tiling_api register platform unified_dlog dl m graph_base)
 target_compile_options(topk_c15_batch_filter_${kind} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
endforeach()
target_compile_definitions(topk_c15_batch_filter_raw PRIVATE C15_RAW=1)
target_compile_definitions(topk_c15_batch_filter_baseline PRIVATE C15_BASELINE=1)
