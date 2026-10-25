# qwen36 MoE runtime (design 5.1). It links neither ggml nor llama: the runtime
# parses the GGUF itself and drives Vulkan directly. Enabled from the root
# CMakeLists.txt with -DLLAMA_BUILD_QRT=ON (the default here).

add_library(qrt_core STATIC
    qrt_config.cpp
    qrt_telemetry.cpp
    gguf_index.cpp
    weights.cpp
    prefill.cpp
    attn_deltanet.cpp
    attn_deltanet_check.cpp
    attn_full.cpp
    attn_full_check.cpp
    kernels/gemm.cpp
    kernels/norm.cpp
    kernels/moe_expert.cpp
    layer.cpp
    ffn.cpp
    vk_dev.cpp)

target_include_directories(qrt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(qrt_core PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# vk_dev.cpp compiles to a stub when the SDK is absent, so qrt still builds and
# --vk-info reports why it cannot open a device.
find_package(Vulkan QUIET)
if (Vulkan_FOUND)
    target_compile_definitions(qrt_core PUBLIC QRT_HAVE_VULKAN)
    target_link_libraries(qrt_core PUBLIC Vulkan::Vulkan)
else()
    message(WARNING "Vulkan SDK not found: qrt builds without --vk-info")
endif()

# --prefill-check dispatches compute shaders, so it needs SPIR-V. qrt-read.comp is the
# M1-4a read probe; kernels/gemm_{q4k,q5k,q6k}.comp and kernels/swiglu_q8.comp are the
# real int8 GEMM leg. The dot-product variant needs VK_KHR_shader_integer_dot_product,
# which vk_dev enables when the device has it.
find_package(Vulkan QUIET COMPONENTS glslc)
if(Vulkan_GLSLC_EXECUTABLE)
    set(_qrt_read_comp "${CMAKE_CURRENT_SOURCE_DIR}/proto/qrt-read.comp")
    set(_qrt_read_spv "${CMAKE_CURRENT_BINARY_DIR}/qrt-read.spv")
    set(_qrt_kernels "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
    set(_qrt_swiglu_spv "${CMAKE_CURRENT_BINARY_DIR}/swiglu_q8.spv")
    # Tile shape per kernel. The expert kernels keep the M2-4 choice (2 rows x 8 tokens);
    # the dense Q8_0 kernel is swept from the command line because its shapes differ:
    # -DQRt_Q8K_ROWS / -DQRt_Q8K_TOKENS / -DQRt_Q8K_WG (RESULTS 48).
    set(_qrt_tokens_q4k 8)
    set(_qrt_tokens_q5k 8)
    set(_qrt_tokens_q6k 8)
    set(_qrt_rows_q4k 2)
    set(_qrt_rows_q5k 2)
    set(_qrt_rows_q6k 2)
    if(NOT DEFINED QRt_Q8K_ROWS)
        set(QRt_Q8K_ROWS 2)
    endif()
    if(NOT DEFINED QRt_Q8K_TOKENS)
        set(QRt_Q8K_TOKENS 8)
    endif()
    if(NOT DEFINED QRt_Q8K_WG)
        set(QRt_Q8K_WG 128)
    endif()
    set(_qrt_rows_q8k ${QRt_Q8K_ROWS})
    set(_qrt_tokens_q8k ${QRt_Q8K_TOKENS})
    set(_qrt_wg_q8k ${QRt_Q8K_WG})
    set(_qrt_gemm_spv "")
    foreach(_q IN ITEMS q4k q5k q6k q8k)
        set(_qrt_wg_${_q} 128)
    endforeach()
    set(_qrt_wg_q8k ${QRt_Q8K_WG})
    foreach(_q IN ITEMS q4k q5k q6k q8k)
        list(APPEND _qrt_gemm_spv "${CMAKE_CURRENT_BINARY_DIR}/gemm_${_q}.spv")
    endforeach()
    add_custom_command(
        OUTPUT  ${_qrt_read_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O ${_qrt_read_comp} -o ${_qrt_read_spv}
        DEPENDS ${_qrt_read_comp}
        COMMENT "Compiling proto/qrt-read.comp for --prefill-check"
        VERBATIM)
    foreach(_q IN ITEMS q4k q5k q6k q8k)
        set(_qrt_wg_${_q} 128)
    endforeach()
    set(_qrt_wg_q8k ${QRt_Q8K_WG})
    foreach(_q IN ITEMS q4k q5k q6k q8k)
        add_custom_command(
            OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/gemm_${_q}.spv
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1 -DQ4K_DOT4
                    -DQ4K_TOKENS=${_qrt_tokens_${_q}} -DQ4K_ROWS=${_qrt_rows_${_q}}
                    -DQ4K_WG_INVOCATIONS=${_qrt_wg_${_q}}
                    -I${_qrt_kernels} ${_qrt_kernels}/gemm_${_q}.comp
                    -o ${CMAKE_CURRENT_BINARY_DIR}/gemm_${_q}.spv
            DEPENDS ${_qrt_kernels}/gemm_${_q}.comp ${_qrt_kernels}/tiles.h
                    ${_qrt_kernels}/common.glsl
            COMMENT "Compiling kernels/gemm_${_q}.comp (dot product variant)"
            VERBATIM)
    endforeach()
    add_custom_command(
        OUTPUT  ${_qrt_swiglu_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1 -I${_qrt_kernels}
                ${_qrt_kernels}/swiglu_q8.comp -o ${_qrt_swiglu_spv}
        DEPENDS ${_qrt_kernels}/swiglu_q8.comp ${_qrt_kernels}/common.glsl
        COMMENT "Compiling kernels/swiglu_q8.comp"
        VERBATIM)
    # RMS norm + Q8_1 pack (M2-6): the activation leg the dense GEMM reads.
    set(_qrt_norm_spv "${CMAKE_CURRENT_BINARY_DIR}/qrt_norm.spv")
    add_custom_command(
        OUTPUT  ${_qrt_norm_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                ${_qrt_kernels}/qrt_norm.comp -o ${_qrt_norm_spv}
        DEPENDS ${_qrt_kernels}/qrt_norm.comp
        COMMENT "Compiling kernels/qrt_norm.comp"
        VERBATIM)
    # The layer chain (M2-6) needs its residual add as its own kernel.
    set(_qrt_layer_add_spv "${CMAKE_CURRENT_BINARY_DIR}/layer_add.spv")
    add_custom_command(
        OUTPUT  ${_qrt_layer_add_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                ${_qrt_kernels}/layer_add.comp -o ${_qrt_layer_add_spv}
        DEPENDS ${_qrt_kernels}/layer_add.comp
        COMMENT "Compiling kernels/layer_add.comp"
        VERBATIM)
    # The FFN half (M2-6): router, top-k, the routed gather, the weighted accumulate and
    # the final residual add.
    set(_qrt_ffn_spv "")
    foreach(_k IN ITEMS router topk gather accum out)
        set(_spv "${CMAKE_CURRENT_BINARY_DIR}/ffn_${_k}.spv")
        add_custom_command(
            OUTPUT  ${_spv}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                    ${_qrt_kernels}/ffn_${_k}.comp -o ${_spv}
            DEPENDS ${_qrt_kernels}/ffn_${_k}.comp
            COMMENT "Compiling kernels/ffn_${_k}.comp"
            VERBATIM)
        list(APPEND _qrt_ffn_spv ${_spv})
    endforeach()
    # The DeltaNet recurrence (M2-6c) keeps the 128x128 state in registers and
    # reduces within 32-lane subgroups, so it needs SPIR-V 1.3 (Vulkan 1.1).
    set(_qrt_deltanet_spv "${CMAKE_CURRENT_BINARY_DIR}/deltanet_recur.spv")
    add_custom_command(
        OUTPUT  ${_qrt_deltanet_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                ${_qrt_kernels}/deltanet_recur.comp -o ${_qrt_deltanet_spv}
        DEPENDS ${_qrt_kernels}/deltanet_recur.comp
        COMMENT "Compiling kernels/deltanet_recur.comp"
        VERBATIM)
    # Wide recurrence variant (M2-6c): a workgroup owns 32 columns of one head, a
    # thread owns 32 rows of one column, so the k/q vector is read once per
    # workgroup instead of once per column pair (RESULTS 63).
    set(_qrt_dn_cols_spv "${CMAKE_CURRENT_BINARY_DIR}/deltanet_recur_cols.spv")
    add_custom_command(
        OUTPUT  ${_qrt_dn_cols_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                ${_qrt_kernels}/deltanet_recur_cols.comp -o ${_qrt_dn_cols_spv}
        DEPENDS ${_qrt_kernels}/deltanet_recur_cols.comp
        COMMENT "Compiling kernels/deltanet_recur_cols.comp"
        VERBATIM)
    set(_qrt_dn_gate_spv "${CMAKE_CURRENT_BINARY_DIR}/deltanet_gate.spv")
    add_custom_command(
        OUTPUT  ${_qrt_dn_gate_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                ${_qrt_kernels}/deltanet_gate.comp -o ${_qrt_dn_gate_spv}
        DEPENDS ${_qrt_kernels}/deltanet_gate.comp
        COMMENT "Compiling kernels/deltanet_gate.comp"
        VERBATIM)
    set(_qrt_dn_gnorm_spv "${CMAKE_CURRENT_BINARY_DIR}/deltanet_gnorm.spv")
    add_custom_command(
        OUTPUT  ${_qrt_dn_gnorm_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                ${_qrt_kernels}/deltanet_gnorm.comp -o ${_qrt_dn_gnorm_spv}
        DEPENDS ${_qrt_kernels}/deltanet_gnorm.comp
        COMMENT "Compiling kernels/deltanet_gnorm.comp"
        VERBATIM)
    set(_qrt_dn_prep_spv "${CMAKE_CURRENT_BINARY_DIR}/deltanet_prep.spv")
    add_custom_command(
        OUTPUT  ${_qrt_dn_prep_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                ${_qrt_kernels}/deltanet_prep.comp -o ${_qrt_dn_prep_spv}
        DEPENDS ${_qrt_kernels}/deltanet_prep.comp
        COMMENT "Compiling kernels/deltanet_prep.comp"
        VERBATIM)
    # Full attention (M2-6d): per-head RMS norm + IMRoPE + KV append.
    set(_qrt_full_qk_spv "${CMAKE_CURRENT_BINARY_DIR}/attn_full_qk.spv")
    add_custom_command(
        OUTPUT  ${_qrt_full_qk_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                ${_qrt_kernels}/attn_full_qk.comp -o ${_qrt_full_qk_spv}
        DEPENDS ${_qrt_kernels}/attn_full_qk.comp
        COMMENT "Compiling kernels/attn_full_qk.comp"
        VERBATIM)
    set(_qrt_full_attn_spv "${CMAKE_CURRENT_BINARY_DIR}/attn_full_attn.spv")
    add_custom_command(
        OUTPUT  ${_qrt_full_attn_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1
                ${_qrt_kernels}/attn_full_attn.comp -o ${_qrt_full_attn_spv}
        DEPENDS ${_qrt_kernels}/attn_full_attn.comp
        COMMENT "Compiling kernels/attn_full_attn.comp"
        VERBATIM)
    add_custom_target(qrt_shaders DEPENDS ${_qrt_read_spv} ${_qrt_gemm_spv} ${_qrt_swiglu_spv}
                      ${_qrt_norm_spv} ${_qrt_layer_add_spv} ${_qrt_deltanet_spv}
                      ${_qrt_dn_cols_spv}
                      ${_qrt_dn_prep_spv} ${_qrt_dn_gate_spv}
                      ${_qrt_dn_gnorm_spv} ${_qrt_full_qk_spv} ${_qrt_full_attn_spv}
                      ${_qrt_ffn_spv})
endif()

add_executable(qrt qrt_main.cpp)
target_link_libraries(qrt PRIVATE qrt_core)
target_include_directories(qrt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(qrt PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# The shaders land next to qrt.exe, which is where the checks look by default. The copy
# must be its own target: a POST_BUILD step on qrt only fires when qrt relinks, so a
# shader-only change would leave a stale .spv next to the binary (that cost a debugging
# round on M2-6c).
if(TARGET qrt_shaders)
    add_dependencies(qrt qrt_shaders)
    add_custom_target(qrt_spv_copy ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_qrt_read_spv} ${_qrt_gemm_spv} ${_qrt_swiglu_spv} ${_qrt_norm_spv}
                ${_qrt_layer_add_spv} ${_qrt_deltanet_spv}
                ${_qrt_dn_cols_spv} ${_qrt_dn_prep_spv} ${_qrt_dn_gate_spv} ${_qrt_dn_gnorm_spv}
                ${_qrt_full_qk_spv} ${_qrt_full_attn_spv} ${_qrt_ffn_spv} $<TARGET_FILE_DIR:qrt>
        DEPENDS ${_qrt_read_spv} ${_qrt_gemm_spv} ${_qrt_swiglu_spv} ${_qrt_norm_spv}
                ${_qrt_layer_add_spv} ${_qrt_deltanet_spv}
                ${_qrt_dn_cols_spv} ${_qrt_dn_prep_spv} ${_qrt_dn_gate_spv} ${_qrt_dn_gnorm_spv}
                ${_qrt_full_qk_spv} ${_qrt_full_attn_spv} ${_qrt_ffn_spv}
        COMMENT "Copying the runtime shaders next to qrt")
    add_dependencies(qrt qrt_spv_copy)
endif()

if (BUILD_TESTING)
    add_test(NAME qrt_selftest COMMAND qrt --selftest)
endif()
