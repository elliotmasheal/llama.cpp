# qwen36 MoE runtime (design 5.1). It links neither ggml nor llama: the runtime
# parses the GGUF itself and drives Vulkan directly. Enabled from the root
# CMakeLists.txt with -DLLAMA_BUILD_QRT=ON (the default here).

add_library(qrt_core STATIC
    qrt_config.cpp
    qrt_telemetry.cpp
    gguf_index.cpp
    weights.cpp
    prefill.cpp
    kernels/gemm.cpp
    kernels/moe_expert.cpp
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
    # Tokens per lane: the expert kernels are best at 8 (M2-4 sweep), but the dense
    # projections run at M=10240 where a wider tile cuts the weight re-reads (RESULTS 46).
    set(_qrt_tokens_q4k 8)
    set(_qrt_tokens_q5k 8)
    set(_qrt_tokens_q6k 8)
    set(_qrt_tokens_q8k 8)
    set(_qrt_gemm_spv "")
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
        add_custom_command(
            OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/gemm_${_q}.spv
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1 -DQ4K_DOT4
                    -DQ4K_TOKENS=${_qrt_tokens_${_q}}
                    -I${_qrt_kernels} ${_qrt_kernels}/gemm_${_q}.comp
                    -o ${CMAKE_CURRENT_BINARY_DIR}/gemm_${_q}.spv
            DEPENDS ${_qrt_kernels}/gemm_${_q}.comp ${_qrt_kernels}/tiles.h
                    ${_qrt_kernels}/common.glsl
            COMMENT "Compiling kernels/gemm_${_q}.comp (dot product variant)"
            VERBATIM)
    endforeach()
    add_custom_command(
        OUTPUT  ${_qrt_swiglu_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O ${_qrt_kernels}/swiglu_q8.comp -o ${_qrt_swiglu_spv}
        DEPENDS ${_qrt_kernels}/swiglu_q8.comp
        COMMENT "Compiling kernels/swiglu_q8.comp"
        VERBATIM)
    add_custom_target(qrt_shaders DEPENDS ${_qrt_read_spv} ${_qrt_gemm_spv} ${_qrt_swiglu_spv})
endif()

add_executable(qrt qrt_main.cpp)
target_link_libraries(qrt PRIVATE qrt_core)
target_include_directories(qrt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(qrt PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# The shaders land next to qrt.exe, which is where the checks look by default.
if(TARGET qrt_shaders)
    add_dependencies(qrt qrt_shaders)
    add_custom_command(TARGET qrt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_qrt_read_spv} ${_qrt_gemm_spv} ${_qrt_swiglu_spv} $<TARGET_FILE_DIR:qrt>
        COMMENT "Copying the runtime shaders next to qrt")
endif()

if (BUILD_TESTING)
    add_test(NAME qrt_selftest COMMAND qrt --selftest)
endif()
