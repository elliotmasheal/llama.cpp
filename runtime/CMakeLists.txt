# qwen36 MoE runtime (design 5.1). It links neither ggml nor llama: the runtime
# parses the GGUF itself and drives Vulkan directly. Enabled from the root
# CMakeLists.txt with -DLLAMA_BUILD_QRT=ON (the default here).

add_library(qrt_core STATIC
    qrt_config.cpp
    qrt_telemetry.cpp
    gguf_index.cpp
    weights.cpp
    prefill.cpp
    kernels/q4k_gemm.cpp
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
# M1-4a read probe (the stand-in compute leg); kernels/q4k_gemm.comp and
# kernels/swiglu_q8.comp are the real int8 GEMM leg. Dot-product variant needs
# VK_KHR_shader_integer_dot_product, which vk_dev enables when the device has it.
find_package(Vulkan QUIET COMPONENTS glslc)
if(Vulkan_GLSLC_EXECUTABLE)
    set(_qrt_read_comp "${CMAKE_CURRENT_SOURCE_DIR}/proto/qrt-read.comp")
    set(_qrt_read_spv "${CMAKE_CURRENT_BINARY_DIR}/qrt-read.spv")
    set(_qrt_kernels "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
    set(_qrt_q4k_spv "${CMAKE_CURRENT_BINARY_DIR}/q4k_gemm.spv")
    set(_qrt_swiglu_spv "${CMAKE_CURRENT_BINARY_DIR}/swiglu_q8.spv")
    add_custom_command(
        OUTPUT  ${_qrt_read_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O ${_qrt_read_comp} -o ${_qrt_read_spv}
        DEPENDS ${_qrt_read_comp}
        COMMENT "Compiling proto/qrt-read.comp for --prefill-check"
        VERBATIM)
    add_custom_command(
        OUTPUT  ${_qrt_q4k_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.1 -DQ4K_DOT4
                -I${_qrt_kernels} ${_qrt_kernels}/q4k_gemm.comp -o ${_qrt_q4k_spv}
        DEPENDS ${_qrt_kernels}/q4k_gemm.comp ${_qrt_kernels}/q4k_tiles.h
        COMMENT "Compiling kernels/q4k_gemm.comp (dot product variant)"
        VERBATIM)
    add_custom_command(
        OUTPUT  ${_qrt_swiglu_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O ${_qrt_kernels}/swiglu_q8.comp -o ${_qrt_swiglu_spv}
        DEPENDS ${_qrt_kernels}/swiglu_q8.comp
        COMMENT "Compiling kernels/swiglu_q8.comp"
        VERBATIM)
    add_custom_target(qrt_shaders DEPENDS ${_qrt_read_spv} ${_qrt_q4k_spv} ${_qrt_swiglu_spv})
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
                ${_qrt_read_spv} ${_qrt_q4k_spv} ${_qrt_swiglu_spv} $<TARGET_FILE_DIR:qrt>
        COMMENT "Copying the runtime shaders next to qrt")
endif()

if (BUILD_TESTING)
    add_test(NAME qrt_selftest COMMAND qrt --selftest)
endif()
