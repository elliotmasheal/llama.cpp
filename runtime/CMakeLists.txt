# qwen36 MoE runtime (design 5.1). It links neither ggml nor llama: the runtime
# parses the GGUF itself and drives Vulkan directly. Enabled from the root
# CMakeLists.txt with -DLLAMA_BUILD_QRT=ON (the default here).

add_library(qrt_core STATIC
    qrt_config.cpp
    qrt_telemetry.cpp
    gguf_index.cpp
    weights.cpp
    prefill.cpp
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

# --prefill-check dispatches a compute shader, so it needs SPIR-V. The shader is the
# M1-4a read probe: it stands in for the int8 GEMM until M2-5 step 2 wires that in.
find_package(Vulkan QUIET COMPONENTS glslc)
if(Vulkan_GLSLC_EXECUTABLE)
    set(_qrt_read_comp "${CMAKE_CURRENT_SOURCE_DIR}/proto/qrt-read.comp")
    set(_qrt_read_spv "${CMAKE_CURRENT_BINARY_DIR}/qrt-read.spv")
    add_custom_command(
        OUTPUT  ${_qrt_read_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O ${_qrt_read_comp} -o ${_qrt_read_spv}
        DEPENDS ${_qrt_read_comp}
        COMMENT "Compiling proto/qrt-read.comp for --prefill-check"
        VERBATIM)
    add_custom_target(qrt_shaders DEPENDS ${_qrt_read_spv})
endif()

add_executable(qrt qrt_main.cpp)
target_link_libraries(qrt PRIVATE qrt_core)
set_target_properties(qrt PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# The shader lands next to qrt.exe, which is where --prefill-check looks by default.
if(TARGET qrt_shaders)
    add_dependencies(qrt qrt_shaders)
    add_custom_command(TARGET qrt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_qrt_read_spv} $<TARGET_FILE_DIR:qrt>
        COMMENT "Copying qrt-read.spv next to qrt")
endif()

if (BUILD_TESTING)
    add_test(NAME qrt_selftest COMMAND qrt --selftest)
endif()
