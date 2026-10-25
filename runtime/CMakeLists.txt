# qwen36 MoE runtime (design 5.1). It links neither ggml nor llama: the runtime
# parses the GGUF itself and drives Vulkan directly. Enabled from the root
# CMakeLists.txt with -DLLAMA_BUILD_QRT=ON (the default here).

add_library(qrt_core STATIC
    qrt_config.cpp
    qrt_telemetry.cpp
    gguf_index.cpp
    weights.cpp
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

add_executable(qrt qrt_main.cpp)
target_link_libraries(qrt PRIVATE qrt_core)
set_target_properties(qrt PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

if (BUILD_TESTING)
    add_test(NAME qrt_selftest COMMAND qrt --selftest)
endif()
