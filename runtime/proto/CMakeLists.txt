# M1-4a prototype: standalone Vulkan pass-cost probe. Links Vulkan only, no ggml/llama.
# Enabled from the root CMakeLists.txt with -DLLAMA_BUILD_QRT_PROTO=ON on a GGML_VULKAN build.

find_package(Vulkan REQUIRED COMPONENTS glslc)

set(_qrt_spv "${CMAKE_CURRENT_BINARY_DIR}/qrt-read.spv")

add_custom_command(
    OUTPUT  ${_qrt_spv}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O ${CMAKE_CURRENT_SOURCE_DIR}/qrt-read.comp -o ${_qrt_spv}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/qrt-read.comp
    COMMENT "Compiling qrt-read.comp"
    VERBATIM)

add_custom_target(qrt_proto_shader DEPENDS ${_qrt_spv})

add_executable(qrt_proto qrt-proto.cpp)
add_dependencies(qrt_proto qrt_proto_shader)

set_target_properties(qrt_proto PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(qrt_proto PRIVATE Vulkan::Vulkan)
