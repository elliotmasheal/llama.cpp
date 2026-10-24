#!/usr/bin/env bash
# Builds the R1c probe: two shader variants (int8 dot, portable) plus the host binary.
# MinGW + Vulkan SDK only; nothing here touches the project CMake.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
sdk=${VULKAN_SDK:-D:/Scoop/apps/vulkan/current}
cxx=${CXX:-D:/Scoop/apps/mingw/current/bin/g++.exe}
glslc="$sdk/Bin/glslc.exe"

out="$here/out"
mkdir -p "$out"

# vulkan1.1 target keeps the module at SPIR-V 1.3 + SPV_KHR_integer_dot_product, which any
# 1.1+ device with VK_KHR_shader_integer_dot_product can consume (not only 1.3 devices).
"$glslc" -O --target-env=vulkan1.1 -DQ4K_DOT4 -I"$here" \
    "$here/q4k-gemm.comp" -o "$out/q4k-gemm-dot.spv"
"$glslc" -O --target-env=vulkan1.0 -I"$here" \
    "$here/q4k-gemm.comp" -o "$out/q4k-gemm-fallback.spv"

"$cxx" -O2 -std=c++17 -Wall \
    "$here/q4k-gemm.cpp" -I"$here" -I"$sdk/Include" -L"$sdk/Lib" -lvulkan-1 \
    -o "$out/q4k-gemm.exe"

echo "built $out/q4k-gemm.exe and the two .spv files"
