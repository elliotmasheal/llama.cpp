# Build llama.cpp locally

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](../include/llama.h).

The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server.

**To get the Code:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

The following sections describe how to build with different backends and options.

* [CPU Build](#cpu-build)
* [Vulkan](#vulkan)
* [Arm® KleidiAI™](#arm-kleidiai)
* [Notes about GPU-accelerated backends](#notes-about-gpu-accelerated-backends)

## CPU Build

Build llama.cpp using `CMake`:

```bash
cmake -B build
cmake --build build --config Release
```

**Notes**:

- For faster compilation, add the `-j` argument to run multiple jobs in parallel, or use a generator that does this automatically such as Ninja. For example, `cmake --build build --config Release -j 8` will run 8 jobs in parallel.
- For faster repeated compilation, install [ccache](https://ccache.dev/)
- For debug builds, there are two cases:

    1. Single-config generators (e.g. default = `Unix Makefiles`; note that they just ignore the `--config` flag):

       ```bash
       cmake -B build -DCMAKE_BUILD_TYPE=Debug
       cmake --build build
       ```

    2. Multi-config generators (`-G` param set to Visual Studio, XCode...):

       ```bash
       cmake -B build -G "Xcode"
       cmake --build build --config Debug
       ```

    For more details and a list of supported generators, see the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).
- For static builds, add `-DBUILD_SHARED_LIBS=OFF`:
  ```
  cmake -B build -DBUILD_SHARED_LIBS=OFF
  cmake --build build --config Release
  ```

- Building for Windows with MSVC: install Visual Studio 2022 (workload "Desktop development with C++",
  components "C++ CMake tools for Windows" and "Git for Windows"), then use a Developer Command Prompt or
  PowerShell for VS2022:
    ```bash
    cmake --preset x64-windows-msvc-release
    cmake --build build-x64-windows-msvc-release
    ```
- Building for Windows with MinGW-w64 and Ninja (the configuration used in this fork):
    ```bash
    cmake -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_C_COMPILER=D:/Scoop/apps/mingw/current/bin/gcc.exe \
      -DCMAKE_CXX_COMPILER=D:/Scoop/apps/mingw/current/bin/g++.exe
    cmake --build build-vulkan -j 8
    ```
- If you want HTTPS/TLS features, you may install OpenSSL development libraries. If not installed, the project will build and run without SSL support.
  - **Debian / Ubuntu:** `sudo apt-get install libssl-dev`
  - **Fedora / RHEL / Rocky / Alma:** `sudo dnf install openssl-devel`
  - **Arch / Manjaro:** `sudo pacman -S openssl`

## Vulkan

### For Windows Users:
**w64devkit**

Download and extract [`w64devkit`](https://github.com/skeeto/w64devkit/releases).

Download and install the [`Vulkan SDK`](https://vulkan.lunarg.com/sdk/home#windows) with the default settings.

Launch `w64devkit.exe` and run the following commands to copy Vulkan dependencies:
```sh
SDK_VERSION=1.3.283.0
cp /VulkanSDK/$SDK_VERSION/Bin/glslc.exe $W64DEVKIT_HOME/bin/
cp /VulkanSDK/$SDK_VERSION/Lib/vulkan-1.lib $W64DEVKIT_HOME/x86_64-w64-mingw32/lib/
cp -r /VulkanSDK/$SDK_VERSION/Include/* $W64DEVKIT_HOME/x86_64-w64-mingw32/include/
cat > $W64DEVKIT_HOME/x86_64-w64-mingw32/lib/pkgconfig/vulkan.pc <<EOF
Name: Vulkan-Loader
Description: Vulkan Loader
Version: $SDK_VERSION
Libs: -lvulkan-1
EOF

```

Switch into the `llama.cpp` directory and build using CMake.
```sh
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

**Git Bash MINGW64**

Download and install [`Git-SCM`](https://git-scm.com/downloads/win) with the default settings

Download and install [`Visual Studio Community Edition`](https://visualstudio.microsoft.com/) and make sure you select `C++`

Download and install [`CMake`](https://cmake.org/download/) with the default settings

Download and install the [`Vulkan SDK`](https://vulkan.lunarg.com/sdk/home#windows) with the default settings.

Go into your `llama.cpp` directory and right click, select `Open Git Bash Here` and then run the following commands

```
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

Now you can load the model in conversation mode using `Vulkan`

```sh
build/bin/Release/llama-cli -m "[PATH TO MODEL]" -ngl 100 -c 16384 -t 10 -n -2 -cnv
```

**MSYS2**

Install [MSYS2](https://www.msys2.org/) and then run the following commands in a UCRT terminal to install dependencies.
```sh
pacman -S git \
    mingw-w64-ucrt-x86_64-gcc \
    mingw-w64-ucrt-x86_64-cmake \
    mingw-w64-ucrt-x86_64-vulkan-devel \
    mingw-w64-ucrt-x86_64-shaderc \
    mingw-w64-ucrt-x86_64-spirv-headers
```

Switch into the `llama.cpp` directory and build using CMake.
```sh
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

### For Docker users:

You don't need to install the Vulkan SDK. It will be installed inside the container.

```sh
# Build the image
docker build -t llama-cpp-vulkan --target light -f .devops/vulkan.Dockerfile .

# Then, use it:
docker run -it --rm -v "$(pwd):/app:Z" --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card1:/dev/dri/card1 llama-cpp-vulkan -m "/app/models/YOUR_MODEL_FILE" -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33
```

### For Linux users:

#### Using the LunarG Vulkan SDK

First, follow the official LunarG instructions for the installation and setup of the Vulkan SDK in the [Getting Started with the Linux Tarball Vulkan SDK](https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html) guide.

> [!IMPORTANT]
> After completing the first step, ensure that you have used the `source` command on the `setup_env.sh` file inside of the Vulkan SDK in your current terminal session. Otherwise, the build won't work. Additionally, if you close out of your terminal, you must perform this step again if you intend to perform a build. However, there are ways to make this persistent. Refer to the Vulkan SDK guide linked in the first step for more information about any of this.

#### Using system packages

On Debian / Ubuntu, you can install the required dependencies using:
```sh
sudo apt-get install libvulkan-dev glslc spirv-headers
```

SPIRV-Headers (`spirv/unified1/spirv.hpp`) are required for the Vulkan backend and are **not** always pulled in by the Vulkan loader dev package alone. Other distros use names such as `spirv-headers` (Ubuntu / Debian / Arch), or `spirv-headers-devel` (Fedora / openSUSE). On Windows, the LunarG Vulkan SDK’s `Include` directory already contains these headers.

#### Common steps

Second, after verifying that you have followed all of the SDK installation/setup steps, use this command to make sure before proceeding:
```bash
vulkaninfo
```

Then, assuming you have `cd` into your llama.cpp folder and there are no errors with running `vulkaninfo`, you can proceed to build llama.cpp using the CMake commands below:
```bash
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release
```

Finally, after finishing your build, you should be able to do something like this:
```bash
# Test the output binary
# "-ngl 99" should offload all of the layers to GPU for most (if not all) models.
./build/bin/llama-cli -m "PATH_TO_MODEL" -p "Hi you how are you" -ngl 99

# You should see in the output, ggml_vulkan detected your GPU. For example:
# ggml_vulkan: Using Intel(R) Graphics (ADL GT2) | uma: 1 | fp16: 1 | warp size: 32
```

### For Mac users:

Generally, follow LunarG's [Getting Started with the MacOS Vulkan SDK](https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html) guide for installation and setup of the Vulkan SDK. There are two options of Vulkan drivers on macOS, both of which implement translation layers to map Vulkan to Metal. They can be hot-swapped by setting the `VK_ICD_FILENAMES` environment variable to point to the respective ICD JSON file.

Check the box for "KosmicKrisp" during the LunarG Vulkan SDK installation.

Set environment variable for the LunarG Vulkan SDK after installation (and optionally add to your shell profile for persistence):
```bash
source /path/to/vulkan-sdk/setup-env.sh
```

#### Using MoltenVK

MoltenVK is the default Vulkan driver installed with the LunarG Vulkan SDK on macOS, so you can use the above environment variable settings as is.

#### Using KosmicKrisp

Override the environment variable for KosmicKrisp:
```bash
export VK_ICD_FILENAMES=$VULKAN_SDK/share/vulkan/icd.d/libkosmickrisp_icd.json
export VK_DRIVER_FILES=$VULKAN_SDK/share/vulkan/icd.d/libkosmickrisp_icd.json
```

#### Build

This is the only step different from [above](#common-steps) instructions.
```bash
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release
```

## Arm® KleidiAI™
KleidiAI provides optimized Arm CPU microkernels used by the ggml CPU backend. Enabling it at build time makes those kernels available; it does not force every operation to use KleidiAI. At runtime, llama.cpp selects the best compatible CPU kernel from the detected CPU features, tensor type, operation shape, and active backend priority.

Supported targets:

| Platform | Supported ABI / architecture | Notes |
| --- | --- | --- |
| Linux | AArch64 / arm64 | Runtime CPU feature detection is automatic. |
| Android | `arm64-v8a` | Use the Android NDK command below for a portable build. |
| Apple | arm64 | Runtime CPU feature detection is automatic. Non-streaming SVE vector length is treated as unavailable. |
| Windows | arm64 | Runtime CPU feature detection is automatic. SMCU count is treated as unknown until a detection path is verified. |

`GGML_CPU_KLEIDIAI=ON` is valid only for AArch64/arm64 builds. Do not enable it for x86, 32-bit Arm, or Android ABIs other than `arm64-v8a`.

### Native AArch64/arm64 build

From the llama.cpp source directory:

```bash
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release
```

### Android arm64-v8a NDK build

Set `ANDROID_NDK` to the Android NDK root, then run the following from the llama.cpp source directory. This command configures a portable Android `arm64-v8a` build with KleidiAI enabled and avoids Android dependencies that are not part of the NDK stable native API set.

```bash
cmake -S . -B build-android \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DLLAMA_OPENSSL=OFF
cmake --build build-android --config Release --parallel
cmake --install build-android --prefix {install-dir} --config Release
```

Important Android options:

- `GGML_CPU_KLEIDIAI=ON` enables KleidiAI for Android `arm64-v8a`.
- `GGML_NATIVE=OFF` is required for cross-compilation because the build host CPU is not the Android target CPU.
- `GGML_OPENMP=OFF` avoids adding an OpenMP runtime dependency to this NDK command-line build.
- `GGML_LLAMAFILE=OFF` avoids the llamafile backend, which is not supported on Android.
- `LLAMA_OPENSSL=OFF` avoids depending on OpenSSL, which is not part of the Android NDK stable native API set.

The Android Studio project under `examples/llama.android` enables KleidiAI automatically for `arm64-v8a`. For Android command-line CMake builds on `arm64-v8a`, pass `-DGGML_CPU_KLEIDIAI=ON` explicitly.

Global -march flags such as `-march=armv8.7a` flag are not required for a portable Android `arm64-v8a` build. Global `-march` flags raise the baseline instruction set for generic code. No manual architecture-specific source selection is required; llama.cpp selects compatible KleidiAI kernels at runtime. The KleidiAI libraries internal CMake handles the -march flags for each particular kernel.

### Verifying the build

Run an installed or in-tree binary:

```bash
./build/bin/llama-cli -m PATH_TO_MODEL -p "What is a car?"
```

If KleidiAI is enabled, the output contains a line similar to:

```
load_tensors: CPU_KLEIDIAI model buffer size =  3474.00 MiB
```

This confirms that the model has tensors allocated through the KleidiAI CPU buffer. It does not prove that every operation, or any specific SME-family operation, used a KleidiAI microkernel. Runtime CPU features, tensor type, operation shape, and backend priority still control dispatch.

Depending on the build target, another backend may have higher priority than the CPU backend. To force CPU execution for a run, disable higher priority backends at build time, for example `-DGGML_VULKAN=OFF`, or use a runtime device option such as `--device none` where supported.

### Runtime dispatch

KleidiAI microkernels use Arm CPU features such as dotprod, i8mm, SVE, and SME/SME2. Build-time configuration makes the kernels available. Runtime dispatch selects a compatible kernel for the detected CPU and operation. Older or lower-feature CPUs fall back automatically to compatible kernels.

KleidiAI accelerates selected `GGML_OP_MUL_MAT` paths for F32 and common quantized formats. Exact coverage depends on the bundled KleidiAI version and the llama.cpp runtime selector, so unsupported tensor types, unsupported operation shapes, or higher priority backends may bypass KleidiAI even when the CPU supports the required Arm feature. This is also why a model may not use SME-family kernels on SME-capable hardware.

The current llama.cpp KleidiAI SVE selector only enables SVE kernels when the runtime SVE vector length is known to be QK8_0 bytes, currently 32 bytes. Linux and Android query this at runtime. Apple reports SVE capability separately from userspace non-streaming SVE availability, so llama.cpp treats the SVE vector length as unknown there. Windows exposes SVE feature presence but not the runtime SVE vector length used by this selector, so that value is also treated as unknown. Windows arm64 also treats SMCU count as unknown until a detection mechanism is verified.

The set of available SME-family kernels depends on the bundled KleidiAI version and the detected CPU capabilities. Production configuration does not require any KleidiAI runtime environment variables.

### Diagnostics and debug overrides

KleidiAI runtime environment variables are diagnostics/debug overrides, not production configuration. Leave them unset for normal use.

`GGML_KLEIDIAI_SME` controls SME-family kernel selection and overrides the maximum number of threads assigned to selected quantized SME-family kernels:

- Not set: use automatic runtime detection.
- `0`: disable SME-family kernels.
- `<n> > 0`: enable compatible SME-family kernels and allow up to `<n>` threads for quantized SME-family kernels.

On Windows arm64, use `GGML_KLEIDIAI_SME=<n>` as the temporary diagnostics/debug override for SME thread-cap calibration until automatic SMCU count detection is verified.

If the CPU does not support the required SME-family capability for a bundled kernel, that kernel is disabled regardless of the environment variable.

---
## Notes about GPU-accelerated backends

The GPU may still be used to accelerate some parts of the computation even when using the `-ngl 0` option. You can fully disable GPU acceleration by using `--device none`.

In most cases, it is possible to build and use multiple backends at the same time. For example, you can build llama.cpp with both CPU and Vulkan support by using the `-DGGML_CPU=ON -DGGML_VULKAN=ON` options with CMake. At runtime, you can specify which backend devices to use with the `--device` option. To see a list of available devices, use the `--list-devices` option.

Backends can be built as dynamic libraries that can be loaded dynamically at runtime. This allows you to use the same llama.cpp binary on different machines with different GPUs. To enable this feature, use the `GGML_BACKEND_DL` option when building.
