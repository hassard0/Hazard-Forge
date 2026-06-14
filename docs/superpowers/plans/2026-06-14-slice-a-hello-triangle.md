# Slice A: Hello Triangle + RHI Seam — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `hello_triangle.exe` — an SDL3 window rendering a colored triangle through a thin Vulkan RHI backend, surviving resize, closing cleanly, with zero Vulkan validation errors.

**Architecture:** A pure-virtual C++ RHI interface (`engine/rhi/`) with no Vulkan symbols, implemented by a Vulkan backend (`engine/rhi_vulkan/`) using vk-bootstrap (bring-up), VMA (allocation), and raw C Vulkan 1.3 dynamic rendering (commands/pipeline/sync). SDL3 provides the window/surface (`engine/hal/`). HLSL shaders compile to SPIR-V via DXC at build time.

**Tech Stack:** C++20, CMake + Ninja + Presets, Conan 2 (sdl, vk-bootstrap, vulkan-memory-allocator, vulkan-headers, vulkan-loader, volk-free), DXC, MSVC.

---

## Prerequisites (verify once, before Task 1)

- [ ] **Verify toolchain present**

Run each; all must succeed:
```bash
cmake --version          # >= 3.25 (presets v6)
ninja --version
conan --version          # >= 2.0
where dxc                 # DXC on PATH (ships with Windows SDK / Vulkan SDK)
where cl                  # MSVC compiler on PATH (run from a Dev shell or use ilammy/msvc env)
```
Expected: each prints a version / path. If `dxc` is missing, install the Vulkan SDK (LunarG) — it bundles `dxc` and the validation layers. If `conan` is missing: `pip install "conan>=2.0"`.

- [ ] **Create a default Conan profile if none exists**

Run:
```bash
conan profile detect --exist-ok
```
Expected: prints a detected profile (compiler=msvc, build_type, etc.).

---

## File Structure

| File | Responsibility |
|---|---|
| `conanfile.py` | Declares deps + generators (CMakeToolchain, CMakeDeps). |
| `CMakeLists.txt` | Top-level: C++20, finds deps, adds subdirs. |
| `CMakePresets.json` | `windows-msvc-debug` / `-release` configure+build presets wired to Conan toolchain. |
| `cmake/CompileShaders.cmake` | `hf_compile_shaders()` — DXC HLSL→SPIR-V build rule. |
| `engine/rhi/rhi.h` | THE SEAM — interfaces, descriptor structs, enums. No `vk*`. |
| `engine/rhi/rhi_factory.h/.cpp` | `hf::rhi::CreateDevice(Backend)` → backend instance. |
| `engine/hal/window.h/.cpp` | SDL3 window + event pump + Vulkan instance-extension query + surface creation. |
| `engine/rhi_vulkan/*` | Vulkan implementation of every RHI interface. |
| `shaders/triangle.vert.hlsl`, `triangle.frag.hlsl` | The triangle (hardcoded NDC). |
| `samples/hello_triangle/main.cpp` | Runnable target wiring it together. |
| `tests/rhi_smoke.cpp` | CTest headless device+swapchain create/destroy. |

---

## Task 0: Project skeleton, Conan, CMake, buildable empty app

**Files:**
- Create: `conanfile.py`, `CMakeLists.txt`, `CMakePresets.json`, `engine/CMakeLists.txt`, `samples/hello_triangle/CMakeLists.txt`, `samples/hello_triangle/main.cpp`

- [ ] **Step 1: Write `conanfile.py`**

```python
from conan import ConanFile
from conan.tools.cmake import cmake_layout


class HazardForge(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("sdl/3.2.4")
        self.requires("vk-bootstrap/1.3.295")
        self.requires("vulkan-memory-allocator/3.1.0")
        self.requires("vulkan-headers/1.3.290.0")
        self.requires("vulkan-loader/1.3.290.0")

    def layout(self):
        cmake_layout(self)
```

> If a listed version is not in your Conan remote, run `conan search <name>` and pick the nearest available; update the version string. Keep `vulkan-headers` and `vulkan-loader` on the same SDK version.

- [ ] **Step 2: Write top-level `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.25)
project(HazardForge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive-)
endif()

find_package(SDL3 REQUIRED)
find_package(vk-bootstrap REQUIRED)
find_package(VulkanMemoryAllocator REQUIRED)
find_package(VulkanHeaders REQUIRED)
find_package(VulkanLoader REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

enable_testing()

add_subdirectory(engine)
add_subdirectory(samples/hello_triangle)
add_subdirectory(tests)
```

- [ ] **Step 3: Write `CMakePresets.json`**

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 25, "patch": 0 },
  "configurePresets": [
    {
      "name": "conan-base",
      "hidden": true,
      "generator": "Ninja",
      "toolchainFile": "${sourceDir}/build/${presetName}/generators/conan_toolchain.cmake",
      "binaryDir": "${sourceDir}/build/${presetName}"
    },
    {
      "name": "windows-msvc-debug",
      "inherits": "conan-base",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug" }
    },
    {
      "name": "windows-msvc-release",
      "inherits": "conan-base",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Release" }
    }
  ],
  "buildPresets": [
    { "name": "windows-msvc-debug", "configurePreset": "windows-msvc-debug" },
    { "name": "windows-msvc-release", "configurePreset": "windows-msvc-release" }
  ],
  "testPresets": [
    {
      "name": "windows-msvc-debug",
      "configurePreset": "windows-msvc-debug",
      "output": { "outputOnFailure": true }
    }
  ]
}
```

> The `toolchainFile` path matches Conan's `cmake_layout` output folder for the matching preset name. We pass `-of=build/<preset>` to Conan in Step 6 so the paths line up.

- [ ] **Step 4: Write `engine/CMakeLists.txt` (stub — grows each task)**

```cmake
add_library(hf_engine STATIC
  # rhi + hal + vulkan sources added in later tasks
  rhi/rhi_factory.cpp
)
target_include_directories(hf_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hf_engine PUBLIC
  SDL3::SDL3
  vk-bootstrap::vk-bootstrap
  GPUOpen::VulkanMemoryAllocator
  Vulkan::Headers
  Vulkan::Loader
)
```

> `rhi/rhi_factory.cpp` is created in Task 9 but the engine lib must compile from Task 2 onward. Create a temporary stub now so the target builds: see Step 5.

- [ ] **Step 5: Write a temporary stub `engine/rhi/rhi_factory.cpp`**

```cpp
// Temporary stub — replaced in Task 9.
namespace hf::rhi { void _link_placeholder() {} }
```

- [ ] **Step 6: Write `samples/hello_triangle/CMakeLists.txt` + minimal `main.cpp`**

`samples/hello_triangle/CMakeLists.txt`:
```cmake
add_executable(hello_triangle main.cpp)
target_link_libraries(hello_triangle PRIVATE hf_engine)
```

`samples/hello_triangle/main.cpp`:
```cpp
#include <cstdio>

int main() {
    std::printf("Hazard Forge: hello_triangle skeleton\n");
    return 0;
}
```

- [ ] **Step 7: Write `tests/CMakeLists.txt` (stub)**

```cmake
# Smoke test added in Task 10.
```

- [ ] **Step 8: Install deps and configure**

Run:
```bash
cd /c/Users/ihass/dev/hazard-forge
conan install . -of=build/windows-msvc-debug -s build_type=Debug --build=missing
cmake --preset windows-msvc-debug
```
Expected: Conan installs/builds deps; CMake configures with "Generating done".

- [ ] **Step 9: Build and run the skeleton**

Run:
```bash
cmake --build --preset windows-msvc-debug
./build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe
```
Expected: prints `Hazard Forge: hello_triangle skeleton`.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "build: project skeleton — conan, cmake presets, empty hello_triangle"
```

---

## Task 1: HAL window (SDL3) + Vulkan surface support

**Files:**
- Create: `engine/hal/window.h`, `engine/hal/window.cpp`
- Modify: `engine/CMakeLists.txt`

- [ ] **Step 1: Write `engine/hal/window.h`**

```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct SDL_Window;
struct VkInstance_T;  using VkInstance = VkInstance_T*;
struct VkSurfaceKHR_T; using VkSurfaceKHR = VkSurfaceKHR_T*;

namespace hf::hal {

struct WindowConfig {
    std::string title = "Hazard Forge";
    int width = 1280;
    int height = 720;
};

// Owns an SDL3 window with the Vulkan flag set. Pumps OS events.
class Window {
public:
    explicit Window(const WindowConfig& cfg);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Process pending OS events. Returns false when the user requested quit.
    bool PumpEvents();

    // True for exactly one PumpEvents cycle after a size/resize event.
    bool ConsumeResized();

    // Vulkan instance extensions SDL requires for this window's surface.
    std::vector<const char*> RequiredVulkanInstanceExtensions() const;

    // Create a Vulkan surface for this window on the given instance.
    VkSurfaceKHR CreateVulkanSurface(VkInstance instance) const;

    int FramebufferWidth() const;
    int FramebufferHeight() const;
    SDL_Window* Raw() const { return window_; }

private:
    SDL_Window* window_ = nullptr;
    bool resized_ = false;
};

} // namespace hf::hal
```

- [ ] **Step 2: Write `engine/hal/window.cpp`**

```cpp
#include "hal/window.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <stdexcept>

namespace hf::hal {

Window::Window(const WindowConfig& cfg) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
    }
    window_ = SDL_CreateWindow(cfg.title.c_str(), cfg.width, cfg.height,
                               SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window_) {
        throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }
}

Window::~Window() {
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

bool Window::PumpEvents() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) return false;
        if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED ||
            e.type == SDL_EVENT_WINDOW_RESIZED) {
            resized_ = true;
        }
    }
    return true;
}

bool Window::ConsumeResized() {
    bool r = resized_;
    resized_ = false;
    return r;
}

std::vector<const char*> Window::RequiredVulkanInstanceExtensions() const {
    Uint32 count = 0;
    const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
    if (!names) {
        throw std::runtime_error(std::string("SDL_Vulkan_GetInstanceExtensions failed: ") +
                                 SDL_GetError());
    }
    return std::vector<const char*>(names, names + count);
}

VkSurfaceKHR Window::CreateVulkanSurface(VkInstance instance) const {
    VkSurfaceKHR surface = nullptr;
    if (!SDL_Vulkan_CreateSurface(window_, instance, nullptr, &surface)) {
        throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface failed: ") +
                                 SDL_GetError());
    }
    return surface;
}

int Window::FramebufferWidth() const {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    return w;
}

int Window::FramebufferHeight() const {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    return h;
}

} // namespace hf::hal
```

> `SDL_Vulkan_CreateSurface` takes `VkInstance`/`VkSurfaceKHR*` as opaque pointers; including `<SDL3/SDL_vulkan.h>` (which pulls Vulkan types via the loader) keeps this `.cpp` the only HAL file that touches Vulkan typedefs. The header uses forward-declared handle typedefs so RHI code including `window.h` needs no Vulkan headers.

- [ ] **Step 3: Add HAL sources to `engine/CMakeLists.txt`**

Change the `add_library(hf_engine STATIC ...)` list to include:
```cmake
add_library(hf_engine STATIC
  hal/window.cpp
  rhi/rhi_factory.cpp
)
```

- [ ] **Step 4: Build to verify it compiles**

Run:
```bash
cmake --build --preset windows-msvc-debug
```
Expected: builds with no errors. (No runtime test yet — exercised in Task 10.)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(hal): SDL3 window with Vulkan surface + event pump"
```

---

## Task 2: RHI interface seam (`engine/rhi/rhi.h`)

**Files:**
- Create: `engine/rhi/rhi.h`

- [ ] **Step 1: Write `engine/rhi/rhi.h`**

```cpp
#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

// Forward declarations of HAL + handle types — NO vulkan headers in this file.
struct VkInstance_T;  using VkInstance = VkInstance_T*;
struct VkSurfaceKHR_T; using VkSurfaceKHR = VkSurfaceKHR_T*;

namespace hf::hal { class Window; }

namespace hf::rhi {

enum class Backend { Vulkan };

enum class Format {
    Undefined,
    RGBA8_UNorm,
    BGRA8_UNorm,
    RG32_Float,    // vec2 vertex attribute
    RGB32_Float,   // vec3 vertex attribute
};

struct ClearColor { float r = 0, g = 0, b = 0, a = 1; };

// --- Resource descriptors ----------------------------------------------------

struct VertexAttribute {
    uint32_t location;   // shader input location
    Format   format;     // attribute element format
    uint32_t offset;     // byte offset within a vertex
};

struct VertexLayout {
    uint32_t stride = 0;                       // bytes per vertex
    std::vector<VertexAttribute> attributes;   // per-attribute layout
};

struct ShaderModuleDesc {
    std::span<const uint32_t> spirv;  // SPIR-V words
};

struct GraphicsPipelineDesc {
    class IShaderModule* vertex = nullptr;
    class IShaderModule* fragment = nullptr;
    VertexLayout vertexLayout;
    Format colorFormat = Format::BGRA8_UNorm;  // must match swapchain format
};

struct BufferDesc {
    uint64_t size = 0;          // bytes
    const void* initialData = nullptr;  // optional upload-on-create
};

// --- Resource interfaces -----------------------------------------------------

class IShaderModule { public: virtual ~IShaderModule() = default; };
class IPipeline     { public: virtual ~IPipeline() = default; };
class IBuffer       { public: virtual ~IBuffer() = default; };

// Records draw commands for one frame's swapchain image.
class ICommandBuffer {
public:
    virtual ~ICommandBuffer() = default;
    virtual void BeginRenderPass(const ClearColor& clear) = 0;
    virtual void BindPipeline(IPipeline& pipeline) = 0;
    virtual void BindVertexBuffer(IBuffer& buffer) = 0;
    virtual void Draw(uint32_t vertexCount, uint32_t firstVertex = 0) = 0;
    virtual void EndRenderPass() = 0;
};

class ISwapchain {
public:
    virtual ~ISwapchain() = default;
    virtual Format ColorFormat() const = 0;
    // Recreate after a window resize (or out-of-date acquire/present).
    virtual void Recreate(uint32_t width, uint32_t height) = 0;
};

// Per-frame handle returned by BeginFrame; passed back to EndFrame.
struct FrameContext {
    ICommandBuffer* cmd = nullptr;  // null if the frame was skipped (e.g. minimized)
};

class IRHIDevice {
public:
    virtual ~IRHIDevice() = default;

    virtual ISwapchain& Swapchain() = 0;

    virtual std::unique_ptr<IShaderModule> CreateShaderModule(const ShaderModuleDesc&) = 0;
    virtual std::unique_ptr<IPipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc&) = 0;
    virtual std::unique_ptr<IBuffer> CreateBuffer(const BufferDesc&) = 0;

    // Acquire next swapchain image + begin command recording.
    // Returns FrameContext{nullptr} when the frame must be skipped this tick.
    virtual FrameContext BeginFrame() = 0;
    // Submit recorded commands and present.
    virtual void EndFrame(const FrameContext&) = 0;

    // Block until the GPU is idle (call before destroying GPU resources).
    virtual void WaitIdle() = 0;
};

} // namespace hf::rhi
```

- [ ] **Step 2: Build to verify the header compiles**

Add a throwaway `.cpp` is unnecessary — instead include it from the stub. Temporarily edit `engine/rhi/rhi_factory.cpp` to:
```cpp
#include "rhi/rhi.h"
namespace hf::rhi { void _link_placeholder() {} }
```
Run:
```bash
cmake --build --preset windows-msvc-debug
```
Expected: compiles clean (header is well-formed C++20).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(rhi): define RHI interface seam (device, swapchain, pipeline, buffer)"
```

---

## Task 3: Shaders + DXC build rule

**Files:**
- Create: `cmake/CompileShaders.cmake`, `shaders/triangle.vert.hlsl`, `shaders/triangle.frag.hlsl`
- Modify: top-level `CMakeLists.txt`, `samples/hello_triangle/CMakeLists.txt`

- [ ] **Step 1: Write `shaders/triangle.vert.hlsl`**

```hlsl
struct VSInput {
    [[vk::location(0)]] float2 pos   : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
};

struct VSOutput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};

VSOutput main(VSInput input) {
    VSOutput o;
    o.pos = float4(input.pos, 0.0, 1.0);
    o.color = input.color;
    return o;
}
```

- [ ] **Step 2: Write `shaders/triangle.frag.hlsl`**

```hlsl
struct PSInput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};

float4 main(PSInput input) : SV_Target {
    return float4(input.color, 1.0);
}
```

- [ ] **Step 3: Write `cmake/CompileShaders.cmake`**

```cmake
# hf_compile_shaders(TARGET <tgt> OUTPUT_DIR <dir> SHADERS <file:stage> ...)
# Each SHADERS entry is "<relative-hlsl-path>:<stage>" where stage is vs|ps.
find_program(DXC_EXECUTABLE NAMES dxc REQUIRED)

function(hf_compile_shaders)
  set(oneValue TARGET OUTPUT_DIR)
  set(multiValue SHADERS)
  cmake_parse_arguments(ARG "" "${oneValue}" "${multiValue}" ${ARGN})

  set(spv_outputs "")
  foreach(entry ${ARG_SHADERS})
    string(REPLACE ":" ";" parts "${entry}")
    list(GET parts 0 src)
    list(GET parts 1 stage)
    if(stage STREQUAL "vs")
      set(profile "vs_6_0")
    elseif(stage STREQUAL "ps")
      set(profile "ps_6_0")
    else()
      message(FATAL_ERROR "Unknown shader stage: ${stage}")
    endif()

    get_filename_component(name "${src}" NAME)
    set(out "${ARG_OUTPUT_DIR}/${name}.spv")
    add_custom_command(
      OUTPUT "${out}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${ARG_OUTPUT_DIR}"
      COMMAND "${DXC_EXECUTABLE}" -spirv -T ${profile} -E main
              -fspv-target-env=vulkan1.3
              "${CMAKE_SOURCE_DIR}/${src}" -Fo "${out}"
      DEPENDS "${CMAKE_SOURCE_DIR}/${src}"
      COMMENT "DXC ${src} -> ${out}"
      VERBATIM)
    list(APPEND spv_outputs "${out}")
  endforeach()

  add_custom_target(${ARG_TARGET}_shaders DEPENDS ${spv_outputs})
  add_dependencies(${ARG_TARGET} ${ARG_TARGET}_shaders)
endfunction()
```

- [ ] **Step 4: Include the module in top-level `CMakeLists.txt`**

After `list(APPEND CMAKE_MODULE_PATH ...)` add:
```cmake
include(CompileShaders)
```

- [ ] **Step 5: Wire shader compilation into the sample (`samples/hello_triangle/CMakeLists.txt`)**

Replace the file with:
```cmake
add_executable(hello_triangle main.cpp)
target_link_libraries(hello_triangle PRIVATE hf_engine)

set(SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders")
hf_compile_shaders(
  TARGET hello_triangle
  OUTPUT_DIR "${SHADER_OUT}"
  SHADERS
    "shaders/triangle.vert.hlsl:vs"
    "shaders/triangle.frag.hlsl:ps"
)
target_compile_definitions(hello_triangle PRIVATE
  "HF_SHADER_DIR=\"${SHADER_OUT}\"")
```

- [ ] **Step 6: Build and verify `.spv` files are produced**

Run:
```bash
cmake --preset windows-msvc-debug
cmake --build --preset windows-msvc-debug --target hello_triangle_shaders
ls build/windows-msvc-debug/samples/hello_triangle/shaders/
```
Expected: `triangle.vert.hlsl.spv` and `triangle.frag.hlsl.spv` exist.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "build(shaders): HLSL triangle + DXC SPIR-V compile rule"
```

---

## Task 4: Vulkan backend — instance, device, swapchain bring-up

**Files:**
- Create: `engine/rhi_vulkan/vulkan_device.h`, `engine/rhi_vulkan/vulkan_device.cpp`, `engine/rhi_vulkan/vulkan_swapchain.h`, `engine/rhi_vulkan/vulkan_swapchain.cpp`, `engine/rhi_vulkan/vk_common.h`
- Modify: `engine/CMakeLists.txt`

- [ ] **Step 1: Write `engine/rhi_vulkan/vk_common.h` (shared helpers + format mapping)**

```cpp
#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"
#include <stdexcept>
#include <string>

namespace hf::rhi::vk {

inline void Check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan error in ") + what +
                                 " (VkResult=" + std::to_string((int)r) + ")");
    }
}

inline VkFormat ToVk(Format f) {
    switch (f) {
        case Format::RGBA8_UNorm: return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::BGRA8_UNorm: return VK_FORMAT_B8G8R8A8_UNORM;
        case Format::RG32_Float:  return VK_FORMAT_R32G32_SFLOAT;
        case Format::RGB32_Float: return VK_FORMAT_R32G32B32_SFLOAT;
        default:                  return VK_FORMAT_UNDEFINED;
    }
}

inline Format FromVk(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: return Format::RGBA8_UNorm;
        case VK_FORMAT_B8G8R8A8_UNORM: return Format::BGRA8_UNorm;
        default:                       return Format::Undefined;
    }
}

} // namespace hf::rhi::vk
```

- [ ] **Step 2: Write `engine/rhi_vulkan/vulkan_device.h`**

```cpp
#pragma once
#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"
#include "rhi_vulkan/vulkan_swapchain.h"
#include "hal/window.h"
#include <memory>

namespace hf::rhi::vk {

// Frames-in-flight; double-buffered.
constexpr uint32_t kFramesInFlight = 2;

class VulkanDevice final : public IRHIDevice {
public:
    explicit VulkanDevice(hf::hal::Window& window);
    ~VulkanDevice() override;

    ISwapchain& Swapchain() override { return *swapchain_; }

    std::unique_ptr<IShaderModule> CreateShaderModule(const ShaderModuleDesc&) override;
    std::unique_ptr<IPipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc&) override;
    std::unique_ptr<IBuffer> CreateBuffer(const BufferDesc&) override;

    FrameContext BeginFrame() override;
    void EndFrame(const FrameContext&) override;
    void WaitIdle() override;

    // Accessors used by sibling Vulkan objects.
    VkDevice device() const { return device_; }
    VmaAllocator allocator() const { return allocator_; }

private:
    void CreateSyncObjects();
    void DestroySyncObjects();

    hf::hal::Window& window_;

    vkb::Instance vkbInstance_{};
    vkb::Device   vkbDevice_{};
    VkSurfaceKHR  surface_ = VK_NULL_HANDLE;
    VkInstance    instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice      device_ = VK_NULL_HANDLE;
    VkQueue       graphicsQueue_ = VK_NULL_HANDLE;
    uint32_t      graphicsQueueFamily_ = 0;
    VmaAllocator  allocator_ = VK_NULL_HANDLE;

    std::unique_ptr<VulkanSwapchain> swapchain_;

    // Per-frame-in-flight sync + command resources.
    struct FrameSync {
        VkCommandPool   pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd  = VK_NULL_HANDLE;
        VkSemaphore     imageAvailable = VK_NULL_HANDLE;
        VkSemaphore     renderFinished = VK_NULL_HANDLE;
        VkFence         inFlight = VK_NULL_HANDLE;
    };
    FrameSync frames_[kFramesInFlight];
    uint32_t  frameIndex_ = 0;
    uint32_t  acquiredImage_ = 0;

    std::unique_ptr<class VulkanCommandBuffer> recorder_;
};

} // namespace hf::rhi::vk
```

- [ ] **Step 3: Write `engine/rhi_vulkan/vulkan_swapchain.h`**

```cpp
#pragma once
#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include "rhi/rhi.h"
#include <vector>

namespace hf::rhi::vk {

class VulkanSwapchain final : public ISwapchain {
public:
    VulkanSwapchain(VkDevice device, vkb::Device& vkbDevice,
                    uint32_t width, uint32_t height);
    ~VulkanSwapchain() override;

    Format ColorFormat() const override { return FromVk(format_); }
    void Recreate(uint32_t width, uint32_t height) override;

    VkSwapchainKHR handle() const { return swapchain_; }
    VkFormat vkFormat() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    VkImage image(uint32_t i) const { return images_[i]; }
    VkImageView view(uint32_t i) const { return views_[i]; }
    uint32_t imageCount() const { return (uint32_t)images_.size(); }

private:
    void Build(uint32_t width, uint32_t height);
    void Destroy();

    VkDevice device_;
    vkb::Device& vkbDevice_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
};

} // namespace hf::rhi::vk
```

- [ ] **Step 4: Write `engine/rhi_vulkan/vulkan_swapchain.cpp`**

```cpp
#include "rhi_vulkan/vulkan_swapchain.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanSwapchain::VulkanSwapchain(VkDevice device, vkb::Device& vkbDevice,
                                 uint32_t width, uint32_t height)
    : device_(device), vkbDevice_(vkbDevice) {
    Build(width, height);
}

VulkanSwapchain::~VulkanSwapchain() { Destroy(); }

void VulkanSwapchain::Build(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder builder{vkbDevice_};
    auto ret = builder
        .set_desired_format(VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM,
                                               VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        .build();
    if (!ret) {
        throw std::runtime_error("Swapchain build failed: " + ret.error().message());
    }
    vkb::Swapchain vkbSwap = ret.value();
    swapchain_ = vkbSwap.swapchain;
    format_ = vkbSwap.image_format;
    extent_ = vkbSwap.extent;
    images_ = vkbSwap.get_images().value();
    views_ = vkbSwap.get_image_views().value();
}

void VulkanSwapchain::Destroy() {
    for (auto v : views_) vkDestroyImageView(device_, v, nullptr);
    views_.clear();
    images_.clear();
    if (swapchain_) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VulkanSwapchain::Recreate(uint32_t width, uint32_t height) {
    vkDeviceWaitIdle(device_);
    Destroy();
    Build(width, height);
}

} // namespace hf::rhi::vk
```

- [ ] **Step 5: Write the bring-up portion of `engine/rhi_vulkan/vulkan_device.cpp` (ctor/dtor + sync)**

> The resource-creation and frame methods are added in Tasks 5–8. Write this file now with the constructor, destructor, sync helpers, and `WaitIdle`. Stub the not-yet-implemented overrides to `throw std::runtime_error("not implemented")` so the class is concrete and links.

```cpp
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"
#include "rhi_vulkan/vulkan_command_buffer.h"  // created in Task 8

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace hf::rhi::vk {

VulkanDevice::VulkanDevice(hf::hal::Window& window) : window_(window) {
    // --- Instance ---
    vkb::InstanceBuilder ib;
    ib.set_app_name("Hazard Forge")
      .require_api_version(1, 3, 0)
#ifndef NDEBUG
      .request_validation_layers(true)
      .use_default_debug_messenger()
#endif
      ;
    for (const char* ext : window_.RequiredVulkanInstanceExtensions()) {
        ib.enable_extension(ext);
    }
    auto instRet = ib.build();
    if (!instRet) throw std::runtime_error("Instance build failed: " + instRet.error().message());
    vkbInstance_ = instRet.value();
    instance_ = vkbInstance_.instance;

    // --- Surface ---
    surface_ = window_.CreateVulkanSurface(instance_);

    // --- Physical device (require dynamic rendering + sync2) ---
    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{vkbInstance_};
    auto physRet = selector
        .set_surface(surface_)
        .set_minimum_version(1, 3)
        .set_required_features_13(f13)
        .select();
    if (!physRet) throw std::runtime_error("GPU select failed: " + physRet.error().message());

    vkb::DeviceBuilder db{physRet.value()};
    auto devRet = db.build();
    if (!devRet) throw std::runtime_error("Device build failed: " + devRet.error().message());
    vkbDevice_ = devRet.value();
    device_ = vkbDevice_.device;
    physical_ = physRet.value().physical_device;

    graphicsQueue_ = vkbDevice_.get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily_ = vkbDevice_.get_queue_index(vkb::QueueType::graphics).value();

    // --- VMA ---
    VmaAllocatorCreateInfo aci{};
    aci.physicalDevice = physical_;
    aci.device = device_;
    aci.instance = instance_;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    Check(vmaCreateAllocator(&aci, &allocator_), "vmaCreateAllocator");

    // --- Swapchain ---
    swapchain_ = std::make_unique<VulkanSwapchain>(
        device_, vkbDevice_, window_.FramebufferWidth(), window_.FramebufferHeight());

    CreateSyncObjects();
    recorder_ = std::make_unique<VulkanCommandBuffer>(*this);
}

VulkanDevice::~VulkanDevice() {
    if (device_) vkDeviceWaitIdle(device_);
    recorder_.reset();
    DestroySyncObjects();
    swapchain_.reset();
    if (allocator_) vmaDestroyAllocator(allocator_);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (device_) vkb::destroy_device(vkbDevice_);
    if (instance_) vkb::destroy_instance(vkbInstance_);
}

void VulkanDevice::CreateSyncObjects() {
    for (auto& fr : frames_) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = graphicsQueueFamily_;
        Check(vkCreateCommandPool(device_, &pci, nullptr, &fr.pool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbi.commandPool = fr.pool;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        Check(vkAllocateCommandBuffers(device_, &cbi, &fr.cmd), "vkAllocateCommandBuffers");

        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        Check(vkCreateSemaphore(device_, &si, nullptr, &fr.imageAvailable), "sem");
        Check(vkCreateSemaphore(device_, &si, nullptr, &fr.renderFinished), "sem");

        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        Check(vkCreateFence(device_, &fi, nullptr, &fr.inFlight), "fence");
    }
}

void VulkanDevice::DestroySyncObjects() {
    for (auto& fr : frames_) {
        if (fr.inFlight) vkDestroyFence(device_, fr.inFlight, nullptr);
        if (fr.imageAvailable) vkDestroySemaphore(device_, fr.imageAvailable, nullptr);
        if (fr.renderFinished) vkDestroySemaphore(device_, fr.renderFinished, nullptr);
        if (fr.pool) vkDestroyCommandPool(device_, fr.pool, nullptr);
        fr = FrameSync{};
    }
}

void VulkanDevice::WaitIdle() { if (device_) vkDeviceWaitIdle(device_); }

// --- Stubs replaced in later tasks ---
std::unique_ptr<IShaderModule> VulkanDevice::CreateShaderModule(const ShaderModuleDesc&) {
    throw std::runtime_error("CreateShaderModule not implemented yet");
}
std::unique_ptr<IPipeline> VulkanDevice::CreateGraphicsPipeline(const GraphicsPipelineDesc&) {
    throw std::runtime_error("CreateGraphicsPipeline not implemented yet");
}
std::unique_ptr<IBuffer> VulkanDevice::CreateBuffer(const BufferDesc&) {
    throw std::runtime_error("CreateBuffer not implemented yet");
}
FrameContext VulkanDevice::BeginFrame() {
    throw std::runtime_error("BeginFrame not implemented yet");
}
void VulkanDevice::EndFrame(const FrameContext&) {
    throw std::runtime_error("EndFrame not implemented yet");
}

} // namespace hf::rhi::vk
```

> Tasks 5–8 replace the five stubs and add the named files (`vulkan_shader.*`, `vulkan_pipeline.*`, `vulkan_buffer.*`, `vulkan_command_buffer.*`). The `#include "rhi_vulkan/vulkan_command_buffer.h"` and `VulkanCommandBuffer` usages are satisfied in Task 8 — to build *this* task in isolation, temporarily comment out that include, the `recorder_` member, and its `make_unique`, then restore them in Task 8. (Noted in Task 8 Step 1.)

- [ ] **Step 6: Add Vulkan sources to `engine/CMakeLists.txt` and define `VK_NO_PROTOTYPES` off**

Update the library list:
```cmake
add_library(hf_engine STATIC
  hal/window.cpp
  rhi/rhi_factory.cpp
  rhi_vulkan/vulkan_device.cpp
  rhi_vulkan/vulkan_swapchain.cpp
)
```

- [ ] **Step 7: Build (with Task 8 references temporarily commented per Step 5 note)**

Run:
```bash
cmake --build --preset windows-msvc-debug
```
Expected: `hf_engine` compiles. Linker may complain about `VulkanDevice` only if instantiated — it is not yet, so this builds.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(vulkan): instance/device/VMA/swapchain bring-up via vk-bootstrap"
```

---

## Task 5: Vulkan shader module

**Files:**
- Create: `engine/rhi_vulkan/vulkan_shader.h`, `engine/rhi_vulkan/vulkan_shader.cpp`
- Modify: `engine/rhi_vulkan/vulkan_device.cpp`, `engine/CMakeLists.txt`

- [ ] **Step 1: Write `engine/rhi_vulkan/vulkan_shader.h`**

```cpp
#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanShaderModule final : public IShaderModule {
public:
    VulkanShaderModule(VkDevice device, std::span<const uint32_t> spirv);
    ~VulkanShaderModule() override;
    VkShaderModule handle() const { return module_; }

private:
    VkDevice device_;
    VkShaderModule module_ = VK_NULL_HANDLE;
};

} // namespace hf::rhi::vk
```

- [ ] **Step 2: Write `engine/rhi_vulkan/vulkan_shader.cpp`**

```cpp
#include "rhi_vulkan/vulkan_shader.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanShaderModule::VulkanShaderModule(VkDevice device, std::span<const uint32_t> spirv)
    : device_(device) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = spirv.size() * sizeof(uint32_t);
    ci.pCode = spirv.data();
    Check(vkCreateShaderModule(device_, &ci, nullptr, &module_), "vkCreateShaderModule");
}

VulkanShaderModule::~VulkanShaderModule() {
    if (module_) vkDestroyShaderModule(device_, module_, nullptr);
}

} // namespace hf::rhi::vk
```

- [ ] **Step 3: Replace the `CreateShaderModule` stub in `vulkan_device.cpp`**

Add `#include "rhi_vulkan/vulkan_shader.h"` near the top, then replace the stub with:
```cpp
std::unique_ptr<IShaderModule> VulkanDevice::CreateShaderModule(const ShaderModuleDesc& d) {
    return std::make_unique<VulkanShaderModule>(device_, d.spirv);
}
```

- [ ] **Step 4: Add to `engine/CMakeLists.txt`**

Add `rhi_vulkan/vulkan_shader.cpp` to the `hf_engine` source list.

- [ ] **Step 5: Build**

Run:
```bash
cmake --build --preset windows-msvc-debug
```
Expected: compiles clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(vulkan): shader module from SPIR-V"
```

---

## Task 6: Vulkan graphics pipeline (dynamic rendering)

**Files:**
- Create: `engine/rhi_vulkan/vulkan_pipeline.h`, `engine/rhi_vulkan/vulkan_pipeline.cpp`
- Modify: `engine/rhi_vulkan/vulkan_device.cpp`, `engine/CMakeLists.txt`

- [ ] **Step 1: Write `engine/rhi_vulkan/vulkan_pipeline.h`**

```cpp
#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanPipeline final : public IPipeline {
public:
    VulkanPipeline(VkDevice device, const GraphicsPipelineDesc& desc);
    ~VulkanPipeline() override;
    VkPipeline handle() const { return pipeline_; }

private:
    VkDevice device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

} // namespace hf::rhi::vk
```

- [ ] **Step 2: Write `engine/rhi_vulkan/vulkan_pipeline.cpp`**

```cpp
#include "rhi_vulkan/vulkan_pipeline.h"
#include "rhi_vulkan/vulkan_shader.h"
#include "rhi_vulkan/vk_common.h"
#include <vector>

namespace hf::rhi::vk {

VulkanPipeline::VulkanPipeline(VkDevice device, const GraphicsPipelineDesc& desc)
    : device_(device) {
    auto* vs = static_cast<VulkanShaderModule*>(desc.vertex);
    auto* fs = static_cast<VulkanShaderModule*>(desc.fragment);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs->handle();
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs->handle();
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = desc.vertexLayout.stride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::vector<VkVertexInputAttributeDescription> attrs;
    for (const auto& a : desc.vertexLayout.attributes) {
        attrs.push_back({a.location, 0, ToVk(a.format), a.offset});
    }

    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
    vi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &blendAtt;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    Check(vkCreatePipelineLayout(device_, &lci, nullptr, &layout_), "vkCreatePipelineLayout");

    VkFormat colorFormat = ToVk(desc.colorFormat);
    VkPipelineRenderingCreateInfo rendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &colorFormat;

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.pNext = &rendering;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &dyn;
    pci.layout = layout_;
    Check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_),
          "vkCreateGraphicsPipelines");
}

VulkanPipeline::~VulkanPipeline() {
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
}

} // namespace hf::rhi::vk
```

- [ ] **Step 3: Replace the `CreateGraphicsPipeline` stub in `vulkan_device.cpp`**

Add `#include "rhi_vulkan/vulkan_pipeline.h"`, replace the stub with:
```cpp
std::unique_ptr<IPipeline> VulkanDevice::CreateGraphicsPipeline(const GraphicsPipelineDesc& d) {
    return std::make_unique<VulkanPipeline>(device_, d);
}
```

- [ ] **Step 4: Add to `engine/CMakeLists.txt`**

Add `rhi_vulkan/vulkan_pipeline.cpp` to the source list.

- [ ] **Step 5: Build**

Run:
```bash
cmake --build --preset windows-msvc-debug
```
Expected: compiles clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(vulkan): graphics pipeline with dynamic rendering"
```

---

## Task 7: Vulkan vertex buffer (VMA)

**Files:**
- Create: `engine/rhi_vulkan/vulkan_buffer.h`, `engine/rhi_vulkan/vulkan_buffer.cpp`
- Modify: `engine/rhi_vulkan/vulkan_device.cpp`, `engine/CMakeLists.txt`

- [ ] **Step 1: Write `engine/rhi_vulkan/vulkan_buffer.h`**

```cpp
#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanBuffer final : public IBuffer {
public:
    VulkanBuffer(VmaAllocator allocator, const BufferDesc& desc);
    ~VulkanBuffer() override;
    VkBuffer handle() const { return buffer_; }

private:
    VmaAllocator allocator_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
};

} // namespace hf::rhi::vk
```

- [ ] **Step 2: Write `engine/rhi_vulkan/vulkan_buffer.cpp`**

> Uses a host-visible, host-coherent vertex buffer (simplest correct upload for a static triangle — no staging buffer needed for Slice A).

```cpp
#include "rhi_vulkan/vulkan_buffer.h"
#include "rhi_vulkan/vk_common.h"
#include <cstring>

namespace hf::rhi::vk {

VulkanBuffer::VulkanBuffer(VmaAllocator allocator, const BufferDesc& desc)
    : allocator_(allocator) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = desc.size;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    Check(vmaCreateBuffer(allocator_, &bci, &aci, &buffer_, &allocation_, &info),
          "vmaCreateBuffer");

    if (desc.initialData && info.pMappedData) {
        std::memcpy(info.pMappedData, desc.initialData, desc.size);
        vmaFlushAllocation(allocator_, allocation_, 0, desc.size);
    }
}

VulkanBuffer::~VulkanBuffer() {
    if (buffer_) vmaDestroyBuffer(allocator_, buffer_, allocation_);
}

} // namespace hf::rhi::vk
```

- [ ] **Step 3: Replace the `CreateBuffer` stub in `vulkan_device.cpp`**

Add `#include "rhi_vulkan/vulkan_buffer.h"`, replace the stub with:
```cpp
std::unique_ptr<IBuffer> VulkanDevice::CreateBuffer(const BufferDesc& d) {
    return std::make_unique<VulkanBuffer>(allocator_, d);
}
```

- [ ] **Step 4: Add to `engine/CMakeLists.txt`**

Add `rhi_vulkan/vulkan_buffer.cpp` to the source list.

- [ ] **Step 5: Build**

Run:
```bash
cmake --build --preset windows-msvc-debug
```
Expected: compiles clean.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(vulkan): host-visible vertex buffer via VMA"
```

---

## Task 8: Command buffer + frame loop (sync2, dynamic rendering)

**Files:**
- Create: `engine/rhi_vulkan/vulkan_command_buffer.h`, `engine/rhi_vulkan/vulkan_command_buffer.cpp`
- Modify: `engine/rhi_vulkan/vulkan_device.cpp`, `engine/CMakeLists.txt`

- [ ] **Step 1: Restore the Task 8 references in `vulkan_device.cpp`**

If you commented out the `#include "rhi_vulkan/vulkan_command_buffer.h"`, the `recorder_` member, or its `make_unique` in Task 4 Step 5, uncomment them now.

- [ ] **Step 2: Write `engine/rhi_vulkan/vulkan_command_buffer.h`**

```cpp
#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanDevice;

// Thin recorder over a VkCommandBuffer owned by VulkanDevice's per-frame sync.
class VulkanCommandBuffer final : public ICommandBuffer {
public:
    explicit VulkanCommandBuffer(VulkanDevice& device);

    // Called by VulkanDevice::BeginFrame to retarget this recorder.
    void Begin(VkCommandBuffer cmd, VkImageView colorView, VkExtent2D extent);

    void BeginRenderPass(const ClearColor& clear) override;
    void BindPipeline(IPipeline& pipeline) override;
    void BindVertexBuffer(IBuffer& buffer) override;
    void Draw(uint32_t vertexCount, uint32_t firstVertex) override;
    void EndRenderPass() override;

private:
    VulkanDevice& device_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkImageView colorView_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
};

} // namespace hf::rhi::vk
```

- [ ] **Step 3: Write `engine/rhi_vulkan/vulkan_command_buffer.cpp`**

```cpp
#include "rhi_vulkan/vulkan_command_buffer.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vulkan_pipeline.h"
#include "rhi_vulkan/vulkan_buffer.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanCommandBuffer::VulkanCommandBuffer(VulkanDevice& device) : device_(device) {}

void VulkanCommandBuffer::Begin(VkCommandBuffer cmd, VkImageView colorView, VkExtent2D extent) {
    cmd_ = cmd;
    colorView_ = colorView;
    extent_ = extent;
}

void VulkanCommandBuffer::BeginRenderPass(const ClearColor& clear) {
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = colorView_;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{clear.r, clear.g, clear.b, clear.a}};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent_};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cmd_, &ri);

    VkViewport vp{0, 0, (float)extent_.width, (float)extent_.height, 0.0f, 1.0f};
    vkCmdSetViewport(cmd_, 0, 1, &vp);
    VkRect2D scissor{{0, 0}, extent_};
    vkCmdSetScissor(cmd_, 0, 1, &scissor);
}

void VulkanCommandBuffer::BindPipeline(IPipeline& pipeline) {
    auto& p = static_cast<VulkanPipeline&>(pipeline);
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, p.handle());
}

void VulkanCommandBuffer::BindVertexBuffer(IBuffer& buffer) {
    auto& b = static_cast<VulkanBuffer&>(buffer);
    VkBuffer vb = b.handle();
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd_, 0, 1, &vb, &offset);
}

void VulkanCommandBuffer::Draw(uint32_t vertexCount, uint32_t firstVertex) {
    vkCmdDraw(cmd_, vertexCount, 1, firstVertex, 0);
}

void VulkanCommandBuffer::EndRenderPass() {
    vkCmdEndRendering(cmd_);
}

} // namespace hf::rhi::vk
```

- [ ] **Step 4: Implement `BeginFrame` / `EndFrame` in `vulkan_device.cpp`**

Replace the `BeginFrame` and `EndFrame` stubs with the following. Add a small static helper for image-layout transitions using sync2.

```cpp
namespace {
void TransitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout oldLayout, VkImageLayout newLayout,
                     VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                     VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = srcStage;  b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;  b.dstAccessMask = dstAccess;
    b.oldLayout = oldLayout;    b.newLayout = newLayout;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}
} // namespace

FrameContext VulkanDevice::BeginFrame() {
    FrameSync& fr = frames_[frameIndex_];
    vkWaitForFences(device_, 1, &fr.inFlight, VK_TRUE, UINT64_MAX);

    VkResult acq = vkAcquireNextImageKHR(device_, swapchain_->handle(), UINT64_MAX,
                                         fr.imageAvailable, VK_NULL_HANDLE, &acquiredImage_);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchain_->Recreate(window_.FramebufferWidth(), window_.FramebufferHeight());
        return FrameContext{nullptr};
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) Check(acq, "vkAcquireNextImageKHR");

    vkResetFences(device_, 1, &fr.inFlight);
    vkResetCommandBuffer(fr.cmd, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(fr.cmd, &bi), "vkBeginCommandBuffer");

    // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
    TransitionImage(fr.cmd, swapchain_->image(acquiredImage_),
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    recorder_->Begin(fr.cmd, swapchain_->view(acquiredImage_), swapchain_->extent());
    return FrameContext{recorder_.get()};
}

void VulkanDevice::EndFrame(const FrameContext& frame) {
    if (!frame.cmd) return;  // skipped frame
    FrameSync& fr = frames_[frameIndex_];

    // COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC
    TransitionImage(fr.cmd, swapchain_->image(acquiredImage_),
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);

    Check(vkEndCommandBuffer(fr.cmd), "vkEndCommandBuffer");

    VkSemaphoreSubmitInfo waitSem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitSem.semaphore = fr.imageAvailable;
    waitSem.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo signalSem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signalSem.semaphore = fr.renderFinished;
    signalSem.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    VkCommandBufferSubmitInfo cmdSub{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdSub.commandBuffer = fr.cmd;

    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &waitSem;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdSub;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signalSem;
    Check(vkQueueSubmit2(graphicsQueue_, 1, &submit, fr.inFlight), "vkQueueSubmit2");

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &fr.renderFinished;
    VkSwapchainKHR sc = swapchain_->handle();
    present.swapchainCount = 1;
    present.pSwapchains = &sc;
    present.pImageIndices = &acquiredImage_;
    VkResult pr = vkQueuePresentKHR(graphicsQueue_, &present);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        swapchain_->Recreate(window_.FramebufferWidth(), window_.FramebufferHeight());
    } else {
        Check(pr, "vkQueuePresentKHR");
    }

    frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
}
```

- [ ] **Step 5: Add to `engine/CMakeLists.txt`**

Add `rhi_vulkan/vulkan_command_buffer.cpp` to the source list.

- [ ] **Step 6: Build**

Run:
```bash
cmake --build --preset windows-msvc-debug
```
Expected: `hf_engine` compiles and links clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(vulkan): frame loop — acquire/record/submit/present with sync2"
```

---

## Task 9: RHI factory

**Files:**
- Create: `engine/rhi/rhi_factory.h`
- Modify: `engine/rhi/rhi_factory.cpp` (replace stub)

- [ ] **Step 1: Write `engine/rhi/rhi_factory.h`**

```cpp
#pragma once
#include <memory>
#include "rhi/rhi.h"

namespace hf::hal { class Window; }

namespace hf::rhi {

// Create a backend device bound to the given window.
std::unique_ptr<IRHIDevice> CreateDevice(Backend backend, hf::hal::Window& window);

} // namespace hf::rhi
```

- [ ] **Step 2: Replace `engine/rhi/rhi_factory.cpp`**

```cpp
#include "rhi/rhi_factory.h"
#include "rhi_vulkan/vulkan_device.h"
#include <stdexcept>

namespace hf::rhi {

std::unique_ptr<IRHIDevice> CreateDevice(Backend backend, hf::hal::Window& window) {
    switch (backend) {
        case Backend::Vulkan:
            return std::make_unique<vk::VulkanDevice>(window);
        default:
            throw std::runtime_error("Unsupported RHI backend");
    }
}

} // namespace hf::rhi
```

- [ ] **Step 3: Build**

Run:
```bash
cmake --build --preset windows-msvc-debug
```
Expected: compiles and links clean.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(rhi): backend factory (CreateDevice -> Vulkan)"
```

---

## Task 10: Wire up `hello_triangle` + headless smoke test

**Files:**
- Modify: `samples/hello_triangle/main.cpp`
- Create: `tests/rhi_smoke.cpp`, `tests/CMakeLists.txt` (replace stub)

- [ ] **Step 1: Write the full `samples/hello_triangle/main.cpp`**

```cpp
#include "hal/window.h"
#include "rhi/rhi.h"
#include "rhi/rhi_factory.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<uint32_t> LoadSpirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open shader: " + path);
    std::streamsize size = f.tellg();
    if (size % 4 != 0) throw std::runtime_error("SPIR-V size not multiple of 4: " + path);
    f.seekg(0);
    std::vector<uint32_t> words(size / 4);
    f.read(reinterpret_cast<char*>(words.data()), size);
    return words;
}

struct Vertex { float pos[2]; float color[3]; };

} // namespace

int main() {
    using namespace hf;
    try {
        hal::Window window({"Hazard Forge — Hello Triangle", 1280, 720});
        auto device = rhi::CreateDevice(rhi::Backend::Vulkan, window);

        auto vsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/triangle.vert.hlsl.spv");
        auto fsWords = LoadSpirv(std::string(HF_SHADER_DIR) + "/triangle.frag.hlsl.spv");
        auto vs = device->CreateShaderModule({std::span<const uint32_t>(vsWords)});
        auto fs = device->CreateShaderModule({std::span<const uint32_t>(fsWords)});

        rhi::VertexLayout layout;
        layout.stride = sizeof(Vertex);
        layout.attributes = {
            {0, rhi::Format::RG32_Float,  offsetof(Vertex, pos)},
            {1, rhi::Format::RGB32_Float, offsetof(Vertex, color)},
        };

        rhi::GraphicsPipelineDesc pdesc;
        pdesc.vertex = vs.get();
        pdesc.fragment = fs.get();
        pdesc.vertexLayout = layout;
        pdesc.colorFormat = device->Swapchain().ColorFormat();
        auto pipeline = device->CreateGraphicsPipeline(pdesc);

        const Vertex verts[3] = {
            {{ 0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
            {{ 0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
            {{-0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
        };
        rhi::BufferDesc bdesc;
        bdesc.size = sizeof(verts);
        bdesc.initialData = verts;
        auto vbuffer = device->CreateBuffer(bdesc);

        bool running = true;
        while (running) {
            running = window.PumpEvents();
            if (window.ConsumeResized()) {
                device->WaitIdle();
                device->Swapchain().Recreate(window.FramebufferWidth(),
                                             window.FramebufferHeight());
            }
            auto frame = device->BeginFrame();
            if (frame.cmd) {
                frame.cmd->BeginRenderPass(rhi::ClearColor{0.02f, 0.02f, 0.05f, 1.0f});
                frame.cmd->BindPipeline(*pipeline);
                frame.cmd->BindVertexBuffer(*vbuffer);
                frame.cmd->Draw(3);
                frame.cmd->EndRenderPass();
            }
            device->EndFrame(frame);
        }

        device->WaitIdle();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
```

- [ ] **Step 2: Build and RUN the triangle (manual visual verification)**

Run:
```bash
cmake --build --preset windows-msvc-debug
./build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe
```
Expected: a 1280×720 window shows a triangle with red/green/blue corners on a dark background. **Resize the window** — the triangle stays centered and correctly proportioned. Close it — process exits 0. **No `VK_DEBUG`/validation messages** print to stderr.

- [ ] **Step 3: Write `tests/rhi_smoke.cpp` (headless create/destroy)**

> SDL can create a hidden window without presenting; this exercises full device + swapchain init/teardown — the regression-prone part — without asserting on rendered pixels.

```cpp
#include "hal/window.h"
#include "rhi/rhi.h"
#include "rhi/rhi_factory.h"
#include <cstdio>

int main() {
    using namespace hf;
    try {
        hal::Window window({"hf-smoke", 64, 64});
        auto device = rhi::CreateDevice(rhi::Backend::Vulkan, window);
        if (device->Swapchain().ColorFormat() == rhi::Format::Undefined) {
            std::fprintf(stderr, "swapchain has undefined format\n");
            return 1;
        }
        device->WaitIdle();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "smoke FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("smoke OK\n");
    return 0;
}
```

- [ ] **Step 4: Write `tests/CMakeLists.txt`**

```cmake
add_executable(rhi_smoke rhi_smoke.cpp)
target_link_libraries(rhi_smoke PRIVATE hf_engine)
add_test(NAME rhi_smoke COMMAND rhi_smoke)
```

- [ ] **Step 5: Build and run the test**

Run:
```bash
cmake --build --preset windows-msvc-debug
ctest --preset windows-msvc-debug
```
Expected: `1/1 Test #1: rhi_smoke .......... Passed`. (Requires a GPU/Vulkan loader present; on a headless CI runner without a GPU this test is expected to be skipped/excluded later.)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(sample): hello_triangle renders + headless RHI smoke test"
```

---

## Definition of Done (verify all)

- [ ] `cmake --build --preset windows-msvc-debug` succeeds from clean.
- [ ] `hello_triangle.exe` shows a red/green/blue triangle on a dark window.
- [ ] Window resizes without crash or distortion; closes cleanly (exit 0).
- [ ] Zero Vulkan validation-layer errors in the debug run.
- [ ] `ctest --preset windows-msvc-debug` passes `rhi_smoke`.
- [ ] No `vk*` symbol appears anywhere under `engine/rhi/` (grep to confirm:
      `grep -rn "vk" engine/rhi/ | grep -iv "rhi\|//"` returns only false positives).

---

## Self-Review Notes (author)

- **Spec coverage:** repo layout ✓ (Task 0); RHI seam interfaces ✓ (Task 2); vk-bootstrap+VMA contained in `rhi_vulkan/` ✓ (Tasks 4–8); HLSL→SPIR-V via DXC ✓ (Task 3); per-frame fences+semaphores, double-buffered ✓ (Task 4/8); validation layers in debug ✓ (Task 4); resize/out-of-date recreation ✓ (Task 8); CTest headless smoke ✓ (Task 10); no-`vk*`-in-`rhi/` rule enforced by DoD grep ✓.
- **Type consistency:** `IShaderModule`/`IPipeline`/`IBuffer`/`ICommandBuffer`/`ISwapchain`/`IRHIDevice`, `FrameContext{cmd}`, `GraphicsPipelineDesc`, `VertexLayout/VertexAttribute`, `BufferDesc{size,initialData}`, `ShaderModuleDesc{spirv}` used identically across Tasks 2, 4–10. `CreateDevice(Backend, Window&)` matches factory header and call site.
- **Known simplifications (intended for Slice A):** host-visible vertex buffer (no staging); single pipeline layout with no descriptors; FIFO present mode; format mapping covers only what the triangle needs.
