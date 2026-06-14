# Slice B: 3D Core (Spinning Cube) — Implementation Plan

> **For agentic workers:** built via subagent-driven-development. Checkbox steps. The Slice A
> toolchain is proven; environment facts are in the "Environment" section — don't re-derive them.

**Goal:** A spinning, depth-correct, perspective 3D cube rendered through the RHI with an MVP
push constant and a camera. Screenshot-verified, zero validation errors.

**Builds on:** Slice A (RHI seam + Vulkan backend, on `master`). This slice is additive.

---

## Environment (proven in Slice A — trust)
- Build inside a VS BuildTools x64 dev shell: `& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64` then run cmake in that session.
- Conan deps already installed. Configure/build: `cmake --preset windows-msvc-debug` then `cmake --build --preset windows-msvc-debug`. Only re-run conan if a package is missing: `conan install . -of=build/windows-msvc-debug -s build_type=Debug -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing`.
- Shaders compile via DXC (HINTS find the SPIR-V-capable standalone dxc; Windows-SDK dxc lacks SPIR-V codegen — do not let it win).
- Do NOT run the interactive sample (infinite loop). Automated gates are `ctest` (`rhi_smoke`, `math_test`). Visual verification is done by the controller via screenshot.
- engine/CMakeLists.txt links: `sdl::sdl`, `vk-bootstrap::vk-bootstrap`, `GPUOpen::VulkanMemoryAllocator`, `vulkan-headers::vulkan-headers`, `Vulkan::Loader`. Add new sources only.

---

## Task B1: Math library + math_test

**Files:** create `engine/math/math.h`; create `tests/math_test.cpp`; modify `tests/CMakeLists.txt`.

- [ ] **Step 1: Create `engine/math/math.h`** (column-major, right-handed, Vulkan [0,1] depth, Y-flipped projection)

```cpp
#pragma once
#include <cmath>
#include <cstdint>

namespace hf::math {

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline Vec3 operator*(const Vec3& a, float s) { return {a.x*s, a.y*s, a.z*s}; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline Vec3 normalize(const Vec3& v) {
    float len = std::sqrt(dot(v, v));
    return len > 0 ? Vec3{v.x/len, v.y/len, v.z/len} : v;
}

// Column-major 4x4: element(row, col) == m[col*4 + row].
struct Mat4 {
    float m[16] = {0};

    static Mat4 Identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    // Right-handed, depth range [0,1] (Vulkan), Y flipped for Vulkan clip space.
    static Mat4 Perspective(float fovYRad, float aspect, float zNear, float zFar) {
        float t = std::tan(fovYRad * 0.5f);
        Mat4 r;  // all zeros
        r.m[0]  = 1.0f / (aspect * t);       // col0,row0
        r.m[5]  = -1.0f / t;                  // col1,row1 (negative = Vulkan Y flip)
        r.m[10] = zFar / (zNear - zFar);      // col2,row2
        r.m[11] = -1.0f;                      // col2,row3
        r.m[14] = (zNear * zFar) / (zNear - zFar); // col3,row2
        return r;
    }

    static Mat4 LookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        Vec3 f = normalize(center - eye);
        Vec3 s = normalize(cross(f, up));
        Vec3 u = cross(s, f);
        Mat4 r = Identity();
        r.m[0] = s.x; r.m[4] = s.y; r.m[8]  = s.z;   // row0 = s
        r.m[1] = u.x; r.m[5] = u.y; r.m[9]  = u.z;   // row1 = u
        r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z; // row2 = -f
        r.m[12] = -dot(s, eye);
        r.m[13] = -dot(u, eye);
        r.m[14] = dot(f, eye);
        return r;
    }

    static Mat4 Translate(const Vec3& t) {
        Mat4 r = Identity();
        r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
        return r;
    }

    static Mat4 RotateY(float rad) {
        float c = std::cos(rad), s = std::sin(rad);
        Mat4 r = Identity();
        r.m[0] = c;  r.m[8] = s;
        r.m[2] = -s; r.m[10] = c;
        return r;
    }

    static Mat4 RotateX(float rad) {
        float c = std::cos(rad), s = std::sin(rad);
        Mat4 r = Identity();
        r.m[5] = c;  r.m[9] = -s;
        r.m[6] = s;  r.m[10] = c;
        return r;
    }
};

// C = A * B  (standard matrix product; C(row,col) = sum_k A(row,k)*B(k,col)).
inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 c;  // zeros
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row) {
            float sum = 0;
            for (int k = 0; k < 4; ++k)
                sum += a.m[k*4 + row] * b.m[col*4 + k];
            c.m[col*4 + row] = sum;
        }
    return c;
}

} // namespace hf::math
```

- [ ] **Step 2: Create `tests/math_test.cpp`** (headless, no GPU, deterministic)

```cpp
#include "math/math.h"
#include <cmath>
#include <cstdio>

using namespace hf::math;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
    // Identity * Identity == Identity
    Mat4 i = Mat4::Identity();
    Mat4 ii = i * i;
    for (int k = 0; k < 16; ++k) check(approx(ii.m[k], i.m[k]), "I*I==I");

    // Translate composes: T(1,2,3) applied to origin via column-major MVP-style multiply.
    Mat4 t = Mat4::Translate({1, 2, 3});
    check(approx(t.m[12], 1) && approx(t.m[13], 2) && approx(t.m[14], 3), "translate col3");

    // LookAt from (0,0,5) toward origin: eye maps to ~origin in view space (translation z = -5 along -f).
    Mat4 v = Mat4::LookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0});
    // For a point at world origin, view-space z should be -5 (in front of camera, RH -z forward).
    // p_view = V * [0,0,0,1]; z component = m[2]*0 + m[6]*0 + m[10]*0 + m[14].
    check(approx(v.m[14], -5.0f), "lookat translates eye to -5 view z");

    // Perspective basic sanity: finite, m[11] == -1, aspect scaling on x.
    Mat4 p = Mat4::Perspective(1.0472f /*60deg*/, 16.0f/9.0f, 0.1f, 100.0f);
    check(approx(p.m[11], -1.0f), "persp m[11]==-1");
    check(p.m[0] > 0 && p.m[5] < 0, "persp x>0, y<0 (Vulkan flip)");

    // Associativity: (P*V)*M == P*(V*M)
    Mat4 m = Mat4::RotateY(0.7f);
    Mat4 lhs = (p * v) * m;
    Mat4 rhs = p * (v * m);
    for (int k = 0; k < 16; ++k) check(approx(lhs.m[k], rhs.m[k], 1e-3f), "assoc");

    if (g_fail == 0) { std::printf("math_test OK\n"); return 0; }
    std::printf("math_test: %d failures\n", g_fail);
    return 1;
}
```

- [ ] **Step 3: Add to `tests/CMakeLists.txt`** (keep existing rhi_smoke)

```cmake
add_executable(rhi_smoke rhi_smoke.cpp)
target_link_libraries(rhi_smoke PRIVATE hf_engine)
add_test(NAME rhi_smoke COMMAND rhi_smoke)

add_executable(math_test math_test.cpp)
target_link_libraries(math_test PRIVATE hf_engine)
add_test(NAME math_test COMMAND math_test)
```

(Math is header-only; `math_test` links `hf_engine` only for the include path — add
`engine/math` to hf_engine's PUBLIC include dir if needed, or include via the existing
`target_include_directories(hf_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})` in engine/, which
already exposes `math/math.h`.)

- [ ] **Step 4: Build, run `ctest` — math_test passes.** Commit: `feat(math): column-major RH math library + tests`

---

## Task B2: RHI seam extensions

**File:** modify `engine/rhi/rhi.h`. Additive only; do not break Slice A users.

- [ ] **Step 1:** In `enum class Format`, add `D32_Float` after the existing entries.
- [ ] **Step 2:** Add a buffer usage enum and field:
```cpp
enum class BufferUsage { Vertex, Index };
```
In `struct BufferDesc` add: `BufferUsage usage = BufferUsage::Vertex;`
- [ ] **Step 3:** In `struct GraphicsPipelineDesc` add:
```cpp
    bool depthTest = true;
    Format depthFormat = Format::D32_Float;
    uint32_t pushConstantSize = 0;   // bytes, vertex stage
```
- [ ] **Step 4:** In `class ICommandBuffer` add three methods (pure virtual):
```cpp
    virtual void BindIndexBuffer(IBuffer& buffer) = 0;
    virtual void DrawIndexed(uint32_t indexCount, uint32_t firstIndex = 0) = 0;
    virtual void PushConstants(const void* data, uint32_t size) = 0;
```
- [ ] **Step 5:** Build (it won't link the sample yet — VulkanCommandBuffer is now abstract until B3 implements the new methods; that's fine within this task as a compile check of the header). Commit after B3 builds. For an isolated header compile check, momentarily not required — proceed straight to B3, then build+commit together.

> Because adding pure-virtual methods makes `VulkanCommandBuffer` temporarily abstract, B2 and B3
> must land together to keep the tree compiling. Implement B2 then B3, build once, commit both
> (or commit B2+B3 as one "feat(rhi): depth, index buffers, push constants" commit).

---

## Task B3: Vulkan backend (depth, index, push constants) + cube shaders + sample

**Files:** modify `engine/rhi_vulkan/*` (device, swapchain, pipeline, buffer, command_buffer,
vk_common); create `shaders/cube.vert.hlsl`, `shaders/cube.frag.hlsl`; modify
`samples/hello_triangle/CMakeLists.txt` + `main.cpp` (extend the existing sample to draw the cube —
keep the target name `hello_triangle`).

### vk_common.h
- [ ] In `ToVk(Format)` add `case Format::D32_Float: return VK_FORMAT_D32_SFLOAT;`.

### Depth image (device-owned)
- [ ] The device creates a depth image + view sized to the swapchain extent, VMA `GPU_ONLY`
  (`VMA_MEMORY_USAGE_AUTO`, `VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT`, format
  `VK_FORMAT_D32_SFLOAT`, aspect `VK_IMAGE_ASPECT_DEPTH_BIT`). Simplest placement: give
  `VulkanSwapchain` an additional depth image it builds in `Build()` and destroys in `Destroy()`,
  exposing `VkImageView depthView()` and `VkFormat depthFormat()`. It recreates with the color
  images on resize. (Needs the VmaAllocator — pass it into the swapchain ctor.)
- [ ] BeginFrame: add a depth image layout transition `UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL`
  (stage `EARLY_FRAGMENT_TESTS|LATE_FRAGMENT_TESTS`, access `DEPTH_STENCIL_ATTACHMENT_WRITE`,
  aspect DEPTH). No transition back needed (depth isn't presented).

### Command buffer
- [ ] `Begin(...)` also receives the depth view. `BeginRenderPass` adds a
  `VkRenderingAttachmentInfo` for depth (`imageView=depthView`,
  `imageLayout=DEPTH_ATTACHMENT_OPTIMAL`, `loadOp=CLEAR`, `storeOp=DONT_CARE`,
  `clearValue.depthStencil={1.0f,0}`) and sets `VkRenderingInfo.pDepthAttachment`.
- [ ] Implement `BindIndexBuffer` (`vkCmdBindIndexBuffer(cmd, buf, 0, VK_INDEX_TYPE_UINT32)`),
  `DrawIndexed` (`vkCmdDrawIndexed(cmd, indexCount, 1, firstIndex, 0, 0)`), and `PushConstants`
  (`vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, size, data)`).
- [ ] `PushConstants` needs the current pipeline layout. Store the bound `VkPipelineLayout` in the
  command buffer when `BindPipeline` is called (have `VulkanPipeline` expose `VkPipelineLayout layout()`).

### Pipeline
- [ ] Add `VkPipelineDepthStencilStateCreateInfo` (`depthTestEnable`/`depthWriteEnable` from
  `desc.depthTest`, `depthCompareOp=VK_COMPARE_OP_LESS`) and wire into the create info.
- [ ] Set `VkPipelineRenderingCreateInfo.depthAttachmentFormat = ToVk(desc.depthFormat)`.
- [ ] If `desc.pushConstantSize > 0`, add a `VkPushConstantRange{VK_SHADER_STAGE_VERTEX_BIT, 0,
  desc.pushConstantSize}` to the pipeline layout create info.
- [ ] Enable backface culling: `rs.cullMode = VK_CULL_MODE_BACK_BIT;` (front face stays
  `COUNTER_CLOCKWISE`). Author cube indices with CCW outward winding. (If the cube renders
  inside-out at screenshot time, the fix is the winding or cull mode — note this risk.)

### Buffer
- [ ] In `VulkanBuffer`, set usage from `desc.usage`: vertex → `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT`,
  index → `VK_BUFFER_USAGE_INDEX_BUFFER_BIT`. Keep host-visible mapped upload as in Slice A.

### Shaders
- [ ] `shaders/cube.vert.hlsl`:
```hlsl
struct VSInput {
    [[vk::location(0)]] float3 pos   : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
};
struct VSOutput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};
[[vk::push_constant]] struct { float4x4 mvp; } pc;

VSOutput main(VSInput input) {
    VSOutput o;
    o.pos = mul(pc.mvp, float4(input.pos, 1.0));
    o.color = input.color;
    return o;
}
```
- [ ] `shaders/cube.frag.hlsl`:
```hlsl
struct PSInput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};
float4 main(PSInput input) : SV_Target { return float4(input.color, 1.0); }
```
- [ ] In `samples/hello_triangle/CMakeLists.txt`, change the `hf_compile_shaders` SHADERS list to
  compile `shaders/cube.vert.hlsl:vs` and `shaders/cube.frag.hlsl:ps` (replace the triangle
  shaders). Keep `HF_SHADER_DIR`.

### Sample (`samples/hello_triangle/main.cpp`)
- [ ] Replace the triangle geometry with a unit cube: 24 vertices (4 per face, so each face gets a
  flat color) OR 8 shared vertices with per-vertex color. Use **8 vertices + 36 indices** with
  distinct per-vertex colors (corner = RGB by position sign) so the 3D shape and depth are legible.
  Vertex = `{float pos[3]; float color[3];}`. Vertex layout: location 0 = `RGB32_Float` at offset 0,
  location 1 = `RGB32_Float` at offset 12, stride 24.
- [ ] Cube corners at ±0.5 on each axis; color = `(pos + 0.5)` per component so each corner is a
  distinct color. 12 triangles / 36 indices, CCW outward.
- [ ] Create the vertex buffer (`BufferUsage::Vertex`) and an index buffer (`BufferUsage::Index`).
- [ ] Pipeline desc: `depthTest=true`, `pushConstantSize=sizeof(float)*16`.
- [ ] Each frame: compute `model = RotateY(t) * RotateX(t*0.5)`, `view = LookAt({2,2,4},{0,0,0},{0,1,0})`,
  `proj = Perspective(60°, width/height, 0.1, 100)`, `mvp = proj * view * model`. Get `t` from
  `std::chrono::steady_clock` (seconds since start). Record:
  `BeginRenderPass(clear)`, `BindPipeline`, `PushConstants(&mvp, 64)`, `BindVertexBuffer`,
  `BindIndexBuffer`, `DrawIndexed(36)`, `EndRenderPass`.
- [ ] On resize, recreate the swapchain (already wired) — aspect recomputed each frame from window
  size, so perspective updates automatically. Update the window title to "Hazard Forge — Hello Cube".

- [ ] **Build, run `ctest` (rhi_smoke + math_test both pass).** Confirm `hello_triangle.exe`
  builds (do NOT run it). Commit: `feat(rhi+vulkan): depth buffer, index buffers, push-constant MVP, spinning cube`

---

## Definition of Done
- [ ] `ctest` passes both `rhi_smoke` and `math_test`.
- [ ] `hello_triangle.exe` builds clean, zero validation errors.
- [ ] Controller screenshot shows a solid, perspective, depth-correct spinning cube.
- [ ] No `vk*` symbols under `engine/rhi/` (grep).
