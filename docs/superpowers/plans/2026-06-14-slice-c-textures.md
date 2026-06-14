# Slice C: Textures — Implementation Plan

> Built via subagent-driven-development. Slice A/B toolchain proven; env facts below.

**Goal:** A textured spinning cube — procedural checkerboard sampled across faces, modulated by
vertex color. Exercises staging upload, samplers, and descriptor sets. Screenshot-verified,
zero validation errors.

**Builds on:** Slices A+B on `master` (RHI seam + Vulkan backend with depth/index/push-constants).
Additive.

---

## Environment (proven — trust)
- VS BuildTools x64 dev shell: `& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64`, then cmake in that session.
- Conan deps installed. `cmake --preset windows-msvc-debug` then `cmake --build --preset windows-msvc-debug`. Reconfigure after adding files/shaders. (Re-run conan only if a package is missing: `conan install . -of=build/windows-msvc-debug -s build_type=Debug -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing`.)
- DXC SPIR-V already wired (don't touch CompileShaders.cmake).
- Gates: `ctest` only (`rhi_smoke`, `math_test`). DO NOT run hello_triangle.exe (infinite loop) — only build it. Controller does visual verification.
- engine/CMakeLists.txt links sdl::sdl, vk-bootstrap::vk-bootstrap, GPUOpen::VulkanMemoryAllocator, vulkan-headers::vulkan-headers, Vulkan::Loader. Add new source files to the hf_engine list (e.g. rhi_vulkan/vulkan_texture.cpp).
- READ existing files first; match their style: engine/rhi/rhi.h, engine/rhi_vulkan/{vulkan_device.h/.cpp, vulkan_pipeline.h/.cpp, vulkan_command_buffer.h/.cpp, vulkan_buffer.cpp, vk_common.h}, samples/hello_triangle/main.cpp + CMakeLists.txt, tests/rhi_smoke.cpp.

---

## Task C1: RHI seam extensions

**File:** `engine/rhi/rhi.h` (additive; no vulkan symbols).

- [ ] Add texture descriptor + interface + device method + pipeline flag + command method:
```cpp
struct TextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    Format format = Format::RGBA8_UNorm;
    const void* data = nullptr;   // tightly packed pixels
    uint64_t dataSize = 0;        // bytes
};

class ITexture { public: virtual ~ITexture() = default; };
```
- In `GraphicsPipelineDesc` add: `bool usesTexture = false;`
- In `IRHIDevice` add: `virtual std::unique_ptr<ITexture> CreateTexture(const TextureDesc&) = 0;`
- In `ICommandBuffer` add: `virtual void BindTexture(ITexture& texture) = 0;`

(Adding pure-virtuals makes the Vulkan classes temporarily abstract — implement C2+C3 before
building. Land C1–C3 together.)

---

## Task C2: Vulkan device — sampler, descriptor layout/pool, staging upload, texture class

**Files:** create `engine/rhi_vulkan/vulkan_texture.h/.cpp`; modify `vulkan_device.h/.cpp`,
`engine/CMakeLists.txt`.

### Device additions (`vulkan_device.h/.cpp`)
- [ ] New members: `VkSampler defaultSampler_`, `VkDescriptorSetLayout texturedSetLayout_`,
  `VkDescriptorPool descriptorPool_`. Accessors: `VkSampler defaultSampler() const`,
  `VkDescriptorSetLayout texturedSetLayout() const`, `VkDescriptorPool descriptorPool() const`,
  plus `VkDevice device() const` (already exists) and `VmaAllocator allocator() const` (exists).
- [ ] In ctor (after device/VMA, before swapchain is fine):
  - **Sampler:** `vkCreateSampler` — `magFilter=minFilter=VK_FILTER_LINEAR`,
    `mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR`, address modes `VK_SAMPLER_ADDRESS_MODE_REPEAT`,
    `maxLod=VK_LOD_CLAMP_NONE`.
  - **Set layout:** one binding: `binding=0, descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    descriptorCount=1, stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT`.
  - **Pool:** `VkDescriptorPoolCreateInfo` with one pool size
    `{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64}`, `maxSets=64`,
    `flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT`.
- [ ] In dtor (after `vkDeviceWaitIdle`, before destroying device; order: textures must already be
  gone — they're owned by the app/sample, destroyed before the device): destroy pool, set layout,
  sampler. (These are device-level and destroyed in `~VulkanDevice` after the swapchain reset, before
  `vkb::destroy_device`.)

### Staging upload helper (`vulkan_device.cpp`, can be a private method or free helper)
- [ ] `void UploadToImage(VkImage image, uint32_t w, uint32_t h, const void* data, uint64_t size)`:
  1. Create host-visible staging buffer via VMA (`VK_BUFFER_USAGE_TRANSFER_SRC_BIT`,
     `HOST_ACCESS_SEQUENTIAL_WRITE | MAPPED`), memcpy `data`, flush.
  2. Allocate a one-time primary command buffer from a transient pool (reuse a transient pool
     created in ctor, or `frames_[0].pool` with reset — simplest: create a short-lived
     `VK_COMMAND_POOL_CREATE_TRANSIENT_BIT` pool here and destroy it after). Begin (ONE_TIME_SUBMIT).
  3. Barrier (sync2 `vkCmdPipelineBarrier2`): `UNDEFINED → TRANSFER_DST_OPTIMAL`, srcStage
     TOP_OF_PIPE/0, dstStage `COPY`/`TRANSFER`, dstAccess `TRANSFER_WRITE`, aspect COLOR.
  4. `vkCmdCopyBufferToImage` — `VkBufferImageCopy{ bufferOffset=0, imageSubresource={COLOR,0,0,1},
     imageExtent={w,h,1} }`, dstLayout TRANSFER_DST_OPTIMAL.
  5. Barrier: `TRANSFER_DST → SHADER_READ_ONLY_OPTIMAL`, srcStage TRANSFER/srcAccess TRANSFER_WRITE,
     dstStage `FRAGMENT_SHADER`/dstAccess `SHADER_READ`.
  6. End, `vkQueueSubmit2` (no semaphores, a fence OR `vkQueueWaitIdle`), `vkQueueWaitIdle(graphicsQueue_)`.
  7. Destroy staging buffer + the transient pool.

### `VulkanTexture` (new)
- [ ] `vulkan_texture.h`:
```cpp
#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {
class VulkanDevice;
class VulkanTexture final : public ITexture {
public:
    VulkanTexture(VulkanDevice& device, const TextureDesc& desc);
    ~VulkanTexture() override;
    VkDescriptorSet descriptorSet() const { return set_; }
private:
    VulkanDevice& device_;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation alloc_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
};
} // namespace
```
- [ ] `vulkan_texture.cpp`:
  - Create device-local `VkImage` via `vmaCreateImage`: `imageType=2D`, `format` from `ToVk(desc.format)`
    (RGBA8 → `VK_FORMAT_R8G8B8A8_UNORM`), extent {w,h,1}, mipLevels=1, arrayLayers=1,
    `usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT`, `tiling=OPTIMAL`,
    `initialLayout=UNDEFINED`, VMA `MEMORY_USAGE_AUTO` device-local.
  - `device_.UploadToImage(image_, w, h, desc.data, desc.dataSize)`.
  - Create `VkImageView` (2D, COLOR, 1 mip/layer).
  - Allocate a descriptor set from `device_.descriptorPool()` with `device_.texturedSetLayout()`.
  - Update it: `VkDescriptorImageInfo{ device_.defaultSampler(), view_,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }`, `VkWriteDescriptorSet` (binding 0,
    `COMBINED_IMAGE_SAMPLER`), `vkUpdateDescriptorSets`.
  - Dtor: destroy view, `vmaDestroyImage`, `vkFreeDescriptorSets(pool, 1, &set_)`.
- [ ] Device `CreateTexture` returns `std::make_unique<VulkanTexture>(*this, desc)`.

---

## Task C3: Pipeline + command buffer + shaders + textured cube sample

**Files:** modify `vulkan_pipeline.h/.cpp`, `vulkan_command_buffer.h/.cpp`; create
`shaders/cube_tex.vert.hlsl`, `shaders/cube_tex.frag.hlsl`; modify
`samples/hello_triangle/{CMakeLists.txt, main.cpp}`, `tests/rhi_smoke.cpp`.

### Pipeline
- [ ] `VulkanPipeline` construction needs the device (for `texturedSetLayout()`). Change the
  pipeline ctor / `CreateGraphicsPipeline` to pass `VulkanDevice&` (or the set layout). When
  `desc.usesTexture`, set `VkPipelineLayoutCreateInfo.setLayoutCount=1`,
  `pSetLayouts=&texturedSetLayout`. Push-constant range stays. When `!usesTexture`, no set layouts
  (Slice A/B path unaffected — but the cube now uses texture).

### Command buffer
- [ ] `BindTexture(ITexture& t)`: `auto& vt = static_cast<VulkanTexture&>(t); VkDescriptorSet s =
  vt.descriptorSet(); vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_,
  0, 1, &s, 0, nullptr);`. (Uses the stored bound pipeline layout — texture must be bound after
  BindPipeline; the sample does so.)

### Shaders
- [ ] `shaders/cube_tex.vert.hlsl`:
```hlsl
struct VSInput {
    [[vk::location(0)]] float3 pos   : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
    [[vk::location(2)]] float2 uv    : TEXCOORD0;
};
struct VSOutput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
    [[vk::location(1)]] float2 uv    : TEXCOORD0;
};
[[vk::push_constant]] struct { float4x4 mvp; } pc;
VSOutput main(VSInput i) {
    VSOutput o;
    o.pos = mul(pc.mvp, float4(i.pos, 1.0));
    o.color = i.color; o.uv = i.uv;
    return o;
}
```
- [ ] `shaders/cube_tex.frag.hlsl`:
```hlsl
[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(0, 0)]] SamplerState gSmp : register(s0);
struct PSInput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
    [[vk::location(1)]] float2 uv    : TEXCOORD0;
};
float4 main(PSInput i) : SV_Target {
    float4 t = gTex.Sample(gSmp, i.uv);
    return float4(t.rgb * i.color, 1.0);
}
```
> `[[vk::binding(0,0)]]` on BOTH the Texture2D and SamplerState makes DXC emit a single combined
> image sampler at set 0 binding 0, matching the `COMBINED_IMAGE_SAMPLER` descriptor. Verify the
> SPIR-V uses one combined binding; if DXC splits them, fall back to two bindings (image at
> binding 0, sampler at binding 1) and update the set layout to two bindings accordingly. Note
> whichever path you took.
- [ ] In samples CMakeLists, compile `cube_tex.vert.hlsl:vs` + `cube_tex.frag.hlsl:ps` (replace the
  cube.* shaders).

### Sample (`main.cpp`)
- [ ] Vertex struct now `{float pos[3]; float color[3]; float uv[2];}` (stride 32). Vertex layout
  adds location 2 = `RG32_Float` at offset 24.
- [ ] **24 vertices** (4 per face) with per-face UVs (0,0),(1,0),(1,1),(0,1) and a per-face tint
  color; **36 indices** (CCW outward, 2 tris per face).
- [ ] Generate a procedural texture before the loop: e.g. 256×256 RGBA8, an 8×8 checkerboard where
  tiles alternate between two colors and the cell color also varies with uv (so faces show a clear
  grid). Store in `std::vector<uint8_t>` (size 256*256*4). Create via
  `device->CreateTexture({256,256,Format::RGBA8_UNorm, pixels.data(), pixels.size()})`.
- [ ] Pipeline desc: `usesTexture=true`, `depthTest=true`, `pushConstantSize=64`.
- [ ] Per frame record: `BeginRenderPass`, `BindPipeline`, `PushConstants(&mvp,64)`,
  `BindTexture(*texture)`, `BindVertexBuffer`, `BindIndexBuffer`, `DrawIndexed(36)`, `EndRenderPass`.
- [ ] Window title "Hazard Forge — Textured Cube".

### Test (`tests/rhi_smoke.cpp`)
- [ ] After device creation, add: build a 2×2 RGBA8 array (16 bytes), `auto tex =
  device->CreateTexture({2,2,Format::RGBA8_UNorm, pixels, 16});` then let it go out of scope before
  `WaitIdle`. Exercises staging + descriptor alloc/free headlessly. Keep the existing checks.

- [ ] **Build, `ctest` (rhi_smoke + math_test pass), confirm hello_triangle.exe builds (don't run).**
  Commit (one combined commit fine): `feat(rhi+vulkan): textures — staging upload, sampler, descriptor sets, textured cube`

---

## Definition of Done
- [ ] `ctest` passes `rhi_smoke` (now with a texture) + `math_test`.
- [ ] `hello_triangle.exe` builds clean, zero validation errors.
- [ ] Controller screenshot shows the textured spinning cube (checkerboard visible on faces).
- [ ] No `vk*` under `engine/rhi/` (grep).

## Notes / risks
- DXC combined-sampler emission (the `[[vk::binding(0,0)]]` on both) is the main uncertainty —
  fallback to separate bindings documented above.
- Descriptor pool `FREE_DESCRIPTOR_SET_BIT` is required because VulkanTexture frees its set in dtor.
- Staging uses `vkQueueWaitIdle` (synchronous) — acceptable for load-time in this slice.
