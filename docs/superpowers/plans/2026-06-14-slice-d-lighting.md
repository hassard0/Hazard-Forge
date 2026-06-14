# Slice D: Real-Time Lighting — Implementation Plan

> Built via subagent-driven-development. Slice A/B/C toolchain proven; env facts below.

**Goal:** A Blinn-Phong-lit textured cube (directional + ambient + specular), via a per-frame
uniform buffer and frequency-based descriptor sets (set 0 = frame, set 1 = material). The shading
difference from Slice C must be obvious. Screenshot-verified, zero validation errors.

**Builds on:** Slices A+B+C on `master`. Additive, but **refactors descriptors**: the texture set
moves from set 0 → set 1; a new per-frame UBO occupies set 0.

---

## Environment (proven — trust)
- VS BuildTools x64 dev shell: `& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64`, then cmake in that session.
- Conan installed. `cmake --preset windows-msvc-debug` (reconfigure after adding files/shaders) then `cmake --build --preset windows-msvc-debug`.
- DXC SPIR-V wired; don't touch CompileShaders.cmake.
- Gates: `ctest` (rhi_smoke + math_test). DO NOT run hello_triangle.exe (infinite loop) — only build. Controller screenshots.
- READ existing code first and match style: engine/rhi/rhi.h, engine/rhi_vulkan/{vulkan_device.h/.cpp, vulkan_texture.cpp, vulkan_pipeline.h/.cpp, vulkan_command_buffer.h/.cpp, vulkan_buffer.cpp, vk_common.h}, samples/hello_triangle/{main.cpp, CMakeLists.txt}, tests/rhi_smoke.cpp. engine/math/math.h has Mat4/Vec3.
- kFramesInFlight is a constant in vulkan_device.h (=2). Per-frame sync lives in frames_[kFramesInFlight].

## Existing facts to integrate with
- Vulkan 1.3 dynamic rendering + sync2. VulkanDevice has device(), allocator(), graphicsQueue_, descriptorPool_ (from Slice C, FREE_DESCRIPTOR_SET_BIT, maxSets=64), a default sampler, and the Slice C material set layout (currently used at set 0 — you will move it to set 1). The command buffer stores boundLayout_ on BindPipeline and has BindTexture (currently firstSet=0 — change to 1). VulkanPipeline takes VulkanDevice& and conditionally adds the material set layout when usesTexture.
- HARD RULE: no vulkan symbols in engine/rhi/. C-D additions (BufferUsage::Uniform, usesFrameUniforms, SetFrameUniforms, BindTexture@set1) stay backend-agnostic.
- Adding the pure-virtual SetFrameUniforms makes VulkanDevice temporarily abstract — implement all tasks together, build once, commit together.

---

## Task D1: RHI seam extensions
**File:** engine/rhi/rhi.h (additive).
- [ ] `enum class BufferUsage { Vertex, Index, Uniform };`
- [ ] `GraphicsPipelineDesc`: add `bool usesFrameUniforms = false;`
- [ ] `IRHIDevice`: add `virtual void SetFrameUniforms(const void* data, uint32_t size) = 0;`
- [ ] (No new ICommandBuffer method — BindTexture's set index changes internally.)

---

## Task D2: Vulkan device — per-frame UBOs, frame descriptor set layout (set 0), pool sizing
**Files:** vulkan_device.h/.cpp.

- [ ] New device members:
  - `VkDescriptorSetLayout frameSetLayout_` — set 0, binding 0 = `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER`, count 1, stage `VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT`.
  - `VkBuffer uboBuffer_[kFramesInFlight]`, `VmaAllocation uboAlloc_[kFramesInFlight]`, `void* uboMapped_[kFramesInFlight]`.
  - `VkDescriptorSet frameSet_[kFramesInFlight]`.
  - Accessor: `VkDescriptorSetLayout frameSetLayout() const`, `VkDescriptorSet currentFrameSet() const { return frameSet_[frameIndex_]; }`.
- [ ] Rename the Slice C material set layout accessor if needed; it now logically lives at **set 1** (the layout object itself is unchanged — set index is decided in the pipeline layout array). Keep `texturedSetLayout()` / `materialSetLayout()` name as-is.
- [ ] Pool sizing: add a `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` pool size (count ≥ kFramesInFlight + a margin, e.g. 16). Keep existing SAMPLED_IMAGE + SAMPLER sizes. Bump maxSets if needed (e.g. 64 stays fine).
- [ ] In ctor (after pool/sampler/material layout exist): create frameSetLayout_; for each frame i: create a host-visible **mapped** uniform buffer sized to a fixed `kFrameUboSize` (use 256 bytes — safely covers the 112-byte FrameData and respects typical minUniformBufferOffsetAlignment; store the size as a constant), keep its mapped pointer (`VmaAllocationInfo.pMappedData`), allocate frameSet_[i] from the pool with frameSetLayout_, and `vkUpdateDescriptorSets` it to point at uboBuffer_[i] (`VkDescriptorBufferInfo{uboBuffer_[i], 0, VK_WHOLE_SIZE}`, type UNIFORM_BUFFER, binding 0).
- [ ] `SetFrameUniforms(const void* data, uint32_t size)` impl: `std::memcpy(uboMapped_[frameIndex_], data, size);` (host-coherent mapped; assert size <= kFrameUboSize). No flush needed if the allocation is HOST_COHERENT — to be safe, request coherent or call vmaFlushAllocation and Check it.
- [ ] Dtor: destroy frameSetLayout_, and for each frame `vmaDestroyBuffer(uboBuffer_[i], uboAlloc_[i])` (frame sets freed with pool or explicitly). Order: in the existing texture-resources teardown block (after swapchain reset, before vmaDestroyAllocator). Sets allocated from descriptorPool_ — either free them before destroying the pool or just destroy the pool (pool destruction frees all sets). Simplest: destroy buffers + layout, then the existing pool destroy frees the sets.

---

## Task D3: Pipeline + command buffer (set indices) + buffer usage
**Files:** vulkan_pipeline.h/.cpp, vulkan_command_buffer.h/.cpp, vulkan_buffer.cpp.

- [ ] **Pipeline layout set array.** Build a `std::vector<VkDescriptorSetLayout> setLayouts;`
  - if `desc.usesFrameUniforms` → `setLayouts.push_back(device.frameSetLayout());`  (this is set 0)
  - if `desc.usesTexture` → `setLayouts.push_back(device.materialSetLayout());`     (this is set 1)
  Set `layoutInfo.setLayoutCount = setLayouts.size(); pSetLayouts = setLayouts.data();`. Push-constant range (model matrix) unchanged. Carry `bool hasFrameSet_ = desc.usesFrameUniforms;` on the VulkanPipeline; expose `bool hasFrameSet() const`.
  > IMPORTANT: the order pushed determines set index. frame first (set 0), material second (set 1). The shader must declare UBO at set 0 and texture/sampler at set 1 to match.
- [ ] **Command buffer BindPipeline:** after binding the pipeline and storing boundLayout_, if `pipeline.hasFrameSet()`, bind the device's current frame set at set 0:
  `VkDescriptorSet fs = device_.currentFrameSet(); vkCmdBindDescriptorSets(cmd_, GRAPHICS, boundLayout_, 0, 1, &fs, 0, nullptr);`
  (The command buffer already has access to the device or can get currentFrameSet — VulkanCommandBuffer holds VulkanDevice&. Verify it does; if not, it already does for other needs — check.)
- [ ] **Command buffer BindTexture:** change `firstSet` from 0 to **1**:
  `vkCmdBindDescriptorSets(cmd_, GRAPHICS, boundLayout_, 1, 1, &set, 0, nullptr);`
- [ ] **Buffer:** add `case BufferUsage::Uniform: usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;` (the device builds its UBOs directly via VMA, but keep the enum mapping complete/correct).

---

## Task D4: Shaders + lit cube sample + test
**Files:** create shaders/lit.vert.hlsl, shaders/lit.frag.hlsl; modify samples/hello_triangle/{main.cpp, CMakeLists.txt}, tests/rhi_smoke.cpp.

- [ ] `shaders/lit.vert.hlsl`:
```hlsl
struct VSInput {
    [[vk::location(0)]] float3 pos    : POSITION;
    [[vk::location(1)]] float3 color  : COLOR;
    [[vk::location(2)]] float2 uv     : TEXCOORD0;
    [[vk::location(3)]] float3 normal : NORMAL;
};
struct VSOutput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
};
struct FrameData { float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos; };
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; } pc;

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(pc.model, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(f.viewProj, world);
    o.wnormal = normalize(mul((float3x3)pc.model, i.normal));
    o.color = i.color; o.uv = i.uv;
    return o;
}
```
- [ ] `shaders/lit.frag.hlsl`:
```hlsl
struct FrameData { float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos; };
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
};
float4 main(PSInput i) : SV_Target {
    float3 N = normalize(i.wnormal);
    float3 L = normalize(-f.lightDir.xyz);
    float  diff = max(dot(N, L), 0.0);
    float3 V = normalize(f.viewPos.xyz - i.wpos);
    float3 H = normalize(L + V);
    float  spec = pow(max(dot(N, H), 0.0), 32.0);
    float  ambient = 0.15;
    float3 tex = gTex.Sample(gSmp, i.uv).rgb * i.color;
    float3 rgb = tex * ((ambient + diff) * f.lightColor.rgb) + spec * f.lightColor.rgb * 0.4;
    return float4(rgb, 1.0);
}
```
> Note bindings: UBO at set 0 binding 0 (cbuffer), texture at set 1 binding 0, sampler at set 1
> binding 1 — matching frameSetLayout (set0) + materialSetLayout (set1). Verify DXC emits these
> set/binding numbers (disassemble if unsure). Material stays SPLIT image+sampler (as Slice C).
- [ ] samples CMakeLists: compile `lit.vert.hlsl:vs` + `lit.frag.hlsl:ps` (replace cube_tex.*).
- [ ] `main.cpp`:
  - Vertex `{float pos[3]; float color[3]; float uv[2]; float normal[3];}` stride 44; layout adds
    location 3 = `RGB32_Float` @ offsetof(normal). 24 verts (4/face) with correct per-face outward
    normal; 36 indices CCW.
  - Procedural texture as in Slice C (keep it).
  - `struct FrameData { float vp[16]; float lightDir[4]; float lightColor[4]; float viewPos[4]; };`
    (matches the shader; 112 bytes).
  - Pipeline desc: `usesFrameUniforms=true, usesTexture=true, depthTest=true, pushConstantSize=64`.
  - Per frame: `view=LookAt(eye,{0,0,0},{0,1,0})` with `eye={2.5f,2.5f,4.0f}`; `proj=Perspective(60°,
    w/h,0.1,100)`; `vp = proj*view`. Fill FrameData (vp = vp.m, lightDir={-0.5,-1,-0.3,0},
    lightColor={1,1,1,1}, viewPos={eye,1}). `device->SetFrameUniforms(&frameData, sizeof(FrameData));`
    Then `model = RotateY(t)*RotateX(t*0.5)`; record BeginRenderPass, BindPipeline,
    PushConstants(&model,64), BindTexture(*tex), BindVertexBuffer, BindIndexBuffer, DrawIndexed(36),
    EndRenderPass. (BindPipeline auto-binds frame set 0; SetFrameUniforms must be called before draw —
    call it once per frame before BeginFrame's recording, after BeginFrame so frameIndex_ is current:
    ORDER = BeginFrame → SetFrameUniforms → record. Confirm BeginFrame advances/sets frameIndex_ such
    that currentFrameSet matches the buffer SetFrameUniforms writes. If BeginFrame uses frameIndex_
    for the current frame and EndFrame advances it, then SetFrameUniforms after BeginFrame writes the
    correct buffer. Verify this ordering in the existing device code.)
  - Title "Hazard Forge — Lit Cube".
- [ ] `tests/rhi_smoke.cpp`: after device + texture creation, call
  `float dummy[28] = {0}; device->SetFrameUniforms(dummy, sizeof(dummy));` (112 bytes) to exercise
  the UBO copy path. Keep existing checks.

- [ ] **Build, `ctest` (both pass), confirm hello_triangle.exe builds (don't run).** Commit:
  `feat(rhi+vulkan): Blinn-Phong lighting via per-frame UBO + frequency-based descriptor sets`

---

## Definition of Done
- [ ] ctest passes rhi_smoke + math_test.
- [ ] hello_triangle.exe builds clean, zero validation errors.
- [ ] Controller screenshot shows clear directional shading (lit/dark faces) + specular — obviously
      different from Slice C's flat texture.
- [ ] No `vk*` under engine/rhi/ (grep).

## Risks
- **Set-index / frameIndex_ timing** is the main risk: BindPipeline must bind the SAME frame set
  whose UBO SetFrameUniforms just wrote (current frameIndex_). Verify BeginFrame/EndFrame frame-index
  semantics and call SetFrameUniforms after BeginFrame.
- DXC set/binding emission for the cbuffer at (set0,binding0) + split sampler at set1 — disassemble
  if the descriptors don't line up; report what you confirmed.
- UBO alignment: kFrameUboSize=256 is safe; FrameData is 112 and host-coherent mapped.
