# Slice G: Offscreen Render Targets + Post-Processing — Plan

> subagent-driven. Toolchain proven. Env facts below.

**Goal:** Render the scene to an offscreen color texture, then a fullscreen post pass (tonemap +
vignette + gamma) samples it → swapchain. Verifiable via `--shot` (visible vignette).

**Builds on:** Slices A–F on `master`.

---

## Environment (proven — trust)
- VS BuildTools x64 dev shell: `& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64`, then cmake.
- `cmake --preset windows-msvc-debug` (reconfigure after adding shaders/files) then `cmake --build --preset windows-msvc-debug`. Add new engine/rhi_vulkan/vulkan_render_target.cpp to engine/CMakeLists.txt.
- `hello_triangle.exe --shot <path>.bmp` renders one frame + EXITS (you MAY run it). Never the no-arg loop.
- Gates: ctest (rhi_smoke + math_test). READ first: engine/rhi/rhi.h, engine/rhi_vulkan/{vulkan_device.h/.cpp, vulkan_texture.h/.cpp, vulkan_pipeline.cpp, vulkan_command_buffer.cpp, vulkan_swapchain.cpp, vk_common.h}, samples/hello_triangle/main.cpp, samples/hello_triangle/CMakeLists.txt.
- HARD RULE: no vulkan symbols in engine/rhi/ or engine/scene/.

## Existing facts
- Vulkan 1.3 dynamic rendering + sync2. Device has descriptorPool_, defaultSampler_, materialSetLayout (set1: b0 SAMPLED_IMAGE, b1 SAMPLER), frameSetLayout (set0 UBO). VulkanTexture shows the pattern for an image+view+descriptor-set (allocate from pool, update with view+sampler). The swapchain format is BGRA8 (vk_common FromVk maps it). TransitionImage sync2 helper exists in vulkan_device.cpp. The per-frame command recorder is recorder_ (VulkanCommandBuffer); frames_[frameIndex_] holds per-frame sync (pool, cmd, fences, sems).
- VulkanPipeline takes VulkanDevice& and builds vertex input from desc.vertexLayout, depth state from desc.depthTest, set layouts from usesFrameUniforms/usesTexture, push range from pushConstantSize.

## Task G1: RHI seam
engine/rhi/rhi.h, additive:
```cpp
class IRenderTarget : public ITexture {  // sampleable offscreen color (+depth) you render into
public:
    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;
};
```
- GraphicsPipelineDesc: add `bool fullscreen = false;`
- IRHIDevice add:
```cpp
    virtual std::unique_ptr<IRenderTarget> CreateRenderTarget(uint32_t w, uint32_t h) = 0;
    virtual FrameContext BeginRenderTargetFrame(IRenderTarget& rt) = 0;
    virtual void EndRenderTargetFrame(const FrameContext&) = 0;
```

## Task G2: VulkanRenderTarget + device methods
Create engine/rhi_vulkan/vulkan_render_target.{h,cpp}. Modify vulkan_device.{h,cpp}, engine/CMakeLists.txt.
- [ ] VulkanRenderTarget(VulkanDevice&, w, h): create
  - color VkImage (format = swapchain format `swapchain_->vkFormat()` — add an accessor if needed, OR pass VK_FORMAT_B8G8R8A8_UNORM; usage COLOR_ATTACHMENT|SAMPLED), color view.
  - depth VkImage (D32, DEPTH_STENCIL_ATTACHMENT), depth view (aspect DEPTH).
  - a descriptor set from device pool with materialSetLayout, updated with {color view (SHADER_READ_ONLY layout in the write), default sampler} — same two-write pattern as VulkanTexture (binding0 SAMPLED_IMAGE, binding1 SAMPLER).
  - expose: colorImage(), colorView(), depthView(), descriptorSet() (so BindTexture works — IRenderTarget IS an ITexture; make descriptorSet() the method VulkanCommandBuffer::BindTexture uses; since BindTexture takes ITexture&, ensure the static_cast path works — simplest: VulkanRenderTarget also stores its set and VulkanCommandBuffer::BindTexture handles both VulkanTexture and VulkanRenderTarget. EASIEST: give both a common way to get the set. Make BindTexture static_cast to a small shared base, OR add a virtual VkDescriptorSet to a backend-internal interface. CLEANEST: add `VkDescriptorSet descriptorSet() const` to BOTH and in BindTexture, dynamic_cast or check type. SIMPLEST ROBUST: define an internal abstract `class ISampledVk { public: virtual VkDescriptorSet vkDescriptorSet() const = 0; };` that both VulkanTexture and VulkanRenderTarget implement, and BindTexture does `dynamic_cast<ISampledVk*>(&texture)`.) Implement the ISampledVk approach.
  - track currentLayout_ for the color image.
  - dtor: destroy views, vmaDestroyImage x2, free descriptor set.
- [ ] Device CreateRenderTarget → make_unique<VulkanRenderTarget>(*this, w, h).
- [ ] Device needs a SECOND command recorder + per-frame-ish sync for the RT pass, OR reuse a transient one-time submit. SIMPLEST CORRECT: BeginRenderTargetFrame uses a dedicated VkCommandBuffer from a transient/dedicated pool (or frames_[frameIndex_] is busy — use a separate rtPool_ + rtCmd_ created in ctor). Steps:
  - BeginRenderTargetFrame(rt): reset rtCmd_, begin; transition rt color UNDEFINED→COLOR_ATTACHMENT, rt depth UNDEFINED→DEPTH_ATTACHMENT; recorder2_->Begin(rtCmd_, rt.colorView(), rt.depthView(), {rt.width(),rt.height()}); return FrameContext{recorder2_.get()}. (Add a second recorder_ instance `rtRecorder_`.)
  - EndRenderTargetFrame: end rendering (recorder’s EndRenderPass already called by sample), transition rt color COLOR_ATTACHMENT→SHADER_READ_ONLY_OPTIMAL, vkEndCommandBuffer, submit (no semaphores; signal a dedicated rtFence_), vkWaitForFences(rtFence_) [or vkQueueWaitIdle]. Update rt.currentLayout_.
  > Use a dedicated rtCmd_/rtPool_/rtFence_ created in the device ctor and destroyed in dtor.
- [ ] CreateRenderTarget color format: add `VkFormat VulkanSwapchain::vkFormat() const` accessor if not present (it exists — check) so RT matches swapchain.

## Task G3: Pipeline fullscreen support + command buffer
- [ ] VulkanPipeline: when desc.fullscreen, set vertex input state to 0 bindings / 0 attributes (ignore desc.vertexLayout). Everything else (depth off recommended for post: the sample sets depthTest=false for the post pipeline), color format = swapchain format. Keep usesTexture → material set at set 0 (since usesFrameUniforms=false, material set is the only set → index 0; the existing materialSetIndex() returns 0 when !hasFrameSet — good, BindTexture will bind at 0).
- [ ] VulkanCommandBuffer::BindTexture: change to resolve the descriptor set via the ISampledVk interface (dynamic_cast) so it works for both VulkanTexture and VulkanRenderTarget. Keep using boundMaterialSet_.

## Task G4: Shaders + sample wiring
- [ ] shaders/post.vert.hlsl (fullscreen triangle, no inputs):
```hlsl
struct VSOutput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
VSOutput main(uint vid : SV_VertexID) {
    VSOutput o;
    float2 p = float2((vid << 1) & 2, vid & 2);   // (0,0),(2,0),(0,2)
    o.uv = p;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);       // covers clip space with a big triangle
    return o;
}
```
- [ ] shaders/post.frag.hlsl (set 0 because no frame UBO; but to match materialSetLayout used at set 0 when no frame set — IMPORTANT: declare at set 0):
```hlsl
[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);
struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
float4 main(PSInput i) : SV_Target {
    float3 c = gTex.Sample(gSmp, i.uv).rgb;
    c = c / (c + 1.0);                         // Reinhard tonemap
    c = pow(c, 1.0 / 2.2);                     // gamma
    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));  // darken corners
    return float4(c * vig, 1.0);
}
```
> The post pipeline has usesFrameUniforms=false, usesTexture=true → material set is the ONLY set → set index 0. So the post shader binds at set 0. Confirm materialSetIndex() returns 0 here and BindTexture binds at 0. (The lit pipeline keeps frame set at 0, material at 1 — unaffected.)
- [ ] samples CMakeLists: also compile post.vert.hlsl:vs + post.frag.hlsl:ps (keep lit.* too).
- [ ] main.cpp:
  - Load lit + post shaders; create the lit pipeline (as now) and a NEW post pipeline
    (`fullscreen=true, usesTexture=true, usesFrameUniforms=false, depthTest=false`, colorFormat=swapchain format, empty vertexLayout).
  - Create `auto rt = device->CreateRenderTarget(W,H)` at the swapchain size (use the window framebuffer size; for --shot the swapchain is 1280x720 — query device swapchain extent or window size). On resize (interactive), recreate rt to match.
  - Frame: `auto rtc = device->BeginRenderTargetFrame(*rt); rtc.cmd->BeginRenderPass(clear); <bind lit pipeline + draw scene renderables>; rtc.cmd->EndRenderPass(); device->EndRenderTargetFrame(rtc);`
    then swapchain pass: `device->CaptureNextFrame()` (only in --shot), `auto fc=device->BeginFrame(); if(fc.cmd){ fc.cmd->BeginRenderPass(clear); fc.cmd->BindPipeline(postPipeline); fc.cmd->BindTexture(*rt); fc.cmd->Draw(3); fc.cmd->EndRenderPass(); } device->EndFrame(fc);`
  - Keep --shot capture + WriteBMP. Title "Hazard Forge — Post".
- [ ] rhi_smoke: add `auto rt = device->CreateRenderTarget(64,64);` then let it drop, before WaitIdle.

## Verify
- Build clean. ctest (rhi_smoke + math_test) pass.
- Run `--shot "$env:TEMP\hf_g.bmp"` → exit 0, BMP ~3.6MB. Paste stdout + size.
- grep: no vulkan in engine/rhi/ or engine/scene/.

## Commit (footer Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>)
`feat(rhi+vulkan): offscreen render targets + fullscreen post-processing (tonemap+vignette)`

## DoD
- `--shot` BMP shows the scene with a visible vignette (darkened corners) + tonemapped colors → proves the offscreen RT + post pass path. ctest green. seam clean.

## Risks
- RT/swapchain sync: EndRenderTargetFrame must finish (fence/waitidle) before the swapchain pass samples the RT. Use rtFence_ + wait.
- The post pipeline color format must equal the swapchain format. RT color format must equal swapchain format too (so the lit pipeline renders into it without a format mismatch).
- BindTexture must resolve descriptor sets for BOTH VulkanTexture and VulkanRenderTarget (ISampledVk dynamic_cast).
