# Hazard Forge — Architecture

## The RHI Seam

The central architectural decision in Hazard Forge is the **Rendering Hardware Interface seam** — a layer of pure C++ abstract interfaces in `engine/rhi/rhi.h` that separates all engine and application code from any GPU backend.

### The hard rule

> No `vk*` type, no `MTL*` type, and no backend header may appear in `engine/rhi/`, `engine/scene/`, `engine/math/`, or `engine/hal/`. Backend symbols are confined to `engine/rhi_vulkan/` and `engine/rhi_metal/` respectively.

The rule is checkable with a single `grep`: if `vk` or `MTL` appears above the backend directories, the seam is broken. This is enforced by convention in every slice plan.

### The seam surface (`engine/rhi/rhi.h`)

All resource types are opaque base classes:

```
IRHIDevice        — device factory; owns swapchain, frame loop, descriptor infrastructure
ISwapchain        — surface + images; resize
ICommandBuffer    — per-frame draw recording (BeginRenderPass, BindPipeline, Draw, ...)
IPipeline         — compiled graphics PSO from GraphicsPipelineDesc
IBuffer           — vertex / index / uniform buffer
ITexture          — device-local image + sampler descriptor (base for render targets too)
IRenderTarget     — sampleable offscreen color (+depth) image; inherits ITexture
```

`GraphicsPipelineDesc` is a small, flat descriptor that encodes what the pipeline needs to know about shaders, vertex layout, depth, push constants, descriptor set membership, and a set of mode flags that has grown additively slice by slice — `fullscreen`, `depthOnly`, `pbrMaterial`, `usesEnvironment`, `usesJointPalette`, `depthWrite`, `alphaBlend`, `cullNone`, `lineList`, `fragmentPushConstants` — all in terms of engine-defined enums and types, no backend types. Every new flag **defaults to the value that leaves all pre-existing pipelines byte-for-byte unchanged**, which is what keeps the committed goldens stable as features land.

There is also a compute path (`IComputePipeline` + `Dispatch`) used by the GPU particle system, on the same seam.

Backends are handed out through the factory:

```cpp
// Vulkan (Windows):
auto device = hf::rhi::CreateDevice(hf::rhi::Backend::Vulkan, window);

// Metal headless (macOS, no window):
auto device = hf::rhi::mtl::CreateMetalDeviceHeadless(width, height);
```

Both return `std::unique_ptr<IRHIDevice>`. The caller never sees a backend type again.

---

## Descriptor / Binding Model

Hazard Forge uses a **frequency-based descriptor set layout** adopted in Slice D (lighting):

| Set | Update frequency | Contents |
|-----|-----------------|----------|
| 0 | Per frame | UBO: `viewProj`, directional light, view position, point lights, `lightViewProj`, camera basis for sky; shadow map depth image (binding 1) + shadow sampler (binding 2) |
| 1 | Per material / per draw | Material textures + samplers. The base set is color + normal; the **PBR** path (`BindMaterialPBR`) widens it to the full 5-texture metallic-roughness set (base / metal-rough / normal / emissive / occlusion). The **environment** (HDR IBL) binds on a dedicated slot via `BindEnvironment` so the base material layouts are unchanged. |
| 2 | Per skinned draw | Joint palette (skinning), bound via `SetJointPalette` at a dedicated vertex binding. |

GPU instancing adds a **second per-instance vertex stream** (binding 1, step-per-instance) rather than a descriptor set — four `RGBA32_Float` attributes carry a column-major `float4x4` per instance.

**Set 0** is owned by the device. The device maintains one UBO per frame-in-flight (double-buffered), each with a pre-baked descriptor set that already points at its buffer. `SetFrameUniforms(data, size)` memcpys into the current frame's UBO; `BindPipeline` auto-binds set 0 when the pipeline has `usesFrameUniforms = true`. The CPU never writes a UBO that the GPU is still reading.

**Set 1** is owned per texture / render target. Each `VulkanTexture` and each `VulkanRenderTarget` allocates one descriptor set from a device pool at creation time and updates it with its image view and the device's default sampler. `BindTexture(ITexture&)` binds that set.

**Push constants** carry the per-object model matrix (64 bytes, `mat4`). `PushConstants(data, size)` records a push to the vertex stage.

When `usesFrameUniforms = false` (the post-processing pipeline), the material set is the only set and occupies set index 0. The `BindTexture` path accounts for this shift.

---

## Per-frame UBO growth

The `FrameData` struct (352 bytes at the time of the skybox slice; `kFrameUboSize` is 512) has grown additively across slices:

- Slice D: `viewProj`, `lightDir`, `lightColor`, `viewPos` — basic per-frame camera + directional light
- Slice H: `ptCount`, `ptPos[3]`, `ptColor[3]` — colored point lights
- Slice I: `lightViewProj` — directional shadow map light-space matrix
- Slice J: `camFwd`, `camRight`, `camUp`, `skyParams` — camera basis for sky ray reconstruction

The `kFrameUboSize` constant is bumped whenever the struct approaches the current allocation. The C++ `FrameData` struct (mirrored byte-for-byte in `metal_headless/visual_test.mm`) and the HLSL `cbuffer FrameData` in every shader that reads it must be kept in sync by hand — the Metal struct layout is asserted to match the Vulkan sample's. Pass-specific parameters that are NOT per-frame (bloom thresholds, SSAO kernel params, per-object material factors, glass tint/alpha) travel via **push constants**, not the UBO — the vertex push constant carries `mat4 model + float4 material`, and the bloom/SSAO fullscreen passes use the `fragmentPushConstants` flag to read their params in the fragment stage.

---

## Multi-Pass Frame Structure

Frames are described by `render::RenderGraph`: passes declare imported targets (shadow map / scene color / swapchain) as reads/writes and are executed in declaration order through the RHI. The baseline lit frame is three passes; feature showcases add more passes to the same graph (a g-buffer prepass + AO + blur + composite for SSAO; a threshold + 5× down + up + composite chain for bloom into an `RGBA16_Float` target; a sorted alpha-blended pass after the opaque pass for transparency; a debug-line draw after opaque geometry). The capstone scene composes **seven distinct opaque pipelines plus a transparent pipeline** into one scene pass.

The baseline three-pass frame:

```
Pass 0: Shadow pass (depth-only)
  BeginShadowPass(*shadowMap)
    → depth-only command buffer, 2048² depth image, no color attachment
    → drawDepthOnly: shadow pipeline (depthOnly=true), PushConstants(model), DrawIndexed
  EndShadowPass → submit + fence-wait → shadow map transitions to SHADER_READ_ONLY

Pass 1: Scene → offscreen render target
  BeginRenderTargetFrame(*rt)
    → command buffer targeting the RT's color+depth
    → sky pipeline (fullscreen, depthTest=false) → Draw(3)   [background fill]
    → lit pipeline: per-renderable [PushConstants, BindTexture, BindVB/IB, DrawIndexed]
  EndRenderTargetFrame → submit + fence-wait → RT color transitions to SHADER_READ_ONLY

Pass 2: Post pass → swapchain
  BeginFrame → acquires swapchain image
    → post pipeline (fullscreen, usesTexture=true, usesFrameUniforms=false)
    → BindTexture(*rt) → Draw(3)
  EndFrame → submit + present (or capture + BMP in --shot mode)
```

Passes 0 and 1 use dedicated command buffers (`rtCmd_`, `rtPool_`, `rtFence_`) to avoid contention with the per-frame swapchain recorder. Both are submitted synchronously (fence-wait) before the swapchain pass begins, so there are no semaphore dependencies to track.

---

## Headless Capture and Golden-Image Testing

Headless capture is a first-class RHI feature, not a test harness hack.

`IRHIDevice::CaptureNextFrame()` arms a flag. On the next `EndFrame`, the Vulkan backend transitions the just-rendered swapchain image to `TRANSFER_SRC_OPTIMAL`, copies it to a host-visible staging buffer via `vkCmdCopyImageToBuffer`, waits idle, maps the buffer, and stores the BGRA8 pixels. `GetCapturedPixels(outBGRA, w, h)` hands them out. Present is skipped in capture mode — the window need not be visible.

The Metal headless target (`metal_headless/visual_test`) uses the same `CaptureNextFrame()` / `GetCapturedPixels()` path, but the "swapchain" is an offscreen `MTLTexture` (no `CAMetalLayer`, no window server). The same `IRHIDevice` / `ICommandBuffer` calls that the Vulkan sample makes drive the Metal backend through the exact same code path.

There are now **15 committed Metal reference renders** under `tests/golden/metal/` (M4, deterministic to byte level — two runs diff `0.0000`), one per `visual_test` showcase flag: `scene_shadow` (default), `skinning`, `pbr_helmet`, `instanced`, `ibl_helmet`, `physics`, `transparency`, `bloom`, `scene_import`, `debug_viz`, `anim_blend`, `ssao`, `capstone`, `camera_pose`, and `gizmo`. `scripts/verify.ps1` (and the self-hosted CI Metal job) build `metal_headless` once and golden-compare **all 15** at threshold `0.0` on every run, so any unintended change to shared shader/render/loader code is caught as a non-zero DIFF on a specific golden.

### Shaders are generated, not hand-written

The Metal shaders are **generated from the shared HLSL sources at build time** — there is no hand-written MSL to drift from the canonical shaders. For each shader the `metal_headless` build runs HLSL → SPIR-V (`glslc -x hlsl`) → MSL (`spirv-cross --msl --msl-decoration-binding`), emitting `*.gen.metal`, which `visual_test` compiles at runtime via `newLibraryWithSource:`. The `--msl-decoration-binding` flag maps each resource's SPIR-V binding directly to its Metal `[[buffer/texture/sampler(n)]]` index; the engine's Metal binding constants (`engine/rhi_metal/metal_common.h`) are chosen to match. The only Metal-specific shader adjustments are guarded by `#ifdef HF_MSL_GEN` (two texture-origin V-flips), so the Vulkan SPIR-V is byte-identical.

---

## Vulkan Backend Notes

- **Vulkan version:** 1.3 dynamic rendering (`VK_KHR_dynamic_rendering`); no `VkRenderPass` / `VkFramebuffer` objects.
- **Synchronization:** Vulkan Synchronization 2 (`vkCmdPipelineBarrier2` / `VkImageMemoryBarrier2`). A `TransitionImage` helper in `vulkan_device.cpp` centralizes the barrier pattern.
- **Memory:** Vulkan Memory Allocator (VMA 3.3) for all images and buffers. Staging uploads use a host-visible VMA buffer, `vkQueueWaitIdle` after the copy (synchronous load-time path).
- **Frames in flight:** `kFramesInFlight = 2` (double-buffered). Per-frame fences + semaphores in `frames_[frameIndex_]`. Per-frame UBOs indexed by `frameIndex_`.
- **Shader pipeline:** HLSL → SPIR-V via DXC (`-spirv`, invoked by a CMake custom command at build time). Shaders are loaded as `.spv` files at runtime via `IShaderModule`.
- **Depth:** swapchain-sized `VK_FORMAT_D32_SFLOAT` image owned by the device, recreated on resize.

## Metal Backend Notes

- **MSL via the shared toolchain.** `CreateShaderModule(ShaderModuleDesc&)` (SPIR-V) throws on the Metal backend; instead MSL **generated from the shared HLSL** (see "Shaders are generated" above) is compiled at runtime via `MTLDevice newLibraryWithSource:` through `MakeShaderModuleFromMSL`. This closes the earlier seam deviation — there is no hand-written MSL.
- **NDC Y-flip (CPU-side).** `hf::math::Perspective` / `Ortho` bake the Vulkan clip-space Y-flip. Metal NDC is +Y up, so `visual_test` flips the projection's and ortho's Y row on the **CPU** before composing view-proj / lightViewProj, rather than flipping in the shader (which would diverge from the shared HLSL). The shared math and Vulkan backend are untouched; only two `#ifdef HF_MSL_GEN`-guarded texture-origin V-flips remain shader-side.
- **No explicit barriers.** Metal's command encoder model uses implicit hazard tracking; there are no `MTLBlitCommandEncoder` barriers equivalent to Vulkan image layout transitions. The Metal backend omits them.
- **Frame pacing.** A `dispatch_semaphore_t inFlight_` (initial count = `kFramesInFlight`) blocks `BeginFrame` until the GPU has finished with the oldest in-flight frame's resources, replacing Vulkan's per-frame fence approach.
- **Full feature parity on Metal, golden-tested.** Offscreen render targets (including `RGBA16_Float` HDR targets for bloom/SSAO), directional shadow mapping (static / skinned / instanced depth-only + PCF), multi-pass post, HDR bloom, SSAO, alpha-blended transparency, GPU instancing, compute particles, skinning, PBR + HDR-IBL materials, and the debug-line pipeline all run on the M4 and are each verified against a committed golden at `DIFF 0.0000` (15 in total). Notable Metal-vs-Vulkan differences: depth bias is set on the encoder (not the PSO); `End*Pass` uses `commit + waitUntilCompleted` in place of Vulkan's barrier+fence; LINE_LIST topology is selected at draw time. The remaining intentional gap is the **windowed present loop** (interactive viewport + mouse-drag editing), which is Vulkan/Windows only; Metal is headless-offscreen-verified.

---

## Scene Layer

`engine/scene/` sits above the RHI and below the application:

- `vertex.h` — canonical `Vertex { pos[3], color[3], uv[2], normal[3], tangent[3] }` (stride 56, tangent added for normal mapping) and `MeshVertexLayout()`; a `SkinnedVertex` (stride 88, + joints/weights) and `SkinnedMeshVertexLayout()`; `InstanceTransformLayout()` (stride 64) for the per-instance stream.
- `transform.h` — `Transform { position, eulerRadians, scale }` with `Matrix()` → TRS `mat4`.
- `mesh.h/.cpp` — `Mesh` owns `IBuffer* vertices` and `IBuffer* indices`; static factories `Cube(device)`, `Plane(device)`, `Sphere(device)`.
- `renderable.h` — `Renderable { Mesh*, ITexture*, Transform }`; `scene_io` (load/dump) and `commands` (undoable edits) round-trip the scene; `instance_grid` builds the deterministic instancing field.

The scene layer — and the `asset`, `anim`, `physics`, `runtime`, `editor`, and `debug` modules above the seam — has no knowledge of which backend is active. They call `IRHIDevice::Create*` through the seam, which is exactly why the backend-agnostic subset compiles into `hf_core` and runs clean under AddressSanitizer in the pure unit tests. This is the same code the Metal headless target uses to render its reference images.
