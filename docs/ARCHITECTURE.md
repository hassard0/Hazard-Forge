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

`GraphicsPipelineDesc` is a small, flat descriptor that encodes what the pipeline needs to know about shaders, vertex layout, depth, push constants, descriptor set membership, and a few mode flags (`fullscreen`, `depthOnly`) — all in terms of engine-defined enums and types, no backend types.

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
| 1 | Per material / per draw | Sampleable color texture + default sampler |

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

The `kFrameUboSize` constant is bumped whenever the struct approaches the current allocation (256 → 512 when shadows were added). The C++ struct and the HLSL `cbuffer FrameData` in every shader that reads it must be kept in sync by hand.

---

## Multi-Pass Frame Structure

A typical frame on the Vulkan backend (post Slice J) executes three passes:

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

The reference render lives at `tests/golden/metal/scene.png` (M4, full Slice-F scene: ground plane + 3×3 lit cube grid, fixed camera and light). The render is deterministic to byte-level; two runs diff to 0.0000.

---

## Vulkan Backend Notes

- **Vulkan version:** 1.3 dynamic rendering (`VK_KHR_dynamic_rendering`); no `VkRenderPass` / `VkFramebuffer` objects.
- **Synchronization:** Vulkan Synchronization 2 (`vkCmdPipelineBarrier2` / `VkImageMemoryBarrier2`). A `TransitionImage` helper in `vulkan_device.cpp` centralizes the barrier pattern.
- **Memory:** Vulkan Memory Allocator (VMA 3.3) for all images and buffers. Staging uploads use a host-visible VMA buffer, `vkQueueWaitIdle` after the copy (synchronous load-time path).
- **Frames in flight:** `kFramesInFlight = 2` (double-buffered). Per-frame fences + semaphores in `frames_[frameIndex_]`. Per-frame UBOs indexed by `frameIndex_`.
- **Shader pipeline:** HLSL → SPIR-V via DXC (`-spirv`, invoked by a CMake custom command at build time). Shaders are loaded as `.spv` files at runtime via `IShaderModule`.
- **Depth:** swapchain-sized `VK_FORMAT_D32_SFLOAT` image owned by the device, recreated on resize.

## Metal Backend Notes

- **No SPIR-V.** `CreateShaderModule(ShaderModuleDesc&)` throws on the Metal backend; the Metal sample calls `CreateShaderModuleMSL(source, entryPoint)` instead, which compiles MSL at runtime via `MTLDevice newLibraryWithSource:`.  This is a seam deviation that will be resolved by a future HLSL → MSL toolchain.
- **NDC Y-flip.** `hf::math::Perspective` bakes the Vulkan clip-space Y-flip (`m[5] = -1/tan`). Metal NDC is +Y up, so the Metal lit shader undoes this: `out.clip.y = -out.clip.y`. The fix is entirely in `shaders/lit.metal`; the shared math and Vulkan backend are untouched.
- **No explicit barriers.** Metal's command encoder model uses implicit hazard tracking; there are no `MTLBlitCommandEncoder` barriers equivalent to Vulkan image layout transitions. The Metal backend omits them.
- **Frame pacing.** A `dispatch_semaphore_t inFlight_` (initial count = `kFramesInFlight`) blocks `BeginFrame` until the GPU has finished with the oldest in-flight frame's resources, replacing Vulkan's per-frame fence approach.
- **Render targets / shadows / post:** not yet implemented on Metal. Methods throw `std::runtime_error`. The headless scene render (Slice F parity) uses only `BeginFrame` / `EndFrame` / `SetFrameUniforms` / `BindTexture`.

---

## Scene Layer

`engine/scene/` sits above the RHI and below the application:

- `vertex.h` — canonical `Vertex { pos[3], color[3], uv[2], normal[3] }` (stride 44) and `MeshVertexLayout()`.
- `transform.h` — `Transform { position, eulerRadians, scale }` with `Matrix()` → TRS `mat4`.
- `mesh.h/.cpp` — `Mesh` owns `IBuffer* vertices` and `IBuffer* indices`; static factories `Cube(device)`, `Plane(device)`, `Sphere(device)`.
- `renderable.h` — `Renderable { Mesh*, ITexture*, Transform }`.

The scene layer has no knowledge of which backend is active. It calls `IRHIDevice::CreateBuffer` and `IRHIDevice::CreateTexture` through the seam. This is the same layer the Metal headless target uses to render its reference image.
