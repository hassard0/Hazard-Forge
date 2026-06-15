# Hazard Forge — Architecture

## The RHI Seam

The central architectural decision in Hazard Forge is the **Rendering Hardware Interface seam** — a layer of pure C++ abstract interfaces in `engine/rhi/rhi.h` that separates all engine and application code from any GPU backend.

### The hard rule

> No `vk*` type, no `MTL*` type, and no backend header may appear in any above-seam module — `engine/rhi/` (the seam itself), `engine/scene/`, `engine/math/`, `engine/render/`, `engine/material/`, `engine/game/`, or the rest. Backend symbols are confined to `engine/rhi_vulkan/` and `engine/rhi_metal/` respectively. (`engine/hal/` is the platform layer: it touches SDL's `SDL_Metal_*` to vend an opaque `CAMetalLayer*` as a `void*`, but pulls in no Metal/Obj-C GPU types.)

The rule is checkable with a single `grep` for `vk[A-Z]|MTL|Metal` over the above-seam dirs: every match is a comment, prose, or a substring false-positive (`kMetallic`, the `Backend::Metal` enumerator name). The only genuine backend-symbol CODE above the seam is the **two-line `rhi_factory` dispatch** (the `CreateMetalDevice` forward-declaration + the `Backend::Metal` → `mtl::CreateMetalDevice` call) — the single sanctioned place the factory names a backend. This baseline is re-checked each consolidation pass and enforced by convention in every slice plan.

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

`GraphicsPipelineDesc` is a small, flat descriptor that encodes what the pipeline needs to know about shaders, vertex layout, depth, push constants, descriptor set membership, and a set of mode flags that has grown additively slice by slice — `fullscreen`, `depthOnly`, `pbrMaterial`, `usesEnvironment`, `usesJointPalette`, `depthWrite`, `alphaBlend`, `cullNone`, `lineList`, `fragmentPushConstants`, `usesLightClusters` — all in terms of engine-defined enums and types, no backend types. Every new flag **defaults to the value that leaves all pre-existing pipelines byte-for-byte unchanged**, which is what keeps the committed goldens stable as features land.

There is also a compute path (`IComputePipeline` + `Dispatch`) used by the GPU particle system and the **GPU-driven culling** pass, on the same seam. Newer per-frame binding entry points have been added additively to the command interface — e.g. `BindLightClusters` (clustered/Forward+ light data) and `BindReflectionProbe(ITexture& probeAtlas)` (the baked reflection/irradiance probe atlas) — each with a default no-op body so existing pipelines and goldens are unaffected.

Three later seam additions follow the same defaulted-no-op discipline, so backends without the path still link and pre-existing goldens are byte-for-byte unchanged:

- **Indirect draw** — `DrawIndexedIndirect(IBuffer& args, ...)` consumes a `VkDrawIndexedIndirectCommand` / `MTLDrawIndexedPrimitivesIndirectArguments` produced on the GPU by the cull compute pass (frustum-cull + ordered compaction), so the survivor draw never round-trips to the CPU. The `vk*`/`MTL*` indirect calls live only in the backend dirs.
- **Resource-state transitions** — `TransitionResource(state prev → next)` lets the render-graph barrier solver request a transition declaratively. On Vulkan it lowers to a `vkCmdPipelineBarrier2` (`VkImageMemoryBarrier2` with the matching layout + stage/access masks); Metal's tracked-hazard model makes it a no-op.
- **Secondary / parallel recording** — passes record into per-thread secondary command buffers that are replayed in deterministic creation order (`vkCmdExecuteCommands` on Vulkan; an `MTLParallelRenderCommandEncoder`'s sub-encoders on Metal). This is what makes a 1-worker render and an N-worker render byte-identical.

Backends are handed out through the factory:

```cpp
// Vulkan (Windows):
auto device = hf::rhi::CreateDevice(hf::rhi::Backend::Vulkan, window);

// Metal headless (macOS, no window):
auto device = hf::rhi::mtl::CreateMetalDeviceHeadless(width, height);

// Metal windowed (macOS): built directly from a CAMetalLayer* (passed as void*), so a windowed,
// presenting Metal device can be constructed WITHOUT pulling in hal/window.h or SDL — used by the
// SDL-free native Cocoa entry (mac_window/) that drives the interactive --fly viewport on macOS.
MetalDevice dev(caMetalLayer /* void* */, width, height);
```

`MetalDevice` therefore has three constructors — windowed-from-`Window&` (SDL HAL), windowed-from-`CAMetalLayer*` (native Cocoa, SDL-free), and headless-offscreen — all behind the same `IRHIDevice` seam.

Both return `std::unique_ptr<IRHIDevice>`. The caller never sees a backend type again.

---

## Descriptor / Binding Model

Hazard Forge uses a **frequency-based descriptor set layout** adopted in Slice D (lighting):

| Set | Update frequency | Contents |
|-----|-----------------|----------|
| 0 | Per frame | UBO: `viewProj`, directional light, view position, point lights, `lightViewProj`, camera basis for sky; shadow map depth image (binding 1) + shadow sampler (binding 2) |
| 1 | Per material / per draw | Material textures + samplers. The base set is color + normal; the **PBR** path (`BindMaterialPBR`) widens it to the full 5-texture metallic-roughness set (base / metal-rough / normal / emissive / occlusion). The **environment** (HDR IBL) binds on a dedicated slot via `BindEnvironment` so the base material layouts are unchanged. |
| 2 | Per skinned draw | Joint palette (skinning), bound via `SetJointPalette` at a dedicated vertex binding. |
| 3 | Per clustered-lit frame | The clustered/Forward+ light data (Slice AG): the per-cluster `Cluster` array (offset/count) at binding 13, the flat light-index list at binding 14, and the `GpuLight` array at binding 15. Bound via `BindLightClusters` when a pipeline sets `usesLightClusters = true`; otherwise the set is absent and pre-existing pipelines are byte-for-byte unchanged. |

GPU instancing adds a **second per-instance vertex stream** (binding 1, step-per-instance) rather than a descriptor set — four `RGBA32_Float` attributes carry a column-major `float4x4` per instance.

The set-3 cluster set follows the **same dedicated-extra-set pattern** established by the environment (HDR IBL) set and the joint palette: a feature that only some pipelines need binds on its own slot through a single `Bind*` call, so the base material/frame layouts — and therefore every existing golden — are untouched. `BindLightClusters` is the first case of **graphics-stage storage buffers** on the seam: the three set-3 bindings are `STORAGE_BUFFER`s read by the fragment shader (`lit_clustered.frag`), not the uniform buffers used by set 0/1. On Vulkan they are SSBO descriptors; on Metal they map to fragment `[[buffer(13/14/15)]]` via `spirv-cross --msl-decoration-binding` (`engine/rhi_metal/metal_common.h`: `kFragClusterBuf=13`, etc.).

**Set 0** is owned by the device. The device maintains one UBO per frame-in-flight (double-buffered), each with a pre-baked descriptor set that already points at its buffer. `SetFrameUniforms(data, size)` memcpys into the current frame's UBO; `BindPipeline` auto-binds set 0 when the pipeline has `usesFrameUniforms = true`. The CPU never writes a UBO that the GPU is still reading.

**Set 1** is owned per texture / render target. Each `VulkanTexture` and each `VulkanRenderTarget` allocates one descriptor set from a device pool at creation time and updates it with its image view and the device's default sampler. `BindTexture(ITexture&)` binds that set.

**Push constants** carry the per-object model matrix (64 bytes, `mat4`). `PushConstants(data, size)` records a push to the vertex stage.

When `usesFrameUniforms = false` (the post-processing pipeline), the material set is the only set and occupies set index 0. The `BindTexture` path accounts for this shift.

---

## Per-frame UBO growth

The `FrameData` struct (352 bytes at the time of the skybox slice; `kFrameUboSize` was 512) has grown additively across slices:

- Slice D: `viewProj`, `lightDir`, `lightColor`, `viewPos` — basic per-frame camera + directional light
- Slice H: `ptCount`, `ptPos[3]`, `ptColor[3]` — colored point lights
- Slice I: `lightViewProj` — directional shadow map light-space matrix
- Slice J: `camFwd`, `camRight`, `camUp`, `skyParams` — camera basis for sky ray reconstruction
- Slices AD–AF: the cascaded / spot / point shadow matrices, per-cascade split distances, and the spot/point light parameters — the shadow set's per-frame data. This is what pushed the struct past 512 B, so **`kFrameUboSize` was bumped 512 → 1024** (`engine/rhi_vulkan/vulkan_device.cpp` and `engine/rhi_metal/metal_device.h`, kept in sync); both backends now allocate a 1024-byte per-frame UBO, and `SetFrameUniforms` asserts the upload fits.

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

### Render-graph resource-state tracker + barrier solver

`render::RenderGraph` no longer relies on each pass to hand-place its own barriers. Every imported resource carries a current **state** (e.g. `RenderTarget`, `ShaderRead`, `TransferSrc`, `Present`); each pass declares the state it needs each resource in (read vs write). Before a pass runs, the **barrier solver** diffs each resource's prior state against the pass's required state and, when they differ, emits exactly one transition via `ICommandBuffer::TransitionResource(prev → next)`, then advances the tracked state. On Vulkan that transition becomes an explicit `vkCmdPipelineBarrier2`; on Metal the tracked-hazard model makes it a no-op. The solver is pure host logic over `rhi/` types (no backend symbols) and is unit-tested (`render_graph_test`). Crucially the auto-inserted barriers are **proven hazard-free by the Khronos synchronization-validation layer**: running every showcase under sync validation yields zero `SYNC-HAZARD-*` lines, which is the oracle that the solver places correct, sufficient, and non-redundant barriers.

### Parallel command recording (worker pool)

Within a pass, draws can be recorded **across threads** without changing the output a single bit. A worker pool fans the pass's draw list out to N per-thread secondary command buffers; each thread records its slice into its own secondary, and the pass replays the secondaries **in deterministic creation order** (`vkCmdExecuteCommands` with the array in order on Vulkan; closing an `MTLParallelRenderCommandEncoder`, which commits its sub-encoders in creation order, on Metal). Because the replay order is fixed and independent of which thread finished first, a `--mt --workers 1` render and a `--mt --workers 4` render produce a **byte-identical** image (the `mt` golden + a 1-vs-N hash test, `parallel_record_test`, both enforce this). The ordering guarantee — not thread count — is what preserves determinism.

### Shadow-atlas tiling (`SetViewport`)

The expanded shadow set (Slices AD–AF) renders **many** light-space depth maps per frame — N directional cascades, a spot map, and the six faces of a point light's cube — without allocating N separate depth textures. Instead the depth pass clears one shadow **atlas** once, then for each sub-map calls a new seam entry point, `ICommandBuffer::SetViewport(x, y, w, h)`, to restrict rasterization to that tile before drawing the geometry for that cascade/face. The lit pass samples the right tile by transforming into the corresponding light-space matrix and offsetting into the atlas region.

`SetViewport` is a defaulted no-op on the base interface, so passes and backends that don't tile an atlas are unaffected (the pre-existing single-shadow-map goldens are byte-for-byte unchanged). On Vulkan it records `vkCmdSetViewport`/`vkCmdSetScissor`; on Metal it sets the encoder viewport + scissor rect. The CSM frustum-split + per-cascade ortho fit, the spot `spotViewProj`, and the point-light 6-face cube view-proj + dominant-axis face/tile mapping are all pure, header-only, unit-tested math (`render/csm.h`, `render/spot.h`, `render/point_shadow.h`) — the goldens prove the GPU side, the unit tests prove the math.

---

## Headless Capture and Golden-Image Testing

Headless capture is a first-class RHI feature, not a test harness hack.

`IRHIDevice::CaptureNextFrame()` arms a flag. On the next `EndFrame`, the Vulkan backend transitions the just-rendered swapchain image to `TRANSFER_SRC_OPTIMAL`, copies it to a host-visible staging buffer via `vkCmdCopyImageToBuffer`, waits idle, maps the buffer, and stores the BGRA8 pixels. `GetCapturedPixels(outBGRA, w, h)` hands them out. Present is skipped in capture mode — the window need not be visible.

The Metal headless target (`metal_headless/visual_test`) uses the same `CaptureNextFrame()` / `GetCapturedPixels()` path, but the "swapchain" is an offscreen `MTLTexture` (no `CAMetalLayer`, no window server). The same `IRHIDevice` / `ICommandBuffer` calls that the Vulkan sample makes drive the Metal backend through the exact same code path.

There are now **29 committed Metal reference renders** under `tests/golden/metal/` (M4, deterministic to byte level — two runs diff `0.0000`), one per `visual_test` showcase flag: `scene_shadow` (default), `skinning`, `pbr_helmet`, `instanced`, `ibl_helmet`, `physics`, `transparency`, `bloom`, `scene_import`, `debug_viz`, `anim_blend`, `ssao`, `capstone`, `camera_pose`, `gizmo`, `csm`, `spot`, `point_shadow`, `clustered`, `ssr`, `volumetric`, `probe`, `taa`, `cull`, `gpu_cull`, `mt`, `mat_graph`, `mat_graph2`, and `game`. `scripts/verify.ps1` (and the self-hosted CI Metal job) build `metal_headless` once and golden-compare **all 29** at threshold `0.0` on every run, so any unintended change to shared shader/render/loader code is caught as a non-zero DIFF on a specific golden. The `clustered` golden additionally proves the Forward+ light-culling path is **byte-identical to brute-force** shading (192 deterministic lights produce the same image as shading every light per fragment).

Alongside the image goldens there is one **non-image golden**: the engine-state introspection JSON (`tests/golden/introspect/default_scene.json`). `editor::DescribeEngine` serializes the live engine state — engine/features/showcases, a commands manifest, scene entities + transforms, camera + lights, and stats, with `backends == ["vulkan","metal"]` — as a deterministic LF-only JSON document. It is backend-agnostic (pure `hf_core`), so `verify.ps1` (and the Windows CI job) byte-match the live `--introspect` output against this golden **on the Windows side only**; the Mac is not needed because the bytes are identical on both backends. This is the agent-facing OBSERVE artifact: a machine can read the JSON to see the engine's state and capabilities, then act through the commands manifest.

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
- **Full feature parity on Metal, golden-tested.** Offscreen render targets (including `RGBA16_Float` HDR targets for bloom/SSAO), directional shadow mapping (static / skinned / instanced depth-only + PCF), cascaded shadow maps, spot-light shadows, omnidirectional point-light cube shadows (all atlas-tiled via `SetViewport`), clustered/Forward+ lighting (set-3 fragment storage buffers via `BindLightClusters`), screen-space reflections, ray-marched volumetric fog / light shafts, reflection + irradiance probes (baked cubemap atlas bound via `BindReflectionProbe`), multi-pass post, HDR bloom, SSAO, alpha-blended transparency, GPU instancing, compute particles, skinning, PBR + HDR-IBL materials, temporal anti-aliasing (Halton-jittered reprojection), CPU + GPU-driven frustum culling (compute cull → compacted indirect draw), multithreaded command recording (a parallel render encoder whose sub-encoders match the single-threaded order byte-for-byte), the data-driven material/shader-graph pipelines, the playable game sample, and the debug-line pipeline all run on the M4 and are each verified against a committed golden at `DIFF 0.0000` (29 in total). Notable Metal-vs-Vulkan differences: depth bias is set on the encoder (not the PSO); `End*Pass` uses `commit + waitUntilCompleted` in place of Vulkan's barrier+fence; LINE_LIST topology is selected at draw time; `SetViewport` sets the encoder viewport+scissor; the set-3 cluster storage buffers bind to fragment `[[buffer(13/14/15)]]`.
- **Windowed Metal present loop (macOS).** Beyond the headless path, Metal now also drives a **windowed, presenting** viewport: `MetalSwapchain` wraps a `CAMetalLayer` and `AcquireNext()` vends the next drawable, and `MetalDevice` can be constructed straight from a `CAMetalLayer*` (passed as `void*`) so the SDL-free native Cocoa entry in `mac_window/` builds a presenting device **without** `hal/window.h` / SDL. This runs the same interactive `--fly` viewport (fly camera + mouse pick + gizmo drag + hot-reload) that Vulkan/Windows runs. The windowed path is **build-verified** by `verify.ps1` / CI on the bench Mac (it compiles and headless-renders all 29 goldens); the on-screen interactive window is exercised manually (user-confirmed), while the deterministic logic beneath it (camera, picking, gizmo, file-watch) is unit- and golden-tested on both backends. There is no longer a feature gap between the backends for offscreen rendering; the only manual-only surface is live on-screen interaction.

---

## Scene Layer

`engine/scene/` sits above the RHI and below the application:

- `vertex.h` — canonical `Vertex { pos[3], color[3], uv[2], normal[3], tangent[3] }` (stride 56, tangent added for normal mapping) and `MeshVertexLayout()`; a `SkinnedVertex` (stride 88, + joints/weights) and `SkinnedMeshVertexLayout()`; `InstanceTransformLayout()` (stride 64) for the per-instance stream.
- `transform.h` — `Transform { position, eulerRadians, scale }` with `Matrix()` → TRS `mat4`.
- `mesh.h/.cpp` — `Mesh` owns `IBuffer* vertices` and `IBuffer* indices`; static factories `Cube(device)`, `Plane(device)`, `Sphere(device)`.
- `renderable.h` — `Renderable { Mesh*, ITexture*, Transform }`; `scene_io` (load/dump) and `commands` (undoable edits) round-trip the scene; `instance_grid` builds the deterministic instancing field.

The scene layer — and the `asset`, `anim`, `physics`, `material`, `game`, `runtime`, `editor`, and `debug` modules above the seam — has no knowledge of which backend is active. They call `IRHIDevice::Create*` through the seam, which is exactly why the backend-agnostic subset compiles into `hf_core` and runs clean under AddressSanitizer in the pure unit tests. This is the same code the Metal headless target uses to render its reference images.

---

## Material / Shader Graph Layer (`engine/material/`)

Phase 4 adds a **data-driven material system** that authors shaders as node graphs instead of hand-written HLSL, while preserving the "shaders are generated, not hand-written" and golden-stability invariants. It is pure host logic above the seam — no `vk*`/`MTL*`/`Backend` symbols — and is split into five concerns plus a build-time tool:

- **`shader_graph` (`shader_graph.h/.cpp`)** — the pure-CPU graph model: typed nodes (constants, texture samples, math ops, fresnel, …) wired into the four PBR fragment inputs (`kBaseColor`, `kMetallic`, `kRoughness`, `kEmissive`). No GPU types; fully unit-tested (`shader_graph_test`).
- **`codegen`** — lowers a validated graph to HLSL that drops into the existing `lit_pbr` fragment contract (same descriptor sets / push constants), so a graph material is just another PBR pipeline as far as the render graph and goldens are concerned.
- **`material_loader`** — parses the JSON authoring format (`assets/materials/*.mat.json`) into a `ShaderGraph`.
- **`runtime_compile` (`runtime_compile.h/.cpp`)** — the **runtime path**: it shells out to the `dxc` compiler as a **subprocess** (temp-file IO only; HLSL → SPIR-V), then builds an `IPipeline` from the result. It fails **safe** — returns `nullopt` + an error on any problem, never throws across the boundary — so a bad live edit cannot crash the host.
- **`live_material` (`live_material.h`)** — the in-editor authoring controller: it watches a material file and re-runs the loader → codegen → runtime compile chain on change, hot-swapping the pipeline for live iteration.

**Two production paths, proven equal.** A **build-time codegen tool** (`tools/material_codegen`) runs during the build to bake the showcase materials into committed generated HLSL (`shaders/generated/mat_showcase*.frag.hlsl`), which compile through the normal offline shader pipeline (the fast, shipping path). The **runtime** `runtime_compile` path generates and compiles the *same* graph live (the authoring path). The two are required to render **byte-identically**: `--material-shot` (build-time) and `--material-live-shot` (runtime) produce the same SHA, and the `mat_graph` / `mat_graph2` goldens pin both. This is the material analogue of the HLSL→MSL "generated, not hand-written" rule — the live authoring convenience can never silently diverge from what ships.

---

## Gameplay Layer (`engine/game/`)

`engine/game/roll_game` is a small **playable game sample** sitting at the very top of the stack — a roll-a-ball: a player sphere rolls along a scripted track collecting pickups. `MakeRollGame` builds the initial `GameState`; `StepGame(state, dt)` advances it by the engine's fixed timestep over `game::ScriptedTrack`. It is pure C++ above the seam (no rendering symbols), unit-tested (`roll_game_test`), and **fully deterministic**: the same inputs always yield the same trajectory, so the end state is fixed (`score:3, won:true, steps:380`). The `--game-shot` showcase steps the game to a fixed mid-track capture frame and renders the player + remaining pickups through the existing static-lit scene path, pinned by the `game` golden. Gameplay state and rendering are cleanly separated — the sample proves the engine can host an actual interactive loop, not just render set-piece showcases.

---

## The validation-clean invariant

As of the render-graph barrier work, **Vulkan-validation-cleanliness is a permanent gate**, not a one-off check. Every `--*-shot` showcase is run under the **Khronos validation layer** with both **core** and **synchronization** validation enabled (`VK_LAYER_KHRONOS_validation`, `VK_LAYER_PATH` pointed at the layer's package bin dir, sync validation switched on). The bar is **zero** `VUID-*`, `SYNC-HAZARD-*`, `UNASSIGNED-*`, or `[ERROR]` lines across all showcases. The synchronization-validation half is specifically the oracle for the render-graph barrier solver: if the solver ever placed a missing or wrong barrier, sync validation would flag a hazard. The only tolerated diagnostics are benign `[WARNING: Performance]` notices — notably the depth-only shadow pipelines binding a full vertex layout produce "vertex attribute at location N not consumed by vertex shader" — which are expected and documented, not errors.

The validation layer is **not** a link-time dependency; it is pulled in via Conan (`vulkan-validationlayers` in `conanfile.py`, debug builds) purely to ship `VkLayer_khronos_validation.{dll,json}` so the gate can actually load the layer on a box where it is not installed system-wide. Without that dependency the oracle would be silently inactive and a validation regression could slip through unnoticed.
