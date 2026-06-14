# Hazard Forge — Slice G: Offscreen Render Targets + Post-Processing

**Date:** 2026-06-14
**Status:** Self-approved (autonomous session)
**Branch:** `slice-g-rendertargets`

## Goal

Render the scene into an **offscreen color texture**, then run a **fullscreen post-processing
pass** that samples it (tonemap + vignette + gamma) and writes the final image to the swapchain.
This introduces render-to-texture — the foundation for shadows, bloom, SSAO, deferred shading,
and the whole post pipeline.

**Definition of Done:** `--shot scene.bmp` shows the multi-object scene with a visible post effect
(darkened corners / vignette + tonemapped colors) — proving the frame went through an offscreen
target and a fullscreen pass. Headless-verified, zero validation errors, ctest green.

## Bold decisions

1. **RT color format == swapchain format.** So the existing lit pipeline renders into the RT
   unchanged (no per-target pipeline variants). The RT owns its own depth.
2. **RT is sampleable like a texture.** `IRenderTarget` exposes a descriptor set (image+sampler at
   the material set layout, set 1) so the post pass binds it with the existing `BindTexture`.
3. **Fullscreen triangle from `SV_VertexID`** — no vertex buffer for the post pass (3 verts
   generated in the shader). Standard, zero-allocation.
4. **One RT sized to the swapchain, recreated on resize.** Owned by the sample for now (not the
   device), created from RHI primitives.

## RHI seam extensions (additive)

- `class IRenderTarget : public ITexture` — an offscreen color image (+depth) you render into and
  can `BindTexture`. (Inheriting ITexture lets the post pass bind it directly.)
- `IRHIDevice::CreateRenderTarget(uint32_t w, uint32_t h) -> std::unique_ptr<IRenderTarget>`
  (color format = swapchain format, with its own depth).
- `IRHIDevice::BeginRenderTargetFrame(IRenderTarget&) -> FrameContext` — returns a command buffer
  recording into the RT's color+depth (dynamic rendering). Sample records the scene as usual.
- `IRHIDevice::EndRenderTargetFrame(const FrameContext&)` — ends recording, submits, and
  transitions the RT color to `SHADER_READ_ONLY_OPTIMAL` so the next pass can sample it.
- `GraphicsPipelineDesc`: add `bool fullscreen = false;` (when true: no vertex input, the post
  pass draws 3 verts; still `usesTexture` for the RT sample).

Hard rule holds: no `vk*` in `engine/rhi/`.

## Frame structure (sample)

Per frame:
1. `auto rtCmd = device->BeginRenderTargetFrame(*rt);` → record the **scene** (lit pipeline,
   per-renderable draws) into the RT. `device->EndRenderTargetFrame(rtCmd);`
2. `auto fc = device->BeginFrame();` (swapchain) → record the **post pass**: bind the fullscreen
   post pipeline, `BindTexture(*rt)` (samples the RT color), `Draw(3)`. `device->EndFrame(fc);`
   (`--shot` arms capture on this swapchain frame as before.)

So there are two pipelines: the existing **lit** pipeline (renders scene → RT) and a new
**post** pipeline (fullscreen, RT → swapchain).

## Vulkan implementation

- **`VulkanRenderTarget`** (new): VMA color image (`COLOR_ATTACHMENT | SAMPLED`, swapchain format),
  color view, own depth image+view (`D32`, DEPTH_ATTACHMENT), and a descriptor set (material set
  layout: sampled image + sampler, like VulkanTexture) for sampling the color. Tracks its current
  layout for correct transitions.
- **BeginRenderTargetFrame:** use a dedicated command buffer (or a second per-frame recorder);
  transition RT color UNDEFINED→COLOR_ATTACHMENT, depth UNDEFINED→DEPTH_ATTACHMENT; begin dynamic
  rendering into them; hand the recorder back.
- **EndRenderTargetFrame:** end rendering, transition RT color COLOR_ATTACHMENT→SHADER_READ_ONLY,
  end+submit the command buffer, wait (or fence) so the swapchain pass can sample it. (Simplest
  correct: a fence + wait, or a barrier-based dependency. For this slice, a `vkQueueWaitIdle` after
  the RT submit is acceptable — one extra sync point, optimize later.)
- **Post pipeline:** `fullscreen=true` → vertex input state with 0 bindings/attributes; the lit
  pipeline path is unchanged. Pipeline layout uses material set (set 1) only (no frame UBO needed
  for post → `usesFrameUniforms=false`); push constants none.
- The post pass binds the RT as the material texture (set 1) and draws 3 vertices.

## Shaders (`shaders/post.*.hlsl`)

- `post.vert.hlsl`: fullscreen triangle from `SV_VertexID` (positions {-1,-1},{3,-1},{-1,3}; uv
  derived), no inputs.
- `post.frag.hlsl`: sample RT color (set 1: `gTex` b0, `gSmp` b1), apply Reinhard tonemap
  `c/(c+1)`, gamma 1/2.2, and a vignette `smoothstep` based on uv distance from center. Output.

## Testing
- `rhi_smoke`: add `device->CreateRenderTarget(64,64)` + destroy to exercise the RT path headlessly.
- `math_test` unchanged.
- Visual: `--shot` (controller views the post-processed scene).

## Out of scope
HDR float RT formats, bloom/blur chains, multiple RTs / MRT, ping-pong, depth-as-texture (that's
the shadow slice next). One LDR color RT + one fullscreen post pass.
