# Metal golden-image test

Golden references, all produced by the **real** Metal RHI backend running headless on an
Apple M4 (`metal_headless/visual_test`), rendering the full Slice-F scene (ground checkerboard
plane + a 3x3 grid of 9 lit, textured cubes). All are deterministic: fixed camera, fixed light,
static transforms, offscreen MTLTexture target, byte-exact readback. Two runs diff to 0.0000.

- **`scene_shadow.png` — current `visual_test` output.** Adds the directional shadow-mapping
  parity path on top of the RT+post flow: a depth-only pass from the light (`CreateShadowMap` /
  `BeginShadowPass` / `EndShadowPass`, `shaders/shadow.metal`) renders the scene into a 2048²
  Depth32Float sampleable shadow map; the lit pass (`shaders/lit.metal`) then projects each
  fragment into the light's clip space and does a 3×3 PCF compare to darken shadowed surfaces. The
  cubes cast shadows on the ground and each other. **This is what `visual_test` writes today** —
  validate against it.
- **`scene_post.png` — RT+post reference (no shadows).** Scene rendered into an offscreen
  `MetalRenderTarget` (BGRA8 color + D32 depth), then a fullscreen post pass (ACES tonemap + gamma
  + cinematic grade + vignette, `shaders/post.metal`) samples the RT into the captured output. This
  exercises the Metal render-target + post-processing parity path
  (`CreateRenderTarget` / `BeginRenderTargetFrame` / `EndRenderTargetFrame` + the `fullscreen`
  pipeline). Retained as the pre-shadow reference.
- **`scene.png` — historical no-post reference.** The earlier direct scene→swapchain render with no
  render target and no post pass. Retained for comparison; `visual_test` no longer produces it
  (the render flow now always goes scene→RT→post). To regenerate it you would have to revert the
  RT+post wiring in `visual_test.mm`.

## How it is produced

```sh
# On the Mac (Command Line Tools only, runtime MSL compile — no Xcode/metal CLI):
source ~/mac-remote-rig/env.sh
cd ~/hazard-forge
cmake -S metal_headless -B build-metal -G Ninja
cmake --build build-metal
./build-metal/visual_test /tmp/metal_post.png
```

`visual_test` constructs the real `MetalDevice` in **headless** mode
(`rhi::mtl::CreateMetalDeviceHeadless(W, H)` -> offscreen BGRA8 color + D32 depth texture, no
window / CAMetalLayer / present), builds the scene from the engine `scene/` layer, creates a lit
pipeline from `shaders/lit.metal` (runtime-compiled MSL), creates a `MetalRenderTarget`, renders
the scene into it (`BeginRenderTargetFrame` -> draws -> `EndRenderTargetFrame`), then runs a
fullscreen post pipeline (`shaders/post.metal`, runtime-compiled MSL) that binds the RT and draws
3 vertices into the captured swapchain output. It reads the result back via `CaptureNextFrame()` +
`GetCapturedPixels()` and writes a PNG. All through the `IRHIDevice` / `ICommandBuffer` seam — the
same calls the Vulkan `hello_triangle` sample makes for its scene→RT→post path.

## How to validate a future run (visual regression)

```sh
# Render a candidate, then compare to this golden. Deterministic => DIFF 0.0000.
./build-metal/visual_test /tmp/out.png
~/mac-remote-rig/compare.sh tests/golden/metal/scene_shadow.png /tmp/out.png 0.0
# prints: DIFF 0.0000 (threshold 0.0)   and exits 0 on a match.
```

`compare.sh` is the rig's stdlib-only mean-per-pixel PNG diff. A non-zero DIFF means the Metal
render changed — investigate before updating this golden.

## NDC-Y note

`math::Perspective` bakes the Vulkan clip-space Y flip (+Y down). Metal NDC is +Y up (like GL), so
the Metal backend undoes the flip in `shaders/lit.metal` (`out.clip.y = -out.clip.y`). The fix
lives entirely in the Metal path; the shared engine math and the Vulkan backend are untouched.

The shadow map follows the same convention: `math::Ortho` (used for `lightViewProj`) bakes the
same Vulkan Y-flip, so `shaders/shadow.metal` undoes it on the depth-pass output the same way
(`out.clip.y = -out.clip.y`). The render flip and the lit-pass sample formula
(`smUV = proj.xy*0.5+0.5`) are self-consistent — the flip cancels — so the shadow UV math is byte-
identical to the Vulkan backend (`shaders/lit.frag.hlsl`). Light-space depth is [0,1] (Ortho), so
`curDepth = proj.z` compares directly. A modest depth bias (encoder `setDepthBias 1.25 / slope
1.75`, matching Vulkan) plus a 0.0025 shader bias fight shadow acne.
