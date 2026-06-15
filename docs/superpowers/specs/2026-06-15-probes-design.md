# Slice AK — Reflection + Irradiance Probes (baked local cubemap GI)

## Goal
Bake the local scene into a cubemap-atlas from a fixed probe position so hero objects reflect and
are lit by their ACTUAL surroundings (local color bleed) — not just the global HDR sky. This is the
Phase-2 GI step: prove **local** reflection/irradiance vs a **global** environment map.

ADDITIVE: existing render paths, pipelines, shaders, and the 21 goldens stay byte-for-byte untouched.

## Reuse (study-first)
- `engine/render/point_shadow.h`: the 6-face cube → 3×2 tile-atlas machinery (FaceDir/FaceUp/
  FaceViewProj/SelectFace/FaceTile). The probe MIRRORS this EXACTLY but renders scene COLOR into an
  RGBA16F atlas (not depth).
- Slice U HDR render target: `CreateRenderTarget(w,h,RGBA16_Float)` is renderable + sampleable.
- Slice R environment binding: `BindEnvironment` binds a sampled image at the dedicated env set/slot
  (Vulkan set 3 binding 11/12; Metal fragment texture(11)/sampler(12)) when a pipeline declares
  `usesEnvironment`. The probe atlas is bound the same way via a NEW `BindReflectionProbe`.

## New pure-math header: `engine/render/probe.h` (no device, no backend symbols)
Namespace `hf::render::probe`. Reuses `point_shadow`'s face math for the BAKE projections and adds
the COMBINED-atlas dir→(tile, uv) mapping that the shader mirrors exactly:

- Atlas layout (ONE RGBA16F texture, so it binds through the single env slot):
  - **Reflection** block: 3×2 grid of `kReflTile`=512² tiles → 1536×1024, at atlas origin (0,0).
  - **Irradiance** block: 3×2 grid of `kIrrTile`=64² tiles → 192×128, placed BELOW the reflection
    block at y-offset 1024. Total atlas = 1536 × 1152.
  - `kAtlasW=1536`, `kAtlasH=1152`.
- `FaceViewProj(P, face)` = `point_shadow::FaceViewProj(P, face, near, range)` (90° FOV, aspect 1).
- `SelectFace(dir)` = `point_shadow::SelectFace(dir)` (dominant abs axis → face 0..5).
- `FaceUVFromDir(dir, face)`: project `dir` onto the selected face and return the [0,1] face UV
  (the planar cube-face parameterization). The shader uses the projected clip-UV instead (it has the
  per-face VP), but the CPU version exists for the unit test (dir→face→uv consistency) and matches
  the standard cube-face S/T parameterization.
- `ReflTileUV(face, faceUV)` and `IrrTileUV(face, faceUV)`: map a face's [0,1] UV into the atlas UV
  for that face's tile in the reflection / irradiance block respectively. Pure float math; the
  shader's `SelectFace`+tile mapping is the exact mirror.

## New shader: `shaders/lit_probe.frag.hlsl`
Base lit PBR (Cook-Torrance direct light, same helpers as `lit_pbr_ibl.frag`), but the IBL comes
from the PROBE atlas (bound at the env slot gEnv/gEnvSmp):

- **Specular** = reflection-block sample at `R = reflect(-V, N)`: `face=SelectFace(R)`, project R by
  `faceVP[face]` to a [0,1] face UV, remap into the reflection tile, sample. (Sharp mirror; roughness
  is folded by blending toward the irradiance sample as roughness rises — cheap "blurrier reflection
  as it roughens".)
- **Diffuse ambient** = irradiance-block sample at `N`: `face=SelectFace(N)`, project N by
  `faceVP[face]`, remap into the irradiance tile, sample × albedo × (1−metallic).
- Tile UVs are clamped to the tile interior (texel inset) so linear filtering never bleeds across
  tile borders — same guard `lit_point.frag` uses for its PCF.
- `HF_MSL_GEN` V-flip on the sampled face UV, matching `lit_point.frag`'s cube-atlas convention.
- Own `FrameData` (< kFrameUboSize=1024): viewProj, lightDir/Color, viewPos, faceVP[6], probePos,
  atlasParams (tile scales + irradiance y-offset + texel), camera basis. ~600 bytes.

## Bake approach
1. **Reflection bake:** render the ROOM (Cornell-style colored walls + a couple static colored
   panels — NOT the dynamic hero objects, to avoid recursion) from the fixed probe center P into the
   reflection block: for each face, `SetViewport(reflection tile)` and draw with `faceVP[face]`.
   Deterministic, done once. Each face tile therefore captures one wall (red left → −X face,
   green right → +X face, etc.).
2. **Irradiance convolution:** a fullscreen pass over the small irradiance block. For each output
   irradiance texel (a direction N built from its tile+uv), average a cosine-weighted set of taps
   from the REFLECTION block sampled around N (a coarse hemisphere convolution). This yields a
   diffuse-convolved irradiance cube atlas — real color bleed (red-ish toward the red wall, etc.),
   cheap because the irradiance tiles are tiny (64²). Implemented as a dedicated fragment shader
   `shaders/probe_irradiance.frag.hlsl` that reads the reflection block (bound as a texture) and
   writes the irradiance block of the SAME atlas in a second render-target pass (separate RT, then
   the lit pass samples the combined atlas). To keep it ONE bound texture, the bake renders BOTH
   blocks of one atlas: reflection pass renders the 6 reflection tiles; irradiance pass renders the
   6 irradiance tiles of the same RT (reading the reflection tiles it just wrote).

## Binding: `BindReflectionProbe(ITexture&)`
New `ICommandBuffer::BindReflectionProbe` (default no-op). Binds the probe atlas at the env set/slot:
- **Metal:** identical to `BindEnvironment` — `setFragmentTexture/SamplerState` at texture(11)/
  sampler(12). A render target is an `IMetalSampled`, so this works directly.
- **Vulkan:** the probe atlas is a `VulkanRenderTarget`; give it a lazily-allocated env-layout
  descriptor set (binding 11 = its colorView, 12 = env sampler) and bind it at the env set index.
  Reuses the existing `environmentSetLayout` / `environmentSampler` — NO new set layout, NO new
  backend type leaks above `engine/rhi/`.

The lit_probe pipeline sets `usesEnvironment=true` so set 3 is reserved exactly like the IBL path.

## Showcase: `hello_triangle --probe-shot <out.bmp>` (Vulkan) / `visual_test --probe` (Metal)
A box room: red left wall (−X), green right wall (+X), neutral floor/ceiling/back/front (Cornell-
box-like). Probe baked at room center. Hero objects: a **metallic sphere** (reflects the red/green
walls on its sides — local, not sky) and a **matte dielectric box** (picks up red bleed on its left,
green on its right from the irradiance). Fixed camera + light, deterministic. Capture → BMP → PNG,
self-inspect. New golden `tests/golden/metal/probe.png` on Metal (two-run diff 0.0000).

## Unit test: `tests/probe_test.cpp` (hf_core / ASan)
Pure math on `probe.h`: +X dir → +X face center; face→dir↔dir→face round-trip consistency; a probe
face view-proj projects an on-axis point to clip center; reflection/irradiance tile UVs land inside
their block. No device.

## Verify
- `scripts/verify.ps1` golden list 21 → 22.
- Register `probe_test` in `tests/CMakeLists.txt`; `lit_probe`/`probe_irradiance` in BOTH CMake
  shader lists.
- All 21 existing goldens stay DIFF 0.0000.

## Spec deviations / decisions
- Single combined atlas (reflection + irradiance in one texture) bound through ONE env-style slot —
  avoids a second descriptor set / second backend binding path. Documented above.
- Walls are tinted via a per-draw push-constant color (lit_probe push constant = model + material +
  tint), since the shared cube mesh has fixed per-face vertex colors.
