# Slice AJ — Volumetric Fog / Light Shafts (god rays) — Design

Phase 2 GI/atmosphere. ADDITIVE: a new fullscreen ray-march pass that produces
volumetric in-scattering (god rays / light shafts) by sampling the existing
directional shadow map at every step along the view ray. No existing path,
pipeline, shader, or golden is touched. The 20 committed goldens stay byte-stable;
this slice adds the 21st (`volumetric`).

## Goal / showcase

A scene lit by a single directional light at a grazing/low angle, with several
shadow-casting occluders (a wall-with-gaps + free-standing pillars/cubes) so the
light streams BETWEEN them, plus fog filling the air. The volumetric pass produces
bright beams where the light reaches the air and dark volumes in the occluders'
shadows. Composited additively over the lit HDR scene, then tonemapped to the
swapchain. Fixed camera/light, deterministic → byte-stable golden.

## Approach (analytic ray-march, reuse the directional shadow map; no froxel grid)

Reuse the proven Slice AH (SSR) pattern almost verbatim for the scaffolding:
HDR scene RT + a view-space normal/linear-depth g-buffer (the SAME SSAO gbuffer
shaders), a single directional shadow map (`CreateShadowMap` / `SetShadowMap`),
the RenderGraph passes, and the fullscreen post pipeline. The new piece is the
ray-march fragment shader.

### Passes
1. **shadow** (existing): render occluders+ground depth into the directional
   shadow map. Single cascade — no CSM needed.
2. **scene** (existing lit path): sky + lit/shadowed geometry → HDR RGBA16F RT.
3. **gbuffer** (existing SSAO gbuffer shaders): view normal.xyz + linear depth.w
   → RGBA16F. Only `.w` (linear depth) is used here — it tells the march where the
   view ray ENDS so fog does not accumulate behind solid geometry.
4. **volumetric** (NEW, fullscreen): ray-march N steps from the camera along the
   per-pixel world ray; accumulate shadowed in-scattering. Output rgb = in-scatter
   colour, a unused. Writes its own HDR RT.
5. **composite** (NEW): `scene + volumetric` (additive) → exposure/ACES/grade/
   vignette (same math as post.frag/ssr_composite, replicated — no shared include)
   → LDR swapchain.

### Volumetric pass math (`shaders/volumetric.frag.hlsl`)

Bindings (a fullscreen pipeline with `usesFrameUniforms + usesTexture +
fragmentPushConstants`):
- set 0 b0 = FrameData (camera basis camFwd/Right/Up + skyParams.x=tanHalfFov,
  .y=aspect; viewPos; lightDir/lightColor; lightViewProj for the per-step shadow).
- set 0 t1/s1 = the directional **shadow map** (bound by `SetShadowMap`, exactly as
  the lit pass samples it).
- set 1 t0/s0 = the **g-buffer** (view normal + linear depth), bound by `BindTexture`.
- fragment push constant = `VolParams` (fog density, g, extinction, step count,
  marchDist, screen size).

**World ray reconstruction** (identical to sky.frag — no matrix inverse):
```
ndc   = uv*2-1
rayU  = camFwd + camRight*ndc.x*tanHalf*aspect + camUp*(-ndc.y)*tanHalf
```
`rayU` has unit camFwd projection, so a point at view-linear-depth `t` along the
ray is `viewPos + rayU * t`. The world direction is `normalize(rayU)`.

**Scene-depth clamp**: read the g-buffer's linear depth `dz = gbuf.w` at this
pixel. If `dz > 0` (geometry hit), clamp the march end to `dz` (the surface). If
`dz <= 0` (background/sky), march to a fixed `marchDist` so the open air still
fogs. End distance `tEnd = (dz > 0) ? min(dz, marchDist) : marchDist`.

**March**: `kSteps` (64) uniform steps from `t≈0` to `tEnd`. A baked 4×4 Bayer
dither offsets the START by a sub-step fraction (deterministic, no RNG) to break
banding. At each sample point `P = viewPos + rayU * t`:
- project `P` into light clip with `lightViewProj`, `smUV = proj.xy*0.5+0.5`
  (Metal: `smUV.y = 1-smUV.y` under HF_MSL_GEN — SAME convention as lit.frag),
  compare `proj.z` (minus a small bias) to the single shadow tap → `lit ∈ {0,1}`.
- in-scatter this step = `lit * density * HG(cosTheta, g) * stepLen`, attenuated by
  the accumulated Beer-Lambert transmittance `T = exp(-extinction * t)`.
- `cosTheta = dot(rayDir, lightDir)` (rayDir = view→sample, lightDir = travel dir
  of photons = `f.lightDir`). HG forward-peaks (g≈0.6) so looking toward the light
  glows; the dot uses the photon travel direction so a camera looking INTO the
  light sees the strongest shafts.
- accumulate `scatter += lightColor * inScatterStep * T`.

Result rgb = `scatter`, written to the volumetric HDR RT.

**Henyey-Greenstein** factored into `engine/render/volumetric.h` (pure, header-only)
and unit-tested:
```
HG(cosTheta, g) = (1 - g^2) / (4*pi * (1 + g^2 - 2*g*cosTheta)^1.5)
```
g=0 → isotropic 1/4π; forward (cosTheta=1) peak > back (cosTheta=-1) for g>0.

### Metal shadow V-flip
The per-step shadow sample uses the SAME `#ifdef HF_MSL_GEN smUV.y = 1 - smUV.y`
flip the lit pass uses, because on Metal `lightViewProj` has its NDC Y-flip baked
in CPU-side (`FlipProjY(Ortho)`), so the render and sample stay self-consistent.

## March params (chosen for clear, smooth shafts)
- steps = 64, density = 0.18, g = 0.6 (forward scatter), extinction = 0.04,
  marchDist = 28.0 (covers the scene + a bit of open air). 4×4 Bayer start dither.

## Files
- NEW `engine/render/volumetric.h` — HG phase (+ a world-ray helper), pure CPU.
- NEW `shaders/volumetric.frag.hlsl` — the ray-march.
- NEW `shaders/volumetric_composite.frag.hlsl` — additive composite + tonemap.
- NEW `tests/volumetric_test.cpp` — HG properties + ray helper. Registered hf_core/ASan.
- Vulkan: `hello_triangle.exe --volumetric-shot <out.bmp>` in main.cpp.
- Metal: `--volumetric` in visual_test.mm → golden `tests/golden/metal/volumetric.png`.
- Register both shaders in BOTH CMake shader lists; verify.ps1 golden list 20→21.

## Seam
No new `vk*`/`MTL*`/`mtl::`/`Backend::Metal` symbols above the backend dirs. The
new header is pure math; the new shaders are HLSL; main.cpp/visual_test.mm use only
the existing RHI interface (the same calls the SSR slice uses).
