# Slice Y — SSAO (Screen-Space Ambient Occlusion) Design Spec

Date: 2026-06-14
Branch: `slice-ssao`

## Goal
Add classic screen-space ambient occlusion via a view-space normal+depth prepass, a
hemisphere-kernel AO pass, a blur, and a composite that multiplies the lit scene's ambient/indirect
term by the blurred AO so contact creases darken. Showcase on the settled physics sphere-pyramid (the
same scene as `--physics-shot`) where spheres touch each other and the ground. All existing pipelines,
shaders, and goldens stay byte-for-byte unchanged: SSAO uses SEPARATE pipelines + shaders.

## Scene
Reuse the deterministic 4-layer square-pyramid sphere pile (30 unit spheres, ground plane, sky) from
`--physics-shot`, stepped to rest (240 steps @ dt=1/120). Same fixed camera (`eye{6.5,4.5,7}`,
`center{0,1,0}`, fovY=60deg) and directional light. This scene has abundant sphere-sphere and
sphere-ground contact crevices — ideal to show localized contact AO.

## Passes (all NEW, separate from existing paths)
1. **Shadow** — reuse existing shadow pipelines (unchanged) so the lit composite is shadowed too.
2. **Lit scene -> HDR RT (`sceneColor`, RGBA16F)** — reuse existing `lit` + `lit_instanced` pipelines
   writing into an RGBA16F target (exactly like the bloom showcase). UNCHANGED shaders.
3. **G-buffer prepass -> `gbuffer` RT (RGBA16F)** — NEW `gbuffer.vert`/`gbuffer.frag`. Renders the same
   geometry (ground plane + instanced spheres) but writes **view-space normal (xyz) + view-space
   linear depth (w = -viewPos.z, positive in front of camera)**. A NEW push constant carries
   `modelView` (mat4) and `modelViewIT-as-normalMatrix` is approximated by `(float3x3)modelView`
   (rotation + uniform scale only — true for the spheres and the uniformly-scaled ground). Instanced
   variant `gbuffer_instanced.vert` builds modelView from the per-instance model * the view push const.
   Background (sky / cleared) is written with depth `w=0` so the AO pass treats it as "no geometry".
4. **SSAO pass -> `ao` RT (RGBA16F, half-res)** — NEW fullscreen `ssao.frag`. Per pixel: read
   view-space normal+linear depth from `gbuffer`; reconstruct view-space position from screen UV +
   linear depth + `tanHalfFovY`/`aspect` (no matrix inverse needed — `viewPos = vec3((uv*2-1) *
   vec2(aspect*t, t) * linDepth, -linDepth)` accounting for the V-down post.vert UV). Build a TBN from
   the normal + a per-pixel rotation vector pulled from a baked 4x4 tiled noise (16 constants, CPU
   baked, NO runtime RNG). For each of 16 baked hemisphere kernel samples: offset the view-space
   position by `TBN*kernel*radius`, project to UV via the projection params, read the neighbor's linear
   depth from `gbuffer`, and if the stored surface is closer to the camera than the sample (sample is
   occluded) AND within `radius` (range check with smooth falloff), count it. `ao = 1 -
   (occlusion/N)*intensity`, clamped. Kernel + noise are baked deterministically in C++.
5. **Blur pass -> `aoBlur` RT (RGBA16F, half-res)** — NEW fullscreen `ssao_blur.frag`: 4x4 box blur over
   `ao` to remove kernel noise.
6. **Composite -> swapchain** — NEW fullscreen `ssao_composite.frag`: samples `sceneColor` (lit HDR) and
   `aoBlur`, outputs `litColor * lerp(1, ao, aoStrength)` then the SAME exposure/ACES/gamma/grade/grain/
   vignette as post.frag (replicated verbatim, like bloom_composite). AO multiplies the WHOLE lit color
   for the showcase (simplest robust path); because AO is ~1 on open surfaces and <1 only in crevices,
   the visible effect is localized contact darkening, not a global dim. Documented deviation from
   "ambient-term-only": multiplying full lit color is the standard showcase approach and reads cleanly
   on this diffuse scene (low specular), matching the slice's stated "simplest robust approach".

## SSAO-off comparison
`--ssao-shot <path>` renders WITH SSAO. `--ssao-shot-off <path>` (or env reuse) renders the identical
scene but the composite uses `aoStrength=0` (AO forced to 1) — same pipeline, so the only difference is
the AO term. This gives a clean on/off comparison for visual verification.

## Reconstruction math (Vulkan, V-down UV from post.vert)
- post.vert gives `uv` with origin top-left, V increasing downward (Vulkan), un-flipped.
- NDC: `ndc.x = uv.x*2-1`, `ndc.y = uv.y*2-1`. The projection bakes a Y-flip (`m[5] = -1/t`), so a
  view-space point with +Y-up maps to NDC -Y-up... we account for this by reconstructing view-space
  with `viewPos.x = ndc.x * (aspect*t) * linDepth`, `viewPos.y = -ndc.y * t * linDepth`,
  `viewPos.z = -linDepth`. The `-ndc.y` undoes the projection Y-flip so the reconstructed view-space Y
  matches the gbuffer's stored view-space normal Y. The forward-projection inside the loop applies the
  same flip in reverse to land back in UV space. This keeps AO consistent on Vulkan; the Metal path
  uses the post.vert HF_MSL_GEN V-flip already, plus the same reconstruction.

## Determinism
- 16-sample hemisphere kernel: generated CPU-side with a fixed integer hash (no `rand()`), scaled so
  samples cluster near the origin (accelerating interpolation `lerp(0.1,1,t*t)`).
- 4x4 (16) rotation-noise vectors: baked CPU-side from the same fixed hash, tiled across the screen.
- Both uploaded as push-constant / constant data; two runs are byte-identical.

## Seam
No `vk*`/`MTL`/`Metal` tokens added to the protected engine dirs (grep stays 12). All new code lives in
`shaders/`, `samples/hello_triangle/main.cpp`, `metal_headless/visual_test.mm`, and the two CMake lists.

## Shaders registered in BOTH lists
`gbuffer.vert`, `gbuffer_instanced.vert`, `gbuffer.frag`, `ssao.frag`, `ssao_blur.frag`,
`ssao_composite.frag` — added to the Vulkan DXC list (samples/hello_triangle/CMakeLists.txt) and the
Metal MSL-gen list (metal_headless/CMakeLists.txt).

## Verification
- `ctest --preset windows-msvc-debug` stays green (11/11).
- Capture SSAO-on and SSAO-off BMP->PNG; visually confirm localized contact darkening present on-vs-off,
  clean (not noisy/global/haloed).
- Metal: add `--ssao` arg; new golden `tests/golden/metal/ssao.png` (two runs DIFF 0.0000) if Mac
  reachable; existing 11 goldens unchanged.
