# Slice T — Alpha-blended Transparency (Design Spec)

Date: 2026-06-14
Branch: `slice-transparency`

## Goal
Add a sorted, alpha-blended translucent ("glass") render pass on top of the existing
opaque scene, plus the RHI plumbing it needs (depth-WRITE control). New pipeline +
shader + showcase + golden are ADDITIVE; existing pipelines, shaders, and goldens are
byte-for-byte untouched.

## 1. RHI depth-write control
- New field `GraphicsPipelineDesc.depthWrite = true` (default `true`).
- Meaning: when `depthTest` is enabled, `depthWrite` controls whether fragments that pass
  the depth test also WRITE depth. Default `true` reproduces today's behavior exactly
  (depth-tested pipelines also write depth), so every existing pipeline is unchanged.
- Backends:
  - Vulkan (`vulkan_pipeline.cpp`): `ds.depthWriteEnable = (desc.depthTest && desc.depthWrite)`.
    Previously `depthWriteEnable = desc.depthTest`. With default `depthWrite=true` this is
    identical for all existing pipelines.
  - Metal (`metal_pipeline.mm`): in the depth-test branch,
    `dsd.depthWriteEnabled = desc.depthWrite ? YES : NO`. Depth-only (shadow) pass keeps
    `YES` unconditionally (it has no color and must write depth).
- The seam header (`engine/rhi/rhi.h`) only gains a plain `bool` + a comment carefully
  worded to introduce NO new `vk*`/`MTL`/`Metal` literal tokens (seam grep stays at 12).

## 2. Transparent pipeline + shader
- New fragment shader `shaders/transparent.frag.hlsl`; the VERTEX stage REUSES
  `lit.vert.hlsl` (same VSOutput; `material.xy` interpolant is repurposed to carry the
  per-object base alpha in `.x` — see push-constant note). No new vertex shader.
- Self-contained glassy look (NOT dependent on the PBR/IBL material sets):
  - Directional diffuse (half-Lambert) + a Blinn-Phong specular highlight.
  - Procedural `SkyColor(R)` environment reflection in the mirror direction (reusing the
    same procedural sky function as `lit.frag`, copied locally — the transparent shader is
    self-contained and binds NO textures).
  - RGB = lerp(tint*lighting, skyReflection, fresnel*0.6) so glass picks up sky sheen at
    grazing angles and shows its tint head-on.
  - Alpha = `lerp(baseAlpha, 1.0, pow(1 - saturate(dot(N,V)), 5))` (Fresnel-style): edges
    read as more opaque than the head-on center -> glass look.
- Per-object data via push constant. The transparent pipeline uses its OWN push range,
  separate from lit. Layout (vertex stage): `{ float4x4 model; float4 tintAlpha; }` = 80
  bytes (tintAlpha.rgb = tint, tintAlpha.w = baseAlpha). This matches lit.vert's existing
  `{ float4x4 model; float4 material; }` so we can REUSE lit.vert.hlsl unchanged: the vertex
  shader forwards `material.xy` -> fragment `material` interpolant. The transparent fragment
  reconstructs base alpha from `material.x`; the tint is forwarded via a custom path: to keep
  lit.vert untouched we pass the FULL tintAlpha through the push constant and the fragment
  reads the SAME push constant is not possible (push constant is vertex-stage only on both
  backends). Therefore the fragment derives tint from the vertex COLOR attribute? No — to stay
  self-contained and per-object we forward tint via the two free interpolant slots already in
  VSOutput: `color` (the per-vertex color) is multiplied by tint on the CPU is not per-vertex.
  DECISION: extend the reuse by passing tint through `material` is only 2 floats. Instead we
  reuse lit.vert AND pass baseAlpha+packed tint... (see "Decision: shader pairing" below.)
- Pipeline state: `alphaBlend=true, depthTest=true, depthWrite=false, cullNone=true`.
  - `cullNone=true` (double-sided): real glass shows both front and back faces; with depth
    write off there is no self-occlusion artifact and both faces blend, which reads as solid
    glass. Documented choice.

### Decision: shader pairing (final)
To keep `lit.vert.hlsl` BYTE-UNCHANGED while giving the fragment both a tint and a base
alpha, the transparent pass uses a DEDICATED vertex shader `transparent.vert.hlsl` that:
- takes the same mesh vertex layout,
- reads push constant `{ float4x4 model; float4 tintAlpha; }` (80 bytes, same size/layout as
  lit's model+material so the HF_MSL_GEN binding story is identical),
- outputs world pos + world normal + forwards `tintAlpha` (rgb tint, w baseAlpha) as a
  `nointerpolation float4` interpolant.
This is cleaner than overloading lit.vert and keeps lit.vert untouched. The fragment shader
`transparent.frag.hlsl` consumes that interpolant. Both are new files; no existing shader
changes.

## 3. Sorted transparent pass
- In the scene RT pass, AFTER sky + all opaque lit objects, render N (=4) translucent
  objects sorted back-to-front by distance from the camera eye (CPU sort each frame;
  deterministic for the fixed camera).
- They depth-TEST against the opaque depth (so opaque geometry in FRONT correctly occludes
  them) but do NOT write depth (`depthWrite=false`), so overlapping glass blends correctly
  and nothing self-occludes. Back-to-front order makes the over-blend correct between the
  translucent objects themselves.

## 4. Showcase + verification
- Vulkan: `hello_triangle.exe --transparency-shot <path>` — self-contained capture path
  (does NOT touch other showcases). Opaque scene = checkerboard ground plane + procedural
  sky + a few opaque lit objects (cubes/spheres) to show THROUGH. Then 4 overlapping tinted
  glass spheres of different colors at different depths in the sorted transparent pass.
  Fixed camera/light. Lit + shadowed opaque behind. One frame -> BMP -> exit.
- Convert BMP->PNG, visually inspect: must see SEE-THROUGH colored glass, tinting on
  overlap, more-opaque grazing edges (Fresnel), correct occlusion by opaque geometry in
  front.
- Metal: same showcase in `metal_headless/visual_test.mm` (`--transparency` arg), new
  golden `tests/golden/metal/transparency.png` (two runs DIFF 0.0000) IF the Mac is
  reachable; else Metal code lands and golden is "pending".

## Non-goals / constraints
- No order-independent transparency; simple sorted-over blend is sufficient for the showcase.
- Existing goldens, pipelines, and shaders untouched. New `depthWrite` defaults true.
- Seam grep stays == 12.
