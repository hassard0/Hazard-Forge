# Slice AE — Spot light + spot-light shadows (design)

Date: 2026-06-15
Branch: `slice-spot-shadows`
Status: implementing

## Goal

Add a **spot light** type that casts a sharp shadow via a **single perspective shadow map**.
Phase-1 fidelity. Strictly ADDITIVE: the directional/CSM/point paths, their shaders, and the 16
existing goldens stay untouched. We add a new lit shader (`lit_spot.frag`), a new
`hello_triangle --spot-shot` showcase, a new Metal `--spot` showcase + golden, a new unit test, and
a 17th entry in `verify.ps1`.

## Spot light model

A spot light is like a point light but with a direction, a cone cutoff, and a perspective shadow:

```
struct SpotLight {
    Vec3  position;
    Vec3  direction;     // unit, the axis the cone points along
    float innerCone;     // radians, half-angle — full brightness inside
    float outerCone;     // radians, half-angle — zero past this; soft edge inner->outer
    Vec3  color;
    float range;         // distance falloff + shadow far plane
};
```

### Shadow projection

```
spotViewProj = Perspective(2*outerCone, aspect=1, near, range) * LookAt(position, position+direction, up)
```

* `Perspective` bakes the Vulkan Y-flip. On **Metal** the matrix is flipped CPU-side with the same
  `FlipProjY` helper the directional/CSM showcases use, so the shared HLSL needs no spot-specific
  flip beyond the existing `HF_MSL_GEN` `smUV.y = 1-smUV.y` on the sample. RENDER (caster) and
  SAMPLE (lit) both use the same CPU-side `spotViewProj`, so they stay self-consistent — exactly the
  directional-shadow pattern.
* FOV = `2*outerCone` so the shadow frustum exactly covers the cone. `aspect=1` (square map).
* `up` is chosen non-parallel to `direction` (use world-up unless the spot points near-vertical, in
  which case use world +Z), matching `LookAt`'s requirement.
* The single shadow map is one `CreateShadowMap(2048)` — no atlas needed (one spot light).

## Shaders

### `shaders/lit_spot.frag.hlsl` (NEW)

Copy the base `lit.frag` PBR/IBL/ambient machinery (Cook-Torrance, SkyColor IBL, small ambient).
The directional + IBL + ambient terms stay so it's a believable lit scene, but the directional light
is **toned down** (the showcase passes a dim directional color) so the SPOT reads as the dominant
light and its shadow is unambiguous.

Add the spot term:

1. `L = normalize(spotPos - wpos)`, `dist = length(spotPos - wpos)`.
2. **Distance falloff**: `att = saturate(1 - dist/range); att *= att;` (smooth, like point lights).
3. **Cone**: `c = smoothstep(cosOuter, cosInner, dot(-L, spotDir))` — 1 on-axis, soft edge from
   inner to outer half-angle, 0 outside. (`-L` points light->fragment; `spotDir` is the cone axis.)
4. **Shadow**: project `wpos` by `spotViewProj`, `smUV = proj.xy*0.5+0.5`, `HF_MSL_GEN` V-flip,
   3×3 PCF with bias, clamped to the `[0,1]` UV + `[0,1]` depth frustum bounds (fragments outside the
   frustum are unshadowed — `shadow=1`, but the cone factor `c` is already 0 there so they stay dark).
5. Spot radiance = `spotColor * intensity * att * c * shadow`, fed through `hfCookTorrance`.

#### Spot FrameData layout (own struct; fits `kFrameUboSize`=1024)

```
float4x4 viewProj;     //   0
float4   lightDir;     //  64  directional (dim in the showcase)
float4   lightColor;   //  80
float4   viewPos;      //  96
float4x4 spotViewProj; // 112  perspective light matrix -> ends 176
float4   spotPos;      // 176  xyz position, w unused
float4   spotDir;      // 192  xyz cone axis (unit), w unused
float4   spotColor;    // 208  rgb color, w unused
float4   spotParams;   // 224  x=cosInner, y=cosOuter, z=range, w=intensity
float4   camFwd;       // 240
float4   camRight;     // 256
float4   camUp;        // 272
float4   skyParams;    // 288
                       // total 304 bytes < 1024
```

The spot shadow binds to set 0 (binding 1 = depth, binding 2 = sampler) exactly like the directional
shadow map — `SetShadowMap` machinery is reused verbatim.

### Caster

Reuse the existing depth-only `shadow_csm.vert.hlsl` (it already takes the view-proj in the push
constant alongside the model matrix). The shadow pass does not care whether the projection is
perspective or ortho. Push constant = `{ spotViewProj(16), model(16) }` = 128 B.

## Showcase (`hello_triangle --spot-shot <out.bmp>`)

Ground plane + a few cubes/spheres, lit primarily by ONE spot light mounted above and angled down so
it casts a clear cone of light with sharp shadows of the objects within the cone; outside the cone is
dark. Fixed camera/light, deterministic. Capture to BMP → PNG → visual inspection.

* Spot above the scene (e.g. `pos = (0, 12, 4)`) aimed down-forward at the cluster, `outerCone ≈ 22°`,
  `innerCone ≈ 16°`, `range ≈ 30`.
* Directional light dim (`~0.2`) + low ambient so the cone pool dominates.
* Shadow map 2048², 3×3 PCF.

## Metal (`visual_test --spot`) + golden `tests/golden/metal/spot.png`

Mirror the Vulkan path (offscreen RT → post blit, like `RunCsmShowcase`). `FlipProjY` on both the
camera proj and the `spotViewProj`. New golden via two-run DIFF 0.0000. The 16 existing goldens stay
DIFF 0.0000. `verify.ps1` golden list goes 16 → 17 (`spot | --spot`).

## Unit test (`tests/spot_test.cpp`, hf_core / ASan)

Pure math, no device. Add a small header-only helper `engine/render/spot.h` with
`SpotViewProj(pos, dir, outerCone, near, range)` (Vulkan clip space, reused by both showcases and the
test). Assert:

* a point at cone center + mid-range projects near clip center (`proj.xy ≈ 0`) and inside `[0,1]` depth;
* a point well outside the cone projects outside `[-1,1]`;
* cone attenuation `smoothstep(cosOuter, cosInner, dot(axis, dirToCenter))` == 1 on-axis and 0 past
  the outer half-angle.

## Architecture / seam

No `vk*`/`MTL*` symbols added above the backend dirs. New header `engine/render/spot.h` is pure math
(like `csm.h`). New shaders registered in BOTH the Vulkan DXC list
(`samples/hello_triangle/CMakeLists.txt`) and the Metal MSL-gen list
(`metal_headless/CMakeLists.txt`). Seam baseline stays 16 benign matches.

## Deviations

None planned. (Will record here if any surface during implementation.)
