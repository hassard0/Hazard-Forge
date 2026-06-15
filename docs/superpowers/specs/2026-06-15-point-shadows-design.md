# Slice AF — Omnidirectional point-light shadows (6-face cube → shadow atlas)

## Goal
Make a point light cast shadows in ALL directions (Phase-1 fidelity) by rendering the
scene from the light's position through **6 perspective frustums** (the 6 cube faces:
+X,-X,+Y,-Y,+Z,-Z), each with FOV=90°, aspect=1, into **6 tiles of one shadow atlas**.
The lit shader picks the dominant-axis face per fragment, projects the fragment by that
face's view-proj, samples that tile with 3×3 PCF, and compares depth.

This is the SAME pattern as CSM (N tiles in one atlas via `SetViewport`) and the spot
(perspective light map), just with 6 axis-aligned faces. No cubemap RHI is added.

## Architecture / hard rules
- ADDITIVE. The 17 goldens, existing lit/csm/spot/shadow shaders + paths are UNTOUCHED.
- New CPU header `engine/render/point_shadow.h` (pure math, header-only, no backend symbols).
- New shader `shaders/lit_point.frag.hlsl` (reuses `shadow_csm.vert` as the depth caster —
  faceViewProj in the push constant, exactly like CSM/spot).
- New showcase path `--point-shadow-shot <out.bmp>` in `samples/hello_triangle/main.cpp`.
- New Metal flag `--point-shadow` + golden `tests/golden/metal/point_shadow.png`.
- New unit test `tests/point_shadow_test.cpp`.
- Zero new actual `vk*`/`MTL*` backend types in any new file.

## Cube face convention (CPU `point_shadow.h` ↔ HLSL `lit_point.frag` MUST match)
Faces are indexed by dominant absolute axis of `dirLightToFrag = wpos - lightPos`:

| face | axis | faceDir (forward) | up         |
|------|------|-------------------|------------|
| 0    | +X   | (+1, 0, 0)        | (0,-1, 0)  |
| 1    | -X   | (-1, 0, 0)        | (0,-1, 0)  |
| 2    | +Y   | (0,+1, 0)         | (0, 0,+1)  |
| 3    | -Y   | (0,-1, 0)         | (0, 0,-1)  |
| 4    | +Z   | (0, 0,+1)         | (0,-1, 0)  |
| 5    | -Z   | (0, 0,-1)         | (0,-1, 0)  |

These are the conventional D3D/Vulkan cube-face basis vectors (with -Y up on the lateral
faces because `Mat4::LookAt` + `Mat4::Perspective` already bake the Vulkan Y-flip; the
test verifies on-axis center maps to clip ~0,0 with depth in [0,1]).

**Face selection** (`SelectFace`): pick the axis with the largest `|component|` of
`dir = wpos - lightPos`. Sign of that component selects +/- face. The HLSL mirrors this
exactly so the render face and the sampled face agree.

Per-face view-proj: `Perspective(90°, 1.0, near, range) * LookAt(P, P + faceDir, up)`.
On Metal the caller wraps the Perspective in `FlipProjY` (CPU-side V-flip) exactly like
every other shadow showcase, and the shader applies the `HF_MSL_GEN` `smUV.y = 1-smUV.y`.

## Atlas layout
- Atlas: **3072 × 2048** D32 sampleable shadow map (one `CreateShadowMap` — but that API
  is square-only, so we use a **3072 × 3072** atlas and only fill a 3×2 grid of 1024² tiles**).
  → Final: **3072² atlas, 3×2 grid, 1024² tiles**. Faces map to (col,row):

  ```
  face 0(+X) -> (0,0)   face 1(-X) -> (1,0)   face 2(+Y) -> (2,0)
  face 3(-Y) -> (0,1)   face 4(+Z) -> (1,1)   face 5(-Z) -> (2,1)
  ```
  i.e. `col = face % 3`, `row = face / 3`. tilesPerRow=3, tilesPerCol=2.
  Tile UV scale = (1/3, 1/2). The shader builds atlas UV =
  `tileOrigin + faceUV * tileScale`, with PCF taps clamped to the tile (inset by a texel)
  so the 3×3 kernel never bleeds across a tile boundary.

## UBO packing (must fit kFrameUboSize = 1024 bytes)
`PointFrameData` (std140-compatible float4 / float4x4 fields):

```
  0  float4x4 viewProj          // camera VP
 64  float4   lightDir          // dim directional fill dir
 80  float4   lightColor        // dim directional fill color
 96  float4   viewPos           // camera eye (xyz), w=1
112  float4x4 faceVP[6]         // 6 face view-projs = 384B -> ends 496
496  float4   ptPos             // point light pos (xyz), w=range
512  float4   ptColor           // rgb color, w=intensity
528  float4   atlasParams       // x=tilesPerRow(3), y=tilesPerCol(2), z=1/atlasTexels, w=near
544  float4   camFwd            // for SkyColor only
560  float4   camRight
576  float4   camUp
592  float4   skyParams         // x=tan(0.5 fovY), y=aspect
-> 608 bytes < 1024. ✓
```
The 6 face matrices are the only large block (384B). Tile rects are NOT stored — the
shader DERIVES (col,row) = (face%3, face/3) and the constant tileScale=(1/3,1/2), so no
extra UBO bytes for rects. atlasParams.z carries 1/atlasSize for the texel inset.

## Render (showcase)
ONE shadow pass loops the 6 faces: for each face `SetViewport(col*1024, row*1024, 1024,
1024)` then draw all casters (ground + ring of cubes/spheres + back wall) with that face's
view-proj as the caster push-constant matrix (`shadow_csm.vert`). Then ONE scene pass
shades with `lit_point.frag`.

Scene: a point light ~3 units above a ground plane, surrounded by a RING of cubes and
spheres at different azimuths + a back wall, so shadows radiate OUTWARD in all directions
(left objects shadow left, right objects shadow right, front/back too) — proving multiple
cube faces contribute. The directional fill is dim so the point light's radial shadows read.

## Shader depth compare (distance-vs-projected-depth)
We compare the fragment's projected clip-space depth (proj.z from the chosen faceVP)
against the stored tile depth — identical to spot/CSM. (The "distance-based" idea reduces
to this because all 6 faces share `Perspective(90,1,near,range)`, so clip depth is a
monotonic function of distance along the face axis; comparing proj.z is equivalent and
reuses the proven CSM/spot PCF path.) Bias 0.0025, 3×3 PCF clamped to tile bounds.

## Tests
`tests/point_shadow_test.cpp` (pure, hf_core, ASan):
- `SelectFace` returns the right face for clear ±X/±Y/±Z directions.
- Each face's view-proj projects a point straight out along its axis to ~clip center
  (proj.xy≈0), depth in [0,1].
- A point behind a face (opposite axis) projects outside the [-1,1] frustum / w<=0.

## Verification
- Vulkan: `hello_triangle --point-shadow-shot ptsh.bmp` → BMP→PNG → visually inspect:
  shadows radiate in different directions from the central light; attached; no acne; no
  hard seams at cube-face tile boundaries.
- Metal: `--point-shadow` → `point_shadow.png` golden, two-run DIFF 0.0000; 17 existing
  goldens stay DIFF 0.0000.
- `scripts/verify.ps1` golden list: 17 → 18 (`point_shadow|--point-shadow`).
