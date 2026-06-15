# Slice AD — Cascaded Shadow Maps (CSM) — Design Spec

Date: 2026-06-15
Branch: `slice-csm`

## Goal
Upgrade directional shadows from a single ortho shadow map to N cascades for crisp shadows
across the full view-distance range. Additive: the existing single-shadow path, its shaders
(`lit.frag`, `shadow.vert`), and all 15 goldens stay byte-for-byte untouched.

## Approach: shadow ATLAS (not a texture array)
One larger shadow map subdivided into N tiles. Each cascade renders into its own tile via a
viewport/scissor sub-rect. The lit-CSM fragment shader picks a cascade by view-space depth,
transforms `wpos` by that cascade's `lightViewProj`, remaps the resulting [0,1] UV into the
cascade's atlas tile, and does 3x3 PCF clamped to the tile bounds.

This reuses the existing single shadow-texture binding (set 0 binding 1/2) with NO new
array-texture RHI.

### Cascade config (chosen)
- **N = 4 cascades**, atlas = **4096x4096**, tiles in a **2x2 grid** of 2048x2048 each.
- Split scheme: practical split (blend of logarithmic + uniform), **lambda = 0.5**, over the
  camera's `[near, far]` shadow range (near = camera near; far = a CSM shadow-distance cutoff,
  e.g. 60 units, well past the scene depth so the far cascade still covers receding ground).
- Per-cascade ortho fit: split the camera frustum into N depth sub-frusta using the practical
  splits; for each, take the 8 world-space corners, transform into light view space, fit a tight
  AABB, and build `Ortho(minX,maxX, minY,maxY, -maxZ_pad, -minZ_pad)` in light view space (Z is
  along -lightDir; we pull the near plane back by a pad so casters behind the slice still cast).

## RHI changes (default-preserving)
- `ICommandBuffer::SetViewport(x,y,w,h)` — sets BOTH viewport and scissor to a sub-rect (default
  no-op in the base interface so existing passes/backends are unaffected; both backends override).
  The shadow-CSM pass calls `BeginRenderPass` once (clears the whole atlas), then per cascade sets
  the tile viewport and draws all casters with that cascade's `lightViewProj`.
- `kFrameUboSize`: **bumped 512 -> 1024** in BOTH backends. 4 cascade mat4 (256B) + the existing
  ~352B base layout + per-cascade split/offset/scale vec4s overflow 512B. This is an additive
  constant change — existing shaders read fewer bytes, so their layout is unchanged and the 15
  goldens are byte-for-byte unaffected.

## CSM uniform packing (`lit_csm.frag.hlsl` / `shadow_csm.vert.hlsl` own struct)
Layout over the per-frame UBO (CSM showcase fills this; other showcases keep their own layout):
```
float4x4 viewProj;          //   0
float4   lightDir;          //  64
float4   lightColor;        //  80
float4   viewPos;           //  96
float4   csmSplits;         // 112  x,y,z,w = view-space far depth of cascade 0..3
float4x4 cascadeVP[4];      // 128  per-cascade lightViewProj (256B) -> ends 384
float4   camFwd;            // 384  (kept for parity, unused by CSM frag)
float4   camRight;          // 400
float4   camUp;             // 416
float4   skyParams;         // 432
float4   csmAtlas;          // 448  x=tilesPerRow, y=tileUVScale (1/tilesPerRow), z=numCascades
```
Total 464B < 1024B. Atlas tile (col,row) for cascade c: col = c % tilesPerRow, row = c / tilesPerRow.
Tile UV = (cascadeUV * tileUVScale) + (float2(col,row) * tileUVScale).

## Cascade selection + sampling (lit_csm.frag)
1. View-space depth of fragment: `vz = dot(wpos - viewPos, camFwd_for_depth)` — but to keep CPU/GPU
   consistent and deterministic we compare against `csmSplits` computed as the **view-space linear
   distance** along the camera forward. We pass `camFwd` (already in UBO) and `viewPos`; depth =
   `dot(wpos - viewPos, normalize(camFwd))`.
2. Pick smallest `c` with `depth <= csmSplits[c]`; clamp to N-1.
3. `lp = mul(cascadeVP[c], wpos); proj = lp.xyz/lp.w; uv = proj.xy*0.5+0.5;` (Metal flips uv.y like
   lit.frag, because the CPU bakes the same Vulkan Y-flip into the ortho and Metal flips CPU-side).
4. Map uv into the tile, clamp PCF taps to the tile's [min,max] so a 3x3 kernel never bleeds into a
   neighbour tile.
5. 3x3 PCF with bias; same ambient/PBR as lit.frag otherwise.

Debug tint (per-cascade color) lives behind `#define HF_CSM_DEBUG_TINT` — OFF in the committed golden.

## Shaders
- NEW `shaders/lit_csm.frag.hlsl` — copy of lit.frag with CSM cascade selection + atlas sampling.
- NEW `shaders/shadow_csm.vert.hlsl` — depth-only caster that reads the per-cascade `lightViewProj`
  from a push constant (model mat4 + cascade index) — actually simpler: the caster reuses the SAME
  `lightViewProj` field but we re-upload the UBO per cascade? No — UBO is one per frame. Instead the
  caster takes the cascade's VP via a **push constant** (mat4 lightVP + mat4 model = 128B). This
  avoids re-uploading the frame UBO between cascades and keeps the shadow caster cascade-agnostic.

## Showcase (`--csm-shot <path>`, Vulkan)
Long ground plane (receding to far Z) + cubes/spheres at near/mid/far + truck, grazing-angle
directional light for long shadows. Fixed deterministic camera. Render the 4-cascade atlas, then the
scene with `lit_csm`. Capture BMP.

## Unit test (`tests/csm_test.cpp`, pure, hf_core/ASan)
- `CsmSplits(near,far,N,lambda)` returns N increasing splits within (near,far].
- `FitCascadeOrtho(...)` bounds all 8 frustum-slice corners inside [-1,1]^2 and [0,1] depth of the
  cascade's clip space.

## Math lib (`engine/render/csm.h`, header-only, pure)
`CsmSplits`, `ComputeFrustumCornersWorld`, `FitCascadeLightMatrix` — used by both the test and the
showcase so the GPU path and the test exercise the SAME code.

## Metal
Add `--csm` to `metal_headless/visual_test.mm` -> new golden `tests/golden/metal/csm.png` (two-run
DIFF 0.0000) if the Mac is reachable; otherwise note "Metal golden pending". Register `lit_csm.frag`
+ `shadow_csm.vert` in the MSL-gen list.
