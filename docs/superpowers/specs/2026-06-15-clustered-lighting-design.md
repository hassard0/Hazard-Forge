# Slice AG — Clustered / Forward+ Lighting (design spec)

**Date:** 2026-06-15  **Branch:** `slice-clustered`  **Status:** implementing

## Goal
Make HUNDREDS of dynamic point lights affordable. Brute-force lit shaders cap at ≤3 point
lights in the per-frame UBO (`ptCount, ptPos[3], ptColor[3]`). To scale to 64–256 lights we
(a) store all lights in a STORAGE buffer, (b) partition the view frustum into a 3D cluster grid
and assign each light's bounding sphere to the clusters it overlaps (CPU, deterministic), and
(c) in a new lit fragment shader compute the fragment's cluster and iterate ONLY that cluster's
lights. This is the Forward+ / clustered-shading technique. **Additive** — existing lit/shadow
paths, pipelines, and the 18 goldens are byte-for-byte untouched.

CPU light culling this slice (deterministic, golden-stable). A compute-shader cull is a future
optimization, explicitly out of scope.

## Cluster grid
- Dimensions `CX×CY×CZ = 16×9×24` (3456 clusters). XY follow the 16:9 framebuffer tiling; Z is
  exponential depth slices.
- A cluster is addressed by `(cx, cy, cz)`; the flat index is
  `idx = cx + cy*CX + cz*CX*CY`.
- **XY ← screen tiles.** For a fragment at pixel `(px, py)` (SV_Position.xy, top-left origin):
  `cx = clamp(floor(px / W * CX), 0, CX-1)`, `cy = clamp(floor(py / H * CY), 0, CY-1)`.
- **Z ← exponential view-depth slice.** With `vz = -viewSpacePos.z` (positive distance in front
  of the camera, since the engine's view space looks down −Z), the slice is
  `sliceZ(k) = znear * (zfar/znear)^(k/CZ)`. Inverting:
  `cz = clamp(floor( log(vz/znear) / log(zfar/znear) * CZ ), 0, CZ-1)`.
  (`vz <= znear → cz=0`; `vz >= zfar → cz=CZ-1`.)

This exact mapping is implemented ONCE in `engine/render/clustered.h` (CPU) and MIRRORED
verbatim in `shaders/lit_clustered.frag.hlsl`. The CPU code documents the formulas; the shader
re-derives them with the same constants passed via the frame uniform.

## CPU culling — `engine/render/clustered.h` (header-only, pure math, no device/backend)
Inputs: camera `view`, `proj`, `znear`, `zfar`, screen `W,H`, grid dims, and a list of point
lights `{ viewSpacePos (Vec3), radius (float), color (Vec3) }` (caller transforms world→view).

Algorithm (per cluster, per light AABB-vs-sphere):
1. For each cluster `(cx,cy,cz)` compute its **view-space AABB**. The Z bounds come from the
   exponential slice: `zNear_k = sliceZ(cz)`, `zFar_k = sliceZ(cz+1)` (both positive distances;
   view-space z spans `[-zFar_k, -zNear_k]`). The XY bounds are reconstructed from the screen
   tile corners unprojected to each of the two z planes (the tile's view-space x/y extent grows
   with depth), taking the min/max over the near & far plane corners — a conservative AABB that
   fully contains the cluster's frustum sub-volume.
2. A light at `viewSpacePos` with `radius` overlaps the cluster iff the squared distance from the
   sphere center to the AABB (clamped-point distance) `≤ radius²`. Standard sphere-AABB test.
3. Two-pass build for tight packing: pass 1 counts per-cluster light hits → fill `clusters[idx] =
   {offset, count}` via prefix sum; pass 2 fills the flat `lightIndices` array.

Outputs (match the three GPU buffers exactly):
- `clusters`  : `CX*CY*CZ` × `{ uint offset; uint count; }`  (8 B each).
- `lightIndices` : flat `uint` array, length = Σcount.
- `lights` : per light `{ float4 posRadius; float4 color; }` (32 B) — **view-space** position in
  xyz, radius in w; color rgb, intensity in w. The shader does its lighting in view space for the
  clustered point lights (it already has `view`), so positions are pre-transformed on the CPU.

Invariant (unit-tested): Σ`clusters[i].count` == `lightIndices.size()`; every
`clusters[i].offset == ` running prefix sum.

## RHI addition — fragment-stage storage buffers (set 3)
- New `GraphicsPipelineDesc.usesLightClusters` (default false → existing pipelines unchanged).
- New `ICommandBuffer::BindLightClusters(IBuffer& clusters, IBuffer& lightIndices, IBuffer&
  lights)` (default no-op).
- **Vulkan:** a new descriptor set layout `clusterSetLayout_` at **set 3** with three
  `STORAGE_BUFFER` bindings (0,1,2), fragment stage, created with
  `PUSH_DESCRIPTOR_BIT_KHR` (mirrors the compute SSBO path — bound inline, no pool sizing). The
  pipeline-layout assembly pads sets 1/2 with placeholders (exactly like `usesEnvironment`) so
  clusters land at set 3. `BindLightClusters` pushes the three buffers via `pushDescriptorFn()`
  with `VK_PIPELINE_BIND_POINT_GRAPHICS`. Existing pipelines (flag false) get NO set-3 layout, so
  their layouts are byte-for-byte unchanged → 18 goldens hold.
- **Metal:** flat fragment buffer slots. HLSL declares the three SSBOs at `[[vk::binding(0/1/2,
  3)]]`; `spirv-cross --msl-decoration-binding` maps space-3 buffers to fragment buffer indices.
  `BindLightClusters` calls `setFragmentBuffer:atIndex:` for each. New `kFragCluster*` constants
  in `metal_common.h`.

## Shader — `shaders/lit_clustered.frag.hlsl` (NEW; lit.frag untouched)
Base PBR (Cook-Torrance) + dim directional fill + procedural sky IBL, identical machinery to
lit_point.frag, PLUS the clustered point-light loop:
1. Compute view-space position: `vpos = mul(view, float4(wpos,1)).xyz`; `vz = -vpos.z`.
2. `cx/cy` from `SV_Position.xy` and screen size; `cz` from `vz` via the exponential-slice inverse.
3. `idx = cx + cy*CX + cz*CX*CY`; read `clusters[idx] = {offset,count}`.
4. Loop `count` lights from `lightIndices[offset + j]`; for each, fetch `lights[li]`; do a point
   PBR contribution with smooth `att = sat(1 - dist/radius)²` distance falloff (matching the CPU
   radius used for culling so no light pops at a cluster boundary). Lighting done in VIEW space
   (transform N and V to view space, or equivalently keep world but use world light pos — we use
   view space consistently since light positions are stored view-space).

FrameData layout (`ClusteredFrameData`, own struct, ≤1024 B): viewProj, view, lightDir,
lightColor, viewPos, plus `clusterParams` = (CX, CY, CZ, znear), `clusterParams2` = (zfar, W, H,
0), and the sky/cam vectors. `static_assert(sizeof ≤ 1024)`.

## Showcase + verification
- **Vulkan:** `hello_triangle.exe --clustered-shot <path>`. Ground plane + a few objects lit by
  **192 deterministic point lights** (16×12 grid, positions/colors/radii derived from the index —
  NO rng). Fixed camera. CPU-build the three cluster buffers, create them as Storage buffers with
  initialData, render via the clustered pipeline, capture to BMP. Convert to PNG and visually
  inspect for many overlapping colored pools, smooth falloff, NO tile banding.
- Optional brute-force reference (loop all 192 lights, same shader path with one giant cluster) to
  confirm visual match.
- **Metal:** `visual_test --clustered` → new golden `tests/golden/metal/clustered.png` (two-run
  DIFF 0.0000) if the Mac is reachable; else "Metal golden pending".
- `scripts/verify.ps1` golden list 18 → 19.

## Unit test — `tests/clustered_test.cpp` (hf_core, ASan)
- z-slice round-trip: a view point at a known depth maps to the cluster whose AABB contains it.
- cluster-index round-trip for several view points.
- a light sphere at a known position+radius is assigned to exactly the clusters its AABB overlaps
  and NOT to far-away clusters.
- consistency: Σcount == lightIndices.size(); offsets == prefix sum.

## Seam
Zero new `vk*`/`MTL*`/`mtl::`/`Backend::Metal` symbols in protected dirs. The cluster math header
is pure `hf::math`. Backend changes live only in `engine/rhi_vulkan` + `engine/rhi_metal`; the
`rhi.h` interface additions are backend-neutral (`usesLightClusters` bool, `BindLightClusters`
virtual).
