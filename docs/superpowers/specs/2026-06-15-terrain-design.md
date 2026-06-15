# Slice BF — Procedural Terrain / Heightmap (Phase 4 #9) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A flagship
> open-world capability that pairs with BD streaming.

**Goal:** Render a procedural terrain patch. A deterministic heightmap function drives a generated mesh
(NxN grid, vertices displaced by height, normals from finite differences), rendered through the existing
lit/PBR scene path. A `--terrain-shot` showcase shows rolling lit/shadowed terrain; golden-verified on
both backends.

## Why procedural (determinism, no asset)

A heightmap loaded from a file adds an asset + decode dependency. Instead the height is a fixed pure
function `Height(x, z)` (e.g. a sum of a few sines + a deterministic integer value-noise — NO RNG/clock).
Same function ⇒ same mesh ⇒ same pixels ⇒ goldens match. (Loading real heightmap images is a future
slice.)

## Design decisions (locked)

1. **Heightmap + mesh gen (engine/terrain/heightmap.{h,cpp}, pure CPU, no RHI/backend symbols).**
   Namespace `hf::terrain`. Reuses `engine/math` + the scene `Mesh`/vertex types.
   - `float Height(float x, float z)` — deterministic: e.g. `A1*sin(f1*x)*cos(f1*z) + A2*sin(f2*x+...)`
     + a small fixed integer value-noise term (hash(ix,iz) → lattice value, bilinear). Document the exact
     formula + constants. Pure, no RNG/clock/float-time.
   - `TerrainMesh BuildTerrain(int n, float worldSize, float heightScale)` — an `n x n` vertex grid over
     `[-worldSize/2, +worldSize/2]^2` in XZ; each vertex `y = Height(x,z) * heightScale`; UVs = grid
     coords; **normals from central finite differences** of `Height` (`N = normalize(cross(dz, dx))`
     using neighbor heights); indices = two triangles per quad. Returns scene vertex/index buffers
     compatible with the existing lit/PBR mesh path. Deterministic vertex/index counts: `n*n` verts,
     `(n-1)*(n-1)*6` indices.

2. **Render reuses the existing lit/PBR scene path.** The terrain mesh is uploaded as a normal scene mesh
   and drawn lit + shadowed (directional light + the shadow map) with a simple material (a height/slope
   tint OR the existing checker — pick what reads clearly; a height-based color ramp makes the relief
   legible). NO new RHI, NO new shader REQUIRED (reuse lit/PBR); if a tiny height-tint helps legibility,
   do it in the material params, not a new shader. Fixed deterministic camera framing the terrain.

3. **Showcase `--terrain-shot <out>` (Vulkan) / `--terrain` (Metal).** Build a terrain patch (e.g.
   n=128, worldSize=20, heightScale tuned so hills are clearly visible but the camera sees the relief),
   render one lit+shadowed frame from a fixed ¾ camera. Print `terrain: {n:128, verts:16384,
   tris:32258, peak:P, ...}` (P = max height, deterministic). New golden `tests/golden/metal/terrain.png`
   (Metal two runs DIFF 0.0000). Existing 34 image goldens UNTOUCHED.

4. **Tests `tests/terrain_test.cpp` (pure CPU, no GPU):**
   - **Height determinism:** `Height(x,z)` returns the same value across runs; a few hand-anchored sample
     points match expected values (within a tiny epsilon if float, or document an exact-value check).
   - **Mesh structure:** `BuildTerrain(n,...)` yields exactly `n*n` verts + `(n-1)*(n-1)*6` indices; every
     index is in range; corner/edge vertices at the expected world positions; vertex Y == Height*scale at
     a sample vertex.
   - **Normals:** a vertex on a flat region (constant Height) has normal ~+Y; a vertex on a known slope
     has the expected tilt direction (finite-difference normal sanity); all normals are unit length.
   - **Determinism:** two `BuildTerrain` calls produce bit-identical buffers.
   - Clean under `windows-msvc-asan`.

5. **Introspect.** Add exactly `procedural-terrain` (features) + `--terrain-shot` (showcases).
   One-pattern rebake, no other drift.

## RHI seam additions (summary)
- **None.** Terrain gen is pure CPU; rendering reuses the existing lit/PBR + shadow path. New files
  (`engine/terrain/heightmap.{h,cpp}`, `tests/terrain_test.cpp`) add ZERO backend symbols. Seam grep
  stays at baseline (2).

## Out of scope (YAGNI)
LOD / geometry clipmaps / quadtree tessellation, GPU tessellation/displacement, multi-material terrain
splatting, heightmap-image loading, terrain physics/collision, streaming-tile integration with BD,
erosion/procedural-generation pipelines, vegetation scattering, holes/caves. One deterministic
single-resolution procedural patch, finite-difference normals, lit/shadowed, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 33) + new `terrain_test` (height determinism +
   sample values, mesh structure/counts, normals, determinism). Clean under `windows-msvc-asan`.
2. `--terrain-shot` on Windows/Vulkan: controller visual review — rolling terrain relief, lit + shadowed,
   coherent (hills/valleys read clearly); the `terrain: {...}` stat line is deterministic (two runs →
   byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --terrain` → new golden `tests/golden/metal/terrain.png`; two runs DIFF 0.0000;
   the terrain stat line matches Vulkan.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `terrain.png` added;
   the other 34 byte-identical.
5. Introspect JSON rebaked exactly `+procedural-terrain` + `--terrain-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `terrain` image
   golden in the Mac round-trip loop.
