# Slice CL — Clustered Light Culling (Forward+) (Phase 4 #37) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The many-lights
> flagship: a clustered (Forward+) light-culling pass scales the engine's point lighting to dozens/hundreds
> of lights, PROVEN byte-identical to brute-force all-lights shading. Render-invariant by construction.

**Goal:** Light many point lights efficiently. The view frustum is partitioned into a 3D cluster grid; a
compute pass assigns each light to the clusters its sphere of influence touches; the lit shader reads each
pixel's cluster and iterates ONLY that cluster's light list. Because a light is assigned to a cluster iff
its (windowed, hard-radius) sphere intersects the cluster AABB — and a light whose sphere misses a cluster
contributes EXACTLY ZERO to every pixel in it — the clustered-shaded image is BYTE-IDENTICAL to brute-force
all-lights shading. Proven: image SHA == brute-force SHA + the per-cluster light count > 1 (real culling
happened) + the GPU cluster assignment matches a CPU reference.

## Why this is render-invariant (the proof that makes it golden-safe)

Point lights use a WINDOWED attenuation with a hard cutoff: `atten(d) = (1/(d²+ε)) * window(d, radius)`
where `window` smoothly reaches EXACTLY 0 at `d == radius` (e.g. `clamp(1 - (d/radius)⁴, 0, 1)²`). So a
light contributes literally zero beyond `radius`. A cluster is a view-space AABB; a light is assigned to it
iff `SphereAABBIntersect(lightPosView, radius, clusterMin, clusterMax)`. If the sphere misses the cluster
AABB, the light is farther than `radius` from EVERY point in the cluster → zero contribution to every pixel
there. Therefore iterating only the cluster's assigned lights yields the IDENTICAL sum as iterating all
lights. Byte-identical. (This is the same airtight equivalence the GPU-cull slices use.)

## Design decisions (locked)

1. **Cluster math (engine/render/cluster.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::cluster`. Mirrors `frustum.h`/`hiz.h` (shared with the shader + `tests/cluster_test.cpp`).
   - `struct ClusterGrid { int dimX, dimY, dimZ; float zNear, zFar; };` — the cluster grid dimensions
     (e.g. 16×9×24) + the depth range. Document the depth slicing (exponential:
     `zSlice(k) = zNear * (zFar/zNear)^(k/dimZ)` — the standard cluster depth distribution; document).
   - `void ClusterViewAABB(const ClusterGrid&, int cx, int cy, int cz, const math::Mat4& invProj,
     int screenW, int screenH, math::Vec3& outMin, math::Vec3& outMax)` — the VIEW-SPACE AABB of cluster
     (cx,cy,cz): unproject the cluster's screen-tile corners at its near/far Z slices to view space and take
     the min/max. Document the convention (right-handed view space, −Z forward; pick + document).
   - `bool SphereAABBIntersect(const math::Vec3& center, float radius, const math::Vec3& aabbMin,
     const math::Vec3& aabbMax)` — standard closest-point test (`distance²(clamp(center,min,max), center) <=
     radius²`). Exact; shared with the shader.
   - `void AssignLights(const ClusterGrid&, const math::Mat4& proj, const math::Mat4& view, int screenW,
     int screenH, span<const PointLight> lights, std::vector<std::vector<uint32_t>>& outPerCluster)` — the
     CPU reference: for each cluster, the list of light indices whose VIEW-space sphere intersects the
     cluster AABB. Deterministic (fixed light set, ordered indices). Used by the test + the count reference.
   - `struct PointLight { math::Vec3 posWorld; float radius; math::Vec3 color; float intensity; };` — shared
     light struct (matches the shader's light SSBO layout; document the std430 packing).

2. **Light-assignment compute `shaders/cluster_assign.comp.hlsl`.** One thread per cluster (or per
   light-cluster pair): compute the cluster's view AABB (same math as `cluster::ClusterViewAABB`), test every
   light's view-space sphere (`SphereAABBIntersect`), and write the cluster's light index list into a packed
   SSBO (`clusterLightGrid[clusterIdx] = {offset, count}` + a flat `clusterLightIndices[]` filled via an
   atomic-allocated cursor — OR, since the light count is small + fixed, a fixed `MAX_LIGHTS_PER_CLUSTER`
   slot array per cluster with a deterministic ORDERED fill; PREFER the deterministic ordered fill — no
   atomics — so the per-cluster list order is identical to the CPU reference; document the choice + the cap +
   the conservative behavior if a cluster overflows the cap = NEVER drop a contributing light: size the cap
   so no cluster overflows for the showcase, and assert no overflow). Deterministic.

3. **Clustered lit shader `shaders/lit_clustered.frag.hlsl` (NEW variant).** A lit variant that, per pixel:
   reconstruct the view-space position (or use the existing G-buffer), compute the pixel's cluster index
   (screen tile from gl_FragCoord, Z slice from view depth — same `ClusterGrid` slicing), read
   `clusterLightGrid[cluster]`, and accumulate ONLY that cluster's point lights (windowed attenuation, same
   `PointLight` data) on top of the existing directional-sun + ambient/IBL. The default lit pass + its
   goldens stay BYTE-IDENTICAL (this is a NEW path behind the showcase flag). HLSL→SPIR-V→MSL via the
   existing toolchain. NO new RHI seam (light SSBO + cluster SSBO via the existing storage-buffer binding
   path used by gpu-driven/MDI).

4. **Showcase `--clustered-lights-shot <out>` (Vulkan) / `--clustered-lights` (Metal).** A scene (the
   ground plane + objects) lit by MANY point lights (e.g. 96 deterministically-placed colored point lights
   on a fixed lattice/spiral, fixed radii) + the directional sun. Pipeline: assign lights to clusters
   (compute) → clustered lit shade. Print `clustered-lights: {lights:96, clusters:DIMX*DIMY*DIMZ,
   maxPerCluster:M, avgPerCluster:A, brimByteIdentical:true}` where the rendered image is BYTE-IDENTICAL
   (SHA) to an INTERNAL brute-force render of the SAME scene iterating ALL 96 lights per pixel (no
   clustering) — assert + fail loudly on any diff (a culling bug = a visible lighting error). Also assert the
   GPU per-cluster assignment count total == the CPU `cluster::AssignLights` reference total. `maxPerCluster
   > 1` (real culling: most clusters see far fewer than 96 lights). New golden
   `tests/golden/metal/clustered_lights.png` (Metal two runs DIFF 0.0000). Existing 57 image goldens
   UNTOUCHED.

5. **Determinism.** Fixed light set (deterministic placement, no RNG — or fixed-seed integer LCG for the
   lattice, documented), fixed cluster grid, fixed camera, ordered cluster fill (no atomics → identical list
   order vs CPU ref). Two runs byte-identical.

6. **Tests `tests/cluster_test.cpp` (pure CPU, no GPU):**
   - **Sphere-AABB intersect:** known inside/touching/outside cases (a sphere centered in the AABB → true;
     a sphere far outside → false; a sphere just touching a face/edge/corner → true at the boundary).
   - **Cluster AABB:** `ClusterViewAABB` for a known grid+proj produces AABBs that tile the frustum (adjacent
     clusters share a face; the near slice is closer than the far slice; the grid covers the screen).
   - **Assignment correctness (the invariance core):** for a known light set + grid, `AssignLights` assigns a
     light to a cluster IFF its sphere intersects that cluster's AABB; a light is in ≥1 cluster; a tiny-radius
     light is in few clusters, a huge-radius light in many; NO contributing cluster is missed (brute-force
     per-cluster check parity → 0 false-negatives, the safety property).
   - **Real culling:** with many lights of moderate radius, `maxPerCluster < totalLights` (clustering
     actually reduces the per-cluster light count) AND every cluster's list is a subset of all lights.
   - **Determinism:** same inputs → same assignment (same lists, same order).
   - Clean under `windows-msvc-asan`.

7. **Introspect.** Add exactly `clustered-light-culling` (features) + `--clustered-lights-shot` (showcases).

## RHI seam additions (summary)
- **None.** The cluster-assign compute + clustered lit shader reuse the existing storage-buffer (SSBO) +
  compute-dispatch path (as used by gpu-driven cull / MDI / bindless). New non-backend files
  (`engine/render/cluster.h`, `shaders/cluster_assign.comp.hlsl`, `shaders/lit_clustered.frag.hlsl`,
  `tests/cluster_test.cpp`) add ZERO above-seam backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Spot lights / area lights / light cones (point lights only — the math generalizes later), shadowed point
lights (shadow atlas/cubemaps — future), tiled deferred (this is clustered forward), 2.5D tiled (use full 3D
clusters), light-list compaction beyond the fixed cap, per-cluster depth-bounds refinement from the actual
depth buffer (use the analytic cluster Z slices — conservative + simpler + still byte-identical), more than
a few hundred lights, dynamic/animated lights (fixed set, fixed-time). One fixed-light-set clustered
Forward+ pass, byte-identical to brute-force, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 57) + new `cluster_test` (sphere-AABB, cluster AABB
   tiling, assignment correctness + no-false-negative parity, real-culling reduction, determinism). Clean
   under `windows-msvc-asan`.
2. **Invariance proof (render-invariance + count):** `--clustered-lights-shot` on Vulkan renders the
   clustered Forward+ shade; the captured image is BYTE-IDENTICAL (SHA) to the INTERNAL brute-force
   all-96-lights render; `maxPerCluster > 1` and `maxPerCluster < 96` (real culling); the GPU assignment
   total == the CPU `cluster::AssignLights` reference. Prints `clustered-lights: {lights:96, clusters:N,
   maxPerCluster:M, avgPerCluster:A, brimByteIdentical:true}`. Two runs identical.
3. `--clustered-lights-shot` visual review (controller): the scene shows many colored point-light pools on
   the ground/objects (a rich many-lights look), coherent with the sun + ambient. Run under the AT
   Vulkan-validation gate → ZERO errors.
4. Metal: `visual_test --clustered-lights` → new golden `tests/golden/metal/clustered_lights.png`; two runs
   DIFF 0.0000.
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `clustered_lights.png`
   added; the other 57 byte-identical.
6. Introspect JSON rebaked exactly `+clustered-light-culling` + `--clustered-lights-shot`; introspect test
   updated; no other drift.
7. Seam grep clean (no new above-seam code symbols). `scripts/verify.ps1` updated to include the new
   `clustered_lights` image golden in the Mac round-trip loop.
