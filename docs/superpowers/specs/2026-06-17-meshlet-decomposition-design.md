# Slice DS — Virtual-Geometry Slice 1: Meshlet/Cluster Decomposition (BEACHHEAD) (Phase 6 #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. **The BEACHHEAD of the
> NANITE-STYLE VIRTUAL-GEOMETRY flagship** (the 2nd flagship toward literal-UE5 parity — virtualized massive-detail
> geometry). DS establishes the cluster data structure + a deterministic mesh→meshlet decomposition + a per-cluster
> colored viz golden that every subsequent slice (DT per-cluster GPU cull, DU Hi-Z occlusion, DV cluster-LOD)
> builds on. Pure CPU, integer-deterministic, NO new RHI, NO GPU math. Namespace `hf::render::vg` (NOT `cluster`
> — that's taken by Forward+ LIGHT clustering). See [[hazard-forge-nanite-roadmap]]. Branch: slice-ds-meshlet.

**Goal:** Add `engine/render/meshlet.h` (pure-CPU, `hf::render::vg`) that partitions a mesh's index buffer into
~128-triangle clusters with conservative per-cluster bounds, and a `--meshlet-viz` showcase that renders a sphere
segmented into flat per-cluster colors (the visible cluster structure). Make-safe by construction: a NEW header +
NEW shader + NEW showcase + NEW golden — nothing existing changes. The decomposition is integer sort + index copy
+ min/max only → bit-identical on every platform and every run (the simplest possible proof surface for a
beachhead).

## Reuse map (file:line — grounded by the scout)
- **`engine/scene/mesh.h`** — `MeshGeometry{std::vector<Vertex> verts; std::vector<uint32_t> indices}` (mesh.h:15)
  is the decomposition INPUT; `CubeGeometry()` / `SphereGeometry(segments,rings)` (mesh.h:19-20) give deterministic
  procedural input meshes; `MeshBounds` AABB (mesh.h:24).
- **`engine/scene/vertex.h`** — `Vertex` (stride 56: pos/color/uv/normal/tangent, vertex.h:8). Decomposition reads
  only `Vertex::pos`.
- **`engine/math`** — `Vec3`, min/max. NO transcendentals (Morton quantization is integer; bounds are min/max).
- **The existing bound-draw loop / per-draw push-constant path** in `samples/hello_triangle/main.cpp` — the viz
  draws each meshlet as an index sub-range with a per-cluster color push constant (the existing draw surface; NO
  new RHI). Model the showcase on an existing simple lit/flat showcase (e.g. the scene-shadow / instanced path).
- **The shader↔CPU-header verbatim-copy convention** — `hashColor(meshletIndex)` is copied verbatim from the
  header into `meshlet_viz.frag.hlsl` (or supplied as the push-constant color computed CPU-side; pick the simplest
  — PREFER computing the color CPU-side and passing it as the push constant, so the shader is a trivial flat-color
  pass and there's no GPU-side hash to keep bit-exact).

## Design decisions (locked)

1. **`engine/render/meshlet.h` (NEW, pure CPU, namespace `hf::render::vg`; 0 above-seam backend symbols — only
   seam-discipline doc comments may mention vk/MTL).**
   ```cpp
   struct Meshlet {
       uint32_t   triOffset;     // first triangle (into MeshletSet::indices, /3)
       uint32_t   triCount;      // tris in this cluster (<= kMaxTrisPerCluster)
       math::Vec3 boundMin, boundMax;   // object-space AABB over the cluster's vertices
       math::Vec3 boundCenter;   // 0.5*(boundMin+boundMax)
       float      boundRadius;   // max |v - boundCenter| over the cluster's verts (conservative sphere)
   };
   struct MeshletSet {
       std::vector<Meshlet>  meshlets;
       std::vector<uint32_t> indices;   // REORDERED index buffer, clusters contiguous;
                                        // meshlet i covers indices[3*triOffset .. 3*(triOffset+triCount))
   };
   static constexpr uint32_t kMaxTrisPerCluster = 128;
   ```
   `MeshletSet BuildMeshlets(std::span<const Vertex> verts, std::span<const uint32_t> indices)` — the deterministic
   spatial-greedy sweep:
   - `T = indices.size() / 3` (any trailing 1-2 indices are IGNORED — document this explicitly). `T == 0` → return
     an empty `MeshletSet{}` (the no-op).
   - Per triangle, centroid `c = (p0+p1+p2) * (1/3)` (ordinary FP; centroid is used ONLY as a sort key, never as
     cross-backend GPU math). Compute the mesh AABB over all referenced vertices.
   - **Quantize** each centroid into the mesh AABB to a 10-bit-per-axis integer `(qx,qy,qz) ∈ [0,1023]`
     (`q = clamp(floor((c-aabbMin)/(aabbMax-aabbMin) * 1024), 0, 1023)`; guard a zero-extent axis → q=0), interleave
     to a 30-bit **Morton code**. **Sort the triangle list by `(mortonCode, originalTriIndex)`** — the secondary key
     guarantees a TOTAL order (no ties → fully deterministic; this is the deliberate choice over float-greedy
     adjacency, which risks tie-order nondeterminism). Provide `MortonCode10(qx,qy,qz)` as a documented bit-interleave
     helper (unit-tested).
   - Walk the sorted triangles, emitting clusters of `kMaxTrisPerCluster` consecutive triangles (the last may be
     partial). For each cluster, append its `3*triCount` indices CONTIGUOUSLY into `MeshletSet::indices` and record
     `triOffset` (running, in triangles) + `triCount`.
   - Per cluster, compute the AABB over its referenced vertices → `boundMin/Max`; `boundCenter = 0.5*(min+max)`;
     `boundRadius = max over the cluster's verts of length(v - boundCenter)` (conservative; DT/DU rely on this exactly
     like `frustum.h`/`hiz.h` rely on their conservative contracts — do NOT under-bound).
   - Everything is integer sort + index copy + min/max + one sqrt for the radius (the radius is host-only viz/cull
     metadata, not cross-backend GPU-shared, so libm sqrt is fine; document). Bit-identical every run/platform.

2. **`shaders/meshlet_viz.frag.hlsl` (+ vert) — flat per-cluster color.** The simplest path: the showcase draws each
   meshlet as an index sub-range with a per-cluster color PUSH CONSTANT computed CPU-side (`hashColor(i)` in the
   header → RGB), so the fragment shader just outputs the pushed flat color (optionally × a fixed Lambert from the
   vertex normal for shape readability — keep it deterministic, no lights/shadows/RNG). `hashColor(uint32_t i)` is a
   documented integer hash (e.g. a fixed multiply-xor-shift → 3 bytes → [0,1] RGB) in `meshlet.h`, unit-tested for
   determinism + spread. Reuse an existing vert (e.g. the lit/instanced vert); do NOT edit any existing shader.

3. **Showcase `--meshlet-viz <out>` (Vulkan) / `--meshlet-viz` (Metal).** Scene: a single `SphereGeometry(48, 32)`
   centered + a fixed camera (deterministic, no clock/RNG), lit flat. `BuildMeshlets` over the sphere → upload
   `MeshletSet::indices` as the index buffer → draw each meshlet as a sub-range (`firstIndex = 3*triOffset`,
   `indexCount = 3*triCount`) with its `hashColor(i)` push constant. Result: the sphere segmented into flat-colored
   cluster patches (the visible cluster structure). Print `meshlet: {tris:T, clusters:N, maxTris:128}`.
   - **Golden** `tests/golden/metal/meshlet_viz.png` (the segmented sphere). Metal two runs DIFF 0.0000, gate on the
     compare.sh EXIT CODE. Cross-backend-identical (same verts, same projection, color = pure integer hash → identical
     bytes). Existing 82 image goldens UNTOUCHED.
   - Print the proof lines: `meshlet partition: COMPLETE (ΣtriCount==T, every tri once)`, `meshlet determinism:
     two builds BYTE-IDENTICAL`.

4. **Determinism / cross-backend.** Integer Morton sort + index copy + min/max → identical on every platform/run.
   No GPU math (the viz draws existing vertices with a flat pushed color). The golden is cross-backend-identical by
   construction (verify + report honestly).

5. **Tests `tests/meshlet_test.cpp` (pure CPU; `hf_add_pure_test(meshlet_test)`):**
   - **Partition completeness:** the multiset of all (reordered) cluster triangles' index-triples == the original
     index multiset (every triangle covered exactly once, none invented/dropped); `Σ meshlet.triCount == T`; each
     `triCount <= kMaxTrisPerCluster` and only the LAST cluster may be partial.
   - **Determinism:** `BuildMeshlets` twice → `memcmp` the `meshlets` array AND the `indices` array bit-identical.
   - **Conservative bounds:** for every meshlet, every referenced vertex lies inside `[boundMin,boundMax]` AND within
     `boundRadius` of `boundCenter` (the conservative contract DT/DU depend on).
   - **Degenerate/empty no-op:** `BuildMeshlets({},{})` → 0 meshlets, 0 indices; a 1-triangle mesh → exactly 1
     meshlet (triCount 1); an index buffer whose length isn't divisible by 3 → trailing 1-2 indices ignored, `T =
     len/3` triangles partitioned (documented + asserted).
   - **MortonCode10 + hashColor:** the bit-interleave round-trips the 3 axes; hashColor is deterministic + spreads
     (distinct adjacent indices → distinct colors).
   - **Stress:** `CubeGeometry()` + several `SphereGeometry` tessellations → completeness + determinism +
     conservative bounds hold for each. Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `virtual-geometry-meshlets` (features) + `--meshlet-viz` (showcases).

## RHI seam additions (summary)
- **None.** Pure-CPU decomposition + a flat-color viz over the EXISTING draw + push-constant surface. New non-backend
  code (`meshlet.h`, `meshlet_viz.frag.hlsl`, `meshlet_test.cpp`, the showcase) adds ZERO above-seam backend symbols.
  rhi.h + rhi_factory (dispatch baseline 2) + the backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — DT and beyond)
GPU per-cluster cull (DT), Hi-Z occlusion (DU), cluster-LOD selection (DV), the visibility buffer / software raster /
cluster-DAG simplification / streaming (the deferred stretch arc). Adjacency-based region-growing meshlet build (the
Morton spatial sweep is the deterministic beachhead; adjacency is a later refinement, not needed for the cluster data
structure + viz). Vertex de-duplication / per-cluster local vertex buffers (DS reuses the shared vertex buffer +
reorders only indices — sufficient for the cull/LOD arc). ONE deterministic mesh→meshlet decomposition with a
completeness + determinism + conservative-bounds + degenerate proof set and a per-cluster-colored viz golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 82) + new `meshlet_test`. Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--meshlet-viz` on Vulkan: the sphere is segmented into coherent flat-colored cluster
   patches (each cluster a contiguous spatial patch — proves the Morton spatial coherence); `meshlet partition:
   COMPLETE` + `meshlet determinism: two builds BYTE-IDENTICAL`; the `meshlet: {...}` line deterministic. Run under
   the AT Vulkan-validation gate → ZERO errors (declare the empty shadow pass per the DQ lesson if the viz runs the
   lit pass).
3. Metal: `visual_test --meshlet-viz` → new golden `tests/golden/metal/meshlet_viz.png`; two runs DIFF 0.0000 (gate
   on the compare.sh EXIT CODE).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `meshlet_viz.png` added; the
   other 82 byte-identical. `git diff master --stat -- tests/golden` = ONLY `meshlet_viz.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/meshlet_viz.png` (the DK stray trap), NO other golden changed.
5. Introspect JSON rebaked exactly `+virtual-geometry-meshlets` + `--meshlet-viz`; introspect test updated; no other
   drift.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI; no existing shader/pipeline changed). `scripts/verify.ps1` updated
   to include the new `meshlet_viz` image golden in the Mac round-trip loop AND `--meshlet-viz` in the `$vkShots`
   validation gate (the DQ hardening — every new showcase that records a draw joins the gate).
