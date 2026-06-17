# Slice DT — Virtual-Geometry Slice 2: GPU Per-Cluster Frustum Cull → Indirect Cluster Draw (Phase 6 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 2nd slice of the
> NANITE-style virtual-geometry arc (after DS meshlet decomposition): cull the DS clusters on the GPU by frustum
> (per-cluster bounding sphere) and draw only the survivors via MDI — the per-cluster analogue of the engine's
> existing per-OBJECT GPU cull (AR/CD). NO new RHI (a cluster is just an object with a per-cluster sphere; reuse
> the compute-compaction + MDI surface). Byte-identical GPU-culled==CPU-culled + exact-count proofs.
> Branch: slice-dt-clustercull. See [[hazard-forge-nanite-roadmap]].

**Goal:** Render an instance grid of DS-clustered spheres, cull on the GPU at the (instance×cluster) granularity by
frustum, and MDI-draw the survivors. Make-safe: a NEW header + NEW compute shader + NEW showcase + NEW golden;
nothing existing changes. Proven by GPU survivor-count==CPU-`frustum.h` reference (ReadBuffer), a byte-identical
GPU-culled==CPU-culled image, and an all-in-frustum==full-draw no-op.

## Reuse map (file:line — from the scout)
- **DS `engine/render/meshlet.h`** — `BuildMeshlets` → per-cluster `boundCenter`/`boundRadius` (the object-space
  bounding sphere DT culls) + the reordered index buffer (each cluster = `indices[3*triOffset .. 3*(triOffset+
  triCount))`) + `hashColor`.
- **`engine/render/gpu_cull.h`** — `InstanceWorldSphere(model, localCenter, localRadius)` (gpu_cull.h:37): transform
  a cluster's local sphere by its instance model matrix → world sphere for the frustum test. `CompactSurvivors`/
  `SurvivorCount` (gpu_cull.h:52,74) — the ordered prefix-sum compaction.
- **`engine/render/gpu_culled.h`** — `CulledObject` (gpu_culled.h:50) + `CullAndCompact` (:79): the pure-CPU mirror
  the test pins. DT's `cluster_cull.h` mirrors this STRUCTURALLY over cluster-instances.
- **`engine/render/frustum.h`** — `FromViewProj` (:70) + `SphereOutside` (:100): the cull predicate (Gribb-Hartmann,
  convention-correct both backends). Copy `SphereOutside` verbatim into the shader (the shared-math rule).
- **`shaders/gpudriven_cull.comp.hlsl`** — the GPU mirror: single-workgroup Hillis-Steele ordered prefix sum
  (:114-121) for deterministic source-index survivor order + `SphereOutside` (:82-88) + compaction into the draw-arg
  + per-draw SSBOs (:124-145). `cluster_cull.comp.hlsl` is a near-copy with cluster-instance records in,
  compacted `MdiCommand`s out.
- **`engine/render/mdi.h`** — `MdiCommand` (20-byte `VkDrawIndexedIndirectCommand`, mdi.h:34-46). A surviving
  cluster-instance → one `MdiCommand{ indexCount = triCount*3, instanceCount = 1, firstIndex = triOffset*3,
  vertexOffset = 0, firstInstance = clusterInstanceIndex }` (firstInstance carries the source index so the
  vertex/fragment shader fetches that cluster-instance's model matrix + hash color from an SSBO).
- **The MDI draw + readback RHI** — `DrawIndexedMultiIndirect(argsBuf, drawCount, stride)` (rhi.h:377),
  `BindStorageBuffer`/`DispatchCompute`/`ComputeToVertexBarrier` (rhi.h:411,423,425), `ReadBuffer` (rhi.h:613).
  ALL EXISTING — **NO new RHI**.
- **`engine/scene/instance_grid.h`** — `BuildInstanceGrid` (:17): the deterministic model-matrix grid.

## Design decisions (locked)

1. **`engine/render/cluster_cull.h` (NEW, pure CPU, `hf::render::vg`; 0 above-seam backend symbols).**
   - `struct ClusterInstance { uint32_t triOffset, triCount; math::Vec3 worldCenter; float worldRadius; uint32_t
     instanceIndex; };` — one per (instance × cluster); `worldCenter/Radius` = `InstanceWorldSphere(instanceModel,
     meshlet.boundCenter, meshlet.boundRadius)`. (sizeof std430-friendly; document padding.)
   - `BuildClusterInstances(span<const Mat4> instanceModels, const MeshletSet& mset) -> std::vector<ClusterInstance>`
     — for each instance i, for each meshlet m: push `{m.triOffset, m.triCount, InstanceWorldSphere(models[i],
     m.boundCenter, m.boundRadius), i}`. Deterministic order: instance-major, cluster-minor.
   - `CullClusterInstances(span<const ClusterInstance>, const Frustum&) -> std::vector<MdiCommand>` — mirror
     `gpu_culled.h::CullAndCompact`: walk in source order, for each `!frustum.SphereOutside(worldCenter,
     worldRadius)` emit an `MdiCommand{triCount*3, 1, triOffset*3, 0, clusterInstanceSourceIndex}` IN SOURCE ORDER
     (ordered compaction — matches the GPU prefix-sum order). `SurvivorClusterCount(...)` convenience.
   - Document: the bounding sphere is the DS-conservative one (never under-bounds) → the cull is conservative
     (never drops a visible cluster), exactly like the per-object cull.

2. **`shaders/cluster_cull.comp.hlsl` (NEW; near-copy of `gpudriven_cull.comp.hlsl`).** One thread per
   cluster-instance; bindings (SSBOs, no textures): the `ClusterInstance[]` (read), the frustum planes (push/UBO),
   the compacted `MdiCommand[]` out + the draw-count out (atomic or the single-workgroup ordered prefix-sum like
   gpudriven_cull). Copy `frustum::SphereOutside` VERBATIM (the shared-math rule). Use the SAME single-workgroup
   Hillis-Steele ordered prefix-sum compaction as `gpudriven_cull.comp` so the survivor order is the deterministic
   source-cluster-instance order (matches the CPU mirror byte-for-byte). `ComputePipelineDesc{ storageBufferCount,
   threadsPerGroupX = 64 }`. NO new RHI. Only `[[vk::binding]]` + `HF_MSL_GEN` above-seam.

3. **Showcase `--cluster-cull-shot <out>` (Vulkan) / `--cluster-cull` (Metal).** A small instance grid of clustered
   spheres: `BuildInstanceGrid` (e.g. a 4×1 row or 3×3 grid wider than the camera FOV) of `SphereGeometry(48,32)`
   (DS's 24 clusters each) → `BuildClusterInstances` → frustum from the showcase camera (framed so a CLEAR subset of
   cluster-instances is outside the frustum) → `cluster_cull.comp` → `DrawIndexedMultiIndirect` the survivors (the
   vertex shader fetches the model matrix + `hashColor` per draw via `firstInstance`/an SSBO; reuse the DS
   `meshlet_viz` shaders or a thin MDI variant). Declare the empty shadow pass per the DQ lesson → validation-clean.
   PROOFS (fail loudly):
   - **(1) GPU survivor count == CPU `frustum.h` reference:** `ReadBuffer` the GPU draw-count; CPU
     `SurvivorClusterCount` over the same cluster-instances + frustum → equal. Print `cluster-cull GPU count == CPU
     frustum.h: <N> EXACT`.
   - **(2) byte-identical GPU-culled == CPU-culled image:** render once via the GPU-compacted MDI args, render once
     via the CPU `CullClusterInstances` MdiCommands (draw exactly those, same shader) → the two captures
     byte-identical (SHA). Print `cluster-cull GPU-culled == CPU-culled: BYTE-IDENTICAL`.
   - **(3) all-in-frustum == full draw:** with a camera that contains ALL cluster-instances, the cull emits every
     cluster (count == total) and the image == drawing all cluster-instances unculled byte-identical.
   - **(4) determinism:** two runs byte-identical (ordered prefix sum → source order).
   - **Golden** = the culled grid render → `tests/golden/metal/cluster_cull.png` (the clustered spheres with the
     off-frustum clusters absent). Metal two runs DIFF 0.0000, gate on compare.sh EXIT CODE. Print `cluster-cull:
     {instances:M, clustersPerMesh:24, clusterInstances:M*24, survivors:N}`. Existing 83 image goldens UNTOUCHED.
     **GOLDEN DISCIPLINE: ONLY `tests/golden/metal/cluster_cull.png` — no loose `tests/golden/cluster_cull.png`.**

4. **Determinism / cross-backend.** `SphereOutside` is the shared `frustum.h` math (Gribb-Hartmann, convention-
   correct both backends via the single `Mat4::Perspective`); the ordered prefix sum gives source-index survivor
   order (no atomics-race nondeterminism); `InstanceWorldSphere` is plain matrix·vector + a max (use `std::fma`/
   `mad` if any multiply-add could contract — match CPU↔shader). Two runs byte-identical; the GPU==CPU count +
   GPU-culled==CPU-culled image proofs pin cross-backend correctness.

5. **Tests `tests/cluster_cull_test.cpp` (pure CPU; `hf_add_pure_test`):**
   - `BuildClusterInstances`: M instances × K clusters → M*K cluster-instances in instance-major order; each
     worldRadius ≥ the local radius scaled by the instance scale (conservative).
   - `CullClusterInstances`: a frustum containing all → all survive (count M*K, MdiCommands in source order); a
     frustum containing none → 0; a half-cut frustum → the exact `frustum.h::SphereOutside` subset, in source order.
   - `MdiCommand` fields: `indexCount==triCount*3`, `firstIndex==triOffset*3`, `firstInstance==sourceIndex`.
   - Determinism; conservative (a cluster whose sphere straddles a plane is KEPT). Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `virtual-geometry-cluster-cull` (features) + `--cluster-cull-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses `ComputePipelineDesc`/`BindStorageBuffer`/`DispatchCompute`/`ComputeToVertexBarrier`/`ReadBuffer`/
  `DrawIndexedMultiIndirect` (the AR/CB/CD/BM surface). New non-backend code (`cluster_cull.h`,
  `cluster_cull.comp.hlsl`, the test, the showcase) adds ZERO above-seam backend symbols. rhi.h + rhi_factory
  (dispatch baseline 2) + the backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — DU/DV)
Per-cluster Hi-Z occlusion (DU), cluster-LOD selection (DV), the visibility buffer / software raster (deferred
stretch arc). Per-cluster backface/normal-cone cull (a later refinement). ONE GPU per-cluster FRUSTUM cull with the
GPU==CPU count + GPU-culled==CPU-culled image + all-in-frustum no-op proofs and the culled-grid golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 83) + new `cluster_cull_test`. Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--cluster-cull-shot` on Vulkan: the grid renders with off-frustum clusters absent (coherent,
   the surviving clusters intact); `cluster-cull GPU count == CPU frustum.h: <N> EXACT` + `GPU-culled == CPU-culled:
   BYTE-IDENTICAL` + all-in-frustum==full-draw + two-run byte-identical; the `cluster-cull: {...}` line
   deterministic. Run under the AT Vulkan-validation gate → ZERO errors (compute→draw barriers SYNC-HAZARD-free;
   empty shadow pass declared).
3. Metal: `visual_test --cluster-cull` → new golden `tests/golden/metal/cluster_cull.png`; two runs DIFF 0.0000
   (gate on compare.sh EXIT CODE). The GPU==CPU count + GPU-culled==CPU-culled proofs also pass on Metal. **The
   implementer MUST wire BOTH the Vulkan (main.cpp) AND the Metal (visual_test.mm) showcase — confirm
   visual_test.mm is in the diff (the DS Metal-wiring lesson).**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `cluster_cull.png` added; the
   other 83 byte-identical. `git diff master --stat -- tests/golden` = ONLY `cluster_cull.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/cluster_cull.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+virtual-geometry-cluster-cull` + `--cluster-cull-shot`; introspect test updated.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `cluster_cull`
   image golden in the Mac round-trip loop AND `--cluster-cull-shot` in the `$vkShots` validation gate.
