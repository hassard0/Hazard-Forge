# Slice DU — Virtual-Geometry Slice 3: GPU Per-Cluster Hi-Z Occlusion Cull (Phase 6 #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 3rd slice of the NANITE
> virtual-geometry arc (after DS decomposition, DT frustum cull): drop clusters fully HIDDEN behind closer
> geometry via a Hi-Z occlusion test, on top of DT's frustum cull. The per-cluster analogue of the engine's
> existing per-OBJECT Hi-Z cull (Slice CJ `--hiz-cull`); `hiz.h` is drop-in. NO new RHI. Conservative → the
> occlusion-culled render is byte-identical to the frustum-only render. Branch: slice-du-clusterhiz.
> See [[hazard-forge-nanite-roadmap]].

**Goal:** Add a per-cluster Hi-Z occlusion test to DT's cluster cull: a depth pre-pass renders an occluder + the
clusters, a Hi-Z pyramid is built from the read-back depth, and the cluster cull additionally drops any
cluster-instance whose bounding AABB is fully behind the Hi-Z (occluded). Make-safe: a NEW shader/showcase/golden;
the occlusion is CONSERVATIVE (only culls clusters guaranteed fully hidden) so the occlusion-on render is
BYTE-IDENTICAL to the frustum-only (DT) render, and an occlusion-DISABLED path is byte-identical to DT.

## Reuse map (file:line)
- **DT `engine/render/cluster_cull.h`** — `ClusterInstance` + `BuildClusterInstances` + `CullClusterInstances`
  (frustum). DU extends the cull with the occlusion test (a 2nd predicate after `SphereOutside`).
- **`engine/render/hiz.h`** — `BuildHiZ(const float* depth, int w, int h, std::vector<HiZMip>& mips)` (hiz.h:60)
  and `IsOccluded(aabbMin, aabbMax, viewProj, screenW, screenH, std::span<const HiZMip> mips) -> bool` (hiz.h:104).
  Conservative (near-plane straddle / off-screen / partially-clipped → KEEP). Copy the GPU mirror from
  `shaders/hiz_cull.comp.hlsl`.
- **Slice CJ `--hiz-cull` showcase** (`samples/hello_triangle/main.cpp` + `metal_headless/visual_test.mm` +
  `shaders/hiz_cull.comp.hlsl`) — the TEMPLATE: depth pre-pass → `ReadRenderTarget`/`ReadBuffer` the depth → flat
  `float[w*h]` → `BuildHiZ` → upload the Hi-Z mips as an SSBO → the cull compute samples them via `IsOccluded`.
  DU does the SAME but the cull units are DT's cluster-instances (not objects). Mirror CJ's depth-prepass +
  Hi-Z-build + the SSBO upload exactly.
- **`shaders/cluster_cull.comp.hlsl` (DT)** — the cluster frustum-cull compute. DU's `cluster_hiz_cull.comp.hlsl`
  is DT's shader + the `hiz_cull.comp.hlsl` occlusion test (the verbatim `IsOccluded` over the cluster AABB + the
  Hi-Z mip SSBO), gated by an `occlusionEnabled` flag.

## Design decisions (locked)

1. **Cluster AABB for the occlusion test.** Add a conservative world AABB per cluster-instance — the bounding
   sphere's AABB `[worldCenter - worldRadius, worldCenter + worldRadius]` (extend `ClusterInstance` with
   `worldAabbMin/Max`, OR compute it inline in the cull from `worldCenter ± worldRadius`). This is conservative
   (encloses the cluster) → `IsOccluded` over it never false-culls a visible cluster. Document that DU uses the
   sphere-AABB (a tighter oriented per-cluster AABB is a later refinement, YAGNI).

2. **`engine/render/cluster_cull.h` extension (pure CPU, `hf::render::vg`).** Add
   `CullClusterInstancesHiZ(span<const ClusterInstance>, const Frustum&, const Mat4& viewProj, int w, int h,
   span<const hiz::HiZMip> mips, bool occlusionEnabled) -> std::vector<MdiCommand>` — the DT frustum cull PLUS:
   for each frustum-surviving cluster, if `occlusionEnabled && hiz::IsOccluded(aabbMin, aabbMax, viewProj, w, h,
   mips)` → DROP it; else emit the MdiCommand (same source-ordered compaction). `occlusionEnabled=false` →
   identical to DT's `CullClusterInstances` (the disabled-path guarantee). `SurvivorClusterCountHiZ(...)`
   convenience. This is the CPU mirror the GPU compute matches byte-for-byte.

3. **`shaders/cluster_hiz_cull.comp.hlsl` (NEW; DT's `cluster_cull.comp` + CJ's `hiz_cull.comp` occlusion).**
   One thread per cluster-instance; bindings: `ClusterInstance[]` (read), the Hi-Z mip SSBO (read, same layout CJ
   uploads), frustum planes + viewProj + screenW/H + `occlusionEnabled` (push/UBO), compacted `MdiCommand[]` + count
   out. Copy `frustum::SphereOutside` + `hiz::IsOccluded` VERBATIM (the shared-math rule). SAME single-workgroup
   ordered prefix-sum compaction (deterministic source order). NO new RHI (reuses the DT/CJ compute + SSBO surface).

4. **Showcase `--cluster-hiz-shot <out>` (Vulkan, main.cpp) AND `--cluster-hiz` (Metal, visual_test.mm — WIRE
   BOTH; confirm visual_test.mm in the diff, the DS lesson).** Scene: an OCCLUDER (a large opaque quad/wall) in
   FRONT of a back row of DT-clustered spheres, plus some spheres NOT occluded, so some cluster-instances are fully
   hidden by the occluder. Pipeline: depth pre-pass (render occluder + clusters) → `ReadRenderTarget` depth → flat
   float → `BuildHiZ` → upload Hi-Z SSBO → `cluster_hiz_cull.comp` → `DrawIndexedMultiIndirect` survivors (the
   occluder is drawn in the color pass in BOTH the occlusion-on and frustum-only renders). Declare the empty shadow
   pass (DQ lesson) → validation-clean. PROOFS (fail loudly):
   - **(1) occlusion-culled == frustum-only BYTE-IDENTICAL:** the occlusion-ON render (occluded clusters dropped)
     is byte-identical (SHA) to the frustum-only (occlusion-OFF, == DT) render — because the dropped clusters were
     100% hidden behind the occluder (the conservative Hi-Z only culls fully-occluded clusters). Print
     `cluster-hiz occlusion-culled == frustum-only: BYTE-IDENTICAL`.
   - **(2) survivor count DROPS:** occlusion-ON survivor count < occlusion-OFF (frustum-only) count — a measurable
     occlusion happened (ReadBuffer both counts). Print `cluster-hiz occluded: <Noff> -> <Non> (<dropped> culled)`.
   - **(3) occlusion-disabled == DT:** `occlusionEnabled=false` → byte-identical to the DT frustum-only cull of the
     same scene (the disabled-path no-op).
   - **(4) GPU==CPU count + GPU-culled==CPU-culled:** the GPU `cluster_hiz_cull` survivor count == CPU
     `SurvivorClusterCountHiZ`; the GPU-culled render == the CPU-`CullClusterInstancesHiZ` render byte-identical.
   - **(5) determinism:** two runs byte-identical.
   - **Golden** = the occlusion-culled render → `tests/golden/metal/cluster_hiz.png` (the occluder + the visible
     clusters; the hidden back clusters culled but invisible anyway → looks like the frustum-only image, which is
     the POINT — the proof is the count drop + byte-identity). Metal two runs DIFF 0.0000, gate on compare.sh EXIT.
     Print `cluster-hiz: {clusterInstances:M*24, frustumSurvivors:Noff, hizSurvivors:Non}`. Existing 83 image
     goldens UNTOUCHED. **GOLDEN DISCIPLINE: ONLY `tests/golden/metal/cluster_hiz.png`.**

5. **Determinism / cross-backend.** `IsOccluded` + `SphereOutside` are shared header math (copied verbatim into the
   shader); the Hi-Z is built from the read-back depth (host BuildHiZ on the same depth bytes the GPU sampled →
   bit-exact, the CJ discipline); ordered prefix sum → source order. The CJ `--hiz-cull` golden already proves the
   Hi-Z build + IsOccluded are convention-correct + cross-backend-stable on both backends; DU reuses that exact
   machinery. (As with DT, the Metal showcase MAY render the survivor set via the CPU bound path per the
   gpudriven/hiz convention — the image is backend-identical and the Vulkan side carries the GPU==CPU proof; pick
   the same convention CJ's Metal `--hiz-cull` uses and document it.)

6. **Tests `tests/cluster_hiz_test.cpp` (pure CPU; `hf_add_pure_test`):**
   - `CullClusterInstancesHiZ` with `occlusionEnabled=false` == `CullClusterInstances` (DT) byte-identical (the
     disabled-path guarantee).
   - With a Hi-Z where a cluster's AABB is fully behind the depth → that cluster is DROPPED; a cluster in front →
     KEPT. Conservative: a near-plane-straddling / off-screen cluster is KEPT (IsOccluded returns false).
   - Survivor count with occlusion ≤ frustum-only count; the dropped set ⊆ the frustum-surviving set.
   - Determinism; MdiCommand fields preserved. Clean under `windows-msvc-asan`.

7. **Introspect.** Add exactly `virtual-geometry-cluster-hiz` (features) + `--cluster-hiz-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses the DT/CJ depth-prepass + `ReadRenderTarget`/`ReadBuffer` + compute-compaction + MDI surface +
  `hiz.h` (header). New non-backend code (`cluster_cull.h` extension, `cluster_hiz_cull.comp.hlsl`, the test, the
  showcase) adds ZERO above-seam backend symbols. rhi.h + rhi_factory (dispatch baseline 2) + the backend dirs
  UNCHANGED. Report the seam.

## Out of scope (YAGNI — DV)
Cluster-LOD selection (DV), oriented/tight per-cluster AABBs (the sphere-AABB is conservative + sufficient), a
two-phase Hi-Z (cull→draw→rebuild→cull-again, the real Nanite two-pass — a later refinement), the visibility
buffer. ONE GPU per-cluster Hi-Z occlusion cull with occlusion-culled==frustum-only byte-identical +
occlusion-disabled==DT + survivor-count-drops + GPU==CPU proofs and the occluded-grid golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 83) + new `cluster_hiz_test`. Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--cluster-hiz-shot` on Vulkan: the occluder + visible clusters render coherently;
   `cluster-hiz occlusion-culled == frustum-only: BYTE-IDENTICAL` + `cluster-hiz occluded: Noff -> Non` (a real
   drop) + `occlusionEnabled=false == DT` + GPU==CPU count + GPU-culled==CPU-culled + two-run byte-identical; the
   `cluster-hiz: {...}` line deterministic. Run under the AT Vulkan-validation gate → ZERO errors (empty shadow
   pass declared).
3. Metal: `visual_test --cluster-hiz` → new golden `tests/golden/metal/cluster_hiz.png`; two runs DIFF 0.0000 (gate
   on compare.sh EXIT CODE). **Confirm visual_test.mm is in the diff (the DS Metal-wiring lesson).**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `cluster_hiz.png` added; the
   other 83 byte-identical. `git diff master --stat -- tests/golden` = ONLY `cluster_hiz.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/cluster_hiz.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+virtual-geometry-cluster-hiz` + `--cluster-hiz-shot`; introspect test updated.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `cluster_hiz`
   image golden in the Mac round-trip loop AND `--cluster-hiz-shot` in the `$vkShots` validation gate.
