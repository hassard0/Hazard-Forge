# Slice DV — Virtual-Geometry Slice 4: Discrete Cluster-LOD Selection by Screen-Space Error (Phase 6 #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The FINAL slice of the NANITE
> virtual-geometry CORE (DS decomposition → DT frustum cull → DU Hi-Z occlusion → DV LOD): per cluster-group, pick
> among 2-3 pre-baked discrete LOD levels by PROJECTED SCREEN-SPACE ERROR, so near geometry draws full detail and
> far geometry draws a coarser LOD — the Nanite LOD primitive (the screen-error-driven detail selection), kept
> deterministic + bit-exact via pre-baked LODs (no runtime simplifier — that hard, non-deterministic-prone DAG
> simplification is the deferred stretch arc). NO new RHI. Disabled-path force-LOD0 → byte-identical to the
> full-detail render. Branch: slice-dv-clusterlod. See [[hazard-forge-nanite-roadmap]].

**Goal:** Render a field of instances at varying distance; each instance's projected screen-space error selects a
discrete LOD (one of 2-3 pre-baked tessellations of the mesh, each DS-cluster-decomposed), drawn via the cluster
MDI path; near instances draw LOD0 (full detail), far instances draw a coarser LOD. Make-safe: a NEW header + NEW
compute shader + NEW showcase + NEW golden; the LOD select is a bit-exact integer function (`std::fma` +
host-precomputed projection scalar → GPU==CPU bit-exact), and a `force-LOD0` / `errorThreshold=0` path selects LOD0
for every instance → byte-identical to the full-detail render.

## Reuse map (file:line)
- **DS `engine/render/meshlet.h`** — `BuildMeshlets` (each pre-baked LOD mesh is DS-decomposed into clusters) +
  `hashColor`.
- **DT `engine/render/cluster_cull.h`** — `ClusterInstance` / `MdiCommand` emission / the cluster MDI draw path
  (the LOD-selected clusters draw exactly as DT's survivors do). The DT `--cluster-cull` showcase (Vulkan +
  Metal) is the draw template.
- **`engine/scene/mesh.h`** — `SphereGeometry(segments, rings)` at 3 tessellations = the 3 pre-baked LODs
  (deterministic procedural meshes; no simplifier needed).
- **`engine/render/frustum.h` / `Mat4::Perspective`** — the projection; `screenH / (2·tan(fovY/2))` is the
  host-precomputed projection scalar for the screen-error.
- **The DH FP discipline** — `std::fma` + the host-precomputed projection scalar so the screen-error → LOD-index
  integer is bit-exact CPU↔Vulkan↔Metal (transcendental `tan(fovY/2)` is host-precomputed once, uploaded as exact
  bits; the per-instance math is `error · (1/dist) · projScale` via `fma`, no per-instance transcendental).

## Design decisions (locked)

1. **`engine/render/cluster_lod.h` (NEW, pure CPU, `hf::render::vg`; 0 above-seam backend symbols).**
   - `struct LodLevel { uint32_t firstCluster, clusterCount; float geometricError; };` — a discrete LOD's cluster
     range (into a shared `MeshletSet` that concatenates all LODs' clusters) + its CONSERVATIVE geometric error
     (object-space deviation from LOD0). For a `SphereGeometry(seg, rings)` tessellation, the conservative error =
     the sagitta `radius · (1 − cos(π / min(seg,rings)))` (the max chord deviation; documented + computed
     deterministically, OR a fixed per-LOD constant — pick the sagitta, it's principled). LOD0 has
     `geometricError = 0` (full detail).
   - `BuildLodMeshes(...) -> { MeshletSet combined; std::array<LodLevel, kNumLods> lods; }` — DS-decompose each
     tessellation, concatenate the clusters + reorder indices into one buffer, record each LOD's cluster range +
     error. `kNumLods = 3`.
   - `float ProjectionScaleForScreenError(float fovYRadians, int screenH)` = `screenH / (2 · tan(fovY/2))` — the
     HOST-PRECOMPUTED scalar (the only transcendental; computed once on the host, passed to both CPU + shader as
     exact bits).
   - `uint32_t SelectLod(float geometricError0Ref, const math::Vec3& instanceCenter, const math::Mat4& view, float
     projScale, std::span<const float> lodErrorThresholds, float errorScale)` — compute the view-space distance
     `dist = length((view · instanceCenter).xyz)` (or the z); `projectedError[n] = lods[n].geometricError · projScale
     / dist` (via `std::fma`/reciprocal — match the shader exactly); select the COARSEST LOD whose
     `projectedError ≤ errorThreshold · errorScale` (i.e. the coarsest LOD still under the allowed screen error),
     clamped to `[0, kNumLods-1]`. `errorScale = 0` (or a "forceLod0" flag) → always LOD0 (the disabled-path).
     Returns the LOD index. Document the exact arithmetic so the shader copy is bit-identical (`std::fma`, the
     reciprocal, the comparison order). NOTE: `dist` uses `length`/`sqrt` (not bit-identical CPU↔GPU per the DH
     lesson) — so to keep the SELECTED-LOD INTEGER bit-exact, compare on `dist²` against `(geometricError ·
     projScale / threshold)²` (squared form, no sqrt), OR host-precompute the per-instance `1/dist` and upload it
     as exact bits both read (PREFER the squared-distance comparison — pure multiply/compare, bit-exact, no sqrt).
     Document this explicitly — it is the make-or-break for GPU==CPU.

2. **`shaders/cluster_lod_select.comp.hlsl` (NEW).** One thread per instance: compute the squared view-distance,
   `SelectLod` (the squared-form comparison, `frustum`-style), then emit that LOD's clusters as `MdiCommand`s into
   the compacted draw buffer (each cluster of the selected LOD → one MdiCommand, `firstInstance` = the instance
   index for the model matrix + color). Copy `SelectLod` (squared form) VERBATIM from the header. Same ordered
   compaction idiom. NO new RHI. (Frustum cull MAY be composed but is not required for DV — keep the LOD the
   isolated new variable; if composed, force-LOD0 must still reproduce the full-detail reference exactly.)

3. **Showcase `--cluster-lod-shot <out>` (Vulkan, main.cpp) AND `--cluster-lod` (Metal, visual_test.mm — WIRE
   BOTH; confirm visual_test.mm in the diff).** A row/grid of instances at INCREASING distance from the camera
   (e.g. 6 spheres marching away) → `BuildLodMeshes` (3 tessellations) → per-instance `SelectLod` →
   `cluster_lod_select.comp` → `DrawIndexedMultiIndirect` the selected-LOD clusters (cluster-hash colored, so the
   LOD is visible as the cluster-patch coarseness). Near instances draw LOD0 (many fine clusters), far instances
   draw LOD2 (few coarse clusters) — a visible detail falloff. Declare the empty shadow pass (DQ lesson) →
   validation-clean. PROOFS (fail loudly):
   - **(1) GPU==CPU selected-LOD bit-exact:** `ReadBuffer` the GPU per-instance selected-LOD integers; CPU
     `SelectLod` over the same instances → the integer arrays are EQUAL (memcmp). Print `cluster-lod GPU==CPU
     selected-LOD: BIT-EXACT (<perLodCounts>)`.
   - **(2) force-LOD0 == full-detail BYTE-IDENTICAL:** `errorScale=0` / forceLod0 → every instance selects LOD0 →
     the render is byte-identical (SHA) to drawing every instance at LOD0 (the full-detail reference). Print
     `cluster-lod forceLod0 == full-detail: BYTE-IDENTICAL`.
   - **(3) LOD actually varies:** with the real threshold, the selected-LOD set spans ≥ 2 distinct LODs (near LOD0,
     far coarser) — a measurable LOD distribution (assert not-all-same). Print the per-LOD instance counts.
   - **(4) monotonicity:** a farther instance never selects a FINER LOD than a nearer one (same error/threshold) —
     the selection is distance-monotonic (assert in the test + a showcase check).
   - **(5) determinism:** two runs byte-identical.
   - **Golden** = the LOD-varied render → `tests/golden/metal/cluster_lod.png` (near fine clusters → far coarse).
     Metal two runs DIFF 0.0000, gate on compare.sh EXIT CODE. Print `cluster-lod: {instances:M, lods:3,
     lod0:n0, lod1:n1, lod2:n2}`. Existing 84 image goldens UNTOUCHED. **GOLDEN DISCIPLINE: ONLY
     `tests/golden/metal/cluster_lod.png`.**

4. **Determinism / cross-backend.** The LOD select is integer output from a squared-distance multiply/compare with
   a host-precomputed `projScale` (no per-instance transcendental, no sqrt on the bit-exact path) → GPU==CPU
   bit-exact + cross-backend-identical. The render of the selected clusters is the DT cluster MDI path (proven
   cross-backend). Two runs byte-identical. (Metal MAY render the selected set via the CPU bound path per the
   DT/CJ convention — image backend-identical, Vulkan carries the GPU==CPU select proof; document.)

5. **Tests `tests/cluster_lod_test.cpp` (pure CPU; `hf_add_pure_test`):**
   - `BuildLodMeshes`: 3 LODs, each a valid `MeshletSet` cluster range; LOD0 error 0; coarser LODs larger error,
     fewer clusters; the combined index buffer covers all LODs' triangles.
   - `SelectLod`: near instance → LOD0; far → coarser; force-LOD0 (errorScale 0) → LOD0 always; distance
     monotonicity (farther ≥ coarser); threshold boundaries (squared-form) select correctly; clamp to
     `[0, kNumLods-1]`.
   - `ProjectionScaleForScreenError` matches `screenH/(2 tan(fovY/2))`.
   - Determinism. Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `virtual-geometry-cluster-lod` (features) + `--cluster-lod-shot` (showcases).

## RHI seam additions (summary)
- **None.** Reuses the cluster MDI draw + compute-compaction + `ReadBuffer` surface (DT). New non-backend code
  (`cluster_lod.h`, `cluster_lod_select.comp.hlsl`, the test, the showcase) adds ZERO above-seam backend symbols.
  rhi.h + rhi_factory (dispatch baseline 2) + the backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI)
A runtime cluster-DAG simplifier with monotonic-error locked-boundary simplification (the discrete pre-baked
tessellation LODs are deterministic + bit-exact; a real simplifier is the deferred DY stretch), per-cluster-group
(sub-mesh) LOD within a single instance (DV selects per-instance — the cluster-group is the instance's cluster set;
sub-instance cluster-DAG LOD is the stretch), continuous/geomorph LOD, the visibility buffer. ONE discrete
screen-error LOD selection with a GPU==CPU bit-exact selected-LOD proof + force-LOD0==full-detail byte-identical +
distance-monotonicity + the visible LOD-falloff golden. **This completes the 4-slice virtual-geometry CORE.**

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 84) + new `cluster_lod_test`. Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--cluster-lod-shot` on Vulkan: near instances render fine cluster patches, far instances
   coarse — a visible detail falloff; `cluster-lod GPU==CPU selected-LOD: BIT-EXACT` + `forceLod0 == full-detail:
   BYTE-IDENTICAL` + LOD-varies (≥2 LODs) + distance-monotonic + two-run byte-identical; the `cluster-lod: {...}`
   line deterministic. Run under the AT Vulkan-validation gate → ZERO errors (empty shadow pass declared).
3. Metal: `visual_test --cluster-lod` → new golden `tests/golden/metal/cluster_lod.png`; two runs DIFF 0.0000 (gate
   on compare.sh EXIT CODE). The GPU==CPU select + forceLod0 proofs also pass. **Confirm visual_test.mm in the diff.**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `cluster_lod.png` added; the
   other 84 byte-identical. `git diff master --stat -- tests/golden` = ONLY `cluster_lod.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/cluster_lod.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+virtual-geometry-cluster-lod` + `--cluster-lod-shot`; introspect test updated.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `cluster_lod`
   image golden in the Mac round-trip loop AND `--cluster-lod-shot` in the `$vkShots` validation gate.
