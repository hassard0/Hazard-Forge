# Slice FR2 — Deterministic Fracture/Destruction: FRAGMENT EXTRACTION (per-cell mass properties) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #14 (DETERMINISTIC
> RIGID-BODY FRACTURE / DESTRUCTION, `hf::sim::fract`). FR1 partitioned the source lattice into N Voronoi cells;
> FR2 turns each non-empty cell into a FRAGMENT RECORD — its integer centroid, AABB, bounding-sphere radius, and
> volume/inverse-mass — via the count→scan→emit compaction the grid-hashes/MC3/CL2 already use. The fragment set
> the FR4 step will spawn as `fpx::FxBody` instances. PURE INT32 → MSL-native (a true GPU pass on BOTH backends,
> the FR1/MC1/GF1 bar). Branch: `slice-fr2`. See [[hazard-forge-fract-roadmap]].

**Goal:** Extend `engine/sim/fract.h` (additive — FR1 byte-unchanged) with `FractFragment` (the per-fragment
record) + `ExtractFragments(field, cells, seedCount, out)` (count→scan→emit the lattice samples grouped by
`cellId`, then reduce each non-empty cell to a fragment in **fixed ascending order**). Add
`shaders/fract_emit_{count,scan,emit}.comp.hlsl` (pure int32, MSL-native) OR a single
`fract_fragments.comp.hlsl` (implementer's call — see §Design). Add `--fract-fragments-shot` (Vulkan) /
`--fract-fragments` (Metal). Bake the integer golden `fract_fragments`. **NO new RHI.**

## Design call: count→scan→emit CSR + per-fragment reduction, PURE INT32 → MSL-native, integer lattice-space
FR2 is the GR2/FL2 `BuildGrainCellTable` (count→scan→emit, `grain.h:301`) applied to FR1's `cellId` array: count
samples per cell → exclusive scan → a CSR (`fragStart[]` + `fragSamples[]` = sample indices grouped by cell, the
**ascending-sample-index emit** of GR2). Then **one fragment per non-empty cell**, reducing its CSR slice in
ascending sample-index order to:
- **centroid** = `(Σ lattice-pos) / count` (per-axis integer sum / integer count — truncating divide, deterministic),
- **AABB** = per-axis `min`/`max` of member lattice coords,
- **boundRadiusSq** = `max` over members of `(pos − centroid)·(pos − centroid)` (squared, int32 — NO sqrt on the
  hot path; an optional `boundRadius = ISqrt(boundRadiusSq)` derived once per fragment, the `mc.h::ISqrt`),
- **volume** = member `count` (the integer voxel count),
- **invMass** = a deterministic integer function of `volume` (e.g. a fixed-point `kOne / volume`, or carry
  `volume` and let FR4 derive `invMass` — implementer's call, but it MUST be pure-integer + documented).

**The crux decision — keep it INTEGER lattice-space → MSL-native.** Centroid/AABB/radius/volume are all small
integers (lattice coords `< ~1024`, squared distances `< ~3.1M` → int32 with headroom — the FR1 bound). NO int64,
NO float, NO Q16.16 world scaling at FR2 → the shader MSL-generates natively (a TRUE GPU pass on both backends,
strict zero-differing-pixel — the FR1 bar). **World-unit Q16.16 fragment positions (for FR4's `fpx::FxBody`) are
DEFERRED to FR4's host-side spawn** (`worldPos = latticeCoord · worldCellSize`); FR2 stays integer lattice-space.

The output is a **COMPACT** array of `F ≤ seedCount` fragments (empty/dominated cells produce NO fragment — the
CL2/MC3 stream-compaction), plus a `cellToFragment[]` remap (cell index → fragment index, or a sentinel for
empty cells) so later slices can map a `cellId` to its fragment. Fragments are ordered by **ascending cell
index** (deterministic).

## The bit-exact reduction (deterministic, race-free)
The GPU mirror builds the CSR exactly like GR2/FL2 (count→scan→emit, the emit scatter using the per-cell write
cursor — single-thread ascending OR an atomic-bump that preserves ascending order; the CPU reference uses the
GR2 single-thread ascending cursor), then **one thread per fragment** reduces its own CSR slice in ascending
sample order → writes its own `FractFragment` record. Each fragment writes ONLY its own record → per-fragment
disjoint, race-free, NO atomics in the reduction, `[numthreads(64,1,1)]`. The fixed ascending order makes the
integer sums/min/max bit-identical CPU↔GPU↔cross-vendor. `ExtractFragments` (CPU) is the byte-for-byte reference
the GPU memcmp's against (the fragment array + the CSR + the remap).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The count→scan→emit CSR to MIRROR (`engine/sim/grain.h:301` `BuildGrainCellTable`):** the exact (1) count
  per cell → (2) exclusive prefix-sum `cellStart[cells+1]` → (3) emit each index into its cell's slice via the
  per-cell ascending cursor. FR2's CSR over `cellId` is identical (cells = `seedCount`, the "particle"→cell map
  is `cells.cellId[sample]` directly, no `CellOf` recompute). Read it for the CSR layout + the ascending-emit
  determinism argument.
- **The stream-compaction precedent (`engine/render/cluster_lod.h` CL2 / `engine/render/mc.h` MC3 emit):** the
  compact-non-empty + the per-element scan→offset→emit. FR2's "only non-empty cells become fragments" is this
  compaction; the `cellToFragment[]` remap is the inverse.
- **The integer square root (`engine/render/mc.h::ISqrt` / `engine/sim/fpx.h::FxISqrt`):** `boundRadius =
  ISqrt(boundRadiusSq)` (once per fragment, pure integer, identical on every vendor). Keep `boundRadiusSq` (int32)
  as the hot comparison; derive `boundRadius` once.
- **FR1 (this branch's `fract.h`, read-only — build on, do not modify FR1 code):** `FractField` (the lattice +
  `sampleCount`/`SampleIndex`), `FractCells{cellId[]}` (the FR1 output FR2 consumes), `FractSeed`, the int32
  discipline. The lattice POSITION of sample index `idx` is its `(x,y,z)` decompose: `x = idx % nx; y = (idx /
  nx) % ny; z = idx / (nx*ny)` (the inverse of `SampleIndex` — add a `SampleCoord(field, idx)` helper).
- **The int32 MSL-native shader precedent (`shaders/fract_classify.comp.hlsl` FR1 / GR2's cell passes / `mc_*`):**
  pure-int32 compute in `hf_gen_msl`. The FR2 shader(s) are the same — ADD to `hf_gen_msl`. Read FR1's shader for
  the SSBO/params binding shape + the flat-id→coord decompose.
- **Showcase + registration:** FR1's `--fract-cells-shot` plumbing; `scripts/verify.ps1`, `engine/editor/
  introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden**), `tests/fract_test.cpp`.
  NOTE: when adding the `--fract-fragments-shot` showcase flag, use the **standalone-loop arg-parse pattern** FR1
  established (NOT another `else if` in main.cpp's ladder — it hit MSVC C1061 blocks-nested-too-deeply).

## Design decisions (locked)
1. **`FractFragment` record (integer lattice-space):** `{ int cx,cy,cz (centroid); int minx,miny,minz,maxx,maxy,
   maxz (AABB); int boundRadiusSq; int boundRadius; uint32_t volume; uint32_t cellId; fx invMass; }` (exact field
   set is the implementer's call, but it MUST be pure-integer-derivable + documented; `invMass` may be `fx` if
   derived by a pure-integer `kOne/volume`). `ExtractFragments(field, cells, seedCount, out)` builds the CSR +
   the compact fragment array + the `cellToFragment[]` remap. Additive — FR1 `ClassifyFractCells` etc. unchanged.
2. **Shader(s):** EITHER three `fract_emit_{count,scan,emit}.comp` (the GR2 split) OR a count→scan→emit + a
   `fract_reduce.comp` — implementer's call, ALL pure int32, ALL in `hf_gen_msl`, all MSL-native. The reduction
   is one thread per fragment over its CSR slice (race-free).
3. **Showcase `--fract-fragments-shot <out>` (Vulkan) AND `--fract-fragments` (Metal) — WIRE BOTH** (standalone
   arg-parse loop). The FR1 lattice + seed scene; run FR2; GPU fragment array + CSR + remap → **memcmp vs the CPU
   `ExtractFragments` reference**. Render: color each lattice sample by its fragment's centroid-hash AND overlay/
   mark the fragment centroids (or render the AABBs) — a viz showing the extracted fragments (the centroids/bounds
   over the cell mosaic). Golden = `tests/golden/metal/fract_fragments.png` (Mac-baked by the CONTROLLER — DO NOT
   commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU fragment array + CSR + remap == the CPU reference byte-for-byte. Print
     `fract-fragments: {cells:<M>, fragments:<F>, samples:<N>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `fract-fragments determinism: two runs BYTE-IDENTICAL`.
   - **(3) mass partition:** `Σ fragment.volume == sampleCount` (every sample belongs to exactly one fragment).
     Print `fract-fragments mass: {Σvol:<S>, samples:<N>} partition-exact` with `Σvol == N`.
   - **(4) conservative bounds:** every member sample lies inside its fragment's AABB AND within `boundRadius` of
     its centroid (no member outside its own bound). Print `fract-fragments bounds: all <N> members inside
     fragment AABB+sphere`.
   - **Golden discipline: ONLY `tests/golden/metal/fract_fragments.png`; do NOT commit it.** Existing 154 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal GPU == golden, ZERO differing pixels.
6. **Tests `tests/fract_test.cpp` additions (pure CPU):** `ExtractFragments` — a 1-seed field → 1 fragment, its
   centroid the lattice center, volume == sampleCount; a 2-seed split → 2 fragments with complementary volumes
   summing to N; a dominated seed (0 samples) → NO fragment + its `cellToFragment` is the sentinel; centroid/AABB
   on a known small field (hand-computed); `Σvol == N`; every member inside its AABB+sphere; `SampleCoord` round-
   trips `SampleIndex`. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-fract-fragments` (features) + `--fract-fragments-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the FR1/GR2 surface). `rhi.h` + backend
  dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple.h` + `couple_grain.h` +
  `couple_gf.h` + `engine/physics/` + all existing shaders UNCHANGED. FR1 `fract.h` code + `fract_classify.comp`
  UNCHANGED (FR2 additive). The only new shaders are the FR2 emit/reduce passes (int32, in `hf_gen_msl`). Report
  the seam empty except the new MSL-native FR2 shaders.

## Out of scope (YAGNI — later FR slices)
The bond graph + break model (FR3), the fracture step through `fpx.h` (FR4 — where world-unit Q16.16 positions
+ the per-fragment `fpx::FxBody` spawn happen), lockstep (FR5), the lit render (FR6). TRUE convex hull geometry
(FR2 produces a centroid + AABB + bounding sphere, NOT a triangulated hull — the sphere-bound fragment is the
honest first cut, the documented FPX3-sphere-contact reuse; a real hull is a later refinement). Anisotropic/
weighted mass. FR2 claims ONLY: a deterministic per-cell fragment extraction (centroid/AABB/bound/volume) with
mass-partition + conservative-bounds proofs, bit-identical CPU↔Vulkan↔Metal, with the integer golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 100) + the new `fract_test` fragment cases. Clean under
   `windows-msvc-asan` (build+run `fract_test` + `introspect_test`).
2. **proofs + visual:** `--fract-fragments-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the fragment centroids/bounds over the cell mosaic (pixel-check; the FR1 lesson).**
3. Metal: `visual_test --fract-fragments` → new golden `tests/golden/metal/fract_fragments.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm the FR2 shaders ARE
   MSL-generated (int32, a true GPU pass on both).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fract_fragments.png` added; the
   other 154 byte-identical (re-run `--fract-cells-shot` → still bit-exact). `git diff master --stat --
   tests/golden` = ONLY `fract_fragments.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fract-fragments` + `--fract-fragments-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple.h`/`couple_grain.h`/
   `couple_gf.h` + `engine/physics/` + FR1 `fract.h`/`fract_classify.comp` byte-unchanged). `scripts/verify.ps1`
   updated: `fract_fragments` golden in the Mac loop + `--fract-fragments-shot` in `$vkShots`. The new FR2 shaders
   ARE in `hf_gen_msl`.
