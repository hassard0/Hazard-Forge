# Slice MC2 — GPU Isosurface Meshing: PER-CELL TRIANGLE COUNT (256-case table + atomic total) (Phase 10 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 2nd MC slice (after MC1's
> per-cell case classification): given each cell's MC case index, look up how many triangles it emits from the
> canonical 256-case Marching-Cubes table, write the per-cell count, and accumulate the GRAND TOTAL via an atomic add
> — the prerequisite for MC3's prefix-sum compaction + triangle emission. Proven GPU==CPU BIT-EXACT (per-cell counts
> AND the atomic total), integer, cross-backend BIT-IDENTICAL. NO new RHI (the `InterlockedAdd` of integer counts is
> the autoexposure-histogram precedent — commutative → order-independent → deterministic). Namespace `hf::render::mc`.
> Branch: `slice-mc2-count`. See [[hazard-forge-mc-roadmap]].

**Goal:** Extend `engine/render/mc.h` with the canonical MC 256×16 triangle table (`kTriTable`) + the derived per-case
triangle-count table (`kTriCount`) + `CountTriangles`/`CountCells` (CPU references) + `TotalTriangles`; add
`shaders/mc_count.comp.hlsl` (one thread per cell: classify → table-lookup count → write per-cell count +
`InterlockedAdd` the total); add a `--mc-count-shot` (Vulkan) / `--mc-count` (Metal) showcase that counts the MC1
sphere field's triangles, reads back the per-cell count buffer + the total, proves both BIT-EXACT vs the CPU
reference, and bakes an integer count-grid golden. Make-safe: header additions + a NEW shader + NEW showcase + NEW
golden; MC1's `mc_classify` golden + everything else UNCHANGED.

## The integer core (extends mc.h)
- **`kTriTable[256][16]`** — the canonical public-domain Marching-Cubes triangle table (Paul Bourke / standard), each
  case row a list of edge indices (0–11) in groups of 3, terminated by `-1`, MATCHING the MC1 locked corner numbering
  (0=(0,0,0)…6=(1,1,1)…7=(0,1,1)) + the edge numbering comment-pinned in MC1. `static_assert`/comment-document the
  source + the convention so it is auditable. (MC3 consumes the full table for edge→vertex emission; MC2 only needs
  the count, but the full table is added now so the count is DERIVED + verifiable, not magic numbers.)
- **`kTriCount[256]`** — derived: `kTriCount[c] = (count of non-negative entries in kTriTable[c]) / 3` (0..5). Provide
  it as a `constexpr`-filled array OR a `TriCountForCase(c)` accessor; either way unit-test that it equals the table
  derivation for all 256 cases, and that `kTriCount[0x00]==0` and `kTriCount[0xFF]==0` (both fully-out / fully-in
  emit no triangles).
- **`int CountTriangles(uint8_t caseIndex)`** = `kTriCount[caseIndex]`.
- **`void CountCells(const VoxelField&, int32_t isovalue, std::vector<uint32_t>& countOut)`** — CPU ref: for each
  cell, `countOut[cellId] = CountTriangles(CaseIndex(corners, iso))` (reuse MC1's `ClassifyCells` math).
- **`uint64_t TotalTriangles(std::span<const uint32_t> counts)`** = the sum (the CPU mirror of the GPU atomic total).
  (Use `uint32_t` total if the field can't overflow — 32768 cells × max 5 tris = 163840 fits u32 easily; document.)

## Reuse map (file:line)
- **MC1 (the input math):** `engine/render/mc.h` — `VoxelField`, `kCornerOffset`, `CaseIndex`, `SampleField`,
  `ClassifyCells`. MC2 reuses `CaseIndex` verbatim (classify-then-count in one pass, self-contained — does NOT depend
  on a persisted MC1 buffer).
- **The integer-atomic total precedent (the make-or-break for the atomic):** `shaders/autoexposure_histogram.comp.hlsl:81`
  `InterlockedAdd(gHistogram[bin], 1u)` + its determinism rationale (`:7-11`: integer atomic add is commutative →
  order-independent → the result is identical regardless of thread race → bit-exact GPU==CPU + cross-backend). MC2's
  `InterlockedAdd(gTotal[0], triCount)` is the same family; plain MSL, no MSL-2.2.
- **The compute marking template:** `shaders/mc_classify.comp.hlsl` (MC1) — one thread per cell, 3+ SSBOs, an enabled
  flag, the math copied verbatim. `mc_count.comp` is this + the table lookup + the atomic.
- **Compute + readback surface (NO new RHI):** `BufferUsage::Storage` (`rhi.h:166`), compute dispatch
  (`rhi.h:412-426`), `ReadBuffer` (`rhi.h:616`).
- **Golden + registration:** the MC1 `--mc-classify-shot`/`--mc-classify` showcase + Z-slice viz template;
  `meshlet.h:79` `hashColor`; `scripts/verify.ps1` `$Goldens`/`$vkShots`, `engine/editor/introspect.cpp`,
  `tests/introspect_test.cpp`.

## Design decisions (locked)

1. **`mc_count.comp.hlsl` (NEW).** ONE thread per cell. Reads 8 corner scalars from `gField` (b0), computes
   `CaseIndex` (copied VERBATIM from mc.h), looks up `triCount = gTriCount[caseIndex]` from the host-uploaded
   256-entry count table SSBO (b1), writes `gCounts[cell] = triCount` (b2), and `InterlockedAdd(gTotal[0], triCount)`
   (b3, a single-uint SSBO). `countEnabled=0` push flag → write 0, no atomic add (the disabled no-op → counts
   all-zero, total 0). `gParams` (b4: {nx,ny,nz,isovalue,countEnabled}). `ComputePipelineDesc{ storageBufferCount=5,
   threadsPerGroupX=64 }`. The `InterlockedAdd` of integer counts is the ONLY atomic, order-independent → deterministic
   + cross-backend bit-identical (the autoexposure argument). Plain integer → NO `--msl-version 20200`. Register in
   BOTH compile lists.
2. **Showcase `--mc-count-shot <out>` (Vulkan, main.cpp) AND `--mc-count` (Metal, visual_test.mm — WIRE BOTH; confirm
   visual_test.mm + `#include "render/mc.h"`).** The SAME MC1 sphere field (33³ corners, radius 12, iso 0). Upload
   `gField` + `gTriCount` (host-built from `kTriTable`), clear `gCounts` + `gTotal` to 0, dispatch → `ReadBuffer`
   `gCounts` + `gTotal`. CPU-run `CountCells` + `TotalTriangles` over the SAME field. Golden = the per-cell count
   field CPU-colored as a grid of Z-slices (each cell colored by its triangle count 0..5 — a fixed count→color ramp
   [e.g. 0=dark, 1..5 = distinct hues], NOT hashColor, so the count is legible) → `tests/golden/metal/mc_count.png`.
   CPU-colored from the integer read-back → identical both backends by construction → DIFF 0.0000 / strict zero diff.
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU per-cell counts bit-exact:** `memcmp(gpuCounts, cpuCounts) == 0`. Print `mc-count GPU==CPU
     counts: <N> cells BIT-EXACT`.
   - **(2) atomic total bit-exact:** `gpuTotal == TotalTriangles(cpuCounts)` (the `InterlockedAdd` total equals the
     CPU sum — the atomic determinism proof). Print `mc-count atomic total: <T> tris == CPU (order-independent)`.
   - **(3) table self-consistency:** for all 256 cases `kTriCount[c]` equals the count derived from `kTriTable[c]`,
     and `kTriCount[0x00]==kTriCount[0xFF]==0`. Print `mc-count tri-table: 256 cases consistent, empty/full=0 OK`.
   - **(4) disabled-path no-op:** `countEnabled=false` → `gCounts` all-zero AND `gTotal==0`. Print `mc-count disabled:
     zero counts + zero total (no-op)`.
   - **(5) determinism:** two dispatches → counts AND total byte-identical (the atomic is order-independent). Print
     `mc-count determinism: two dispatches BYTE-IDENTICAL`.
   - **(6) {stats}:** `mc-count: {cells:<C>, surface-cells:<k>, total-tris:<T>, max-tris/cell:<m>}`.
   - **Golden discipline: ONLY `tests/golden/metal/mc_count.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 100 image goldens UNTOUCHED.
4. **Tests `tests/mc_test.cpp` additions (pure CPU):** `kTriCount` == derived-from-`kTriTable` for all 256 cases +
   `[0x00]==[0xFF]==0` + every count in `[0,5]`; `kTriTable` rows are well-formed (groups of 3 then `-1`, all edge
   indices in `[0,11]`); `CountCells` over a known tiny field → known counts; `TotalTriangles` == Σ; `countEnabled`
   off modeled → all-zero + total 0; determinism. Clean under `windows-msvc-asan`.
5. **Introspect.** Add exactly `gpu-isosurface-meshing-count` (features) + `--mc-count-shot` (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + compute + `InterlockedAdd` (the autoexposure family) + `ReadBuffer`. ZERO
  above-seam backend symbols. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — MC3+)
Prefix-sum compaction + vertex/index emission (MC3 — this slice only COUNTS, no geometry), fixed-point interpolated
vertex placement (MC4), rendering the mesh (MC5), normals/shading (MC6). The full `kTriTable` is ADDED here but only
its derived COUNT is used on the GPU; MC3 uploads + consumes the full table for edge→vertex emission. ONE compute pass
that counts triangles per cell + an atomic grand total with the GPU==CPU bit-exact proof (counts AND total) +
table self-consistency + disabled no-op + determinism and the integer count-grid golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 91) + the new `mc_test` count/table cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--mc-count-shot` on Vulkan: a coherent per-cell count-grid viz (the sphere's surface cells
   colored by triangle count across Z-slices); all 6 proof lines. Run under the Vulkan-validation gate → ZERO errors
   (the atomic-add compute SYNC-HAZARD-free).
3. Metal: `visual_test --mc-count` → new golden `tests/golden/metal/mc_count.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). The GPU==CPU + atomic-total + determinism proofs also pass on Metal (integer atomic).
   **Confirm visual_test.mm in the diff; confirm mc_count.comp MSL-generates + the InterlockedAdd lowers on Metal
   (atomic on a uint SSBO — should need no MSL-2.2; if it fails, report).** Integer golden → a strict cross-backend
   pixel compare must show ZERO differing pixels.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `mc_count.png` added; the other 100
   byte-identical. `git diff master --stat -- tests/golden` = ONLY `mc_count.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+gpu-isosurface-meshing-count` + `--mc-count-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated: `mc_count` golden in the Mac loop
   + `--mc-count-shot` in `$vkShots`.
