# Slice GR2 — Deterministic GPU Granular/Sand: GRID-HASH NEIGHBOR SEARCH (int32, MSL-native) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #10
> (DETERMINISTIC GPU GRANULAR / SAND, `hf::sim::grain`). Builds the per-grain CANDIDATE NEIGHBOR LIST over
> the GR1 grain pool via a uniform spatial-hash grid with the proven count→scan→emit compaction — the
> candidate set the GR3 frictionless contact solve + GR4 Coulomb friction iterate. **PURE INT32 →
> MSL-native on BOTH backends** (a true GPU pass on Vulkan AND Metal, unlike GR1's int64-Vulkan-only
> integrate) → the STRONGEST cross-backend proof (strict zero-differing-pixel). The FL2/FPX2/CL2 twin.
> Branch: `slice-gr2`. See [[hazard-forge-grain-roadmap]].

**Goal:** Extend `engine/sim/grain.h` (additive — GR1 byte-unchanged) with the bounded-dense-grid neighbor
search: `GrainGrid`/`MakeGrainGrid`/`GrainCellOf`/`FlatGrainCellId`/`GrainCellCount`, `BuildGrainCellTable`
(CSR cell→grain buckets), `GrainNeighborAccept` (the per-axis box reject), `CountGrainNeighbors`/
`BuildGrainNeighborList` (CSR per-grain candidate lists over the 27-cell stencil) — all via count→scan→emit,
pure int32. Add 6 int32 shaders `grain_cell_{count,scan,emit}.comp.hlsl` + `grain_neighbor_{count,scan,emit}.comp.hlsl`
(**MSL-native — IN `hf_gen_msl`**). Add `--grain-neighbors-shot` (Vulkan) / `--grain-neighbors` (Metal). Bake
the integer golden `grain_neighbors` (the per-grain neighbor-count heat viz). NO new RHI.

## Design call: PURE INT32 → MSL-native (the FL2/FPX2/CL2 precedent) — strict zero-diff BOTH backends
The whole neighbor search is integer INDEX arithmetic: cell ids are `FloorDiv` per axis (`fpx::BroadphaseCell`
at the cell size), the cell bucketing is count→scan→emit of grain indices, the candidate reject is a per-axis
`|a.axis − b.axis| < hSearch` compare (`fx` is int32 → a PURE INT32 compare, NO squaring, NO products, NO
int64, NO sqrt). So **the GPU shaders MSL-generate natively** and run a TRUE GPU pass on both Vulkan and
Metal (the FL2 win — unlike GR1/GR3/GR4's int64). The bar is the STRICT integer bar: Vulkan == Metal ZERO
differing pixels, and the GPU `{cellTable, neighborList}` == the CPU reference byte-for-byte. **This is the
flagship's strongest proof — a genuine GPU pass bit-identical across vendors.**

## The contact-search-radius decision (the one parameter delta vs FL2 — document it)
FL2's cell-size / candidate radius is the PBF smoothing radius `h`. GR2's is the **contact search radius
`hSearch`** — the range within which two grains are CANDIDATE contact pairs (their surfaces could touch).
For uniform radius `r`, two grains contact when centre distance < 2·r (the diameter). To find ALL contact
candidates the search radius must be ≥ the contact diameter; to also make the GR1 non-overlapping block
(spacing = 2·r) a NON-DEGENERATE neighbor graph for the heat viz, pick `hSearch` modestly LARGER than the
block spacing (e.g. `hSearch ≈ 1.5 × spacing`, a documented host-snapped Q16.16 constant) so each interior
grain sees its 26 lattice neighbours (the FL2 "interior dense / surface sparse" structure). The EXACT
radial contact-overlap cull (distance < r_i + r_j) is DEFERRED to GR3's contact solve (the contact
projection is a no-op for non-overlapping candidate pairs — exactly FL2's "over-inclusive box candidate,
exact cull deferred to FL3" discipline). The box reject `|dx| < hSearch && |dy| < hSearch && |dz| < hSearch`
is pure int32. **`hSearch` ≥ the GR3 contact diameter is a HARD requirement** (a smaller search radius would
miss real contacts in GR3) — the implementer asserts `hSearch ≥ 2·maxRadius` and documents the chosen value.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FL2 grid-hash neighbor search to MIRROR near-verbatim (`engine/sim/fluid.h`):** `FluidGrid`
  (`fluid.h:226-230`), `CellOf` (`fluid.h:234`), `FlatCellId` (`fluid.h:241`), `CellCount` (`fluid.h:247`),
  `MakeGrid` (`fluid.h:255`), `FluidCellTable` + `BuildCellTable` (the count→scan→emit on grains,
  `fluid.h:283-315`), `NeighborAccept` (the per-axis box reject, `fluid.h:321-326`), `FluidNeighborList` +
  `CountNeighbors` + `BuildNeighborList` (the 27-cell-stencil count→scan→emit, `fluid.h:335-415`). GR2 is
  the SAME code with `Grain` types, `hSearch` for the radius, and `GrainParticle.pos` as the position.
- **The Q16.16 grid toolbox (read-only, `engine/sim/fpx.h`):** `FloorDiv` (`fpx.h:177`, deterministic
  floor-division correct for negative coords), `FxCell` (`fpx.h:183`), `CellId` (`fpx.h:196`). DO NOT
  modify fpx.h / cloth.h / fluid.h — grain is the additive sibling.
- **The int32 grid SHADERS to mirror (`shaders/fluid_cell_{count,scan,emit}.comp.hlsl` +
  `fluid_neighbor_{count,scan,emit}.comp.hlsl`):** the 6 FL2 shaders, all int32, all in `hf_gen_msl`. GR2's
  6 shaders are the SAME structure → ADD all 6 to BOTH the DXC list (samples CMake) AND `hf_gen_msl`
  (metal_headless CMake) — they ARE MSL-native (contrast GR1's int64 grain_integrate, DXC-only).
- **The integer-golden discipline + showcase:** FL2's `--fluid-neighbors-shot` / `--fluid-neighbors` (the
  per-particle neighbor-count heat viz: color each grain by its neighbor count) — mirror for
  `--grain-neighbors-shot` / `--grain-neighbors`. GR1's `--grain-integrate-shot` is the immediate template
  for the grain SSBO upload / dispatch / readback / memcmp / debug-viz plumbing.
- **Registration:** `scripts/verify.ps1` ($Goldens + $vkShots), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp`, `tests/grain_test.cpp`, the CMake shader lists. main.cpp has `/bigobj`.

## Design decisions (locked)
1. **Bounded dense grid at cell-size `hSearch` (the FL2 scheme verbatim).** A grain's cell coord is
   `FloorDiv(pos.axis, hSearch)` per axis; `MakeGrainGrid` sizes the grid to the grain AABB (every grain's
   cell in `[0,gridDim)`); the flat cell id is `fpx::CellId` of `(coord − cellMin)` into `gridDim`. Total,
   collision-free, deterministic. Empty pool → a 1×1×1 grid (deterministic degenerate). Pure int32.
2. **`BuildGrainCellTable` (count→scan→emit on grains).** CSR `cellStart` (cellCount+1 exclusive prefix-sum)
   + `cellGrains` (grain indices grouped by cell, ASCENDING grain index within a cell — deterministic). The
   GPU `grain_cell_{count,scan,emit}` mirror this byte-for-byte. (DET-CRUX, the FL2 lesson: the EMIT is the
   single-thread ascending-grain scatter — a parallel atomic cursor would make the within-cell order
   GPU-scheduling-dependent → non-deterministic. The cell COUNT + the neighbor passes are per-grain-disjoint
   and race-free; only the cell-emit scatter is the ordered pass. Mirror the FL2 cell_emit exactly.)
3. **`GrainNeighborAccept` + `BuildGrainNeighborList` (27-cell stencil, count→scan→emit).** For each grain
   i, scan its 3×3×3 = 27-cell stencil; for each j≠i in those cells, accept iff `|pos_i.axis − pos_j.axis| <
   hSearch` on EVERY axis (the pure-int32 box reject, NO self-neighbor). Emit accepted j into i's disjoint
   CSR slice in the FIXED order (ascending stencil cell dz,dy,dx −1..+1, then ascending j within a cell) →
   fully deterministic. Stencil cells outside the grid are clamped/skipped. CSR `neighborStart`
   (grainCount+1) + `neighbors[]`. The GPU `grain_neighbor_{count,scan,emit}` do the SAME three passes
   (per-grain-disjoint → race-free) and memcmp byte-for-byte.
4. **6 int32 shaders, ALL in `hf_gen_msl` (the FL2 contrast to GR1).** `grain_cell_{count,scan,emit}` +
   `grain_neighbor_{count,scan,emit}`, pure int32 → MSL-native, a TRUE GPU pass on both backends. Report
   that all 6 are in `hf_gen_msl` (and grain_integrate from GR1 is NOT — the int64/int32 split).
5. **Showcase `--grain-neighbors-shot <out>` (Vulkan) AND `--grain-neighbors` (Metal) — WIRE BOTH.** Run
   `InitGrainBlock` (the GR1 1000-grain block; OPTIONALLY integrate a few GR1 steps to a representative
   state, but the static block already gives a rich neighbor graph at `hSearch ≈ 1.5·spacing`) → build the
   grid + cell table + neighbor list on the GPU → **memcmp vs the CPU `BuildGrainCellTable` /
   `BuildGrainNeighborList` reference** → color each grain by its neighbor count (a heat viz: interior dense
   / surface sparse). Golden = `tests/golden/metal/grain_neighbors.png` (baked on the Mac by the CONTROLLER
   — DO NOT commit).
6. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU `{cellStart, cellGrains, neighborStart, neighbors}` == the CPU
     reference byte-for-byte. Print `grain-neighbors: {particles:<N>, cells:<C>, neighbors:<T>, maxPer:<M>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two builds → identical. Print `grain-neighbors determinism: two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the neighbor counts are coherent (interior grains many neighbours, surface
     fewer; all accepted pairs within `hSearch`). Print `grain-neighbors coverage: <T> candidate pairs
     (interior dense, surface sparse)`.
   - **(4) sparse / disabled no-op:** a single grain (or grains spread > hSearch apart) → ZERO neighbours.
     Print `grain-neighbors sparse: 0 neighbours (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/grain_neighbors.png`; do NOT commit it.** Existing 130
     image goldens (incl GR1 `grain_integrate`) UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict — and the STRONGEST: a real GPU pass on both):** Vulkan GPU ==
   Metal GPU == the committed Metal golden, **ZERO differing pixels** (NOT a CPU-reference fallback like
   GR1's int64 — GR2's int32 shaders run a true GPU pass on Metal too).
8. **Tests `tests/grain_test.cpp` additions (pure CPU):** `MakeGrainGrid` (correct cellMin/gridDim over a
   known pool); `BuildGrainCellTable` (CSR offsets + ascending within-cell order); `GrainNeighborAccept`
   (box accept/reject hand-checked); `BuildGrainNeighborList` (a tiny hand-laid pool → the exact neighbor
   counts + the fixed stencil order); the no-self-neighbor + sparse cases. Clean under `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-grain-neighbors` (features) + `--grain-neighbors-shot`
   (showcases). Rebake the introspect JSON golden + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the FL2/GR1 surface). `rhi.h` +
  `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `cloth.h` + `fluid.h` + `engine/physics/`
  UNCHANGED. GR1's `grain.h` integrator + `grain_integrate.comp` UNCHANGED (GR2 is purely additive). Report
  the seam is empty.

## Out of scope (YAGNI — later GR slices)
The frictionless contact projection (GR3 — the candidate list is GR2's job, the SOLVE is GR3's), Coulomb
friction (GR4), lockstep (GR5), the lit render (GR6). Exact radial overlap cull (deferred to GR3's solve,
which is a no-op beyond overlap), variable-radius spatial hashing, sort-based reordering. GR2 claims ONLY:
the per-grain candidate neighbor list, bit-identical CPU↔Vulkan↔Metal (a true GPU pass on both), with the
heat-viz integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 96) + the new `grain_test` neighbor cases. Clean
   under `windows-msvc-asan`.
2. **proofs + visual:** `--grain-neighbors-shot` on Vulkan: the 4 proofs + exit 0. Run under the
   Vulkan-validation gate → ZERO VUID in the OUTPUT (set BOTH `VK_LAYER_PATH` to the conan layers dir AND
   `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the rendered image actually shows the coherent
   neighbor-count heat viz (pixel-check the colored grain region) — do NOT trust the proof's claim alone
   (the NAV6/CL6 lesson).**
3. Metal: `visual_test --grain-neighbors` → new golden `tests/golden/metal/grain_neighbors.png`; two runs
   DIFF 0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm all 6 grain
   grid shaders MSL-generate (they ARE in `hf_gen_msl`).** The cross-vendor Vulkan-vs-Metal delta is
   **STRICT ZERO** (a true GPU integer pass on both — the strongest bar).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `grain_neighbors.png` added;
   the other 130 byte-identical (GR1 `grain_integrate` + all existing untouched; re-run
   `--grain-integrate-shot` → still bit-exact). `git diff master --stat -- tests/golden` = ONLY
   `grain_neighbors.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-grain-neighbors` + `--grain-neighbors-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; `engine/sim/fpx.h` + `cloth.h` + `fluid.h` +
   `engine/physics/` + GR1's `grain_integrate.comp` byte-unchanged). `scripts/verify.ps1` updated:
   `grain_neighbors` golden in the Mac loop + `--grain-neighbors-shot` in `$vkShots`. All 6 new grain grid
   shaders in BOTH the DXC list AND `hf_gen_msl`.
