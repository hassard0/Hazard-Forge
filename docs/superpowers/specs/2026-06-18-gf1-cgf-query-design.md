# Slice GF1 — Deterministic Grain↔Fluid Coupling: UNIFIED TWO-POOL WORLD + SHARED-GRID CROSS QUERY (the BEACHHEAD of FLAGSHIP #13) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice of FLAGSHIP #13 (DETERMINISTIC
> TWO-WAY GRAIN↔FLUID COUPLING, `hf::sim::cgf`) — the THIRD material-interaction pairing, and the FIRST
> **particle↔particle** coupling (vs the two existing rigid↔particle ones: rigid↔fluid CP, rigid↔grain CG).
> Two emergent materials co-resident in ONE Q16.16 world — a frictional sand bed (`grain.h`) + an
> incompressible PBF fluid (`fluid.h`): WET SAND / MUD / SLURRY. Fluid seeps into the sand, sand fluidizes
> into slurry, and two peers re-simulate it bit-for-bit from inputs alone. UE5 has no deterministic granular
> at all, let alone deterministic two-phase granular↔fluid. GF1 is ONLY the unified two-pool world + the
> cross-pool neighbour QUERY (each grain's nearby fluid particles AND each fluid particle's nearby grains) —
> the link, NO exchange yet (GF2 buoyancy/seepage, GF3 contact reaction, GF4 the coupled step, GF5 lockstep,
> GF6 render). The GR2/CG1 grid-hash query applied CROSS-POOL. Branch: `slice-gf1`. See
> [[hazard-forge-couple-gf-roadmap]] (to be created).

**Goal:** Create `engine/sim/couple_gf.h` (header-only, namespace `hf::sim::cgf`, `#include "sim/grain.h"` +
`"sim/fluid.h"` read-only) with a `CGFWorld` (a grain pool + a fluid pool + config) and `BuildCGFNeighbors(world)`:
build ONE shared grid covering BOTH pools, bucket each pool into a cell table, and build the two cross-pool
candidate lists — `gfNeighbors` (per grain → its nearby fluid particles) + `fgNeighbors` (per fluid particle →
its nearby grains) — via count→scan→emit over the 27-cell stencil. Add the cross-query shaders (int32 →
**MSL-native**). Add `--cgf-query-shot` (Vulkan) / `--cgf-query` (Metal). Bake the integer golden `cgf_query`.
NO new RHI. `grain.h`/`fluid.h`/`fpx.h`/`couple.h`/`couple_grain.h` + their golden sets UNTOUCHED (couple_gf is
the additive sibling).

## Design call: PURE INT32 → MSL-native (the FL2/GR2/CG1 precedent) — strict zero-diff BOTH backends
The cross-pool query is integer index arithmetic + a per-axis `|dx| < h` box reject (`grain::GrainNeighborAccept`
/ `fluid::NeighborAccept` shape — `fx` is int32 → a PURE INT32 compare, NO products, NO int64, NO sqrt). So
the GPU shaders MSL-generate natively → a TRUE GPU pass on both Vulkan AND Metal (the strongest cross-vendor
proof, like GR2/CG1 — strict zero-differing-pixel). The exact radial cull is DEFERRED to GF2/GF3 (the
exchange force/contact is 0 beyond `h`), so the over-inclusive box candidate is correct — the FL2/GR2
discipline. Bar: strict INTEGER — Vulkan GPU == Metal GPU == golden, ZERO differing pixels; the GPU
`{gfNeighbors, fgNeighbors}` == the CPU reference byte-for-byte.

## The shared grid + cross query (the new shape — two pools, ONE grid)
The two pools share the SAME Q16.16 world units (both `#include sim/fpx.h`; `FluidParticle` and `GrainParticle`
are near-identical std430 packings). Build ONE shared grid `CGFGrid` at cell-size `h` (the coupling radius)
covering the UNION of both pools' AABBs (`MakeGrid`/`MakeGrainGrid` are structurally identical — `FloorDiv`
cell coords + `CellId` linearization; the union is `min(grainCellMin, fluidCellMin)` .. `max(...)` per axis so
the linearization stays total + collision-free over BOTH pools — pin this edge case). Bucket EACH pool into
its own cell table over that shared grid (reuse the existing `grain_cell_*` / `fluid_cell_*` count→scan→emit
passes). Then:
- `gfNeighbors` (grain → fluid): for each grain, scan its 27-cell stencil of the FLUID cell table; accept each
  fluid particle with `|dx| < h` per axis. count→scan→emit CSR.
- `fgNeighbors` (fluid → grain): for each fluid particle, scan its 27-cell stencil of the GRAIN cell table;
  accept each grain with `|dx| < h` per axis. count→scan→emit CSR.
Both are pure int32. (DET-CRUX, the GR2/CG1 lesson: the per-particle EMIT is the fixed-order scatter; count +
the per-particle lists are per-particle-disjoint → race-free. The reused cell-EMIT is single-thread ascending.)

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The GR2/FL2 grid-hash + neighbour search to MIRROR (`engine/sim/grain.h` GR2 + `engine/sim/fluid.h` FL2,
  read-only):** `GrainGrid`/`MakeGrainGrid` (`grain.h:238-284`), `FluidGrid`/`MakeGrid` (`fluid.h:226-274`),
  `BuildGrainCellTable`/`BuildCellTable` (the count→scan→emit cell bucketing), `GrainNeighborAccept`/
  `NeighborAccept` (the per-axis box reject, `grain.h:335`/`fluid.h:321`), `BuildGrainNeighborList`/
  `BuildNeighborList` (the 27-cell-stencil count→scan→emit CSR). GF1's `gfNeighbors`/`fgNeighbors` are the SAME
  count→scan→emit, CROSS-POOL (the query pool vs the target pool's cell table).
- **The CG1 unified-world + cross-query CSR shape (`engine/sim/couple_grain.h`):** `CGrainWorld`,
  `GatherBodyGrains` / `CGrainQuery{bodyStart, bodyGrains}` — the additive-sibling unified-world pattern + the
  per-query CSR. GF1's `CGFWorld` + `CGFNeighbors{gfStart, gfNeighbors, fgStart, fgNeighbors}` are the SAME
  shape, two pools instead of bodies+grains.
- **The Q16.16 + particle structs (read-only):** `grain::GrainParticle` (`grain.h:85`), `fluid::FluidParticle`
  (`fluid.h:82`), `fpx::FloorDiv`/`FxCell`/`CellId`. DO NOT modify grain.h/fluid.h/fpx.h/couple.h/couple_grain.h
  — couple_gf is the NEW additive sibling `#include`-ing grain.h + fluid.h read-only.
- **The int32 grid SHADERS to mirror (`shaders/grain_neighbor_{count,scan,emit}.comp.hlsl` GR2 +
  `shaders/grain_cell_*`/`fluid_cell_*`):** the FL2/GR2 int32 passes — GF1's `cgf_gf_{count,scan,emit}.comp`
  (grain→fluid) + `cgf_fg_{count,scan,emit}.comp` (fluid→grain) are the SAME shape over the cross-pool cell
  table. REUSE the existing `grain_cell_*` + `fluid_cell_*` passes for the two cell tables (sized to the shared
  grid). ALL int32 → in `hf_gen_msl`. Report which existing passes are reused vs new.
- **Showcase + registration:** GR2's `--grain-neighbors-shot` / CG1's `--cgrain-query-shot` (the heat viz);
  `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect
  JSON golden** — the GR2/CP2 lesson), `tests/cgf_test.cpp` (NEW test target — register in `tests/CMakeLists.txt`).
  main.cpp has `/bigobj`.

## Design decisions (locked)
1. **`CGFWorld` (the unified two-pool world).** `struct CGFWorld { std::vector<grain::GrainParticle> grains;
   std::vector<fluid::FluidParticle> fluid; fpx::FxVec3 gravity; fpx::fx dt, groundY, h; }` (GF1 needs `grains`
   + `fluid` + the coupling cell-size `h`; the rest carried for GF2–GF6). Both pools share the SAME Q16.16 world.
2. **`MakeCGFGrid` (the shared grid over BOTH pools' union AABB).** Compute the per-axis cell-bound union of the
   grain cells and the fluid cells (the `MakeGrainGrid`/`MakeGrid` cell bounds), at cell-size `h`. Returns a
   `CGFGrid{h, cellMin, gridDim}` total over BOTH pools (every grain AND fluid cell in `[0,gridDim)`). Empty
   pools → a 1×1×1 grid (deterministic). Pure int32.
3. **`BuildCGFNeighbors` (the two cross-pool count→scan→emit lists).** Build the shared grid + the grain cell
   table + the fluid cell table (reuse GR2/FL2). Then `gfNeighbors` (per grain, the fluid cell table 27-cell
   stencil, `|dx|<h` accept) + `fgNeighbors` (per fluid, the grain cell table 27-cell stencil, `|dx|<h` accept),
   each a CSR (`gfStart`/`gfNeighbors`, `fgStart`/`fgNeighbors`), grouped by the query particle ascending,
   fixed stencil order → deterministic. The GPU `cgf_gf_*`/`cgf_fg_*` mirror this byte-for-byte.
4. **The int32 shaders (ALL in `hf_gen_msl`) + the reused cell-table passes.** `cgf_gf_{count,scan,emit}` +
   `cgf_fg_{count,scan,emit}` (the two cross-pool queries) — pure int32 → MSL-native, a TRUE GPU pass on both
   backends. Reuse the existing `grain_cell_*` + `fluid_cell_*` for the two cell tables (sized to the shared
   grid — pass the shared `CGFGrid` cellMin/gridDim). Report all in `hf_gen_msl`.
5. **Showcase `--cgf-query-shot <out>` (Vulkan) AND `--cgf-query` (Metal) — WIRE BOTH.** A small scene: a grain
   bed (`InitGrainBlock`, settled) + a fluid block (`InitBlock`) dropped/overlapping it (so the two pools
   interpenetrate → non-trivial cross neighbours). Build the shared grid + both cell tables + both cross lists
   on the GPU → **memcmp vs the CPU `BuildCGFNeighbors` reference** → color each particle by its cross-pool
   neighbour count (a heat viz: grains by their fluid-neighbour count, fluid by their grain-neighbour count).
   Golden = `tests/golden/metal/cgf_query.png` (Mac-baked by the CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU `{grainCellTable, fluidCellTable, gfStart, gfNeighbors, fgStart,
     fgNeighbors}` == the CPU reference byte-for-byte. Print `cgf-query: {grains:<G>, fluid:<F>, gf:<X>, fg:<Y>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two builds → identical. Print `cgf-query determinism: two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the cross-neighbour counts are coherent (particles in the overlap region
     have cross-pool neighbours; particles far from the other pool have 0). Print `cgf-query coverage: {gf:<X>,
     fg:<Y>} cross-pool pairs (overlap populated, separated empty)`. (Symmetry sanity: every grain↔fluid pair
     appears once in `gf` and once in `fg` — assert `X == Y` if the same `h`.)
   - **(4) empty no-op:** the two pools fully separated (no overlap) → 0 cross neighbours. Print `cgf-query
     separated: 0 cross-pool neighbours (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/cgf_query.png`; do NOT commit it.** Existing 147 image
     goldens UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict — a real GPU pass on both):** Vulkan GPU == Metal GPU == golden, ZERO
   differing pixels.
8. **Tests `tests/cgf_test.cpp` (NEW, pure CPU):** `MakeCGFGrid` (the union bounds over a known two-pool set);
   `BuildCGFNeighbors` (a hand-laid grain+fluid set → the exact `gfNeighbors`/`fgNeighbors` cross lists, the
   symmetry `X==Y`, none for separated pools); the CSR offsets correct. Clean under `windows-msvc-asan`.
   Register `cgf_test` in `tests/CMakeLists.txt`.
9. **Introspect.** Add exactly `deterministic-cgf-query` (features) + `--cgf-query-shot` (showcases). **REBAKE
   `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2/CP2 lesson —
   `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the FL2/GR2/CG1 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` +
  `couple.h` + `couple_grain.h` + `engine/physics/` UNCHANGED (couple_gf is a NEW additive sibling header
  `#include`-ing grain.h + fluid.h read-only). Report the seam is empty.

## Out of scope (YAGNI — later GF slices)
Buoyancy/seepage fluid→grain (GF2), contact reaction grain→fluid (GF3), the coupled step (GF4), lockstep
(GF5), the lit render (GF6). The exact radial cull (deferred to GF2/GF3), three-pool coupling. GF1 claims
ONLY: a unified grain+fluid world and the two cross-pool neighbour lists, bit-identical CPU↔Vulkan↔Metal (a
true GPU pass on both), with the heat-viz integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 98) + the new `cgf_test`. Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--cgf-query-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image
   shows the cross-pool neighbour heat viz (pixel-check the overlap region — grains + fluid coloured by their
   cross-neighbour count) — do NOT trust the proof's claim alone (the NAV6/CL6 lesson).**
3. Metal: `visual_test --cgf-query` → new golden `tests/golden/metal/cgf_query.png`; two runs DIFF 0.0000 (gate
   on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm the cgf cross-query shaders
   MSL-generate (in `hf_gen_msl`).** Cross-vendor Vulkan-vs-Metal STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgf_query.png` added; the other
   147 byte-identical. `git diff master --stat -- tests/golden` = ONLY `cgf_query.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgf-query` + `--cgf-query-shot`; introspect test updated.
   (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` + `couple.h` +
   `couple_grain.h` + `engine/physics/` byte-unchanged). `scripts/verify.ps1` updated: `cgf_query` golden in
   the Mac loop + `--cgf-query-shot` in `$vkShots`. The cgf cross-query shaders in BOTH the DXC list AND
   `hf_gen_msl`.
