# Slice FL2 — Deterministic GPU Fluid: GRID-HASH NEIGHBOR SEARCH (int32-native) (Phase 14 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #9
> (DETERMINISTIC GPU FLUID via Position-Based Fluids, `hf::sim::fluid`, header `engine/sim/fluid.h`).
> Builds the per-particle NEIGHBOR LIST over the FL1 particle pool via a uniform spatial-hash grid — the
> candidate set the FL3 density gather + FL4 PBF solve iterate. Built by the proven count→scan→emit
> compaction, PURE INT32 → Metal-MSL-native (a true GPU pass on both backends, unlike FL1's int64
> integrate — the FPX2/CL2 precedent). Strict zero-diff cross-backend (the integer bar). ZERO new RHI.
> Branch: `slice-fl2`. See [[hazard-forge-fluid-roadmap]].

**Goal:** Extend `engine/sim/fluid.h` with `CellOf` (a particle's int32 grid cell at cell-size `h` =
the PBF smoothing radius, via `fpx::BroadphaseCell`/`FloorDiv`), `BuildCellTable` (bucket the particles
into cells — count→scan→emit of particle indices per cell), and `BuildNeighborList` (per particle, gather
the candidate neighbours from the 27-cell stencil — same + adjacent cells — with a PURE INT32 per-axis
`|dx| < h` reject; the exact radial `r < h` cull is deferred to FL3's kernel which is 0 beyond `h`). Lay
the variable-count output out with the count→scan→emit compaction. Add `shaders/fluid_cell_{count,scan,
emit}.comp.hlsl` + `shaders/fluid_neighbor_{count,scan,emit}.comp.hlsl` (or a combined set — int32 →
Metal-native), the `fluid_neighbors` integer golden (the neighbor-list / per-cell-count viz, CPU-colored
from the integer read-back → strict zero-diff cross-backend), `--fluid-neighbors-shot` (Vulkan) /
`--fluid-neighbors` (Metal), and `tests/fluid_test.cpp` additions. Reuse FL1 verbatim (FluidParticle /
InitBlock) — FL2 is additive (FL1's integrate pipeline + golden stay byte-identical).

## Design call: INTEGER bit-exact, int32-NATIVE (the FPX2 twin)
The whole neighbor search is integer INDEX arithmetic: cell ids are `BroadphaseCell` (FloorDiv per axis,
int32), the cell bucketing is count→scan→emit of particle indices, and the candidate reject is a per-axis
`|p_i.axis - p_j.axis| < h` compare — `fx` is int32, so this is a PURE INT32 compare (no squaring, no
int64). **The exact radial distance `r < h` (the circle-vs-box refinement) is NOT done in FL2** — it is
deferred to FL3, where the kernel `W(r,h)` is naturally 0 for `r >= h`, so an over-inclusive box-candidate
list is correct (the extra candidates contribute 0 density). This keeps FL2 pure int32 → the shaders
MSL-generate NATIVELY and run as true GPU passes on both backends (the FPX2 `fpx_pair_*` / CL2
`cloth_edge_*` precedent — int32, MSL-native, strict zero-diff by construction, unlike the int64
fluid_integrate). The golden is the CPU-colored integer neighbor buffer → Vulkan==Metal BIT-IDENTICAL. NO
new RHI.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FL1 pool (the input):** `engine/sim/fluid.h` — `FluidParticle`, `InitBlock`. FL2 adds `CellOf` +
  `BuildCellTable` + `BuildNeighborList` + the `NeighborList` layout to this same header; FL1 functions
  UNCHANGED.
- **The grid-hash to REUSE (read-only):** `engine/sim/fpx.h::BroadphaseCell`/`CellId`/`FloorDiv` (the
  int32 spatial cell math — FPX2's broadphase). The cell-size is `h` (the PBF radius); `CellOf(p, h)` =
  `BroadphaseCell` per axis.
- **The count→scan→emit structural template (int32, MSL-native):** `engine/sim/fpx.h::CountPairs`/
  `BuildPairs` + `shaders/fpx_pair_count.comp.hlsl` + `fpx_pair_scan.comp.hlsl` (`[numthreads(1,1,1)]`
  serial exclusive prefix-sum) + `fpx_pair_emit.comp.hlsl` — the FPX2 twin. Also CL2's `cloth_edge_*` +
  NAV1's `nav_raster_*`. Mirror EXACTLY: count per particle (its candidate neighbours), single-thread
  prefix-sum, per particle emit at offset in a FIXED order (ascending cell, then ascending particle index
  — so the neighbor list is deterministic).
- **The integer-golden showcase discipline:** FL1's `--fluid-integrate-shot` + FPX2's `--fpx-pairs-shot`
  (ReadBuffer the integer result, memcmp GPU==CPU, CPU-color the integer buffer, strict zero-diff). For
  neighbours, draw the per-particle neighbor count (a heat color) or the cell-occupancy grid.
- **RHI compute envelope (must fit, ZERO additions):** `engine/rhi/rhi.h` — BindStorageBuffer,
  DispatchCompute, ComputeToComputeBarrier, ComputePushConstants, ReadBuffer, InterlockedAdd (the FL1/FPX2
  set).
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--fluid-neighbors-shot` standalone arg
  branch + DXC list), `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunFluidNeighborsShowcase` +
  `--fluid-neighbors` + the new shaders in `hf_gen_msl` — int32 → Metal-native),
  `engine/editor/introspect.cpp` (+`deterministic-fluid-neighbors` feature + `--fluid-neighbors-shot`
  showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1` (`fluid_neighbors`
  golden in the Mac loop + `--fluid-neighbors-shot` in `$vkShots`).

## Design decisions (locked)
1. **The grid + cell table.** Cell-size = `h` (the PBF radius, a config Q16.16). `CellOf(p, h)` =
   `(FloorDiv(p.x, h), FloorDiv(p.y, h), FloorDiv(p.z, h))` → a flat `CellId` (the `fpx` linearization,
   bounded to a grid extent, or a hash mod table size — pick a deterministic scheme, document; a bounded
   dense grid over the dam-break AABB is simplest + fully deterministic). `BuildCellTable`: count particles
   per cell → exclusive prefix-sum → scatter particle indices into `cellParticles[]` at the cell offset
   (the count→scan→emit). Particles within a cell ordered by ascending particle index (deterministic).
2. **The neighbor list.** `BuildNeighborList(particles, cellTable, h, maxNeighbors)`: per particle `i`,
   scan the 27 cells of its 3×3×3 stencil; for each particle `j != i` in those cells, accept as a candidate
   iff the per-axis `|p_i.axis - p_j.axis| < h` reject passes (PURE INT32 — `fx` is int32). Emit the
   accepted `j` indices into `neighbors[]` at `i`'s offset, in a FIXED order (ascending stencil-cell, then
   ascending `j`). Count→scan→emit lays out the variable-length per-particle lists. (Cap per-particle
   neighbours at a `maxNeighbors` if needed for buffer sizing — document; the dam-break density bounds it.)
   **No radial `r<h` cull, no int64 — the box-candidate list is correct since FL3's kernel is 0 beyond h.**
3. **GPU pipeline (count→scan→emit, the FPX2 mirror).** The cell-table + neighbor-list passes are int32
   count→scan→emit (one thread per particle/cell counts, single-thread prefix-sum scan, one thread emits at
   offset). **All int32 → in `hf_gen_msl`, Metal-native.** GPU `{cellTable, neighbors, offsets}` == the CPU
   `BuildCellTable`+`BuildNeighborList` reference byte-for-byte.
4. **Showcase `--fluid-neighbors-shot <out>` (Vulkan) AND `--fluid-neighbors` (Metal — WIRE BOTH; confirm
   visual_test.mm + `#include "sim/fluid.h"`).** The FL1 dam-break block (a settled or mid-fall state) →
   BuildCellTable → BuildNeighborList. ReadBuffer the integer `neighbors` + `offsets` (+ per-particle
   neighbor count); **memcmp GPU == the CPU reference (the make-or-break)**; CPU-color a per-particle
   neighbor-count heat viz (or the cell-occupancy grid) → `tests/golden/metal/fluid_neighbors.png` (baked
   on the Mac by the CONTROLLER — DO NOT commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance / GPU==CPU (make-or-break):** `neighbors` + `offsets` (+ cell table) equal the CPU
     reference byte-for-byte. Print `fluid-neighbors: {particles:<N>, cells:<C>, neighbors:<E>, maxPer:<M>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs BYTE-IDENTICAL. Print `fluid-neighbors determinism: two runs
     BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the neighbor lists are symmetric-ish + bounded (every neighbor j of i
     is within h per-axis of i; interior particles have more neighbours than surface ones — a coherent
     density). Print `fluid-neighbors coverage: <E> neighbor-pairs over <N> particles (all within h)`.
   - **(4) empty / sparse:** a single particle (or particles spread > h apart) → 0 neighbours → cleared.
     Print `fluid-neighbors sparse: 0 neighbors (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/fluid_neighbors.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 124 image goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict).** The neighbor buffer is pure int32, host-snapped → ZERO GPU
   float → Vulkan==Metal BIT-IDENTICAL: the golden is the CPU-colored integer read-back; the controller's
   cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare. Any nonzero cross-backend diff is a bug.
7. **Tests `tests/fluid_test.cpp` additions (pure CPU):** `CellOf` (incl negative coords via FloorDiv); a
   2-particle case within h → mutual neighbours; > h apart → no neighbours; the per-axis reject (a particle
   just inside/outside h on one axis); a small block → the expected neighbor counts; no particle is its own
   neighbour; determinism. Clean under `windows-msvc-asan`.
8. **Introspect.** Add exactly `deterministic-fluid-neighbors` (features) + `--fluid-neighbors-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None expected.** Pure SSBO compute (BindStorageBuffer, DispatchCompute, ComputeToComputeBarrier,
  ComputePushConstants, ReadBuffer, InterlockedAdd — the FL1/FPX2 set). `rhi.h` + `rhi_factory` (baseline
  2) + backend dirs UNCHANGED. FL1's `fluid_integrate.comp` + `engine/sim/fpx.h` + `engine/sim/cloth.h` +
  `engine/physics/` UNCHANGED. Report the seam.

## Out of scope (YAGNI — later FL slices)
The density + λ kernel (FL3 — consumes this neighbor list), the PBF solve (FL4), lockstep (FL5), the float
render (FL6). The exact radial `r<h` cull (deferred to FL3's kernel = 0 beyond h). FL2 is ONLY the
spatial-hash candidate neighbor list + its bit-exact golden. No density, no float, no int64.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 95) + the new `fluid_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fluid-neighbors-shot` on Vulkan: GPU==CPU memcmp BIT-EXACT + determinism +
   coverage + sparse no-op; a coherent neighbor-density image. Run under the Vulkan-validation gate → ZERO
   VUID (set BOTH `VK_LAYER_PATH` to the conan `...\.conan2\p\...\layers` dir AND
   `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer LOADED = zero "not found").
3. Metal: `visual_test --fluid-neighbors` → new golden `tests/golden/metal/fluid_neighbors.png`; two runs
   DIFF 0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm the new
   fluid cell/neighbor shaders MSL-generate (int32 → in `hf_gen_msl`).** Cross-backend = STRICT
   ZERO-DIFFERING-PIXEL, NOT the float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fluid_neighbors.png`
   added; the other 124 byte-identical (FL1 fluid_integrate + all existing untouched). `git diff master
   --stat -- tests/golden` = ONLY `fluid_neighbors.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fluid-neighbors` + `--fluid-neighbors-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI; shaders int32-only, no int64_t). `scripts/verify.ps1`
   updated: `fluid_neighbors` golden in the Mac loop + `--fluid-neighbors-shot` in `$vkShots`. FL1 +
   `engine/sim/fpx.h` + `engine/sim/cloth.h` + `engine/physics/` UNTOUCHED.
