# Slice CP1 — Deterministic Rigid↔Fluid Coupling: UNIFIED COUPLED WORLD + BODY→FLUID GRID-HASH QUERY (the BEACHHEAD of FLAGSHIP #11) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice of FLAGSHIP #11 (DETERMINISTIC
> TWO-WAY RIGID↔FLUID COUPLING, `hf::sim::couple`) — the natural 5th act of the deterministic-sim arc: not
> another isolated body, but making the EXISTING bodies INTERACT. The four sim members (rigid `fpx`, cloth,
> fluid, grain) each live in their own world and only touch STATIC kinematic colliders — no two simulated
> bodies ever exchange momentum. This flagship couples a dynamic `fpx::FxBody` to the bit-exact PBF fluid:
> buoyancy + drag + displacement, all in one Q16.16 world, lockstep/rollback-replayable. The headline: a
> barrel BOBBING in water that two netcode peers re-simulate bit-for-bit from inputs alone — a genuine first
> (UE5's fluid-rigid coupling is float/non-deterministic). CP1 is ONLY the unified world + the body→fluid
> neighbour QUERY (which fluid particles each body contains) — the link, NO exchange yet (CP2 buoyancy/drag,
> CP3 displacement, CP4 the coupled step, CP5 lockstep, CP6 render). Branch: `slice-cp1`. See [[hazard-forge-couple-roadmap]] (to be created).

**Goal:** Create `engine/sim/couple.h` (header-only, namespace `hf::sim::couple`, `#include "sim/fpx.h"` +
`"sim/fluid.h"` read-only) with a `CoupleWorld` (a `std::vector<fpx::FxBody>` + the `fluid` particle pool +
config) and `GatherBodyParticles(world)`: for each body, find the fluid particles inside its sphere via the
FL2 grid-hash, producing a per-body CSR candidate list (count→scan→emit). Add a `couple_body_query.comp`
trio (int32 → **MSL-native**). Add `--couple-query-shot` (Vulkan) / `--couple-query` (Metal). Bake the
integer golden `couple_query`. NO new RHI. `fpx.h`/`fluid.h`/`cloth.h`/`grain.h` + their 4 golden sets
UNTOUCHED (couple is the additive sibling).

## Design call: PURE INT32 → MSL-native (the FL2/GR2 precedent) — strict zero-diff BOTH backends
The body→fluid query is integer index arithmetic + a per-axis box reject (`|body.pos.axis − particle.pos.axis|
< body.radius`, the FL2 `NeighborAccept` shape — `fx` is int32 → a PURE INT32 compare, NO products, NO int64,
NO sqrt). So the GPU shaders MSL-generate natively → a TRUE GPU pass on both Vulkan AND Metal (the strongest
cross-vendor proof, like GR2 — strict zero-differing-pixel). The exact radial cull (`|p−body| < radius`
sphere test) is DEFERRED to CP2's force (the buoyant/drag impulse is 0 outside the sphere), so the
over-inclusive box candidate is correct — exactly FL2/GR2's "over-inclusive box, exact cull deferred"
discipline. Bar: strict INTEGER — Vulkan GPU == Metal GPU == golden, ZERO differing pixels; the GPU
`{cellTable, perBodyList}` == the CPU reference byte-for-byte.

## The body→fluid query (the one new shape vs GR2's grain→grain)
GR2 finds, per GRAIN, its neighbour grains in a 27-cell stencil (the search radius ≈ a grain diameter, so
the stencil is fixed 3×3×3). CP1 finds, per BODY, the fluid particles inside the body's sphere — and a body
radius is typically MANY fluid cells wide, so the body spans a RANGE of cells (its `BodyAabb` in cell space),
NOT a fixed 27-cell stencil. So `GatherBodyParticles` iterates the cell range `[CellOf(body.pos − radius) ..
CellOf(body.pos + radius)]` (the `fpx::BodyAabb` quantised to fluid cells), and for each fluid particle in
those cells accepts iff the per-axis box reject `|body.pos.axis − p.pos.axis| < body.radius` passes. Built
by count→scan→emit (CSR `bodyStart[bodyCount+1]` + `bodyParticles[]`, grouped by body, ascending particle
index — the GR2/FL2 EMIT-order discipline, fully deterministic). Pure int32.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The fluid grid-hash to REUSE (`engine/sim/fluid.h`, read-only):** `FluidGrid`/`MakeGrid`/`CellOf`/
  `FlatCellId`/`CellCount` (`fluid.h:226-274`), `FluidCellTable`/`BuildCellTable` (the count→scan→emit cell
  bucketing of fluid particles, `fluid.h:283-315`), `FluidParticle` (`fluid.h:82-88`). The body→particle
  query iterates the body's cell range over THIS cell table.
- **The rigid body to REUSE (`engine/sim/fpx.h`, read-only):** `FxBody` (`fpx.h:116-131` — pos, vel,
  invMass, flags, radius, orient, angVel), `FxAabb`/`BodyAabb` (`fpx.h:210-220`, the body's integer AABB),
  `kFlagDynamic` (`fpx.h:133`). DO NOT modify fpx.h/fluid.h/cloth.h/grain.h — couple is the additive sibling.
- **The query mold to MIRROR (`engine/sim/grain.h` GR2, this is the closest twin):** `GrainNeighborAccept`
  (the per-axis box reject), `CountGrainNeighbors`/`BuildGrainNeighborList` (the count→scan→emit CSR over a
  cell stencil) — CP1's `CountBodyParticles`/`GatherBodyParticles` are the SAME shape with a body-AABB cell
  RANGE instead of a 27-cell stencil, and `body.radius` instead of `hSearch`.
- **The int32 grid SHADERS to mirror (`shaders/grain_neighbor_{count,scan,emit}.comp.hlsl` +
  `grain_cell_*`):** the GR2 int32 passes — CP1 reuses the fluid cell-table build (the FL2 `fluid_cell_*`
  passes already exist + are MSL-native) and adds `couple_body_{count,scan,emit}.comp` (the per-body query,
  int32 → in `hf_gen_msl`). Report which existing passes are reused vs new.
- **Showcase + registration:** GR2's `--grain-neighbors-shot` / FL2's `--fluid-neighbors-shot` (the heat
  viz); `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden** — the GR2 lesson), `tests/couple_test.cpp` (NEW test target — register in
  `tests/CMakeLists.txt`). main.cpp has `/bigobj`.

## Design decisions (locked)
1. **`CoupleWorld` (the unified world — bodies + fluid in one Q16.16 frame).** `struct CoupleWorld {
   std::vector<fpx::FxBody> bodies; std::vector<fluid::FluidParticle> particles; fluid::FluidKernel kernel;
   fpx::FxVec3 gravity; fpx::fx dt, groundY; }` (CP1 only needs `bodies` + `particles` + the grid cell-size
   from `kernel.h`; the rest is carried for CP2–CP6). The bodies and the fluid share the SAME world units
   (the CL4/GR3 deformable-meets-rigid precedent — `fpx::FxBody` and `fluid::FluidParticle` are both Q16.16).
2. **`GatherBodyParticles` (the count→scan→emit body→fluid query).** Build the fluid `FluidGrid` +
   `FluidCellTable` (reuse FL2). For each body, iterate the cell range covering `BodyAabb(body)`; for each
   fluid particle `p` in those cells, accept iff `|body.pos.axis − p.pos.axis| < body.radius` on every axis
   (pure int32 box reject). Emit accepted particle indices into the body's CSR slice in ascending order.
   Returns `CoupleQuery{ std::vector<uint32_t> bodyStart; std::vector<uint32_t> bodyParticles; }`. The GPU
   `couple_body_{count,scan,emit}` mirror this byte-for-byte. (DET-CRUX, the GR2/FL2 lesson: the per-body
   EMIT is the fixed-order scatter; the count + the per-body lists are per-body-disjoint → race-free.)
3. **`couple_body_{count,scan,emit}.comp` (3 int32 shaders, ALL in `hf_gen_msl`) + the reused FL2
   `fluid_cell_*` passes.** Pure int32 → MSL-native, a TRUE GPU pass on both backends. Report all 3 in
   `hf_gen_msl`. (CP1 has NO int64 — it is the int32-native beachhead like GR2.)
4. **Showcase `--couple-query-shot <out>` (Vulkan) AND `--couple-query` (Metal) — WIRE BOTH.** A small scene:
   a fluid pool (the FL1 `InitBlock`, e.g. settled a few steps) + 1–few `FxBody` spheres placed partly
   submerged in the pool. Build the grid + cell table + per-body query on the GPU → **memcmp vs the CPU
   `GatherBodyParticles` reference** → color each fluid particle by which body contains it (a per-body heat
   viz: particles inside body 0 one colour, body 1 another, un-gathered grey). Golden =
   `tests/golden/metal/couple_query.png` (Mac-baked by the CONTROLLER — DO NOT commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU `{cellStart, cellParticles, bodyStart, bodyParticles}` == the CPU
     reference byte-for-byte. Print `couple-query: {bodies:<B>, particles:<N>, gathered:<G>, maxPerBody:<M>}
     GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two builds → identical. Print `couple-query determinism: two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the gathered particles are coherent (each submerged body gathers the
     particles in its sphere; a body floating clear of the pool gathers 0). Print `couple-query coverage: <G>
     body-particle pairs (submerged bodies populated, clear bodies empty)`.
   - **(4) empty no-op:** zero bodies (or all bodies clear of the fluid) → 0 gathered. Print `couple-query
     empty: 0 gathered (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/couple_query.png`; do NOT commit it.** Existing 135 image
     goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict — the strongest, a real GPU pass on both):** Vulkan GPU == Metal GPU
   == golden, ZERO differing pixels.
7. **Tests `tests/couple_test.cpp` (NEW, pure CPU):** `GatherBodyParticles` — a hand-laid pool + one body →
   the exact gathered set (particles inside the box, none outside); a body clear of the pool → empty; two
   bodies → disjoint per-body lists in ascending order; the CSR offsets correct. Clean under
   `windows-msvc-asan`. Register `couple_test` in `tests/CMakeLists.txt`.
8. **Introspect.** Add exactly `deterministic-couple-query` (features) + `--couple-query-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2
   lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the FL2/GR2 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `fluid.h` + `cloth.h` + `grain.h` +
  `engine/physics/` UNCHANGED (couple is a NEW additive sibling header `#include`-ing fpx.h + fluid.h
  read-only). Report the seam is empty.

## Out of scope (YAGNI — later CP slices)
Buoyancy/drag forces fluid→body (CP2), the fluid displacement body→fluid (CP3), the coupled step (CP4),
lockstep/rollback (CP5), the lit render (CP6). The exact radial sphere cull (deferred to CP2's force, a no-op
outside the sphere), body-body coupling, multiple fluids. CP1 claims ONLY: a unified bodies+fluid world and
the per-body fluid-particle query, bit-identical CPU↔Vulkan↔Metal (a true GPU pass on both), with the heat-viz
integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 96) + the new `couple_test`. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--couple-query-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the per-body gathered-particle heat viz (pixel-check the coloured submerged regions) — do NOT
   trust the proof's claim alone (the NAV6/CL6 lesson).**
3. Metal: `visual_test --couple-query` → new golden `tests/golden/metal/couple_query.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm the 3 couple
   query shaders MSL-generate (in `hf_gen_msl`).** Cross-vendor Vulkan-vs-Metal STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `couple_query.png` added; the
   other 135 byte-identical. `git diff master --stat -- tests/golden` = ONLY `couple_query.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-couple-query` + `--couple-query-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h` + `fluid.h` + `cloth.h` + `grain.h` +
   `engine/physics/` byte-unchanged). `scripts/verify.ps1` updated: `couple_query` golden in the Mac loop +
   `--couple-query-shot` in `$vkShots`. The 3 couple query shaders in BOTH the DXC list AND `hf_gen_msl`.
