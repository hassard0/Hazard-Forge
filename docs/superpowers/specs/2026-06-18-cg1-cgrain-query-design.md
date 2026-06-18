# Slice CG1 — Deterministic Rigid↔Grain Coupling: UNIFIED WORLD + BODY→GRAIN GRID-HASH QUERY (the BEACHHEAD of FLAGSHIP #12) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice of FLAGSHIP #12 (DETERMINISTIC
> TWO-WAY RIGID↔GRAIN COUPLING, `hf::sim::cgrain`) — the SECOND material-interaction pairing (after the
> rigid↔fluid CP flagship): a dynamic `fpx::FxBody` coupled to the bit-exact FRICTIONAL granular pile. Drop a
> heavy body onto/into a poured sand bed — it SINKS, the sand piles around and SUPPORTS it, it settles
> half-buried, and two peers re-simulate the whole sink + pile bit-for-bit. Strictly harder than CP: grain has
> the extra friction physics (GR4) and supports the body through a MANY-CONTACT bed (not a buoyant volume).
> UE5 has no deterministic granular at all, let alone deterministic rigid↔granular coupling. CG1 is ONLY the
> unified world + the body→grain neighbour QUERY (which grains each body contains) — the link, NO exchange yet
> (CG2 support/drag, CG3 displacement, CG4 the coupled step, CG5 lockstep, CG6 render). The CP1 twin with the
> grain grid. Branch: `slice-cg1`. See [[hazard-forge-couple-grain-roadmap]] (to be created).

**Goal:** Create `engine/sim/couple_grain.h` (header-only, namespace `hf::sim::cgrain`, `#include "sim/fpx.h"` +
`"sim/grain.h"` read-only) with a `CGrainWorld` (a `std::vector<fpx::FxBody>` + the `grain` pool + config) and
`GatherBodyGrains(world)`: for each body, find the grain indices inside its sphere via the GR2 grid-hash,
producing a per-body CSR candidate list (count→scan→emit). Add a `cgrain_body_query.comp` trio (int32 →
**MSL-native**). Add `--cgrain-query-shot` (Vulkan) / `--cgrain-query` (Metal). Bake the integer golden
`cgrain_query`. NO new RHI. `fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple.h` + their golden sets UNTOUCHED
(couple_grain is the additive sibling).

## Design call: PURE INT32 → MSL-native (the CP1/FL2/GR2 precedent) — strict zero-diff BOTH backends
The body→grain query is integer index arithmetic + a per-axis box reject (`|body.pos.axis − grain.pos.axis| <
body.radius`, the CP1/FL2 `BodyParticleAccept` shape — `fx` is int32 → a PURE INT32 compare, NO products, NO
int64, NO sqrt). So the GPU shaders MSL-generate natively → a TRUE GPU pass on both Vulkan AND Metal (the
strongest cross-vendor proof, like CP1/GR2 — strict zero-differing-pixel). The exact radial sphere cull (`|g−
body| < radius`) is DEFERRED to CG2's force/CG3's projection (the support/displacement is 0 outside the
sphere), so the over-inclusive box candidate is correct — exactly the CP1/FL2/GR2 "over-inclusive box, exact
cull deferred" discipline. Bar: strict INTEGER — Vulkan GPU == Metal GPU == golden, ZERO differing pixels;
the GPU `{cellTable, perBodyList}` == the CPU reference byte-for-byte.

## The body→grain query (the CP1 twin — grain grid instead of fluid grid)
This is `couple.h`'s CP1 `GatherBodyParticles` near-verbatim, with the FLUID grid swapped for the GRAIN grid:
build the grain `GrainGrid` + `GrainCellTable` (reuse GR2 `MakeGrainGrid`/`BuildGrainCellTable` at the grain's
`hSearch` cell-size); for each body, iterate the cell range covering `BodyAabb(body)` quantised to grain cells
(`grain::GrainCellOf`); for each grain in those cells accept iff `|body.pos.axis − g.pos.axis| < body.radius`
on every axis (pure int32 box reject). Built by count→scan→emit (CSR `bodyStart[bodyCount+1]` +
`bodyGrains[]`, grouped by body, ascending grain index — the CP1/GR2 EMIT-order discipline, fully
deterministic). Pure int32. (The body radius is typically MANY grain cells wide, so the body spans a cell
RANGE, not a 27-cell stencil — exactly CP1's body-AABB cell-range walk.)

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CP1 body→fluid query to MIRROR near-verbatim (`engine/sim/couple.h`):** `CoupleWorld`,
  `BodyParticleAccept` (the per-axis box reject), `CountBodyParticles`/`GatherBodyParticles` (the body-AABB
  cell-range count→scan→emit CSR, `couple.h:105-197`), `CoupleQuery{bodyStart, bodyParticles}`. CG1 is the
  SAME with `grain::GrainParticle` instead of `fluid::FluidParticle` and the grain grid.
- **The grain grid-hash to REUSE (`engine/sim/grain.h`, read-only):** `GrainGrid`/`MakeGrainGrid`/`GrainCellOf`/
  `FlatGrainCellId`/`GrainCellCount` (GR2), `GrainCellTable`/`BuildGrainCellTable` (the count→scan→emit cell
  bucketing of grains), `GrainParticle` (pos, radius, flags). The body→grain query iterates the body's cell
  range over THIS cell table at the grain `hSearch`.
- **The rigid body to REUSE (`engine/sim/fpx.h`, read-only):** `FxBody` (pos, vel, invMass, flags, radius),
  `FxAabb`/`BodyAabb` (the body's integer AABB), `kFlagDynamic`. DO NOT modify fpx.h/grain.h/fluid.h/cloth.h/
  couple.h — couple_grain is the additive sibling.
- **The int32 grid SHADERS to mirror (`shaders/couple_body_{count,scan,emit}.comp.hlsl` — CP1):** the CP1
  int32 query passes — CG1's `cgrain_body_{count,scan,emit}.comp` are the SAME shape over the grain cell table.
  Reuse the GR2 `grain_cell_*` passes for the cell-table build (already exist + MSL-native) + add the 3
  `cgrain_body_*` passes (int32 → in `hf_gen_msl`). Report which existing passes are reused vs new.
- **Showcase + registration:** CP1's `--couple-query-shot` / GR2's `--grain-neighbors-shot` (the heat viz);
  `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect
  JSON golden** — the GR2/CP2 lesson), `tests/cgrain_test.cpp` (NEW test target — register in
  `tests/CMakeLists.txt`). main.cpp has `/bigobj`.

## Design decisions (locked)
1. **`CGrainWorld` (the unified world — bodies + grains in one Q16.16 frame).** `struct CGrainWorld {
   std::vector<fpx::FxBody> bodies; std::vector<grain::GrainParticle> grains; fpx::FxVec3 gravity; fpx::fx dt,
   groundY, hSearch; }` (CG1 needs `bodies` + `grains` + the grain `hSearch` cell-size; the rest carried for
   CG2–CG6). Bodies and grains share the SAME Q16.16 world units (the GR3 `GrainSphereFromBody` precedent —
   `fpx::FxBody` and `grain::GrainParticle` already interoperate).
2. **`GatherBodyGrains` (the count→scan→emit body→grain query).** Build the grain `GrainGrid` +
   `GrainCellTable` (reuse GR2). For each body, iterate the cell range covering `BodyAabb(body)`; for each
   grain `g` in those cells, accept iff `|body.pos.axis − g.pos.axis| < body.radius` on every axis (pure int32
   box reject). Emit accepted grain indices into the body's CSR slice in ascending order. Returns
   `CGrainQuery{ std::vector<uint32_t> bodyStart; std::vector<uint32_t> bodyGrains; }`. The GPU
   `cgrain_body_{count,scan,emit}` mirror this byte-for-byte. (DET-CRUX, the CP1/GR2 lesson: per-body EMIT is
   the fixed-order scatter; count + per-body lists are per-body-disjoint → race-free.)
3. **`cgrain_body_{count,scan,emit}.comp` (3 int32 shaders, ALL in `hf_gen_msl`) + the reused GR2
   `grain_cell_*` passes.** Pure int32 → MSL-native, a TRUE GPU pass on both backends. Report all 3 in
   `hf_gen_msl`. (CG1 has NO int64 — the int32-native beachhead like CP1/GR2.)
4. **Showcase `--cgrain-query-shot <out>` (Vulkan) AND `--cgrain-query` (Metal) — WIRE BOTH.** A small scene:
   a grain bed (the GR1 `InitGrainBlock`, e.g. settled a few `StepGrainFriction` steps into a pile) + 1–few
   `FxBody` spheres placed partly buried in the bed. Build the grid + cell table + per-body query on the GPU
   → **memcmp vs the CPU `GatherBodyGrains` reference** → color each grain by which body contains it (a
   per-body heat viz: grains inside body 0 one colour, body 1 another, un-gathered grey). Golden =
   `tests/golden/metal/cgrain_query.png` (Mac-baked by the CONTROLLER — DO NOT commit).
5. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU `{cellStart, cellGrains, bodyStart, bodyGrains}` == the CPU reference
     byte-for-byte. Print `cgrain-query: {bodies:<B>, grains:<N>, gathered:<G>, maxPerBody:<M>} GPU==CPU
     BIT-EXACT`.
   - **(2) determinism:** two builds → identical. Print `cgrain-query determinism: two runs BYTE-IDENTICAL`.
   - **(3) coverage / coherence:** the gathered grains are coherent (each buried body gathers the grains in
     its sphere; a body clear of the bed gathers 0). Print `cgrain-query coverage: <G> body-grain pairs
     (buried bodies populated, clear bodies empty)`.
   - **(4) empty no-op:** zero bodies (or all bodies clear of the bed) → 0 gathered. Print `cgrain-query
     empty: 0 gathered (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/cgrain_query.png`; do NOT commit it.** Existing 141 image
     goldens UNTOUCHED.
6. **Cross-backend bar (INTEGER, strict — the strongest, a real GPU pass on both):** Vulkan GPU == Metal GPU
   == golden, ZERO differing pixels.
7. **Tests `tests/cgrain_test.cpp` (NEW, pure CPU):** `GatherBodyGrains` — a hand-laid bed + one body → the
   exact gathered set (grains inside the box, none outside); a body clear of the bed → empty; two bodies →
   disjoint per-body lists in ascending order; the CSR offsets correct. Clean under `windows-msvc-asan`.
   Register `cgrain_test` in `tests/CMakeLists.txt`.
8. **Introspect.** Add exactly `deterministic-cgrain-query` (features) + `--cgrain-query-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2/CP2
   lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the CP1/GR2 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` +
  `couple.h` + `engine/physics/` UNCHANGED (couple_grain is a NEW additive sibling header `#include`-ing fpx.h
  + grain.h read-only). Report the seam is empty.

## Out of scope (YAGNI — later CG slices)
Support/drag forces grain→body (CG2), the grain displacement body→grain (CG3), the coupled step (CG4),
lockstep/rollback (CG5), the lit render (CG6). The exact radial sphere cull (deferred to CG2/CG3, a no-op
outside the sphere), body-body coupling, multiple beds. CG1 claims ONLY: a unified bodies+grains world and the
per-body grain query, bit-identical CPU↔Vulkan↔Metal (a true GPU pass on both), with the heat-viz integer
golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 97) + the new `cgrain_test`. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cgrain-query-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the per-body gathered-grain heat viz (pixel-check the coloured buried regions) — do NOT trust
   the proof's claim alone (the NAV6/CL6 lesson).**
3. Metal: `visual_test --cgrain-query` → new golden `tests/golden/metal/cgrain_query.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm the 3 cgrain query
   shaders MSL-generate (in `hf_gen_msl`).** Cross-vendor Vulkan-vs-Metal STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgrain_query.png` added; the
   other 141 byte-identical. `git diff master --stat -- tests/golden` = ONLY `cgrain_query.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgrain-query` + `--cgrain-query-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple.h` +
   `engine/physics/` byte-unchanged). `scripts/verify.ps1` updated: `cgrain_query` golden in the Mac loop +
   `--cgrain-query-shot` in `$vkShots`. The 3 cgrain query shaders in BOTH the DXC list AND `hf_gen_msl`.
