# Slice CX4 — Deterministic Convex Contacts: THE FULL CONVEX STEP (a settling stack) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #19
> (DETERMINISTIC CONVEX RIGID-BODY CONTACTS, `hf::sim::convex`). CX1 gave the SAT axis, CX2 the manifold, CX3 the
> angular impulse (velocity-only). CX4 assembles them into the FULL per-tick STEP and adds the one missing piece —
> POSITION de-penetration — so a STACK of boxes on a static floor SETTLES into a coherent resting tower (boxes
> interlock and rest instead of rolling off / sinking, impossible in the sphere-sphere fpx solver). Deterministic
> integer host multi-pass driver (the FR4/CP4/CG4/GF4 mold). INTEGER-bit-exact over many ticks. int64 → the
> `convex_step.comp` shader is Vulkan-only + a Metal CPU reference. CX1+CX2+CX3's `convex.h` code + shaders are
> BYTE-FROZEN (CX4 is additive). Branch: `slice-cx4`. See [[hazard-forge-convex-roadmap]].

**Goal:** Extend `engine/sim/convex.h` (additive — CX1+CX2+CX3 byte-unchanged) with the full step: a `ConvexWorld`
(a small set of `FxBody` + `FxBox` colliders, some static) + `StepConvexWorld(world, cfg)` (one deterministic tick:
predict-integrate → all-pairs narrowphase → impulse solve → position de-penetration) + `StepConvexWorldN`. Add the
new int64 shader `shaders/convex_step.comp.hlsl` + `--convex-stack-shot` (Vulkan) / `--convex-stack` (Metal). Bake
the integer golden `convex_stack`. **NO new RHI.**

## Design call: the per-tick multi-pass step (the FR4/CP4 host-driver mold) + position de-penetration

A `ConvexWorld` holds `N` boxes (each an `FxBody{pos, vel, invMass, flags, orient, angVel}` + an `FxBox
{halfExtents}`), a fixed number static (`invMass==0`, e.g. the floor). `StepConvexWorld(world, cfg)` runs ONE tick,
ALL orders PINNED:
1. **Predict-integrate:** for each DYNAMIC body (`invMass != 0` && `flags & kFlagDynamic`), `IntegrateBodyFull(b,
   gravity, dt)` (fpx.h — vel += gravity·dt; pos += vel·dt; orient integrate from angVel). Static bodies untouched.
2. **All-pairs narrowphase (fixed `i<j` order):** for every body pair `(i,j)`, `i<j`, skipping static-static pairs
   (both `invMass==0`), run `BoxSat(bi,boxi,bj,boxj)`. Collect the OVERLAPPING pairs' `SatResult` + the
   `BuildManifold` manifold into a fixed-order list. (HONEST SIMPLIFICATION: all-pairs, NOT the FPX2 BuildPairs
   broadphase — the CX4 scene is a SMALL stack [≤ ~6 boxes]; the AABB broadphase is a scaling refinement, deferred.
   Documented.)
3. **Impulse solve (Gauss-Seidel over the pair list, `cfg.solveIters` sweeps):** for each sweep, for each
   overlapping pair in fixed order, `SolveManifoldImpulse(bi, bj, invIiW, invIjW, manifold, restitution, 1)` (ONE
   inner sweep per outer sweep — the world-level Gauss-Seidel; the world inverse inertias `WorldInvInertia`
   recomputed per outer sweep from the current orient, OR once per tick — DECIDE + document; once-per-tick is the
   common cheap choice and stays deterministic). The per-pair impulse mutates the bodies in place (later pairs see
   earlier updates — the FR4 world Gauss-Seidel).
4. **Position de-penetration (the NEW bit, fixed pair order):** after the velocity solve, re-run `BoxSat` per pair
   (positions have not moved yet this sub-step, but re-deriving keeps the depth current) OR reuse the manifold's max
   depth; for each still-overlapping pair push the two bodies APART along the A→B-corrected normal by
   `corrected = fxmul(max(0, penetration − slop), beta)` split by inverse mass:
   `wi = fxdiv(invMassA, invMassA+invMassB)`, `wj = kOne − wi` (both static → skip; one static → the dynamic takes
   all); `posA -= n·(corrected·wi); posB += n·(corrected·wj)`. `slop` (allowed penetration, ~1/64 unit) + `beta`
   (correction fraction, ~0.8 in Q16.16) reduce jitter — both in `cfg`, FIXED. This is what makes the stack REST
   instead of sink. (Linear position correction only — no positional angular/rotational correction; documented.)
5. (Orientation already integrated in step 1.)

`StepConvexWorldN(world, cfg, ticks)` runs `ticks` steps → the stack settles. `ConvexStepConfig {FxVec3 gravity; fx
dt; uint32_t solveIters; fx restitution; fx slop; fx beta;}`. `MeasureStack(world)` → max body speed (rest test) +
max pairwise penetration (interpenetration test) + the dynamic-body count, deterministic.

**THE int64 REALITY (the CX1/CX2/CX3/FPX3 lesson):** the whole chain is int64 (the inertia/impulse/SAT products).
DXC compiles int64 (Vulkan); glslc cannot. So `convex_step.comp` is **VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl)**; the
Metal `--convex-stack` runs the CPU `StepConvexWorldN` — byte-identical to the Vulkan GPU result BY CONSTRUCTION,
while the Vulkan side carries the GPU==CPU memcmp proof. Document this.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **CX1+CX2+CX3 `engine/sim/convex.h` (read it; APPEND only after `ResolveContactPair`):** `BoxSat`/`SatResult`,
  `BuildManifold`/`ContactManifold`, `FxBoxInvInertiaBody`/`WorldInvInertia`/`SolveManifoldImpulse`,
  `ContactSolveConfig`. CX4 ASSEMBLES these into the world step + ADDS position de-penetration. CX1-CX3 byte-frozen.
- **fpx world + integrate + de-penetration precedent (`engine/sim/fpx.h`, read-only):** `FxBody`, `IntegrateBodyFull`
  (fpx.h:479), `ResolvePair`/the split-by-invMass position correction (fpx.h:337-348 — the `wi = fxdiv(a.invMass,
  invSum)` split CX4's step-4 mirrors for boxes), `kFlagDynamic`, `FxScale`/`FxSub`/`FxAdd`, `fxmul`/`fxdiv`,
  `kOne`/`kFrac`. **CONFIRM signatures. DO NOT modify fpx.h.**
- **The host-driver + shader mold (`shaders/fpx_solve.comp.hlsl` + the FR4/CP4 `--fract-step`/`--couple-step`
  showcases in `samples/hello_triangle/main.cpp`):** the int64 single-thread WHOLE-WORLD step shader + how the host
  drives it per-tick (or the shader loops the ticks internally) + the GPU==CPU final-state memcmp. `convex_step.comp`
  follows this: ONE thread runs the whole `StepConvexWorldN` over the SMALL body set (all-pairs, the few boxes),
  copying the CPU `StepConvexWorld` VERBATIM, writing the final bodies. Confirm `convex_step` NOT in hf_gen_msl.
- **CX3 shader precedent (`shaders/convex_solve.comp.hlsl`):** the int64 ops, the body pack, the VERBATIM CPU copy.
- **The showcase precedents:** CX3's `--convex-tumble-shot`/`--convex-tumble` (the per-pair dispatch → the K-tick
  free-integrate → the 2D render) and FR4's `--fract-step` (the settling-bodies side-view). Mirror them.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/convex_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/convex.h`** (CX1-CX3 byte-frozen): `ConvexWorld {std::vector<FxBody> bodies;
   std::vector<FxBox> boxes;}` (parallel arrays, body[i] ↔ box[i]), `ConvexStepConfig`, `StepConvexWorld(world,
   cfg)` (the 5-pass tick above), `StepConvexWorldN(world, cfg, ticks)`, `MeasureStack(world)`. Pure integer, FIXED
   orders. **NEW shader** `convex_step.comp.hlsl` (int64, Vulkan-only, ONE thread runs the whole world step —
   copies `StepConvexWorldN` VERBATIM). NOT in hf_gen_msl; Metal runs the CPU path.
2. **Showcase `--convex-stack-shot <out>` (Vulkan) AND `--convex-stack` (Metal) — WIRE BOTH** (standalone
   arg-parse). The SCENE: a static FLOOR box (large, `invMass==0`) + a STACK of 3-4 dynamic boxes dropped slightly
   offset so they settle into a coherent tower (boxes interlock + rest). Run `StepConvexWorldN` ~K=120-200 ticks.
   Vulkan: the GPU `convex_step.comp` → **memcmp the GPU final body world vs the CPU `StepConvexWorldN`** (NO
   tolerance). Metal: the CPU reference. Render a PURE-INTEGER 2D side-view (XY) of the settled boxes (each an
   oriented rectangle outline at its final `pos>>kFrac` + orient, the floor distinct, boxes colored by index).
   Golden = `tests/golden/metal/convex_stack.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU final body world == the CPU `StepConvexWorldN` byte-for-byte. Print
     `convex-stack: {bodies:<N>, dynamic:<D>, ticks:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `convex-stack determinism: two runs BYTE-IDENTICAL`.
   - **(3) settled:** after K ticks the max dynamic-body speed is below a rest threshold (the stack came to rest)
     AND no pair interpenetrates beyond `slop + an integer epsilon` (the position correction held) AND the dynamic
     boxes are STACKED (their final `pos.y` are ordered + separated by ~the box height, not collapsed). Print
     `convex-stack settled: {atRest:true, noInterpenetration:true, stacked:true}`; assert all.
   - **(4) ground control:** a SINGLE box dropped on the floor comes to rest ON the floor (final `pos.y −
     halfExtent.y ≈ floorTop` within an epsilon; it did NOT sink through). Print
     `convex-stack ground: {boxRestsOnFloor:true}`; assert it.
   - **Golden discipline: ONLY `tests/golden/metal/convex_stack.png`; do NOT commit it.** Existing 186 image
     goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests — APPEND to `tests/convex_test.cpp` (pure CPU):** a single box dropped on a static floor settles to rest
   on the floor (pos.y stable, not sinking) within K ticks; a 2-3 box stack settles with the boxes ordered in y +
   not interpenetrating beyond slop; the step is a no-op on a world with no overlaps (free-fall only changes
   pos/vel by gravity); `StepConvexWorldN` two runs byte-identical; static bodies never move. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-convex-step` (features) + `--convex-stack-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/fpx.h` + all sibling sim headers + **CX1+CX2+CX3's convex.h code + convex_sat.comp +
  convex_manifold.comp + convex_solve.comp** + `engine/nav/` + `engine/anim/` + `engine/physics/` + all EXISTING
  shaders UNCHANGED. The ONLY new shader is `convex_step.comp.hlsl` (int64, Vulkan-only, NOT in hf_gen_msl).
  `convex.h` APPEND-only. Report the seam empty.

## Out of scope (YAGNI — later CX slices)
Lockstep + rollback (CX5 — pure CPU, the FR5/CP5 twin), the lit 3D render (CX6 — CX4's render is the 2D settled-
stack side-view). The FPX2 BuildPairs AABB broadphase (CX4 uses all-pairs over the small stack; broadphase is a
scaling refinement). Friction (the tangential Coulomb impulse — CX3 deferred it; CX4 stays frictionless-normal +
position correction; a sliding box won't be held by friction — documented). Rotational/positional angular
correction (CX4's de-penetration is LINEAR split-by-invMass only). Stacks larger than ~6 boxes (all-pairs +
single-thread GPU). Arbitrary convex hulls / GJK / EPA (BOXES only). CX4 claims ONLY: a deterministic integer full
convex step (predict → narrowphase → impulse → position de-penetration) that settles a small box stack into a
coherent resting tower, bit-identical CPU↔Vulkan↔Metal over many ticks, with the integer golden + the four proofs.
NOTE (honest): boxes only; all-pairs (small scene); diagonal inertia; non-accumulated Gauss-Seidel + linear position
correction (stack stable within a band, stiffness ∝ iters, residual not analytic); frictionless-normal; int64 →
Vulkan-GPU + Metal-CPU-ref (the FPX3 proof strength, not MSL-native strict-zero).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 105 incl. CX1-CX3's `convex_test` + the appended CX4 cases).
   Clean under `windows-msvc-asan` (build+run `convex_test` + `introspect_test`).
2. **proofs + visual:** `--convex-stack-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows a
   coherent SETTLED STACK (boxes resting in a tower on the floor — NOT collapsed into one blob, NOT scattered, NOT
   sunk through the floor).**
3. Metal: `visual_test --convex-stack` → new golden `tests/golden/metal/convex_stack.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `convex_step.comp` NOT in `hf_gen_msl`.** Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `convex_stack.png` added; the other
   186 byte-identical. `git diff master --stat -- tests/golden` = ONLY `convex_stack.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-convex-step` + `--convex-stack-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h` + all sibling sim headers + **CX1+CX2+CX3's convex.h code
   + convex_sat.comp + convex_manifold.comp + convex_solve.comp** + `engine/nav/` + `engine/anim/` +
   `engine/physics/` + ALL existing shaders byte-unchanged). `scripts/verify.ps1` updated: `convex_stack` golden in
   the Mac loop + `--convex-stack-shot` in `$vkShots`. **The ONLY new shader is `convex_step.comp.hlsl` (int64, NOT
   in `hf_gen_msl`).**
