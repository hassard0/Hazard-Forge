# Slice FC4 — Deterministic Contact Friction: THE FRICTION-LOCKED WORLD STEP — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #20
> (DETERMINISTIC TANGENTIAL CONTACT FRICTION, `hf::sim::fric`) — THE MONEY-PHYSICS BEAT. FC1-FC3 built the tangent
> basis, the per-contact state, and the cone-clamped impulse solver. FC4 wires friction into the full per-tick
> WORLD step: a box released on a tilted static box **grips and holds** (static cone) or **slides and decelerates
> to rest** (kinetic cone), and the settling box stack stands **with angular damping turned OFF** — real friction
> physically holding the tower, retiring the convex-stack's documented `angDamp` stability aid. INTEGER-bit-exact
> over many ticks. int64 → the `fric_step.comp` shader is Vulkan-only + a Metal CPU reference. FC1-FC3's `fric.h`
> code + CX1-CX6's `convex.h` are BYTE-FROZEN (FC4 is additive). Branch: `slice-fc4`. See
> [[hazard-forge-fric-roadmap]].

**Goal:** Extend `engine/sim/fric.h` (additive — FC1-FC3 + convex.h byte-unchanged) with `FrictionStepConfig` (the
`convex::ConvexStepConfig` fields + `mu`) + `StepFrictionWorld(world, cfg)` (the CX4 5-pass tick with the FC3
friction solve swapped in for the normal-only impulse) + `StepFrictionWorldN` + `MeasureFrictionStack`. Add the new
int64 shader `shaders/fric_step.comp.hlsl` + `--fric-ramp-shot`/`--fric-stack-shot` (Vulkan) /
`--fric-ramp`/`--fric-stack` (Metal). Bake the integer goldens `fric_ramp` + `fric_stack`. **NO new RHI.**

## Design call: the CX4 step with friction (the money-physics beat)

`convex::StepConvexWorld` (frozen, `convex.h:856`) is the proven CX4 5-pass tick — predict-integrate → all-pairs
narrowphase → world Gauss-Seidel impulse → position de-penetration → orientation. FC4 writes a NEW
`StepFrictionWorld` in `fric.h` that mirrors that structure exactly but swaps the **impulse pass** from the
normal-only `convex::SolveManifoldImpulse` to the FC3 normal+friction `SolveFrictionImpulse`. (It cannot modify the
frozen `StepConvexWorld`; it reproduces the step shell, reusing the frozen helpers.)

`StepFrictionWorld(world, cfg)` runs ONE tick over a `convex::ConvexWorld` (reused — the body+box arrays), ALL
orders PINNED (the CX4 order):
1. **Predict-integrate** every DYNAMIC body (`convex::IsDynamic`) via `fpx::IntegrateBodyFull(b, cfg.gravity,
   cfg.dt)`, then the per-tick `cfg.linDamp`/`cfg.angDamp` retain (kOne = off — and the FC4 headline is that with
   friction the stack holds at `angDamp = kOne`).
2. **World inverse inertias** per body, once per tick (`convex::FxBoxInvInertiaBody` + `convex::WorldInvInertia`,
   frozen).
3. **Impulse solve — world Gauss-Seidel:** `cfg.solveIters` outer sweeps; each sweep iterates the all-pairs `i<j`
   list (skip static-static) in fixed order; per overlapping pair run `BuildFrictionPoints` (FC2) →
   `SolveFrictionImpulse(bi, bj, invIiW, invIjW, fm, cfg.restitution, cfg.mu, 1)` (ONE inner sweep — the world loop
   is the outer Gauss-Seidel). The friction solve mutates the bodies in place.
4. **Position de-penetration** (the CX4 step-4, reproduced): `cfg.posIters` sweeps; each over the pairs in fixed
   order; for each still-overlapping pair (`BoxSatStable`) push the two bodies apart along the A→B-corrected SAT
   axis by `fxmul(max(0, pen − cfg.slop), cfg.beta)` split by inverse mass (`wi = fxdiv(invMassA, invMassA+invMassB)`,
   `wj = kOne − wi`; both static → skip). LINEAR only (the CX4 convention). This is `convex.h:907-932` reproduced.
5. (Orientation already integrated in step 1.)

`StepFrictionWorldN(world, cfg, ticks)` runs `ticks` steps. `FrictionStepConfig` = the `ConvexStepConfig` fields
(`gravity, dt, solveIters, restitution, slop, beta, linDamp, angDamp, posIters`) + `fx mu`. `MeasureFrictionStack`
(or reuse `convex::MeasureStack`) → max body speed + max pairwise penetration + dynamic count.

**THE int64 REALITY (the CX4/FC3 lesson):** the whole chain is int64. DXC compiles int64 (Vulkan); glslc cannot.
So `fric_step.comp` is **VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl)**, single-thread over the small body set (the
`convex_step.comp` convention); the Metal `--fric-ramp`/`--fric-stack` runs the CPU `StepFrictionWorldN` —
byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU memcmp proof.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **FC3 `engine/sim/fric.h` (read it; APPEND only after `ResolveContactFriction`):** `SolveFrictionImpulse`,
  `BuildFrictionPoints`, `FrictionManifold`, `FricSolveConfig`. FC1-FC3 lines byte-frozen.
- **convex.h (read-only — do NOT edit):** `convex::ConvexWorld` (`convex.h:741`), `convex::ConvexStepConfig`
  (`convex.h:749`, the FC4 config mirrors + adds `mu`), `convex::StepConvexWorld` (`convex.h:856` — the 5-pass
  shell FC4 REPRODUCES with the friction solve), the **position de-penetration** loop (`convex.h:907-932`, reproduce
  it), `convex::IsDynamic` (`convex.h:786`), `convex::BoxSatStable` (`convex.h:807`), `convex::FxBoxInvInertiaBody`/
  `convex::WorldInvInertia`, `convex::MeasureStack` (`convex.h:942`). FC4 reuses these read-only.
- **fpx.h (read-only):** `IntegrateBodyFull` (`fpx.h:479`), `FxScale`/`FxSub`/`FxAdd`, `kFlagDynamic`, `fxmul`/
  `fxdiv`, `kOne`. **DO NOT modify fpx.h or convex.h.**
- **The shader + showcase precedent:** CX4's `shaders/convex_step.comp.hlsl` (the int64 Vulkan-only single-thread
  WHOLE-WORLD step that copies `StepConvexWorldN` VERBATIM + the GPU==CPU final-state memcmp), the
  `--convex-stack-shot` Vulkan showcase in `samples/hello_triangle/main.cpp` + the `--convex-stack` Metal block.
  `fric_step.comp` is the twin (copies `StepFrictionWorldN`). Confirm `fric_step` NOT in hf_gen_msl.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/fric_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/fric.h`** (FC1-FC3 byte-frozen): `FrictionStepConfig`, `StepFrictionWorld(world, cfg)`
   (the 5-pass tick above), `StepFrictionWorldN(world, cfg, ticks)`, `MeasureFrictionStack` (or reuse MeasureStack).
   Pure integer, FIXED orders. **NEW shader** `fric_step.comp.hlsl` (int64, Vulkan-only, ONE thread runs the whole
   world step — copies `StepFrictionWorldN` VERBATIM). NOT in hf_gen_msl; Metal runs the CPU path.
2. **Showcases — WIRE BOTH backends for TWO scenes:**
   - **`--fric-ramp-shot`/`--fric-ramp`:** a DYNAMIC box released on a TILTED static box (a ramp). Run
     `StepFrictionWorldN` ~K ticks. With the showcase `mu` HIGH the box GRIPS (rests near its release point); a
     SECOND control body with `mu` LOW (or a separate low-μ run) SLIDES down the ramp. Render the settled box(es)
     on the ramp (2D side-view, the convex_stack render style).
   - **`--fric-stack-shot`/`--fric-stack`:** the CX4 settling-stack scene (static floor + 3 dynamic boxes) but with
     `angDamp = kOne` (OFF) — friction physically holds the tower. Run ~K ticks; the tower stands.
   Vulkan: the GPU `fric_step.comp` → **memcmp the GPU final body world vs the CPU `StepFrictionWorldN`** (NO
   tolerance), per scene. Metal: the CPU reference. Goldens = `tests/golden/metal/fric_ramp.png` +
   `tests/golden/metal/fric_stack.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU final body world == the CPU `StepFrictionWorldN` byte-for-byte, BOTH
     scenes. Print `fric-step: {scene:ramp, bodies:<N>, ticks:<K>} GPU==CPU BIT-EXACT` and the same for `stack`.
   - **(2) determinism:** two runs → identical. Print `fric-step determinism: two runs BYTE-IDENTICAL`.
   - **(3) THE MONEY PHYSICS — friction locks:** (a) the HIGH-μ ramp box GRIPPED (its final down-ramp displacement
     is below a small threshold — it did NOT slide away) AND (b) the stack scene came to REST + stays STACKED + no
     interpenetration **with `angDamp = kOne`** (friction held it, no angular-damping aid). Print `fric-step locked:
     {rampGripped:true, stackStandsNoDamp:true}`; assert both.
   - **(4) control — low μ slides:** the LOW-μ ramp box SLID down the ramp (its down-ramp displacement EXCEEDS the
     high-μ box's by a clear margin) — proving friction is what holds the high-μ box, not the geometry. Print
     `fric-step control: {lowMuSlides:true}`; assert.
   - **Golden discipline: ONLY `fric_ramp.png` + `fric_stack.png`; do NOT commit them.** Existing 192 image goldens
     UNTOUCHED.
   - **HONESTY GATE:** if friction at `angDamp = kOne` does NOT hold the stack for the K-tick window (the spurious
     Gauss-Seidel residual torque survives), DO NOT fake it — report it; the fallback is a documented small
     `angDamp` (still less than CX4's) with an honest caveat that friction REDUCES but does not fully retire the
     aid. The controller will judge. Prefer the clean `angDamp = kOne` result if it genuinely holds.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels, both
   scenes.
5. **Tests — APPEND to `tests/fric_test.cpp` (pure CPU):** a high-μ box on a ramp settles with small down-ramp
   displacement; the same box at low μ slides farther (a strict inequality); the stack scene at `angDamp = kOne`
   comes to rest + stays ordered + non-interpenetrating within K ticks; `StepFrictionWorldN` two runs byte-identical;
   static bodies never move. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-friction-step` (features) + `--fric-ramp-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. (One feature + one representative showcase
   flag is enough; the `--fric-stack-shot` flag also exists but the manifest entry for the capability is the one
   feature.) **Do NOT rebake the JSON golden — the controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/convex.h` + `fpx.h` + **FC1-FC3's fric.h code + fric_basis/points/solve.comp** + all other sim headers
  + `engine/nav/` + `engine/anim/` + `engine/physics/` + all EXISTING shaders UNCHANGED. The ONLY new shader is
  `fric_step.comp.hlsl` (int64, Vulkan-only, NOT in hf_gen_msl). `fric.h` APPEND-only. Report the seam empty.

## Out of scope (YAGNI — later FC slices)
Lockstep + rollback (FC5 — pure CPU), the lit 3D render capstone (FC6 — FC4's goldens are the 2D settled side-views).
Cross-tick warm-starting; rolling/spinning friction; the FPX2 broadphase (all-pairs small scene, the CX4 scope).
FC4 claims ONLY: a deterministic integer friction-locked world step that grips/slides a box on a ramp and stands a
box stack (ideally with `angDamp = kOne`), bit-identical CPU↔Vulkan↔Metal over many ticks, with the two integer
goldens + the four proofs. NOTE: boxes only; isotropic single-μ; coupled-iteration μ approx; linear position
correction (stable within a band); int64 → Vulkan-GPU + Metal-CPU-ref.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 106 incl. FC1-FC3's `fric_test` + the appended FC4 cases).
   Clean under `windows-msvc-asan` (build+run `fric_test` + `introspect_test`).
2. **proofs + visual:** `--fric-ramp-shot` AND `--fric-stack-shot` on Vulkan: the 4 proofs + exit 0, under the
   Vulkan-validation gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED).
   **VERIFY the ramp image shows the box GRIPPED on the ramp (high-μ) / SLID (low-μ), and the stack image shows a
   coherent RESTING TOWER (boxes stacked, not collapsed/scattered/sunk) at `angDamp = kOne`.**
3. Metal: `visual_test --fric-ramp` + `--fric-stack` → new goldens `tests/golden/metal/fric_ramp.png` +
   `fric_stack.png`; two runs DIFF 0.0000 each. **Confirm `visual_test.mm` in the diff; confirm `fric_step.comp`
   NOT in `hf_gen_msl`.** Cross-vendor STRICT ZERO, both.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fric_ramp.png` + `fric_stack.png`
   added; the other 192 byte-identical. `git diff master --stat -- tests/golden` = ONLY those two (metal) + the
   introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-friction-step` + `--fric-ramp-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fpx.h` + **FC1-FC3's fric.h code + fric_basis/points/
   solve.comp** + ALL other sim headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing shaders
   byte-unchanged). `scripts/verify.ps1` updated: `fric_ramp` + `fric_stack` goldens in the Mac loop +
   `--fric-ramp-shot` + `--fric-stack-shot` in `$vkShots`. **The ONLY new shader is `fric_step.comp.hlsl` (int64,
   NOT in `hf_gen_msl`).**
