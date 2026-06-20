# Slice FC3 — Deterministic Contact Friction: THE CONE-CLAMPED TANGENT-IMPULSE SOLVER — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #20
> (DETERMINISTIC TANGENTIAL CONTACT FRICTION, `hf::sim::fric`) — THE SOLVER, where friction actually bites. FC1
> built the tangent basis, FC2 the per-contact `FrictionPoint[]` state. FC3 adds the **Coulomb friction cone**: a
> tangent impulse at each contact, clamped to `±μ·jn` (μ × the normal impulse), applied to BOTH linear AND angular
> velocity through the inertia tensor — so a box sliding on a static box has its slide ARRESTED (static cone) or
> DECELERATED (kinetic cone), and the contact drag produces TORQUE. INTEGER-bit-exact. int64 → the
> `fric_solve.comp` shader is Vulkan-only + a Metal CPU reference. FC1/FC2's `fric.h` code + CX1-CX6's `convex.h`
> are BYTE-FROZEN (FC3 is additive). Branch: `slice-fc3`. See [[hazard-forge-fric-roadmap]].

**Goal:** Extend `engine/sim/fric.h` (additive — FC1/FC2 + convex.h byte-unchanged) with `FricSolveConfig` (the
`restitution`/`mu`/`iters`) + `SolveFrictionImpulse` (the combined normal + cone-clamped tangent Gauss-Seidel over a
`FrictionManifold`) + `ResolveContactFriction` (one box-box pair end-to-end). Add the new int64 shader
`shaders/fric_solve.comp.hlsl` + `--fric-solve-shot` (Vulkan) / `--fric-solve` (Metal). Bake the integer golden
`fric_solve`. **NO new RHI.**

## Design call: the Coulomb friction cone on the manifold impulse (the GR4 cone in impulse form)

CX3's `SolveManifoldImpulse` (frozen, `convex.h:651`) already does the NORMAL impulse — the inverse-mass +
inertia-tensor velocity response along `n`. FC3 writes a NEW combined solver that does the normal impulse the SAME
way AND adds the tangent (friction) impulse. The normal part must reproduce CX3's result so that `μ=0` is a clean
control (frictionless == CX3).

Per Gauss-Seidel sweep, for each `FrictionPoint` in fixed order:
- **Normal impulse (the CX3 form, reproduced — do NOT modify the frozen `SolveManifoldImpulse`):** `rA = p −
  bodyA.pos`, `rB = p − bodyB.pos`; `vn = FxDot(vpB − vpA, n)` (with `vpA = vA + ωA×rA`); skip if `vn ≥ 0`;
  `kn = invMa + invMb + n·((I⁻¹ₐ(rA×n))×rA) + n·((I⁻¹_b(rB×n))×rB)`; `jn = fxdiv(−fxmul(kOne+restitution, vn), kn)`,
  clamp `jn ≥ 0`; apply `J = n·jn` to both bodies (`vA −= J·invMa; ωA −= I⁻¹ₐ(rA×J)`; `vB += …`). This is
  `convex.h:662-689` reproduced verbatim into the friction solver.
- **Tangent friction impulse (THE NEW PHYSICS), for each of `t1`, `t2` (the FC2 basis, fixed order t1 then t2):**
  `vt = FxDot(vpB − vpA, t)` (recompute the contact-point velocities AFTER the normal apply — sequential impulse);
  `kt = invMa + invMb + t·((I⁻¹ₐ(rA×t))×rA) + t·((I⁻¹_b(rB×t))×rB)` (the effective mass along `t`, the SAME form as
  `kn` with `t` for `n`); `jt = fxdiv(−vt, kt)`; **CLAMP to the Coulomb cone** `jt = clamp(jt, −fxmul(mu, jn),
  +fxmul(mu, jn))` (μ × THIS sweep's `jn` — the coupled-iteration approximation, documented; a `jt` below the cone
  → static stick, at the cone → kinetic slip); apply `Jt = t·jt` to both bodies (`vA −= Jt·invMa; ωA −= I⁻¹ₐ(rA×Jt)`;
  `vB += …`). Store the last `jt` into `FrictionPoint.tangentImpulse1/2` + `jn` into `.normalImpulse` (the warm-start
  hooks, diagnostic only in the baseline).
- **Gauss-Seidel order PINNED:** `iters` sweeps; each sweep iterates the manifold points `0..count-1`; within a
  point, normal first, then t1, then t2; mutating vel/angVel in place. Fixed double loop, single-thread on the GPU
  (the CX3 mirror).

**`SolveFrictionImpulse(bodyA, bodyB, invIaW, invIbW, fm, restitution, mu, iters)`** — mutates `bodyA`/`bodyB`
vel+angVel + the `fm` accumulators by the above. **`ResolveContactFriction(bodyA, boxA, bodyB, boxB, cfg)`** =
`convex::BoxSatStable` → `BuildFrictionPoints` (FC2) → if `count==0` return → `convex::FxBoxInvInertiaBody` +
`convex::WorldInvInertia` for each (frozen) → `SolveFrictionImpulse`. `cfg = {restitution, mu, iters}`.
Velocity-only (NO position de-penetration — that is FC4).

**THE int64 REALITY (the CX3/FC1 lesson):** the inertia `fxdiv` + the `FxDot`/`FxCross`/`FxMat3MulVec` products are
int64. DXC compiles int64 (Vulkan); glslc cannot. So `fric_solve.comp` is **VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl)**;
the Metal `--fric-solve` runs the CPU `ResolveContactFriction` — byte-identical to the Vulkan GPU result BY
CONSTRUCTION, while the Vulkan side carries the GPU==CPU memcmp proof.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **FC2 `engine/sim/fric.h` (read it; APPEND only after `MeasureFrictionPoints`):** `FrictionPoint`/
  `FrictionManifold`/`BuildFrictionPoints` (FC2), `TangentBasis`/`MakeTangentBasis` (FC1), the pulled
  `fx`/`FxVec3`/`FxDot`/`FxCross`. FC1/FC2 lines byte-frozen.
- **convex.h (read-only — do NOT edit):** `convex::SolveManifoldImpulse` (`convex.h:651` — the normal-impulse body
  FC3 REPRODUCES the normal part of, and whose result `μ=0` must match), `convex::FxBoxInvInertiaBody`
  (`convex.h:606`), `convex::WorldInvInertia` (`convex.h:622` → `FxMat3`), `convex::FxMat3MulVec` (`convex.h:93`),
  `convex::BoxSatStable` (`convex.h:807`), `convex::ContactSolveConfig` (`convex.h:594` — the `restitution`/`iters`
  shape FC3's config mirrors + adds `mu`). The Coulomb-cone tangent clamp precedent is `grain.h` GR4 (the static-
  cancel / kinetic-scale pattern, here in impulse form).
- **fpx.h (read-only):** `FxScale`/`FxSub`/`FxAdd`, `FxBody{vel, angVel, invMass}`, `fxmul`/`fxdiv`, `kOne`. **DO
  NOT modify fpx.h or convex.h.**
- **The shader + showcase precedent:** CX3's `shaders/convex_solve.comp.hlsl` (the int64 Vulkan-only one-thread-per-
  pair solver that copies `ResolveContactPair` VERBATIM + the resolved-body pack + the GPU==CPU memcmp), and the
  `--convex-tumble-shot` Vulkan showcase + `--convex-tumble` Metal block. `fric_solve.comp` is the DIRECT twin
  (copies `ResolveContactFriction`). Confirm `fric_solve` NOT in hf_gen_msl.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/fric_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/fric.h`** (FC1/FC2 byte-frozen): `FricSolveConfig {fx restitution; fx mu; uint32_t
   iters;}`, `SolveFrictionImpulse`, `ResolveContactFriction`. Pure integer, FIXED Gauss-Seidel order (sweeps ×
   points × {normal, t1, t2}), the cone clamp vs the current-sweep `jn`. **NEW shader** `fric_solve.comp.hlsl`
   (int64, Vulkan-only, one thread per pair — copies `ResolveContactFriction` VERBATIM). NOT in hf_gen_msl; Metal
   runs the CPU path.
2. **Showcase `--fric-solve-shot <out>` (Vulkan) AND `--fric-solve` (Metal) — WIRE BOTH** (standalone arg-parse).
   The SCENE: a small fixed deterministic array of contact pairs — at minimum (a) a DYNAMIC box pressed against a
   STATIC box with an incoming TANGENTIAL (sliding) velocity (friction should arrest/reduce the slide + impart a
   little spin), and (b) a CONTROL pair identical but with `mu = 0` (frictionless → the tangential velocity is
   UNCHANGED by the solve, and the resolved body == the CX3 `ResolveContactPair` result). Vulkan: the GPU
   `fric_solve.comp` resolves each pair → **memcmp the GPU resolved bodies vs the CPU `ResolveContactFriction`** (NO
   tolerance). Metal: the CPU reference. For the visual: render a PURE-INTEGER 2D diagnostic of the before/after
   tangential velocity per pair (e.g. the body's velocity vector before vs after, the slide visibly shortened under
   friction). Golden = `tests/golden/metal/fric_solve.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU resolved-body array == the CPU `ResolveContactFriction` byte-for-byte.
     Print `fric-solve: {pairs:<N>, resolved:<M>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `fric-solve determinism: two runs BYTE-IDENTICAL`.
   - **(3) THE NEW PHYSICS — friction arrests the slide:** the high-μ sliding pair's post-solve tangential speed is
     STRICTLY LESS than its pre-solve tangential speed (friction removed tangential momentum) AND its `angVel`
     changed (the contact drag imparted spin). Print `fric-solve newphysics: {slideReduced:true, frictionTorque:
     true}`; assert both.
   - **(4) control — μ=0 is frictionless == CX3:** the `mu=0` pair's resolved body is BYTE-IDENTICAL to
     `convex::ResolveContactPair(...)` on the same pair (the friction solve at μ=0 leaves the normal-only result
     exactly). Print `fric-solve control: {muZeroEqualsFrictionless:true}`; assert the memcmp.
   - **Golden discipline: ONLY `tests/golden/metal/fric_solve.png`; do NOT commit it.** Existing 191 image goldens
     UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests — APPEND to `tests/fric_test.cpp` (pure CPU):** a sliding box vs a static box with high μ → the
   tangential velocity magnitude drops after `SolveFrictionImpulse`; with μ=0 the tangential velocity is unchanged
   AND the body equals `convex::ResolveContactPair`; the friction impulse stays within the cone (`|jt| ≤ μ·jn` per
   point); a separating contact (`vn ≥ 0`) applies no impulse; two runs byte-identical. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-friction-solve` (features) + `--fric-solve-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/convex.h` + `fpx.h` + **FC1/FC2's fric.h code + fric_basis.comp + fric_points.comp** + all other sim
  headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + all EXISTING shaders UNCHANGED. The ONLY new shader
  is `fric_solve.comp.hlsl` (int64, Vulkan-only, NOT in hf_gen_msl). `fric.h` APPEND-only. Report the seam empty.

## Out of scope (YAGNI — later FC slices)
The friction-locked world step / ramp grip+slide / the tower standing with `angDamp=kOne` (FC4 — FC3 only resolves
ONE pair's velocity, NO position correction, NO multi-tick integration), lockstep (FC5), the lit 3D render (FC6).
Cross-tick warm-starting (the accumulators are stored but the baseline re-solves from zero each call). Rolling/
spinning friction (FC3 is tangential-sliding only). FC3 claims ONLY: a deterministic integer Coulomb-cone tangent
impulse (clamped to ±μ·jn, applied to linear + angular velocity) that arrests a slide, bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the four proofs (incl. μ=0 == frictionless == CX3). NOTE: boxes only;
isotropic single-μ; coupled-iteration μ approx (cone vs current-sweep jn); int64 → Vulkan-GPU + Metal-CPU-ref.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 106 incl. FC1/FC2's `fric_test` + the appended FC3 cases).
   Clean under `windows-msvc-asan` (build+run `fric_test` + `introspect_test`).
2. **proofs + visual:** `--fric-solve-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the friction arresting the slide (a coherent before/after velocity diagnostic, not scrambled).**
3. Metal: `visual_test --fric-solve` → new golden `tests/golden/metal/fric_solve.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `fric_solve.comp` NOT in `hf_gen_msl`.** Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fric_solve.png` added; the other
   191 byte-identical. `git diff master --stat -- tests/golden` = ONLY `fric_solve.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-friction-solve` + `--fric-solve-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fpx.h` + **FC1/FC2's fric.h code + fric_basis.comp +
   fric_points.comp** + ALL other sim headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing
   shaders byte-unchanged). `scripts/verify.ps1` updated: `fric_solve` golden in the Mac loop + `--fric-solve-shot`
   in `$vkShots`. **The ONLY new shader is `fric_solve.comp.hlsl` (int64, NOT in `hf_gen_msl`).**
