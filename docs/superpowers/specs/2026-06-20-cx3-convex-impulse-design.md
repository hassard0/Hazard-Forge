# Slice CX3 — Deterministic Convex Contacts: THE ANGULAR CONTACT IMPULSE (the new physics) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #19
> (DETERMINISTIC CONVEX RIGID-BODY CONTACTS, `hf::sim::convex`) — **THE NEW-PHYSICS BEAT** (the FR3/GR4-equivalent).
> CX1 gave the SAT axis, CX2 the contact manifold (1-4 points). CX3 turns the manifold into the VELOCITY response:
> the inverse-mass + INERTIA-TENSOR contact impulse, applied to BOTH linear velocity AND angular velocity. This is
> the **FIRST time in the whole engine that `angVel` is driven by a contact** — a box hitting a static box
> OFF-CENTER gains spin and TUMBLES (impossible in the sphere-sphere fpx solver). Deterministic integer
> Gauss-Seidel. INTEGER-bit-exact. int64 → the `convex_solve.comp` shader is Vulkan-only + a Metal CPU reference
> (the convex_sat/convex_manifold/fpx_solve split). CX1+CX2's `convex.h` code + shaders are BYTE-FROZEN (CX3 is
> additive). Branch: `slice-cx3`. See [[hazard-forge-convex-roadmap]].

**Goal:** Extend `engine/sim/convex.h` (additive — CX1+CX2 byte-unchanged) with the angular contact impulse: the box
inertia tensor (`FxBoxInvInertiaBody` + `WorldInvInertia` → an `FxMat3`, the type CX1 introduced and CX3 finally
USES) + `SolveManifoldImpulse` (the Gauss-Seidel velocity+angular impulse over a `ContactManifold`) + a one-pair
`ResolveContactPair` (BoxSat → BuildManifold → world inertias → solve). Add the new int64 shader
`shaders/convex_solve.comp.hlsl` + `--convex-tumble-shot` (Vulkan) / `--convex-tumble` (Metal). Bake the integer
golden `convex_tumble`. **NO new RHI.**

## Design call: the inverse-mass + inertia-tensor contact impulse in Q16.16

### The box inertia tensor (body-space diagonal → world via R·I⁻¹·Rᵀ)
A box with half-extents `(hx,hy,hz)` and mass `m` has the analytic DIAGONAL body-space inertia
`Ixx = (m/3)(hy²+hz²)`, `Iyy = (m/3)(hx²+hz²)`, `Izz = (m/3)(hx²+hy²)` (derived from the full extents `2h`). We store
`invMass = 1/m` on the body, so the body-space INVERSE inertia diagonal is
`invIbody = ( 3·invMass / (hy²+hz²), 3·invMass / (hx²+hz²), 3·invMass / (hx²+hy²) )` — each a Q16.16
`fxdiv(3*invMass, fxmul(h,h)+fxmul(h,h))` (int64 inside `fxmul`/`fxdiv`). A STATIC body (`invMass==0`) → `invIbody =
(0,0,0)` (infinite inertia, takes no angular impulse).
- **`FxBoxInvInertiaBody(box, invMass)` → FxVec3** = the 3 diagonal body-space inverse inertias (above).
- **`WorldInvInertia(body, invIbody)` → FxMat3** = `R · diag(invIbody) · Rᵀ` where `R`'s columns are the body's
  WORLD face axes (`BoxAxes(body, axes)` — already in CX1). Build it explicitly into an `FxMat3` (the type's first
  real use): `I⁻¹_world = Σ_k invIbody[k] · (axis_k ⊗ axis_k)` (the outer product of each world axis with itself,
  scaled by its inverse inertia — that IS `R·diag·Rᵀ` for an orthonormal R). Symmetric. Use the CX1 `FxMat3`
  helpers; `FxMat3MulVec(I⁻¹_world, x)` gives `I⁻¹_world · x`.

### The impulse (per manifold point, Gauss-Seidel)
For each contact point `p` (from the CX2 `ContactManifold`), with the contact normal `n` **SIGN-CORRECTED to point
from A toward B** (`n = manifold.normal`; if `FxDot(n, FxSub(bodyB.pos, bodyA.pos)) < 0` flip it — this sidesteps
the CX2 ref→inc ambiguity; do it ONCE for the manifold):
- lever arms `rA = FxSub(p, bodyA.pos)`, `rB = FxSub(p, bodyB.pos)` (world, from each center of mass).
- contact-point velocities `vpA = vA + ωA × rA`, `vpB = vB + ωB × rB` (`×` = `FxCross`; ω = `body.angVel`, world).
- relative normal velocity `vn = FxDot(FxSub(vpB, vpA), n)`. If `vn >= 0` (separating or resting) → SKIP this point
  (no impulse — deterministic).
- effective-mass denominator
  `k = invMassA + invMassB + FxDot(n, FxCross(FxMat3MulVec(invIaW, FxCross(rA, n)), rA)) + FxDot(n,
  FxCross(FxMat3MulVec(invIbW, FxCross(rB, n)), rB))`. (`invIaW`/`invIbW` = the world inverse inertias.) If
  `k <= 0` → skip (degenerate; shouldn't happen with a real contact).
- scalar impulse `jn = fxdiv( -fxmul(kOne + restitution, vn), k )`. CLAMP `jn` to `>= 0` (a contact only PUSHES —
  no sticking; non-accumulated per-iteration, the simple Gauss-Seidel).
- apply `J = n·jn` to BOTH bodies (statics with invMass==0 are unaffected since invMass/invI are 0):
  `vA -= FxScale(J, invMassA); ωA -= FxMat3MulVec(invIaW, FxCross(rA, J))`;
  `vB += FxScale(J, invMassB); ωB += FxMat3MulVec(invIbW, FxCross(rB, J))`.
- **Gauss-Seidel order PINNED:** iterate `iters` sweeps; in each sweep iterate the manifold points `0..count-1` in
  order, mutating the body vel/angVel in place (so later points see earlier updates — the Gauss-Seidel the
  fpx_solve.comp single-thread mirror reproduces bit-for-bit). Fixed double loop, single-thread on the GPU.

**`SolveManifoldImpulse(bodyA, bodyB, invIaW, invIbW, manifold, restitution, iters)`** — mutates `bodyA`/`bodyB`
vel+angVel by the above. **`ResolveContactPair(bodyA, boxA, bodyB, boxB, cfg)`** = `BoxSat` → if `!overlap` return
(no-op) → `BuildManifold` → `WorldInvInertia` for each → `SolveManifoldImpulse`. `cfg = {restitution, iters}`.
**Velocity-only** (NO position de-penetration — that is CX4's settling step; CX3 proves the impulse imparts the
correct spin, the showcase free-integrates to SHOW the tumble).

**THE int64 REALITY (the CX1/CX2/FPX3 lesson):** the inertia `fxdiv` + the `FxDot`/`FxCross`/`FxMat3MulVec` Q16.16
products are int64. DXC compiles int64 (Vulkan); glslc cannot. So `convex_solve.comp` is **VULKAN-SPIR-V-ONLY (NOT in
hf_gen_msl)**; the Metal `--convex-tumble` runs the CPU `ResolveContactPair` — byte-identical to the Vulkan GPU
result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU memcmp proof. Document this.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **CX1+CX2 `engine/sim/convex.h` (read it; do NOT edit CX1/CX2's code — APPEND only after `MeasureManifold`):**
  `FxMat3`/`FxMat3Diagonal`/`FxMat3MulVec`/`FxMat3Transpose` (CX1, finally used), `FxDot`/`FxCross`, `FxBox`,
  `BoxAxes`, `SatResult`/`BoxSat`, `ContactManifold`/`BuildManifold`, `SatPair`. CX3 APPENDS its inertia + impulse
  helpers; CX1+CX2 lines byte-frozen.
- **fpx Q16.16 + body + integrate toolbox (`engine/sim/fpx.h`, read-only):** `FxBody{pos, vel, invMass, flags,
  orient, angVel}` (angVel is WORLD-frame — `IntegrateOrientation` left-multiplies `ω⊗q`, fpx.h:461-468), `FxScale`
  (fpx.h:70), `FxSub`/`FxAdd`, `FxDot`, `FxCross`, `IntegrateBodyFull(b, gravity, dt)` (fpx.h:479 — vel+=g·dt;
  pos+=vel·dt; IntegrateOrientation; NO ground clamp), `IntegrateOrientation` (fpx.h:464), `kOne`/`kFrac`/`kHalf`,
  `fxmul`/`fxdiv`, `kFlagDynamic`. **CONFIRM the exact signatures. DO NOT modify fpx.h.** The CX3 showcase
  free-integrates the resolved body with `IntegrateBodyFull` (the FPX4 6-DOF integrate — angVel now NON-ZERO drives
  a real orientation change).
- **The shader precedent:** CX2's `shaders/convex_manifold.comp.hlsl` (the int64 Vulkan-only one-thread-per-pair
  pattern, the `ManifoldGpu` pack, the VERBATIM CPU copy) AND `shaders/fpx_solve.comp.hlsl` (the int64 single-thread
  Gauss-Seidel solver mold — the FxQuat/FxRotate int64 ops, how it is registered for DXC SPIR-V but excluded from
  hf_gen_msl). `convex_solve.comp.hlsl` follows BOTH: copies `ResolveContactPair` (BoxSat→BuildManifold→inertia→
  SolveManifoldImpulse) VERBATIM, one thread per pair, writes the resolved bodies. Confirm `convex_solve` NOT in
  hf_gen_msl (grep 0).
- **The showcase precedents (`samples/hello_triangle/main.cpp` + `metal_headless/visual_test.mm`):** CX2's
  `--convex-manifold-shot` / `--convex-manifold` (the per-pair GPU dispatch + GPU==CPU memcmp + the 2D render). And
  FPX4's `--fpx-shot` (rendering an ORIENTED body via the integer orient). Mirror these.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/convex_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/convex.h`** (CX1+CX2 byte-frozen): `ContactSolveConfig {fx restitution; uint32_t iters;}`,
   `FxBoxInvInertiaBody`, `WorldInvInertia` (→ FxMat3), `SolveManifoldImpulse`, `ResolveContactPair`. Pure integer,
   FIXED Gauss-Seidel order, sign-corrected A→B normal. **NEW shader** `convex_solve.comp.hlsl` (int64, Vulkan-only,
   one thread per pair — copies `ResolveContactPair` VERBATIM). NOT in hf_gen_msl; Metal runs the CPU path.
2. **Showcase `--convex-tumble-shot <out>` (Vulkan) AND `--convex-tumble` (Metal) — WIRE BOTH** (standalone
   arg-parse). The SCENE: a small fixed deterministic array of contact pairs — at minimum (a) a DYNAMIC box moving
   so a CORNER strikes a STATIC box OFF-CENTER (the off-center contact torques it → non-zero angVel), and (b) a
   CONTROL pair where the dynamic box hits the static box DEAD-CENTER, head-on, symmetric (→ ~zero angVel). Vulkan:
   the GPU `convex_solve.comp` resolves each pair → **memcmp the GPU resolved bodies (pos/vel/invMass/orient/angVel
   pack) vs the CPU `ResolveContactPair`** (NO tolerance). Metal: the CPU reference. For the VISUAL golden:
   free-integrate the resolved dynamic box of pair (a) forward N (~24-48) ticks with `IntegrateBodyFull` (NO further
   contact) so the box visibly ROTATES, and render a PURE-INTEGER 2D side-view (XY) of the box's oriented outline at
   a few sampled ticks (a motion-trail of the tumbling box) — the tumble is the money visual. Golden =
   `tests/golden/metal/convex_tumble.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU resolved-body array == the CPU `ResolveContactPair` byte-for-byte. Print
     `convex-tumble: {pairs:<N>, resolved:<M>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `convex-tumble determinism: two runs BYTE-IDENTICAL`.
   - **(3) THE NEW PHYSICS — contact drove spin:** the OFF-CENTER pair's dynamic body has `angVel != 0` after the
     resolve (the contact imparted angular velocity — the first contact-driven spin in the engine) AND its post-hit
     normal-relative velocity at the deepest contact is non-approaching (the impulse removed the approach). Print
     `convex-tumble newphysics: {contactDroveSpin:true, approachRemoved:true}`; assert both.
   - **(4) control — centered hit, no spin:** the DEAD-CENTER symmetric pair's dynamic body has `angVel == 0` (a
     head-on symmetric contact produces NO rotation). Print `convex-tumble control: {centeredHit:noSpin}`; assert
     `angVel == {0,0,0}` for that pair.
   - **Golden discipline: ONLY `tests/golden/metal/convex_tumble.png`; do NOT commit it.** Existing 185 image
     goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests — APPEND to `tests/convex_test.cpp` (pure CPU):** `FxBoxInvInertiaBody` matches the analytic
   `(m/3)(h²+h²)` inverse for a unit box; `WorldInvInertia` of an identity-orient box == the diagonal body inertia,
   and of a 90°-rotated box permutes the diagonal correctly; `SolveManifoldImpulse` on a 1-point OFF-CENTER contact
   produces non-zero angVel of the right sign; a DEAD-CENTER symmetric contact produces zero angVel; an
   already-separating contact (`vn >= 0`) is a no-op; momentum sanity (the static body unaffected; the dynamic
   body's normal approach is removed); two runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-convex-impulse` (features) + `--convex-tumble-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `vehicle.h` +
  `active.h` + `boids.h` + `engine/nav/` + `engine/anim/` + `engine/physics/` + **CX1+CX2's convex.h code +
  convex_sat.comp + convex_manifold.comp** + all EXISTING shaders UNCHANGED. The ONLY new shader is
  `convex_solve.comp.hlsl` (int64, Vulkan-only, NOT in hf_gen_msl). `convex.h` APPEND-only. Report the seam empty.

## Out of scope (YAGNI — later CX slices)
The full multi-pass per-tick step driver over MANY ticks for a SETTLING STACK of multiple boxes (CX4 — CX3 proves
ONE off-center contact imparts the correct spin + the box tumbles when free-integrated; CX4 owns BuildPairs
broadphase + the per-tick predict→detect→solve→integrate loop + position de-penetration so a stack RESTS), lockstep
(CX5), the lit 3D render (CX6 — CX3's render is the 2D tumble side-view diagnostic). Position/Baumgarte
de-penetration (CX3 is velocity-impulse-only; the box does not need to rest, only to spin from contact). Accumulated/
warm-started impulses, friction (the tangential Coulomb impulse — a later refinement; CX3 is the NORMAL angular
impulse). Arbitrary convex hulls / GJK / EPA (BOXES only). CX3 claims ONLY: a deterministic integer inertia-tensor
contact impulse that drives BOTH linear AND angular velocity (the first contact-driven spin), bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE (honest): boxes only; diagonal inertia only;
non-accumulated single-pass-per-iteration Gauss-Seidel (residual not analytic, stiffness ∝ iters); velocity-only (no
position correction); int64 → Vulkan-GPU + Metal-CPU-ref (the FPX3 proof strength, not MSL-native strict-zero).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 105 incl. CX1/CX2's `convex_test` + the appended CX3 cases).
   Clean under `windows-msvc-asan` (build+run `convex_test` + `introspect_test`).
2. **proofs + visual:** `--convex-tumble-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the box VISIBLY ROTATED / tumbling across the sampled ticks (a coherent tumble, not a translating un-rotated
   box).**
3. Metal: `visual_test --convex-tumble` → new golden `tests/golden/metal/convex_tumble.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `convex_solve.comp` NOT in `hf_gen_msl`.** Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `convex_tumble.png` added; the other
   185 byte-identical. `git diff master --stat -- tests/golden` = ONLY `convex_tumble.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-convex-impulse` + `--convex-tumble-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h`/`vehicle.h`/`active.h`/`boids.h` + `engine/nav/` + `engine/anim/` + `engine/physics/` + **CX1+CX2's
   convex.h code + convex_sat.comp + convex_manifold.comp** + ALL existing shaders byte-unchanged). `scripts/
   verify.ps1` updated: `convex_tumble` golden in the Mac loop + `--convex-tumble-shot` in `$vkShots`. **The ONLY new
   shader is `convex_solve.comp.hlsl` (int64, NOT in `hf_gen_msl`).**
