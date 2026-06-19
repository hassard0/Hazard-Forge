# Slice JT2 — Deterministic Articulated-Body Ragdoll: ANGULAR LIMITS (hinge + cone) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #15 (DETERMINISTIC
> ARTICULATED-BODY RAGDOLL, `hf::sim::joint`) — **THE NEW-PHYSICS BEAT** (the GR4-friction-equivalent). JT1 pinned
> the anchors (a ball joint); JT2 adds the ANGULAR LIMIT: after the JT1 positional projection, project the
> RELATIVE orientation `qrel = qA⁻¹·qB` back into the joint's allowed cone/hinge via a quaternion **swing-twist
> decomposition + cone clamp** — a quaternion analog of the GR4 Coulomb-cone clamp. A hinge holds its axis; a
> cone-twist limits the swing. INTEGER-bit-exact, int64 → Vulkan-only shader + Metal CPU reference (the FPX4/CL3
> split). Single-thread Gauss-Seidel. Branch: `slice-jt2`. See [[hazard-forge-joint-roadmap]].

**Goal:** Extend `engine/sim/joint.h` (additive — JT1 byte-unchanged) with a SEPARATE `FxAngularLimit` record +
`SolveAngularLimit` (the swing-twist + cone clamp + nlerp inverse-mass apply) + `StepArticulated` (integrate → K
Gauss-Seidel passes of {JT1 ball | JT2 angular} → ground). Add `shaders/joint_angular_solve.comp.hlsl` (int64 →
Vulkan-only + Metal CPU). Add `--joint-hinge-shot` (Vulkan) / `--joint-hinge` (Metal). Bake the integer golden
`joint_hinge`. **NO new RHI.**

## Design call: a SEPARATE FxAngularLimit (JT1 FxJoint frozen), integer swing-twist + host-cos cone clamp
To keep JT1's `FxJoint` + `joint_ball_solve.comp` **byte-frozen**, JT2 adds a PARALLEL record (NOT a field on
`FxJoint`): `FxAngularLimit{uint32 bodyA, bodyB; FxVec3 axis (UNIT, body-local on A); fx cosHalfLimit; fx
sinHalfLimit; uint32 kind;}`. A ragdoll edge (JT4) carries BOTH a `FxJoint` (position) and a `FxAngularLimit`
(orientation). `kind`: `kAngularHinge = 0` (cone limit 0 → swing forced to identity, only twist about `axis`),
`kAngularCone = 1` (swing limited to the cone half-angle).

**The limit as host cos/sin (NO runtime acos):** the cone half-angle is supplied as `cosHalfLimit = cos(θ/2)` +
`sinHalfLimit = sin(θ/2)` — host-snapped Q16.16 constants (the caller computes them once in float at scene
build; the SIM does ZERO transcendentals). A HINGE is `cosHalfLimit = kOne, sinHalfLimit = 0` (cone angle 0).

## The swing-twist + cone clamp (the new physics — all integer)
For an `FxAngularLimit` over bodies A,B (read-only of the JT1-projected positions; this is the ORIENTATION pass):
```
qA = world.bodies[A].orient;  qB = world.bodies[B].orient
qrel = FxQuatMul(QConj(qA), qB)                         // B's orientation in A's frame; QConj = {-x,-y,-z,w}
// --- swing-twist decomposition about `axis` (the hinge/cone axis, body-local on A) ---
proj = FxDot(qrel.xyz, axis)                            // qrel's rotation component along the axis (int64)
twist = FxQuatNormalize({axis.x*proj, axis.y*proj, axis.z*proj, qrel.w})   // the on-axis rotation (degenerate -> identity)
swing = FxQuatMul(qrel, QConj(twist))                  // the off-axis part (qrel = swing * twist)
// --- cone clamp the SWING (host cos/sin limit, NO acos) ---
if (swing.w < cosHalfLimit) {                           // swing half-angle exceeds the cone -> clamp
    nhat = FxNormalize(swing.xyz)                       // the swing rotation axis (degenerate -> skip, swing≈identity)
    swing = { sinHalfLimit*nhat.x, sinHalfLimit*nhat.y, sinHalfLimit*nhat.z, cosHalfLimit }   // re-synth at the limit
}                                                       // (hinge: cosHalfLimit=kOne -> always clamps swing -> identity)
qrelClamped = FxQuatNormalize(FxQuatMul(swing, twist)) // recompose
// --- the correction targets + the nlerp inverse-mass apply (NO slerp) ---
qBtarget = FxQuatMul(qA, qrelClamped)                  // qB such that qA⁻¹·qB == qrelClamped
qAtarget = FxQuatMul(qB, QConj(qrelClamped))           // qA such that qA⁻¹·qB == qrelClamped (the mirror)
wsum = invMassA + invMassB; if (wsum==0) skip
wA = fxdiv(invMassA, wsum); wB = fxdiv(invMassB, wsum)
qB = FxQuatNormalize(QNlerp(qB, qBtarget, wB))         // QNlerp(p,q,t) = normalize(p + t*(q-p)) component-wise
qA = FxQuatNormalize(QNlerp(qA, qAtarget, wA))         // a pinned body (invMass 0 -> w 0) is NOT rotated
```
**The integer-clean choices (no transcendentals):** `cosHalfLimit`/`sinHalfLimit` are host constants → no `acos`/
`sin` at runtime; the swing re-synthesis is `FxNormalize`(int64 `FxISqrt`) + `fxmul`; the inverse-mass apply is
**nlerp** (component lerp + `FxQuatNormalize`) — the standard integer-friendly small-angle slerp approximation
(valid per Gauss-Seidel iteration, the cloth/PBD precedent), NOT a slerp/quaternion-power. int64 →
`joint_angular_solve.comp` Vulkan-only + the Metal showcase runs the CPU `SolveAngularLimit` (byte-identical).
Single-thread `[numthreads(1,1,1)]` Gauss-Seidel (order-dependent; a ragdoll has ~15 limits).

**HONEST CAVEAT (state it in the header + proofs):** the swing-twist + host-cos clamp + nlerp apply is a
DETERMINISTIC PROXY for an angular limit, exact-deterministic + bit-identical cross-backend (the headline), but
NOT an analytic constraint-mechanics solution (the GR4-angle-of-repose caveat shape — a within-band limit). The
nlerp small-angle apply leaves a deterministic-but-nonzero residual (more iters → tighter).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The fpx quaternion math to REUSE VERBATIM (`engine/sim/fpx.h:409-472`):** `FxQuatMul` (:412, Hamilton
  product), `FxQuatNormalize` (:427, int64 normalize), `FxRotate` (:440). Add small inline helpers `QConj`
  (negate xyz), `FxDot(FxVec3,FxVec3)` (if not present), `QNlerp` (component lerp + normalize) in `joint.h` — do
  NOT modify fpx.h. The **int64/glslc lesson (fpx.h:400-407)**: int64 → `joint_angular_solve.comp` Vulkan-only,
  NOT in `hf_gen_msl`.
- **JT1 (this branch's `joint.h`, read-only — build on, DON'T modify):** `FxJoint`, `SolveBallJoint`,
  `WorldAnchor`, `StepJointWorld` (JT2's `StepArticulated` mirrors it + the angular pass), `AnchorGap`. The JT1
  `joint_ball_solve.comp` stays byte-frozen → the `joint_ball` golden MUST remain byte-identical.
- **The GR4 cone-clamp precedent (`engine/sim/grain.h` `SolveGrainFriction`):** the Coulomb-cone tangential clamp
  — JT2's swing clamp is the quaternion analog (clamp the off-axis rotation to a cone). Read it for the "clamp a
  vector/rotation into a cone, host-limit, deterministic" shape + the within-band caveat framing.
- **The int64 Vulkan-only single-thread solve precedent (`shaders/joint_ball_solve.comp.hlsl` JT1 /
  `fpx_orient.comp` FPX4 / `cloth_solve.comp` CL3):** the per-pass dispatch driver + the `[numthreads(1,1,1)]`
  Gauss-Seidel + the int64 quaternion ops. `joint_angular_solve.comp` is the same shape.
- **Showcase + registration:** JT1's `--joint-ball-shot` plumbing — **standalone arg-parse loop** (the FR1 C1061
  lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden**), `tests/joint_test.cpp`.

## Design decisions (locked)
1. **`FxAngularLimit` (separate record, JT1 `FxJoint` frozen)** + `kAngularHinge=0`/`kAngularCone=1` +
   `SolveAngularLimit(world, lim)` (the swing-twist + cone clamp + nlerp apply above) + helpers `QConj`/`QNlerp`/
   `FxDot` + a `SwingAngleCos(world, lim)` metric (the swing `.w` = `cos(half-angle)`, for the "within the cone"
   proof). int64. `joint_angular_solve.comp` copies `SolveAngularLimit` VERBATIM.
2. **`StepArticulated(world, joints, angularLimits, dt, iters)`** — integrate (`IntegrateBodyFull` all) → K
   Gauss-Seidel passes EACH doing {all `SolveBallJoint` in order, then all `SolveAngularLimit` in order} → ground
   clamp. (JT1's `StepJointWorld` stays as-is; JT3 will be the full multi-body step adding contacts.) +
   `StepArticulatedSteps`.
3. **Showcase `--joint-hinge-shot <out>` (Vulkan) AND `--joint-hinge` (Metal) — WIRE BOTH** (standalone
   arg-parse). A SWINGING DOOR / pendulum: a pinned (invMass 0) frame body + a door body, joined by a `FxJoint`
   (ball, at the hinge line) AND a `FxAngularLimit` (HINGE about the vertical axis — the door may swing about the
   hinge but is held in the hinge plane, no off-axis flop). Seed an impulse so the door swings; settle K
   `StepArticulated` steps. Vulkan: the GPU ball+angular solve → **memcmp vs the CPU `StepArticulated`**. Metal:
   the CPU reference. Render a top/side view (the door swinging about the hinge). Golden =
   `tests/golden/metal/joint_hinge.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body world after K steps == the CPU `StepArticulated` reference
     byte-for-byte. Print `joint-hinge: {bodies:<N>, joints:<J>, limits:<L>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `joint-hinge determinism: two runs BYTE-IDENTICAL`.
   - **(3) the hinge HOLDS its axis:** after settling, the door's off-axis swing is within the cone (the swing
     `.w >= cosHalfLimit` within an LSB band — the door did NOT flop off the hinge plane), AND the door actually
     ROTATED about the hinge axis (the twist angle is non-trivial — it swung). Print `joint-hinge axis:
     {swungAboutAxis:true, offAxisWithinCone:true}`; assert both.
   - **(4) the free control swings off-axis:** a NO-LIMIT control (`cosHalfLimit = -kOne`, a 180° cone → never
     clamps) lets the door flop off-axis (its off-axis swing exceeds the hinge band), proving the limit does the
     work. Print `joint-hinge control: free door flops off-axis (limit does the work)`.
   - **Golden discipline: ONLY `tests/golden/metal/joint_hinge.png`; do NOT commit it.** Existing 160 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
6. **Tests `tests/joint_test.cpp` additions (pure CPU):** `QConj`/`QNlerp`/swing-twist round-trip (recompose
   `swing*twist == qrel` within an LSB band); `SolveAngularLimit` — a HINGE drives an off-axis-rotated body back
   into the hinge plane (its swing `.w` rises toward `cosHalfLimit`); a CONE clamps a beyond-cone orientation to
   the cone; an in-limit orientation is a (near) no-op; a pinned body (invMass 0) is not rotated; a free limit
   (`cosHalfLimit=-kOne`) never clamps. `StepArticulated` two runs byte-identical. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-joint-hinge` (features) + `--joint-hinge-shot` (showcases). **REBAKE
   `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the JT1/cloth surface). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` +
  `engine/physics/` + all existing shaders UNCHANGED. JT1 `joint.h` code (FxJoint/SolveBallJoint/StepJointWorld)
  + `joint_ball_solve.comp` UNCHANGED (JT2 additive). The only new shader is `joint_angular_solve.comp` (int64,
  Vulkan-only — NOT in `hf_gen_msl`). Report the seam empty except the new Vulkan-only angular shader.

## Out of scope (YAGNI — later JT slices)
The full articulated multi-body step interleaving joints with CONTACTS (JT3 — JT2's `StepArticulated` has no
broadphase/contacts yet), the skeleton→ragdoll bind (JT4), lockstep (JT5), the lit render (JT6). The TWIST-range
limit (JT2 ships the SWING cone clamp + the hinge twist-free-about-axis; a separate twist min/max range is a
small documented extension). Soft/spring drives, motors. Inertia tensor (the fpx deferral). JT2 claims ONLY: a
deterministic angular swing/cone limit that holds a hinge's axis, bit-identical CPU↔Vulkan↔Metal, with the
integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 101) + the new `joint_test` angular cases. Clean under
   `windows-msvc-asan` (build+run `joint_test` + `introspect_test`).
2. **proofs + visual:** `--joint-hinge-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the door swung about the hinge but held in its plane (not flopped off-axis — pixel-check; the JT1 lesson).**
3. Metal: `visual_test --joint-hinge` → new golden `tests/golden/metal/joint_hinge.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm `joint_angular_solve.comp` is
   correctly NOT MSL-generated (int64, Vulkan-only); the JT1 `joint_ball_solve.comp` still ABSENT from MSL.**
   Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `joint_hinge.png` added; the other
   160 byte-identical (re-run `--joint-ball-shot` → still bit-exact, the JT1 golden UNCHANGED). `git diff master
   --stat -- tests/golden` = ONLY `joint_hinge.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-joint-hinge` + `--joint-hinge-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/`fract.h` +
   `engine/physics/` + JT1 `joint.h` FxJoint/SolveBallJoint/StepJointWorld + `joint_ball_solve.comp`
   byte-unchanged). `scripts/verify.ps1` updated: `joint_hinge` golden in the Mac loop + `--joint-hinge-shot` in
   `$vkShots`. `joint_angular_solve.comp` NOT in `hf_gen_msl`.
