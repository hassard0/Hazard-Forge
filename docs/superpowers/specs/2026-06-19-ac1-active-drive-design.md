# Slice AC1 — Deterministic Active Ragdoll: THE ANGULAR POSE-DRIVE PRIMITIVE — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the BEACHHEAD) of FLAGSHIP #17
> (DETERMINISTIC ACTIVE RAGDOLL / PHYSICAL-ANIMATION BLENDING, `hf::sim::active`). Flagship #15 (joint) built a
> ragdoll; flagship #16 (vehicle) built the spring soft-constraint. AC1 adds the ONE genuinely-new primitive
> active ragdoll needs: an **angular DRIVE-TO-TARGET torque** — a motor that drives a joint's relative
> orientation toward a target quaternion by a stiffness fraction (the swing-twist toolbox of `SolveAngularLimit`
> with the cone-clamp replaced by an nlerp-toward-target). INTEGER-bit-exact. The ONLY new shader of the whole
> flagship: `active_drive_solve.comp` (int64, Vulkan-only + Metal CPU ref, the JT2/VH1 split). Branch:
> `slice-ac1`. See [[hazard-forge-active-roadmap]].

**Goal:** Create `engine/sim/active.h` (header-only, namespace `hf::sim::active`, `#include "sim/joint.h"` +
`"sim/fpx.h"` read-only ONLY) with `FxAngularDrive` (the drive record) + `SolveAngularDrive` (the per-drive
solver) + `StepDriveWorld` (the JT2 `StepArticulated` mold with a drive pass) + `StepDriveWorldSteps` +
`DriveAngleCos` (the held-to-target metric). Add the new int64 shader `shaders/active_drive_solve.comp.hlsl` +
`--active-drive-shot` (Vulkan) / `--active-drive` (Metal). Bake the integer golden `active_drive`. **NO new RHI.**

## Design call: the drive is SolveAngularLimit's apply with the cone-clamp replaced by nlerp-toward-target
`joint.h::SolveAngularLimit` (joint.h:267) already does the full machinery: `qrel = FxQuatMul(QConj(qA), qB)` (B's
orientation in A's frame), a swing-twist cone clamp producing `qrelClamped`, then an inverse-mass nlerp apply that
rotates both bodies so `qA⁻¹·qB → qrelClamped`. AC1 keeps the *apply* verbatim and replaces the *clamp* with a
**drive toward a target**:

`SolveAngularDrive(world, drv)`:
1. `FxBody& a = bodies[drv.bodyA]; FxBody& b = bodies[drv.bodyB];` skip if out-of-range or `wsum =
   a.invMass + b.invMass == 0`.
2. `qrel = FxQuatMul(QConj(a.orient), b.orient)` (B in A's frame — the SolveAngularLimit line verbatim).
3. **The drive (replaces the cone clamp):** `qrelDriven = FxQuatNormalize(QNlerp(qrel, drv.qTarget, drv.stiffness))`
   — nlerp the current relative orientation toward the target by `stiffness ∈ [0, kOne]` (the VH1 spring
   `stiffness` soft-constraint pattern lifted to orientation; `stiffness = 0` → no drive, `stiffness = kOne` →
   snap to target). `QNlerp` + `FxQuatNormalize` are the existing joint.h int64 helpers (joint.h:258, 282).
   *(Optional sign-fix for the shortest arc: if `FxDot4(qrel, qTarget) < 0` negate `qTarget`'s components before
   the nlerp — quaternion double-cover; document if included. Keep it deterministic / fixed-order.)*
4. **The inverse-mass nlerp apply (VERBATIM SolveAngularLimit:299-304):** `qBtarget = FxQuatMul(a.orient,
   qrelDriven)`, `qAtarget = FxQuatMul(b.orient, QConj(qrelDriven))`, `wA = fxdiv(a.invMass, wsum)`, `wB =
   fxdiv(b.invMass, wsum)`, `b.orient = FxQuatNormalize(QNlerp(b.orient, qBtarget, wB))`, `a.orient =
   FxQuatNormalize(QNlerp(a.orient, qAtarget, wA))`. A pinned body (invMass 0 → share 0) is NOT rotated.

`FxAngularDrive { uint32 bodyA, bodyB; FxQuat qTarget; fx stiffness; }` — std430-packable plain int32s (2×uint32 +
4×int32 quat + 1×int32 stiffness = 28 bytes; pad to a 16-byte-aligned stride if the GPU mirror needs it — match
the `FxAngularLimit` packing discipline, joint.h:230, and document the exact stride).

`StepDriveWorld(world, joints, angularLimits, drives, dt, iters)` = the `StepArticulated` mold (joint.h:332) with
the drive pass added to the Gauss-Seidel interleave: integrate (FPX4 `IntegrateBodyFull`) → K passes EACH {all
`SolveBallJoint` (fixed order) → all `SolveAngularLimit` (fixed order) → all `SolveAngularDrive` (fixed order)} →
ground floor clamp. (AC1's scene uses only joints + drives; the limits list may be empty.) Sequential
single-thread, fixed op order → two-run bit-identical AND bit-exact GPU==CPU. `StepDriveWorldSteps` runs K steps.
`DriveAngleCos(world, drv)` → the `.w` of `qrel·qTarget⁻¹` (== cos(half the residual angle to target); a held
drive keeps this near kOne — the "drove to target" metric, the `SwingAngleCos` shape, joint.h:311).

**The new shader `active_drive_solve.comp.hlsl`** copies `SolveAngularDrive`'s body VERBATIM (the
`joint_angular_solve.comp` mold — one thread, the whole-world drive pass, int64 quaternion math). Because the
quaternion `FxQuatMul`/`FxQuatNormalize`/`fxdiv` use int64, the shader is **DXC/Vulkan-SPIR-V-only** (glslc can't
parse int64) → NOT added to `hf_gen_msl`; the Metal `--active-drive` runs the CPU `StepDriveWorld` (byte-identical
by construction — the JT2/VH1/CL3/FPX3 split). The Vulkan `--active-drive-shot` GPU driver dispatches
`joint_ball_solve.comp` (ball pass) + `joint_angular_solve.comp` (limit pass, if any) + `active_drive_solve.comp`
(drive pass) per Gauss-Seidel iteration in the locked order, then memcmp's the GPU body world vs the CPU
`StepDriveWorld` → BIT-EXACT.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The drive solver template (`engine/sim/joint.h`):** `SolveAngularLimit` (joint.h:267 — copy the apply
  verbatim, replace the cone clamp with the nlerp-to-target), `QConj` (248), `QNlerp` (258), `FxDot` (251),
  `SwingAngleCos` (311 — the `DriveAngleCos` template), `StepArticulated` (332 — the `StepDriveWorld` mold),
  `FxAngularLimit` (237 — the std430 packing discipline). **DO NOT modify joint.h** (JT byte-frozen).
- **The fpx quaternion toolbox (`engine/sim/fpx.h`, read-only):** `FxQuat`, `FxQuatMul`, `FxQuatNormalize`,
  `FxBody{orient, invMass, ...}`, `FxWorld`, `IntegrateBodyFull`, `fxmul`/`fxdiv`/`FxLength`/`FxNormalize`. **DO
  NOT modify fpx.h.**
- **The VH1 spring stiffness pattern (`engine/sim/vehicle.h`):** the `stiffness ∈ [0,kOne]` soft-constraint
  scalar (the AC1 drive strength is the same idea on orientation). Read-only.
- **The new-shader showcase precedent (`samples/hello_triangle/main.cpp`):** study `--joint-hinge-shot` /
  `--vehicle-spring-shot` (the int64-Vulkan-only new-shader GPU driver + the per-pass dispatch + the GPU==CPU
  memcmp + the standalone arg-parse). `--joint-step-shot` for the multi-pass interleave driver. Mirror these.
- **The int64-shader build wiring:** how `joint_angular_solve.comp.hlsl` / `vehicle_spring_solve.comp.hlsl` are
  registered for DXC SPIR-V compile but EXCLUDED from `hf_gen_msl` (the Metal MSL-gen list) — `active_drive_solve.
  comp.hlsl` follows the SAME wiring (CMake/CompileShaders + the hf_gen_msl exclusion). Confirm grep: `active`
  must NOT appear in the hf_gen_msl list.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp`
  (**controller rebakes the JSON golden — do NOT**), a NEW `tests/active_test.cpp` (+ CMake wiring for the test).

## Design decisions (locked)
1. **NEW `engine/sim/active.h`** (header-only, namespace `hf::sim::active`, `#include "sim/joint.h"` + `"sim/fpx.h"`
   read-only): `FxAngularDrive{bodyA, bodyB, FxQuat qTarget, fx stiffness}` + `SolveAngularDrive` (above) +
   `StepDriveWorld`/`StepDriveWorldSteps` + `DriveAngleCos`. Pure integer, fixed op order.
2. **NEW `shaders/active_drive_solve.comp.hlsl`** (int64, Vulkan-only, `[numthreads(1,1,1)]` whole-world drive
   pass — the joint_angular_solve.comp mold; copies SolveAngularDrive's body verbatim). Registered for DXC SPIR-V,
   EXCLUDED from hf_gen_msl. The ONLY new shader of the flagship.
3. **Showcase `--active-drive-shot <out>` (Vulkan) AND `--active-drive` (Metal) — WIRE BOTH** (standalone
   arg-parse). The SCENE: a 3-link chain (a pinned invMass-0 root + 3 dynamic bodies ball-jointed end-to-end, like
   the JT1 hanging chain) with an `FxAngularDrive` on each joint whose `qTarget` bends the chain into an L-shape
   (e.g. a 90° relative rotation about Z on the first joint). Settle K `StepDriveWorld` steps × iters → the chain
   is DRIVEN to + HOLDS the L-pose against gravity (vs a stiffness-0 control that hangs straight down). Vulkan: the
   GPU multi-pass (ball + drive shaders, locked order) → **memcmp vs the CPU StepDriveWorld**. Metal: the CPU
   reference. Render a PURE-INTEGER 2D side-view (each body a disc at `pos>>kFrac`, the joints as segments, the
   pinned root distinct, the target pose implied by the held bend). Golden = `tests/golden/metal/active_drive.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body world after K steps == the CPU `StepDriveWorld` byte-for-byte. Print
     `active-drive: {bodies:<N>, joints:<J>, drives:<D>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `active-drive determinism: two runs BYTE-IDENTICAL`.
   - **(3) drove to target:** after settling, each drive's `DriveAngleCos >= a band` (the relative orientation
     reached/held the target within the deterministic Gauss-Seidel residual) AND the chain bent (the driven
     bodies' positions differ from the straight-hang control by a margin). Print `active-drive held: {meanCos:<C>,
     bentFromRest:true}`; assert both.
   - **(4) control:** a `stiffness = 0` drive set leaves the chain hanging straight down (the drive does the work).
     Print `active-drive control: {stiffness0:hangsStraight}`.
   - **Golden discipline: ONLY `tests/golden/metal/active_drive.png`; do NOT commit it.** Existing 171 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
6. **Tests `tests/active_test.cpp` (NEW, pure CPU):** `SolveAngularDrive` — a single joint driven with stiffness
   kOne snaps qrel to qTarget (within an LSB band); stiffness 0 leaves qrel unchanged; a pinned body is not
   rotated; inverse-mass split rotates the lighter body more. `StepDriveWorld` — the 3-link chain drives to + holds
   the L-pose (DriveAngleCos near kOne); the stiffness-0 control hangs straight; two runs byte-identical. Clean
   under `windows-msvc-asan`. Wire the new test into CMake (the joint_test/vehicle_test pattern).
7. **Introspect.** Add exactly `deterministic-active-drive` (features) + `--active-drive-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the JT/VH/fpx surface). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` +
  `fract.h` + `vehicle.h` + `engine/anim/` + `engine/physics/` + all EXISTING shaders UNCHANGED. The ONLY new
  shader is `active_drive_solve.comp.hlsl` (int64, Vulkan-only, NOT in hf_gen_msl). `active.h` is a NEW additive
  sibling (`#include`s joint.h + fpx.h read-only). Report the seam empty (only `active.h` + the new shader + the
  showcase/test/introspect are new/changed).

## Out of scope (YAGNI — later AC slices)
Per-joint blend weight (AC2), the anim-clip target step (AC3 — AC1's `qTarget` is a host-fixed test pose, NOT yet
sampled from a clip), the active→limp→recover blend factor + impact (AC4), lockstep/rollback (AC5), the lit
skinned render (AC6). A mass-correct PD controller / N·m torque limits / critical damping. AC1 claims ONLY: a
deterministic angular drive-to-target primitive that bends a jointed chain to a held target pose, bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE (honest, the JT2/VH1 caveat shape): the drive is
a stiffness-scaled nlerp toward target (a soft angular constraint), NOT analytic motor mechanics; the held angle
is a deterministic Gauss-Seidel residual (near-target within a band, not exact); bones have no inertia tensor
(inherited fpx/JT caveat). Determinism + cross-platform bit-identity is the headline.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 102 + the NEW `active_test`). Clean under `windows-msvc-asan`
   (build+run `active_test` + `introspect_test`).
2. **proofs + visual:** `--active-drive-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the chain DRIVEN to the L-pose (held bend), not hanging straight (pixel-check; the VH1 lesson).** Re-run
   `--joint-hinge-shot` + `--joint-ball-shot` → still bit-exact (JT render-invariance).
3. Metal: `visual_test --active-drive` → new golden `tests/golden/metal/active_drive.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm `active_drive_solve.comp` is
   NOT in `hf_gen_msl` (the Metal path runs the CPU StepDriveWorld).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `active_drive.png` added; the other
   171 byte-identical. `git diff master --stat -- tests/golden` = ONLY `active_drive.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-active-drive` + `--active-drive-shot` added in introspect.cpp; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h`/`vehicle.h` + `engine/anim/` + `engine/physics/` + ALL existing shaders byte-unchanged). `scripts/
   verify.ps1` updated: `active_drive` golden in the Mac loop + `--active-drive-shot` in `$vkShots`. **The ONLY new
   shader is `active_drive_solve.comp.hlsl`; `active` does NOT appear in `hf_gen_msl`.**
