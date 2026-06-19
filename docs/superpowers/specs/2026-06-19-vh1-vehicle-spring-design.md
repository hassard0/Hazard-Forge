# Slice VH1 — Deterministic Vehicle Physics: SUSPENSION SPRING JOINT — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the BEACHHEAD) of FLAGSHIP #16
> (DETERMINISTIC VEHICLE PHYSICS, `hf::sim::vehicle`). A vehicle is the JT articulated mechanism specialized — and
> the ONE genuinely-new primitive it needs is a **suspension SPRING JOINT**: a damped distance constraint that
> holds two bodies at a REST LENGTH (not coincident like the JT1 ball joint), the `cloth::SolveDistanceConstraint`
> mold with `restLen != 0` + a stiffness scale + relative-velocity damping. This is the chassis↔wheel suspension.
> INTEGER-bit-exact (int64 → Vulkan-only shader + Metal CPU reference, the FPX3/CL3/JT1 split). Single-thread
> Gauss-Seidel. Branch: `slice-vh1`. See [[hazard-forge-vehicle-roadmap]].

**Goal:** Create `engine/sim/vehicle.h` (header-only, namespace `hf::sim::vehicle`, `#include "sim/joint.h"` +
`"sim/fpx.h"` read-only) with `FxSpringJoint` (the damped spring record) + `SolveSpringJoint` (the rest-length
positional correction + normal-velocity damping, inverse-mass split) + `StepSpringWorld` (integrate + K
Gauss-Seidel spring passes + ground — the StepJointWorld mold). Add `shaders/vehicle_spring_solve.comp.hlsl`
(int64 → Vulkan-only + Metal CPU). Add `--vehicle-spring-shot` (Vulkan) / `--vehicle-spring` (Metal). Bake the
integer golden `vehicle_spring`. **NO new RHI.**

## Design call: a damped spring = JT1 ball joint with restLen != 0 + a stiffness scale + velocity damping
A `FxSpringJoint` ties two `fpx::FxBody`s at body-local anchors (like `joint::FxJoint`) but holds them at a
**rest length** `restLen` along the anchor-to-anchor line, with a `stiffness` (the per-iteration correction
fraction — a soft constraint) and a `damping` (the relative-velocity damping along the spring normal). The world
anchor of body `b` is `WorldAnchor(b, anchorLocal) = b.pos + fpx::FxRotate(b.orient, anchorLocal)` (reuse
`joint::WorldAnchor`). The solve is **`cloth::SolveDistanceConstraint` (cloth.h:324) generalized**:
```
pa = WorldAnchor(a, j.anchorA);  pb = WorldAnchor(b, j.anchorB)
d  = pb - pa;  len = FxLength(d);  if (len == 0) skip      // coincident -> no deterministic normal
n   = FxNormalize(d);  pen = len - j.restLen               // != 0: the spring REST LENGTH (JT1 used restLen 0)
wsum = invMassA + invMassB;  if (wsum == 0) skip
wa  = fxdiv(invMassA, wsum);  wb = fxdiv(invMassB, wsum)
// (1) POSITIONAL spring restore (stiffness-scaled — a soft constraint; stiffness in [0,kOne]):
corr = fxmul(pen, j.stiffness)                              // partial correction toward restLen per iter
a.pos += FxScale(n, fxmul(corr, wa));  b.pos -= FxScale(n, fxmul(corr, wb))
// (2) NORMAL-velocity DAMPING (the spring damper — converts oscillation to heat deterministically):
vRelN = FxDot(FxSub(b.vel, a.vel), n)                      // relative velocity along the spring normal (int64)
dvel  = fxmul(vRelN, j.damping)                            // the damped impulse magnitude
a.vel += FxScale(n, fxmul(dvel, wa));  b.vel -= FxScale(n, fxmul(dvel, wb))
```
**The integer-clean choices:** `pen = len − restLen` (the spring deflection), `corr = fxmul(pen, stiffness)` (a
soft per-iteration restore — `stiffness < kOne` means partial, the standard PBD soft-constraint), and the
velocity damping `fxmul(vRelN, damping)` (the damper). All int64 (`FxLength`/`FxNormalize`/`FxDot`/`fxmul`/
`fxdiv`). `vehicle_spring_solve.comp` is **Vulkan-only** (DXC compiles int64; glslc cannot) + the Metal showcase
runs the CPU `SolveSpringJoint` (byte-identical — the FPX3/CL3/JT1 split). **Single-thread `[numthreads(1,1,1)]`**
Gauss-Seidel (order-dependent; a vehicle has ~4 springs → far under the TDR watchdog). **HONEST CAVEAT:** the
suspension stiffness is `∝ iterations` (the cloth/JT Gauss-Seidel-residual caveat — a stiff spring under heavy
load sags within a deterministic-but-nonzero band; tune stiffness/iters, NOT zero residual). Exact-deterministic
+ bit-identical cross-backend, NOT analytic spring mechanics.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The bilateral distance constraint to GENERALIZE (`engine/sim/cloth.h:324` `SolveDistanceConstraint`):** the
  `pen = len − restLen` deflection (VH1 restLen != 0, unlike JT1's restLen 0), the inverse-mass split, the
  `FxLength`/`FxNormalize`/pinned-skip. VH1 adds the `stiffness` scale + the velocity damping.
- **The JT1 ball-joint precedent (`engine/sim/joint.h`):** `FxJoint`, `SolveBallJoint` (the restLen-0 world-anchor
  projection VH1's spring generalizes), `WorldAnchor` (REUSE for the world anchors), `AnchorGap`, `StepJointWorld`
  (the integrate → K Gauss-Seidel → ground loop `StepSpringWorld` mirrors). DO NOT modify joint.h — vehicle.h is
  the new additive sibling `#include "sim/joint.h"` read-only.
- **The fpx substrate (`engine/sim/fpx.h`, read-only):** `FxBody{pos,vel,invMass,flags,radius,orient,angVel}`,
  `FxWorld`, `kFlagDynamic`, `FxRotate`, `IntegrateBodyFull`, `FxVec3` math (`FxAdd`/`FxSub`/`FxScale`/
  `FxNormalize`/`FxLength`/`fxmul`/`fxdiv`). Add a `FxDot(FxVec3,FxVec3)` inline in vehicle.h if joint.h doesn't
  export one (JT2 added `FxDot` — reuse `joint::FxDot` if present). **DO NOT modify fpx.h/joint.h.**
- **The int64 Vulkan-only single-thread solve precedent (`shaders/joint_ball_solve.comp.hlsl` JT1 /
  `joint_angular_solve.comp` JT2 / `cloth_solve.comp` CL3):** the per-pass dispatch driver + `[numthreads(1,1,1)]`
  + the int64 ops NOT in `hf_gen_msl`. `vehicle_spring_solve.comp` is the same shape.
- **Showcase + registration:** the JT `--joint-*-shot` plumbing — **standalone arg-parse loop** (the FR1 C1061
  lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden**), a NEW `tests/vehicle_test.cpp` (wire into the CMake test list like `joint_test`).

## Design decisions (locked)
1. **`vehicle.h` NEW header (namespace `hf::sim::vehicle`):** `FxSpringJoint{uint32 bodyA, bodyB; FxVec3 anchorA,
   anchorB; fx restLen; fx stiffness; fx damping;}` + `SolveSpringJoint(world, j)` (the CPU reference above) +
   `SpringLength(world, j)` helper (the current anchor-to-anchor length, for the proof) + `StepSpringWorld(world,
   springs, dt, iters)` (the StepJointWorld mold: `IntegrateBodyFull` all → K Gauss-Seidel spring passes in FIXED
   order → ground clamp) + `StepSpringWorldSteps`. Pure integer, header-only, NO device/backend symbols, NO
   `<cmath>`.
2. **`shaders/vehicle_spring_solve.comp.hlsl`** — the `SolveSpringJoint` body VERBATIM, int64, single-thread
   `[numthreads(1,1,1)]` over the spring list in FIXED order, **NOT in `hf_gen_msl`** (Vulkan-only). The host
   dispatches it once per Gauss-Seidel pass over the spring SSBO + the body SSBO.
3. **Showcase `--vehicle-spring-shot <out>` (Vulkan) AND `--vehicle-spring` (Metal) — WIRE BOTH** (standalone
   arg-parse). A BODY BOBBING ON A SPRING: a static (pinned, invMass 0) anchor body + a dynamic body hung below
   it by a `FxSpringJoint` (restLen R, stiffness, damping); drop the dynamic body (start it displaced from
   restLen), it oscillates on the spring and DAMPS to rest at ~restLen. Settle K `StepSpringWorld` steps. Vulkan:
   the GPU spring solve → **memcmp vs the CPU `StepSpringWorld`**. Metal: the CPU reference. Render a 2D side view
   (bodies as discs, the spring as a segment between anchors). Golden = `tests/golden/metal/vehicle_spring.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body world after K steps == the CPU `StepSpringWorld` reference
     byte-for-byte. Print `vehicle-spring: {bodies:<N>, springs:<S>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `vehicle-spring determinism: two runs BYTE-IDENTICAL`.
   - **(3) the spring restores + damps:** after settling, the spring length is within a small band of `restLen`
     (the spring restored the rest length), AND the body came to REST (its speed dropped below a threshold — the
     damper killed the oscillation). Print `vehicle-spring settled: {len:<L>, restLen:<R>, atRest:true}` with
     `|L − R|` below a documented bound; assert atRest.
   - **(4) the damper does the work:** a `damping=0` control keeps OSCILLATING (its end speed is NOT below the
     rest threshold) — proving the damper, not just the position solve, brings it to rest. Print `vehicle-spring
     control: {damping0:oscillating}`; assert the undamped end-speed exceeds the band.
   - **Golden discipline: ONLY `tests/golden/metal/vehicle_spring.png`; do NOT commit it.** Existing 165 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels (int64
   GPU vs Metal CPU reference — the GR3/CL3/JT1 bar).
6. **Tests `tests/vehicle_test.cpp` (NEW, pure CPU):** `SolveSpringJoint` — a body stretched beyond restLen is
   pulled IN (length decreases toward restLen); a body compressed below restLen is pushed OUT; at exactly restLen
   the positional correction is ~0; a pinned body (invMass 0) → only the dynamic one moves; the velocity damping
   reduces the relative normal speed; `damping=0` → no velocity change. `StepSpringWorld` — a displaced body on a
   damped spring settles at ~restLen at rest; two runs byte-identical; the undamped control keeps moving. Clean
   under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-vehicle-spring` (features) + `--vehicle-spring-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the JT1/cloth surface). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` +
  `fract.h` + `engine/anim/` + `engine/physics/` + all existing shaders UNCHANGED (`vehicle.h` is the new additive
  sibling). The only new shader is `vehicle_spring_solve.comp` (int64, Vulkan-only — NOT in `hf_gen_msl`). Report
  the seam empty except the new Vulkan-only spring shader.

## Out of scope (YAGNI — later VH slices)
The vehicle rig assembly (VH2 — chassis + 4 wheels + 4 springs + 4 hinges), drive/steer commands + the vehicle
tick (VH3), wheel-ground traction/friction (VH4), lockstep (VH5), the lit render (VH6). Angular spring / torsion
bar. VH1 claims ONLY: a deterministic damped spring joint that holds two bodies at a rest length + damps to rest,
bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 101) + the NEW `vehicle_test`. Clean under
   `windows-msvc-asan` (build+run `vehicle_test` + `introspect_test`).
2. **proofs + visual:** `--vehicle-spring-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image
   shows the body hung on the spring below the anchor, settled at rest (pixel-check; the JT1 lesson).**
3. Metal: `visual_test --vehicle-spring` → new golden `tests/golden/metal/vehicle_spring.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm
   `vehicle_spring_solve.comp` is correctly NOT MSL-generated (int64, Vulkan-only).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `vehicle_spring.png` added; the
   other 165 byte-identical. `git diff master --stat -- tests/golden` = ONLY `vehicle_spring.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-vehicle-spring` + `--vehicle-spring-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h` + `engine/anim/` + `engine/physics/` byte-unchanged). `scripts/verify.ps1` updated: `vehicle_spring`
   golden in the Mac loop + `--vehicle-spring-shot` in `$vkShots`. `vehicle_spring_solve.comp` NOT in `hf_gen_msl`.
   New `vehicle_test` wired into the CMake test list.
