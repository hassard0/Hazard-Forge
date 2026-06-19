# Slice JT1 — Deterministic Articulated-Body Ragdoll: JOINT GRAPH + BALL-JOINT CONSTRAINT — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the BEACHHEAD) of FLAGSHIP #15
> (DETERMINISTIC ARTICULATED-BODY RAGDOLL, `hf::sim::joint`). The rigid solver `fpx.h` is already 6-DOF but has
> NO bilateral constraint tying two bodies — JT1 adds the MISSING PRIMITIVE: a BALL JOINT (the anchors of two
> bodies pinned coincident), the verbatim generalization of `cloth::SolveDistanceConstraint` to rest-length 0
> with body-local anchors rotated by `fpx::FxRotate`. This turns "piles of independent spheres" into a CHAIN /
> mechanism. INTEGER-bit-exact (int64 → Vulkan-only shader + Metal CPU reference, the FPX3/cloth split). Single-
> thread Gauss-Seidel (joints are order-dependent). Branch: `slice-jt1`. See [[hazard-forge-joint-roadmap]].

**Goal:** Create `engine/sim/joint.h` (header-only, namespace `hf::sim::joint`, `#include "sim/fpx.h"` read-only)
with `FxJoint` (the joint record) + `SolveBallJoint` (the world-anchor distance-0 projection with inverse-mass
split) + `StepJointWorld` (integrate + K Gauss-Seidel ball-joint passes + ground — the StepCloth mold over
`fpx::FxWorld`). Add `shaders/joint_ball_solve.comp.hlsl` (int64 → Vulkan-only + Metal CPU). Add `--joint-ball-shot`
(Vulkan) / `--joint-ball` (Metal). Bake the integer golden `joint_ball`. **NO new RHI.**

## Design call: a ball joint = cloth `SolveDistanceConstraint` over BODY anchors, restLen 0
A `FxJoint` ties two `fpx::FxBody`s at a pair of **body-local anchor points** (`anchorA`/`anchorB`, Q16.16 offsets
from each body's centre). The world anchor of body `b` is `worldAnchor = b.pos + fpx::FxRotate(b.orient,
anchorLocal)` (the anchor offset rotated into the body's current orientation — `FxRotate` exists, fpx.h:440). The
BALL constraint pins the two world anchors COINCIDENT (a distance-0 constraint), resolved by an inverse-mass-
weighted **positional** correction of the two body centres — **literally `cloth::SolveDistanceConstraint`
(cloth.h:324) with `restLen = 0`, the constraint endpoints being the world anchors instead of the particle
positions, and the correction applied to `b.pos`**:
```
pa = posA + FxRotate(orientA, anchorA);  pb = posB + FxRotate(orientB, anchorB)
d  = pb - pa;  len = FxLength(d);  if (len == 0) skip      // coincident -> no deterministic normal
wsum = invMassA + invMassB;  if (wsum == 0) skip            // both pinned
n   = FxNormalize(d);  pen = len - 0 = len                  // restLen 0: pull the anchors together
wa  = fxdiv(invMassA, wsum);  wb = fxdiv(invMassB, wsum)
posA += FxScale(n, fxmul(pen, wa))   // A's centre moves toward the midpoint
posB -= FxScale(n, fxmul(pen, wb))   // B's centre moves toward the midpoint
```
**THE JT1 SIMPLIFICATION (honest, documented):** JT1 satisfies the anchor constraint by TRANSLATING the body
centres (the cloth positional split). It does NOT yet rotate a body to align an off-centre anchor (no lever-arm /
angular coupling — that is the JT2/JT3 angular work). With anchors at the link ends, a translation-only ball
joint already produces a hanging, swinging CHAIN (each link's anchor pulled to its neighbour's); orientation is
driven by `IntegrateBodyFull`'s angular integrate, not yet by the joint. int64 (`FxRotate` fxmul + `fxdiv`/
`FxLength`) → `joint_ball_solve.comp` is **Vulkan-only** (DXC compiles int64; glslc cannot) + the Metal showcase
runs the CPU `SolveBallJoint` (byte-identical by construction — the FPX3/cloth split). **Single-thread Gauss-
Seidel `[numthreads(1,1,1)]`** (each joint reads the centres the earlier joints THIS pass already moved → order-
dependent, the cloth CL3 / fpx FPX3 discipline; a chain/ragdoll has ~15 joints → far under the TDR watchdog).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The bilateral constraint to GENERALIZE (`engine/sim/cloth.h:324` `SolveDistanceConstraint`):** the inverse-
  mass split (`wsum`, `wi`/`wj` via `fxdiv`), the `FxLength`/`FxNormalize` direction, the pinned-skip, the
  `pos += n·fxmul(pen,w)` correction. JT1's `SolveBallJoint` is THIS with `restLen 0` over the world anchors.
  Also `StepCloth` (cloth.h:353) — the integrate → K Gauss-Seidel passes (FIXED order, sequential) → ground
  clamp loop `StepJointWorld` mirrors.
- **The fpx substrate to REUSE VERBATIM (`engine/sim/fpx.h`):** `FxBody{pos,vel,invMass,flags,radius,orient,
  angVel}` (:116), `FxWorld{gravity,groundY,bodies}` (:135), `kFlagDynamic` (:133), `FxRotate` (:440 — the
  body-local→world anchor rotation), `IntegrateBody`/`IntegrateBodyFull` (:149/:479 — JT1 uses `IntegrateBodyFull`
  for the 6-DOF integrate; the chain links carry orientation), `ResolveGround`/the floor clamp, `FxVec3` math
  (`FxAdd`/`FxSub`/`FxScale`/`FxNormalize`/`FxLength`/`fxmul`/`fxdiv`). **DO NOT modify fpx.h** — `joint.h` is the
  new additive sibling, `#include "sim/fpx.h"` read-only.
- **The int64 Vulkan-only + Metal-CPU split precedent (`shaders/grain_contact.comp.hlsl` GR3 / `cloth_solve.comp`
  CL3 / `cgf_buoyancy.comp` GF2):** an int64 single-thread compute shader NOT in `hf_gen_msl` (Vulkan-only via
  DXC) + the Metal showcase runs the CPU reference. `joint_ball_solve.comp` is the same. The host driver dispatches
  it once per Gauss-Seidel pass (the cloth_solve per-pass driver), memcmp vs the CPU `StepJointWorld`.
- **The fract/cloth showcase + golden mold:** the `--cloth-*`/`--fract-*` showcases (a settle → render a 2D/3D
  view + the proofs). **Add `--joint-ball-shot` using the STANDALONE arg-parse loop pattern** (the FR1 C1061
  lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden**), a NEW `tests/joint_test.cpp` (wire into the CMake test list like `fract_test`).

## Design decisions (locked)
1. **`joint.h` NEW header (namespace `hf::sim::joint`):** `kJointBall = 0` (kind; hinge/cone are JT2) +
   `FxJoint{uint32 bodyA, bodyB; FxVec3 anchorA, anchorB; uint32 kind; fx limit;}` (limit unused at JT1, carried
   for JT2) + `SolveBallJoint(world, joint)` (the CPU reference above) + `WorldAnchor(body, anchorLocal)` helper +
   `StepJointWorld(world, joints, dt, iters)` (the StepCloth mold: `IntegrateBodyFull` all → K Gauss-Seidel ball
   passes in FIXED joint order → ground clamp) + `StepJointWorldSteps`. Pure integer, header-only, NO device/
   backend symbols, NO `<cmath>`.
2. **`shaders/joint_ball_solve.comp.hlsl`** — the `SolveBallJoint` body VERBATIM, int64, single-thread
   `[numthreads(1,1,1)]` over the joint list in FIXED order (Gauss-Seidel), **NOT in `hf_gen_msl`** (Vulkan-only).
   The host dispatches it once per Gauss-Seidel pass over the joint SSBO + the body SSBO.
3. **Showcase `--joint-ball-shot <out>` (Vulkan) AND `--joint-ball` (Metal) — WIRE BOTH** (standalone arg-parse).
   A HANGING CHAIN: a static (pinned, invMass 0) root body at the top + a chain of ~6-10 dynamic bodies, each
   linked to its parent by a ball joint (anchorA at the parent's lower end, anchorB at the child's upper end).
   Settle K `StepJointWorld` steps under gravity → the chain hangs/swings and the links stay connected at their
   anchors. Vulkan: the GPU joint solve → **memcmp vs the CPU `StepJointWorld`**. Metal: the CPU reference. Render
   a 2D side view (each body a disc at `pos`, the joints as line segments between world anchors). Golden =
   `tests/golden/metal/joint_ball.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body world after K steps == the CPU `StepJointWorld` reference
     byte-for-byte. Print `joint-ball: {bodies:<N>, joints:<J>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `joint-ball determinism: two runs BYTE-IDENTICAL`.
   - **(3) the joints HOLD:** after settling, every joint's world-anchor gap `|pb − pa|` is within a small Q16.16
     band (the constraint is satisfied — the chain is connected, not flying apart). Print `joint-ball connected:
     max anchor gap <G> within band` with `G` below a documented bound. (Gauss-Seidel residual is deterministic-
     but-nonzero — the cloth/fract caveat shape; assert the gap is SMALL, not zero.)
   - **(4) the pinned root HELD + the chain hung:** the static root body is UNCHANGED (invMass 0) AND the chain's
     mean pos.y dropped below the root (it hung under gravity). Print `joint-ball hang: {rootHeld:true,
     dropped:true}`.
   - **Golden discipline: ONLY `tests/golden/metal/joint_ball.png`; do NOT commit it.** Existing 159 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels (int64
   GPU vs Metal CPU reference — the GR3/CL3/GF2 bar).
6. **Tests `tests/joint_test.cpp` (NEW, pure CPU):** `WorldAnchor` rotates the local anchor by orient + adds pos
   (identity orient → pos + anchor). `SolveBallJoint` — two equal-mass bodies with offset anchors → both move
   half the gap toward coincidence (one projection halves the gap by the inverse-mass split); a pinned body
   (invMass 0) → only the dynamic one moves; coincident anchors → no-op; both pinned → no-op. `StepJointWorld` —
   a 1-joint chain (pinned root + 1 dynamic) settles with the gap shrinking + the dynamic body below the root;
   two runs byte-identical. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-joint-ball` (features) + `--joint-ball-shot` (showcases). **REBAKE
   `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the cloth/fract surface). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` +
  `engine/physics/` + all existing shaders UNCHANGED (`joint.h` is the new additive sibling). The only new shader
  is `joint_ball_solve.comp` (int64, Vulkan-only — NOT in `hf_gen_msl`). Report the seam empty except the new
  Vulkan-only solve shader.

## Out of scope (YAGNI — later JT slices)
Angular limits / hinge / cone-twist (JT2 — JT1 is the positional ball joint only, no orientation coupling), the
articulated multi-body step interleaving joints with contacts (JT3), the skeleton→ragdoll bind (JT4), lockstep
(JT5), the lit render (JT6). Lever-arm / angular ball-joint coupling (an off-centre anchor rotating its body —
JT1 translates only). Inertia tensor / torque-from-contact (the documented fpx deferral). JT1 claims ONLY: a
deterministic positional ball-joint constraint that hangs a chain, bit-identical CPU↔Vulkan↔Metal, with the
integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 100) + the NEW `joint_test`. Clean under `windows-msvc-asan`
   (build+run `joint_test` + `introspect_test`).
2. **proofs + visual:** `--joint-ball-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   a coherent hanging/swinging chain — connected links dangling from the pinned root (pixel-check; the cloth/FR
   lesson).**
3. Metal: `visual_test --joint-ball` → new golden `tests/golden/metal/joint_ball.png`; two runs DIFF 0.0000 (gate
   on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm `joint_ball_solve.comp` is correctly
   NOT MSL-generated (int64, Vulkan-only).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `joint_ball.png` added; the other
   159 byte-identical. `git diff master --stat -- tests/golden` = ONLY `joint_ball.png` (metal) + the introspect
   json.
5. Introspect JSON rebaked exactly `+deterministic-joint-ball` + `--joint-ball-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/`fract.h` +
   `engine/physics/` byte-unchanged). `scripts/verify.ps1` updated: `joint_ball` golden in the Mac loop +
   `--joint-ball-shot` in `$vkShots`. `joint_ball_solve.comp` NOT in `hf_gen_msl`. New `joint_test` wired into the
   CMake test list.
