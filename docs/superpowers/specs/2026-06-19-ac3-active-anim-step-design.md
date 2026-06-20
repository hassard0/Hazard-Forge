# Slice AC3 — Deterministic Active Ragdoll: THE ANIM-TARGET STEP (THE PILLAR BRIDGE) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #17 (DETERMINISTIC
> ACTIVE RAGDOLL / PHYSICAL-ANIMATION BLENDING, `hf::sim::active`). AC1 built the drive primitive; AC2 the
> per-joint blend weight. AC3 is **the pillar bridge**: each tick, SAMPLE an anim clip → write the sampled
> per-bone local rotations into the joints' drive targets → run the AC1/AC2 integer drive step, so a character
> **tracks an animation clip via physics torques** while still colliding/yielding. This unites the anim pillar
> (`engine/anim/`) and the physics moat (`hf::sim`). INTEGER-bit-exact sim; the clip-sample is the documented
> deterministic FLOAT crossing OUTSIDE the bit-exact loop (the JT4 bind shape). **NO new shader, NO new RHI.**
> Branch: `slice-ac3`. See [[hazard-forge-active-roadmap]].

**Goal:** Extend `engine/sim/active.h` (additive — AC1/AC2 byte-frozen) with `ActiveRagdoll` (a JT4 `joint::Ragdoll`
+ a parallel `std::vector<FxAngularDrive>` — one drive per non-root edge) + `ActiveFromSkeleton` (build the ragdoll
+ the drives) + `WriteClipTargets` (the per-tick clip-sample → qTarget snap, the float crossing) + `StepActive`
(WriteClipTargets then `StepDriveWorld`) + `StepActiveSteps`. Add `--active-step-shot` (Vulkan) / `--active-step`
(Metal). Bake the integer golden `active_step`. **NO new shader, NO new RHI.**

## Design call: sample the clip → snap each bone's local rotation into its drive's qTarget → integer step
A drive's `qTarget` is the desired RELATIVE orientation of the child body in the parent body's frame (qrel =
qParent⁻¹·qChild). The anim system's `SampleLocalPose(skeleton, clip, time)` (animation.h:45) returns, per joint, a
`JointPose{t, r, s}` whose `.r` is **exactly the bone's local rotation in its parent's frame** — i.e. precisely
the qTarget for the joint that connects the bone to its parent. So the bridge is direct: for each non-root edge,
`drive.qTarget = snap_to_Q16.16(SampleLocalPose(...)[boneOfEdge].r)`.

**`ActiveFromSkeleton(skeleton, ragdollCfg, stiffness, driveWeightFn)`** builds:
- the JT4 `joint::Ragdoll ragdoll = RagdollFromSkeleton(skeleton, ragdollCfg)` (the bodies + the ball joints +
  the cone limits — REUSED VERBATIM, the float→Q16.16 bind, joint.h:546), then
- a parallel `std::vector<FxAngularDrive> drives` — ONE per `ragdoll.joints[e]` (same non-root-edge ordering), with
  `bodyA = joints[e].bodyA` (parent), `bodyB = joints[e].bodyB` (child), `stiffness` from the arg, `driveWeight`
  from `driveWeightFn(e)` (AC2 — default kOne; lets a showcase make some bones limp). `qTarget` is left identity;
  `WriteClipTargets` fills it each tick.

**`WriteClipTargets(active, skeleton, clip, time)`** (the FLOAT crossing, host, deterministic): `pose =
SampleLocalPose(skeleton, clip, time)` (float); for each edge `e`, `active.drives[e].qTarget = FxQuatFromFloat(
pose[boneOfEdge(e)].r)` where `FxQuatFromFloat` snaps each float component to Q16.16 by **round-to-nearest** (the
JT4 / BuildPileWorld host-snap idiom: `(fx)llround(component * (double)kOne)` or the existing fpx float→fx
helper if one exists — reuse, don't reinvent). `boneOfEdge(e)` is the child joint index of edge `e` (the same
index `RagdollFromSkeleton` used to build `joints[e]` — the implementer MUST confirm the edge↔joint-index mapping
in `RagdollFromSkeleton` and mirror it; if the Ragdoll exposes the child index per joint, use it, else rebuild the
non-root-edge list identically). The snap is RNG/clock-free and `time` advances by a fixed `dt` → bit-reproducible.

**`StepActive(active, skeleton, clip, time, dt, iters)`** = `WriteClipTargets(active, skeleton, clip, time)` then
`active::StepDriveWorld(active.ragdoll.world, active.ragdoll.joints, active.ragdoll.limits, active.drives, dt,
iters)`. `StepActiveSteps(active, skeleton, clip, dt, iters, steps, startTime)` advances `time = startTime + s*dt`
each step (deterministic). The clip-sample + snap is the per-tick host pre-pass (OUTSIDE the bit-exact loop); the
`StepDriveWorld` is the integer bit-exact part — the GPU showcase runs the SAME host pre-pass (sample+snap+upload
the drives) then the AC1 `active_drive_solve.comp` per tick, memcmp'ing the GPU body world vs the CPU `StepActive`.

**The GPU==CPU contract (the JT4/FL6 bridge):** the qTargets are computed HOST-side identically for both paths
(the float sample+snap is deterministic and shared), so the only thing the GPU/CPU bit-exactly reproduce is the
integer `StepDriveWorld` over those qTargets — exactly the AC1 proof, now with per-tick-varying qTargets. No new
shader (the drive solve is AC1's `active_drive_solve.comp`; the sample+snap is host C++).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **AC1/AC2 (this branch's `active.h`, read-only — build on, DON'T modify):** `FxAngularDrive{bodyA, bodyB,
  qTarget, stiffness, driveWeight}`, `SolveAngularDrive`, `StepDriveWorld`/`StepDriveWorldSteps`, `DriveAngleCos`,
  `FxDot4`. DO NOT modify the AC1/AC2 functions. AC3 APPENDS.
- **JT4 ragdoll bind (`engine/sim/joint.h`):** `struct Ragdoll{fpx::FxWorld world; std::vector<FxJoint> joints;
  std::vector<FxAngularLimit> limits;}` (joint.h:535), `struct RagdollConfig` (joint.h:523), `RagdollFromSkeleton`
  (joint.h:546 — REUSE VERBATIM for the bodies+joints+limits), `PoseToPalette` (joint.h:630 — carried for AC6),
  `FxJoint{bodyA, bodyB, ...}` (joint.h:86 — read the child/parent fields to map edge↔bone). **DO NOT modify
  joint.h.** Confirm the EXACT non-root-edge → joint-index → child-bone-index mapping `RagdollFromSkeleton` uses.
- **The anim sampler (`engine/anim/animation.h`):** `SampleLocalPose(skeleton, animation, time) ->
  std::vector<JointPose>` (animation.h:45), `JointPose{math::Vec3 t; math::Quat r; math::Vec3 s;}` (animation.h:36
  — `.r` is the per-bone local rotation = the qTarget). `struct Animation`/`Channel` (animation.h:16-30), `struct
  Skeleton` (`anim/skeleton.h`). **DO NOT modify anim/.** (read-only).
- **The float→Q16.16 snap:** reuse the existing fpx/JT4 host float→fx helper (grep `FxFromFloat`/`llround`/the
  RagdollFromSkeleton bind snap); `FxQuat` is 4×`fx`. Do NOT reinvent a snap — mirror the JT4 bind's exactly so
  the convention matches.
- **The skeleton+clip source (`samples/hello_triangle/main.cpp` / `metal_headless/visual_test.mm`):** how
  `--joint-ragdoll-shot`/`--joint-render-shot` (JT4/JT6) load the Fox skeleton + an `Animation` clip (or build a
  synthetic skeleton — JT4 used a synthetic humanoid). REUSE that skeleton+clip setup for the AC3 scene. Standalone
  arg-parse (the FR1 C1061 lesson). The 2D render: the AC1/AC2 side-view of the posed ragdoll bodies.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), `tests/active_test.cpp`.

## Design decisions (locked)
1. **`struct ActiveRagdoll { joint::Ragdoll ragdoll; std::vector<FxAngularDrive> drives; }`** +
   **`ActiveFromSkeleton(skeleton, ragdollCfg, stiffness, driveWeightFn=kOne)`** (build the ragdoll + one drive per
   non-root edge, parent/child from `joints[e]`). **`FxQuatFromFloat(math::Quat)`** (round-to-nearest snap — the
   JT4 convention). **`WriteClipTargets(active, skeleton, clip, time)`** (sample + snap into the qTargets).
   **`StepActive`/`StepActiveSteps`** (above). Pure: the sim step is integer/fixed-order; the per-tick sample+snap
   is deterministic host float.
2. **Showcase `--active-step-shot <out>` (Vulkan) AND `--active-step` (Metal) — WIRE BOTH** (standalone arg-parse).
   The SCENE: a skeleton (the Fox or a synthetic humanoid, REUSED from JT4/JT6) bound to an `ActiveRagdoll`, driven
   to track a clip (a walk/idle `Animation`) at a fixed `time` (or a few ticks advancing `time` by `dt`) via
   `StepActive` — the ragdoll holds the clip's pose through physics torques (vs an `stiffness=0`/no-drive control
   that collapses limp). Vulkan: the host sample+snap+upload per tick + the GPU `active_drive_solve.comp` → memcmp
   vs the CPU `StepActive`. Metal: the CPU reference. Render the AC1 2D side-view of the posed bodies (discs +
   joint segments + distinct root). Golden = `tests/golden/metal/active_step.png` (Mac-baked by the CONTROLLER —
   DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body world after K `StepActive` ticks == the CPU `StepActive`
     byte-for-byte (the per-tick qTargets are host-shared; the integer step is bit-exact). Print `active-step:
     {bones:<N>, drives:<D>, ticks:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `active-step determinism: two runs BYTE-IDENTICAL`.
   - **(3) tracks the clip:** the driven ragdoll's per-edge `DriveAngleCos` to the clip's sampled targets is within
     a band (the pose is held toward the clip) AND the driven pose DIFFERS from the no-drive limp-collapse control
     by a margin. Print `active-step tracked: {meanCos:<C>, posedNotLimp:true}`; assert both.
   - **(4) clip-advance is live:** sampling the clip at two different `time`s yields DIFFERENT qTargets (the drive
     follows the animation, not a fixed pose) — OR a `time=0` rest-pose holds the bind pose. Print `active-step
     clip: {targetsAdvance:true}` (or `{restHoldsBind:true}`). Pick the clearer of the two and document it.
   - **Golden discipline: ONLY `tests/golden/metal/active_step.png`; do NOT commit it.** Existing 173 image goldens
     UNTOUCHED (incl AC1 `active_drive.png` + AC2 `active_blend.png`).
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels. (The
   float sample+snap is host-shared + deterministic → the integer step + render are bit-identical cross-backend.)
5. **Tests `tests/active_test.cpp` additions (pure CPU):** `FxQuatFromFloat` round-trips a unit quat within an LSB
   band; `ActiveFromSkeleton` builds one drive per non-root edge with the right parent/child; `WriteClipTargets`
   sets each drive's qTarget to the snapped sampled bone rotation; `StepActive` drives the ragdoll toward the clip
   pose (DriveAngleCos within band) while a no-drive control collapses; two runs byte-identical. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-active-step` (features) + `--active-step-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the AC1 compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED. `engine/
  sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `vehicle.h` + `engine/
  anim/` + `engine/physics/` + ALL shaders (incl `active_drive_solve.comp`) UNCHANGED. AC1/AC2 `active.h`
  functions UNCHANGED (AC3 additive — only the ActiveRagdoll + the clip-bridge + the step + the showcase). **NO new
  shader** (the drive solve is AC1's shader; the sample+snap is host C++). Report the seam empty.

## Out of scope (YAGNI — later AC slices)
The active→limp→recover blend factor + impact (AC4 — AC3 tracks the clip at a fixed/advancing time but does NOT yet
have a global physicality knob or an impulse that drops it then recovers), lockstep/rollback (AC5 — note for AC5:
the clip `time` is per-tick state that must be snapshotted, the VH5 lesson), the lit skinned render (AC6 — AC3's
render is the AC1 2D diagnostic, not the skinned character; AC6 swaps in `PoseToPalette` → `lit_skinned`). Root
motion, foot IK, blend trees. AC3 claims ONLY: a deterministic anim-clip-tracking active ragdoll (the bones driven
toward the clip's sampled local rotations via the integer drive), bit-identical CPU↔Vulkan↔Metal, with the integer
golden + the four proofs. NOTE (honest): the clip-sample + the float→Q16.16 qTarget snap are a deterministic FLOAT
crossing OUTSIDE the bit-exact loop (the JT4/FL6 bridge — the TARGETS are not in the GPU==CPU memcmp, only the
integer `StepDriveWorld` is); the tracking is the AC1 soft-drive proxy (within the Gauss-Seidel residual), NOT a
mass-correct motor.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 103 + the new `active_test` AC3 cases). Clean under
   `windows-msvc-asan` (build+run `active_test` + `introspect_test`).
2. **proofs + visual:** `--active-step-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the ragdoll POSED toward the clip (a coherent posed skeleton, not a limp collapse and not scrambled).** Re-run
   `--active-drive-shot` + `--active-blend-shot` → still bit-exact AND their goldens byte-identical (AC1/AC2
   render-invariance — AC3 is additive, must not perturb them).
3. Metal: `visual_test --active-step` → new golden `tests/golden/metal/active_step.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm NO new shader (`hf_gen_msl` UNCHANGED; `active` still absent).**
   Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `active_step.png` added (NOT
   `active_drive.png`/`active_blend.png`); the other 173 byte-identical. `git diff master --stat -- tests/golden` =
   ONLY `active_step.png` (metal) + the introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-active-step` + `--active-step-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + the frozen sim/anim/physics headers + ALL shaders incl `active_drive_solve.comp`
   byte-unchanged; ONLY `active.h` extended additively + the showcase/test/introspect). `scripts/verify.ps1`
   updated: `active_step` golden + `--active-step-shot` in `$vkShots`. **NO new shader; `active` still NOT in
   `hf_gen_msl`.**
