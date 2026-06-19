# Slice VH2 — Deterministic Vehicle Physics: THE VEHICLE RIG + WHEEL HINGE — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #16 (DETERMINISTIC
> VEHICLE PHYSICS, `hf::sim::vehicle`). VH1 built the suspension SPRING joint; VH2 ASSEMBLES the car — a chassis
> body + four wheel bodies tied by four VH1 springs (suspension) + four `joint::FxAngularLimit` hinges (the wheels
> spin about their axle but don't flop out of plane) — and settles it on its springs so it rests at ride height,
> wheels on the ground. The `RagdollFromSkeleton` twin (a host-built body+constraint rig) over the vehicle.
> INTEGER-bit-exact. **NO new shader** — the step drives the EXISTING VH1 `vehicle_spring_solve.comp` + JT2
> `joint_angular_solve.comp` (the JT3 host-driven multi-pass mold). Branch: `slice-vh2`. See
> [[hazard-forge-vehicle-roadmap]].

**Goal:** Extend `engine/sim/vehicle.h` (additive — VH1 byte-unchanged) with `VehicleConfig` + `struct Vehicle`
(the `fpx::FxWorld` + the spring list + the hinge list + chassis/wheel indices) + `VehicleFromConfig` (the host
assembly) + `StepVehicleRig` (integrate → K Gauss-Seidel {spring | hinge} → ground — the settle) +
`StepVehicleRigSteps` + `MeasureVehicleRig`. Add `--vehicle-rig-shot` (Vulkan) / `--vehicle-rig` (Metal). Bake
the integer golden `vehicle_rig`. **NO new shader, NO new RHI.**

## Design call: assemble the car (the RagdollFromSkeleton twin), settle on springs (no inter-body contacts yet)
A vehicle is the JT articulated mechanism specialized: `VehicleFromConfig(cfg)` host-builds an `fpx::FxWorld` of
**1 chassis `FxBody`** (a box, sphere-bound radius from its half-extents, at the config ride height) + **4 wheel
`FxBody`s** (at the four chassis-corner positions, dropped by the suspension length), plus:
- **4 `FxSpringJoint`s (VH1)** — one per corner, `bodyA = chassis`, `bodyB = wheel`, anchors at the chassis
  corner (local) and the wheel centre (local), `restLen = cfg.suspensionLen`, `stiffness`/`damping` from cfg. The
  suspension: gravity pulls the chassis down, the springs push back, the chassis floats at ride height.
- **4 `joint::FxAngularLimit` hinges (JT2, reused VERBATIM)** — one per corner, `kAngularHinge`, `axis` = the
  wheel spin axis (the lateral car axis, body-local on the chassis), so the wheel may SPIN about its axle but the
  hinge clamps the off-axis swing (the wheel stays in its rolling plane — no flopping). Reuse `SolveAngularLimit`
  + the host cos/sin hinge limit (cone limit 0 = pure hinge).

`StepVehicleRig(vehicle, dt, iters)` = the `StepSpringWorld`/`StepJointWorld` mold with BOTH passes: integrate
(`IntegrateBodyFull` all) → K Gauss-Seidel iters EACH {all `SolveSpringJoint` then all `SolveAngularLimit`} →
ground clamp (every DYNAMIC body `pos.y >= groundY + radius` so the WHEELS rest on the floor). **NO inter-body
broadphase/contacts here** — the wheels rest on the ground via the clamp, and the chassis floats on the springs;
the full broadphase+contact tick (for ramps/obstacles + drive) is VH3. So the car SETTLES: the chassis descends
until the four springs (compressed by its weight) balance gravity, the wheels resting on the ground at ride
height. Pure integer, fixed op order → two-run bit-identical AND bit-exact GPU==CPU. **The GPU showcase drives
the EXISTING `vehicle_spring_solve.comp` + `joint_angular_solve.comp` (both int64, Vulkan-only) in the locked
order, memcmp vs the CPU `StepVehicleRig`; Metal runs the CPU reference. NO new shader.**

## Reuse map (file:line — the implementer MUST ground these before coding)
- **VH1 (this branch's `vehicle.h`, read-only — build on, DON'T modify):** `FxSpringJoint`, `SolveSpringJoint`,
  `SpringLength`, `StepSpringWorld` (the integrate → K spring passes → ground mold VH2 extends with the hinge
  pass). DO NOT modify VH1 code or `vehicle_spring_solve.comp`.
- **JT angular limits to REUSE VERBATIM (`engine/sim/joint.h`):** `FxAngularLimit`, `SolveAngularLimit`,
  `kAngularHinge`, `SwingAngleCos`, the host `cosHalfLimit`/`sinHalfLimit` (a hinge = cone limit 0). `joint.h`'s
  `joint_angular_solve.comp` is the existing Vulkan-only shader the GPU rig drives. DO NOT modify joint.h.
- **The assembly pattern (`engine/sim/joint.h::RagdollFromSkeleton` — JT4):** the host body+constraint builder
  (bodies at world positions, joints+limits per edge, anchors in body-local frames via `FxRotate(QConj(orient),
  worldPos − bodyPos)`). `VehicleFromConfig` is the SAME shape — a fixed 5-body rig instead of a skeleton tree.
- **The host-driven multi-pass step (`engine/sim/joint.h::StepArticulatedContacts` — JT3 / `fract.h::StepFracture`
  — FR4):** the integrate → K {pass | pass} → ground tick + the GPU showcase that drives the existing int64
  shaders in the locked order and memcmp's vs the CPU reference. VH2's `StepVehicleRig` is this with {spring |
  hinge} (NO contacts yet — VH3 adds broadphase+contacts).
- **The fpx substrate (`engine/sim/fpx.h`, read-only):** `FxBody{pos,vel,invMass,flags,radius,orient,angVel}`,
  `FxWorld`, `kFlagDynamic`, `IntegrateBodyFull`, `FxBodyTransform` (carried for VH6). **DO NOT modify fpx.h.**
- **Showcase + registration:** VH1's `--vehicle-spring-shot` plumbing — **standalone arg-parse loop** (the FR1
  C1061 lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden**), `tests/vehicle_test.cpp`.

## Design decisions (locked)
1. **`VehicleConfig{ fx rideHeight, suspensionLen, springStiffness, springDamping, chassisHalfX/Y/Z (or radius),
   wheelRadius, fx wheelBaseX, wheelBaseZ (the 4 corner offsets), hingeCosHalf/SinHalf, FxVec3 gravity, fx
   groundY, fx chassisInvMass, wheelInvMass; }`** + **`struct Vehicle{ fpx::FxWorld world; std::vector<FxSpringJoint>
   springs; std::vector<joint::FxAngularLimit> hinges; uint32_t chassisIndex; uint32_t wheelIndex[4]; }`**.
   `VehicleFromConfig(cfg) → Vehicle` (the host assembly above). Pure integer, host-fixed.
2. **`StepVehicleRig(vehicle, dt, iters)`** — integrate → K Gauss-Seidel {all `SolveSpringJoint` | all
   `SolveAngularLimit`} → ground clamp (wheels `pos.y >= groundY + wheelRadius`, chassis floats). +
   `StepVehicleRigSteps`. `MeasureVehicleRig(vehicle, cfg)` → the chassis ride height (pos.y), the mean spring
   length (compression), the wheels-on-ground stat, the hinge off-axis swing — deterministic Q16.16 stats. NO new
   shader (reuses VH1 + JT2 shaders).
3. **Showcase `--vehicle-rig-shot <out>` (Vulkan) AND `--vehicle-rig` (Metal) — WIRE BOTH** (standalone arg-parse),
   host-driven multi-pass. Build a car (`VehicleFromConfig`), drop it, settle K `StepVehicleRig` steps → the
   chassis at ride height + the 4 wheels resting on the ground, suspension compressed. Vulkan: the GPU multi-pass
   (spring + angular shaders, locked order) → **memcmp vs the CPU `StepVehicleRig`**. Metal: the CPU reference.
   Render a 2D side view (chassis box + 2 visible wheels as discs + the suspension springs as segments + the
   ground line). Golden = `tests/golden/metal/vehicle_rig.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU vehicle world after K steps == the CPU `StepVehicleRig` reference
     byte-for-byte. Print `vehicle-rig: {bodies:<N>, springs:4, hinges:4, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `vehicle-rig determinism: two runs BYTE-IDENTICAL`.
   - **(3) the car settled on its suspension:** after settling, the 4 wheels rest at/above `groundY + wheelRadius`
     AND the chassis floats ABOVE the wheels at ~ride height (chassis pos.y in the expected band) AND the springs
     are COMPRESSED (mean spring length < restLen — they hold the chassis up). Print `vehicle-rig settled:
     {chassisY:<C>, wheelsOnGround:true, springCompressed:true}`; assert all.
   - **(4) the hinges hold the wheels in-plane:** every wheel's off-axis swing is within the hinge band (the
     wheels didn't flop out of their rolling plane). Print `vehicle-rig hinges: all 4 wheels in-plane`.
   - **Golden discipline: ONLY `tests/golden/metal/vehicle_rig.png`; do NOT commit it.** Existing 166 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
6. **Tests `tests/vehicle_test.cpp` additions (pure CPU):** `VehicleFromConfig` — 5 bodies (1 chassis + 4
   wheels), 4 springs (chassis↔wheel), 4 hinges; the chassis is at ride height, the wheels below it by ~suspension
   length; `chassisInvMass`/`wheelInvMass` set per cfg. `StepVehicleRig` — a dropped car settles with the chassis
   floating above the wheels, the wheels on the ground, the springs compressed (length < restLen); two runs
   byte-identical; the hinges keep the wheels in-plane. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-vehicle-rig` (features) + `--vehicle-rig-shot` (showcases). **REBAKE
   `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the VH1/JT surface). `rhi.h` + backend
  dirs UNCHANGED. `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` +
  `engine/anim/` + `engine/physics/` + all existing shaders UNCHANGED. VH1 `vehicle.h` code + `vehicle_spring_solve.
  comp` UNCHANGED (VH2 additive — only the rig + the step + the showcase). **NO new shader** (drives VH1's spring
  shader + JT2's angular shader). Report the seam empty.

## Out of scope (YAGNI — later VH slices)
Drive/steer commands + the full broadphase+contact tick (VH3 — VH2's step has no inter-body contacts; the wheels
rest via the ground clamp), wheel-ground traction/friction (VH4), lockstep (VH5), the lit render (VH6). Real tyre
geometry / a steering rack / an anti-roll bar. VH2 claims ONLY: a deterministic vehicle RIG (chassis + 4 wheels +
4 springs + 4 hinges) that settles at ride height on its suspension, bit-identical CPU↔Vulkan↔Metal, with the
integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 102) + the new `vehicle_test` rig cases. Clean under
   `windows-msvc-asan` (build+run `vehicle_test` + `introspect_test`).
2. **proofs + visual:** `--vehicle-rig-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the car settled at ride height — chassis floating on the springs, wheels on the ground (pixel-check; the VH1
   lesson).** ALSO re-run `--vehicle-spring-shot` → still bit-exact (VH1 render-invariance).
3. Metal: `visual_test --vehicle-rig` → new golden `tests/golden/metal/vehicle_rig.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (VH2 reuses the
   VH1 + JT2 shaders — `hf_gen_msl` UNCHANGED).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `vehicle_rig.png` added; the other
   166 byte-identical. `git diff master --stat -- tests/golden` = ONLY `vehicle_rig.png` (metal) + the introspect
   json.
5. Introspect JSON rebaked exactly `+deterministic-vehicle-rig` + `--vehicle-rig-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h` + `engine/anim/` + `engine/physics/` + VH1 `vehicle.h`/`vehicle_spring_solve.comp` byte-unchanged).
   `scripts/verify.ps1` updated: `vehicle_rig` golden in the Mac loop + `--vehicle-rig-shot` in `$vkShots`. **NO
   new entry in `hf_gen_msl`.**
