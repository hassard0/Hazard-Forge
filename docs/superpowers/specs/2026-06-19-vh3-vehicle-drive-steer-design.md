# Slice VH3 — Deterministic Vehicle Physics: DRIVE + STEER COMMANDS + THE LOCKED VEHICLE TICK — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #16 (DETERMINISTIC
> VEHICLE PHYSICS, `hf::sim::vehicle`). VH1 built the suspension spring; VH2 assembled + settled the car (chassis +
> 4 wheels + 4 springs + 4 hinges). VH3 makes it RESPOND TO INPUT: two new integer `fpx::FxCommand` kinds —
> `kCmdDriveTorque` (spin a wheel about its axle) + `kCmdSteer` (rotate the front-wheel hinge axes) — fed through a
> single locked vehicle tick `StepVehicle` that adds the fpx broadphase + sphere-sphere CONTACT pass to VH2's
> spring+hinge solve (the JT3 contacts-as-a-trailing-block mold). INTEGER-bit-exact. **NO new shader** — the GPU
> showcase drives the EXISTING VH1 `vehicle_spring_solve.comp` + JT2 `joint_angular_solve.comp` + fpx
> `fpx_solve.comp` in the locked order. Branch: `slice-vh3`. See [[hazard-forge-vehicle-roadmap]].

**Goal:** Extend `engine/sim/vehicle.h` (additive — VH1/VH2 byte-unchanged) with `kCmdDriveTorque` + `kCmdSteer`
(2 new `fpx::FxCommand` kind values + `ApplyVehicleCommand`) + `StepVehicle` (the locked tick: apply this tick's
commands → integrate → K Gauss-Seidel {spring | hinge} → `fpx::BuildPairs` once → `fpx::StepWorld(dt=0)` contacts
+ ground) + `StepVehicleSteps` + a deterministic drive/steer stat. Add `--vehicle-step-shot` (Vulkan) /
`--vehicle-step` (Metal). Bake the integer golden `vehicle_step`. **NO new shader, NO new RHI.**

## Design call: input as deterministic integer commands, contacts as a trailing block (the JT3 mold)
A drivable vehicle needs (a) input that perturbs the rig deterministically and (b) inter-body contacts so the car
rests/rolls on the ground and obstacles rather than only the VH2 per-wheel ground-clamp. Both reuse existing
machinery:

**Drive + steer = 2 new integer `fpx::FxCommand` kinds.** `fpx::FxCommand{tick, kind, target, arg}` already exists
(the FPX5/lockstep substrate). VH3 adds two NEW `kind` values (defined in `vehicle.h`, NOT in fpx.h — fpx.h stays
frozen; the values are vehicle-local constants above fpx's existing range, applied by `ApplyVehicleCommand`, never
by `fpx::ApplyCommand`):
- **`kCmdDriveTorque`** — `target` = a wheel body index, `arg` = a signed Q16.16 magnitude. Adds an angular
  impulse about that wheel's spin axis to the wheel's `angVel` (`wheel.angVel += FxScale(spinAxis, arg)` via the
  fpx integer vector ops). The throttle: spins the driven wheels.
- **`kCmdSteer`** — `target` = a front hinge index (0/1), `arg` = a signed Q16.16 steer angle. Rotates that
  hinge's `axis` about the chassis up-axis by a **host-snapped** steer quaternion (`QFromAxisAngleSnapped` — build
  the half-angle cos/sin with the fpx fixed-point trig LUT / host round-to-nearest, the BuildPileWorld idiom, so
  the command stream is bit-reproducible) and re-normalizes the axis. Steering re-aims the front wheels' rolling
  plane; the hinge then holds the wheel in that re-aimed plane.

Static/invalid targets (out-of-range index, a kFlagStatic body) are no-ops. Integer adds + a host-snapped
quaternion only — NO runtime transcendentals, NO float in the sim path.

**`StepVehicle(vehicle, cfg, commands, tick, dt, iters, solveIters)`** = the locked tick, the
`joint::StepArticulatedContacts` (JT3) / `fract::StepFracture` (FR4) mold with the VH2 spring pass folded in:
1. **Apply this tick's commands** in ARRAY ORDER (`ApplyVehicleCommand` for each cmd with `cmd.tick == tick`).
2. **Integrate** — `fpx::IntegrateBodyFull` all bodies (the VH2 integrate).
3. **K Gauss-Seidel constraint iters**, each: all `SolveSpringJoint` (VH1) then all `SolveAngularLimit` (JT2) — the
   VH2 solve loop VERBATIM. (No ball joints in a vehicle — springs + hinges only.)
4. **Contacts as a trailing block** (the JT3 deviation, reused): `fpx::BuildPairs` ONCE over the world, then
   `fpx::StepWorld(dt=0, solveIters)` — the dt=0 integrate-suppressed ground + FPX3 sphere-sphere contacts, so the
   wheels/chassis rest on the ground and on each other as a coherent body. (dt=0 keeps the whole-step int64
   shaders bit-exact; per-iteration interleave is NOT bit-idempotent for orientation — the JT3 finding.)

`StepVehicleSteps` runs K ticks. `MeasureVehicleDrive(vehicle, cfg)` → the chassis world position + forward
displacement (did it move under drive), the mean driven-wheel `angVel` magnitude (did the throttle take), the
front-hinge axis heading (did steer re-aim it), the residual contact overlaps — deterministic Q16.16 stats. Pure
integer, fixed op order → two-run bit-identical AND bit-exact GPU==CPU.

**The GPU showcase decomposition (NO new shader, the JT3 precedent):** apply the commands HOST-side (shared input,
identical both paths), then the GPU drives `vehicle_spring_solve.comp` + `joint_angular_solve.comp` (steps=1,
iters=K, ground sentinel far below so their floor-clamp is dead) for (2)+(3), then host-rebuilds the FPX2 pair
list and drives `fpx_solve.comp` (dt=0, real groundY) for (4) — the SAME ops as the CPU `StepVehicle` → GPU body
world memcmp BIT-EXACT. Both shaders int64 → Vulkan-only; Metal `--vehicle-step` runs the CPU `StepVehicle`
(byte-identical by construction — the FPX3/CL3/JT1 split).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **VH1/VH2 (this branch's `vehicle.h`, read-only — build on, DON'T modify):** `FxSpringJoint`,
  `SolveSpringJoint`, `VehicleConfig`, `struct Vehicle`, `VehicleFromConfig`, `StepVehicleRig` (the integrate → K
  {spring | hinge} → ground mold VH3 extends with command-apply + the contact trailing block). DO NOT modify
  VH1/VH2 code, `vehicle_spring_solve.comp`, or the VH2 showcase.
- **fpx command + step substrate to REUSE VERBATIM (`engine/sim/fpx.h`, read-only):** `FxCommand{tick,kind,target,
  arg}`, `FxBody{pos,vel,invMass,flags,radius,orient,angVel}`, `FxWorld`, `IntegrateBodyFull`, `BuildPairs`,
  `StepWorld(dt, solveIters)` (the dt=0 contacts-only call), `FxScale`/`FxAdd`/`FxNormalize`/`FxRotate`/`fxmul`
  (the Q16.16 vector ops), the fixed-point trig used by `QFromAxisAngle` if present. **DO NOT modify fpx.h** —
  define the 2 new command-kind constants + `ApplyVehicleCommand` in `vehicle.h`.
- **JT angular limits to REUSE VERBATIM (`engine/sim/joint.h`):** `FxAngularLimit`, `SolveAngularLimit`,
  `kAngularHinge` (the hinge whose `axis` `kCmdSteer` re-aims). DO NOT modify joint.h.
- **The host-driven multi-pass + contacts-as-trailing-block tick (`engine/sim/joint.h::StepArticulatedContacts`
  — JT3; `fract.h::StepFracture` — FR4):** the integrate → K {pass | pass} → BuildPairs → StepWorld(dt=0) tick +
  the GPU showcase that drives the existing int64 shaders in the locked order and memcmp's vs the CPU reference.
  VH3's `StepVehicle` is this with the spring pass folded in + the command-apply prologue.
- **The command-stream idiom (`engine/sim/joint.h` JT5 `SimRagdollTick` / `fpx.h` FPX5 `SimTick`):** apply a
  tick's commands in array order before stepping; out-of-range/static targets no-op. VH3's `ApplyVehicleCommand` +
  the tick's command-apply prologue follow this shape (NO snapshot/rollback here — that's VH5).
- **Showcase + registration:** VH2's `--vehicle-rig-shot` plumbing — **standalone arg-parse loop** (the FR1 C1061
  lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden**), `tests/vehicle_test.cpp`.

## Design decisions (locked)
1. **2 new command kinds in `vehicle.h`** (vehicle-local constants, NOT added to fpx.h): `kCmdDriveTorque`,
   `kCmdSteer`. **`ApplyVehicleCommand(vehicle, cfg, cmd)`** — `kCmdDriveTorque` adds `FxScale(spinAxis, cmd.arg)`
   to `world.bodies[cmd.target].angVel` (skip if target out-of-range or static); `kCmdSteer` rotates
   `hinges[cmd.target].axis` by a host-snapped steer quaternion about the chassis up-axis (skip if target ∉
   {front hinges}). Integer only; host-snapped quaternion (round-to-nearest) so the stream is bit-reproducible.
2. **`StepVehicle(vehicle, cfg, commands, tick, dt, iters, solveIters)`** — apply tick commands → integrate → K
   {all `SolveSpringJoint` | all `SolveAngularLimit`} → `fpx::BuildPairs` once → `fpx::StepWorld(dt=0, solveIters)`
   → done. + **`StepVehicleSteps(vehicle, cfg, commands, dt, ticks, iters, solveIters)`** (runs ticks 0..K-1, each
   applying its own commands). + **`MeasureVehicleDrive(vehicle, cfg)`** → chassis pos/forward-displacement, mean
   driven-wheel angVel magnitude, front-hinge heading, residual contact overlaps — deterministic Q16.16. NO new
   shader (reuses VH1 + JT2 + fpx shaders).
3. **Showcase `--vehicle-step-shot <out>` (Vulkan) AND `--vehicle-step` (Metal) — WIRE BOTH** (standalone
   arg-parse), host-driven multi-pass. Build a car (`VehicleFromConfig`), feed a SCRIPTED command stream — a few
   ticks of `kCmdDriveTorque` on the rear wheels + a `kCmdSteer` on a front hinge — over K `StepVehicle` ticks →
   the car drives forward and the front wheels re-aim. Vulkan: the GPU multi-pass (spring + angular + fpx contact
   shaders, locked order) → **memcmp vs the CPU `StepVehicle`**. Metal: the CPU reference. Render a 2D
   side/top view (chassis box + wheels as discs at their bit-exact `pos>>kFrac` + the suspension springs as
   segments + the ground line; the driven path traced). Golden = `tests/golden/metal/vehicle_step.png` (Mac-baked
   by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU vehicle world after K command-driven ticks == the CPU `StepVehicle`
     reference byte-for-byte. Print `vehicle-step: {bodies:<N>, commands:<C>, ticks:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `vehicle-step determinism: two runs BYTE-IDENTICAL`.
   - **(3) drive took / the car moved:** under the drive stream the chassis forward displacement > a band AND the
     driven wheels' mean angVel magnitude > 0 (the throttle spun them). Print `vehicle-step drive: {forward:<F>,
     wheelSpin:<W>, moved:true}`; assert both. A NO-command control stays put (forward ≈ 0) — print `vehicle-step
     control: {noCommand:idle}`.
   - **(4) steer re-aimed the front wheels:** after a `kCmdSteer`, the front-hinge axis heading changed from the
     rest heading by the expected (host-snapped) amount AND the rear-hinge headings are unchanged. Print
     `vehicle-step steer: {frontReaimed:true, rearUnchanged:true}`.
   - **Golden discipline: ONLY `tests/golden/metal/vehicle_step.png`; do NOT commit it.** Existing 167 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
6. **Tests `tests/vehicle_test.cpp` additions (pure CPU):** `ApplyVehicleCommand` — a `kCmdDriveTorque` raises the
   target wheel's angVel by the snapped amount, a static/out-of-range target no-ops; a `kCmdSteer` rotates only
   the targeted front hinge's axis, leaves the rears. `StepVehicle` — a drive stream moves the chassis forward +
   spins the driven wheels; a no-command run leaves the car at its VH2 settled pose (forward ≈ 0); two runs
   byte-identical; residual contact overlaps within band (the car non-interpenetrating). Clean under
   `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-vehicle-drive` (features) + `--vehicle-step-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the VH1/VH2/JT/fpx surface). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` +
  `fract.h` + `engine/anim/` + `engine/physics/` + all existing shaders UNCHANGED. VH1/VH2 `vehicle.h` code +
  `vehicle_spring_solve.comp` + the VH2 showcase UNCHANGED (VH3 additive — only the 2 commands + the tick + the
  showcase). **NO new shader** (drives VH1's spring shader + JT2's angular shader + fpx's contact shader). Report
  the seam empty.

## Out of scope (YAGNI — later VH slices)
Wheel-ground traction/friction (VH4 — VH3's drive spins wheels + moves the chassis via the command/integrate path;
the tangential tyre-friction beat that couples spin→ground grip is VH4), lockstep/rollback (VH5), the lit render
(VH6). Real tyre slip curves, a differential, gearbox, aerodynamic drag, an anti-roll bar. VH3 claims ONLY: a
deterministic drivable vehicle TICK (drive-torque + steer integer commands + spring/hinge solve + fpx contacts)
that moves + re-aims the car under a scripted input stream, bit-identical CPU↔Vulkan↔Metal, with the integer
golden + the four proofs. NOTE (honest, the FR4/JT3 caveat carried forward): fpx contacts are sphere-sphere with
NO inertia tensor — a contact does NOT spin a body, so wheel rolling-from-ground-contact is NOT emergent; drive
comes from the command/integrate path and VH4's traction is a deterministic tangential PROXY, not analytic tyre
mechanics. Determinism + cross-platform bit-identity is the headline, NOT "more physically correct."

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 102) + the new `vehicle_test` drive/steer cases. Clean
   under `windows-msvc-asan` (build+run `vehicle_test` + `introspect_test`).
2. **proofs + visual:** `--vehicle-step-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the car driven forward with the front wheels re-aimed (pixel-check; the VH1 lesson).** ALSO re-run
   `--vehicle-rig-shot` + `--vehicle-spring-shot` → still bit-exact (VH1/VH2 render-invariance).
3. Metal: `visual_test --vehicle-step` → new golden `tests/golden/metal/vehicle_step.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (VH3 reuses the
   VH1 + JT2 + fpx shaders — `hf_gen_msl` UNCHANGED).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `vehicle_step.png` added; the
   other 167 byte-identical. `git diff master --stat -- tests/golden` = ONLY `vehicle_step.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-vehicle-drive` + `--vehicle-step-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h` + `engine/anim/` + `engine/physics/` + VH1/VH2 `vehicle.h` (the pre-VH3 functions) /
   `vehicle_spring_solve.comp` byte-unchanged). `scripts/verify.ps1` updated: `vehicle_step` golden in the Mac
   loop + `--vehicle-step-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`.**
