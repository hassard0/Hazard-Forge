# Slice VH4 — Deterministic Vehicle Physics: WHEEL-GROUND TRACTION / FRICTION (THE NEW-PHYSICS BEAT) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #16 (DETERMINISTIC
> VEHICLE PHYSICS, `hf::sim::vehicle`) — the NEW-PHYSICS slice (the GR4-friction / JT2-cone-limit beat). VH3 drove
> the car forward with an honest velocity SEED (the integrate-path throttle) because sphere contacts have no
> inertia tensor. VH4 replaces that seed with a REAL deterministic TRACTION model: at each grounded wheel, a
> Coulomb-friction-cone-clamped tangential ground force converts the wheel's spin (from `kCmdDriveTorque`) into
> chassis forward motion — "the car actually drives". INTEGER-bit-exact. **NO new shader** — traction is a
> pure-integer HOST pass applied identically on the GPU and CPU paths (the VH3 command-apply precedent), folded
> into a NEW additive tick `StepVehicleDriven` (VH3's `StepVehicle` stays byte-frozen). Branch: `slice-vh4`. See
> [[hazard-forge-vehicle-roadmap]].

**Goal:** Extend `engine/sim/vehicle.h` (additive — VH1/VH2/VH3 byte-unchanged) with `ApplyWheelTraction` (the
Coulomb-cone tangential ground-traction pass) + `StepVehicleDriven` (= the VH3 `StepVehicle` tick with the
traction pass inserted) + `StepVehicleDrivenSteps` + a traction stat. Add `--vehicle-traction-shot` (Vulkan) /
`--vehicle-traction` (Metal). Bake the integer golden `vehicle_traction`. **NO new shader, NO new RHI.**

## Design call: traction = a deterministic Coulomb-cone tangential ground force at each grounded wheel
The honest VH3 gap (carried from FR4/JT3): fpx contacts are sphere-sphere with NO inertia tensor, so a
ground contact can neither spin a wheel nor be spun BY a wheel — wheel-roll-drives-chassis is not emergent. VH4
supplies it as an explicit deterministic PROXY (the GR4 dry-friction / the couple_grain Coulomb-cone idiom):

For each WHEEL body that is GROUNDED (its bottom `pos.y - wheelRadius <= groundY + kContactEps`):
1. **The rolling-forward direction** `fwd` = `FxNormalize(FxCross(groundNormal, axleWorld))` where `groundNormal`
   = `(0, kOne, 0)` (the flat ground up) and `axleWorld` = the wheel's spin axis rotated into world by the
   hinge's CURRENT `axis` (so STEERED front wheels push along their re-aimed heading — VH3's `kCmdSteer` now
   actually turns the car). If `|fwd| == 0` (degenerate), skip the wheel (no-op).
2. **The no-slip target ground speed** `vTarget` = `fxmul(wheelAngVelAboutAxle, wheelRadius)` — the wheel's
   contact-patch surface speed (spin × radius), the speed the ground point would move at if rolling without slip.
   `wheelAngVelAboutAxle` = `FxDot(wheel.angVel, axleWorld)` (the signed spin about its own axle).
3. **The slip** `slip` = `vTarget - FxDot(chassis.vel, fwd)` (target minus the chassis's current forward ground
   speed). 
4. **The cone-clamped traction impulse** `j` = clamp(`fxmul(slip, kGripK)`, `-kMuMax`, `+kMuMax`) — a
   stiffness-scaled correction toward no-slip, CLAMPED to the friction cone `±kMuMax` (the Coulomb limit: a wheel
   can only transmit so much tangential force before it slips; large slip saturates). Apply `chassis.vel +=
   FxScale(fwd, fxmul(j, kChassisShare))` (the chassis accelerates toward rolling) AND bleed the wheel:
   `wheelAngVelAboutAxle` reduced by `fxmul(j, kWheelBleed)` projected back onto the axle (momentum leaves the
   spin as the tyre grips — so a freely-spinning driven wheel loses spin as it grips, and the chassis gains
   speed; deterministic, fixed op order). Constants `kGripK`/`kMuMax`/`kChassisShare`/`kWheelBleed` are
   host-fixed Q16.16 in `vehicle.h`.

`ApplyWheelTraction(v, cfg)` runs this over all four wheels in FIXED index order (Gauss-Seidel; the chassis vel
updated in place, read by later wheels — deterministic because the order is fixed). Pure integer (FxCross/FxDot/
FxNormalize/fxmul/FxScale + integer clamp), NO transcendentals.

**`StepVehicleDriven(v, cfg, commands, tick, dt, iters, solveIters)`** = VH3's `StepVehicle` body with ONE added
pass: apply tick commands → PHASE A spring → PHASE B hinge → **PHASE C `ApplyWheelTraction`** (the new beat, after
the wheels' grounded state is set by the integrate+constraints, before the contact solve) → PHASE D fpx
BuildPairs + StepWorld(dt=0) contacts. VH3's `StepVehicle` is NOT modified — `StepVehicleDriven` is a new additive
function that duplicates the VH3 phase order with PHASE C inserted (keep them in lockstep; a comment cross-refs
StepVehicle so a future reader keeps them aligned). `StepVehicleDrivenSteps` runs K ticks. `MeasureVehicleDrive`
(VH3, REUSED) reports the chassis forward displacement + driven-wheel spin; VH4 adds `tractionForward` (the
forward speed gained purely from traction) to a small new stat or reuses the existing one.

**The GPU showcase (NO new shader — traction is a HOST pass, the VH3 command precedent):** the Vulkan
`--vehicle-traction-shot` drives the EXISTING vehicle_spring_solve.comp + joint_angular_solve.comp + fpx_solve.comp
for PHASES A/B/D exactly as VH3 does, and runs `ApplyWheelTraction` HOST-side between the constraint dispatches
and the contact dispatch (the host already does work there — BuildPairs — so a host velocity adjustment is
consistent; the world is staged host→GPU per the existing "fresh initialData upload carries state forward"
pattern). The resulting GPU body world after K ticks is memcmp'd vs the CPU `StepVehicleDriven` → BIT-EXACT.
Metal `--vehicle-traction` runs the CPU `StepVehicleDriven` (byte-identical by construction).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **VH1/VH2/VH3 (this branch's `vehicle.h`, read-only — build on, DON'T modify):** `FxSpringJoint`/
  `SolveSpringJoint`, `VehicleConfig`/`struct Vehicle`/`VehicleFromConfig`, `StepVehicleRig`, the VH3
  `kCmdDriveTorque`/`kCmdSteer`/`ApplyVehicleCommand`/`QFromAxisAngleSnapped`/`kChassisUpAxis`/`kWheelSpinAxis`/
  `IsFrontHinge`, `StepVehicle`/`StepVehicleSteps`, `MeasureVehicleDrive`/`VehicleDriveState`. DO NOT modify any of
  these. VH4 APPENDS after the VH3 block.
- **fpx Q16.16 vector ops to REUSE VERBATIM (`engine/sim/fpx.h`, read-only):** `FxDot`, `FxCross` (confirm it
  exists; if the symbol differs, use the actual one), `FxNormalize`, `FxLength`, `FxScale`, `FxAdd`, `FxSub`,
  `fxmul`, `FxRotate` (to rotate the axle by the hinge axis into world). `FxBody{pos,vel,angVel,orient,radius,
  flags}`. The Coulomb-clamp idiom: a min/max integer clamp (see `grain.h` GR4 friction / `joint.h` SolveAngular
  cone clamp for the cone-clamp shape). **DO NOT modify fpx.h.**
- **The grounded test + the contact substrate:** VH2/VH3's ground handling (`groundY`, `wheelRadius`, the wheel
  `pos.y` clamp / the fpx ground in StepWorld). The wheel is grounded when its bottom is at/below `groundY +
  kContactEps`.
- **The host-pass-between-dispatches precedent:** VH3's `ApplyVehicleCommand` is applied host-side to the staged
  world; VH3's GPU showcase host-rebuilds the FPX2 pairs between dispatches. `ApplyWheelTraction` slots into the
  SAME host seam. Study the VH3 `--vehicle-step-shot` GPU decomposition in `samples/hello_triangle/main.cpp`.
- **Showcase + registration:** VH3's `--vehicle-step-shot` plumbing — **standalone arg-parse loop** (the FR1 C1061
  lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE handled by
  the controller — do NOT rebake the JSON golden**), `tests/vehicle_test.cpp`.

## Design decisions (locked)
1. **`ApplyWheelTraction(Vehicle& v, const VehicleConfig& cfg)`** — the per-wheel Coulomb-cone tangential
   traction above, FIXED wheel-index order, in-place chassis vel + wheel-spin bleed. Host-fixed Q16.16 constants
   `kContactEps`, `kGripK`, `kMuMax`, `kChassisShare`, `kWheelBleed` defined in vehicle.h (documented values).
   Pure integer. Steered front wheels push along their re-aimed heading (the axle rotated by the hinge axis).
2. **`StepVehicleDriven(v, cfg, commands, tick, dt, iters, solveIters)`** — VH3 phase order + PHASE C
   `ApplyWheelTraction` (between hinge and contacts). + **`StepVehicleDrivenSteps(...)`**. VH3 `StepVehicle`
   UNCHANGED. (Optional: a `VehicleTractionState`/extended `MeasureVehicleDrive` overload reporting the
   traction-only forward gain; keep it minimal.)
3. **Showcase `--vehicle-traction-shot <out>` (Vulkan) AND `--vehicle-traction` (Metal) — WIRE BOTH** (standalone
   arg-parse). Build a car, feed a drive stream (`kCmdDriveTorque` on the driven wheels over K ticks) with **NO
   chassis velocity seed** (the VH3 seed is GONE — forward must come from TRACTION). Run K `StepVehicleDriven`
   ticks → the spun wheels grip the ground and the car drives forward. Vulkan: the GPU A/B/D dispatches + the host
   PHASE-C traction → **memcmp vs the CPU `StepVehicleDriven`**. Metal: the CPU reference. Render the 2D side view
   (chassis box + wheels + springs + ground line + the driven path), the VH3 render REUSED. Golden =
   `tests/golden/metal/vehicle_traction.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU vehicle world after K driven ticks == the CPU `StepVehicleDriven`
     byte-for-byte. Print `vehicle-traction: {bodies:<N>, commands:<C>, ticks:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `vehicle-traction determinism: two runs BYTE-IDENTICAL`.
   - **(3) traction drives the car (NO seed):** under the drive stream WITH NO velocity seed, the chassis forward
     displacement > a band AND it is produced by traction (the driven wheels spin AND the chassis gained forward
     speed). Print `vehicle-traction drive: {forward:<F>, wheelSpin:<W>, fromTraction:true}`; assert forward > band.
   - **(4) the friction cone bounds it / a control:** a `kMuMax = 0` control (no grip) leaves the car NOT driving
     forward despite the wheels spinning (slip saturates to zero force) — proving the traction (not a seed) does
     the work. Print `vehicle-traction control: {noGrip:idle, wheelsStillSpin:true}`. (And/or: a free-spin wheel
     with grip loses spin as the chassis gains speed — momentum transfer.)
   - **Golden discipline: ONLY `tests/golden/metal/vehicle_traction.png`; do NOT commit it.** Existing 168 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
6. **Tests `tests/vehicle_test.cpp` additions (pure CPU):** `ApplyWheelTraction` — a grounded wheel spinning
   forward accelerates the chassis along `fwd`; a non-grounded wheel (chassis lifted) contributes no traction; a
   `kMuMax = 0` config produces zero chassis acceleration (cone saturates); a steered front wheel pushes along its
   re-aimed heading (the chassis gains a lateral component). `StepVehicleDriven` — a drive stream with NO seed
   moves the chassis forward (traction-driven); a no-grip control stays put; two runs byte-identical. Clean under
   `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-vehicle-traction` (features) + `--vehicle-traction-shot`
   (showcases) in `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON
   golden — the controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `engine/anim/` +
  `engine/physics/` + all existing shaders UNCHANGED. VH1/VH2/VH3 `vehicle.h` code + `vehicle_spring_solve.comp` +
  the VH2/VH3 showcases UNCHANGED (VH4 additive — only the traction pass + the driven tick + the showcase). **NO
  new shader** (traction is a host pass; A/B/D drive VH1's spring + JT2's angular + fpx's contact shaders).
  Report the seam empty.

## Out of scope (YAGNI — later VH slices)
Lockstep/rollback (VH5), the lit render (VH6). Real Pacejka/Magic-Formula tyre slip curves, longitudinal vs
lateral slip separation, load-dependent grip, a differential, ABS/TCS, rolling resistance, aero. VH4 claims ONLY:
a deterministic wheel-ground TRACTION proxy (a Coulomb-cone tangential ground force converting wheel spin → chassis
motion, steered wheels turning the car) that drives the car forward WITHOUT the VH3 velocity seed, bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE (honest, the GR4/JT2 caveat): traction is a
deterministic tangential PROXY (a slip-proportional cone-clamped force), NOT analytic tyre mechanics; the headline
is DETERMINISM + cross-platform bit-identity + "it drives from spin, not a seed", NOT physical tyre fidelity. The
chassis has no inertia tensor (fpx limitation) so the car does not pitch/roll under traction — planar drive only.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 102) + the new `vehicle_test` traction cases. Clean under
   `windows-msvc-asan` (build+run `vehicle_test` + `introspect_test`).
2. **proofs + visual:** `--vehicle-traction-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image
   shows the car driven forward by traction (pixel-check; the VH1 lesson).** ALSO re-run `--vehicle-step-shot` +
   `--vehicle-rig-shot` + `--vehicle-spring-shot` → still bit-exact (VH1/VH2/VH3 render-invariance).
3. Metal: `visual_test --vehicle-traction` → new golden `tests/golden/metal/vehicle_traction.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (VH4
   reuses VH1 + JT2 + fpx shaders — `hf_gen_msl` UNCHANGED).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `vehicle_traction.png` added; the
   other 168 byte-identical. `git diff master --stat -- tests/golden` = ONLY `vehicle_traction.png` (metal) (the
   introspect json is rebaked by the controller post-merge-gate).
5. Introspect: exactly `+deterministic-vehicle-traction` + `--vehicle-traction-shot` added in introspect.cpp;
   introspect test updated. (JSON golden rebake = controller.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h` + `engine/anim/` + `engine/physics/` + VH1/VH2/VH3 `vehicle.h` (the pre-VH4 functions) /
   `vehicle_spring_solve.comp` byte-unchanged). `scripts/verify.ps1` updated: `vehicle_traction` golden in the Mac
   loop + `--vehicle-traction-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`.**
