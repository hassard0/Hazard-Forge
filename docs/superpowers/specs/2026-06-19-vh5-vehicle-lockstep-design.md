# Slice VH5 — Deterministic Vehicle Physics: LOCKSTEP + ROLLBACK (THE NETCODE HEADLINE) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #16 (DETERMINISTIC
> VEHICLE PHYSICS, `hf::sim::vehicle`) — the NETCODE HEADLINE, the FPX5/FR5/GR5/CG5/JT5 twin. VH1-VH4 built a
> drivable car (suspension → rig → drive/steer → traction). VH5 proves the bit-exact driven tick is true
> cross-platform LOCKSTEP + ROLLBACK: two peers fed ONLY the input command stream (NOT full state) re-derive the
> authority's exact car trajectory bit-for-bit, and a mispredicted input is corrected by rolling back to a saved
> snapshot + re-simulating. **PURE CPU** (NO GPU dispatch, NO new shader, NO new RHI) — both Vulkan-Windows
> (`--vehicle-lockstep-shot`) and Metal-Mac (`--vehicle-lockstep`) run the IDENTICAL CPU harness, so the
> converged-car golden is bit-identical cross-backend BY CONSTRUCTION (that cross-platform bit-identity IS the
> lockstep evidence). Branch: `slice-vh5`. See [[hazard-forge-vehicle-roadmap]].

**Goal:** Extend `engine/sim/vehicle.h` (additive — VH1-VH4 byte-unchanged) with a `VehicleSnapshot` + `SnapshotVehicle`/
`RestoreVehicle` (the VH twist: snapshot the body world **AND the mutable hinge axes** — `kCmdSteer` rotates them,
so they are part of the replayable state) + `SimVehicleTick` (apply a tick's commands already happens inside
`StepVehicleDriven`; SimVehicleTick = one `StepVehicleDriven` step) + `RunVehicleLockstep` + `RunVehicleRollback`.
Add `--vehicle-lockstep-shot` (Vulkan) / `--vehicle-lockstep` (Metal). Bake the integer golden `vehicle_lockstep`.
**NO GPU, NO new shader, NO new RHI.**

## Design call: the FPX5 command/snapshot machinery, reused, with the hinge axes added to the snapshot
The deterministic-sim flagships all share one netcode harness shape (fpx FPX5 → fract FR5 → grain GR5 → cgrain CG5
→ cgf GF5 → joint JT5): a pure-CPU loop that (a) runs an authority + a replica fed the SAME command stream and
asserts they stay BIT-IDENTICAL, and (b) snapshots, mispredicts, rolls back to the snapshot, re-simulates, and
asserts the corrected state == authority. VH5 is the JT5 twin with `StepVehicleDriven` as the per-tick step.

**THE ONE VH-SPECIFIC TWIST — the snapshot must include the hinge axes.** The ragdoll JT5 snapshot was just the
`fpx::FxWorld` (bodies). But a vehicle's `kCmdSteer` MUTATES `hinges[i].axis` (the steered heading), and
`ApplyWheelTraction` reads those axes — so the hinge axes are LIVE replayable state. `SnapshotVehicle` therefore
deep-copies BOTH the body world (`world.bodies`) AND the four `hinges[i].axis` (the steered state). `RestoreVehicle`
restores both. (The springs are immutable — `restLen`/`stiffness`/`damping` never change — so they need not be
snapshotted; capture only what mutates: bodies + hinge axes. The chassis/wheel indices are structural constants.)

`SimVehicleTick(v, cfg, commands, tick, dt, iters, solveIters)` = one `StepVehicleDriven` step (which already
applies `cmd.tick == tick` commands in array order at its prologue — the VH3/VH4 contract). 

`RunVehicleLockstep(cfg, initialVehicle, commands, ticks, dt, iters, solveIters)`: build an `authority` + a
`replica` from the SAME initial vehicle, step BOTH with `SimVehicleTick` over the SAME command stream for `ticks`
ticks, and assert at every tick the two are bit-identical (memcmp the bodies + the hinge axes). Returns the
converged authority (+ a bool all-matched). 

`RunVehicleRollback(cfg, initialVehicle, authorityCommands, mispredictCommands, snapshotTick, divergeTick, ...)`:
run the authority over `authorityCommands`; run a peer that MISPREDICTS (uses `mispredictCommands` — a wrong steer/
drive at `divergeTick`) until it notices, then ROLLS BACK to a `VehicleSnapshot` taken at `snapshotTick` and
re-simulates with the corrected `authorityCommands` → assert the corrected peer == authority bit-for-bit, and that
the mispredicted (pre-rollback) state HAD diverged (a real divergence was fixed).

**The showcase is PURE CPU.** Both `--vehicle-lockstep-shot` (Vulkan-Windows) and `--vehicle-lockstep` (Metal-Mac)
run the IDENTICAL `RunVehicleLockstep`/`RunVehicleRollback` C++ (NO GPU dispatch) and render the converged car via
the VH3/VH4 2D side-view path REUSED VERBATIM. The golden is bit-identical cross-backend BY CONSTRUCTION. There is
NO GPU==CPU memcmp here (it is all CPU) — the proof is authority==replica + rollback==authority + the
cross-platform bit-identical golden.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **VH1-VH4 (this branch's `vehicle.h`, read-only — build on, DON'T modify):** `struct Vehicle` (world + springs +
  hinges + chassisIndex + wheelIndex), `VehicleConfig`, `VehicleFromConfig`, the VH3 `kCmdDriveTorque`/`kCmdSteer`/
  `ApplyVehicleCommand`, `StepVehicleDriven`/`StepVehicleDrivenSteps` (VH4 — the per-tick step VH5 wraps),
  `MeasureVehicleDrive`. DO NOT modify any of these. VH5 APPENDS after the VH4 block.
- **fpx FPX5 command/snapshot machinery to MIRROR (`engine/sim/fpx.h`, read-only):** `FxCommand{tick,kind,bodyId,
  arg}`, `SnapshotWorld`/`RestoreWorld` (or the actual fpx snapshot symbols — confirm the names), the `FxWorld`/
  `FxBody` shape. **DO NOT modify fpx.h.** (VH5 may reuse `fpx::SnapshotWorld` for the body half and add the hinge
  axes alongside, OR roll its own small deep-copy — whichever keeps fpx.h frozen; prefer reuse.)
- **The lockstep/rollback harness to TWIN (`engine/sim/joint.h` JT5 — `RunRagdollLockstep`/`RunRagdollRollback`/
  `SimRagdollTick`; ALSO `fract.h` FR5, `fpx.h` FPX5):** the authority/replica loop + the snapshot/mispredict/
  rollback structure. VH5's three functions are the verbatim shape with `SimVehicleTick`/`StepVehicleDriven`
  substituted and the snapshot extended to the hinge axes. Read JT5 closely — match its proof structure.
- **Showcase + registration:** the VH4 `--vehicle-traction-shot` plumbing — **standalone arg-parse loop** (the FR1
  C1061 lesson) — but PURE CPU (no GPU dispatch path). `scripts/verify.ps1`, `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT**), `tests/vehicle_test.cpp`.

## Design decisions (locked)
1. **`struct VehicleSnapshot { std::vector<fpx::FxBody> bodies; FxVec3 hingeAxis[4]; }`** (or a small struct
   capturing exactly the mutable state). **`SnapshotVehicle(const Vehicle&) -> VehicleSnapshot`** (deep-copy
   bodies + the 4 hinge axes). **`RestoreVehicle(Vehicle&, const VehicleSnapshot&)`** (restore both). Bit-exact
   round-trip.
2. **`SimVehicleTick(v, cfg, commands, tick, dt, iters, solveIters)`** = one `StepVehicleDriven` step. **
   `RunVehicleLockstep(...)`** (authority + replica over the same stream, bit-identical every tick — memcmp bodies
   + hinge axes). **`RunVehicleRollback(...)`** (snapshot → mispredict diverges → rollback → corrected ==
   authority). Mirror the JT5 signatures/returns. Pure integer/host, fixed op order.
3. **Showcase `--vehicle-lockstep-shot <out>` (Vulkan-Windows) AND `--vehicle-lockstep` (Metal-Mac) — WIRE BOTH**
   (standalone arg-parse), PURE CPU. Build a car, a scripted drive+steer `authStream` (so the car drives a
   non-trivial path), run `RunVehicleLockstep` (authority==replica) + `RunVehicleRollback` (a mispredicted steer
   at one tick, rolled back). Render the converged car via the VH3/VH4 2D side-view path REUSED VERBATIM. Golden =
   `tests/golden/metal/vehicle_lockstep.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) authority==replica:** two peers fed only the input stream stay bit-identical for all ticks. Print
     `vehicle-lockstep: {bodies:<N>, ticks:<T>} authority==replica BIT-IDENTICAL`.
   - **(2) rollback==authority:** the corrected (rolled-back + re-simulated) peer == authority byte-for-byte.
     Print `vehicle-lockstep rollback: corrected==authority BIT-EXACT`.
   - **(3) mispredict diverged:** the mispredicted (pre-rollback) state HAD diverged from authority (a real
     divergence was fixed, not a no-op). Print `vehicle-lockstep mispredict: diverged before rollback (real
     divergence fixed)`.
   - **(4) determinism:** two full runs → identical. Print `vehicle-lockstep determinism: two runs BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/vehicle_lockstep.png`; do NOT commit it.** Existing 169 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan-Windows == Metal-Mac == golden, ZERO differing pixels (the CPU
   harness is identical on both, so the rendered converged car is bit-identical by construction).
6. **Tests `tests/vehicle_test.cpp` additions (pure CPU):** `SnapshotVehicle`/`RestoreVehicle` round-trip
   (snapshot → mutate (drive+steer a few ticks) → restore → bit-identical to the snapshot, INCLUDING the hinge
   axes). `RunVehicleLockstep` — authority==replica over a drive+steer stream. `RunVehicleRollback` — a
   mispredicted steer diverges, rollback corrects to authority; the snapshot/restore preserves the steered hinge
   state. Two runs byte-identical. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-vehicle-lockstep` (features) + `--vehicle-lockstep-shot`
   (showcases) in `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON
   golden — the controller does that.**

## RHI seam additions (summary)
- **None.** PURE CPU — NO GPU dispatch, NO compute, NO new shader, NO new RHI. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `engine/anim/` +
  `engine/physics/` + all existing shaders + `hf_gen_msl` UNCHANGED. VH1-VH4 `vehicle.h` code + the VH2/VH3/VH4
  showcases UNCHANGED (VH5 additive — only the snapshot + the harness + the showcase). Report the seam empty.

## Out of scope (YAGNI — later VH slices)
The lit render (VH6). A network transport (VH5 proves the lockstep MATH — inputs-only re-derivation + rollback —
not a socket layer; the GR5/JT5 precedent). Input delay/prediction-window tuning, partial-state delta
compression, multi-vehicle lockstep beyond the showcase's one car, further structural change mid-replay. VH5
claims ONLY: the bit-exact driven vehicle tick is deterministic LOCKSTEP + ROLLBACK (two peers re-derive the car
from inputs alone; a mispredict rolls back to a snapshot — INCLUDING the steered hinge state — and corrects),
bit-identical CPU↔Vulkan-Windows↔Metal-Mac, with the integer golden + the four proofs. This is the cross-platform
rollback-replayable VEHICLE UE5's float Chaos Vehicles cannot do.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 102) + the new `vehicle_test` lockstep/rollback/snapshot
   cases. Clean under `windows-msvc-asan` (build+run `vehicle_test` + `introspect_test`).
2. **proofs + visual:** `--vehicle-lockstep-shot` on Vulkan-Windows: the 4 proofs + exit 0, under the
   Vulkan-validation gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED —
   even though it is CPU, the showcase still spins up the device/swapchain for the render). **VERIFY the image
   shows the converged driven car (pixel-check; the VH1 lesson).** ALSO re-run `--vehicle-traction-shot` +
   `--vehicle-step-shot` + `--vehicle-rig-shot` + `--vehicle-spring-shot` → still bit-exact (VH1-VH4
   render-invariance).
3. Metal-Mac: `visual_test --vehicle-lockstep` → new golden `tests/golden/metal/vehicle_lockstep.png`; two runs
   DIFF 0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader
   (`hf_gen_msl` UNCHANGED).** Cross-vendor STRICT ZERO (the CPU harness is identical on both backends).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `vehicle_lockstep.png` added; the
   other 169 byte-identical. `git diff master --stat -- tests/golden` = ONLY `vehicle_lockstep.png` (metal) (the
   introspect json rebake = controller, post-gate).
5. Introspect: exactly `+deterministic-vehicle-lockstep` + `--vehicle-lockstep-shot` added in introspect.cpp;
   introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h` + `engine/anim/` + `engine/physics/` + VH1-VH4 `vehicle.h` (the pre-VH5 functions) byte-unchanged).
   `scripts/verify.ps1` updated: `vehicle_lockstep` golden in the Mac loop + `--vehicle-lockstep-shot` in
   `$vkShots`. **NO new entry in `hf_gen_msl`.**
