# Slice AC5 — Deterministic Active Ragdoll: LOCKSTEP + ROLLBACK (THE NETCODE HEADLINE) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #17 (DETERMINISTIC
> ACTIVE RAGDOLL / PHYSICAL-ANIMATION BLENDING, `hf::sim::active`). AC1-AC4 built a clip-tracking, blendable,
> hit-reacting active ragdoll. AC5 proves it is true cross-platform LOCKSTEP + ROLLBACK: two peers fed ONLY the
> input HIT stream re-derive the exact clip-driven ragdoll trajectory bit-for-bit, and a mispredicted hit is
> corrected by rolling back to a saved snapshot + re-simulating — the FPX5/FR5/GR5/CG5/GF5/JT5/VH5 twin.
> **PURE CPU** (NO GPU dispatch, NO new shader, NO new RHI) — both Vulkan-Windows (`--active-lockstep-shot`) and
> Metal-Mac (`--active-lockstep`) run the IDENTICAL CPU harness, so the converged golden is bit-identical
> cross-backend BY CONSTRUCTION (that cross-platform bit-identity IS the lockstep evidence). Branch:
> `slice-ac5`. See [[hazard-forge-active-roadmap]].

**Goal:** Extend `engine/sim/active.h` (additive — AC1-AC4 byte-frozen) with `ActiveCommand` (a hit event) +
`ActiveSnapshot` + `SnapshotActive`/`RestoreActive` (the VH5 twist: snapshot the body world AND the `tick` — the
clip-time/physicality anchor) + `SimActiveTick` (apply this tick's hits + `StepActive`) + `RunActiveLockstep` +
`RunActiveRollback`. Add `--active-lockstep-shot` (Vulkan) / `--active-lockstep` (Metal). Bake the integer golden
`active_lockstep`. **NO GPU, NO new shader, NO new RHI.**

## Design call: the FPX5 harness over StepActive, with the tick (the clip-time anchor) in the snapshot
The deterministic-sim flagships all share one netcode harness shape: a pure-CPU loop that (a) runs an authority + a
replica fed the SAME command stream and asserts they stay BIT-IDENTICAL, and (b) snapshots, mispredicts, rolls
back, re-simulates, and asserts the corrected state == authority. AC5 is that harness over the active ragdoll.

**THE INPUT is a HIT stream.** `ActiveCommand { uint32 tick; uint32 body; FxVec3 dv; }` — a velocity-impulse event
(the AC4 `ApplyImpulse`, now an INPUT rather than a scripted constant). The clip-drive (AC3 `StepActive`) holds the
ragdoll on the animation; the hits perturb it; the drive recovers it. (AC5 uses full physicality / `StepActive`
clip-tracking; the AC4 physicality ramp is NOT replayed here — the lockstep proof is over the hit stream + the
deterministic clip-drive. A hit is the only nondeterminism-free "input event".)

**THE VH5 TWIST — snapshot the world AND the tick.** AC3's `StepActive` recomputes the qTargets each tick from the
clip at `time = startTime + tick*dt` (a host pre-pass), and the drives' base weights are constant — so the only
MUTABLE replayable state is the `fpx::FxWorld` (bodies). BUT the clip `time` is derived from the TICK index, so a
snapshot must record WHICH tick it was taken at, or the resumed sim samples the clip at the wrong time. Therefore
`ActiveSnapshot { fpx::FxWorld world; int tick; }` (the VH5 hinge-axes / GF5 two-pool analog: snapshot the world +
the per-tick anchor). `SnapshotActive` deep-copies the world (via `fpx::SnapshotWorld`) + stores the tick;
`RestoreActive` restores the world + returns the tick so the harness resumes the clip-time from there.

`SimActiveTick(active, skeleton, clip, commands, tick, startTime, dt, iters)`: for each `cmd` with `cmd.tick ==
tick` (ARRAY ORDER), `ApplyImpulse(active.ragdoll.world, cmd.body, cmd.dv)`; then `StepActive(active, skeleton,
clip, startTime + tick*dtSeconds, dt, iters)`.

`RunActiveLockstep(skeleton, clip, initialActive, commands, ticks, startTime, dt, iters)`: build an `authority` + a
`replica` from the SAME initial active ragdoll, step BOTH with `SimActiveTick` over the SAME hit stream for `ticks`
ticks, assert bit-identical every tick (memcmp the world bodies). Returns the converged authority.

`RunActiveRollback(skeleton, clip, initialActive, authorityCmds, mispredictCmds, divergeTick, ticks, ...)`: advance
to `divegeTick` with `authorityCmds`; `SnapshotActive` (world + tick); speculatively advance a few ticks with
`mispredictCmds` (a WRONG hit — different body/dv/tick); `RestoreActive` to the snapshot (world + tick) +
re-simulate `divergeTick..ticks` with `authorityCmds`; assert the corrected peer == authority bit-for-bit AND the
mispredicted (pre-rollback) state HAD diverged.

**The showcase is PURE CPU** (both backends run the identical harness, the converged ragdoll rendered via the AC1
2D side-view; bit-identical cross-backend by construction; NO GPU==CPU memcmp — the proof is authority==replica +
rollback==authority + the cross-platform golden).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **AC1-AC4 (this branch's `active.h`, read-only — build on, DON'T modify):** `ActiveRagdoll`, `ActiveFromSkeleton`,
  `StepActive` (the per-tick clip-drive), `ApplyImpulse` (AC4 — the hit), `WriteClipTargets`, `DriveAngleCos`. DO
  NOT modify the AC1-AC4 functions. AC5 APPENDS.
- **fpx snapshot machinery to MIRROR (`engine/sim/fpx.h`, read-only):** `SnapshotWorld`/`RestoreWorld` (the body
  deep-copy/restore — REUSE for the world half), `FxWorld`/`FxBody`, `FxVec3`. **DO NOT modify fpx.h.**
- **The lockstep/rollback harness to TWIN (`engine/sim/vehicle.h` VH5 — `SnapshotVehicle`/`RestoreVehicle`/
  `SimVehicleTick`/`RunVehicleLockstep`/`RunVehicleRollback`; ALSO `joint.h` JT5, `fpx.h` FPX5):** the authority/
  replica loop + the snapshot/mispredict/rollback structure + the "snapshot the mutable extra" (VH5 hinge axes →
  AC5 tick). VH5's `VehicleSnapshot{bodies; hingeAxis[4]}` is the exact precedent — AC5's `ActiveSnapshot{world;
  tick}`. Match VH5's signatures/returns + proof structure.
- **Showcase + registration:** the AC4 `--active-recover-shot` plumbing — **standalone arg-parse loop** — but PURE
  CPU (no GPU dispatch path). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp`
  (**controller rebakes the JSON golden — do NOT**), `tests/active_test.cpp`.

## Design decisions (locked)
1. **`struct ActiveCommand { uint32_t tick; uint32_t body; FxVec3 dv; }`** + **`struct ActiveSnapshot {
   fpx::FxWorld world; int tick; }`** + **`SnapshotActive(active, tick)->ActiveSnapshot`** (deep-copy world via
   `fpx::SnapshotWorld` + store tick) + **`RestoreActive(active, snap)->int`** (restore world via
   `fpx::RestoreWorld`, return the tick) + **`SimActiveTick`** (apply tick's hits + `StepActive`) +
   **`RunActiveLockstep`** + **`RunActiveRollback`** (above). Mirror VH5. Pure integer-sim + the deterministic
   host clip pre-pass.
2. **Showcase `--active-lockstep-shot <out>` (Vulkan-Windows) AND `--active-lockstep` (Metal-Mac) — WIRE BOTH**
   (standalone arg-parse), PURE CPU. The AC3 humanoid + bend clip, a scripted hit `authStream` (a couple of
   impulses on the torso/limbs at a few ticks — so the ragdoll is perturbed off the clip then driven back), run
   `RunActiveLockstep` (authority==replica) + `RunActiveRollback` (a mispredicted hit at one tick, rolled back).
   Render the converged ragdoll via the AC1 2D side-view REUSED. Golden = `tests/golden/metal/active_lockstep.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) authority==replica:** two peers fed only the hit stream stay bit-identical for all ticks. Print
     `active-lockstep: {bones:<N>, ticks:<T>, hits:<H>} authority==replica BIT-IDENTICAL`.
   - **(2) rollback==authority:** the corrected (rolled-back + re-simulated) peer == authority byte-for-byte. Print
     `active-lockstep rollback: corrected==authority BIT-EXACT`.
   - **(3) mispredict diverged:** the mispredicted (pre-rollback) state HAD diverged from authority. Print
     `active-lockstep mispredict: diverged before rollback (real divergence fixed)`.
   - **(4) determinism:** two full runs → identical. Print `active-lockstep determinism: two runs BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/active_lockstep.png`; do NOT commit it.** Existing 175 image
     goldens UNTOUCHED (incl AC1-AC4).
4. **Cross-backend bar (INTEGER, strict):** Vulkan-Windows == Metal-Mac == golden, ZERO differing pixels (the CPU
   harness is identical on both → the rendered converged ragdoll is bit-identical by construction).
5. **Tests `tests/active_test.cpp` additions (pure CPU):** `SnapshotActive`/`RestoreActive` round-trip (snapshot →
   mutate (a few hit+step ticks) → restore → world bit-identical to the snapshot AND the returned tick matches);
   `RunActiveLockstep` authority==replica over a hit stream; `RunActiveRollback` a mispredicted hit diverges,
   rollback corrects to authority. Two runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-active-lockstep` (features) + `--active-lockstep-shot` (showcases)
   in `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** PURE CPU — NO GPU dispatch, NO compute, NO new shader, NO new RHI. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `vehicle.h` +
  `engine/anim/` + `engine/physics/` + ALL shaders + `hf_gen_msl` UNCHANGED. AC1-AC4 `active.h` functions UNCHANGED
  (AC5 additive — only the command + snapshot + harness + showcase). Report the seam empty.

## Out of scope (YAGNI — AC6 only remains)
The lit skinned render (AC6 — AC5's render is the AC1 2D diagnostic; AC6 swaps in `PoseToPalette` → `lit_skinned`).
A network transport (AC5 proves the lockstep MATH — inputs-only re-derivation + rollback — not a socket layer, the
GR5/JT5/VH5 precedent). Replaying the AC4 physicality ramp in lockstep (AC5 uses full clip-drive + hit inputs;
the physicality/limp episode is AC4's). AC5 claims ONLY: the bit-exact clip-driven active ragdoll tick is
deterministic LOCKSTEP + ROLLBACK (two peers re-derive it from the hit stream alone; a mispredict rolls back to a
snapshot — world + tick — and corrects), bit-identical CPU↔Vulkan-Windows↔Metal-Mac, with the integer golden + the
four proofs. This is the cross-platform rollback-replayable physical-animation UE5's float Chaos cannot do.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 103 + the new `active_test` AC5 cases). Clean under
   `windows-msvc-asan` (build+run `active_test` + `introspect_test`).
2. **proofs + visual:** `--active-lockstep-shot` on Vulkan-Windows: the 4 proofs + exit 0, under the
   Vulkan-validation gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED —
   the showcase still spins up the device for the render). **VERIFY the image shows the converged clip-driven
   ragdoll (a coherent posed skeleton, not scrambled).** Re-run `--active-recover-shot` + `--active-step-shot` +
   `--active-blend-shot` + `--active-drive-shot` → still bit-exact AND their goldens byte-identical (AC1-AC4
   render-invariance).
3. Metal-Mac: `visual_test --active-lockstep` → new golden `tests/golden/metal/active_lockstep.png`; two runs DIFF
   0.0000. **Confirm `visual_test.mm` in the diff; confirm NO new shader (`hf_gen_msl` UNCHANGED; `active`
   absent).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `active_lockstep.png` added; the
   other 175 byte-identical. `git diff master --stat -- tests/golden` = ONLY `active_lockstep.png` (metal) + the
   introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-active-lockstep` + `--active-lockstep-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + the frozen sim/anim/physics headers + ALL shaders byte-unchanged; ONLY `active.h`
   extended additively + the showcase/test/introspect). `scripts/verify.ps1` updated: `active_lockstep` golden +
   `--active-lockstep-shot` in `$vkShots`. **NO new shader; `active` still NOT in `hf_gen_msl`.**
