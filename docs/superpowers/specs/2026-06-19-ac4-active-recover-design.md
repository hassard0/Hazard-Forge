# Slice AC4 — Deterministic Active Ragdoll: ACTIVE → LIMP → RECOVER (THE HEADLINE BEHAVIOR) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #17 (DETERMINISTIC
> ACTIVE RAGDOLL / PHYSICAL-ANIMATION BLENDING, `hf::sim::active`). AC1-AC3 built the drive, the per-joint blend,
> and the anim-clip tracking. AC4 is **the money behavior**: a global `physicality ∈ [0, kOne]` knob that scales
> every drive — the character **animates** (physicality kOne, tracks the clip), **gets hit** (an impulse + a
> physicality drop → it goes LIMP, ragdoll wins), then the drives **recover** (physicality ramps back → it returns
> to the clip pose). The classic hit-reaction / stagger / get-up. INTEGER-bit-exact. **NO new shader, NO new RHI.**
> Branch: `slice-ac4`. See [[hazard-forge-active-roadmap]].

**Goal:** Extend `engine/sim/active.h` (additive — AC1-AC3 byte-frozen) with `ApplyImpulse` (a host body-velocity
kick) + `StepActivePhysicality` (the AC3 `StepActive` with a global physicality scaling every drive's weight into a
scratch set — NO persistent mutation) + `StepActiveRecover` (the scripted anim→struck→recover episode driver) +
`PhysicalityAtTick` (the deterministic recovery ramp). Add `--active-recover-shot` (Vulkan) / `--active-recover`
(Metal). Bake the integer golden `active_recover`. **NO new shader, NO new RHI.**

## Design call: physicality scales the drive weights into a scratch set; an impulse + a deterministic ramp
`physicality ∈ [0, kOne]` is the global "how anim-driven is the character right now" alpha (UE5's physical-blend
master alpha). It scales EVERY drive's effective weight: `effectiveWeight[e] = fxmul(active.drives[e].driveWeight,
physicality)`. At `physicality = kOne` → the AC3 drives unchanged (tracks the clip); at `physicality = 0` → all
weights 0 → no correction → pure physics (limp ragdoll). To keep the AC1-AC3 records/functions byte-frozen and
avoid compounding mutation, the scaling builds a SCRATCH `std::vector<FxAngularDrive>` copy each tick (the base
drives are never overwritten):

**`StepActivePhysicality(active, skeleton, clip, time, physicality, dt, iters)`:** `WriteClipTargets(active,
skeleton, clip, time)` (AC3, fills the base qTargets) → build `scratch = active.drives` with `scratch[e].
driveWeight = fxmul(active.drives[e].driveWeight, physicality)` → `StepDriveWorld(active.ragdoll.world,
active.ragdoll.joints, active.ragdoll.limits, scratch, dt, iters)`. At `physicality = kOne`, `fxmul(w, kOne) == w`
→ byte-identical to AC3 `StepActive` (the equivalence contract).

**`ApplyImpulse(world, bodyIndex, FxVec3 dv)`:** `if dynamic and in-range: world.bodies[bodyIndex].vel =
FxAdd(vel, dv)` (a host velocity kick — the hit; the fpx/command idiom, fpx.h frozen, this is a host mutation of
the world state). Deterministic integer add.

**`PhysicalityAtTick(tick, struckTick, limpTicks, recoverTicks)`:** the deterministic recovery ramp — `kOne` before
`struckTick` (active/anim), `0` for `limpTicks` ticks after the hit (limp), then a LINEAR integer ramp `0 → kOne`
over `recoverTicks` ticks (recover), then `kOne` (re-tracked). Pure integer (a clamped linear interpolation, no
float) → bit-reproducible.

**`StepActiveRecover(active, skeleton, clip, dt, iters, struckTick, impulseBody, impulseDv, limpTicks, recoverTicks,
totalTicks, startTime)`:** the episode driver — for each tick `t`: `physicality = PhysicalityAtTick(t, ...)`; at
`t == struckTick` call `ApplyImpulse(world, impulseBody, impulseDv)` (the hit); then `StepActivePhysicality(active,
skeleton, clip, startTime + t*dt, physicality, dt, iters)`. Deterministic, fixed order. The GPU showcase runs the
SAME host per-tick logic (compute physicality, apply the impulse at the struck tick, build the scratch drives,
upload, dispatch the AC1 `active_drive_solve.comp` for one step), memcmp'ing the GPU body world vs the CPU
`StepActiveRecover` → BIT-EXACT (the physicality scale + the impulse are host-shared integer ops; only the integer
`StepDriveWorld` is the memcmp — the AC3 contract).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **AC1-AC3 (this branch's `active.h`, read-only — build on, DON'T modify):** `FxAngularDrive{...driveWeight}`,
  `SolveAngularDrive`, `StepDriveWorld`, `ActiveRagdoll`, `ActiveFromSkeleton`, `WriteClipTargets`, `StepActive`,
  `FxQuatFromFloat`, `DriveAngleCos`. DO NOT modify the AC1-AC3 functions. AC4 APPENDS.
- **fpx vector ops (`engine/sim/fpx.h`, read-only):** `FxVec3`, `FxAdd`, `FxBody{vel, flags}`, `kFlagDynamic`,
  `fxmul`, `kOne`. **DO NOT modify fpx.h.**
- **The episode/impulse precedent:** the JT5/VH5/FR5 lockstep harnesses apply scripted impulses (`FxCommand`/
  vel kicks) per tick; AC4's `ApplyImpulse` + `StepActiveRecover` are the same shape (host per-tick mutation +
  step). The AC3 `StepActive`/`StepActiveSteps` is the per-tick mold.
- **Showcase + registration:** the AC3 `--active-step-shot` plumbing (the host per-tick sample+snap+upload +
  dispatch; the 2D ragdoll render) — copy for `--active-recover-shot` / `--active-recover` (standalone arg-parse).
  `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON
  golden — do NOT**), `tests/active_test.cpp`.

## Design decisions (locked)
1. **`StepActivePhysicality`** (the scratch-scaled step), **`ApplyImpulse`** (the host vel kick),
   **`PhysicalityAtTick`** (the integer recovery ramp), **`StepActiveRecover`** (the episode driver) — all above.
   Pure: the sim step is integer/fixed-order; the physicality + impulse are deterministic integer host ops.
2. **Showcase `--active-recover-shot <out>` (Vulkan) AND `--active-recover` (Metal) — WIRE BOTH** (standalone
   arg-parse). The SCENE: the AC3 humanoid + bend clip, run `StepActiveRecover` over 3 phases — (anim:
   physicality kOne, tracks the clip) → (struck: an impulse on the torso at `struckTick` + physicality 0 for
   `limpTicks` → the ragdoll flies/goes limp) → (recover: physicality ramps 0→kOne over `recoverTicks` → returns
   to the clip pose). **CAPTURE 3 frames** (anim, struck, recovered) — render them side-by-side in ONE golden
   image (a triptych) OR pick the most telling single frame (the recovered pose with a faint anim+struck overlay);
   choose the clearer and document it. Vulkan: the host per-tick logic + the GPU dispatch → **memcmp vs the CPU
   StepActiveRecover** (at the final tick; optionally memcmp the 3 captured states). Metal: the CPU reference.
   Golden = `tests/golden/metal/active_recover.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU body world after the episode == the CPU `StepActiveRecover` byte-for-byte.
     Print `active-recover: {bones:<N>, drives:<D>, ticks:<K>, struckTick:<S>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `active-recover determinism: two runs BYTE-IDENTICAL`.
   - **(3) the three phases:** the ANIM pose tracks the clip (`DriveAngleCos` high) → the STRUCK pose is LIMP +
     displaced (the impulse moved the torso AND `DriveAngleCos` dropped — it left the clip pose) → the RECOVERED
     pose returns toward the clip (`DriveAngleCos` recovered near the anim level). Print `active-recover phases:
     {animCos:<A>, struckCos:<B>, recoverCos:<C>}` with assertions `A high, B < A (went limp), C > B (recovered)`.
   - **(4) equivalence:** a `physicality = kOne`-throughout, no-impulse run == the AC3 `StepActive` episode
     byte-for-byte (the render-invariance / AC3-equivalence proof). Print `active-recover equiv:
     {fullPhysicality==AC3:true}`.
   - **Golden discipline: ONLY `tests/golden/metal/active_recover.png`; do NOT commit it.** Existing 174 image
     goldens UNTOUCHED (incl AC1/AC2/AC3).
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests `tests/active_test.cpp` additions (pure CPU):** `PhysicalityAtTick` returns kOne pre-struck, 0 in the
   limp window, ramps 0→kOne over recover, kOne after (exact integer values at the boundaries); `ApplyImpulse`
   raises the target body's vel (static/out-of-range no-op); `StepActivePhysicality` at physicality kOne ==
   `StepActive` (byte-identical), at 0 == pure-physics (no drive); `StepActiveRecover` — animCos high, struckCos <
   animCos, recoverCos > struckCos; two runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-active-recover` (features) + `--active-recover-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the AC1 compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED. `engine/
  sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `vehicle.h` + `engine/
  anim/` + `engine/physics/` + ALL shaders (incl `active_drive_solve.comp`) UNCHANGED. AC1-AC3 `active.h` functions
  UNCHANGED (AC4 additive — only the physicality step + the impulse + the episode + the showcase). **NO new
  shader.** Report the seam empty.

## Out of scope (YAGNI — later AC slices)
Lockstep/rollback (AC5 — note: physicality is per-tick-derived from the tick index, the impulse is a scripted
event, and the clip `time` is per-tick state; AC5 snapshots the world + the clip-time/tick), the lit skinned render
(AC6). A real hit-detection system / contact-triggered limp / a blend tree / get-up animations. AC4 claims ONLY: a
deterministic active→limp→recover episode (a global physicality knob scaling the drives + a scripted impulse + a
recovery ramp), bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE (honest): the
limp/recover is the AC1 soft-drive proxy scaled by physicality (deterministic, within the Gauss-Seidel residual),
NOT a mass-correct controlled fall; the impulse is a host velocity kick, not a contact-resolved hit.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 103 + the new `active_test` AC4 cases). Clean under
   `windows-msvc-asan` (build+run `active_test` + `introspect_test`).
2. **proofs + visual:** `--active-recover-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image
   shows the episode — the anim pose, the struck/limp displacement, the recovered pose (a coherent ragdoll in
   each, not scrambled).** Re-run `--active-step-shot` + `--active-blend-shot` + `--active-drive-shot` → still
   bit-exact AND their goldens byte-identical (AC1-AC3 render-invariance — AC4 is additive).
3. Metal: `visual_test --active-recover` → new golden `tests/golden/metal/active_recover.png`; two runs DIFF
   0.0000. **Confirm `visual_test.mm` in the diff; confirm NO new shader (`hf_gen_msl` UNCHANGED; `active`
   absent).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `active_recover.png` added (NOT
   active_drive/active_blend/active_step); the other 174 byte-identical. `git diff master --stat -- tests/golden` =
   ONLY `active_recover.png` (metal) + the introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-active-recover` + `--active-recover-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + the frozen sim/anim/physics headers + ALL shaders byte-unchanged; ONLY `active.h`
   extended additively + the showcase/test/introspect). `scripts/verify.ps1` updated: `active_recover` golden +
   `--active-recover-shot` in `$vkShots`. **NO new shader; `active` still NOT in `hf_gen_msl`.**
