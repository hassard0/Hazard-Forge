# Slice JT5 — Deterministic Articulated-Body Ragdoll: LOCKSTEP + ROLLBACK (the netcode headline) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #15 (DETERMINISTIC
> ARTICULATED-BODY RAGDOLL, `hf::sim::joint`). THE NETCODE HEADLINE, PURE CPU: a peer fed the input command
> stream ALONE re-derives the authority's exact ragdoll collapse — every bone's tumble + the settled slumped
> pose — bit-for-bit, and a rollback re-sim corrects a mispredicted impact EXACTLY. This is the FPX5/FR5/GF5/CG5
> lockstep harness applied to the JT3 articulated step — **a deterministic, rollback-replayable, bit-identical-
> cross-platform RAGDOLL, which UE5's float Chaos physical-animation structurally cannot provide**. NO new shader,
> NO new RHI — pure CPU over the JT3 `StepArticulatedContacts`. Branch: `slice-jt5`. See [[hazard-forge-joint-
> roadmap]].

**Goal:** Extend `engine/sim/joint.h` (additive — JT1-JT4 byte-unchanged) with the lockstep/rollback harness over
the ragdoll world: `SimRagdollTick` + `RunRagdollLockstep` + `RunRagdollRollback` over an `fpx::FxWorld`,
**reusing fpx's proven FPX5 command + snapshot machinery VERBATIM** (`fpx::FxCommand`/`ApplyCommand`/
`SnapshotWorld`/`RestoreWorld`), swapping only the per-tick step to `StepArticulatedContacts` (threading the
ragdoll's joints + angular limits, which are CONSTANT across ticks). Add `--joint-lockstep-shot` (Vulkan) /
`--joint-lockstep` (Metal). Bake the integer golden `joint_lockstep`. **NO new shader, NO new RHI.**

## Design call: the FPX5 harness over StepArticulatedContacts, MAXIMAL reuse (lowest risk)
The ragdoll world IS an `fpx::FxWorld` (JT4's `Ragdoll.world` — the bone bodies). fpx already ships the bit-exact
FPX5 lockstep/rollback machinery over `FxWorld`: `FxCommand` (an input impulse/spin on a body), `ApplyCommand`,
`SnapshotWorld` (deep-copy the world), `RestoreWorld`, `RunLockstep`, `RunRollback`. **JT5 reuses ALL of it
VERBATIM and changes ONE thing: the per-tick step is `joint::StepArticulatedContacts` (JT3: integrate → K
{ball|angular} → broadphase → contacts → ground) instead of fpx's `StepWorld`.** So JT5 is a thin harness —
`SimRagdollTick(world, joints, angularLimits, stream, tick, dt, iters, solveIters)` (apply the tick's
`fpx::FxCommand`s via `fpx::ApplyCommand`, then `StepArticulatedContacts`), and `RunRagdollLockstep`/
`RunRagdollRollback` (the loop over `SimRagdollTick`, mirroring fpx's `RunLockstep`/`RunRollback` control flow
byte-for-byte). The `joints`/`angularLimits` (the ragdoll TOPOLOGY) are CONSTANT across ticks — passed through,
NOT part of the snapshot; the snapshot is `fpx::SnapshotWorld`/`RestoreWorld` VERBATIM (the `FxWorld` bone-body
deep-copy). Pure CPU, integer, deterministic → lowest-risk slice (the direct FR5 twin over the ragdoll step).

### What is replayed
The ragdoll TOPOLOGY (the bones + joints + cone limits) is the deterministic `init` (JT4 `RagdollFromSkeleton`,
bit-reproducible). JT5 replays the COLLAPSE DYNAMICS — every bone's fall, swing, tumble, self-collision, and
settle — from the input impact stream, bit-for-bit. The command is a body impulse (`kCmdImpulse`, "punch a bone")
/ spin (`kCmdSetAngVel`) on a ragdoll bone, which two peers re-simulate identically. A peer with only the impacts
re-derives the EXACT slumped pose; rollback corrects a mispredicted punch.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FPX5 harness to REUSE VERBATIM (`engine/sim/fpx.h:506-603`):** `kCmdImpulse`/`kCmdSetAngVel`, `FxCommand`
  (:509), `ApplyCommand` (:519), `SnapshotWorld` (:553), `RestoreWorld` (:559), `RunLockstep` (:567),
  `RunRollback` (:583). JT5's `SimRagdollTick` is the `fpx::SimTick` (:538) twin with `StepArticulatedContacts`
  substituted; `RunRagdollLockstep`/`RunRagdollRollback` mirror `RunLockstep`/`RunRollback`'s exact control flow
  (snapshot at mispredictTick, ≤3 speculative ticks, restore + re-sim).
- **The FR5 lockstep PRECEDENT (`engine/sim/fract.h::SimFractTick`/`RunFractLockstep`/`RunFractRollback`):** the
  IDENTICAL thin-harness shape (reuse fpx command/snapshot, swap the step) over `StepFracture`. JT5 is this over
  `StepArticulatedContacts` — read it and translate (the extra `joints`/`angularLimits`/`iters`/`solveIters`
  params threaded through). Also `couple_gf.h::SimCGFTick`/`RunCGFLockstep` (the kernel-threaded twin).
- **The JT3/JT4 ragdoll (this branch's `joint.h`, read-only — DON'T modify):** `StepArticulatedContacts(world,
  joints, angularLimits, dt, iters, solveIters)` (the step JT5 drives), `RagdollFromSkeleton`/`Ragdoll`/
  `RagdollConfig` (build the init ragdoll world), `MeasureRagdoll`, `PoseToPalette`. `fpx::FxWorld`/`FxBody`.
- **The FR5/CG5 lockstep showcase mold (`--fract-lockstep`/`--cgrain-lockstep`):** the PURE-CPU showcase (run
  lockstep twice + rollback once, memcmp authority==replica + rollback==authority, then render the final state
  via the JT4 `joint_ragdoll` render path reused VERBATIM). FR5 is the closest twin. **Standalone arg-parse loop**
  (the FR1 C1061 lesson).
- **Showcase + registration:** JT1-JT4's `--joint-*-shot` plumbing; `scripts/verify.ps1`, `engine/editor/
  introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden**), `tests/joint_test.cpp`.

## Design decisions (locked)
1. **`SimRagdollTick(world, joints, angularLimits, stream, tick, dt, iters, solveIters)`** — apply ALL `stream`
   commands with `.tick == tick` in ARRAY ORDER via `fpx::ApplyCommand`, then `StepArticulatedContacts(world,
   joints, angularLimits, dt, iters, solveIters)`. The `fpx::SimTick` twin (+ the ragdoll step). Reuse
   `fpx::FxCommand` (NO new command type).
2. **`RunRagdollLockstep(init, joints, angularLimits, stream, ticks, dt, iters, solveIters)`** and
   **`RunRagdollRollback(init, joints, angularLimits, authStream, mispredictStream, ticks, mispredictTick, dt,
   iters, solveIters)`** — the `fpx::RunLockstep`/`RunRollback` control flow over `SimRagdollTick`. Snapshot via
   `fpx::SnapshotWorld`/`RestoreWorld` VERBATIM. (`init` is the `Ragdoll.world`; `joints`/`angularLimits` are the
   constant topology, passed by const-ref.)
3. **Showcase `--joint-lockstep-shot <out>` (Vulkan) AND `--joint-lockstep` (Metal) — WIRE BOTH (PURE CPU,
   standalone arg-parse).** Build the JT4 ragdoll (the synthetic skeleton → `RagdollFromSkeleton`, free or pinned
   root) as the init + an impact stream (punch a couple of bones at a few ticks). Run `RunRagdollLockstep` twice
   (authority + replica) + `RunRagdollRollback` once; assert authority==replica (memcmp the FxWorld) AND
   rollback==authority AND mispredicted≠authority. Render the final collapsed ragdoll via the JT4 `joint_ragdoll`
   render path. Golden = `tests/golden/metal/joint_lockstep.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) lockstep authority==replica:** two independent runs from the same init+stream → identical world.
     Print `joint-lockstep: {bones:<N>, joints:<J>, ticks:<T>} authority==replica BIT-IDENTICAL`.
   - **(2) rollback==authority:** `RunRagdollRollback` final == `RunRagdollLockstep(authStream)` final,
     byte-for-byte. Print `joint-lockstep rollback: corrected==authority BIT-EXACT`.
   - **(3) the misprediction was real:** the speculative (pre-rollback) state DIFFERED from authority. Print
     `joint-lockstep mispredict: diverged before rollback (real divergence fixed)`.
   - **(4) determinism:** two renders of the final state → byte-identical. Print `joint-lockstep determinism: two
     runs BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/joint_lockstep.png`; do NOT commit it.** Existing 163 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** the final lockstep render is Vulkan == Metal CPU-ref == golden, ZERO
   differing pixels (`StepArticulatedContacts` is already integer-bit-exact cross-backend — JT3 proved it; JT5
   only sequences it via inputs).
6. **Tests `tests/joint_test.cpp` additions (pure CPU):** `SimRagdollTick` advances the ragdoll deterministically;
   `RunRagdollLockstep` authority==replica on a tiny ragdoll; `RunRagdollRollback` == `RunRagdollLockstep(auth)`
   AND mispredicted≠authority; an impact command changes a bone's trajectory (the stream does work); snapshot
   round-trip (`fpx::SnapshotWorld`/`RestoreWorld`) bit-exact. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-joint-lockstep` (features) + `--joint-lockstep-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** PURE CPU — no compute, no dispatch. `rhi.h` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h`
  + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `engine/anim/` + `engine/physics/` + all existing shaders
  UNCHANGED. JT1-JT4 `joint.h` code + shaders UNCHANGED (JT5 additive — only the harness + the showcase). **NO new
  shader.** Report the seam empty.

## Out of scope (YAGNI — later JT slice)
The lit SKINNED render capstone (JT6 — JT5's render reuses the JT4 `joint_ragdoll` path as-is). Driving a
re-bind / topology change mid-replay. Network transport / delta-compression / prediction models (the harness
proves bit-exact replay + rollback; the transport is the existing net stack's job). JT5 claims ONLY: a
deterministic lockstep + rollback over the ragdoll collapse dynamics, bit-identical across two runs and (by
`StepArticulatedContacts`'s proven bit-exactness) across platforms, with the integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 101) + the new `joint_test` lockstep cases. Clean under
   `windows-msvc-asan` (build+run `joint_test` + `introspect_test`).
2. **proofs + visual:** `--joint-lockstep-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   rendered final state shows the collapsed ragdoll (the JT4 joint_ragdoll scene — pixel-check; the JT4 lesson).**
   ALSO re-run `--joint-ball/hinge/step/ragdoll-shot` → still bit-exact (JT1-JT4 render-invariance).
3. Metal: `visual_test --joint-lockstep` → new golden `tests/golden/metal/joint_lockstep.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (JT5 is
   pure CPU — `hf_gen_msl` UNCHANGED).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `joint_lockstep.png` added; the
   other 163 byte-identical. `git diff master --stat -- tests/golden` = ONLY `joint_lockstep.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-joint-lockstep` + `--joint-lockstep-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/`fract.h` +
   `engine/anim/` + `engine/physics/` + JT1-JT4 `joint.h`/shaders byte-unchanged). `scripts/verify.ps1` updated:
   `joint_lockstep` golden in the Mac loop + `--joint-lockstep-shot` in `$vkShots`. **NO new entry in
   `hf_gen_msl`.**
