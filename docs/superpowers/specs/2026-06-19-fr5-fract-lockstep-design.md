# Slice FR5 — Deterministic Fracture/Destruction: LOCKSTEP + ROLLBACK (the netcode headline) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #14 (DETERMINISTIC
> RIGID-BODY FRACTURE / DESTRUCTION, `hf::sim::fract`). THE NETCODE HEADLINE, PURE CPU: a peer fed the input
> command stream ALONE re-derives the authority's exact destroyed state — every dislodged chunk's tumble + the
> settled rubble pile — bit-for-bit, and a rollback re-sim corrects a mispredicted input EXACTLY. This is the
> FPX5/GR5/GF5/CG5 lockstep harness applied to the fracture step — the first time a DESTRUCTION sim is shown
> deterministically replayable cross-platform (UE5's float Chaos fracture cannot replay a break bit-for-bit).
> NO new shader, NO new RHI — pure CPU over the FR4 `StepFracture`. Branch: `slice-fr5`. See
> [[hazard-forge-fract-roadmap]].

**Goal:** Extend `engine/sim/fract.h` (additive — FR1-FR4 byte-unchanged) with the lockstep/rollback harness over
the fracture world: `SimFractTick` + `RunFractLockstep` + `RunFractRollback` over an `fpx::FxWorld`, **reusing
fpx's proven FPX5 command + snapshot machinery VERBATIM** (`fpx::FxCommand`/`ApplyCommand`/`SnapshotWorld`/
`RestoreWorld`), swapping only the per-tick step (`StepFracture` instead of fpx's `StepWorld`). Add
`--fract-lockstep-shot` (Vulkan) / `--fract-lockstep` (Metal). Bake the integer golden `fract_lockstep`. **NO new
shader, NO new RHI.**

## Design call: the FPX5 harness over StepFracture, MAXIMAL reuse (lowest risk)
The fracture world IS an `fpx::FxWorld` (FR4's `SpawnFractWorld` output — the broken fragment bodies, anchor
static + dislodged chunks dynamic). fpx already ships the bit-exact FPX5 lockstep/rollback machinery over
`FxWorld`: `FxCommand` (an input impulse/spin on a body), `ApplyCommand`, `SnapshotWorld` (deep-copy the world),
`RestoreWorld`, `RunLockstep`, `RunRollback`. **FR5 reuses ALL of it VERBATIM and changes ONE thing: the per-tick
step is `fract::StepFracture` (FR4: re-broadphase + IntegrateBodyFull + SolveContacts + ground) instead of fpx's
`StepWorld`+`IntegrateOrientation`.** So FR5 is a thin harness — `SimFractTick(world, stream, tick, dt,
solveIters)` (apply the tick's `fpx::FxCommand`s via `fpx::ApplyCommand`, then `StepFracture`), and
`RunFractLockstep`/`RunFractRollback` (the loop over `SimFractTick`, mirroring fpx's `RunLockstep`/`RunRollback`
control flow byte-for-byte, but calling `SimFractTick`). The snapshot is `fpx::SnapshotWorld`/`RestoreWorld`
VERBATIM (the `FxWorld` deep-copy — bodies + scalars). Pure CPU, integer, deterministic → lowest-risk slice.

### What is replayed (the break is in the init; the settle is replayed)
The DESTRUCTION = the break (FR3) + the spawn (FR4 `SpawnFractWorld`, which sets which bodies are dynamic from
the impact). That break is itself DETERMINISTIC + bit-reproducible (a fixed impact → a fixed severed set → a
fixed dynamic/static body assignment), so the `init` `FxWorld` two peers start from is identical by construction.
FR5 then replays the **rubble dynamics** — every dislodged chunk's fall, collision, tumble, and settle — from
the input shove stream, bit-for-bit. The snapshot is the `FxWorld` (the fragment-body set); the bond/severed
state is fixed after the initial break, so it does not change during the settle (it lives in the init's body
flags). **(Driving a FURTHER break mid-replay from a stream command — re-severing bonds + releasing more chunks
during the fall — is a documented extension; FR5 replays the post-break settle, the lowest-risk faithful version
that matches GF5/CG5/FPX5.)** The command is a body shove (`kCmdImpulse`) / spin (`kCmdSetAngVel`) on a dislodged
chunk — "kick a falling chunk", which two peers re-simulate identically.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FPX5 harness to REUSE VERBATIM (`engine/sim/fpx.h:506-603`):** `kCmdImpulse`/`kCmdSetAngVel`, `FxCommand`
  (:509), `ApplyCommand` (:519), `SnapshotWorld` (:553), `RestoreWorld` (:559), `RunLockstep` (:567),
  `RunRollback` (:583). FR5's `SimFractTick` is the `fpx::SimTick` (:538) twin with `StepFracture` substituted for
  the `StepWorld`+`IntegrateOrientation` body; `RunFractLockstep`/`RunFractRollback` mirror `RunLockstep`/
  `RunRollback`'s exact control flow (snapshot at mispredictTick, ≤3 speculative ticks, restore + re-sim).
- **The FR4 step to DRIVE (this branch's `fract.h`, read-only):** `StepFracture(world, dt, solveIters)`,
  `SpawnFractWorld`/`FractStepConfig` (build the init broken world), `MeasureFractRubble`. `fpx::FxWorld`/
  `FxBody`/`kFlagDynamic`. **DO NOT modify fpx.h or FR1-FR4 code** — FR5 only ADDS the three harness functions +
  the showcase.
- **The GF5/CG5 lockstep showcase mold (`--cgf-lockstep`/`--cgrain-lockstep`):** the PURE-CPU showcase (run
  lockstep twice + rollback once, memcmp authority==replica + rollback==authority, then render the final state
  via the FR4 `fract_step` render path reused VERBATIM). FR5 wires `--fract-lockstep-shot`/`--fract-lockstep` the
  same way — **standalone arg-parse loop** (the FR1 C1061 lesson).
- **Showcase + registration:** FR1-FR4's `--fract-*-shot` plumbing; `scripts/verify.ps1`, `engine/editor/
  introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden**), `tests/fract_test.cpp`.

## Design decisions (locked)
1. **`SimFractTick(world, stream, tick, dt, solveIters)`** — apply ALL `stream` commands with `.tick == tick` in
   ARRAY ORDER via `fpx::ApplyCommand`, then `StepFracture(world, dt, solveIters)`. The `fpx::SimTick` twin (+
   StepFracture). Reuse `fpx::FxCommand` (NO new command type).
2. **`RunFractLockstep(init, stream, ticks, dt, solveIters)`** and **`RunFractRollback(init, authStream,
   mispredictStream, ticks, mispredictTick, dt, solveIters)`** — the `fpx::RunLockstep`/`RunRollback` control flow
   over `SimFractTick`. Snapshot via `fpx::SnapshotWorld`/`RestoreWorld` VERBATIM.
3. **Showcase `--fract-lockstep-shot <out>` (Vulkan) AND `--fract-lockstep` (Metal) — WIRE BOTH (PURE CPU,
   standalone arg-parse).** Build the FR4 broken-and-spawned world (the init) + a shove stream (kick a couple of
   dislodged chunks at a few ticks). Run `RunFractLockstep` twice (authority + replica) + `RunFractRollback`
   once; assert authority==replica (memcmp the FxWorld) AND rollback==authority AND mispredicted≠authority.
   Render the final rubble via the FR4 `fract_step` render path. Golden = `tests/golden/metal/fract_lockstep.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) lockstep authority==replica:** two independent runs from the same init+stream → identical world.
     Print `fract-lockstep: {bodies:<N>, dynamic:<D>, ticks:<T>} authority==replica BIT-IDENTICAL`.
   - **(2) rollback==authority:** `RunFractRollback` final == `RunFractLockstep(authStream)` final, byte-for-byte.
     Print `fract-lockstep rollback: corrected==authority BIT-EXACT`.
   - **(3) the misprediction was real:** the speculative (pre-rollback) state DIFFERED from authority. Print
     `fract-lockstep mispredict: diverged before rollback (real divergence fixed)`.
   - **(4) determinism:** two renders of the final state → byte-identical. Print `fract-lockstep determinism: two
     runs BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/fract_lockstep.png`; do NOT commit it.** Existing 157 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** the final lockstep render is Vulkan == Metal CPU-ref == golden, ZERO
   differing pixels (StepFracture is already integer-bit-exact cross-backend; FR5 only sequences it via inputs).
6. **Tests `tests/fract_test.cpp` additions (pure CPU):** `SimFractTick` advances the world deterministically;
   `RunFractLockstep` authority==replica on a tiny broken world; `RunFractRollback` == `RunFractLockstep(auth)`
   AND mispredicted≠authority; a shove command changes a chunk's trajectory (the stream does work); snapshot
   round-trip (`fpx::SnapshotWorld`/`RestoreWorld`) bit-exact. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-fract-lockstep` (features) + `--fract-lockstep-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** PURE CPU — no compute, no dispatch. `rhi.h` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h`
  + `fluid.h` + `cloth.h` + `couple.h` + `couple_grain.h` + `couple_gf.h` + `engine/physics/` + all existing
  shaders UNCHANGED. FR1-FR4 `fract.h` code + shaders UNCHANGED (FR5 additive — only the harness + the showcase).
  **NO new shader.** Report the seam empty.

## Out of scope (YAGNI — later FR slice)
The lit 3D render capstone (FR6 — FR5's render reuses the FR4 `fract_step` path as-is). Driving a FURTHER break
mid-replay from a stream command (re-severing bonds during the fall — the documented extension). Network
transport / delta-compression / prediction models (the harness proves bit-exact replay + rollback; the transport
is the existing net stack's job). FR5 claims ONLY: a deterministic lockstep + rollback over the fracture rubble
dynamics, bit-identical across two runs and (by `StepFracture`'s proven bit-exactness) across platforms, with the
integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 100) + the new `fract_test` lockstep cases. Clean under
   `windows-msvc-asan` (build+run `fract_test` + `introspect_test`).
2. **proofs + visual:** `--fract-lockstep-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   rendered final state shows the settled rubble (the FR4 fract_step scene — pixel-check; the FR4 lesson).**
3. Metal: `visual_test --fract-lockstep` → new golden `tests/golden/metal/fract_lockstep.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (FR5 is
   pure CPU — `hf_gen_msl` UNCHANGED).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fract_lockstep.png` added; the
   other 157 byte-identical. `git diff master --stat -- tests/golden` = ONLY `fract_lockstep.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fract-lockstep` + `--fract-lockstep-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h` +
   `engine/physics/` + FR1-FR4 `fract.h`/shaders byte-unchanged). `scripts/verify.ps1` updated: `fract_lockstep`
   golden in the Mac loop + `--fract-lockstep-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`.**
