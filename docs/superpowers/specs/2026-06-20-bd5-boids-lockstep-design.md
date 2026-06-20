# Slice BD5 — Deterministic GPU Crowds: LOCKSTEP + ROLLBACK (THE NETCODE HEADLINE) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #18 (DETERMINISTIC GPU
> CROWDS, `hf::sim::boids`). BD1-BD4 built a path-following flocking crowd. BD5 proves it is true cross-platform
> LOCKSTEP + ROLLBACK — the one-sentence moat: **a 256-agent crowd that's rollback-netcode-safe**, where UE5's
> Mass/AI is float and non-deterministic. Two peers fed ONLY the input perturbation stream re-derive the exact
> crowd trajectory bit-for-bit; a mispredicted perturbation is corrected by rolling back to a saved snapshot +
> re-simulating — the FPX5/FR5/GR5/CG5/GF5/JT5/VH5/AC5 twin. **PURE CPU** (NO GPU dispatch, NO new shader, NO new
> RHI) — both Vulkan-Windows (`--boids-lockstep-shot`) and Metal-Mac (`--boids-lockstep`) run the IDENTICAL CPU
> harness, so the converged golden is bit-identical cross-backend BY CONSTRUCTION. **THE BD-SPECIFIC SIMPLIFICATION:
> the snapshot is JUST the agent world** — `SteerPath` is STATELESS (recomputed each tick from the agent positions)
> and the corridor is a const input, so there is NO mutable extra to snapshot (unlike VH5's hinge axes / AC5's
> tick). Branch: `slice-bd5`. See [[hazard-forge-boids-roadmap]].

**Goal:** Extend `engine/sim/boids.h` (additive — BD1-BD4 byte-frozen) with `BoidsCommand` (a perturbation event) +
`ApplyBoidsCommand` + `BoidsSnapshot` + `SnapshotBoids`/`RestoreBoids` (just the agent world) + `SimBoidsTick`
(apply this tick's commands + `StepFlockPath`) + `RunBoidsLockstep` + `RunBoidsRollback`. Add `--boids-lockstep-shot`
(Vulkan) / `--boids-lockstep` (Metal). Bake the integer golden `boids_lockstep`. **NO GPU, NO new shader, NO new
RHI.**

## Design call: the FPX5 harness over StepFlockPath, snapshot = just the agents (stateless SteerPath)
The deterministic-sim flagships all share one netcode harness shape: a pure-CPU loop that (a) runs an authority + a
replica fed the SAME command stream and asserts they stay BIT-IDENTICAL, and (b) snapshots, mispredicts, rolls
back, re-simulates, and asserts the corrected state == authority. BD5 is that harness over the crowd.

**THE INPUT is a PERTURBATION stream.** `BoidsCommand { uint32 tick; uint32 agent; FxVec3 dv; }` — a velocity kick
to one agent (a "scare"/"shove" event; the AC5 `ApplyImpulse` analog, lifted to a replayable input).
`ApplyBoidsCommand(agents, cmd)`: `if agent in-range: agents[cmd.agent].vel = FxAdd(vel, cmd.dv)`. A perturbation
scatters part of the flock; the flock+path drive re-coheres it. (Other input kinds — a goal-waypoint change — are a
BD-future; the velocity kick is the simplest replayable input.)

**THE SNAPSHOT IS JUST THE AGENT WORLD (the BD simplification).** `StepFlockPath` rebuilds the grid+neighbors from
the current positions each tick (no persistent grid state), `SteerPath` is recomputed each tick from the positions
(no per-agent path-state), and the corridor `BoidsPath` is a CONST input (not mutated). So the ONLY mutable
replayable state is `std::vector<Agent>`. `BoidsSnapshot { std::vector<Agent> agents; }`; `SnapshotBoids` deep-copies
the agent vector; `RestoreBoids` restores it. Simpler than VH5 (hinge axes) / AC5 (tick) — document this as the BD
win (the stateless steering pays off here).

`SimBoidsTick(agents, cfg, path, commands, tick, dt)`: for each `cmd` with `cmd.tick == tick` (ARRAY ORDER),
`ApplyBoidsCommand`; then `StepFlockPath(agents, cfg, path, dt)`.

`RunBoidsLockstep(cfg, path, initialAgents, commands, ticks, dt)`: build an `authority` + a `replica` from the SAME
initial agents, step BOTH with `SimBoidsTick` over the SAME command stream for `ticks` ticks, assert bit-identical
every tick (memcmp the agent vector). Returns the converged authority.

`RunBoidsRollback(cfg, path, initialAgents, authorityCmds, mispredictCmds, divergeTick, ticks, dt)`: advance to
`divergeTick` with `authorityCmds`; `SnapshotBoids`; speculatively advance a few ticks with `mispredictCmds` (a
WRONG perturbation — a different agent/dv/tick); `RestoreBoids` to the snapshot + re-simulate `divergeTick..ticks`
with `authorityCmds`; assert the corrected peer == authority bit-for-bit AND the mispredicted (pre-rollback) state
HAD diverged.

**The showcase is PURE CPU** (both backends run the identical harness; the converged crowd rendered via the BD4 2D
top-down view; bit-identical cross-backend by construction; NO GPU==CPU memcmp — the proof is authority==replica +
rollback==authority + the cross-platform golden).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **BD1-BD4 (this branch's `boids.h`, read-only — build on, DON'T modify):** `Agent`, `FlockConfig`, `BoidsPath`,
  `StepFlockPath` (the per-tick step), `MeasureFlockPath`. DO NOT modify the BD1-BD4 functions. BD5 APPENDS.
- **The lockstep/rollback harness to TWIN (`engine/sim/active.h` AC5 — `ActiveCommand`/`ActiveSnapshot`/
  `SnapshotActive`/`RestoreActive`/`SimActiveTick`/`RunActiveLockstep`/`RunActiveRollback`; ALSO `vehicle.h` VH5,
  `joint.h` JT5, `fpx.h` FPX5):** the authority/replica loop + the snapshot/mispredict/rollback structure + the
  proof assertions. BD5's snapshot is SIMPLER (just `std::vector<Agent>` — no mutable extra). Match the AC5
  signatures/returns + proof structure. (No `fpx::SnapshotWorld` here — the agent vector is a plain deep-copy.)
- **fpx (`engine/sim/fpx.h`, read-only):** `FxVec3`/`FxAdd`, `Agent` is boids-local. `<cstring>` for `std::memcmp`.
  **DO NOT modify fpx.h.**
- **Showcase + registration:** the BD4 `--boids-path-shot` plumbing (the navmesh+corridor build + the BD4 2D
  render) — copy for `--boids-lockstep-shot` / `--boids-lockstep`, but PURE CPU (no GPU dispatch path). Standalone
  arg-parse. `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), `tests/boids_test.cpp`.

## Design decisions (locked)
1. **`struct BoidsCommand { uint32_t tick; uint32_t agent; FxVec3 dv; }`** + **`ApplyBoidsCommand(agents, cmd)`** +
   **`struct BoidsSnapshot { std::vector<Agent> agents; }`** + **`SnapshotBoids(agents) -> BoidsSnapshot`** +
   **`RestoreBoids(agents, snap)`** + **`SimBoidsTick`** + **`RunBoidsLockstep`** + **`RunBoidsRollback`** (above).
   Mirror AC5. Pure integer/host.
2. **Showcase `--boids-lockstep-shot <out>` (Vulkan-Windows) AND `--boids-lockstep` (Metal-Mac) — WIRE BOTH**
   (standalone arg-parse), PURE CPU. Build the BD4 navmesh+corridor + a flock at the start, a scripted perturbation
   `authStream` (a few velocity kicks scattering parts of the flock at a few ticks — so the crowd is perturbed off
   the path then re-coheres + re-follows), run `RunBoidsLockstep` (authority==replica) + `RunBoidsRollback` (a
   mispredicted kick at one tick, rolled back). Render the converged crowd via the BD4 2D view REUSED. Golden =
   `tests/golden/metal/boids_lockstep.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) authority==replica:** two peers fed only the perturbation stream stay bit-identical for all ticks. Print
     `boids-lockstep: {agents:<N>, ticks:<T>, perturbations:<P>} authority==replica BIT-IDENTICAL`.
   - **(2) rollback==authority:** the corrected (rolled-back + re-simulated) peer == authority byte-for-byte. Print
     `boids-lockstep rollback: corrected==authority BIT-EXACT`.
   - **(3) mispredict diverged:** the mispredicted (pre-rollback) state HAD diverged from authority. Print
     `boids-lockstep mispredict: diverged before rollback (real divergence fixed)`.
   - **(4) determinism:** two full runs → identical. Print `boids-lockstep determinism: two runs BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/boids_lockstep.png`; do NOT commit it.** Existing 181 image
     goldens UNTOUCHED (incl BD1-BD4).
4. **Cross-backend bar (INTEGER, strict):** Vulkan-Windows == Metal-Mac == golden, ZERO differing pixels (the CPU
   harness is identical on both → the rendered converged crowd is bit-identical by construction).
5. **Tests `tests/boids_test.cpp` additions (pure CPU):** `SnapshotBoids`/`RestoreBoids` round-trip (snapshot →
   mutate (a few perturb+step ticks) → restore → agent vector bit-identical to the snapshot); `ApplyBoidsCommand`
   kicks the target agent's vel (out-of-range no-op); `RunBoidsLockstep` authority==replica over a perturbation
   stream; `RunBoidsRollback` a mispredicted kick diverges, rollback corrects to authority. Two runs byte-identical.
   Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-boids-lockstep` (features) + `--boids-lockstep-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** PURE CPU — NO GPU dispatch, NO compute, NO new shader, NO new RHI. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `joint.h` + `couple*.h` + `fract.h` + `vehicle.h` +
  `active.h` + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL shaders (incl the boids shaders) +
  `hf_gen_msl` UNCHANGED. BD1-BD4 `boids.h` code UNCHANGED (BD5 additive — only the command + snapshot + harness +
  showcase). Report the seam empty.

## Out of scope (YAGNI — BD6 only remains)
The lit 3D render (BD6 — BD5's render is the BD4 2D diagnostic). A network transport (BD5 proves the lockstep MATH
— inputs-only re-derivation + rollback — not a socket layer, the GR5/JT5/VH5/AC5 precedent). Goal-change commands /
dynamic corridor in lockstep (BD5 uses the velocity-kick perturbation + the fixed corridor). BD5 claims ONLY: the
bit-exact path-following flocking crowd is deterministic LOCKSTEP + ROLLBACK (two peers re-derive it from the
perturbation stream alone; a mispredict rolls back to a snapshot — just the agent world — and corrects), bit-
identical CPU↔Vulkan-Windows↔Metal-Mac, with the integer golden + the four proofs. This is the cross-platform
rollback-replayable CROWD UE5's float Mass/AI cannot do.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 104 + the new `boids_test` lockstep cases). Clean under
   `windows-msvc-asan` (build+run `boids_test` + `introspect_test`).
2. **proofs + visual:** `--boids-lockstep-shot` on Vulkan-Windows: the 4 proofs + exit 0, under the
   Vulkan-validation gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED —
   the showcase still spins up the device for the render). **VERIFY the image shows the converged crowd (a coherent
   flock on/near the corridor — not scattered, not stuck).** Re-run `--boids-path-shot` + `--boids-flock-shot` +
   `--boids-neighbors-shot` + `--boids-steer-shot` → still bit-exact (BD1-BD4 render-invariance).
3. Metal-Mac: `visual_test --boids-lockstep` → new golden `tests/golden/metal/boids_lockstep.png`; two runs DIFF
   0.0000. **Confirm `visual_test.mm` in the diff; confirm NO new shader (`hf_gen_msl` UNCHANGED).** Cross-vendor
   STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `boids_lockstep.png` added; the
   other 181 byte-identical. `git diff master --stat -- tests/golden` = ONLY `boids_lockstep.png` (metal) + the
   introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-boids-lockstep` + `--boids-lockstep-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + the frozen sim/nav/anim/physics headers + ALL shaders byte-unchanged; ONLY `boids.h`
   extended additively + the showcase/test/introspect). `scripts/verify.ps1` updated: `boids_lockstep` golden +
   `--boids-lockstep-shot` in `$vkShots`. **NO new shader; the boids shaders all unchanged.**
