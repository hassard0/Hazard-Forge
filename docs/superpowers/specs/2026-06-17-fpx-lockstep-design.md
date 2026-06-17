# Slice FPX5 — Deterministic Fixed-Point Physics: LOCKSTEP + ROLLBACK PROOF (the beyond-UE5 headline) (Phase 11 #5) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 5th FPX slice and the
> FLAGSHIP'S HEADLINE: prove that the fixed-point sim is true cross-platform LOCKSTEP + ROLLBACK netcode-ready — two
> peers fed ONLY a deterministic input/command stream (NOT full state) re-simulate to BIT-IDENTICAL physics state, and
> a mispredicted input is corrected by ROLLING BACK to a saved snapshot and re-simulating to the authoritative state.
> This is the concrete capability UE5's (float) Chaos cannot provide: float physics diverges across machines, so
> inputs-only state sync (the basis of lockstep/rollback netcode) is impossible. fpx makes it work. Pure-CPU harness
> over the FPX1–FPX4 sim (no GPU dispatch needed — lockstep/rollback is a determinism property; the cross-backend
> zero-diff golden IS the cross-platform-lockstep evidence). NO new RHI, NO new shader. Namespace `hf::sim::fpx`.
> Branch: `slice-fpx-lockstep`. See [[hazard-forge-fpx-roadmap]].

**Goal:** Add a deterministic command/input model + a lockstep+rollback harness to `engine/sim/fpx.h`
(`FxCommand`, `ApplyCommand`, `SimTick`, `SnapshotWorld`/`RestoreWorld`, `RunLockstep`, `RunRollback`) + a
`--fpx-lockstep-shot` (Vulkan) / `--fpx-lockstep` (Metal) showcase that runs an authority + a replica + a rollback
scenario over a scripted command stream, proves authority==replica==rolled-back state BIT-FOR-BIT, and bakes the
converged-state golden. Make-safe: header additions (FPX1–FPX4 UNCHANGED) + a NEW showcase + NEW golden; the float
`engine/physics/` UNCHANGED. The cross-backend bit-identity (the converged state is the same on Vulkan-Windows and
Metal-Mac) is the cross-platform-lockstep demonstration; the in-showcase lockstep + rollback proofs are the netcode
guarantee.

## Why this is the headline (the concrete UE5 gap)
The engine already shipped snapshot replication + client-prediction (`net/snapshot.h:99` `Replicator::Capture(... const
physics::World&)`), but those ride the FLOAT `physics::World`, which diverges across vendors/compilers — so true
lockstep (peers re-simulate from inputs alone to identical state) and rollback (re-simulate a corrected input from a
saved tick) are unattainable today, exactly as in UE5's Chaos. The fpx fixed-point sim is bit-identical
CPU↔Vulkan↔Metal AND run-to-run (proven FPX1–FPX4), so this slice DEMONSTRATES the payoff: inputs-only lockstep +
rollback convergence, bit-for-bit. This is the "more powerful than UE5" claim made concrete and golden-verified.

## The lockstep/rollback core (extends fpx.h)
- **`struct FxCommand { uint32_t tick; uint32_t kind; uint32_t bodyId; FxVec3 arg; };`** — a deterministic per-tick
  input (the thing a netcode layer puts on the wire): e.g. `kind=0` apply-impulse `arg` to `bodyId`, `kind=1`
  set-angVel, `kind=2` spawn (fixed). A `std::vector<FxCommand>` is the command STREAM (sorted by tick, deterministic).
- **`void ApplyCommand(FxWorld&, const FxCommand&)`** — apply one command (integer: add to vel / set angVel / etc.).
- **`void SimTick(FxWorld&, std::span<const FxPair> pairs, const std::vector<FxCommand>& stream, uint32_t tick, fx dt,
  int solveIters)`** — the deterministic per-tick step: apply all commands for `tick` (in deterministic order), then
  `StepWorld` (integrate + solve) (+ `IntegrateBodyFull` for orientation). The pair list is rebuilt per tick from the
  current positions (`BuildPairs`) OR fixed — DECISION: rebuild per tick (`BuildPairs` is deterministic) so the
  lockstep is realistic.
- **Snapshot/restore (the rollback primitive):** `FxWorld SnapshotWorld(const FxWorld&)` = a deep copy (the
  `std::vector<FxBody>` + scalars); `void RestoreWorld(FxWorld&, const FxWorld& snap)`. Deterministic (full integer
  state).
- **`FxWorld RunLockstep(const FxWorld& init, const std::vector<FxCommand>& stream, int ticks, fx dt, int iters)`** —
  run the sim `ticks` ticks applying the stream → the final world. THE peer entry point: authority = `RunLockstep(init,
  stream, N)`, replica = `RunLockstep(init, stream, N)` from the SAME init + stream (inputs only — no state shared) →
  identical by determinism. (The proof asserts they're bit-for-bit equal.)
- **`FxWorld RunRollback(const FxWorld& init, const std::vector<FxCommand>& authStream, const std::vector<FxCommand>&
  mispredictStream, int ticks, int mispredictTick, fx dt, int iters)`** — the rollback harness: run ticks `0..mispredictTick`
  saving a snapshot at `mispredictTick`; from `mispredictTick`, advance a few ticks using the MISPREDICTED stream (a
  wrong input); then "receive" the correct input → `RestoreWorld` to the snapshot + re-simulate `mispredictTick..ticks`
  with the CORRECT (`authStream`) → the final world. The proof asserts this equals the authority's `RunLockstep`
  (rollback corrected the misprediction exactly), AND that the mispredicted-before-rollback state DIFFERED from the
  authority (the rollback actually fixed something, not a no-op).

## Reuse map (file:line)
- **FPX1–FPX4 (the sim):** `engine/sim/fpx.h` — `FxWorld`/`FxBody`, `IntegrateStep`/`StepWorld` (FPX1/FPX3),
  `BuildPairs` (FPX2), `IntegrateBodyFull` (FPX4). `SimTick`/`RunLockstep` compose these. NO changes to them.
- **The netcode framing (conceptual, not a code dependency):** `engine/net/snapshot.h:99` (`Replicator` —
  inputs-only/state-on-wire; FPX5's command stream is what would ride it; FPX5 stays self-contained — do NOT couple to
  the float `physics::World` path).
- **The integer-from-readback debug-viz golden:** the FPX1/FPX3 side-view (`pos>>kFrac → pixel`, `hashColor`).
- **Showcase + registration:** the FPX shapes; `verify.ps1`/`introspect.cpp`/`introspect_test.cpp`.

## Design decisions (locked)

1. **Pure-CPU harness — NO new GPU shader, NO new RHI.** The lockstep/rollback is a determinism property of the CPU
   fpx sim; the showcase runs the harness on the CPU (identically on Vulkan-Windows and Metal-Mac → the golden is
   bit-identical cross-backend BY CONSTRUCTION — which IS the cross-platform-lockstep evidence). (No GPU dispatch is
   needed; this is acceptable — SW1 was pure-CPU. If a GPU==CPU angle is wanted later, the existing fpx_solve.comp can
   run the authority sim — DEFER; the lockstep proofs are the substance.)
2. **Showcase `--fpx-lockstep-shot <out>` (Vulkan, main.cpp) AND `--fpx-lockstep` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm + `#include "sim/fpx.h"`).** A deterministic `FxWorld` (a handful of bodies on a ground) + a
   scripted `authStream` (e.g. impulses applied at known ticks to make the bodies move/collide/settle interestingly) +
   a `mispredictStream` (a wrong impulse at one tick) + `mispredictTick`. Run authority `RunLockstep`, replica
   `RunLockstep`, and `RunRollback`; assert all converge. Golden = the converged final state side-view (the bodies'
   integer positions, CPU-colored) → `tests/golden/metal/fpx_lockstep.png`. (A richer viz: show the authority state +
   overlay that the replica/rolled-back match — but the simplest correct viz is the single converged state, since A,
   B, B' are identical.)
3. **PROOFS (fail loudly; exact lines):**
   - **(1) LOCKSTEP (the headline):** `memcmp(authority.bodies, replica.bodies) == 0` after N ticks (replica fed
     INPUTS ONLY re-derives the authority's exact state). Print `fpx-lockstep replica==authority: <N> bodies BIT-EXACT
     (<T> ticks, inputs-only)`.
   - **(2) ROLLBACK (the headline):** `memcmp(rolledBack.bodies, authority.bodies) == 0` (a mispredicted input is
     corrected by rollback+re-sim to the exact authoritative state), AND the pre-rollback mispredicted state DIFFERED
     from the authority (the rollback fixed a real divergence). Print `fpx-lockstep rollback: corrected to authority
     BIT-EXACT (mispredict@tick<m> diverged then converged)`.
   - **(3) determinism:** running the whole harness twice → byte-identical converged state. Print `fpx-lockstep
     determinism: two runs BYTE-IDENTICAL`.
   - **(4) snapshot round-trip:** `RestoreWorld(SnapshotWorld(w))` == `w` (the rollback primitive is lossless). Print
     `fpx-lockstep snapshot: round-trip BIT-EXACT`.
   - **(5) {stats}:** `fpx-lockstep: {bodies:<N>, ticks:<T>, commands:<C>, mispredict-tick:<m>}`.
   - **Golden discipline: ONLY `tests/golden/metal/fpx_lockstep.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 109 image goldens UNTOUCHED. **The golden being cross-backend ZERO-DIFF (Vulkan-Windows ==
     Metal-Mac) is itself the cross-platform-lockstep proof — the same simulated state on two platforms from the same
     inputs.**
4. **Determinism / cross-backend.** The harness is fixed-point integer CPU code, deterministic by construction
   (FPX1–FPX4); run identically on both platforms → the converged-state golden is bit-identical → cross-platform
   lockstep demonstrated. Run under the Vulkan sync-validation gate (the showcase still creates the headless device +
   writes the image) → SYNC-HAZARD-free / 0 VUID.
5. **Tests `tests/fpx_test.cpp` additions (pure CPU):** `ApplyCommand` (impulse/angVel apply correctly);
   `SnapshotWorld`/`RestoreWorld` round-trip == original; `SimTick` deterministic; `RunLockstep` — two calls from the
   same init+stream are bit-identical (lockstep); `RunRollback` — converges to `RunLockstep`(authStream) AND the
   mispredicted path differs (a positive + a negative control); the command stream applied in deterministic order.
   Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-fixedpoint-physics-lockstep` (features) + `--fpx-lockstep-shot`
   (showcases).

## RHI seam additions (summary)
- **None.** Pure-CPU harness over the existing fpx sim; the showcase reuses the headless device + image-write path.
  ZERO above-seam backend symbols, NO new shader, NO new RHI. `rhi.h` + `rhi_factory` (baseline 2) + backend dirs
  UNCHANGED. `engine/physics/` + FPX1–FPX4 UNTOUCHED. Report the seam.

## Out of scope (YAGNI — FPX6 / beyond)
A real network transport (the command stream is a deterministic scripted sequence — the "wire" is conceptual; an
actual UDP/socket peer is a future netcode slice), coupling to the float `engine/net` `physics::World` path (fpx stays
self-contained), input prediction heuristics / interpolation, the float 3D render of the sim (FPX6), a box-SAT
collider. Claim DETERMINISM + cross-platform BIT-IDENTITY + inputs-only lockstep + rollback convergence — the netcode
primitive UE5's float Chaos lacks. ONE lockstep + rollback harness with the replica==authority + rollback-converges +
snapshot-round-trip + determinism proofs and the converged-state golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 92) + the new `fpx_test` lockstep/rollback cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fpx-lockstep-shot` on Vulkan: a coherent converged-state viz (the bodies after the scripted
   sim); `replica==authority BIT-EXACT` + `rollback corrected to authority BIT-EXACT` + determinism + snapshot
   round-trip + the `{...}` line. Run under the Vulkan-validation gate → ZERO VUID in the OUTPUT.
3. Metal: `visual_test --fpx-lockstep` → new golden `tests/golden/metal/fpx_lockstep.png`; two runs DIFF 0.0000 (gate
   on compare.sh EXIT CODE). Pure-CPU integer harness → MSL-gen N/A (no shader). **Confirm visual_test.mm in the
   diff.** Integer golden → a strict cross-backend pixel compare must show ZERO differing pixels — the cross-platform
   lockstep proof (the same simulated state on Vulkan-Windows + Metal-Mac).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fpx_lockstep.png` added; the other 109
   byte-identical (FPX1–FPX4 untouched). `git diff master --stat -- tests/golden` = ONLY `fpx_lockstep.png` (metal) +
   the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fixedpoint-physics-lockstep` + `--fpx-lockstep-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI, no new shader). `scripts/verify.ps1` updated: `fpx_lockstep`
   golden in the Mac loop + `--fpx-lockstep-shot` in `$vkShots`.
