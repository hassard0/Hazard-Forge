# Slice BY — Client Prediction + Server Reconciliation (Phase 4 #24) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Completes the
> NETWORKING TRILOGY: replication (BQ) → transport + interpolation (BU) → **prediction + reconciliation**
> (this). Deterministic; still NO sockets (in-process channel from BU).

**Goal:** Hide network latency for the LOCAL player via client-side prediction, and stay authoritative via
server reconciliation. The client runs the deterministic game simulation locally and applies its OWN
inputs IMMEDIATELY (prediction — no waiting for the server round-trip). The authority runs the same sim
PLUS scripted SERVER-ONLY effects the client cannot predict; when an authoritative snapshot arrives
(delayed via the BU channel), the client RECONCILES — rewinds to the acknowledged authoritative state and
replays its unacknowledged inputs — correcting any misprediction. Proven deterministic; the reconciled
client CONVERGES to the authority, and a real misprediction is created + corrected (so the test isn't
trivial).

## Why this is verifiable (deterministic prediction + a real misprediction)

If the client and authority ran the IDENTICAL sim from the same inputs, prediction would always be exactly
right and reconciliation a no-op — a trivial, unconvincing test. To create a REAL misprediction to
correct: the authority applies SERVER-ONLY impulses (scripted/seeded ticks — e.g. a "bumper" hit) the
client doesn't know about until the authoritative snapshot reveals them. So the client's predicted path
diverges (peak misprediction > 0), then reconciliation snaps it back onto the authoritative path. All
deterministic (scripted inputs + scripted server-only events + the seeded BU channel) ⇒ bit-stable, two
runs identical. (Real transport remains future; the BU in-process channel carries the snapshots.)

## Design decisions (locked)

1. **Prediction + reconciliation (engine/net/prediction.{h,cpp}, pure CPU, no backend symbols).** Namespace
   `hf::net`. Reuses the deterministic roll-game (AX) sim, BQ `Snapshot`/`Replicator`, BU `SimChannel`.
   - `struct InputCmd { int tick; game::GameInput input; };` — the client's per-tick input (from the AX
     scripted track).
   - `class PredictedClient` — holds a local game `World`/`GameState`, a ring of recent `InputCmd`s
     (unacknowledged), and the last acknowledged authoritative `Snapshot` (tick + state).
     - `void PredictTick(const InputCmd&)` — apply the input to the LOCAL sim immediately (one `StepGame`),
       advancing the predicted state ahead of the server.
     - `void OnAuthoritative(const Snapshot& authSnap)` — RECONCILE: reset the local sim to `authSnap`
       (the acknowledged authoritative state at its tick), then REPLAY every buffered `InputCmd` with
       `tick > authSnap.tick` (the unacknowledged inputs) to re-derive the predicted "now". Drop inputs
       `<= authSnap.tick` (acknowledged). Track `lastMisprediction = distance(predictedBefore,
       predictedAfterReconcile)` and `maxMisprediction`.
     - `PredictedState() const` — the current predicted state (for rendering).
   - The AUTHORITY side: runs the roll-game sim + applies scripted SERVER-ONLY impulses at documented ticks
     (e.g. a fixed lateral impulse to the player at tick T), captures snapshots, sends via the BU
     `SimChannel`. The client `PredictTick`s every tick and `OnAuthoritative`s when the channel delivers a
     snapshot.

2. **Showcase `--netpredict-shot <out>` (Vulkan) / `--netpredict` (Metal).** Run authority (roll-game +
   scripted server-only impulses) + a `PredictedClient` over the scripted track through the BU channel
   (fixed latency so prediction is meaningfully ahead; optional small loss). At a FIXED render tick render
   the client's PREDICTED+RECONCILED scene (player + pickups), lit + shadowed. Print `netpredict:
   {predTicks:P, reconciles:R, maxMisprediction:M, finalError:E, converged:true}` where `maxMisprediction
   > 0` (a real misprediction occurred — server-only impulse) and `converged` asserts the reconciled
   client matches the authority's true state at the render tick within tolerance (`finalError` small). New
   golden `tests/golden/metal/netpredict.png` (Metal two runs DIFF 0.0000). Existing 47 image goldens
   UNTOUCHED.

3. **Determinism.** Scripted inputs + scripted server-only impulses + seeded BU channel + fixed latency →
   bit-stable. Two runs byte-identical. The reconciliation (rewind + replay) is deterministic.

4. **Tests `tests/prediction_test.cpp` (pure CPU, no GPU):**
   - **Prediction advances:** `PredictTick` moves the predicted state ahead under local input (position
     changes in the input direction), without waiting for any snapshot.
   - **Reconciliation with NO misprediction:** when the authority sim == the client's predicted sim (no
     server-only effects), `OnAuthoritative` leaves the predicted "now" UNCHANGED (replay reproduces the
     same state) — `lastMisprediction ≈ 0`.
   - **Reconciliation WITH misprediction:** inject a server-only impulse the client didn't predict; before
     reconcile the predicted state differs from authority; after `OnAuthoritative` + replay, the predicted
     "now" matches the authority's extrapolation of the corrected state (the correction is applied + inputs
     replayed) — `lastMisprediction > 0` and the result equals the expected corrected state.
   - **Input buffer:** acknowledged inputs (`tick <= authSnap.tick`) are dropped; unacknowledged are
     replayed in order; the buffer doesn't grow unbounded.
   - **Convergence/determinism:** over the full run, the reconciled client converges to authority
     (finalError within tolerance, maxMisprediction > 0); two runs identical.
   - Clean under `windows-msvc-asan`.

5. **Introspect.** Add exactly `client-prediction` (features) + `--netpredict-shot` (showcases).

## RHI seam additions (summary)
- **None.** Pure CPU prediction/reconciliation over the AX sim + BQ/BU; the render reuses the lit/shadowed
  scene path. New files (`engine/net/prediction.{h,cpp}`, `tests/prediction_test.cpp`) add ZERO backend
  symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Actual sockets, lag compensation / server-side rewind for hit detection, entity interpolation of REMOTE
players (BU covers remote-entity interpolation; this slice is LOCAL-player prediction), input redundancy/
ack reliability beyond the buffer, rollback netcode for many entities, delta-compressed input streams,
adaptive prediction window, multiple clients. One local-player predict + reconcile against a
server-authoritative sim with a scripted misprediction, golden + convergence-asserted.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 47) + new `prediction_test` (prediction advances,
   reconcile no-misprediction ≈0, reconcile WITH misprediction corrects, input-buffer ack/replay,
   convergence/determinism). Clean under `windows-msvc-asan`.
2. `--netpredict-shot` on Windows/Vulkan: controller visual review — the predicted+reconciled client scene
   (player + remaining pickups) renders coherently; the `netpredict: {... maxMisprediction:M>0,
   converged:true}` line is deterministic (two runs → byte-identical capture) AND maxMisprediction>0 (a
   real misprediction was corrected). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --netpredict` → new golden `tests/golden/metal/netpredict.png`; two runs DIFF
   0.0000; the stat line (incl. maxMisprediction + converged) matches Vulkan.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `netpredict.png` added;
   the other 47 byte-identical.
5. Introspect JSON rebaked exactly `+client-prediction` + `--netpredict-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `netpredict`
   image golden in the Mac round-trip loop.
