# Slice VD5 — Deterministic Gameplay/Netcode: WHOLE-WORLD LOCKSTEP + ROLLBACK (the netcode headline) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP
> #27 (DETERMINISTIC GAMEPLAY / NETCODE PRODUCT LAYER, `hf::game::verdict`). VD1-VD4 built the entity world, the
> gameplay systems, the composed `StepWorld` tick (gameplay + the frozen physics sim), and the heterogeneous
> snapshot/restore. VD5 is THE PRODUCT CLAIM: the ENTIRE game world — entities + gameplay rules + a physics sim —
> is **lockstep- and rollback-replayable** from an input stream alone, bit-identical CPU/Vulkan/Metal. Two peers
> fed only commands re-derive the whole world byte-for-byte; a rollback re-sims from a snapshot bit-for-bit, and a
> misprediction that diverges across BOTH gameplay (a spawned/despawned entity, a score) AND physics (a perturbed
> body) is corrected. This is the artifact UE5 cannot ship: a deterministic, cross-platform-bit-identical,
> rollback-replayable *game* world (its Chaos physics is float/non-deterministic and its gameplay is not
> lockstep-deterministic). It falls out by retargeting the WH5 lockstep harness over `StepWorld` + the VD4
> snapshot/restore — PURE CPU, no shader, no RHI → both backends run the identical harness → bit-identical BY
> CONSTRUCTION (cross-vendor 0 px). APPEND to `engine/game/verdict.h` (VD1-VD4 + warmhull/ecs BYTE-FROZEN). Branch:
> `slice-vd5`. See [[hazard-forge-verdict-roadmap]], [[hazard-forge-warmhull-roadmap]], [[hazard-forge-docs-style]],
> [[hazard-forge-metal-showcase-gate]].

**Goal:** Extend `engine/game/verdict.h` (additive — VD1-VD4 byte-unchanged) with `SimVerdictTick` (=`StepWorld`,
the WH5 `SimWarmHullTick` analog), `RunVerdictLockstep` (two peers from the SAME initial world + input stream →
`VerdictStatesEqual` over the whole world), and `RunVerdictRollback` (advance → `SnapshotWorld` → speculatively
mispredict → `RestoreWorld` → re-sim with the correct stream → equals authority). **THE NON-COPYABLE CONSTRAINT:**
`VerdictWorld` is NOT copyable (the `ecs::Registry` holds `unique_ptr` pools), so the two peers CANNOT be made by
copying `world0` — they are cloned via the VD4 `SnapshotWorld`/`RestoreWorld` (the determinism-faithful clone), OR
built by a `WorldBuilder` callback that constructs a fresh world deterministically. The harness takes an initial
`VerdictSnapshot` (or a builder) and restores it into each peer. Add the showcase `--vd5-net-shot` (Vulkan) /
`--vd5-net` (Metal) — both run the pure-CPU harness over the composed gameplay+physics scene + a command stream, and
render the converged authority world. Bake the integer golden `vd5_net`. **NO new shader, NO new RHI.**

## Design call: the WH5 harness over the whole world; peers cloned via the VD4 snapshot (not a copy)

`StepWorld` (VD3) is a deterministic integer tick; the whole world (entities + components + the sim TRIPLE) is
exactly what VD4 snapshots/restores/compares. So VD5 is the WH5 twin with the step = `StepWorld` and the
snapshot/equality = the VD4 ones, with ONE structural difference forced by the non-copyable world:
- **Cloning peers (the non-copyable fix):** a `VerdictSnapshot world0Snap` (taken from the initial world) is
  `RestoreWorld`'d into each fresh peer (`authority`, `replica`). This is the determinism-faithful clone — the
  restored peers are `VerdictStatesEqual` to `world0`. (Alternatively a `WorldBuilder` callback that builds the
  identical initial world; the snapshot path is preferred — it reuses VD4 and is provably faithful.)
- **`SimVerdictTick(world, cache..., cfg, commands, tick)`** = `StepWorld(world, commands, tick, hazard, player,
  collectRadius, cfg)` (the VD3 composed tick — the gameplay verbs + the lowered sim verbs in one bus).
- **`RunVerdictLockstep(world0Snap, params, commands, ticks, outIdentical)`** → restore `world0Snap` into BOTH
  `authority` + `replica`, BOTH run `SimVerdictTick` for `ticks` with the SAME command stream → `*outIdentical =
  VerdictStatesEqual(authority, replica)` (the WHOLE world). Return the converged authority (or its snapshot — the
  caller renders it).
- **`RunVerdictRollback(world0Snap, params, authStream, mispredictStream, ticks, rollbackAt,
  outCorrectedEqAuthority, outMispredictDiverged)`** → the VD4/WH5 rollback control flow over `SimVerdictTick`:
  restore `world0Snap`, advance to `rollbackAt`, `SnapshotWorld` (the VD4 heterogeneous snapshot), speculatively
  mispredict (≤3 ticks with a WRONG stream that diverges across gameplay AND physics — e.g. a mis-spawned entity +
  a wrong `kCmdImpulse`), `RestoreWorld`, re-sim the correct stream; set corrected==authority (`VerdictStatesEqual`)
  + mispredict-diverged flags.

**The golden scene:** the VD3 gameplay+physics scene + a deterministic command stream (a `kCmdImpulse` nudges the
player body; a `kCmdSpawn` adds an entity; gameplay collects a pickup). Render the converged authority world (the
VD3-style pure-integer composed view). PURE CPU (no GPU dispatch → no TDR).

## Reuse map (file:line)
- **VD1-VD4 `engine/game/verdict.h` (APPEND after `VerdictStatesEqual`):** `StepWorld`/`StepWorldN`,
  `VerdictWorld`, `Command`, `SnapshotWorld`/`RestoreWorld`/`VerdictSnapshot`/`VerdictStatesEqual`/
  `VerdictSnapshotsEqual`, the components, `HazardRegion`. VD1-VD4 byte-frozen.
- **warmhull.h / manifold.h (read-only — the WH5 control-flow SHAPE):** `warmhull::RunWarmHullLockstep`/
  `RunWarmHullRollback` (the harness to mirror with `StepWorld` + the VD4 snapshot), `warmhull::HullSleepConfig`.
- **The showcase precedent:** WH5 `--wh5-lockstep` / MF5 (the pure-CPU 4-proof + converged-world render). Mirror
  for `--vd5-net`. The VD3/VD4 composed-world render is the render base.
- **Registration:** `scripts/verify.ps1` (`vd5_net` + `--vd5-net-shot`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — verify it stages as ` M `**), append to
  `tests/verdict_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/game/verdict.h`** (VD1-VD4 byte-frozen): `SimVerdictTick`, `RunVerdictLockstep`,
   `RunVerdictRollback` (+ a small `VerdictParams` bundling the `StepWorld` scene knobs — `hazard`, `player`,
   `collectRadius`, `cfg` — so the harness signatures stay clean). Pure integer, FIXED command + peer order. The
   peers are cloned via `RestoreWorld(world0Snap)` (NOT a copy — `VerdictWorld` is non-copyable). **NO new shader,
   NO new RHI.**
2. **Showcase `--vd5-net-shot <out>` (Vulkan) AND `--vd5-net` (Metal) — WIRE BOTH (grep your own `visual_test.mm`
   for `--vd5-net` BEFORE reporting DONE).** BOTH run the IDENTICAL pure-CPU `RunVerdictLockstep` +
   `RunVerdictRollback` over the composed scene + a fixed command stream — NO GPU dispatch. Render the converged
   authority world. Golden = `tests/golden/metal/vd5_net.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1)** `vd5-net: {entities:<N>, bodies:<B>, ticks:<K>, commands:<C>} authority==replica BIT-IDENTICAL
     (whole world)` — the two peers' WHOLE world (entities + components + sim) is byte-equal from inputs alone.
   - **(2)** `vd5-net determinism: two runs BYTE-IDENTICAL`.
   - **(3)** `vd5-net rollback: corrected==authority BIT-EXACT (whole world)`.
   - **(4) THE HETEROGENEOUS DIVERGENCE:** `vd5-net mispredict: diverged across gameplay+physics, corrected` — the
     mispredicted pre-rollback world differs from the authority in BOTH a gameplay component (a spawned/despawned
     entity / a score) AND a physics body (a perturbed `pos`), and the rollback corrects it. **Proves the whole
     heterogeneous world is replayable, not just the bodies.**
   - Golden discipline: ONLY `tests/golden/metal/vd5_net.png`; do NOT commit it. Existing 237 goldens UNTOUCHED.
4. **Cross-backend bar (PURE CPU → strict):** both backends run the identical harness → bit-identical BY
   CONSTRUCTION; cross-vendor ZERO differing pixels (the strict-zero integer render, the VD1-VD4 lineage).
5. **Tests — APPEND to `tests/verdict_test.cpp` (pure CPU):** `RunVerdictLockstep` authority==replica (whole
   world); `RunVerdictRollback` corrected==authority AND the mispredict diverged across gameplay AND physics; two
   runs byte-identical; the peers are cloned via the VD4 snapshot (NOT a copy — assert the restored peers are
   `VerdictStatesEqual` to the initial world); a command stream that genuinely moves BOTH the gameplay state (a
   collect / a spawn) AND the sim (an impulse). Clean under `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `verdict-lockstep` (features) + `--vd5-net-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does (and verifies it stages).**

## RHI seam additions (summary)
- **None — and NO new shader.** Pure CPU. `engine/game/verdict.h` APPEND-only (VD1-VD4 frozen); warmhull.h/ecs.h +
  ALL sim headers + ALL existing shaders UNCHANGED. Report the seam empty (only the verdict.h APPEND + the
  showcase/test/introspect are new/changed; NO shaders/ change at all).

## Out of scope (YAGNI — VD6)
The playable lit-3D render capstone (VD6). Real network transport / wire serialization (VD5 is the in-memory
lockstep + rollback; the wire format + the net trilogy's prediction/reconciliation adapter onto this Q16.16 world
are deferred — VD5 provides the cross-platform-bit-identical substrate the net trilogy always wanted). Composing
more than one sim. VD5 claims ONLY: the whole composed game world (entities + gameplay + the physics sim) is
lockstep-deterministic (two peers converge from inputs alone) and rollback-replayable (the heterogeneous snapshot
restores + re-sims bit-for-bit, correcting a divergence that spans gameplay AND physics), bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the four proofs. CAVEATS inherited: the WH4 within-band settle /
demonstrated-N stability of the embedded sim.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "verdict|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--vd5-net-shot` on Vulkan: the 4 proofs + exit 0. VERIFY a coherent converged world. (PURE
   CPU — no GPU compute, no TDR/VUID risk from this slice.)
3. Metal: `visual_test --vd5-net` → `tests/golden/metal/vd5_net.png`; two runs DIFF 0.0000. **Confirm `--vd5-net`
   wired in `visual_test.mm` (grep it) BEFORE the Mac bake** — NO shader added (pure CPU). Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `vd5_net.png` added; the other 237 byte-identical (+ controller introspect rebake,
   verified staged as ` M `).
5. Introspect: exactly `+verdict-lockstep` + `--vd5-net-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + VD1-VD4 verdict.h code + warmhull.h/ecs.h + ALL sim headers + ALL shaders
   byte-unchanged; verdict.h APPEND-only; NO shaders/ change). `vd5_net` in the Mac loop + `--vd5-net-shot` in
   `$vkShots`.
