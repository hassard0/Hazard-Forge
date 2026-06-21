# Slice WH5 — Warm-Started Hull Contacts: LOCKSTEP + ROLLBACK over the warm+sleep triple (the netcode headline) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of
> FLAGSHIP #26 (WARM-STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, `hf::sim::warmhull`). WH1-WH4 built
> the feature ID, the cache, the warm solver, and the sleeping-island STABLE STACK (a deterministic N=4 hull tower
> the frozen #25 step topples). WH5 proves the whole warm+sleep stack sim is **lockstep- and
> rollback-replayable** — the netcode payoff. THE NEW WRINKLE vs the earlier lockstep slices (GJ5/CD5/MF5): the
> replayable state is now a TRIPLE — the body world PLUS the persistent impulse cache (`HullCache`) PLUS the
> per-body sleep state (`HullSleepState[]`). A correct rollback must snapshot and restore ALL THREE, or a peer
> that mispredicts and rolls back would resume with the wrong warm-start impulses or the wrong sleep timers and
> diverge. This is the persist.h PS5 lesson (the box version snapshots bodies+cache+sleep) generalized to hulls.
> Because `StepWarmSleepHullWorld` is a pure deterministic integer function whose mutable state is exactly that
> triple, this falls out by retargeting the frozen lockstep harness over it — PURE CPU, no shader, no RHI → both
> backends run the IDENTICAL harness → the golden is bit-identical BY CONSTRUCTION (cross-vendor 0 px). APPEND to
> `engine/sim/warmhull.h` (WH1-WH4 + manifold/gjk/persist/etc BYTE-FROZEN). Branch: `slice-wh5`. See
> [[hazard-forge-warmhull-roadmap]], [[hazard-forge-manifold-roadmap]], [[hazard-forge-docs-style]],
> [[hazard-forge-metal-showcase-gate]].

**Goal:** Extend `engine/sim/warmhull.h` (additive — WH1-WH4 byte-unchanged) with `SimWarmHullTick`
(`gjk::ApplyHullCommands` + `StepWarmSleepHullWorld`), `WarmHullSnapshot` (the TRIPLE: a deep copy of the body
vector + the `HullCache` + the `HullSleepState[]`), `SnapshotWarmHull` / `RestoreWarmHull`, the equality helpers
`HullCacheEntriesEqual` / `HullSleepStatesEqual` / `WarmHullStatesEqual`, and `RunWarmHullLockstep` /
`RunWarmHullRollback`, reusing the frozen `convex::ConvexCommand` + `gjk::ApplyHullCommands`. Add the showcase
`--wh5-lockstep-shot` (Vulkan) / `--wh5-lockstep` (Metal) — both run the SAME pure-CPU harness over the WH4 tower
scene + a command stream, and render the converged authority world. Bake the integer golden `wh5_lockstep`. **NO
new shader, NO new RHI.**

## Design call: the pure-CPU lockstep harness over the TRIPLE (the GJ5/MF5 twin, PS5-extended)

`StepWarmSleepHullWorld` (WH4) is a fully deterministic integer tick; its mutable replayable state is the TRIPLE
(bodies + cache + sleep) — the hulls are immutable, and the manifolds/inertia are re-derived. So WH5 is the GJ5/MF5
twin with TWO changes: the step is `StepWarmSleepHullWorld`, and the snapshot is the TRIPLE (not bodies-only).
- **Commands — REUSE frozen `convex::ConvexCommand` + `gjk::ApplyHullCommands`** (the WH-line / MF5 verbatim). A
  command can NUDGE the tower (an add-impulse that wakes the island) — making the wake/re-settle the lockstep
  scene. `SimWarmHullTick(world, cache, sleep, cfg, commands, tick)` = `gjk::ApplyHullCommands(world, commands,
  tick)` then `StepWarmSleepHullWorld(world, cache, sleep, cfg)`.
- **`WarmHullSnapshot` (the TRIPLE):** `{ std::vector<FxBody> bodies; HullCache cache; std::vector<HullSleepState>
  sleep; uint32_t tick; }`. `SnapshotWarmHull` deep-copies all three; `RestoreWarmHull` restores all three. (A
  bodies-only snapshot would mis-warm-start / mis-sleep on rollback — the PS5 correctness point.)
- **Equality — `HullCacheEntriesEqual`** (the cache entries byte-equal, FIXED order), **`HullSleepStatesEqual`**
  (the sleep states byte-equal), **`WarmHullStatesEqual`** = `gjk::HullBodiesEqual` AND `HullCacheEntriesEqual` AND
  `HullSleepStatesEqual` (the TRIPLE byte-equal — the make-or-break comparison).
- **`RunWarmHullLockstep(world0, cfg, commands, ticks, outIdentical)`** → two peers (authority + replica) BOTH from
  `world0` (with fresh empty cache + sleep), BOTH run `SimWarmHullTick` for `ticks` with the SAME command stream →
  `*outIdentical = WarmHullStatesEqual(...)` (the TRIPLE). Return the converged authority world. (Mirror
  `manifold::RunHullLockstepHardened`, with the triple equality.)
- **`RunWarmHullRollback(world0, cfg, authStream, mispredictStream, ticks, rollbackAt, outCorrectedEqAuthority,
  outMispredictDiverged)`** → the MF5/PS5 rollback control flow over `SimWarmHullTick`: advance to `rollbackAt`,
  `SnapshotWarmHull` (the TRIPLE), speculatively mispredict (≤3 ticks, diverges), `RestoreWarmHull` (the TRIPLE),
  re-sim the correct stream; set corrected==authority (the TRIPLE) + mispredict-diverged flags. cfg + streams
  CONSTANT.

**The golden scene:** the WH4 tower + a deterministic command stream (an impulse wakes the asleep tower; it
re-settles + re-sleeps; a later command perturbs). Render the converged authority world (the WH4-style pure-integer
side-view → strict-zero cross-vendor). PURE CPU (no GPU dispatch → no TDR).

## Reuse map (file:line)
- **WH1-WH4 `engine/sim/warmhull.h` (APPEND after `MeasureHullSleep`):** `StepWarmSleepHullWorld`/`…N`,
  `HullCache`, `HullSleepState`, `HullSleepConfig`. WH1-WH4 frozen.
- **manifold.h / gjk.h (read-only — REUSE verbatim):** `manifold::RunHullLockstepHardened`/`RunHullRollbackHardened`
  (the SHAPE to mirror with the step + triple swap), `gjk::ApplyHullCommands`/`HullBodiesEqual`/`HullSnapshot`
  (bodies equality + the snapshot idiom), `convex::ConvexCommand`/`ConvexBodiesEqual`.
- **persist.h (read-only — the PS5 TRIPLE template):** `persist::PersistSnapshot` (the bodies+cache+sleep triple
  shape, the snapshot/restore/equality over all three — the proof that the cache + sleep MUST be snapshotted).
- **The showcase precedent:** MF5 `--mf5-lockstep` / GJ5 / CD5 (the pure-CPU 4-proof + converged-world render, NO
  GPU dispatch). Mirror for `--wh5-lockstep`.
- **Registration:** `scripts/verify.ps1` (`wh5_lockstep` + `--wh5-lockstep-shot`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes**), append to `tests/warmhull_test.cpp`. (No shader.)

## Design decisions (locked)
1. **APPEND to `engine/sim/warmhull.h`** (WH1-WH4 byte-frozen): `SimWarmHullTick`, `WarmHullSnapshot`,
   `SnapshotWarmHull`, `RestoreWarmHull`, `HullCacheEntriesEqual`, `HullSleepStatesEqual`, `WarmHullStatesEqual`,
   `RunWarmHullLockstep`, `RunWarmHullRollback`. Pure integer, FIXED command + peer order. **NO new shader, NO new
   RHI** (the seam incl. shaders — EMPTY).
2. **Showcase `--wh5-lockstep-shot <out>` (Vulkan) AND `--wh5-lockstep` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--wh5-lockstep` BEFORE reporting DONE).** BOTH run the IDENTICAL pure-CPU
   `RunWarmHullLockstep` + `RunWarmHullRollback` over the WH4 tower + a fixed command stream — NO GPU dispatch.
   Render the converged authority world. Golden = `tests/golden/metal/wh5_lockstep.png` (Mac-baked by the
   CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1)** `wh5-lockstep: {bodies:<N>, ticks:<K>, commands:<C>} authority==replica BIT-IDENTICAL (triple)` — the
     two peers' TRIPLE (bodies+cache+sleep) is byte-equal.
   - **(2)** `wh5-lockstep determinism: two runs BYTE-IDENTICAL`.
   - **(3)** `wh5-lockstep rollback: corrected==authority BIT-EXACT (triple)` — restore-the-triple + re-sim ==
     authority over all three.
   - **(4)** `wh5-lockstep mispredict: diverged before rollback (triple) (real divergence corrected)`.
   - Golden discipline: ONLY `tests/golden/metal/wh5_lockstep.png`; do NOT commit it. Existing 231 goldens
     UNTOUCHED.
4. **Cross-backend bar (PURE CPU → strict):** both backends run the identical harness → bit-identical BY
   CONSTRUCTION; cross-vendor ZERO differing pixels (the strict-zero integer render, the WH4 lineage).
5. **Tests — APPEND to `tests/warmhull_test.cpp` (pure CPU):** `RunWarmHullLockstep` authority==replica (triple);
   `RunWarmHullRollback` corrected==authority (triple) AND mispredict diverged; two runs byte-identical; **a
   bodies-ONLY snapshot/restore (omitting the cache or sleep) makes the rollback DIVERGE** (the PS5 proof that the
   triple is necessary — restore only bodies, show corrected != authority; then restore the triple, show ==); a
   command stream that wakes the asleep tower (the sleep state genuinely exercised). Clean under `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `warmhull-lockstep` (features) + `--wh5-lockstep-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** Pure CPU. `engine/sim/warmhull.h` APPEND-only (WH1-WH4 frozen); manifold.h/gjk.h/
  persist.h + ALL other sim headers + ALL existing shaders UNCHANGED. Report the seam empty (only the warmhull.h
  APPEND + the showcase/test/introspect are new/changed; NO shaders/ change at all).

## Out of scope (YAGNI — WH6)
The render capstone polish (WH6). Real network transport. WH5 claims ONLY: the warm+sleep stack sim is
lockstep-deterministic (two peers converge from inputs alone over the TRIPLE) and rollback-replayable (the triple
snapshot/restore re-sims bit-for-bit), bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs +
the bodies-only-diverges necessity proof. CAVEATS inherited: the WH4 within-band settle / demonstrated-N stability.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "warmhull|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--wh5-lockstep-shot` on Vulkan: the 4 proofs + exit 0. VERIFY a coherent converged world
   (the settled tower). (PURE CPU — no GPU compute, no TDR/VUID risk from this slice.)
3. Metal: `visual_test --wh5-lockstep` → `tests/golden/metal/wh5_lockstep.png`; two runs DIFF 0.0000. **Confirm
   `--wh5-lockstep` wired in `visual_test.mm` (grep it) BEFORE the Mac bake** — NO shader added (pure CPU).
   Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `wh5_lockstep.png` added; the other 231 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+warmhull-lockstep` + `--wh5-lockstep-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + WH1-WH4 warmhull.h code + manifold.h/gjk.h/persist.h + ALL other sim headers + ALL
   existing shaders byte-unchanged; warmhull.h APPEND-only; NO shaders/ change). `wh5_lockstep` in the Mac loop +
   `--wh5-lockstep-shot` in `$vkShots`.
