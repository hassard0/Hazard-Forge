# Slice PS5 — Deterministic Persistent Contacts: LOCKSTEP + ROLLBACK (the netcode headline) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #21
> (DETERMINISTIC WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, `hf::sim::persist`) — THE NETCODE HEADLINE. PS1-PS4
> built the contact key, the cache, the warm-started solver, and sleeping islands. PS5 proves the WHOLE
> warm+sleeping sim is **lockstep- and rollback-replayable**: two peers fed only an input-command stream warm-start,
> sleep, wake, and re-derive the entire world — bodies AND the persistent impulse cache AND the per-body sleep state
> — byte-identical, and a rollback re-sims from a snapshot bit-for-bit. The KEY difference from CX5/FC5: the
> replayable state now INCLUDES the cache + sleep state, not just the bodies. The moat sentence: **warm-started,
> sleeping rigid-body contacts that are lockstep-replayable.** PURE CPU (NO GPU shader, NO new RHI) → both backends
> run the IDENTICAL harness → bit-identical golden BY CONSTRUCTION (cross-vendor 0 px). PS1-PS4's `persist.h` code +
> CX/FC's `convex.h`/`fric.h` are BYTE-FROZEN (PS5 is additive). Branch: `slice-ps5`. See
> [[hazard-forge-persist-roadmap]].

**Goal:** Extend `engine/sim/persist.h` (additive — PS1-PS4 + fric.h byte-unchanged) with the lockstep harness over
`StepWarmSleepWorld`: `PersistSnapshot` (bodies + cache + sleep + tick) + `SnapshotPersist`/`RestorePersist` +
`SimPersistTick` + `PersistStatesEqual` + `RunPersistLockstep` + `RunPersistRollback`. Add `--persist-lockstep-shot`
(Vulkan) / `--persist-lockstep` (Metal) — both run the SAME pure-CPU harness. Bake the integer golden
`persist_lockstep`. **NO new shader, NO new RHI.**

## Design call: the pure-CPU lockstep harness over `StepWarmSleepWorld` (snapshot = bodies + cache + sleep)

`StepWarmSleepWorld` (PS4) is fully deterministic, but unlike the earlier sims its replayable state is THREE things:
the body world, the persistent impulse `cache` (PS2 — last tick's accumulated impulses), and the per-body
`SleepState[]` (PS4 — energy/quietTicks/asleep). A peer that only snapshotted the bodies would diverge (it would
warm-start from the wrong impulses / mis-time the sleep). So PS5's snapshot captures ALL THREE.
- **Commands — REUSE the frozen `convex::ConvexCommand`** (the warm world is a `convex::ConvexWorld`):
  `kConvexCmdAddImpulse` (a wake-impulse — strikes a body, raising its energy → wakes its island) /
  `kConvexCmdSetAngVel`. `convex::ApplyConvexCommands` applies them. (A wake-impulse on a sleeping body is exactly
  the PS4 wake event.)
- **`SimPersistTick(world, cache, sleep, cfg, commands, tick)`** — `convex::ApplyConvexCommands(world, commands,
  tick)` then `StepWarmSleepWorld(world, cache, sleep, cfg)` (the PS4 warm+sleep tick). ONE deterministic tick.
  (NOTE: a command that adds velocity to a body must take effect BEFORE the KE/sleep evaluation — apply commands
  first, then the step measures the now-energetic body and wakes it.)
- **`PersistSnapshot { std::vector<FxBody> bodies; PersistentCache cache; std::vector<SleepState> sleep; uint32_t
  tick; }`** — the FULL replayable state. `SnapshotPersist(world, cache, sleep, tick)` deep-copies all three;
  `RestorePersist(world, cache, sleep, snap)` restores all three.
- **`PersistStatesEqual(...)`** — byte-for-byte equality of the (bodies, cache.entries, sleep) triple (the
  make-or-break comparison — `ConvexBodiesEqual` for the bodies + a memcmp/field-compare for the cache entries +
  the sleep array). Two states are equal only if ALL THREE match.
- **`RunPersistLockstep(world0, cache0, sleep0, cfg, commands, ticks, outIdentical)`** → two peers (authority +
  replica) BOTH start from the same initial (world0, cache0, sleep0), BOTH run `SimPersistTick` for `ticks` with the
  SAME command stream; set `*outIdentical` to whether the two final (bodies, cache, sleep) triples are byte-identical
  (`PersistStatesEqual`) + return the converged authority state (for the golden).
- **`RunPersistRollback(world0, cache0, sleep0, cfg, authStream, mispredictStream, ticks, rollbackAt, ...)`** → the
  CX5/FC5 rollback control flow: advance to `rollbackAt`, `SnapshotPersist` (all three), speculatively mispredict
  (diverges), `RestorePersist` (all three), re-sim the correct stream; set `*outCorrectedEqAuthority` (corrected ==
  authority byte-for-byte over all three) + `*outMispredictDiverged` (the speculative state genuinely differed —
  proving a real divergence corrected, NOT a no-op). cfg + streams are CONSTANT, NOT snapshotted.

**The golden scene:** the PS4 warm+sleep stack + a deterministic command stream — a couple of early perturbations,
then the tower settles + SLEEPS, then a wake-impulse command at a fixed later tick wakes + topples it. Render the
converged authority world (the PS4 side-view). Both backends produce the identical image BY CONSTRUCTION. PURE CPU.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **PS4 `engine/sim/persist.h` (read it; APPEND only after `SleepMeasure`/`StepWarmSleepWorldN`):**
  `StepWarmSleepWorld`, `SleepState`, `SleepConfig`, `PersistentCache`. PS1-PS4 byte-frozen.
- **convex.h CX5 machinery (read-only — REUSE, do NOT redefine):** `convex::ConvexCommand`,
  `convex::kConvexCmdAddImpulse`/`kConvexCmdSetAngVel`, `convex::ApplyConvexCommands`, `convex::ConvexBodiesEqual`,
  `convex::ConvexWorld`. PS5's harness mirrors `convex::RunConvexLockstep`/`RunConvexRollback` (and PS5's prior
  `fric::RunFricLockstep`) with `StepWarmSleepWorld` swapped in + the cache+sleep added to the snapshot.
- **The showcase precedent:** FC5's `--fric-lockstep-shot`/`--fric-lockstep` (the pure-CPU harness + the 4 proofs +
  the converged-world render, NO GPU dispatch) in `samples/hello_triangle/main.cpp` +
  `metal_headless/visual_test.mm`. Mirror for `--persist-lockstep`.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/persist_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/persist.h`** (PS1-PS4 byte-frozen): `PersistSnapshot`, `SnapshotPersist`,
   `RestorePersist`, `PersistStatesEqual`, `SimPersistTick`, `RunPersistLockstep`, `RunPersistRollback`. Pure
   integer, FIXED command + peer order. **NO new shader, NO new RHI** (the seam incl. shaders — EMPTY; pure-CPU).
2. **Showcase `--persist-lockstep-shot <out>` (Vulkan) AND `--persist-lockstep` (Metal) — WIRE BOTH** (standalone
   arg-parse). BOTH run the IDENTICAL pure-CPU `RunPersistLockstep` + `RunPersistRollback` over the PS4 warm+sleep
   scene + a fixed command stream (settle → sleep → a wake-impulse topples it) — NO GPU dispatch. Render the
   converged authority world. Golden = `tests/golden/metal/persist_lockstep.png` (Mac-baked by the CONTROLLER — DO
   NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) lockstep:** authority == replica byte-for-byte ACROSS ALL THREE (bodies + cache + sleep). Print
     `persist-lockstep: {bodies:<N>, ticks:<K>, commands:<C>} authority==replica BIT-IDENTICAL (bodies+cache+sleep)`;
     assert.
   - **(2) determinism:** two runs → identical. Print `persist-lockstep determinism: two runs BYTE-IDENTICAL`.
   - **(3) rollback:** the corrected re-sim == the authority byte-for-byte (all three). Print `persist-lockstep
     rollback: corrected==authority BIT-EXACT`; assert.
   - **(4) mispredict real:** the mispredicted intermediate genuinely DIVERGED (in at least one of bodies/cache/
     sleep) before the rollback. Print `persist-lockstep mispredict: diverged before rollback (real divergence
     corrected)`; assert the divergence was non-zero.
   - **Golden discipline: ONLY `tests/golden/metal/persist_lockstep.png`; do NOT commit it.** Existing 201 image
     goldens UNTOUCHED.
4. **Cross-backend bar (PURE CPU → strict):** both backends run the identical harness → the golden is bit-identical
   BY CONSTRUCTION; cross-vendor ZERO differing pixels.
5. **Tests — APPEND to `tests/persist_test.cpp` (pure CPU):** `RunPersistLockstep` over the warm+sleep scene + a
   command stream → authority==replica (all three); `RunPersistRollback` → corrected==authority AND the mispredict
   diverged; `SnapshotPersist`/`RestorePersist` round-trips all three exactly; a snapshot taken while the tower is
   ASLEEP restores the sleep state so the replica stays asleep (the sleep state is part of the replayable state);
   two runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-persist-lockstep` (features) + `--persist-lockstep-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None — and NO new shader.** Pure CPU. `rhi.h` + backend dirs UNCHANGED. `engine/sim/convex.h` + `fric.h` +
  `fpx.h` + **PS1-PS4's persist.h code + ALL persist shaders (persist_key/cache/warm/sleep.comp)** + all other sim
  headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing shaders UNCHANGED. `persist.h`
  APPEND-only. Report the seam empty (only the persist.h APPEND + the showcase/test/introspect are new/changed; NO
  shaders/ change at all).

## Out of scope (YAGNI — PS6)
The lit 3D render capstone (PS6 — PS5's render is the 2D converged-state side-view). Real network transport. PS5
claims ONLY: the warm+sleeping contact sim is lockstep-deterministic (two peers converge from inputs alone, ALL
THREE of bodies+cache+sleep) and rollback-replayable (a snapshot re-sim is bit-exact), bit-identical
CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE: boxes only; the same within-band warm/sleep
caveats as PS3/PS4.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 107 incl. PS1-PS4's `persist_test` + the appended PS5 cases).
   Clean under `windows-msvc-asan` (build+run `persist_test` + `introspect_test`).
2. **proofs + visual:** `--persist-lockstep-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID. **VERIFY the image shows a coherent converged warm+sleep world (the settled/toppled stack).**
3. Metal: `visual_test --persist-lockstep` → new golden `tests/golden/metal/persist_lockstep.png`; two runs DIFF
   0.0000. **Confirm `visual_test.mm` in the diff; confirm NO shader added (pure CPU).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `persist_lockstep.png` added; the
   other 201 byte-identical. `git diff master --stat -- tests/golden` = ONLY `persist_lockstep.png` (metal) + the
   introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-persist-lockstep` + `--persist-lockstep-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fric.h`/`fpx.h` + **PS1-PS4's persist.h code + ALL
   persist shaders** + ALL other sim headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing
   shaders byte-unchanged; **NO shaders/ change at all — pure CPU**). `scripts/verify.ps1` updated:
   `persist_lockstep` golden in the Mac loop + `--persist-lockstep-shot` in `$vkShots`.
