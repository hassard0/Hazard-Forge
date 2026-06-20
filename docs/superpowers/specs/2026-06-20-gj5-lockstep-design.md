# Slice GJ5 — General Convex-Hull Contacts: LOCKSTEP + ROLLBACK (the netcode beat) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #22 (DETERMINISTIC
> GENERAL CONVEX-HULL CONTACTS via integer GJK + EPA, `hf::sim::gjk`). GJ1-GJ4 built the narrowphase trio + the
> general-hull world step (`StepHullWorld`) — arbitrary convex polyhedra settle bit-exact CPU↔Vulkan↔Metal. GJ5
> proves the whole hull sim is **lockstep- and rollback-replayable**: two peers fed only an input-command stream
> re-derive the entire hull world byte-identical, and a rollback re-sims from a snapshot bit-for-bit. Because
> `StepHullWorld` is a pure, deterministic integer function, this falls out by retargeting the frozen CX5
> lockstep harness over `HullWorld` — PURE CPU, no GPU shader, no new RHI → both backends run the IDENTICAL
> harness → the golden is bit-identical BY CONSTRUCTION (cross-vendor 0 px). APPEND to `engine/sim/gjk.h`
> (GJ1-GJ4 + convex.h/fric.h/persist.h/fpx.h BYTE-FROZEN). Branch: `slice-gj5`. See [[hazard-forge-gjk-roadmap]],
> [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/gjk.h` (additive — GJ1-GJ4 byte-unchanged) with `ApplyHullCommands` (reusing the
frozen `convex::ConvexCommand` over `HullWorld.bodies`) + `SimHullTick` + `HullSnapshot` +
`SnapshotHull`/`RestoreHull` + `HullBodiesEqual` + `RunHullLockstep` + `RunHullRollback`. Add the showcase
`--gjk-lockstep-shot` (Vulkan) / `--gjk-lockstep` (Metal) — both run the SAME pure-CPU harness. Bake the integer
golden `gjk_lockstep`. **NO new shader, NO new RHI.**

## Design call: the pure-CPU lockstep harness over `StepHullWorld` (the CX5/FC5/PS5 twin)

`StepHullWorld` (GJ4) is a fully deterministic integer tick whose only mutable state is the `bodies` vector (the
`hulls` are immutable/shared geometry, like a `convex::ConvexWorld`'s `boxes`). So GJ5 is the direct CX5 twin —
the same command/snapshot/lockstep/rollback shapes `convex::RunConvexLockstep`/`RunConvexRollback` (and the
later `fric::RunFricLockstep` / `persist::RunPersistLockstep`) use, with `StepConvexWorld`→`StepHullWorld` swapped.
- **Commands — REUSE the frozen `convex::ConvexCommand`** (a hull body is an `fpx::FxBody`, identical to a convex
  body): `convex::kConvexCmdAddImpulse` (a launch impulse) / `convex::kConvexCmdSetAngVel` (a spin). Add a thin
  `ApplyHullCommands(HullWorld& world, const std::vector<convex::ConvexCommand>& cmds, uint32_t tick)` that
  applies them to `world.bodies` (the same per-command logic as the frozen `convex::ApplyConvexCommands`, which
  takes a `ConvexWorld&` and so can't be called on a `HullWorld` directly — reproduce its body verbatim over
  `world.bodies`, including the out-of-range-bodyId guard).
- **`SimHullTick(world, cfg, commands, tick)`** — `ApplyHullCommands(world, commands, tick)` then
  `StepHullWorld(world, cfg)`. ONE deterministic tick (commands BEFORE the step so an impulse integrates this
  tick).
- **`HullSnapshot { std::vector<fpx::FxBody> bodies; uint32_t tick; }`** — the replayable state is the body
  vector only (hulls immutable). `SnapshotHull(world, tick)` deep-copies `bodies`; `RestoreHull(world, snap)`
  restores them (leaving `hulls` untouched). (The `convex::ConvexSnapshot` analog.)
- **`HullBodiesEqual(a, b)`** — byte/field equality of two `std::vector<FxBody>` (the make-or-break comparison —
  reproduce `convex::ConvexBodiesEqual`'s field compare over the FxBody POD; or, if `convex::ConvexBodiesEqual`
  is callable on a `std::vector<FxBody>&` directly, REUSE it — check its signature and prefer reuse).
- **`RunHullLockstep(world0, cfg, commands, ticks, outIdentical)`** → two peers (authority + replica) BOTH start
  from `world0`, BOTH run `SimHullTick` for `ticks` with the SAME command stream; set `*outIdentical` to whether
  the two final body vectors are byte-identical (`HullBodiesEqual`) + return the converged authority world (for
  the golden render). The peer step order is PINNED.
- **`RunHullRollback(world0, cfg, authStream, mispredictStream, ticks, rollbackAt, outCorrectedEqAuthority,
  outMispredictDiverged)`** → the CX5 rollback control flow: advance to `rollbackAt`, `SnapshotHull`,
  speculatively mispredict (≤3 ticks, diverges), `RestoreHull`, re-sim the correct stream; set
  `*outCorrectedEqAuthority` (corrected == authority byte-for-byte) + `*outMispredictDiverged` (the speculative
  state genuinely differed — proving a real divergence corrected, NOT a no-op). cfg + streams are CONSTANT, NOT
  snapshotted.

**The golden scene:** the GJ4 settle scene (floor + tetra + octa + wedge + static box) + a deterministic command
stream — a couple of launch-impulses that knock the hulls around, then they re-settle. Render the converged
authority world (the GJ4 side-view). Both backends produce the identical image BY CONSTRUCTION. PURE CPU.

> NOTE on TDR (the GJ4 lesson, [[hazard-forge-gpu-tdr-chunking]]): GJ5 is PURE CPU — NO GPU dispatch — so the TDR
> chunking is N/A here. (The lockstep harness runs entirely on the CPU on both backends.)

## Reuse map (file:line — the implementer MUST ground these before coding)
- **GJ4 `engine/sim/gjk.h` (read it; APPEND only after `MeasureHullStack`, before the namespace close):**
  `HullWorld`, `StepHullWorld`, `MeasureHullStack`. GJ1-GJ4 byte-frozen.
- **convex.h CX5 machinery (read-only — REUSE, do NOT redefine):** `convex::ConvexCommand`,
  `convex::kConvexCmdAddImpulse`/`kConvexCmdSetAngVel`, `convex::ApplyConvexCommands` (the body to reproduce over
  `HullWorld.bodies`), `convex::ConvexSnapshot`/`SnapshotConvex`/`RestoreConvex` (the snapshot shape),
  `convex::ConvexBodiesEqual` (REUSE if it takes a `std::vector<FxBody>&`), `convex::RunConvexLockstep`/
  `RunConvexRollback` (the control flow to mirror). GJ5's harness mirrors these with `StepHullWorld` swapped in.
- **The showcase precedent:** `convex::`'s `--convex-lockstep-shot` / `--convex-lockstep` (and the later
  `--fric-lockstep` / `--persist-lockstep`) — the pure-CPU harness + the 4 proofs + the converged-world render,
  NO GPU dispatch. Mirror for `--gjk-lockstep`.
- **Registration:** `scripts/verify.ps1` (append `gjk_lockstep` + `--gjk-lockstep-shot` to `$vkShots`),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT**),
  append to `tests/gjk_test.cpp`. (No shader → nothing to add to `hf_gen_msl`.)

## Design decisions (locked)
1. **APPEND to `engine/sim/gjk.h`** (GJ1-GJ4 byte-frozen): `ApplyHullCommands`, `SimHullTick`, `HullSnapshot`,
   `SnapshotHull`, `RestoreHull`, `HullBodiesEqual` (or reuse `convex::ConvexBodiesEqual`), `RunHullLockstep`,
   `RunHullRollback`. Pure integer, FIXED command + peer order. **NO new shader, NO new RHI** (the seam incl.
   shaders — EMPTY; pure-CPU).
2. **Showcase `--gjk-lockstep-shot <out>` (Vulkan) AND `--gjk-lockstep` (Metal) — WIRE BOTH** (standalone
   arg-parse). BOTH run the IDENTICAL pure-CPU `RunHullLockstep` + `RunHullRollback` over the GJ4 settle scene +
   a fixed command stream (launch-impulses → re-settle) — NO GPU dispatch. Render the converged authority world
   (the GJ4 side-view). Golden = `tests/golden/metal/gjk_lockstep.png` (Mac-baked by the CONTROLLER — DO NOT
   commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) lockstep:** authority == replica byte-for-byte. Print `gjk-lockstep: {bodies:<N>, ticks:<K>,
     commands:<C>} authority==replica BIT-IDENTICAL`; assert.
   - **(2) determinism:** two runs → identical. Print `gjk-lockstep determinism: two runs BYTE-IDENTICAL`.
   - **(3) rollback:** the corrected re-sim == the authority byte-for-byte. Print `gjk-lockstep rollback:
     corrected==authority BIT-EXACT`; assert.
   - **(4) mispredict real:** the mispredicted intermediate genuinely DIVERGED before the rollback. Print
     `gjk-lockstep mispredict: diverged before rollback (real divergence corrected)`; assert the divergence was
     non-zero.
   - **Golden discipline: ONLY `tests/golden/metal/gjk_lockstep.png`; do NOT commit it.** Existing 207 image
     goldens UNTOUCHED.
4. **Cross-backend bar (PURE CPU → strict):** both backends run the identical harness → the golden is
   bit-identical BY CONSTRUCTION; cross-vendor ZERO differing pixels.
5. **Tests — APPEND to `tests/gjk_test.cpp` (pure CPU):** `RunHullLockstep` over the settle scene + a command
   stream → authority==replica; `RunHullRollback` → corrected==authority AND the mispredict diverged;
   `SnapshotHull`/`RestoreHull` round-trips the bodies exactly; two runs byte-identical; a command stream
   actually moved the hulls non-trivially (the scene is not a frozen no-op). Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-hull-lockstep` (features) + `--gjk-lockstep-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None — and NO new shader.** Pure CPU. `rhi.h` + backend dirs UNCHANGED. `engine/sim/gjk.h` APPEND-only
  (GJ1-GJ4 frozen); convex.h/fric.h/persist.h/fpx.h + ALL other sim headers + ALL existing shaders +
  `engine/physics/`/`nav/`/`anim/` UNCHANGED. Report the seam empty (only the gjk.h APPEND + the
  showcase/test/introspect are new/changed; NO shaders/ change at all).

## Out of scope (YAGNI — GJ6)
The lit 3D render capstone (GJ6 — GJ5's render is the GJ4 2D side-view of the converged state). Real network
transport. GJ5 claims ONLY: the general-hull sim is lockstep-deterministic (two peers converge from inputs alone)
and rollback-replayable (a snapshot re-sim is bit-exact), bit-identical CPU↔Vulkan↔Metal, with the integer golden
+ the four proofs. NOTE: convex polyhedra only; the GJ2-GJ4 within-band caveats inherited.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 108 incl. the appended GJ5 `gjk_test` cases). Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--gjk-lockstep-shot` on Vulkan: the 4 proofs + exit 0. **VERIFY the image shows a
   coherent converged hull world (the re-settled scene).** (No GPU compute → no VUID risk from this slice, but
   the render path still goes through Vulkan — confirm no validation regression.)
3. Metal: `visual_test --gjk-lockstep` → new golden `tests/golden/metal/gjk_lockstep.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm NO shader added (pure CPU).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `gjk_lockstep.png` added; the
   other 207 byte-identical. `git diff master --stat -- tests/golden` = ONLY `gjk_lockstep.png` (metal) + the
   introspect json (controller rebake).
5. Introspect: exactly `+deterministic-hull-lockstep` + `--gjk-lockstep-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + GJ1-GJ4 gjk.h code + convex.h/fric.h/persist.h/fpx.h + ALL other sim headers + ALL
   existing shaders byte-unchanged; gjk.h APPEND-only; **NO shaders/ change at all — pure CPU**).
   `scripts/verify.ps1` updated: `gjk_lockstep` golden in the Mac loop + `--gjk-lockstep-shot` in `$vkShots`.
