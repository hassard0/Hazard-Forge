# Slice VD3 — Deterministic Gameplay/Netcode: COMPOSING THE PHYSICS SUBSYSTEM — ONE WORLD TICK — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP
> #27 (DETERMINISTIC GAMEPLAY / NETCODE PRODUCT LAYER, `hf::game::verdict`). VD1 built the entity world + command
> bus; VD2 the gameplay systems on a pinned schedule. VD3 is the slice that makes it a *physics game*: ONE
> deterministic `StepWorld` tick that runs the gameplay systems AND drives the embedded Q16.16 physics sim (the
> frozen `warmhull` warm+sleep hull world from flagship #26) in a single pinned order, bridged by a deterministic
> bidirectional `BodyRef` sync. THE NEW CAPABILITY: gameplay (entities/components/rules) + a real physics sim
> advancing together, bit-identical CPU/Vulkan/Metal — the first time gameplay LOGIC and a deterministic PHYSICS
> sim are composed in one tick. The frozen `warmhull::StepWarmSleepHullWorld` is reused VERBATIM; VD3 only adds the
> sync + the schedule. PURE CPU integer (strict 0px). APPEND to `engine/game/verdict.h` (VD1/VD2 + warmhull/gjk/sim
> BYTE-FROZEN). Branch: `slice-vd3`. See [[hazard-forge-verdict-roadmap]], [[hazard-forge-warmhull-roadmap]],
> [[hazard-forge-docs-style]], [[hazard-forge-metal-showcase-gate]].

**Goal:** Extend `engine/game/verdict.h` (additive — VD1/VD2 byte-unchanged) with `SyncComponentsToBodies(world)`
(write each `BodyRef`-bound entity's gameplay-driven velocity into `world.sim.bodies[simBodyIndex]`, in FIXED
`order[]` sequence), `SyncBodiesToComponents(world)` (read each `BodyRef`-bound body's settled `pos`/`orient` back
into the entity's `Transform2D`, fixed order), and `StepWorld(world, commands, tick, cfg)` — the single
deterministic world tick: `ApplyCommands` → `StepGameplay` (VD2 systems) → `SyncComponentsToBodies` →
`LowerToHullCommands` → `gjk::ApplyHullCommands` → `warmhull::StepWarmSleepHullWorld(sim, cache, sleep, cfg)` →
`SyncBodiesToComponents`, in a FIXED pinned order. Add `StepWorldN`. Add the showcase `--vd3-world-shot` (Vulkan) /
`--vd3-world` (Metal) — both build a gameplay+physics scene (a player entity bound to a hull body + a small hull
stack + pickups), step `StepWorldN`, and render the pure-integer world. Bake the integer golden `vd3_world`. **NO
new render RHI, NO new shader, NO new compute.**

## Design call: one pinned tick; the frozen sim verbatim; a deterministic BodyRef bridge

The embedded sim is `warmhull::StepWarmSleepHullWorld` (flagship #26, WH4) — a pure deterministic integer step over
a `gjk::HullWorld` + a `HullCache` + a `HullSleepState[]` (the TRIPLE). VD3 composes it into the world tick WITHOUT
modifying it:
- **`SyncComponentsToBodies(world)`** — for each entity (in `order[]` sequence) with a `BodyRef` bound
  (`simBodyIndex != kNoBody`) + a gameplay component that drives the body (e.g. a `Velocity2D` the gameplay set, or
  a desired-velocity), write it into `world.sim.bodies[simBodyIndex].vel` (the deterministic gameplay→sim push).
  FIXED `order[]` order, integer.
- **The sim commands** (the `kCmdImpulse`/`kCmdSetAngVel` verbs) are lowered by `LowerToHullCommands` (VD1) and
  applied via `gjk::ApplyHullCommands` (frozen) BEFORE the step — the `ApplyHullCommands`-before-step contract,
  identical to every sim's lockstep tick.
- **`warmhull::StepWarmSleepHullWorld(sim, cache, sleep, cfg)`** — the FROZEN WH4 step, called verbatim. VD3 must
  NOT perturb its determinism (proof #2 below).
- **`SyncBodiesToComponents(world)`** — for each `BodyRef`-bound entity (in `order[]` sequence), read
  `world.sim.bodies[simBodyIndex].pos`/`orient` back into the entity's `Transform2D` (the deterministic sim→gameplay
  read-back). FIXED order, integer.
- **`StepWorld`** pins the order: gameplay first (so a `kCmdMove` / a collected pickup is reflected before the sim
  reads bodies), then the sim, then the read-back. The schedule is documented; swapping the sync halves changes the
  result (proving the order is pinned).

> NOTE: the body index `simBodyIndex` is a STABLE index into `world.sim.bodies` (the sim bodies are NOT recycled
> within a match — like the hull worlds in WH1-WH6). An out-of-range / `kNoBody` `BodyRef` is a deterministic no-op
> in both syncs (an entity with no body is gameplay-only). Pure CPU, no GPU dispatch.

## Reuse map (file:line)
- **VD1/VD2 `engine/game/verdict.h` (APPEND after `SpawnMover`):** `VerdictWorld` (incl. `sim`/`cache`/`sleep`),
  `BodyRef`, `Transform2D`/`Velocity2D`, `ApplyCommands`, `StepGameplay`, `LowerToHullCommands`, `order[]`,
  `MeasureVerdict`. VD1/VD2 byte-frozen.
- **warmhull.h (read-only — the embedded sim, REUSE verbatim):** `warmhull::StepWarmSleepHullWorld`
  (`warmhull.h:949` — the WH4 step), `warmhull::HullCache`, `warmhull::HullSleepState`, `warmhull::HullSleepConfig`.
- **gjk.h (read-only):** `gjk::HullWorld`, `gjk::ApplyHullCommands` (`gjk.h:1315` — applies the lowered sim
  commands), `gjk::FxHull`/canonical builders (to build the scene's hull bodies), `fpx::FxBody`.
- **The pure-CPU showcase precedent:** VD2 `--vd2-tick` + the WH4 `--wh4-stack` (the hull-stack 2D-integer render).
  Mirror for `--vd3-world` (gameplay entities + the hull bodies in one pure-integer scene).
- **Registration:** `scripts/verify.ps1` (`vd3_world` + `--vd3-world-shot`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — verify it stages as ` M `**), append to
  `tests/verdict_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/game/verdict.h`** (VD1/VD2 byte-frozen): `SyncComponentsToBodies`, `SyncBodiesToComponents`,
   `StepWorld`, `StepWorldN`, and any needed binding helper (e.g. `BindBody(world, id, simBodyIndex)` setting the
   entity's `BodyRef`). **warmhull.h/gjk.h and ALL sim headers BYTE-UNCHANGED.** The whole tick is pure integer; the
   schedule order is FIXED + documented.
2. **Showcase `--vd3-world-shot <out>` (Vulkan) AND `--vd3-world` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--vd3-world` BEFORE reporting DONE).** BOTH build a gameplay+physics scene (a player entity
   bound to a dynamic hull body + a small hull stack on a static support + a couple of pickup entities), step
   `StepWorldN` with a fixed command stream (a `kCmdImpulse` nudges the player body; gameplay collects a pickup),
   and render the pure-integer world (the hull bodies + the gameplay entity markers). Golden =
   `tests/golden/metal/vd3_world.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) the composed world is deterministic:** `vd3-world: {entities:<N>, bodies:<B>, ticks:<K>} two-run
     BYTE-IDENTICAL` — a fixed K-tick `StepWorldN` script produces byte-identical state across the WHOLE world
     (entities + components + the sim TRIPLE) on two runs (`MeasureVerdict` + `warmhull::WarmHullStatesEqual` on the
     sim).
   - **(2) Verdict didn't perturb the frozen sim:** `vd3-world: embedded sim == standalone (bodies BIT-EXACT)` —
     the embedded `sim` after K ticks of `StepWorld` is byte-identical to a STANDALONE
     `warmhull::StepWarmSleepHullWorldN` fed the SAME impulses/scene (proving the composition didn't alter the
     frozen sim's determinism — the sync only reads/writes, the step is verbatim).
   - **(3) the sync is a pure function:** `vd3-world: sync pure {p2b, b2p} two-call BYTE-EQUAL` —
     `SyncComponentsToBodies` / `SyncBodiesToComponents` called twice on the same world are byte-equal (no hidden
     state), and a body-bound entity's `Transform2D` tracks its bound body's settled `pos`.
   - Golden discipline: ONLY `tests/golden/metal/vd3_world.png`; do NOT commit it. Existing 235 goldens UNTOUCHED.
4. **Cross-backend bar:** the NUMERIC proofs are pure integer → strict and backend-independent. The golden IMAGE is
   the pure-integer world render → strict-zero cross-vendor.
5. **Tests — APPEND to `tests/verdict_test.cpp` (pure CPU):** `StepWorld` deterministic over a fixed script (whole
   world two-run byte-equal, incl. the sim TRIPLE via `WarmHullStatesEqual`); the embedded sim == a standalone
   `StepWarmSleepHullWorldN` (no perturbation); the sync is a pure function + a body-bound entity tracks its body;
   an unbound (`kNoBody`) entity is a sync no-op; the schedule order is pinned (swapping the sync halves changes the
   result). Clean under `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `verdict-world-step` (features) + `--vd3-world-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does (and verifies it stages).**

## RHI seam additions (summary)
- **None — NO new shader, NO new compute, NO new render RHI.** Pure-CPU integer; the frozen sim step is called, not
  re-implemented; the render reuses the existing marker/hull draw. `engine/game/verdict.h` APPEND-only (VD1/VD2
  frozen); warmhull.h/gjk.h + ALL sim headers + ALL shaders UNCHANGED. Report the seam: verdict.h APPEND-only; NO
  rhi.h change, NO shader, NO frozen-file edit.

## Out of scope (YAGNI — later slices)
The heterogeneous snapshot/restore (VD4 — VD3 advances the composed world but does not yet snapshot it). The
world-level lockstep/rollback (VD5). The render capstone (VD6). Composing MORE than one sim (VD3 composes exactly
ONE sim subsystem — the warm+sleep hull world — to keep the determinism claim provable; cloth/fluid/grain in one
world is a future generalization, explicitly deferred). VD3 claims ONLY: one deterministic `StepWorld` tick that
composes the gameplay systems + the frozen warm+sleep hull sim via a pure-function `BodyRef` sync, bit-identical
CPU/Vulkan/Metal, with the embedded-sim-unperturbed proof + the integer golden + the three proofs.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "verdict|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--vd3-world-shot` on Vulkan: the 3 proof lines (incl. the embedded-sim-unperturbed proof)
   + exit 0 under the conan validation layer → ZERO VUID. VERIFY the composed render is coherent (the player body +
   the hull stack + pickup markers), no garbage/NaN.
3. Metal: `visual_test --vd3-world` → `tests/golden/metal/vd3_world.png`; two runs DIFF 0.0000. **Confirm
   `--vd3-world` wired in `visual_test.mm` (grep it) BEFORE the Mac bake** — NO shader added. Cross-vendor STRICT
   ZERO.
4. **Render-invariance:** ONLY `vd3_world.png` added; the other 235 byte-identical (+ controller introspect rebake,
   verified staged as ` M `).
5. Introspect: exactly `+verdict-world-step` + `--vd3-world-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + VD1/VD2 verdict.h code + warmhull.h/gjk.h + ALL sim headers + ALL shaders
   byte-unchanged; verdict.h APPEND-only; NO shader change). `vd3_world` in the Mac loop + `--vd3-world-shot` in
   `$vkShots`.
