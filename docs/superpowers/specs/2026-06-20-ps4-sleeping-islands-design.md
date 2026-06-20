# Slice PS4 — Deterministic Persistent Contacts: SLEEPING ISLANDS (the new physics) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP #21
> (DETERMINISTIC WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, `hf::sim::persist`) — THE NEW-PHYSICS BEAT. PS1-PS3
> built the contact key, the cache, and the warm-started solver. PS4 adds DETERMINISTIC SLEEPING: a per-body
> integer kinetic-energy accumulator + a fixed wake/sleep hysteresis, with contact-graph ISLAND propagation so a
> resting tower sleeps and wakes ATOMICALLY — the warm-started step skips integrate+solve for sleeping islands.
> Sleeping is the scale + stability aid every shipping engine uses, but float energy thresholds diverge
> machine-to-machine; here it is bit-identical CPU/Vulkan/Metal. The headline: a warm-started tower rests and goes
> to sleep (exactly zero residual motion), then a thrown box WAKES the whole island deterministically. INTEGER-bit-
> exact. int64 → the `persist_sleep.comp` shader is Vulkan-only + a Metal CPU reference. PS1-PS3's `persist.h` code +
> CX/FC's `convex.h`/`fric.h` are BYTE-FROZEN (PS4 is additive). Branch: `slice-ps4`. See
> [[hazard-forge-persist-roadmap]].

**Goal:** Extend `engine/sim/persist.h` (additive — PS1-PS3 + fric.h byte-unchanged) with sleeping: `SleepState`
(per-body energy + timer + asleep flag) + `SleepConfig` + the island propagation + `StepWarmSleepWorld` (the
warm step that sleeps/wakes islands and skips sleeping bodies). Add the new int64 shader
`shaders/persist_sleep.comp.hlsl` + `--persist-sleep-shot`/`--persist-wake-shot` (Vulkan) /
`--persist-sleep`/`--persist-wake` (Metal). Bake the integer goldens `persist_sleep` + `persist_wake`. **NO new RHI.**

## Design call: integer KE hysteresis + fixed-order island propagation

A body is a candidate to sleep when its motion stays low for a while; a whole contact-connected ISLAND sleeps/wakes
together (a tower is one island — if any block is disturbed, the whole tower wakes).
- **The per-body kinetic measure (integer, no products beyond the existing toolbox):** `KineticEnergy(b)` = a
  deterministic integer motion magnitude — e.g. `FxLength(b.vel) + FxLength(b.angVel)` (the Q16.16 `FxLength`,
  reused; an L2 motion sum). Below `cfg.sleepThreshold` = "quiet".
- **`SleepState { fx energy; uint32_t quietTicks; bool asleep; }`** per body, in a PARALLEL array (FxBody is
  byte-frozen). `quietTicks` increments each tick the body is quiet, resets to 0 when it is energetic. A body
  becomes ASLEEP when `quietTicks >= cfg.sleepDelay` (the hysteresis delay), and WAKES (quietTicks=0, asleep=false)
  when its energy exceeds `cfg.wakeThreshold` (>= sleepThreshold — a hysteresis BAND so it doesn't flicker).
- **The island propagation (fixed-order, union-find-free):** a sleeping body must stay awake if ANY body it contacts
  is awake (else a tower would half-sleep). Build the contact adjacency from THIS tick's overlapping pairs (the
  all-pairs `BoxSatStable` overlaps). Then propagate WAKEFULNESS to a fixed point: repeat (a fixed bound of N
  passes, or until no change) in FIXED body order — a body is AWAKE if (its own `quietTicks < sleepDelay`) OR (any
  adjacent body is awake). A STATIC body is never an island-waker (it is always "asleep"/inert and does not keep its
  neighbours awake — the floor doesn't keep the tower awake). The final per-body asleep flag = quiet AND no awake
  neighbour. Deterministic (fixed order, fixed pass bound).
- **`StepWarmSleepWorld(world, cache, sleep[], cfg)`** — the PS3 `StepWarmWorld` tick with sleeping:
  1. Compute each dynamic body's `KineticEnergy` (from the PRE-integrate state) + update its `quietTicks`/energy.
  2. Build the contact adjacency (all-pairs overlap) + propagate wakefulness → the per-body `asleep` flag.
  3. **SKIP** integrate for ASLEEP bodies (they stay exactly put — zero drift); only AWAKE dynamic bodies integrate.
  4. The warm impulse solve + position de-penetration run ONLY over pairs with at least one AWAKE dynamic body
     (a fully-asleep pair is skipped — its cached impulses persist untouched).
  5. Rebuild the cache (PS2 UpdateCache) over the solved pairs; the sleeping pairs keep their prior cache entries.
  A WAKE event (an external impulse / a `persist` command on a sleeping body, OR a new contact from a moving body)
  raises that body's energy → resets `quietTicks` → the propagation wakes the whole island next tick.

**THE int64 REALITY:** the whole chain is int64. `persist_sleep.comp` is **VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl)**,
single-thread over the small world; the Metal `--persist-sleep`/`--persist-wake` runs the CPU `StepWarmSleepWorldN`
— byte-identical to the Vulkan GPU result BY CONSTRUCTION, the Vulkan side carries the GPU==CPU memcmp.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **PS3 `engine/sim/persist.h` (read it; APPEND only after `StepWarmWorldN`/`WarmMeasure`):** `SolveFrictionWarm`,
  `StepWarmWorld`, `WarmStepConfig`, `PersistentCache`, `BuildKeyedManifold`, `MatchCache`, `UpdateCache`. PS1-PS3
  byte-frozen.
- **fric.h (read-only — do NOT edit):** `fric::FrictionStepConfig`, `fric::StepFrictionWorld` (the FC4 shell PS4
  extends via PS3's StepWarmWorld). **DO NOT modify.**
- **convex.h (read-only):** `convex::ConvexWorld`, `IsDynamic`, `BoxSatStable` (the overlap adjacency), the position
  de-penetration. **DO NOT modify.** fpx.h: `FxLength` (the KE magnitude), `IntegrateBodyFull`, `kFlagDynamic`.
- **The shader + showcase precedent:** PS3's `shaders/persist_warm.comp.hlsl` (the int64 Vulkan-only whole-world
  warm step + the per-tick cache) — `persist_sleep.comp` adds the sleep state + island propagation + the skip.
  Confirm `persist_sleep` NOT in hf_gen_msl.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/persist_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/persist.h`** (PS1-PS3 byte-frozen): `SleepState`, `SleepConfig` (`sleepThreshold`,
   `wakeThreshold`, `sleepDelay`, + the PS3 `WarmStepConfig` knobs — or a composed `SleepStepConfig`),
   `KineticEnergy`, the island propagation (a helper `PropagateWake`), `StepWarmSleepWorld(world, cache, sleep[],
   cfg)`, `StepWarmSleepWorldN`, a `SleepMeasure` (asleep count, max speed). Pure integer, FIXED orders. **NEW
   shader** `persist_sleep.comp.hlsl` (int64, Vulkan-only, one thread runs the whole world step + sleep — copies
   `StepWarmSleepWorldN` VERBATIM). NOT in hf_gen_msl.
2. **Showcases — WIRE BOTH backends for TWO scenes:**
   - **`--persist-sleep-shot`/`--persist-sleep`:** the warm stack (floor + 3-4 dynamic boxes) run K ticks → it
     settles AND every dynamic body goes ASLEEP (its `asleep` flag set, its motion exactly zero because asleep
     bodies don't integrate). Render the sleeping tower (the FC4 side-view, sleeping bodies tinted distinctly).
   - **`--persist-wake-shot`/`--persist-wake`:** the same stack settled + asleep, THEN a thrown box (a dynamic body
     with an incoming velocity, OR a `persist` wake-impulse command) strikes the tower → the struck body wakes →
     the propagation wakes the WHOLE island → the tower topples/shifts. Render the woken/shifted tower.
   Vulkan: the GPU `persist_sleep.comp` → **memcmp the GPU final body world + sleep states vs the CPU
   `StepWarmSleepWorldN`** (NO tolerance), per scene. Metal: the CPU reference. Goldens =
   `tests/golden/metal/persist_sleep.png` + `persist_wake.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU final body world + sleep states == the CPU byte-for-byte, BOTH scenes.
     Print `persist-sleep: {scene:sleep, bodies:<N>, ticks:<K>, asleep:<A>} GPU==CPU BIT-EXACT` and the `wake` one.
   - **(2) determinism:** two runs → identical. Print `persist-sleep determinism: two runs BYTE-IDENTICAL`.
   - **(3) THE NEW PHYSICS — sleep + wake:** (a) the sleep scene's tower is ALL ASLEEP after K ticks AND its max
     dynamic speed is EXACTLY ZERO (asleep bodies don't move — the deterministic-rest headline); (b) the wake scene:
     after the throw, the previously-sleeping island is AWAKE (asleep count drops) and moving. Print `persist-sleep
     newphysics: {towerAsleep:true, zeroResidual:true, thrownBoxWakesIsland:true}`; assert all.
   - **(4) island atomicity:** waking ONE body in the resting island wakes ALL contact-connected dynamic bodies of
     that island (not just the struck one) — the island sleeps/wakes as a unit. Print `persist-sleep island:
     {wakesAtomically:true}`; assert (after the wake, every dynamic body in the connected island is awake).
   - **Golden discipline: ONLY `persist_sleep.png` + `persist_wake.png`; do NOT commit them.** Existing 199 image
     goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels, both
   scenes.
5. **Tests — APPEND to `tests/persist_test.cpp` (pure CPU):** a quiet body's `quietTicks` rises and it sleeps after
   `sleepDelay`; a sleeping body skipped by the step does NOT move (zero drift); an energetic body stays awake; a
   wake-impulse on a sleeping body wakes its whole contact island (propagation); a static floor does NOT keep the
   tower awake; the hysteresis band prevents flicker; two runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-persist-sleep` (features) + `--persist-sleep-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/convex.h` + `fric.h` + `fpx.h` + **PS1-PS3's persist.h code + persist_key/cache/warm.comp** + all
  other sim headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + all EXISTING shaders UNCHANGED. The ONLY
  new shader is `persist_sleep.comp.hlsl` (int64, Vulkan-only, NOT in hf_gen_msl). `persist.h` APPEND-only. Report
  the seam empty.

## Out of scope (YAGNI — later PS slices)
Lockstep + rollback (PS5 — pure CPU; the snapshot extends to include the cache + sleep state), the lit 3D render
capstone (PS6). PS4 claims ONLY: deterministic integer sleeping islands (KE hysteresis + fixed-order island
propagation + skip-sleeping-bodies) that put a warm tower to rest at exactly zero residual and wake the whole island
on a throw, bit-identical CPU↔Vulkan↔Metal, with the two integer goldens + the four proofs. NOTE: boxes only; the
sleep thresholds are FIXED integer hysteresis constants (a deterministic heuristic, the angDamp/slop honesty
lineage — deterministic, not physical truth); all-pairs island build (the small-scene convention — true
broadphase-scaled islands a deferred scaling refinement); int64 → Vulkan-GPU + Metal-CPU-ref.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 107 incl. PS1-PS3's `persist_test` + the appended PS4 cases).
   Clean under `windows-msvc-asan` (build+run `persist_test` + `introspect_test`).
2. **proofs + visual:** `--persist-sleep-shot` AND `--persist-wake-shot` on Vulkan: the 4 proofs + exit 0, under the
   Vulkan-validation gate → ZERO VUID. **VERIFY the sleep image shows a coherent RESTING TOWER (asleep, distinctly
   tinted) and the wake image shows the tower TOPPLED/SHIFTED (the thrown box woke + disturbed the island).**
3. Metal: `visual_test --persist-sleep` + `--persist-wake` → new goldens `persist_sleep.png` + `persist_wake.png`;
   two runs DIFF 0.0000 each. **Confirm `visual_test.mm` in the diff; confirm `persist_sleep.comp` NOT in
   `hf_gen_msl`.** Cross-vendor STRICT ZERO, both.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `persist_sleep.png` +
   `persist_wake.png` added; the other 199 byte-identical. `git diff master --stat -- tests/golden` = ONLY those two
   (metal) + the introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-persist-sleep` + `--persist-sleep-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fric.h`/`fpx.h` + **PS1-PS3's persist.h code +
   persist_key/cache/warm.comp** + ALL other sim headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL
   existing shaders byte-unchanged). `scripts/verify.ps1` updated: `persist_sleep` + `persist_wake` goldens in the
   Mac loop + `--persist-sleep-shot` + `--persist-wake-shot` in `$vkShots`. **The ONLY new shader is
   `persist_sleep.comp.hlsl` (int64, NOT in `hf_gen_msl`).**
