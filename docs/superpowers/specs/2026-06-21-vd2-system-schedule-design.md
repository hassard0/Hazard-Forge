# Slice VD2 — Deterministic Gameplay/Netcode: THE SYSTEM SCHEDULE + THE GAMEPLAY TICK — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of
> FLAGSHIP #27 (DETERMINISTIC GAMEPLAY / NETCODE PRODUCT LAYER, `hf::game::verdict`). VD1 built the deterministic
> entity world (pinned monotonic `EntityId`, a fixed `order[]`, integer-only components) + the unified
> input-command bus (`ApplyCommands`). VD2 adds the part that makes it a *game*: **deterministic gameplay
> SYSTEMS** running on a **pinned schedule** over a **pinned iteration order**. The crux it fixes is the raw ECS's
> non-determinism: `ecs::Registry::view<>` (ecs.h:248) drives off the smallest pool in dense (insertion) order,
> which renumbers under entity churn — so two peers could visit entities in different sequences and diverge. VD2's
> `OrderedView` iterates strictly in `order[]` (spawn order), so every system visits every entity in the identical
> sequence on every peer/platform. The gameplay rules are pure integer (Q16.16). PURE CPU — the strictest
> determinism tier (strict 0px). APPEND to `engine/game/verdict.h` (VD1 + ecs/warmhull/sim BYTE-FROZEN). Branch:
> `slice-vd2`. See [[hazard-forge-verdict-roadmap]], [[hazard-forge-warmhull-roadmap]], [[hazard-forge-docs-style]],
> [[hazard-forge-metal-showcase-gate]].

**Goal:** Extend `engine/game/verdict.h` (additive — VD1 byte-unchanged) with `OrderedView<Ts...>(VerdictWorld&)`
(iterate live entities in `order[]` sequence, yielding each entity whose handle has ALL of `Ts...`), a small set of
deterministic integer gameplay systems (`SystemMovement` — integrate each entity's `Transform2D` by a velocity
component / a per-entity rule; `SystemDamage` — apply a deterministic hazard/decay to `Health`; `SystemCollect` —
an integer overlap test that "collects" pickups and bumps a score), and `StepGameplay(VerdictWorld&, span<Command>,
tick)` = `ApplyCommands` then run the systems in a FIXED schedule order over `OrderedView`. Add the showcase
`--vd2-tick-shot <out>` (Vulkan) / `--vd2-tick` (Metal) — both run a deterministic gameplay-only script (no physics
yet) for N ticks and render the pure-integer world state. Bake the integer golden `vd2_tick`. **NO new render RHI,
NO new shader, NO new compute.**

## Design call: pin the iteration order at the gameplay layer; systems are pure integer rules

The raw `ecs::Registry::view<T...>()` (ecs.h:248 `View::Resolve`) is fast but its iteration order is the dense
(insertion) order of the smallest matching pool — NOT a pinned cross-peer/cross-platform order, and it renumbers
when components are added/removed. A gameplay system that mutates state while iterating in that order is
non-deterministic. VD2 fixes this WITHOUT modifying the ECS:
- **`OrderedView<Ts...>(world)`** walks `world.order` (the VD1 spawn-order live-id list) in sequence; for each id it
  looks up the `ecs::Entity` handle and yields it IFF `reg.valid(e) && (reg.has<Ts>(e) && ...)`. The iteration
  order is `order[]` — identical on every peer/platform. (It is a thin range/callback over `order[]` +
  `reg.get<>`, NOT a new container.) The raw `reg.view<>` is used ONLY where order is provably irrelevant (and that
  is asserted, not assumed).
- **The systems are pure integer rules** over `OrderedView`, in a FIXED schedule:
  - `SystemMovement(world)` — for each entity with `Transform2D` (+ a velocity carried in a component or a fixed
    rule), advance `pos` by an integer delta (the Q16.16 `fpx::FxAdd`). Deterministic.
  - `SystemDamage(world)` — for each entity with `Health`, apply a deterministic integer decay/hazard (e.g. −1 hp
    when in a hazard region by an integer position test). Deterministic.
  - `SystemCollect(world)` — an integer overlap test (Q16.16 distance² compare, no float, no sqrt) between a
    "player" entity and "pickup" entities; on overlap, despawn the pickup + bump an integer score. Deterministic.
- **`StepGameplay(world, commands, tick)`** = `ApplyCommands(world, commands, tick)` (VD1) → `SystemMovement` →
  `SystemDamage` → `SystemCollect` → `world.tick++`. The SCHEDULE order is FIXED and documented (later systems see
  earlier systems' mutations — a single deterministic pass, the Gauss-Seidel-in-fixed-order discipline the sims
  use).

> NOTE: VD2 runs GAMEPLAY systems only — it does NOT yet step the embedded physics sim (VD3 composes
> `warmhull::StepWarmSleepHullWorld` into the tick). Pure CPU, no GPU dispatch. The scene is gameplay entities
> (a player + pickups + hazards), not hull bodies.

## Reuse map (file:line)
- **VD1 `engine/game/verdict.h` (APPEND after `VerdictMeasuresEqual`):** `VerdictWorld`, `order[]`, `Transform2D`/
  `Health`/`BodyRef`, `Command`/`ApplyCommands`, `SpawnEntity`/`DespawnEntity`, `MeasureVerdict`. VD1 byte-frozen.
- **ecs.h (read-only — REUSE, do NOT edit):** `ecs::Registry::get`/`has`/`valid` (ecs.h:177/170), `ecs::Entity`.
  `OrderedView` is the deterministic alternative to `Registry::view<>` (ecs.h:248) — it does NOT modify ecs.h.
- **fpx.h (read-only):** `fpx::FxAdd`/`FxSub`/`fxmul`/`FxDot`/`kFrac` (the integer movement + the distance² overlap
  test — NO float, NO sqrt).
- **The pure-CPU showcase precedent:** VD1 `--vd1-world` (the pure-integer 2D world-state render). Mirror for
  `--vd2-tick` (render the world after N gameplay ticks: the player moved, pickups collected/absent, health
  decayed — visible deterministic gameplay).
- **Registration:** `scripts/verify.ps1` (`vd2_tick` + `--vd2-tick-shot`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — verify it stages as ` M `**), append to
  `tests/verdict_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/game/verdict.h`** (VD1 byte-frozen): `OrderedView`, `SystemMovement`, `SystemDamage`,
   `SystemCollect`, `StepGameplay`, and any needed component (e.g. `Velocity2D{FxVec3 vel}`, `Pickup{int32 value}`,
   `Score{int32 points}` — all integer). **ecs.h and ALL sim headers BYTE-UNCHANGED.** Systems are pure integer
   (Q16.16). The schedule order is a FIXED, documented sequence.
2. **Showcase `--vd2-tick-shot <out>` (Vulkan) AND `--vd2-tick` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--vd2-tick` BEFORE reporting DONE).** BOTH build a deterministic gameplay scene (a player
   + a few pickups + a hazard), run `StepGameplay` for N ticks with a fixed command stream (move the player toward
   the pickups), and render the pure-integer world state (player + remaining pickups + a score readout). Golden =
   `tests/golden/metal/vd2_tick.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) the iteration-order fix:** `vd2-tick: OrderedView order-stable (order[] != reg.view churn)` —
     `OrderedView` yields entities in `order[]` sequence regardless of pool sizes / add-remove churn; falsify
     against a raw `reg.view<>` that yields a DIFFERENT (renumbered) order for the same entities (assert
     `OrderedView` == `order[]`-filtered AND `!= reg.view order` for a churned scene). **The determinism contract.**
   - **(2) deterministic gameplay:** `vd2-tick: {ticks:<K>, entities:<N>, score:<S>} two-run BYTE-IDENTICAL` — a
     fixed K-tick gameplay script produces byte-identical world state (`MeasureVerdict` + the score) on two runs.
   - **(3) a hand-checked rule:** `vd2-tick: move rule {start:<p0>, end:<p1>} exact` — one move command over the
     movement system yields the EXACT hand-computed Q16.16 position; and a collect removes exactly the overlapping
     pickup + bumps the score by its integer value.
   - Golden discipline: ONLY `tests/golden/metal/vd2_tick.png`; do NOT commit it. Existing 234 goldens UNTOUCHED.
4. **Cross-backend bar:** the NUMERIC proofs are pure integer → strict and backend-independent. The golden IMAGE is
   the pure-integer world render → strict-zero cross-vendor.
5. **Tests — APPEND to `tests/verdict_test.cpp` (pure CPU):** `OrderedView` yields `order[]` sequence (and differs
   from a churned `reg.view`); `StepGameplay` is deterministic over a fixed script (two runs byte-equal);
   `SystemMovement` exact Q16.16 delta; `SystemCollect` removes the overlapping pickup + bumps the score by its
   value, leaves non-overlapping pickups; `SystemDamage` deterministic decay; the schedule order is fixed (swapping
   two systems changes the result — proving order matters and is pinned). Clean under `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `verdict-systems` (features) + `--vd2-tick-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does (and verifies it stages).**

## RHI seam additions (summary)
- **None — NO new shader, NO new compute, NO new render RHI.** Pure-CPU integer gameplay; the render reuses the
  existing marker draw. `engine/game/verdict.h` APPEND-only (VD1 frozen); ecs.h + ALL sim headers + ALL shaders
  UNCHANGED. Report the seam: verdict.h APPEND-only; NO rhi.h change, NO shader, NO frozen-file edit.

## Out of scope (YAGNI — later slices)
Composing the physics subsystem (VD3 — VD2 is gameplay systems only). The heterogeneous snapshot/restore (VD4). The
world-level lockstep/rollback (VD5). The render capstone (VD6). A general system-dependency graph / parallel
scheduler (the schedule is a FIXED hand-ordered sequence — deterministic by construction; a dependency solver is
out of scope). A scripting language. VD2 claims ONLY: deterministic gameplay systems on a pinned schedule over a
pinned `order[]` iteration, pure integer, bit-identical CPU/Vulkan/Metal, with the integer golden + the three
proofs (including the OrderedView-vs-raw-view determinism contract).

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "verdict|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--vd2-tick-shot` on Vulkan: the 3 proof lines (incl. the OrderedView contract) + exit 0
   under the conan validation layer → ZERO VUID. VERIFY the gameplay render is coherent (player moved, pickups
   collected/remaining, a score), no garbage/NaN.
3. Metal: `visual_test --vd2-tick` → `tests/golden/metal/vd2_tick.png`; two runs DIFF 0.0000. **Confirm `--vd2-tick`
   wired in `visual_test.mm` (grep it) BEFORE the Mac bake** — NO shader added. Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `vd2_tick.png` added; the other 234 byte-identical (+ controller introspect rebake,
   verified staged as ` M `).
5. Introspect: exactly `+verdict-systems` + `--vd2-tick-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + VD1 verdict.h code + ecs.h + ALL sim headers + ALL shaders byte-unchanged; verdict.h
   APPEND-only; NO shader change). `vd2_tick` in the Mac loop + `--vd2-tick-shot` in `$vkShots`.
