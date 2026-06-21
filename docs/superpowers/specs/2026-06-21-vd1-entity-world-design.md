# Slice VD1 — Deterministic Gameplay/Netcode: THE ENTITY WORLD + THE INPUT-COMMAND BUS (the beachhead) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice of
> FLAGSHIP #27 (DETERMINISTIC GAMEPLAY / NETCODE PRODUCT LAYER, `hf::game::verdict`). The engine has a complete
> deterministic Q16.16 simulation suite (every sim bit-identical CPU/Vulkan/Metal + lockstep/rollback-replayable)
> and a networking trilogy — but the sims are bare body-soups with no *gameplay*, and the net trilogy + the
> playable "game" sample sit on the FLOAT `physics::World` (single-build determinism only, `snapshot.h:15`). An ECS
> exists (`engine/ecs/ecs.h`) but has NO determinism contract (process-global type ids, smallest-pool iteration
> order, no snapshot/command bus). Flagship #27 is the BRIDGE: a deterministic ECS-game *world* with ONE generic
> rollback/lockstep loop that composes the frozen Q16.16 sim as a subsystem — the first artifact that is a *game*
> (entities + rules + a physics sim) that is lockstep/rollback bit-identical across platforms. VD1 is the
> beachhead: the deterministic entity world + the unified input-command bus the whole loop is built on. It is PURE
> CPU integer — the strictest determinism tier (strict 0px). NEW header `engine/game/verdict.h` (`#include
> "sim/warmhull.h"` + `"ecs/ecs.h"` READ-ONLY/BYTE-FROZEN). Branch: `slice-vd1`. See [[hazard-forge-verdict-roadmap]],
> [[hazard-forge-warmhull-roadmap]], [[hazard-forge-docs-style]], [[hazard-forge-metal-showcase-gate]].

**Goal:** Create `engine/game/verdict.h` (namespace `hf::game::verdict`) with: `EntityId` (a monotonic `uint32`
NEVER recycled within a match — the pinned *sim identity*, mapped to an `ecs::Entity` handle); `VerdictWorld` (the
deterministic world: an `ecs::Registry` + the `EntityId↔ecs::Entity` map + a fixed `order[]` of live entity ids +
`nextId` + the embedded Q16.16 sim `gjk::HullWorld` + its `warmhull::HullCache` + `warmhull::HullSleepState[]` +
`tick`); integer-only components `Transform2D` / `Health` / `BodyRef` (Q16.16, ZERO float); `Command` (the
generalization of `convex::ConvexCommand` carrying gameplay verbs + the lowered sim verbs); `SpawnEntity` /
`DespawnEntity` (deterministic id alloc, never recycled); `LowerToHullCommands` (the sim-verb subset →
`std::vector<convex::ConvexCommand>`); and `ApplyCommands` (fixed array order, only `.tick==tick`, dead/out-of-range
target → no-op). Add the showcase `--vd1-world-shot <out>` (Vulkan) / `--vd1-world` (Metal) — both build a
deterministic spawn/despawn + command script, run `ApplyCommands`, and render a pure-integer world-state
visualization. Bake the integer golden `vd1_world`. **NO new render RHI, NO new shader, NO new compute** — VD1 is
pure-CPU integer gameplay state.

## Design call: pin identity + the command contract at the gameplay layer; reuse the ECS as the store

The raw `ecs::Registry` (`ecs.h:122`) is a fine sparse-set store but is NOT a determinism contract: `create()`
recycles indices via a free-list + bumps generations, and `view<>` (`ecs.h:248`) drives off the smallest pool in
dense (insertion) order — neither is a pinned cross-peer/cross-platform wire order. VD1 layers the contract ON TOP,
without modifying the ECS:
- **`EntityId` = a monotonic `uint32` (start 1, NEVER recycled within a match).** It is the deterministic identity
  used everywhere (snapshot order, command targeting, sim body index) — a pure function of the command stream (a
  `kCmdSpawn` at tick T in array order → `nextId++`), so it is identical on every peer and survives rollback. The
  `ecs::Entity` handle (whose free-list/generation churn is non-deterministic-by-design) is an implementation
  detail behind `VerdictWorld::handle[EntityId]`. `order[]` is the live-entity id list in spawn order — the pinned
  iteration sequence (VD2's `OrderedView` will use it).
- **`Command` generalizes `convex::ConvexCommand`** (`convex.h:988` `{tick, kind, bodyId, arg}`): `Command {uint32
  tick; uint32 kind; EntityId target; FxVec3 arg;}` with the verb set `kCmdSpawn / kCmdDespawn / kCmdMove /
  kCmdAbility` (gameplay) ∪ `kCmdImpulse / kCmdSetAngVel` (sim, which `LowerToHullCommands` maps straight to
  `convex::ConvexCommand` for `gjk::ApplyHullCommands`). One stream drives the whole world.
- **`ApplyCommands(world, commands, tick)`** applies only the commands whose `.tick == tick`, in FIXED array order,
  BEFORE any step (the `convex::ApplyConvexCommands` contract, `convex.h:1000`): `kCmdSpawn` allocates an id +
  pushes to `order[]`; `kCmdDespawn` removes from `order[]` + frees the registry handle (but NEVER recycles the
  id); a command whose `target` is dead / out-of-range is a deterministic NO-OP (never a crash, never undefined).
  Pure integer, fixed order → bit-identical on every peer/platform.

> NOTE: VD1 is the data + the bus ONLY — it does NOT yet run gameplay systems (VD2) or step the physics (VD3). It
> proves the entity world + the command bus are deterministic in isolation. Pure CPU, no GPU dispatch.

## Reuse map (file:line)
- **ecs.h (read-only — REUSE the store, do NOT edit):** `ecs::Registry` (`ecs.h:122`), `ecs::Entity` (`ecs.h:36`),
  `Registry::create`/`get`/`has`/`remove` (`ecs.h:126/177/170/191`), `ComponentPool` (`ecs.h:63`). VD1 adds the
  determinism contract over these; it does NOT modify `ecs.h` (the smallest-pool `view<>` non-determinism is
  sidestepped by `order[]`, not fixed in-place).
- **convex.h / gjk.h (read-only — the command contract VD1 generalizes + the sim it embeds):**
  `convex::ConvexCommand` (`convex.h:988`), `convex::ApplyConvexCommands` (`convex.h:1000` — the apply contract to
  mirror), `gjk::HullWorld` (`gjk.h:1174`), `gjk::ApplyHullCommands` (`gjk.h:1315` — the target of
  `LowerToHullCommands`), `gjk::FxHull`/canonical builders, `fpx::FxVec3`/`FxQuat`.
- **warmhull.h (read-only — the embedded sim subsystem, used from VD3; VD1 only declares the world fields):**
  `warmhull::HullCache`, `warmhull::HullSleepState`. (VD1 carries them in `VerdictWorld` but does not step them.)
- **introspect (read-only — the agentic hook):** `introspect::DescribeEngine(ecs::Registry&, ...)` (`introspect.h:83`
  — already ECS-keyed); the `kShowcases[]` manifest (`introspect.cpp:73`). VD1 registers the `--vd1-world-shot` /
  `--verdict` stub entries (additive list entries).
- **Registration:** `engine/game/` CMake (new `verdict.h` is header-only; the showcase wiring goes in
  `samples/hello_triangle/main.cpp` + `metal_headless/visual_test.mm`), `scripts/verify.ps1` (`vd1_world` +
  `--vd1-world-shot`), `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON
  golden**), a NEW `tests/verdict_test.cpp` (registered in the test CMakeLists).

## Design decisions (locked)
1. **NEW header `engine/game/verdict.h`** (`namespace hf::game::verdict`, `#include "sim/warmhull.h"` +
   `"ecs/ecs.h"`): `EntityId`, `VerdictWorld`, `Transform2D`/`Health`/`BodyRef`, `Command` + the `kCmd*` enum,
   `SpawnEntity`, `DespawnEntity`, `LowerToHullCommands`, `ApplyCommands`, a `VerdictMeasure`/summary helper. **ecs.h
   and ALL sim headers BYTE-UNCHANGED.** Components are Q16.16 integer ONLY (NO float anywhere in the world state).
2. **Showcase `--vd1-world-shot <out>` (Vulkan) AND `--vd1-world` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--vd1-world` BEFORE reporting DONE).** BOTH build a deterministic script (spawn a few
   entities with components, fire move/impulse commands over a few ticks, despawn one), run `ApplyCommands` per
   tick, and render a PURE-INTEGER world-state visualization (entity markers at their `Transform2D.pos >> kFrac`,
   id-tinted; the despawned one absent) → strict-zero cross-vendor. Golden = `tests/golden/metal/vd1_world.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) deterministic world:** `vd1-world: {entities:<N>, order:[<ids>], nextId:<K>} two-run BYTE-IDENTICAL` —
     a fixed spawn/despawn/command script yields a byte-identical `order[]` + `nextId` + component state on two runs.
   - **(2) id alloc independent of ECS churn:** `vd1-world: id alloc churn-independent (nextId:<K>)` — a
     spawn→despawn→spawn sequence re-derives the SAME next `EntityId` regardless of the `ecs::Registry` free-list
     recycle (the pinned-identity contract — assert the id is the monotonic value, NOT a recycled registry index).
   - **(3) ApplyCommands guard:** `vd1-world: ApplyCommands {tickFiltered, fixedOrder, deadTargetNoOp} OK` — only
     `.tick==tick` commands apply, in array order, and a dead/out-of-range `target` is a no-op (assert the world is
     unchanged by an out-of-range command).
   - **(4) the bus composes:** `vd1-world: LowerToHullCommands == hand-written stream (cmds:<C>)` — the sim-verb
     subset lowered to `convex::ConvexCommand` is byte-identical to a hand-written reference stream (the bus
     doesn't diverge from the frozen sim command contract).
   - Golden discipline: ONLY `tests/golden/metal/vd1_world.png`; do NOT commit it. Existing 233 goldens UNTOUCHED.
4. **Cross-backend bar:** the NUMERIC proofs are pure integer → strict and backend-independent. The golden IMAGE is
   the pure-integer world-state render → strict-zero cross-vendor (both backends compute the identical integer
   world; the 2D marker draw is deterministic).
5. **Tests — NEW `tests/verdict_test.cpp` (pure CPU):** `SpawnEntity` allocates monotonic ids + appends to `order[]`;
   `DespawnEntity` removes from `order[]` + frees the handle but the next spawn does NOT reuse the id; `ApplyCommands`
   tick-filter + fixed order + dead-target no-op; `LowerToHullCommands` == a hand-written `ConvexCommand` stream;
   two-run byte-equality of the whole world over a fixed script; component get/has round-trips. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `verdict-world` (features) + `--vd1-world-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.** (Per
   [[hazard-forge-introspect-golden-staleness]]: the controller verifies the golden actually stages before the bake
   commit.)

## RHI seam additions (summary)
- **None — NO new shader, NO new compute, NO new render RHI.** VD1 is pure-CPU integer gameplay state; the render is
  the existing instanced/marker draw reused. `engine/game/verdict.h` is a brand-new sibling that `#include`s
  `sim/warmhull.h` + `ecs/ecs.h` read-only; ecs.h + ALL sim headers + ALL shaders UNCHANGED. Report the seam: NEW
  header `verdict.h` only; NO rhi.h change, NO shader, NO frozen-file edit.

## Out of scope (YAGNI — later slices)
The deterministic system schedule + gameplay tick (VD2). Composing the physics subsystem (VD3 — VD1 carries the sim
fields but does NOT step them). The heterogeneous snapshot/restore (VD4). The world-level lockstep/rollback (VD5).
The playable render capstone (VD6). A scripting VM, a visual editor, hot-reload, an archetype-ECS rewrite. Rebuilding
the net trilogy. VD1 claims ONLY: a deterministic entity world (pinned monotonic ids, a fixed `order[]`,
integer-only components) + a unified tick-ordered input-command bus that generalizes `convex::ConvexCommand` and
lowers cleanly to the frozen sim command contract — the same on CPU/Vulkan/Metal, with the integer golden + the
four proofs.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "verdict|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--vd1-world-shot` on Vulkan: the 4 proof lines + exit 0 under the conan validation layer →
   ZERO VUID. VERIFY the world-state visualization is coherent (entity markers at their positions; the despawned
   one absent), no garbage/NaN.
3. Metal: `visual_test --vd1-world` → `tests/golden/metal/vd1_world.png`; two runs DIFF 0.0000. **Confirm
   `--vd1-world` is wired in `visual_test.mm` (grep it) BEFORE the Mac bake** — NO shader added. Cross-vendor STRICT
   ZERO (pure-integer world + render).
4. **Render-invariance:** ONLY `vd1_world.png` added; the other 233 byte-identical (+ controller introspect rebake;
   verify the golden stages as ` M ` before the bake commit).
5. Introspect: exactly `+verdict-world` + `--vd1-world-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + ecs.h + warmhull.h/gjk.h/convex.h + ALL other sim headers + ALL shaders byte-unchanged;
   `verdict.h` is a NEW sibling `#include`ing `sim/warmhull.h` + `ecs/ecs.h`; NO shader change). `vd1_world` in the
   Mac loop + `--vd1-world-shot` in `$vkShots`.
