# Slice AX — Playable Game Sample (deterministic gameplay loop) — Phase 4 #3 — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The engine->game-engine
> milestone: a real gameplay loop, not another render showcase.

**Goal:** Demonstrate Hazard Forge running an ACTUAL game — a minimal "roll-a-ball, collect the pickups"
sample. A player rigid-body sphere is driven each fixed step by a SCRIPTED input track (for
determinism), the deterministic physics `World` integrates it, gameplay logic detects player↔pickup
contacts (collect → remove pickup, increment score), and the scene renders the player + remaining
pickups. A scripted playthrough is golden-verified AND its final game state (score, player position) is
asserted — proving the simulation is deterministic and the systems (physics + scene + camera + gameplay)
integrate into a playable whole.

## Why scripted input (determinism)

Live input is nondeterministic. To stay golden + state-verifiable, the sample is driven by a fixed
**scripted input track**: a `std::vector<GameInput>` (one entry per fixed step, e.g. a normalized
move direction + a "jump" bit). The fixed-timestep sim consumes the track in order. Same track + same
build ⇒ bit-identical playthrough ⇒ identical final frame and identical final state. (Live keyboard play
is a trivial extension in the interactive `--fly`-style window — OUT OF SCOPE here; this slice is the
headless deterministic proof.)

## Design decisions (locked)

1. **Gameplay module `engine/game/roll_game.{h,cpp}` (pure CPU, no RHI/backend symbols).** Namespace
   `hf::game`. Reuses `engine/physics/World` + `RigidBody` + `engine/math`. Types:
   - `struct GameInput { math::Vec3 moveDir; bool jump; };` (moveDir is a planar XZ direction, magnitude
     0..1).
   - `struct Pickup { math::Vec3 pos; float radius; bool collected; };`
   - `struct GameState { int playerBodyIndex; std::vector<Pickup> pickups; int score; int step; bool won; };`
   - `GameState MakeRollGame(World& world)` — sets up the World: a static ground, a dynamic player sphere
     (via `MakeDynamicSphere`) at a fixed start, and a deterministic set of N=3 pickups at fixed
     positions. (Optionally a few static obstacle spheres for the ball to navigate.)
   - `void StepGame(World& world, GameState& gs, const GameInput& in, float dt)` — apply the input as a
     planar force/impulse to the player body's `linVel` (a fixed accel constant; jump adds +Y velocity
     only when grounded — `player.position.y <= radius + eps`), call `world.Step(dt)`, then for each
     uncollected pickup test `distance(player.pos, pickup.pos) < player.radius + pickup.radius` → mark
     collected + `score++`; set `won = (score == N)`; `step++`. Deterministic, RNG/clock-free.
   - `std::vector<GameInput> ScriptedTrack()` — a fixed input track (hand-authored) that, over a fixed
     step count, rolls the player through all 3 pickups so the playthrough WINS (score 3). This is the
     deterministic scenario the golden + state assertions pin.

2. **Showcase `--game-shot <out>` (Vulkan) / `--game` (Metal).** Build the roll game, run `StepGame` over
   the full `ScriptedTrack()` (fixed step count, fixed dt = the engine's fixed timestep), then render ONE
   frame: ground + player sphere (at its final settled position) + the remaining (uncollected) pickups as
   distinct meshes + a fixed camera framing the play area, lit + shadowed. Print
   `game: {score:3, won:true, steps:S, player:[x,y,z]}`. Deterministic. (Render the END state — by the
   end of the winning track all pickups are collected, so to make the golden visually show pickups, also
   render a couple of static obstacle markers OR capture at a FIXED mid-track step where some pickups
   remain — choose a fixed capture step that shows player + at least one remaining pickup so the image is
   legible; document the chosen capture step. The state assertion covers the full winning run.)

3. **Pickups + player rendering reuse the scene path.** No new RHI. Player = the engine sphere mesh
   scaled to the collider; pickups = small spheres (a distinct color/material); obstacles = the existing
   cube/sphere. Everything goes through the normal lit+shadowed scene render (the same path as
   `--physics-shot`). The camera is a fixed deterministic pose.

4. **Tests `tests/roll_game_test.cpp` (pure CPU, no GPU):**
   - **Determinism:** running `ScriptedTrack()` twice yields bit-identical final `GameState` (score,
     player position, step, collected flags).
   - **Win condition:** the scripted track collects all 3 pickups → `score == 3`, `won == true`.
   - **Collection logic:** a player placed exactly at a pickup collects it (and only it); a distant player
     collects none; a pickup is collected at most once.
   - **Physics integration:** the player actually moves under input (position changes in the input
     direction) and gravity keeps it on the ground (y ≈ radius, doesn't sink/explode — bound via
     `World::KineticEnergy`).
   - Clean under `windows-msvc-asan`.

5. **Verification golden.** New `tests/golden/metal/game.png` (Metal two runs DIFF 0.0000). Existing 28
   image goldens UNTOUCHED. Introspect rebaked exactly `+gameplay-sample` (features) + `--game-shot`
   (showcases).

## RHI seam additions (summary)
- **None.** Gameplay is pure CPU above physics+scene; rendering reuses the existing lit/shadowed scene
  path. New files (`engine/game/roll_game.{h,cpp}`, `tests/roll_game_test.cpp`) add ZERO backend symbols.
  Seam grep stays at the benign baseline (2 pre-existing factory dispatch lines).

## Out of scope (YAGNI)
Live keyboard play (interactive window), AI/enemies, a HUD/score text overlay (no text renderer yet),
audio, multiple levels, save/load of game state, a general ECS behavior/scripting system, win/lose UI,
respawn, timers. One scripted deterministic roll-a-ball-collect-3 sample, golden + state-asserted.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 29) + new `roll_game_test` (determinism, win
   condition, collection logic, physics integration). Clean under `windows-msvc-asan`.
2. `--game-shot` on Windows/Vulkan: controller visual review — the play area with the player sphere and
   remaining pickup(s) at the fixed capture step, lit + shadowed, coherent. The `game: {score,won,steps,
   player}` line is deterministic and shows the winning final state. Run under the AT Vulkan-validation
   gate → ZERO errors.
3. Metal: `visual_test --game` → new golden `tests/golden/metal/game.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `game.png` added; the
   other 28 byte-identical.
5. Introspect JSON rebaked exactly `+gameplay-sample` + `--game-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `game` image
   golden in the Mac round-trip loop.
