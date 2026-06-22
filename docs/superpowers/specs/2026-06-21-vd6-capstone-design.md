# Slice VD6 — Deterministic Gameplay/Netcode: THE PLAYABLE LIT 3D CAPSTONE (the money-shot) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of
> FLAGSHIP #27 (DETERMINISTIC GAMEPLAY / NETCODE PRODUCT LAYER, `hf::game::verdict`). VD1-VD5 built the deterministic
> entity world, the gameplay systems, the composed `StepWorld` tick, the heterogeneous snapshot, and the
> whole-world lockstep/rollback. VD6 is the money-shot: it renders a **playable deterministic game** — a player
> rolling through a hull-stack arena, collecting pickups, the stack settling — as a LIT 3D scene, whose ENTIRE world
> (entities + gameplay + physics) is the bit-exact VD1-VD5 deterministic world and is lockstep-replayable. This is
> the artifact the whole flagship was built to produce: a real, rendered game that is deterministic and
> rollback-replayable across platforms — the thing a float engine cannot ship. The render is the ONE FLOAT crossing
> (outside the bit-exact integer loop), so its bar is the FLOAT visresolve in-band metric. PURE REUSE: the sim
> bodies render via the FROZEN `gjk::HullToRenderInstances`; the gameplay entities (pickups, the score-tinted
> player) overlay through the existing instanced-lit pipeline. ZERO new render shader, ZERO new RHI. APPEND to
> `engine/game/verdict.h` (VD1-VD5 + gjk/warmhull BYTE-FROZEN). Branch: `slice-vd6`. See [[hazard-forge-verdict-roadmap]],
> [[hazard-forge-warmhull-roadmap]], [[hazard-forge-docs-style]], [[hazard-forge-metal-showcase-gate]].

**Goal:** Extend `engine/game/verdict.h` (additive — VD1-VD5 byte-unchanged) with `VerdictToRenderInstances(const
VerdictWorld& world)` — the render payload for the game: the sim bodies via the frozen `gjk::HullToRenderInstances`
(the WH6/CD6 delegate, matte) PLUS the gameplay-entity overlays (pickup markers + a score-tinted player), a PURE
FUNCTION of the world (two calls byte-equal). Add the showcase `--vd6-game-shot <out>` (Vulkan) / `--vd6-game`
(Metal) — both build the playable arena scene, run the deterministic game (`StepWorldN` / a lockstep replay), and
draw the world LIT 3D through the EXISTING instanced-lit pipeline. Bake the float golden `vd6_game`. **NO new shader,
NO new RHI.** This is the FINAL slice → FLAGSHIP #27 COMPLETE.

## Design call: render the bit-exact deterministic game; the world stays integer-exact

The world is VD1-VD5's deterministic Q16.16 result (entities + components + the embedded sim). VD6 adds ONLY a
render bridge: it maps that frozen world to FLOAT geometry for display. The integer world is NOT mutated; the
provenance contract (two `VerdictToRenderInstances` calls on the same world → byte-equal output) proves the render
is a pure function of the deterministic game.
- **`VerdictToRenderInstances(const VerdictWorld& world)`** — the sim bodies via `gjk::HullToRenderInstances(world.sim)`
  (the frozen render bridge, render-only, OUTSIDE the bit-exact loop, MATTE to dodge the iridescence trap), plus the
  gameplay-entity overlays: a marker mesh per non-body entity at its `Transform2D.pos` (a pickup tinted by its
  type, the player tinted by its `Score`). A PURE FUNCTION of `world` (two calls byte-equal). The integer sim/world
  is untouched.
- **The scene = the playable arena.** Build a deterministic arena (a hull stack on a static support + a player
  entity bound to a dynamic hull body + pickup entities), run the deterministic game for N ticks (the player moves
  via commands, collects pickups, the stack settles), draw `VerdictToRenderInstances(world)` lit. The money-shot:
  a real deterministic game — a player among settling polyhedra, a score — rendered in lit 3D, the whole of which is
  lockstep-replayable (the VD5 guarantee).

> NOTE: VD6 runs the game ON THE CPU (the bit-exact `StepWorldN`) and renders the result — a normal draw, NOT a
> heavy compute dispatch, so NO TDR concern.

## Reuse map (file:line)
- **VD1-VD5 `engine/game/verdict.h` (APPEND after `RunVerdictRollback`):** `VerdictWorld`, `StepWorldN`,
  `RunVerdictLockstep`, `VerdictParams`, the components, `order[]`. VD1-VD5 frozen.
- **gjk.h render bridge (read-only — REUSE verbatim):** `gjk::HullToRenderInstances` (the settled-hull → float
  world-space triangle mesh, matte), `gjk::HullRenderMesh`/`HullRenderMeshEqual` (the provenance memcmp). The
  WH6/CD6/MF6 `--*-render` showcases are the precedent (the lit instanced 3D draw + the float visresolve-bar + the
  provenance proof). Mirror for `--vd6-game`.
- **Registration:** `scripts/verify.ps1` (append `vd6_game` to the Mac loop + `--vd6-game-shot` to `$vkShots`),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden — verify it
  stages as ` M `**), append to `tests/verdict_test.cpp`. (No new shader.)

## Design decisions (locked)
1. **APPEND to `engine/game/verdict.h`** (VD1-VD5 byte-frozen): `VerdictToRenderInstances` (the sim-body delegate +
   the entity overlays). Render-only float, OUTSIDE the bit-exact loop. **NO new shader, NO new RHI** (reuse the
   instanced-lit pipeline). gjk.h/warmhull.h and ALL sim headers UNCHANGED.
2. **Showcase `--vd6-game-shot <out>` (Vulkan) AND `--vd6-game` (Metal) — WIRE BOTH (grep your own `visual_test.mm`
   for `--vd6-game` BEFORE reporting DONE).** BOTH build the playable arena, run the deterministic game
   (`StepWorldN`, or a `RunVerdictLockstep` replay), draw `VerdictToRenderInstances(world)` LIT 3D (matte,
   directional light) through the EXISTING pipeline. Golden = `tests/golden/metal/vd6_game.png` (Mac-baked by the
   CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) provenance:** `vd6-game: {bodies:<B>, entities:<N>, tris:<T>} provenance two-calls BYTE-EQUAL`.
   - **(2) the headline:** `vd6-game: {settled:true, score:<S>, replayable:true}` — the rendered world is the
     deterministic game state (the stack settled, the score from collected pickups), AND a `RunVerdictLockstep`
     replica of the same scene is `VerdictStatesEqual` to it (the rendered game is lockstep-replayable).
   - **(3) sim-unmutated:** `vd6-game: world byte-unmutated by render (statesEqual:true)` — `VerdictStatesEqual` of
     the world before vs after the render call is true (the render is a pure read).
   - **(4) determinism:** the Vulkan render path exits 0 + writes the image; Metal two-run DIFF 0.0000 (controller).
   - Golden discipline: ONLY `tests/golden/metal/vd6_game.png`; do NOT commit it. Existing 238 goldens UNTOUCHED.
4. **Cross-backend bar:** the COMMITTED golden is the Mac-Metal bake; verify.ps1 re-renders + compares at 0.0000
   (same-backend determinism — the gate). The CONTROLLER measures the cross-vendor visresolve mean as a DIAGNOSTIC —
   a FLOAT render is in-band (~20-55, the WH6/CD6/MF6 lineage), NOT strict-zero.
5. **Tests — APPEND to `tests/verdict_test.cpp` (pure CPU):** `VerdictToRenderInstances` provenance (two calls
   byte-equal); the body/entity/triangle counts correct; the render of the played-out world vs a fresh world
   differ; the integer world is byte-unmutated by the render call (`VerdictStatesEqual` pre/post); the rendered
   scene is `RunVerdictLockstep`-replayable (the capstone composition). Clean under `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `verdict-render` (features) + `--vd6-game-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does (and verifies it stages).**

## RHI seam additions (summary)
- **None — and NO new shader.** Render reuses the existing instanced-lit pipeline (the sim bodies via the frozen
  `gjk::HullToRenderInstances`; the entity overlays via the existing marker/instanced draw). `engine/game/verdict.h`
  APPEND-only (VD1-VD5 frozen); gjk.h/warmhull.h + ALL sim headers + ALL shaders UNCHANGED. Report the seam: NO
  shaders/ change, no RHI change, no frozen-file edit, verdict.h append-only.

## Out of scope (YAGNI)
A general convex-hull triangulator (reuses the GJ6 canonical-hull meshes). A real input device / interactive
window (the showcase replays a fixed command script — the deterministic core, not live input). Networked transport.
VD6 claims ONLY: the bit-exact deterministic game world renders as a coherent LIT 3D playable scene (a player among
settling polyhedra + a score), the render is a PURE FUNCTION of the deterministic game (provenance byte-equal +
world byte-unmutated), the scene is lockstep-replayable, and the Metal bake is per-backend deterministic (two runs
0.0000) + cross-vendor in-band (float visresolve). This is the FINAL slice → FLAGSHIP #27 COMPLETE.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "verdict|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--vd6-game-shot` on Vulkan: the proofs + exit 0 under the conan validation layer → ZERO
   VUID. VERIFY a coherent LIT 3D playable scene — a player among the hull stack / pickups, under directional light,
   no iridescence, no garbage/NaN.
3. Metal: `visual_test --vd6-game` → `tests/golden/metal/vd6_game.png`; two runs DIFF 0.0000. Confirm `visual_test.mm`
   in the diff; NO shader added. Cross-vendor = FLOAT visresolve in-band (~20-55).
4. **Render-invariance:** ONLY `vd6_game.png` added; the other 238 byte-identical (+ controller introspect rebake,
   verified staged as ` M `).
5. Introspect: exactly `+verdict-render` + `--vd6-game-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + VD1-VD5 verdict.h code + gjk.h/warmhull.h + ALL sim headers + ALL shaders
   byte-unchanged; verdict.h APPEND-only; NO shaders/ change). `vd6_game` in the Mac loop + `--vd6-game-shot` in
   `$vkShots`. **FLAGSHIP #27 COMPLETE → consolidation #61 (full verify.ps1 + ARCHITECTURE verdict section +
   golden/ctest count bumps).**
