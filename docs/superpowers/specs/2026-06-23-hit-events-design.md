# Issue #40 — Deterministic collision event callbacks (OnHit) — design

**Goal:** the deterministic rigid-body sim already computes contacts bit-exactly; surface them as a
deterministic `OnHit(actorA, actorB, point, impulse)` event list on the game layer so gameplay samples
stop re-rolling their own AABB checks. Pure CPU, NO GPU compute, NO new shader, NO new RHI.

## Key facts (already verified)
- `engine/game/verdict.h` (hf::game, the GAME LAYER — this is where new code goes) carries the embedded
  sim TRIPLE on `VerdictWorld`: `gjk::HullWorld sim; warmhull::HullCache cache; std::vector<warmhull::HullSleepState> sleep;`
- `verdict::StepWorld(world, ...)` calls the FROZEN `warmhull::StepWarmSleepHullWorld(world.sim, world.cache, world.sleep, cfg)`.
  That step **rewrites `world.cache` to EXACTLY this tick's solved contacts** (warmhull.h:767).
- `warmhull::HullCache { std::vector<CachedHullContact> entries; }`; `CachedHullContact { warmhull::HullContactKey key; fx normalImpulse; }`.
- `warmhull::HullContactKey { uint32_t bodyA; uint32_t bodyB; uint32_t refFaceId; uint32_t incVertId; }`
  — bodyA/bodyB are the GLOBAL sim body indices, ORDER-NORMALIZED (bodyA < bodyB). One entry per contact POINT.
- `verdict::BodyRef { uint32_t simBodyIndex; }` per entity (component); `world.order` is the FIXED live-entity id sequence.
- Body world position: `world.sim` (gjk::HullWorld) exposes the fpx bodies; a body's position is `body.pos` (FxVec3).
  Confirm the exact accessor (likely `world.sim.bodies[idx].pos` or `world.sim.world.bodies[idx]`) by reading gjk.h/warmhull.h.

## Design — APPEND-ONLY to verdict.h (VD1–VD6 are BYTE-FROZEN; do NOT modify any existing line, do NOT touch StepWorld)
Add, after the last VD slice:
```cpp
// Issue #40: a deterministic collision event (the per-pair OnHit gameplay reads after StepWorld).
struct HitEvent {
    EntityId a = kNoEntity;   // the two colliding entities (a < b by EntityId, deterministic order)
    EntityId b = kNoEntity;
    FxVec3   point{};         // contact location = deterministic midpoint of the two bodies' positions (Q16.16)
    fx       impulse = 0;     // summed normalImpulse over the pair's contact points (Q16.16) — collision strength
};

// Pure function of the POST-StepWorld state (world.cache + world.order + world.sim). Bit-identical
// cross-platform because world.cache is (the persist flagship proved it). Gameplay calls StepWorld then this.
inline std::vector<HitEvent> CollectHitEvents(const VerdictWorld& world);
```
Implementation of `CollectHitEvents`:
1. Build a FIXED-order reverse map sim-body-index -> EntityId by scanning `world.order` (skip entities without
   a bound BodyRef / simBodyIndex==kNoBody). If two entities map to the same body index that's a bug — first wins.
2. Iterate `world.cache.entries` in FIXED order. For each, resolve key.bodyA/key.bodyB -> EntityIds via the map
   (skip if either is unmapped, e.g. a static-only body with no gameplay entity). Aggregate per ENTITY-PAIR:
   sum `normalImpulse` across that pair's contact points (a manifold has up to 4 points -> one HitEvent per pair).
   Order the (a,b) pair canonically by EntityId (a < b).
3. For each aggregated pair, `point` = midpoint of the two bodies' positions: `FxScale(FxAdd(posA,posB), half)`
   where half = kOne>>1 (use the fpx Q16.16 helpers; NO float). Document honestly that this is the inter-body
   midpoint, not a per-contact manifold point (precise points are re-derivable via the manifold query later).
4. Return the events sorted by (a, b) ascending (a FIXED deterministic total order) so the list is canonical.

This adds NO field to VerdictWorld and modifies NO existing function -> VD4 snapshot/VD5 lockstep determinism
untouched (HitEvents are a derived per-tick OUTPUT, recomputed each call, never part of the snapshot).

## Showcase (model on the existing `--vd3-world` VD slice showcase — find it in BOTH renderers)
- Vulkan: `--hit-events-shot <out.bmp>` in `samples/hello_triangle/main.cpp`. **FOLD the flag into an existing
  arg-parse branch with `||` — do NOT add a new `else if` (the chain hits MSVC C1061 block-nesting).** Or place
  the parse next to the other VD `*-shot` flags if that block has room; verify it compiles.
- Metal: `--hit-events <out.png>` in `metal_headless/visual_test.mm` (a new `if (argc>1 && strcmp(argv[1],"--hit-events")==0)` dispatch is fine there — that file uses a flat if-chain, not the nested one).
- Scene: a small deterministic arena — a static support hull + 2–3 dynamic hulls bound to gameplay entities,
  dropped so they collide and settle (StepWorldN, e.g. 120–300 ticks). Then `CollectHitEvents`.
- Render: the SAME pure-integer 2D side-view as `--vd3-world` (hull outlines + id-tinted entity markers),
  PLUS a small marker at each HitEvent.point (so the collisions are visible). Bit-identical cross-backend BY
  CONSTRUCTION (strict zero-differing-pixel — the VD bar, NOT the float visresolve bar).
- Print these EXACT proof lines (fail loudly with a nonzero exit if any fails):
  1. `hit-events: {ticks:<K>, events:<N>, pairs:[(a,b)...], impulses:[...]} two-run BYTE-IDENTICAL`
     (run StepWorldN + CollectHitEvents TWICE from the same init -> the two HitEvent vectors memcmp-equal).
  2. `hit-events: deterministic order (sorted by (a,b), canonical)` — assert the returned list is sorted.
  3. `hit-events: every event maps to two LIVE bound entities (a<b, impulse>0)` — validate each event.
  4. `hit-events: zero collisions -> empty (no-op)` — a control scene with bodies far apart yields 0 events.

## Constraints (HARD)
- verdict.h: APPEND-ONLY. Do NOT edit any existing VD1–VD6 line. sim headers (fpx/gjk/convex/warmhull/persist/manifold)
  are READ-ONLY — do NOT modify them.
- Branch `fix-issue-40` off master. Commit there. Do NOT merge. Do NOT touch `scripts/verify.ps1`.
- Do NOT create or commit any Metal golden PNG (`tests/golden/metal/*`) — the controller bakes those on the Mac.
- Build Windows: from the repo root, `cmd /c "<vcvars64.bat path> && cmake --build build/windows-msvc-release --target hello_triangle"`
  (vcvars: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`).
  Also build + run the ctest for verdict if one exists. Render `--hit-events-shot` on Windows and confirm it
  exits 0 + the 4 proof lines print. Render a zero-collision control too.
- Report: the branch name, the commit hash, the exact proof-line output, and the new showcase flag names.
