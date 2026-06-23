# FR7 — Recursive Fracture-on-Impact (Issue #37) — implementation-ready spec

Append **Slice FR7** to `engine/sim/fract.h` (after `FractToRenderInstances` ~line 917, inside
`namespace hf::sim::fract`, before the namespace close). FR1–FR6 are BYTE-FROZEN; do NOT modify them.
Sibling sim headers (fpx/gjk/convex/warmhull/persist/manifold) are READ-ONLY.

## Honest framing (state this in the FR7 doc-block)
Issue #40's `warmhull::HullCache.normalImpulse` / `verdict::CollectHitEvents` live on the **warmhull HULL
solver**. The FR4 rubble world is an `fpx::FxWorld` solved by `fpx::SolveContacts` (sphere-sphere), which
does NOT populate `world.cache`. FR7 therefore triggers on the **sphere-contact impulse proxy** computed the
same way `fpx::SolveContacts` does — the faithful sphere-world analog of the #40 hull impulse. A true
hull-rubble recursion reading `normalImpulse` verbatim is a future flagship. Do NOT over-claim FR7 "uses the
#40 field". Children are sub-SPHERES (FR4's bounding-sphere collider), not re-tessellated Voronoi shards —
an honest simplification consistent with the existing flagship; document it.

## New types/functions (all in hf::sim::fract, APPEND-ONLY)
```cpp
struct FractChunk { int32_t volume=0; uint32_t depth=0; uint32_t parentId=0; uint32_t chunkId=0; uint8_t retired=0; };
inline constexpr uint32_t kNoChunk = 0xFFFFFFFFu;
struct FractRecursiveWorld { fpx::FxWorld world; std::vector<FractChunk> chunks; uint32_t nextChunkId=0; };
struct FractRecursiveConfig { fx worldCellSize=fpx::kOne/4; fx reFractureImpulse=3*fpx::kOne; int32_t minVolume=8; int childPieces=2; fx gravityY=0; fx groundY=0; };
struct FractChild { fpx::FxBody body; FractChunk chunk; };
struct FractCascadeState { uint32_t liveChunks=0, retired=0, maxDepth=0; int32_t minLiveVolume=0; uint32_t atFloor=0; };

inline uint32_t FractReFractureHash(uint32_t parentChunkId, uint32_t tick, uint32_t salt); // fixed shift/xor/add avalanche, PURE INT32, NO rand
inline std::vector<fx> AccumulateContactImpulse(const fpx::FxWorld& world);                // per-body Q16.16 impulse this tick (T1, see below)
inline std::vector<FractChild> SplitChunk(uint32_t parentIndex, const FractRecursiveWorld&, const FractRecursiveConfig&, uint32_t tick);
inline FractRecursiveWorld BuildRecursiveWorld(const fpx::FxWorld& spawnWorld, const FractFragments&, const std::vector<uint32_t>& clusters);
inline void StepFractureRecursive(FractRecursiveWorld&, const FractRecursiveConfig&, fx dt, int solveIters, uint32_t tick);
inline void StepFractureRecursiveSteps(FractRecursiveWorld&, const FractRecursiveConfig&, fx dt, int solveIters, int steps);
inline FractCascadeState MeasureFractCascade(const FractRecursiveWorld&, const FractRecursiveConfig&);
```

## Algorithm — plane-split (CHOSEN over re-Voronoi: simplest fully-deterministic, clearly terminates)
A chunk is a bounding sphere (`pos,radius,invMass,volume`). On a hard impact, split into `childPieces=2` sub-spheres:
- split-plane normal `n` = a FIXED integer direction-table entry (6 or 13 host Q16.16 unit-ish dirs, NO trig) indexed by `FractReFractureHash(parentChunkId,tick,salt)`.
- `child.radius = fxmul(parent.radius, kCbrtHalf)` where `kCbrtHalf` = host-precomputed Q16.16 literal ≈0.7937 (=2^(−1/3); NO runtime cbrt).
- `child.volume = parent.volume / childPieces` (integer divide); `child.invMass = FractInvMass(child.volume)` (FR2 helper reused).
- `child.pos = parent.pos ± n·(parent.radius/2 + kSeparation)` (kSeparation = host Q16.16 const); `child.vel=parent.vel; child.angVel=parent.angVel; child.orient=parent.orient; flags=kFlagDynamic`.

## Trigger (T1) — sphere-contact impulse proxy from the FR4 solve (pure int64, fixed pair order)
`AccumulateContactImpulse`: re-run `fpx::BuildPairs(world,…)` (the SAME broadphase FR4 uses), and for each
overlapping pair sum into BOTH bodies a Q16.16 magnitude = `fxmul(closingSpeed, overlapDepth)` where
`closingSpeed = max(0, -(vB−vA)·n)`, `overlapDepth = (rA+rB) − |posB−posA|`, all via fpx int64 helpers
(FxSub/FxDot/FxLength/fxmul). Threshold `cfg.reFractureImpulse` (host-snapped Q16.16, tuned so a hard slam
exceeds it and a gentle settle does not; start 3.0). Use a fixed `>` compare.

## StepFractureRecursive — ONE deterministic tick
1. `StepFracture(world, dt, solveIters)` — the FR4 tick VERBATIM (frozen).
2. `imp = AccumulateContactImpulse(world)`.
3. Capture `size_t n0 = world.bodies.size()` BEFORE splitting. Scan indices `[0,n0)` in ASCENDING order;
   to-split = dynamic AND !retired AND `volume > cfg.minVolume` AND `imp[i] > cfg.reFractureImpulse`.
4. For each to-split index (ascending): **retire parent IN PLACE** (`invMass=0; flags&=~kFlagDynamic; chunks[i].retired=1` — NEVER erase, keeps indices+chunks[] aligned) then append children via SplitChunk (`push_back` body+chunk; `chunk.chunkId=nextChunkId++; depth=parent.depth+1; parentId=parent.chunkId`). Children spawned this tick are NOT re-evaluated until next tick (iterate the n0 snapshot).
Guards: `minVolume=max(1,minVolume)`, `childPieces>=2`. Retired bodies excluded from the live/render set.

## Determinism (two runs + both backends bit-identical)
Trigger is int64 over BuildPairs' fixed order (same primitives as cross-backend-proven FR4). Split is
table-lookup normal + fxmul + integer divide, lineage-hashed. Ascending-index iteration over the n0 snapshot
+ append-only + monotonic nextChunkId → identical lineage on every peer/run. StepFracture reused verbatim.
BREAKS IT: threshold straddling (fixed `>`, fixed accum order, tune clear of settle band); erase-instead-of-
retire (index shift → desync — MANDATORY retire-in-place); re-evaluating fresh children same tick (use n0);
float in kCbrtHalf/kSeparation/dir-table (host integer literals only); minVolume<1 or childPieces<2.

## Showcase — `--fract-recursive-shot` (Vulkan) / `--fract-recursive` (Metal)
Model on `RunFractStepShowcase` (visual_test.mm ~18629) + Vulkan `--fract-step-shot`. The recursion logic is
PURE-CPU host code on BOTH backends (like FR4 bond/break); the per-tick StepFracture drives the FR4 int64
`fpx_solve.comp` Vulkan-side / CPU Metal-side → byte-identical by construction. Same FR1 lattice + 16-seed
scene as `--fract-step`. **Golden = strict-zero integer 2D side-view** `tests/golden/metal/fract_recursive.png`
(settled bodies as discs colored by `depth`, retired parents not drawn) — NOT a float lit render. Two runs
DIFF 0.0000, Vulkan==Metal exactly.
EXACT proof lines (fail the run on any violated assertion):
```
fract-recursive: {root-bodies:%u, ticks:%d, final-bodies:%u, max-depth:%u} GPU==CPU BIT-EXACT
fract-recursive cascade: piece-count {tick0:%u, tickK/4:%u, tickK/2:%u, final:%u}
fract-recursive determinism: two runs BYTE-IDENTICAL
fract-recursive floor: min-live-volume:%u <= minVolume:%u reached, at-floor:%u chunks (none split further)
fract-recursive trigger: hard={refractured:%u, max-depth:%u} soft={refractured:0, depth:0}
```
Assertions: cascade liveChunks non-decreasing AND final>tick0 (cascade happened); two-run memcmp over
world.bodies AND chunks == 0; minLiveVolume<=minVolume AND atFloor>0 AND N extra ticks add ZERO bodies
(terminate); soft control (very high reFractureImpulse) → refractured==0 AND final==root (impulse-driven, not
unconditional). Add FR7 cases to `tests/fract_test.cpp` modeled on the FR4 cases.

## Constraints (HARD)
fract.h APPEND-ONLY (FR1-6 byte-frozen); sim headers read-only; FxBody frozen → lineage in the parallel
FractChunk[] only (NEVER add a field to FxBody); retire-in-place never erase. Branch `fix-issue-37`, commit
there, do NOT merge, do NOT touch verify.ps1, do NOT create/commit any `tests/golden/metal/*` PNG (controller
bakes on the Mac). Arg-parse in main.cpp: place `--fract-recursive-shot` beside the other `--fract-*-shot`
flags; if it triggers MSVC C1061 (blocks nested too deeply) fold it into an existing branch with `||`.
Build: `cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`
(vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`);
build+run the fract ctest; run `--fract-recursive-shot` on Windows, confirm exit 0 + all proof lines.
