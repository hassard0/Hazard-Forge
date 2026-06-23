# Slice PT1 — Deterministic GPU particle emitter + integrator (Issue #19, flagship #19 beachhead)

NEW header `engine/sim/particles.h` (`hf::sim::particles`), `#include "sim/fpx.h"` + `#include "sim/grain.h"`
READ-ONLY. The 6th deterministic-sim family member. PT1 is the beachhead: a deterministic emitter + Q16.16
integrator with a fixed pool + a determinism-safe free-list. The MOAT = bit-identical CPU/Vulkan/Metal +
run-to-run reproducible (Niagara is float/non-deterministic). Follow the grain.h (GR1) mold exactly.

## Data structures
```cpp
struct FxParticle {            // 48 bytes, NO padding (memcmp-able); mirror grain::GrainParticle's 48B packing
    fpx::FxVec3 pos;           // Q16.16 world position
    fpx::FxVec3 vel;           // Q16.16 velocity (world units / second)
    fpx::fx     age = 0;       // Q16.16 seconds since spawn (0 at birth)
    fpx::fx     lifetime = 0;  // Q16.16 max age; dies when age >= lifetime
    uint32_t    seed = 0;      // spawn hash (drives initial vel + render hashColor; 0 == empty slot)
    uint32_t    flags = 0;     // bit0 = ALIVE
};
static_assert(sizeof(FxParticle) == 48, "FxParticle std430 layout");
inline constexpr uint32_t kFlagAlive = 1u;
inline constexpr uint32_t kNoSlot = 0xFFFFFFFFu;

struct ParticlePool {
    std::vector<FxParticle> particles;   // FIXED capacity (set at init; NEVER resized at runtime)
    std::vector<uint32_t>   freeList;    // LIFO stack of free slot indices (DETERMINISM-CRITICAL)
    uint32_t spawnCursor = 0;            // monotone spawn counter (feeds the hash; NEVER reset)
    uint32_t tick = 0;                   // current sim tick (snapshotted)
};
struct EmitterConfig { fpx::FxVec3 origin; fpx::fx ratePerTick; fpx::fx lifetime; fpx::fx speed; uint32_t emitterId; };
```
`InitParticlePool(capacity)`: all slots dead (seed=0, flags=0); `freeList` initialized DESCENDING
`[capacity-1 … 1 0]` so the first pop yields slot 0 (ascending birth order). Pure integer.

## Free-list determinism contract (the make-or-break of the whole flagship)
`freeList` is a LIFO stack. Spawn pops from the back (O(1)); death pushes the dead slot back. BOTH the pop
(in the single-thread emit pass) and the push (in the ascending-slot recycle pass) happen in FIXED host order
per tick → the slot a particle lands in is a pure function of (capacity, command stream), identical on every
peer/platform. **NO atomic cursor, EVER** (the grain.h cell-emit DET-CRUX applied to spawn). The spawn/emit
pass is HOST/single-thread-ordered; the integrate pass is per-slot-independent multi-thread.

## Functions
```cpp
// Pure uint32 wrapping avalanche, modeled on fract::FractReFractureHash. NO rand, NO clock, NO float.
inline uint32_t ParticleHash(uint32_t emitterId, uint32_t spawnIndex);
// Map the hash to a unit-ish integer direction from a FIXED ~13-entry host Q16.16 dir-table biased +Y (a
// fountain cone), scaled by speed via fxmul. NO trig, NO float. (Model on fract's split-dir table.)
inline fpx::FxVec3 EmitInitialVelocity(uint32_t seed, fpx::fx speed);
// Pop ONE free slot (LIFO); write a fresh ALIVE particle: seed=ParticleHash(emitterId, pool.spawnCursor++);
// pos=origin; vel=EmitInitialVelocity(seed,speed); age=0; lifetime=cfg.lifetime; flags=kFlagAlive. Free-list
// empty -> no-op (pool full, deterministic). Returns the slot or kNoSlot. SINGLE-THREAD host-ordered.
inline uint32_t EmitParticle(ParticlePool& pool, const EmitterConfig& cfg);
inline void Emit(ParticlePool& pool, const EmitterConfig& cfg);   // spawn ratePerTick particles (fixed loop)
// IntegrateParticle: if !alive return. vel += gravity*dt (component fxmul); vel -= FxScale(vel, dragK) (linear
// drag); pos += vel*dt; age += dt; if (age>=lifetime) flags &= ~kFlagAlive. Pure integer, fixed op order,
// per-particle INDEPENDENT. NO ground clamp (that's PT3). The shader copies THIS body VERBATIM.
inline void IntegrateParticle(FxParticle& p, const fpx::FxVec3& gravity, fpx::fx dragK, fpx::fx dt);
inline void IntegrateParticles(ParticlePool& pool, const fpx::FxVec3& gravity, fpx::fx dragK, fpx::fx dt);
// RecycleDead: single-thread, ASCENDING slot order — for each slot just marked dead (alive cleared AND
// seed!=0), push its index onto freeList + clear seed. Deterministic free-list maintenance.
inline void RecycleDead(ParticlePool& pool);
// One PT1 tick: Emit (single-thread) -> IntegrateParticles (multi-thread) -> RecycleDead (ascending) -> ++tick.
inline void StepEmitIntegrate(ParticlePool& pool, const EmitterConfig& cfg, const fpx::FxVec3& gravity,
                              fpx::fx dragK, fpx::fx dt);
```

## Showcase (`--pt1-emit-shot <out.bmp>` Vulkan / `--pt1-emit` Metal) — clone the GR1/grain-integrate block
Recipe (host-snapped Q16.16): capacity 256; emitter origin (0,0,0); ratePerTick 8; lifetime kOne*2; speed
kOne*4; gravity (0,-9.8,0) host-snapped; dragK kOne/50; dt kOne/60; **K=90 steps**. At K=90 the pool reaches
a steady-state fountain spread (particles at many distinct x,y — the GR1 "catch it mid-flight, not all on one
line" lesson). GPU: host does the deterministic Emit/RecycleDead between dispatches (tiny, host-side); the
GPU dispatch is the K-step `particles_integrate.comp` (int64 → gravity*dt+drag overflow int32 → VULKAN-ONLY,
the FL1/GR1/CL1 lesson). Metal `--pt1-emit` runs the CPU StepEmitIntegrate byte-identical by construction.
The buffer is the 48-byte FxParticle array (reuse the grain shader-buffer plumbing).

EXACT integer proof lines (fail loudly):
```
pt1-emit: {capacity:256, spawned:S, alive:A, steps:90} GPU==CPU BIT-EXACT
pt1-emit determinism: two runs BYTE-IDENTICAL
pt1-emit no-op: emitEnabled=false -> pool UNCHANGED (0 spawned, 0 alive)
pt1-emit lifecycle: spawned S, died D, alive A == S-D (free-list churn balanced)
```
Assertions: (1) full-buffer `memcmp(gpuPool, cpuPool)==0` NO tolerance; (2) two GPU runs memcmp==0; (3)
emitEnabled=false → 0 spawned, all-dead; (4) track spawned/died counters, assert `alive == spawned-died` AND
`freeList.size() == capacity - alive` (the free-list-determinism signature check).
**Strict-zero 2D side-view golden:** each ALIVE particle `(pos.x>>kFrac, pos.y>>kFrac)` → pixel via the GR1
fixed integer transform, drawn as a hashColor(seed) dot (the fountain plume side-view). Identical both
backends BY CONSTRUCTION. Register `pt1_emit` in verify.ps1 $Goldens + add `--pt1-emit-shot` to $vkShots.

## Constraints (HARD)
- NEW header particles.h, append-only as the flagship grows (PT1 is the first content). fpx.h/grain.h
  READ-ONLY (do not modify). NO existing shader/golden/file changed except the additive showcase + verify.ps1
  registration + the new shader.
- Determinism: pure Q16.16/integer in the sim path. NO float, NO rand, NO clock. NO atomic cursor in the
  free-list. Spawn host-single-thread-ordered; integrate per-particle-independent.
- Branch `fix-issue-19-pt1`, commit there, do NOT merge, do NOT commit any `tests/golden/metal/*` (controller
  bakes). DO register the golden name in verify.ps1.
- Build Windows: `cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`
  (vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`);
  also build+run a particles ctest if you add one (model on grain_test). 
- COMPLETION CRITERIA — do NOT commit until: `--pt1-emit-shot` runs exit 0 AND all 4 proof lines print with
  GPU==CPU BIT-EXACT and the free-list invariant holding (spawned/died/alive consistent). If the GPU != CPU,
  the int64 shader or the emit ordering is wrong — fix before committing.
- If main.cpp arg-parse hits MSVC C1061 (blocks nested too deeply), give the flag its OWN parse loop or fold
  into an existing branch with `||` (do NOT add a new else-if to the giant chain).
