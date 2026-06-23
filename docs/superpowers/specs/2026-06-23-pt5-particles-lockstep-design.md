# Slice PT5 — Lockstep + rollback (the NETCODE HEADLINE) (Issue #19, flagship #19, 5th slice)

APPEND to `engine/sim/particles.h` (PT1-PT4 BYTE-FROZEN). PT5 is the moat headline: two peers fed ONLY an
input-command stream re-derive the EXACT particle pool bit-for-bit, and a rollback re-sims from a snapshot
bit-exact. PURE CPU — NO GPU dispatch, NO new shader, NO new RHI. It's a determinism PROPERTY of the bit-exact
PT4 StepParticles; the cross-backend zero-diff golden IS the lockstep evidence. Mirrors the grain/fpx lockstep
harness VERBATIM in shape (grain::GrainCommand/ApplyGrainCommand/SimGrainTick/SnapshotGrain/RestoreGrain/
RunGrainLockstep/RunGrainRollback — READ them for the exact pattern).

## Append to particles.h
```cpp
// A deterministic per-tick INPUT command (NOT state). Applied before StepParticles on its tick.
inline constexpr uint32_t kCmdBurst      = 0u; // spawn `arg.x` extra particles at point arg... (a burst)
inline constexpr uint32_t kCmdGust       = 1u; // add a velocity delta (arg) to ALL alive particles (a wind gust)
inline constexpr uint32_t kCmdMoveEmitter= 2u; // relocate the emitter origin to arg (point)
struct ParticleCommand { uint32_t tick = 0; uint32_t kind = kCmdBurst; FxVec3 arg{}; int32_t argi = 0; };

// ApplyParticleCommand(pool, cfg, cmd): mutate pool/cfg deterministically. kCmdBurst -> EmitParticle argi
// times at arg (single-thread, free-list LIFO — reuse EmitParticle with a temp cfg whose origin=arg,
// rate=argi); kCmdGust -> for each ALIVE particle vel += arg; kCmdMoveEmitter -> cfg.origin = arg. FIXED order.
inline void ApplyParticleCommand(ParticlePool& pool, EmitterConfig& cfg, const ParticleCommand& cmd);

// SimParticleTick: apply THIS tick's commands (those with .tick == pool.tick) in ARRAY ORDER, then
// StepParticles. The deterministic per-tick advance fed by the input stream.
inline int SimParticleTick(ParticlePool& pool, EmitterConfig& cfg, const ParticleCommand* cmds, uint32_t cmdCount,
                           const ForceField* fields, uint32_t fc, const FxVec3& g, fx dragK, fx dt,
                           fx groundY, fx radius, fx e, const ParticleSphereCollider* spheres, uint32_t sc);

// THE SNAPSHOT CRUX: capture the ENTIRE pool — particles + freeList + spawnCursor + tick — because spawn/death
// IS the sim. A snapshot that omits the free-list/cursor re-spawns into WRONG slots on restore -> divergence.
struct ParticleSnapshot { std::vector<FxParticle> particles; std::vector<uint32_t> freeList; uint32_t spawnCursor; uint32_t tick; EmitterConfig cfg; };
inline ParticleSnapshot SnapshotParticles(const ParticlePool& pool, const EmitterConfig& cfg);  // deep copy ALL
inline void RestoreParticles(ParticlePool& pool, EmitterConfig& cfg, const ParticleSnapshot& s); // overwrite ALL
inline bool ParticleStatesEqual(const ParticlePool& a, const ParticlePool& b);                   // full compare

// RunParticleLockstep(init, cfg, stream, T, ...): clone two peers from `init` (RestoreParticles, not copy if
// the pool is non-copyable — here it's copyable, but use Snapshot/Restore for parity with the family); both
// run SimParticleTick over the SAME command stream for T ticks; assert ParticleStatesEqual every tick (or at
// the end). Returns the converged authority snapshot. (The inputs-only re-derivation = the lockstep proof.)
inline ParticleSnapshot RunParticleLockstep(const ParticleSnapshot& init, const ParticleCommand* stream,
                                            uint32_t streamCount, uint32_t T, /*scene params...*/);
// RunParticleRollback: advance to rollbackAt, SnapshotParticles, speculatively mispredict <=N ticks with a
// WRONG stream (a divergent command), RestoreParticles back, re-sim the CORRECT stream -> corrected==authority
// BIT-EXACT AND the pre-rollback mispredicted state DIFFERED (positive + negative control).
inline /*result*/ RunParticleRollback(/*init, streams, rollbackAt, ...*/);
```

## Showcase `--pt5-lockstep-shot` (Vulkan) / `--pt5-lockstep` (Metal) — PURE CPU, NO GPU
Both backends run the IDENTICAL CPU harness over a PT4-style scene (emitter + a vortex field + ground + a
sphere) + a fixed command stream (a kCmdGust at a known tick shoves the spray sideways; a kCmdBurst adds a
puff; lifetime kOne*3, T≈240). Render the converged AUTHORITY pool via the PT4 side-view VERBATIM (ground line
+ sphere outlines + hashColor(seed) dots). Bit-identical Vulkan-Windows == Metal-Mac BY CONSTRUCTION (pure
CPU, same code, same integer pool). **Scene + stream IDENTICAL in both renderers (the PT2 lesson).**

EXACT proof lines (fail loudly):
```
pt5-lockstep: replica==authority {alive:A, ticks:T, commands:C} BIT-EXACT (inputs-only)
pt5-lockstep rollback: corrected==authority BIT-EXACT (mispredict@tick<M> diverged then converged)
pt5-lockstep determinism: two runs BYTE-IDENTICAL + snapshot round-trip exact
pt5-lockstep snapshot-completeness: omit freeList/spawnCursor -> diverges (the crux control)
```
Assertions: (1) two peers from the same init + stream -> ParticleStatesEqual at the end (and ideally each
tick); (2) rollback: corrected==authority AND mispredicted-intermediate != authority (real divergence fixed);
(3) two full runs byte-identical AND SnapshotParticles->RestoreParticles round-trip == original; (4) THE CRUX
CONTROL: a snapshot that captures ONLY particles (NOT freeList/spawnCursor) restored + re-advanced DIVERGES
from the full-snapshot reference (proves capturing the free-list/cursor is necessary — spawn/death is sim
state). Strict-zero golden `pt5_lockstep` in verify.ps1 $Goldens. (NO $vkShots entry needed if there's no GPU
dispatch — but add `--pt5-lockstep-shot` to $vkShots only if the Vulkan showcase still opens a device; if it's
pure-CPU + just writes a BMP, no validation pass is needed — match how the grain/vd lockstep slices register.)
Add a PT5 case to particles_test.cpp (lockstep inputs-only equal, rollback corrects, snapshot round-trip,
snapshot-completeness control).

## Constraints (HARD)
- particles.h APPEND-ONLY (PT1-PT4 byte-frozen). fpx.h/grain.h READ-ONLY. Pure Q16.16/integer. PURE CPU — NO
  GPU dispatch, NO new shader. The snapshot MUST capture particles + freeList + spawnCursor + tick + cfg.
- **Scene + command stream IDENTICAL in main.cpp AND visual_test.mm.**
- Branch `fix-issue-19-pt5`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`).
- COMPLETION CRITERIA — do NOT commit until: `--pt5-lockstep-shot` runs exit 0, all 4 proof lines print
  (lockstep equal, rollback corrects, determinism + round-trip, snapshot-completeness control diverges). The
  CONTROLLER pixel-compares Metal vs Vulkan (MUST be 0-diff) + eyeballs the PNG.
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
