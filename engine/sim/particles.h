#pragma once
// Slice PT1 — Deterministic GPU Particle Emitter + Integrator: a Q16.16 fixed-point PARTICLE POOL with a
// DETERMINISM-SAFE free-list emitter + a per-particle semi-implicit-Euler integrator (the BEACHHEAD of
// FLAGSHIP #19: DETERMINISTIC GPU PARTICLES — a bit-identical CPU/Vulkan/Metal AND run-to-run reproducible
// particle system, the 6th member of the deterministic-sim family (rigid FPX -> cloth -> fluid -> grain ->
// ... -> PARTICLES). The MOAT vs a float Niagara-style system: a particle emitter whose spawn slot, initial
// velocity, age, and death are a PURE FUNCTION of (capacity, command stream) — identical on every peer and
// platform. PT1 is ONLY the emitter + integrator beachhead: a fixed POOL + a determinism-safe FREE-LIST +
// gravity/linear-drag integrate. NO ground clamp (that is PT3), NO neighbours/collision, NO lockstep, NO
// float render — NO float on the bit-exact path. Pure CPU, header-only, NO device, NO backend symbols, NO
// <cmath>. Namespace hf::sim::particles. The STRUCTURAL TWIN of the GR1 integer beachhead (engine/sim/
// grain.h): a pure-integer per-particle update proven GPU==CPU BIT-EXACT, with a cross-backend BIT-IDENTICAL
// integer golden — over a particle POOL with a deterministic emitter on top.
//
// TWO DELTAS vs the GrainParticle twin the PT1 spec locks (everything else is the IntegrateGrainParticle
// body shape): (1) a deterministic EMITTER + FREE-LIST — particles are SPAWNED/RECYCLED (a grain block is
// static-population), so the make-or-break is the free-list determinism (single-thread host-ordered spawn,
// ascending-slot recycle, NO atomic cursor); (2) a `seed`/`age`/`lifetime`/`flags` field set (the spawn hash
// drives the initial velocity + the render color, age vs lifetime drives death) replacing grain's
// prev/invMass/radius — still 48-byte std430 packing (memcmp-able).
//
// The integrator shader shaders/particles_integrate.comp.hlsl copies IntegrateParticle's per-particle math
// VERBATIM (the gravity*dt + linear-drag + integrate + age/death), so tests/particles_test.cpp + the GPU
// pass exercise the EXACT math — which is what makes the integrated particle array bit-identical GPU==CPU AND
// cross-backend. The emit + recycle stay HOST-side (between the GPU integrate dispatches) — they are the
// deterministic single-thread passes; only the per-particle-INDEPENDENT integrate is the GPU dispatch.
//
// THE CROSS-BACKEND CRUX (the make-or-break for GPU==CPU, like grain.h's host-snapped GrainParticle array):
// the GPU consumes host-snapped Q16.16 INTEGERS (the FxParticle array) and does ZERO floating point — every
// step is `(int64)a*b >> kFrac` (an ARITHMETIC right shift on int64, deterministic + identical on every
// compiler/vendor) + integer add + integer compare. In the integrate pass each particle is INDEPENDENT (no
// inter-particle coupling), so the GPU per-thread write is order-independent / race-free with NO atomics, and
// two runs are byte-identical.
//
// THE int32-vs-int64 DECISION (the GR1/FL1/CL1 lesson, documented): the integrate is `vel += gravity*dt;
// vel -= vel*dragK; pos += vel*dt`, all componentwise fxmul — the SAME form as grain_integrate.comp /
// fluid_integrate.comp, which need int64 because the (int64)a*b product before the >>kFrac shift exceeds
// int32 for Q16.16 gravity*dt (gravity ≈ -9.8*65536 = -642253; products of two Q16.16 world-scale values
// blow past 2^31). To stay bit-exact to this int64-intermediate reference WITHOUT any overflow fragility,
// shaders/particles_integrate.comp.hlsl uses int64 (like grain_integrate.comp) and is therefore VULKAN-
// SPIR-V-ONLY (glslc — the Metal HLSL->SPIR-V->MSL frontend — cannot parse int64_t in HLSL), NOT in the
// Metal hf_gen_msl list; the Metal --pt1-emit showcase runs the CPU StepEmitIntegrate (the SAME bit-exact
// reference the Vulkan GPU==CPU memcmp compares against) -> byte-identical to the Vulkan GPU result BY
// CONSTRUCTION. Same established convention as grain_integrate.comp (grain.h:31-40).
//
// REUSE MAP (file:line): the Q16.16 toolbox is engine/sim/fpx.h — fx (int32 Q16.16, fpx.h:46), fxmul
// (fpx.h:54, the int64-intermediate multiply), FxVec3 + FxAdd/FxSub/FxScale (fpx.h:59-72), kOne/kFrac
// (fpx.h:47-48). The integer-avalanche ParticleHash copies fract.h::FractReFractureHash (fract.h:1023-1029);
// the EmitInitialVelocity dir-table copies fract.h::FractSplitDir (fract.h:1036-1057), biased +Y for a
// fountain cone. The per-particle integrate mirrors grain.h::IntegrateGrainParticle (grain.h:155-174) without
// the ground rest, with the linear-drag term — READ, NOT modified (particles is the additive sibling; fpx.h
// #included read-only + stays byte-unchanged, exactly as grain.h:51 does).

#include <cstdint>
#include <vector>

#include "sim/fpx.h"     // read-only: fx / fxmul / FxVec3 / FxAdd / FxSub / FxScale / kOne / kFrac
#include "sim/grain.h"   // read-only: the GR1 mold this slice is modeled on (no symbol used; #included for
                         // documentation parity per the spec, like fluid.h/cloth.h cross-reference siblings).

namespace hf::sim {
namespace particles {

// Reuse the fpx Q16.16 scalar + vector toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::fxmul;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxScale;
inline constexpr int kFrac = fpx::kFrac;     // Q16.16 fractional bits (== fpx::kFrac, MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;      // 1.0 in Q16.16 (65536)

// ----- The particle (the std430 GPU mirror; the GrainParticle 48-byte packing discipline) ----------------
// A single particle. std430-packable as plain int32s (pos.xyz, vel.xyz, age, lifetime, seed, flags, +2
// reserved) = 12 x 4-byte = 48 bytes, NO padding holes (memcmp-able; the GPU FxParticle mirror, the
// GrainParticle 48-byte packing discipline). Treats FxVec3 as 3 plain int32s (NOT a 16-byte-aligned vec3), so
// array stride 48 is a multiple of the 4-byte scalar alignment. The two reserved int32s round the struct to
// the 48-byte std430 stride (GrainParticle's prev/invMass/radius space) — RESERVED for later PT slices (e.g.
// PT2 size/PT3 collision radius); ZEROED at init + carried byte-identically (they are part of the memcmp).
//   * pos      : current Q16.16 world position.
//   * vel      : Q16.16 velocity (world units / second).
//   * age      : Q16.16 seconds since spawn (0 at birth; +dt each integrate step).
//   * lifetime : Q16.16 max age — the particle DIES (alive bit cleared) when age >= lifetime.
//   * seed     : the spawn hash — drives the initial velocity (EmitInitialVelocity) + the render hashColor.
//                seed == 0 marks an EMPTY slot (a dead recycled slot has seed cleared back to 0).
//   * flags    : bit0 = ALIVE (the particle is live + integrated). An empty slot has flags 0.
//   * rsv0/rsv1: reserved (0) — the 48-byte std430 stride padding (carried byte-identically by the memcmp).
struct FxParticle {
    FxVec3   pos;             // Q16.16 current position
    FxVec3   vel;             // Q16.16 velocity (world units / second)
    fx       age = 0;        // Q16.16 seconds since spawn (0 at birth)
    fx       lifetime = 0;   // Q16.16 max age (dies when age >= lifetime)
    uint32_t seed = 0;       // spawn hash (drives initial vel + render hashColor; 0 == empty slot)
    uint32_t flags = 0;      // bit0 = ALIVE
    int32_t  rsv0 = 0;       // reserved (0) — 48-byte std430 stride padding (PT2+/memcmp-carried)
    int32_t  rsv1 = 0;       // reserved (0)
};
static_assert(sizeof(FxParticle) == 48, "FxParticle std430 layout");

inline constexpr uint32_t kFlagAlive = 1u;          // bit0: the particle is live (integrated each step)
inline constexpr uint32_t kNoSlot    = 0xFFFFFFFFu; // EmitParticle returns this when the pool is full

// ----- The particle pool: a FIXED-capacity slot array + a DETERMINISM-CRITICAL free-list ------------------
// `particles` is a FIXED capacity (set at init; NEVER resized at runtime — the GPU buffer is fixed). `freeList`
// is a LIFO stack of free slot indices: spawn pops the back (O(1)); death pushes the dead slot back. `spawnCursor`
// is a monotone spawn counter that feeds the hash (NEVER reset). `tick` is the current sim tick (snapshotted).
struct ParticlePool {
    std::vector<FxParticle> particles;   // FIXED capacity (set at init; NEVER resized at runtime)
    std::vector<uint32_t>   freeList;    // LIFO stack of free slot indices (DETERMINISM-CRITICAL)
    uint32_t spawnCursor = 0;            // monotone spawn counter (feeds the hash; NEVER reset)
    uint32_t tick = 0;                   // current sim tick (snapshotted)
};

// ----- The emitter config (host-snapped Q16.16 knobs) ----------------------------------------------------
struct EmitterConfig {
    FxVec3   origin;             // Q16.16 spawn position (the fountain mouth)
    fx       ratePerTick = 0;    // particles to spawn per Emit call (the fixed loop bound; an integer count in Q16.16? -> use the integer part)
    fx       lifetime    = 0;    // Q16.16 max age handed to each new particle
    fx       speed       = 0;    // Q16.16 initial speed scale (EmitInitialVelocity)
    uint32_t emitterId   = 0;    // a salt so multiple emitters hash to distinct streams
};

// ----- InitParticlePool: all slots dead, free-list DESCENDING so the first pop yields slot 0 --------------
// Every slot starts EMPTY (seed=0, flags=0, zero pos/vel/age/lifetime). `freeList` is initialized DESCENDING
// `[capacity-1 ... 1 0]` so the first LIFO pop (from the BACK) yields slot 0 -> particles land in ASCENDING
// birth order (slot 0, 1, 2, ...) the very first tick (a clean, replay-stable layout). Pure integer.
inline ParticlePool InitParticlePool(uint32_t capacity) {
    ParticlePool pool;
    pool.particles.assign((size_t)capacity, FxParticle{});
    pool.freeList.assign((size_t)capacity, 0u);
    for (uint32_t i = 0; i < capacity; ++i)
        pool.freeList[(size_t)i] = capacity - 1u - i;   // [cap-1, cap-2, ..., 1, 0]; back() == 0
    pool.spawnCursor = 0;
    pool.tick = 0;
    return pool;
}

// ----- ParticleHash: a pure uint32 wrapping avalanche (modeled on fract::FractReFractureHash) -------------
// A fixed integer avalanche over the emitter id + the monotone spawn index — drives the initial velocity dir
// + the render hashColor, so spawns at the same (emitterId, spawnIndex) pick the same replay-stable stream.
// Pure uint32 wrapping arithmetic (defined, identical on every vendor/compiler) — NO RNG, NO clock, NO float.
// The xorshift/multiply mixing is the standard integer hash shape (fract::FractReFractureHash verbatim shape).
inline uint32_t ParticleHash(uint32_t emitterId, uint32_t spawnIndex) {
    uint32_t h = emitterId * 2654435761u;          // Knuth multiplicative
    h ^= (spawnIndex + 0x9E3779B9u + (h << 6) + (h >> 2));
    h += spawnIndex * 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}

// ----- The fixed +Y-biased fountain direction table (13 host Q16.16 entries — NO trig, NO float) ----------
// kEmitDirCount unit-ish integer directions, ALL with a positive (or zero) Y bias so the emitter throws
// particles UPWARD in a cone (a fountain plume). They need NOT be exactly unit (the velocity is scaled by
// speed via fxmul); host integer literals only -> deterministic + cross-vendor identical. Modeled on
// fract::FractSplitDir, but every entry biased +Y (a fountain, not an isotropic split). Index = hash % count.
inline constexpr int kEmitDirCount = 13;
inline FxVec3 EmitDir(uint32_t idx) {
    const uint32_t i = idx % (uint32_t)kEmitDirCount;
    // Q16.16 host literals (NO sqrt): d ~0.707 (face diagonal), c ~0.577 (corner), b ~0.354 (a shallow tilt).
    const fx d = (fx)((int64_t)kOne * 181 / 256);   // ~0.707
    const fx c = (fx)((int64_t)kOne * 148 / 256);   // ~0.577
    const fx b = (fx)((int64_t)kOne *  91 / 256);   // ~0.355
    switch (i) {
        case 0:  return FxVec3{ 0,        kOne, 0        };   // straight up (the core jet)
        case 1:  return FxVec3{ b,        d,    0        };   // tilted +x
        case 2:  return FxVec3{-b,        d,    0        };   // tilted -x
        case 3:  return FxVec3{ 0,        d,    b        };   // tilted +z
        case 4:  return FxVec3{ 0,        d,   -b        };   // tilted -z
        case 5:  return FxVec3{ b,        d,    b        };   // +x +z corner
        case 6:  return FxVec3{ b,        d,   -b        };   // +x -z corner
        case 7:  return FxVec3{-b,        d,    b        };   // -x +z corner
        case 8:  return FxVec3{-b,        d,   -b        };   // -x -z corner
        case 9:  return FxVec3{ c,        c,    c        };   // wide corner spray
        case 10: return FxVec3{-c,        c,    c        };
        case 11: return FxVec3{ c,        c,   -c        };
        default: return FxVec3{-c,        c,   -c        };
    }
}

// ----- EmitInitialVelocity: map the spawn hash to a +Y-biased direction scaled by speed -------------------
// Pick a fountain direction from the fixed table by the hash, then scale by `speed` (FxScale, componentwise
// fxmul). NO trig, NO float. The +Y bias means every particle launches upward -> gravity pulls them back into
// the fountain arc. Pure integer.
inline FxVec3 EmitInitialVelocity(uint32_t seed, fx speed) {
    return FxScale(EmitDir(seed), speed);
}

// ----- EmitParticle: pop ONE free slot (LIFO) + write a fresh ALIVE particle (SINGLE-THREAD host-ordered) --
// Pop the back of the free-list (O(1)). Compute seed = ParticleHash(emitterId, pool.spawnCursor++) — the
// monotone cursor advances EXACTLY once per spawn so the hash stream is replay-stable. Write a fresh particle:
// pos = origin; vel = EmitInitialVelocity(seed, speed); age = 0; lifetime = cfg.lifetime; flags = kFlagAlive.
// Free-list EMPTY (pool full) -> no-op, return kNoSlot (deterministic full-pool behaviour). Returns the slot.
// THE DETERMINISM CONTRACT: this is called from the SINGLE-THREAD host emit pass in a FIXED order, so the slot
// a particle lands in is a pure function of (capacity, command stream) — NO atomic cursor, EVER.
inline uint32_t EmitParticle(ParticlePool& pool, const EmitterConfig& cfg) {
    if (pool.freeList.empty()) return kNoSlot;          // pool full -> deterministic no-op
    const uint32_t slot = pool.freeList.back();
    pool.freeList.pop_back();
    const uint32_t seed = ParticleHash(cfg.emitterId, pool.spawnCursor);
    ++pool.spawnCursor;                                 // monotone, NEVER reset
    FxParticle p{};
    p.pos      = cfg.origin;
    p.vel      = EmitInitialVelocity(seed, cfg.speed);
    p.age      = 0;
    p.lifetime = cfg.lifetime;
    p.seed     = seed | 1u;                             // ensure non-zero (0 == empty); OR in bit0 of the hash
    p.flags    = kFlagAlive;
    pool.particles[(size_t)slot] = p;
    return slot;
}

// ----- Emit: spawn ratePerTick particles this tick (a FIXED single-thread loop) --------------------------
// Spawn (integer part of) cfg.ratePerTick particles by calling EmitParticle in a fixed loop. A full pool
// stops early (EmitParticle returns kNoSlot, deterministically). The integer rate is `ratePerTick >> kFrac`
// (the caller host-snaps an integer count, e.g. 8 -> kOne*8; or passes a plain small integer count — we read
// the integer part). SINGLE-THREAD host-ordered (the spawn-determinism pass).
inline void Emit(ParticlePool& pool, const EmitterConfig& cfg) {
    // ratePerTick is a count: accept either a Q16.16-scaled count (>> kFrac) OR — if the caller passes a small
    // plain integer < kOne — that integer directly. Both host-snapped, deterministic. Use the Q16.16 integer
    // part when >= kOne, else the raw value (so cfg.ratePerTick = 8 spawns 8, and kOne*8 also spawns 8).
    int count = (cfg.ratePerTick >= kOne) ? (int)(cfg.ratePerTick >> kFrac) : (int)cfg.ratePerTick;
    if (count < 0) count = 0;
    for (int k = 0; k < count; ++k)
        EmitParticle(pool, cfg);
}

// ----- IntegrateParticle: the deterministic per-particle semi-implicit-Euler + drag step (SHADER math) ----
// For a single particle, if ALIVE (flags & kFlagAlive):
//   vel += gravity * dt        (component-wise fxmul)            — the velocity integrate
//   vel -= FxScale(vel, dragK) (component-wise fxmul)            — LINEAR drag (vel *= (1 - dragK))
//   pos += vel * dt            (component-wise fxmul)            — the position integrate
//   age += dt                                                    — advance the age
//   if (age >= lifetime) flags &= ~kFlagAlive                    — DEATH (cleared; recycle happens in RecycleDead)
// NO ground clamp (that is PT3). Pure integer, fixed op order, per-particle INDEPENDENT. Each particle is
// INDEPENDENT of every other (no inter-particle coupling in PT1), so the order over particles does NOT matter
// -> two-run bit-identical AND the GPU per-thread write is race-free with NO atomics. The shader runs THIS
// exact per-particle body. (The GrainParticle IntegrateGrainParticle body shape, without the ground rest +
// with the linear-drag term.)
inline void IntegrateParticle(FxParticle& p, const FxVec3& gravity, fx dragK, fx dt) {
    if (!(p.flags & kFlagAlive)) return;
    // (1) integrate velocity: vel += gravity * dt.
    p.vel.x += fxmul(gravity.x, dt);
    p.vel.y += fxmul(gravity.y, dt);
    p.vel.z += fxmul(gravity.z, dt);
    // (2) linear drag: vel -= vel * dragK  (== vel *= (1 - dragK)).
    p.vel.x -= fxmul(p.vel.x, dragK);
    p.vel.y -= fxmul(p.vel.y, dragK);
    p.vel.z -= fxmul(p.vel.z, dragK);
    // (3) integrate position: pos += vel * dt.
    p.pos.x += fxmul(p.vel.x, dt);
    p.pos.y += fxmul(p.vel.y, dt);
    p.pos.z += fxmul(p.vel.z, dt);
    // (4) age + death (NO ground clamp — PT3). Recycle of the freed slot happens in RecycleDead (ascending).
    p.age += dt;
    if (p.age >= p.lifetime) p.flags &= ~kFlagAlive;
}

// ----- IntegrateParticles: one integrate STEP over the whole pool (the make-or-break GPU reference) -------
// Apply IntegrateParticle to every slot once. The reference the GPU memcmp's against. Order-independent
// (particles are independent in PT1) -> bit-identical regardless of GPU scheduling. Dead/empty slots are
// skipped by the alive-bit guard inside IntegrateParticle (their bytes stay byte-identical).
inline void IntegrateParticles(ParticlePool& pool, const FxVec3& gravity, fx dragK, fx dt) {
    const size_t n = pool.particles.size();
    for (size_t i = 0; i < n; ++i)
        IntegrateParticle(pool.particles[i], gravity, dragK, dt);
}

// ----- RecycleDead: ASCENDING-slot single-thread free-list maintenance (DETERMINISM-CRITICAL) -------------
// For each slot in ASCENDING index order, if it was just marked dead (NOT alive AND seed != 0 — a live
// particle that died this step, not an already-empty slot), push its index onto the free-list (LIFO) + clear
// its seed (mark it empty). ASCENDING order is the determinism contract: the order dead slots re-enter the
// free-list is a pure function of (capacity, command stream), so the NEXT tick's spawns land in replay-stable
// slots — NO atomic cursor, NO GPU-scheduling dependence. Single-thread host pass (between integrate
// dispatches). Pure integer.
inline void RecycleDead(ParticlePool& pool) {
    const uint32_t n = (uint32_t)pool.particles.size();
    for (uint32_t i = 0; i < n; ++i) {
        FxParticle& p = pool.particles[(size_t)i];
        if (!(p.flags & kFlagAlive) && p.seed != 0u) {   // a slot that died this step (was live, now empty)
            pool.freeList.push_back(i);                  // return the slot to the LIFO free-list
            p.seed = 0u;                                 // mark empty (so it is not recycled twice)
        }
    }
}

// ----- StepEmitIntegrate: one full PT1 tick (Emit -> IntegrateParticles -> RecycleDead -> ++tick) ---------
// The make-or-break reference the GPU host-driven driver memcmp's against:
//   (1) Emit (SINGLE-THREAD host-ordered LIFO spawn — the spawn-determinism pass).
//   (2) IntegrateParticles (per-particle INDEPENDENT multi-thread — the GPU dispatch).
//   (3) RecycleDead (ASCENDING-slot single-thread free-list maintenance — the recycle-determinism pass).
//   (4) ++tick.
// Pure integer, fixed pass order -> two-run bit-identical AND bit-exact GPU==CPU (the GPU runs (2) as the
// K-step integrate dispatch; the host does (1)+(3) between dispatches — tiny, deterministic).
inline void StepEmitIntegrate(ParticlePool& pool, const EmitterConfig& cfg, const FxVec3& gravity,
                              fx dragK, fx dt) {
    Emit(pool, cfg);                          // (1) spawn (single-thread host-ordered)
    IntegrateParticles(pool, gravity, dragK, dt);  // (2) integrate (per-particle independent)
    RecycleDead(pool);                        // (3) recycle dead slots (ascending-slot order)
    ++pool.tick;                              // (4) advance the tick
}

// ----- CountAlive: the deterministic count of live particles (a reporting/stat helper) -------------------
// Pure integer compare -> bit-exact CPU<->GPU. Used by the showcase's free-list-invariant proof.
inline uint32_t CountAlive(const ParticlePool& pool) {
    uint32_t n = 0;
    for (const FxParticle& p : pool.particles)
        if (p.flags & kFlagAlive) ++n;
    return n;
}

}  // namespace particles
}  // namespace hf::sim
