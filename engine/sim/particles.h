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

// ===== Slice PT2 — Deterministic integer FORCE FIELDS (Issue #19, flagship #19, 2nd slice) ==============
// APPEND-ONLY: everything above (PT1) is BYTE-FROZEN. PT2 adds deterministic integer FORCE FIELDS — point
// attractor/repeller, vortex swirl, uniform wind — accumulated PER-PARTICLE in a FIXED array order and folded
// into the integrate as an extra acceleration alongside gravity. Per-particle INDEPENDENT (no grid, no
// neighbours, no atomics, NO TDR — each particle reads the SAME read-only field list). Pure Q16.16/integer
// (NO float/rand/clock). int64 in the falloff/normalize/cross math -> the new shader particles_forces.comp is
// VULKAN-SPIR-V-ONLY (glslc can't parse int64 in HLSL); Metal runs the CPU StepEmitForcesIntegrate byte-
// identical by construction (the PT1/GR3/FL4 split). The make-or-break: force==0 must equal PT1's
// IntegrateParticle EXACTLY (the no-op control), and the fields must accumulate in FIXED array order so the
// sum is associative-order-pinned -> bit-exact GPU==CPU + cross-backend.
//
// fpx.h gives FxLength / FxNormalize / fxdiv (read-only); FxDot / FxCross are NOT in fpx.h (they live in
// convex.h, which particles.h does NOT include — particles.h stays minimal, fpx.h+grain.h only), so we add
// the two as small LOCAL inline helpers here, VERBATIM the convex.h int64 form (the FxRotate-internal cross),
// so the shader copies the SAME ops. NO new fixed-point primitive — just the dot/cross composed of fxmul.

using fpx::fxdiv;          // Q16.16 divide (int64 shift then truncating divide; b==0 -> 0)
using fpx::FxLength;       // Q16.16 vector length (FxISqrt of the int64 sum-of-squares)
using fpx::FxNormalize;    // Q16.16 unit vector (len==0 -> (0,kOne,0) fallback)

// ----- FxDot / FxCross (LOCAL, VERBATIM the convex.h int64 form — fpx.h lacks them) ----------------------
// a·b = (ax*bx + ay*by + az*bz) >> kFrac, the sum kept in int64 then ONE arithmetic right shift -> Q16.16
// (the FxLength sum-of-squares discipline). Deterministic, identical CPU/HLSL. The shader copies THIS body.
inline fx FxDot(const FxVec3& a, const FxVec3& b) {
    int64_t d = (int64_t)a.x * (int64_t)b.x + (int64_t)a.y * (int64_t)b.y + (int64_t)a.z * (int64_t)b.z;
    return (fx)(d >> kFrac);
}
// a×b = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx), each fxmul an int64 product (the FxRotate cross). The
// vortex tangent is FxCross(axis, rPerp). The shader copies THIS body VERBATIM.
inline FxVec3 FxCross(const FxVec3& a, const FxVec3& b) {
    return FxVec3{
        fxmul(a.y, b.z) - fxmul(a.z, b.y),
        fxmul(a.z, b.x) - fxmul(a.x, b.z),
        fxmul(a.x, b.y) - fxmul(a.y, b.x),
    };
}

// ----- The force-field kinds (a fixed integer enum) -----------------------------------------------------
inline constexpr uint32_t kFieldPoint  = 0u;  // attractor (strength>0) / repeller (strength<0), radial falloff
inline constexpr uint32_t kFieldVortex = 1u;  // swirl about `axis` through `center`, tangential, radial falloff
inline constexpr uint32_t kFieldWind   = 2u;  // uniform constant force = axis*strength (no falloff, no radius)

// ----- ForceField: a single deterministic Q16.16 force field (the std430 GPU mirror) --------------------
// std430-packable as plain int32s: kind (uint), center.xyz, axis.xyz, strength, radius (9 x 4-byte = 36
// bytes; the GPU pack rounds to a std430-friendly stride). center/axis are Q16.16 vectors; strength is a
// SIGNED Q16.16 magnitude (point: + attract toward center / - repel away); radius is the Q16.16 influence
// cutoff (point/vortex; ignored for wind). axis is a host-snapped unit-ish direction (vortex spin axis /
// wind direction). All host-snapped integers -> the GPU consumes them verbatim, ZERO float.
struct ForceField {
    uint32_t kind = kFieldPoint;
    FxVec3   center{};     // Q16.16 field origin (point/vortex)
    FxVec3   axis{};       // Q16.16 unit-ish: vortex spin axis, OR wind direction (host-snapped unit)
    fx       strength = 0; // Q16.16 force magnitude (signed for point: + attract, - repel)
    fx       radius = 0;   // Q16.16 influence radius (point/vortex); ignored for wind
};

// ----- AccumulateForce: sum the force on a particle over the fields in FIXED ARRAY ORDER ------------------
// PURE int64 Q16.16, fields summed in ascending index order (the deterministic associative-order contract —
// the SAME order on CPU + GPU, so the integer sum is bit-identical). Per field kind:
//   kFieldPoint:  d = center - p.pos; dist = FxLength(d); if (0 < dist < radius) {
//                   dir = FxNormalize(d); falloff = fxdiv(radius - dist, radius);  // 1 at center -> 0 at edge
//                   force += FxScale(dir, fxmul(strength, falloff)); }             // strength sign = attract/repel
//   kFieldVortex: r = p.pos - center; rPerp = r - FxScale(axis, FxDot(r, axis));   // perpendicular to axis
//                 dist = FxLength(rPerp); if (0 < dist < radius) {
//                   tan = FxCross(axis, rPerp); if (FxLength(tan) > 0) {
//                     dir = FxNormalize(tan); falloff = fxdiv(radius - dist, radius);
//                     force += FxScale(dir, fxmul(strength, falloff)); } }
//   kFieldWind:   force += FxScale(axis, strength);                                 // constant, no falloff
// dead/empty particles still evaluate the same (the caller gates on ALIVE), so the result is order-pinned.
// The shader copies THIS body VERBATIM. (Uses fpx::FxLength/FxNormalize/fxdiv + the local FxDot/FxCross.)
inline FxVec3 AccumulateForce(const FxParticle& p, const ForceField* fields, uint32_t count) {
    FxVec3 force{0, 0, 0};
    for (uint32_t f = 0; f < count; ++f) {
        const ForceField& fld = fields[f];
        if (fld.kind == kFieldPoint) {
            const FxVec3 d = FxSub(fld.center, p.pos);
            const fx dist = FxLength(d);
            if (dist > 0 && dist < fld.radius) {
                const FxVec3 dir = FxNormalize(d);
                const fx falloff = fxdiv(fld.radius - dist, fld.radius);
                const fx mag = fxmul(fld.strength, falloff);
                force.x += fxmul(dir.x, mag);
                force.y += fxmul(dir.y, mag);
                force.z += fxmul(dir.z, mag);
            }
        } else if (fld.kind == kFieldVortex) {
            const FxVec3 r = FxSub(p.pos, fld.center);
            const fx along = FxDot(r, fld.axis);
            const FxVec3 rPerp = FxSub(r, FxScale(fld.axis, along));
            const fx dist = FxLength(rPerp);
            if (dist > 0 && dist < fld.radius) {
                const FxVec3 tang = FxCross(fld.axis, rPerp);
                if (FxLength(tang) > 0) {
                    const FxVec3 dir = FxNormalize(tang);
                    const fx falloff = fxdiv(fld.radius - dist, fld.radius);
                    const fx mag = fxmul(fld.strength, falloff);
                    force.x += fxmul(dir.x, mag);
                    force.y += fxmul(dir.y, mag);
                    force.z += fxmul(dir.z, mag);
                }
            }
        } else if (fld.kind == kFieldWind) {
            force.x += fxmul(fld.axis.x, fld.strength);
            force.y += fxmul(fld.axis.y, fld.strength);
            force.z += fxmul(fld.axis.z, fld.strength);
        }
    }
    return force;
}

// ----- IntegrateParticleWithForce: the PT1 integrate with `force` ADDED to gravity -----------------------
// EXACTLY IntegrateParticle, but the velocity integrate adds (gravity + force) instead of gravity:
//   vel += (gravity + force) * dt; vel -= vel*dragK; pos += vel*dt; age += dt; death. When force == (0,0,0)
//   this is BYTE-IDENTICAL to IntegrateParticle (the no-op control). The shader copies THIS body VERBATIM.
inline void IntegrateParticleWithForce(FxParticle& p, const FxVec3& force, const FxVec3& gravity,
                                       fx dragK, fx dt) {
    if (!(p.flags & kFlagAlive)) return;
    // (1) integrate velocity: vel += (gravity + force) * dt.
    p.vel.x += fxmul(gravity.x + force.x, dt);
    p.vel.y += fxmul(gravity.y + force.y, dt);
    p.vel.z += fxmul(gravity.z + force.z, dt);
    // (2) linear drag: vel -= vel * dragK.
    p.vel.x -= fxmul(p.vel.x, dragK);
    p.vel.y -= fxmul(p.vel.y, dragK);
    p.vel.z -= fxmul(p.vel.z, dragK);
    // (3) integrate position: pos += vel * dt.
    p.pos.x += fxmul(p.vel.x, dt);
    p.pos.y += fxmul(p.vel.y, dt);
    p.pos.z += fxmul(p.vel.z, dt);
    // (4) age + death (NO ground clamp — PT3).
    p.age += dt;
    if (p.age >= p.lifetime) p.flags &= ~kFlagAlive;
}

// ----- IntegrateParticlesWithForces: per-particle accumulate + force-integrate over the whole pool --------
// For each slot: force = AccumulateForce(p, fields, count); IntegrateParticleWithForce(p, force, ...). The
// reference the GPU memcmp's against. Per-particle INDEPENDENT (every particle reads the SAME read-only field
// list) -> order-independent -> bit-identical regardless of GPU scheduling. count==0 -> force is always
// (0,0,0) -> EXACTLY IntegrateParticles (the no-op control). The shader runs THIS per particle.
inline void IntegrateParticlesWithForces(ParticlePool& pool, const ForceField* fields, uint32_t count,
                                         const FxVec3& gravity, fx dragK, fx dt) {
    const size_t n = pool.particles.size();
    for (size_t i = 0; i < n; ++i) {
        const FxVec3 force = AccumulateForce(pool.particles[i], fields, count);
        IntegrateParticleWithForce(pool.particles[i], force, gravity, dragK, dt);
    }
}

// ----- StepEmitForcesIntegrate: one full PT2 tick ---------------------------------------------------------
// (1) Emit (single-thread host-ordered LIFO spawn) -> (2) IntegrateParticlesWithForces (per-particle
// accumulate + force-integrate, the GPU dispatch) -> (3) RecycleDead (ascending-slot) -> (4) ++tick. count==0
// reduces to PT1's StepEmitIntegrate EXACTLY (the forces-idle control). Pure integer, fixed pass order ->
// two-run bit-identical AND bit-exact GPU==CPU (the GPU runs (2); the host does (1)+(3) between dispatches).
inline void StepEmitForcesIntegrate(ParticlePool& pool, const EmitterConfig& cfg, const ForceField* fields,
                                    uint32_t count, const FxVec3& gravity, fx dragK, fx dt) {
    Emit(pool, cfg);                                                  // (1) spawn (single-thread host-ordered)
    IntegrateParticlesWithForces(pool, fields, count, gravity, dragK, dt);  // (2) accumulate + integrate
    RecycleDead(pool);                                               // (3) recycle dead slots (ascending)
    ++pool.tick;                                                     // (4) advance the tick
}

// ===== Slice PT3 — Deterministic particle-vs-world COLLISIONS (Issue #19, flagship #19, 3rd slice) =======
// APPEND-ONLY: everything above (PT1 + PT2) is BYTE-FROZEN. PT3 adds particle-vs-world collision — a ground
// plane + a set of static spheres — with a deterministic velocity BOUNCE (restitution). It PORTS the grain
// collision MATH (engine/sim/grain.h::CollideGrainPlane / CollideGrainSphere, read-only) onto FxParticle (a
// FIXED particle radius, not a per-particle one) AND adds the velocity reflection grains don't need: grains
// are position-based (their velocity is implied by pos - prev), but FxParticle carries an EXPLICIT vel, so a
// bounce must reflect the inward velocity component (vn = vel·nrm; if vn<0, vel -= nrm*(1+e)*vn). Pure
// Q16.16/integer, per-particle INDEPENDENT (no grid, no neighbours, NO atomics, NO TDR — each particle reads
// the SAME read-only collider list, writes only its OWN slot). The make-or-break: a particle CLEAR of every
// collider must equal PT1's IntegrateParticle EXACTLY (the no-op control), and the plane + sphere passes run
// in a FIXED order (plane, then spheres ascending) so the result is order-pinned -> bit-exact GPU==CPU +
// cross-backend.
//
// int64 (the FxLength/FxNormalize/FxDot snap + the (1+e)*vn reflect) -> the new shader particles_collide.comp
// is VULKAN-SPIR-V-ONLY (glslc can't parse int64 in HLSL); Metal runs the CPU StepEmitIntegrateCollide byte-
// identical by construction (the PT1/PT2/GR3 split). Reuses fpx::FxLength/FxNormalize/FxAdd/FxScale + the PT2
// local FxDot — NO new fixed-point primitive. (The CollideGrainSphere body shape + the explicit-vel bounce.)

// ----- ParticleSphereCollider: a static sphere collider (the GrainSphereCollider shape) ------------------
// center / radius are Q16.16. std430-packable as plain int32s (center.xyz + radius + pad) on the GPU side.
struct ParticleSphereCollider { FxVec3 center{}; fx radius = 0; };

// ----- The PT3 collision constants (host Q16.16 literals) ------------------------------------------------
inline constexpr fx kParticleRadius      = kOne / 4;   // 0.25 world units (the particle's collision radius)
inline constexpr fx kParticleRestitution = kOne / 2;   // 0.5 bounce (Q16.16); 0 = stick, kOne = elastic
inline constexpr fx kCollideEps          = kOne / 256; // containment tolerance for the proof scans

// ----- CollideParticlePlane: clamp the particle SURFACE to the ground + reflect the downward velocity -----
// The particle's SURFACE rests on the plane: pos.y >= groundY + radius (restY). If pos.y < restY, clamp
// pos.y = restY AND, if the velocity is DOWNWARD (vel.y < 0), reflect+damp it: vel.y = -fxmul(e, vel.y)
// (e = restitution; e==0 sticks, e==kOne is fully elastic). Frictionless — the tangential vel.x/vel.z are
// UNCHANGED. Pure integer, no divide, no sqrt. The shader copies THIS body VERBATIM.
inline void CollideParticlePlane(FxParticle& p, fx groundY, fx radius, fx e) {
    const fx restY = groundY + radius;
    if (p.pos.y < restY) {
        p.pos.y = restY;
        if (p.vel.y < 0) p.vel.y = -fxmul(e, p.vel.y);   // reflect+damp the downward component
    }
}

// ----- CollideParticleSphere: project the particle CENTRE out of a static sphere + reflect inward vel -----
// The surfaces touch at surf = sphereR + radius. int32 AABB reject (against surf) first; then if the centre
// distance < surf, snap the CENTRE out along the outward normal AND reflect the inward velocity component:
//   d = pos - center; dist = FxLength(d); nrm = FxNormalize(d) (d==0 -> +Y fallback);
//   pos = center + nrm*surf; vn = FxDot(vel, nrm); if (vn < 0) vel -= nrm*fxmul(kOne + e, vn).
// Returns true iff it was a contact (the CollideGrainSphere body + the explicit-vel bounce). Pure integer
// except the int64 FxLength/FxNormalize/FxDot. The shader copies THIS body VERBATIM.
inline bool CollideParticleSphere(FxParticle& p, const ParticleSphereCollider& s, fx radius, fx e) {
    const fx surf = s.radius + radius;                   // the surfaces-touch distance
    const fx dx = p.pos.x - s.center.x;
    const fx dy = p.pos.y - s.center.y;
    const fx dz = p.pos.z - s.center.z;
    const fx ax = dx < 0 ? -dx : dx;
    const fx ay = dy < 0 ? -dy : dy;
    const fx az = dz < 0 ? -dz : dz;
    if (ax > surf || ay > surf || az > surf) return false;   // outside the AABB -> no overlap
    const FxVec3 d = FxVec3{dx, dy, dz};
    const fx dist = FxLength(d);
    if (dist >= surf) return false;                          // outside the (expanded) sphere -> untouched
    const FxVec3 nrm = FxNormalize(d);                       // dist==0 -> {0,kOne,0} fallback
    p.pos = FxAdd(s.center, FxScale(nrm, surf));             // snap the centre to sphereR + radius
    const fx vn = FxDot(p.vel, nrm);                         // velocity along the outward normal
    if (vn < 0) {                                            // moving INTO the sphere -> reflect+damp it
        const fx j = fxmul(kOne + e, vn);
        p.vel.x -= fxmul(nrm.x, j);
        p.vel.y -= fxmul(nrm.y, j);
        p.vel.z -= fxmul(nrm.z, j);
    }
    return true;
}

// ----- CollideParticleWorld: per ALIVE particle, the plane then each sphere (FIXED order) ----------------
// For each ALIVE particle (index order), run CollideParticlePlane then, per sphere (ascending index, fixed
// order), CollideParticleSphere. Dead/empty slots are skipped (their bytes stay byte-identical). Returns the
// contact count. Per-particle INDEPENDENT (every particle reads the SAME read-only collider list) -> order-
// independent -> bit-identical regardless of GPU scheduling. The shader runs THIS per particle.
inline int CollideParticleWorld(ParticlePool& pool, fx groundY, fx radius, fx e,
                                const ParticleSphereCollider* spheres, uint32_t sphereCount) {
    int contacts = 0;
    const size_t n = pool.particles.size();
    for (size_t i = 0; i < n; ++i) {
        FxParticle& p = pool.particles[i];
        if (!(p.flags & kFlagAlive)) continue;
        CollideParticlePlane(p, groundY, radius, e);
        for (uint32_t s = 0; s < sphereCount; ++s)
            if (CollideParticleSphere(p, spheres[s], radius, e)) ++contacts;
    }
    return contacts;
}

// ----- StepEmitIntegrateCollide: one full PT3 tick --------------------------------------------------------
// (1) Emit (single-thread host-ordered LIFO spawn) -> (2) IntegrateParticles (PT1 gravity+drag, the GPU
// integrate dispatch) -> (3) CollideParticleWorld (per-particle independent plane+spheres, the GPU collide
// dispatch / second half of the shader) -> (4) RecycleDead (ascending-slot) -> (5) ++tick. NO force fields
// here (PT3 is collision; PT4 composes all). A particle clear of every collider is UNTOUCHED by step (3) ->
// the tick reduces to PT1's StepEmitIntegrate EXACTLY for it (the no-op control). Pure integer, fixed pass
// order -> two-run bit-identical AND bit-exact GPU==CPU (the GPU runs (2)+(3) in ONE dispatch; the host does
// (1)+(4) between dispatches).
inline void StepEmitIntegrateCollide(ParticlePool& pool, const EmitterConfig& cfg, const FxVec3& gravity,
                                     fx dragK, fx dt, fx groundY, fx radius, fx e,
                                     const ParticleSphereCollider* spheres, uint32_t sphereCount) {
    Emit(pool, cfg);                                              // (1) spawn (single-thread host-ordered)
    IntegrateParticles(pool, gravity, dragK, dt);                // (2) integrate (per-particle independent)
    CollideParticleWorld(pool, groundY, radius, e, spheres, sphereCount);  // (3) collide (per-particle indep.)
    RecycleDead(pool);                                           // (4) recycle dead slots (ascending)
    ++pool.tick;                                                 // (5) advance the tick
}

// ===== Slice PT4 — The composed StepParticles tick (Issue #19, flagship #19, 4th slice) =================
// APPEND-ONLY: everything above (PT1 + PT2 + PT3) is BYTE-FROZEN. PT4 composes the PT1 emitter + PT2 force
// fields + PT3 collision into ONE deterministic tick: emit -> forces+integrate -> collide -> recycle -> ++tick.
// A full VFX effect (a fountain with wind/vortex + ground/sphere collision) in one call. PT4 adds NO new
// physics math — it ONLY ORDERS the EXISTING PTn functions, each called VERBATIM: Emit (PT1) ->
// IntegrateParticlesWithForces (PT2, the force accumulate + integrate) -> CollideParticleWorld (PT3, plane +
// spheres + bounce) -> RecycleDead (PT1, ascending-slot free-list maintenance) -> ++tick. Per-particle
// independent (the PT2 force-integrate + the PT3 collide both read the SAME read-only field/collider lists and
// write only their OWN slot) -> race-free, NO atomics, NO grid, NO TDR. v1 is a PURE EMITTER — NO particle-
// particle interaction (justified: Niagara's default modules are emitter + global fields + world collision;
// per-particle-independent = stronger determinism). Pure Q16.16/integer (NO float/rand/clock).
//
// THE GPU SPLIT (the established PT1/PT2/PT3 convention): the per-particle PT2 force-integrate + PT3 collide is
// the GPU dispatch (shaders/particles_step.comp — the particles_forces.comp accumulate+integrate body THEN the
// particles_collide.comp plane+spheres body, in ONE pass, avoiding a double-integrate); it is int64 ->
// VULKAN-SPIR-V-ONLY (glslc can't parse int64 in HLSL), NOT in the Metal hf_gen_msl list. The host runs Emit +
// RecycleDead between dispatches (the deterministic single-thread free-list passes). This StepParticles is the
// bit-exact CPU reference the GPU memcmp's against. Because it is EXACTLY the four PTn sub-stages in order, the
// composition == applying Emit -> IntegrateParticlesWithForces -> CollideParticleWorld -> RecycleDead by hand
// (the PT4 composition proof — nothing new in the math).

// ----- StepParticles: one deterministic PT4 world tick, composing PT1-PT3 -------------------------------
//   (1) Emit(pool, cfg)                                                  — PT1 host single-thread (free-list LIFO)
//   (2) IntegrateParticlesWithForces(pool, fields, count, g, dragK, dt)  — PT2 (force accumulate + integrate)
//   (3) CollideParticleWorld(pool, groundY, radius, e, spheres, sc)      — PT3 (plane + spheres + bounce)
//   (4) RecycleDead(pool)                                                — PT1 (ascending-slot free-list maint.)
//   (5) ++pool.tick
// Every sub-stage is the EXISTING PTn function called VERBATIM (PT4 only orders them). Returns the contact
// count from step (3) (a coverage stat). count==0 reduces (2) to PT1's gravity-only integrate; sphereCount==0
// + a clear pool reduces (3) to a no-op -> a particle clear of all colliders with no fields == PT1 free-fall.
inline int StepParticles(ParticlePool& pool, const EmitterConfig& cfg, const ForceField* fields, uint32_t count,
                         const FxVec3& gravity, fx dragK, fx dt, fx groundY, fx radius, fx e,
                         const ParticleSphereCollider* spheres, uint32_t sphereCount) {
    Emit(pool, cfg);                                                     // (1) spawn (single-thread host-ordered)
    IntegrateParticlesWithForces(pool, fields, count, gravity, dragK, dt);  // (2) accumulate + force-integrate
    const int contacts = CollideParticleWorld(pool, groundY, radius, e, spheres, sphereCount);  // (3) collide
    RecycleDead(pool);                                                  // (4) recycle dead slots (ascending)
    ++pool.tick;                                                        // (5) advance the tick
    return contacts;
}

// ----- StepParticlesN: run `steps` StepParticles ticks (a convenience driver) ---------------------------
// Runs StepParticles `steps` times over the SAME pool (the per-tick contact count is discarded — the caller
// can re-run StepParticles for the final-tick stat). Pure composition; identical determinism guarantees.
inline void StepParticlesN(ParticlePool& pool, const EmitterConfig& cfg, const ForceField* fields,
                           uint32_t count, const FxVec3& gravity, fx dragK, fx dt, fx groundY, fx radius, fx e,
                           const ParticleSphereCollider* spheres, uint32_t sphereCount, int steps) {
    for (int s = 0; s < steps; ++s)
        StepParticles(pool, cfg, fields, count, gravity, dragK, dt, groundY, radius, e, spheres, sphereCount);
}

// ===== Slice PT5 — Lockstep + rollback (the NETCODE HEADLINE) (Issue #19, flagship #19, 5th slice) =======
#include <cstring>   // PT5: std::memcmp for the bit-exact ParticleStatesEqual slot-array compare (append-only,
                     // mid-file include is legal C++ — PT1-PT4 lines above stay byte-frozen).
// APPEND-ONLY: everything above (PT1 + PT2 + PT3 + PT4) is BYTE-FROZEN. PT5 is the moat headline: two peers fed
// ONLY an input-command stream re-derive the EXACT particle pool bit-for-bit, and a rollback re-sims from a
// snapshot bit-exact. PURE CPU — NO GPU dispatch, NO new shader, NO new RHI. It is a determinism PROPERTY of the
// bit-exact PT4 StepParticles (the cross-backend zero-diff golden IS the lockstep evidence). Mirrors the
// grain/fpx lockstep harness VERBATIM in shape (grain.h::GrainCommand/ApplyGrainCommand/SimGrainTick/
// SnapshotGrain/RestoreGrain/RunGrainLockstep/RunGrainRollback, themselves the fpx.h::FxCommand/SnapshotWorld/
// RestoreWorld/RunLockstep/RunRollback twins) — the SAME shape over ParticlePool + EmitterConfig.
//
// A ParticleCommand is the deterministic per-tick INPUT a netcode layer would put on the wire (NOT full state),
// applied (in ARRAY ORDER) BEFORE StepParticles on its .tick. kCmdBurst spawns `argi` extra particles at `arg`
// (a burst — a temp EmitterConfig whose origin=arg, ratePerTick=argi, reusing EmitParticle's single-thread
// free-list LIFO spawn so the slots are replay-stable); kCmdGust adds `arg` (a velocity delta) to EVERY ALIVE
// particle (a wind gust); kCmdMoveEmitter relocates cfg.origin to `arg`. A std::vector<ParticleCommand> is the
// command STREAM, processed in ARRAY ORDER for each tick (the deterministic-order contract — the same order on
// every peer/platform), so authority + replica fed the same stream re-derive the same pool exactly.

inline constexpr uint32_t kCmdBurst       = 0u; // spawn `argi` extra particles at point `arg` (a burst)
inline constexpr uint32_t kCmdGust        = 1u; // add velocity delta `arg` to ALL alive particles (a wind gust)
inline constexpr uint32_t kCmdMoveEmitter = 2u; // relocate the emitter origin to `arg` (a point)

struct ParticleCommand {
    uint32_t tick = 0;          // the tick this input applies on
    uint32_t kind = kCmdBurst;  // kCmdBurst / kCmdGust / kCmdMoveEmitter
    FxVec3   arg{};             // the Q16.16 payload (burst point / gust delta-velocity / new emitter origin)
    int32_t  argi = 0;          // an integer payload (kCmdBurst: the extra-particle count)
};

// ApplyParticleCommand(pool, cfg, cmd): apply ONE input command to the pool/cfg deterministically (FIXED order,
// pure integer). kCmdBurst spawns argi particles at arg via EmitParticle (single-thread free-list LIFO — a temp
// EmitterConfig whose origin=arg, ratePerTick=argi, the rest of cfg carried), so the burst slots are replay-
// stable; kCmdGust adds arg to every ALIVE particle's velocity (skip dead/empty slots, so the byte image stays
// pinned); kCmdMoveEmitter sets cfg.origin = arg. Unknown kind is a no-op. The input event the lockstep/rollback
// streams are made of. The grain.h::ApplyGrainCommand twin (over the particle pool + emitter config).
inline void ApplyParticleCommand(ParticlePool& pool, EmitterConfig& cfg, const ParticleCommand& cmd) {
    if (cmd.kind == kCmdBurst) {
        EmitterConfig burst = cfg;          // carry lifetime/speed/emitterId; override origin + count
        burst.origin = cmd.arg;
        burst.ratePerTick = (fx)cmd.argi;   // a small plain integer count (Emit reads the raw value < kOne)
        Emit(pool, burst);                  // single-thread host-ordered LIFO spawn (replay-stable slots)
    } else if (cmd.kind == kCmdGust) {
        for (FxParticle& p : pool.particles) {
            if (!(p.flags & kFlagAlive)) continue;   // dead/empty slots untouched (byte image pinned)
            p.vel.x += cmd.arg.x;
            p.vel.y += cmd.arg.y;
            p.vel.z += cmd.arg.z;
        }
    } else if (cmd.kind == kCmdMoveEmitter) {
        cfg.origin = cmd.arg;
    }
}

// SimParticleTick(pool, cfg, cmds, cmdCount, fields, fc, g, dragK, dt, groundY, radius, e, spheres, sc): the
// deterministic per-tick step. (1) apply ALL commands in `cmds` whose .tick == pool.tick, in ARRAY ORDER (the
// deterministic input-order contract); (2) StepParticles one full PT4 tick (Emit -> forces+integrate -> collide
// -> recycle -> ++tick). Pure integer, fixed order -> bit-identical on every peer/platform. The grain.h::
// SimGrainTick twin (StepParticles replaces StepGrainFriction). Note: the commands gate on pool.tick (BEFORE the
// StepParticles ++tick), so a command with .tick==T is applied exactly when the pool is at tick T.
inline void SimParticleTick(ParticlePool& pool, EmitterConfig& cfg, const ParticleCommand* cmds, uint32_t cmdCount,
                            const ForceField* fields, uint32_t fc, const FxVec3& g, fx dragK, fx dt,
                            fx groundY, fx radius, fx e, const ParticleSphereCollider* spheres, uint32_t sc) {
    for (uint32_t c = 0; c < cmdCount; ++c)
        if (cmds[c].tick == pool.tick) ApplyParticleCommand(pool, cfg, cmds[c]);
    StepParticles(pool, cfg, fields, fc, g, dragK, dt, groundY, radius, e, spheres, sc);
}

// THE SNAPSHOT CRUX: capture the ENTIRE pool — particles + freeList + spawnCursor + tick — PLUS the emitter cfg,
// because spawn/death IS the sim. A snapshot that omits the free-list/spawnCursor re-spawns into WRONG slots
// (or with a wrong hash) on restore -> divergence (the PT5 snapshot-completeness control proves this). The
// grain.h::SnapshotGrain twin, but a STRUCT (the pool is more than one vector — the GrainParticle array is the
// whole state there; here the free-list + cursor + cfg are sim state too).
struct ParticleSnapshot {
    std::vector<FxParticle> particles;   // the full slot array (alive + dead + empty — the whole byte image)
    std::vector<uint32_t>   freeList;    // the LIFO free-list (DETERMINISM-CRITICAL — which slots the next spawn uses)
    uint32_t                spawnCursor = 0;  // the monotone spawn counter (DETERMINISM-CRITICAL — the spawn hash stream)
    uint32_t                tick = 0;         // the current sim tick
    EmitterConfig           cfg{};            // the emitter config (kCmdMoveEmitter mutates it — part of the sim state)
};

// SnapshotParticles(pool, cfg): a DEEP copy of the entire pool + the emitter config (the rollback primitive — a
// lossless saved tick). (std::vector copy is a deep copy.) The grain.h::SnapshotGrain twin.
inline ParticleSnapshot SnapshotParticles(const ParticlePool& pool, const EmitterConfig& cfg) {
    ParticleSnapshot s;
    s.particles   = pool.particles;     // deep copy
    s.freeList    = pool.freeList;      // deep copy (DETERMINISM-CRITICAL)
    s.spawnCursor = pool.spawnCursor;   // (DETERMINISM-CRITICAL)
    s.tick        = pool.tick;
    s.cfg         = cfg;
    return s;
}

// RestoreParticles(pool, cfg, s): overwrite the pool + cfg from a saved snapshot (the rollback). Bit-exact round-
// trip with SnapshotParticles (RestoreParticles(p, c, SnapshotParticles(p0, c0)) leaves p == p0 + c == c0 byte-
// for-byte). The grain.h::RestoreGrain twin.
inline void RestoreParticles(ParticlePool& pool, EmitterConfig& cfg, const ParticleSnapshot& s) {
    pool.particles   = s.particles;
    pool.freeList    = s.freeList;
    pool.spawnCursor = s.spawnCursor;
    pool.tick        = s.tick;
    cfg              = s.cfg;
}

// ParticleStatesEqual(a, b): full bit-exact compare of two pools — the slot array (memcmp, 48-byte std430 stride)
// AND the free-list AND the spawnCursor AND the tick. The lockstep/rollback proof memcmps with THIS (capturing
// the free-list/cursor is necessary — they are sim state, not bookkeeping; the snapshot-completeness control
// proves omitting them diverges). Returns true iff every byte matches.
inline bool ParticleStatesEqual(const ParticlePool& a, const ParticlePool& b) {
    if (a.particles.size() != b.particles.size()) return false;
    if (a.particles.size() != 0 &&
        std::memcmp(a.particles.data(), b.particles.data(), a.particles.size() * sizeof(FxParticle)) != 0)
        return false;
    if (a.freeList != b.freeList) return false;
    if (a.spawnCursor != b.spawnCursor) return false;
    if (a.tick != b.tick) return false;
    return true;
}

// RunParticleLockstep(init, stream, streamCount, T, ...): THE peer entry point. Clone a peer from `init`
// (RestoreParticles, the family-parity path) + run T SimParticleTicks applying the command stream -> the final
// pool. authority = RunParticleLockstep(...); replica = RunParticleLockstep(...) from the SAME init + stream
// (inputs ONLY — no state shared) -> BIT-IDENTICAL by determinism (the lockstep proof ParticleStatesEqual's
// them). The grain.h::RunGrainLockstep twin. Returns the converged ParticleSnapshot (so the caller can render
// the AUTHORITY pool). The scene params are passed through to StepParticles verbatim.
inline ParticleSnapshot RunParticleLockstep(const ParticleSnapshot& init, const ParticleCommand* stream,
                                            uint32_t streamCount, uint32_t T,
                                            const ForceField* fields, uint32_t fc, const FxVec3& g, fx dragK,
                                            fx dt, fx groundY, fx radius, fx e,
                                            const ParticleSphereCollider* spheres, uint32_t sc) {
    ParticlePool  pool;
    EmitterConfig cfg;
    RestoreParticles(pool, cfg, init);            // clone the peer from init (inputs-only, no state shared)
    for (uint32_t t = 0; t < T; ++t)
        SimParticleTick(pool, cfg, stream, streamCount, fields, fc, g, dragK, dt, groundY, radius, e, spheres, sc);
    return SnapshotParticles(pool, cfg);
}

// RunParticleRollback(init, authStream, authCount, mispredictStream, mispredictCount, T, rollbackAt, ...): the
// rollback harness. (1) run ticks 0..rollbackAt from init applying authStream, then SAVE a snapshot AT rollbackAt
// (before that tick is simulated); (2) speculatively advance a few ticks from the snapshot with the MISPREDICTED
// stream (the wrong input) — the client prediction that diverges; (3) "receive" the authoritative input ->
// RestoreParticles to the snapshot + RE-SIMULATE rollbackAt..T with the CORRECT authStream -> the corrected pool.
// The proof asserts this == RunParticleLockstep(init, authStream, T) (rollback corrected the misprediction
// EXACTLY) AND that the mispredicted-before-rollback state DIFFERED (a real divergence was fixed). The grain.h::
// RunGrainRollback twin. Returns the corrected ParticleSnapshot.
inline ParticleSnapshot RunParticleRollback(const ParticleSnapshot& init,
                                            const ParticleCommand* authStream, uint32_t authCount,
                                            const ParticleCommand* mispredictStream, uint32_t mispredictCount,
                                            uint32_t T, uint32_t rollbackAt,
                                            const ForceField* fields, uint32_t fc, const FxVec3& g, fx dragK,
                                            fx dt, fx groundY, fx radius, fx e,
                                            const ParticleSphereCollider* spheres, uint32_t sc) {
    ParticlePool  pool;
    EmitterConfig cfg;
    RestoreParticles(pool, cfg, init);
    // (1) advance 0..rollbackAt with the authoritative stream.
    for (uint32_t t = 0; t < rollbackAt; ++t)
        SimParticleTick(pool, cfg, authStream, authCount, fields, fc, g, dragK, dt, groundY, radius, e, spheres, sc);
    // (2) SAVE the snapshot at rollbackAt (the rollback restore point).
    const ParticleSnapshot snap = SnapshotParticles(pool, cfg);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (the wrong input) — the divergent
    // client prediction. Bounded to the remaining ticks (<=3).
    uint32_t specTicks = (T > rollbackAt) ? (T - rollbackAt) : 0u;
    if (specTicks > 3u) specTicks = 3u;
    for (uint32_t s = 0; s < specTicks; ++s)
        SimParticleTick(pool, cfg, mispredictStream, mispredictCount, fields, fc, g, dragK, dt, groundY, radius, e,
                        spheres, sc);
    // (3) ROLLBACK: restore the snapshot + re-simulate rollbackAt..T with the CORRECT authStream.
    RestoreParticles(pool, cfg, snap);
    for (uint32_t t = rollbackAt; t < T; ++t)
        SimParticleTick(pool, cfg, authStream, authCount, fields, fc, g, dragK, dt, groundY, radius, e, spheres, sc);
    return SnapshotParticles(pool, cfg);
}

// ===== Slice PT6 — Lit 3D render capstone (the FLOAT money-shot, COMPLETES flagship #19) =================
// APPEND-ONLY: everything above (PT1 + PT2 + PT3 + PT4 + PT5) is BYTE-FROZEN. PT6 turns the bit-exact particle
// pool into a lit 3D INSTANCED-sphere render — a deterministic spark FOUNTAIN. These helpers are the ONLY float
// in particles.h and are STRICTLY RENDER-ONLY: they READ the pool's bit-exact integer pos (never mutate it),
// so the sim path above stays pure integer. They mirror grain.h::GrainToFloat / GrainVertToWorld /
// GrainParticleTransform / GrainToRenderInstances (grain.h:991-1021) VERBATIM over an FxParticle instead of a
// GrainParticle. The provenance: every render transform IS a bit-exact FxParticle::pos (the settled output of
// the bit-exact StepParticlesN). A particle sphere is rotation-invariant, so the transform is translate * scale
// (NO rotation — the GR6 grain / FL6 droplet case, NOT the FPX6 rigid-body orient). NO new shader, NO new RHI —
// the caller feeds the output into the EXISTING instanced lit-sphere pipeline (lit_instanced.vert + lit.frag +
// scene::InstanceTransformLayout). THE GATE IS A FLOAT visresolve-bar (UNLIKE PT1-PT5): the SIM/pool is bit-
// exact + byte-identical cross-backend, but the final raster/shade is float -> the rendered image is NOT bit-
// identical Vulkan-vs-Metal. The gate is Metal two-run BYTE-IDENTICAL + provenance + a documented cross-vendor
// mean (the GR6/FPX6 float baseline), NOT a strict-zero cross-backend pixel compare.

// ParticleToFloat(v): the single host fixed-point->float conversion, v in Q16.16 -> float world units (the
// grain::GrainToFloat twin; v / (float)kOne). The ONE place particle render touches float.
inline float ParticleToFloat(fx v) { return (float)v / (float)kOne; }

// ParticleVertToWorld(p): the float world position of a Q16.16 vector (pos.xyz / (float)kOne). The ONE host
// float crossing the PT6 render uses — render-only; the bit-exact integer pos is untouched. The grain::
// GrainVertToWorld twin.
inline math::Vec3 ParticleVertToWorld(const FxVec3& p) {
    return math::Vec3{ParticleToFloat(p.x), ParticleToFloat(p.y), ParticleToFloat(p.z)};
}

// ParticleTransform(p, radius): the render-only model matrix for ONE particle — a unit sphere TRANSLATED to the
// particle's float world position (ParticleVertToWorld(p.pos)) and SCALED by the particle radius (a float world-
// unit radius). translate(pos/kOne) * scale(radius) (no rotation — a spark sphere is rotation-invariant). Pure
// deterministic host float (no RNG, no clock). The provenance: the transform IS the bit-exact FxParticle::pos.
// The grain::GrainParticleTransform twin.
inline math::Mat4 ParticleTransform(const FxParticle& p, float radius) {
    const math::Vec3 t = ParticleVertToWorld(p.pos);
    return math::Mat4::Translate(t) * math::Mat4::Scale(math::Vec3{radius, radius, radius});
}

// ParticleToRenderInstances(pool, radius): build ONE per-instance model matrix per ALIVE particle (dead/empty
// slots SKIPPED — the alive-bit guard, so the instance set is exactly the live spark count). Output is a flat
// std::vector<math::Mat4> (the caller copies each into a scene::InstanceData the EXISTING instanced lit pipeline
// consumes). A PURE FUNCTION of the pool (two calls byte-equal — no RNG, no clock). Empty/all-dead pool -> empty
// output (the empty no-op: zero instances -> the cleared base scene). Render-only, deterministic, NO sim
// mutation. The grain::GrainToRenderInstances twin over the FxParticle pool (with the alive-bit guard).
inline std::vector<math::Mat4> ParticleToRenderInstances(const ParticlePool& pool, float radius) {
    std::vector<math::Mat4> out;
    out.reserve(pool.particles.size());
    for (const FxParticle& p : pool.particles)
        if (p.flags & kFlagAlive)   // dead/empty slots skipped — exactly one transform per ALIVE particle
            out.push_back(ParticleTransform(p, radius));
    return out;
}

}  // namespace particles
}  // namespace hf::sim
