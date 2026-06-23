#pragma once
// Slice PCG1 — Seeded hash-PRNG primitive (the BEACHHEAD of FLAGSHIP #22: DETERMINISTIC PCG). The irreducible
// primitive every later PCG slice (scatter / mask / transform / placement) consumes: a DETERMINISTIC SEEDED
// INTEGER HASH-PRNG in Q16.16, pure int32 (MSL-native — NO int64 fxmul, NO trig, NO float, NO clock, NO
// <cmath>). Header-only, namespace hf::pcg. NO device, NO backend symbols, NO new RHI, NO shader.
//
// The determinism story is IDENTICAL to the particle emitter's: the hash is the SAME integer avalanche shape
// as engine/sim/particles.h::ParticleHash (a fixed uint32 wrapping xorshift/multiply mixing — defined +
// identical on every vendor/compiler) and the no-trig direction comes from the SAME engine/sim/particles.h::
// EmitDir host-Q16.16 direction table. We do NOT improvise a new hash; reusing the proven one keeps the
// cross-platform bit-exactness story unchanged. fpx.h (fx / kOne / kFrac / fxmul / FxVec3) is reused read-only.
//
// THE CROSS-BACKEND CRUX: every value PcgHash/PcgRand01/PcgRandRange/PcgUnitDir returns is a pure function of
// (seed, index[, salt]) computed with ONLY uint32 wrapping arithmetic + arithmetic shifts/masks + (for ranges)
// the int64-intermediate fxmul — deterministic + identical on every compiler/vendor. PcgRand01 takes the TOP
// 16 bits of the 32-bit hash as the Q16.16 fraction (a pure shift — NO division, NO float), so it lands in
// [0, kOne) strictly (top16 of a uint32 is in [0, 65535] = [0, kOne)). This is what makes a CPU-rendered
// PCG point-field byte-identical CPU<->Vulkan<->Metal BY CONSTRUCTION (a strict zero-differing-pixel golden).

#include <cstdint>

#include "sim/fpx.h"            // Q16.16 toolbox (read-only): fx / kOne / kFrac / fxmul / FxVec3
#include "sim/particles.h"     // ParticleHash (the avalanche shape) + EmitDir (the no-trig direction table), read-only

namespace hf::pcg {

// Pull the Q16.16 primitives from fpx (the SAME ones particles.h re-exports) so callers can `using` them.
using hf::sim::fpx::fx;
using hf::sim::fpx::kOne;
using hf::sim::fpx::kFrac;
using hf::sim::fpx::fxmul;
using hf::sim::fpx::FxVec3;

// ----- PcgHash: the deterministic seeded integer avalanche over (seed, index) ------------------------------
// The SAME integer hash SHAPE as engine/sim/particles.h::ParticleHash (verbatim ops) — a fixed uint32 wrapping
// xorshift/multiply mixing. Pure uint32 arithmetic (defined + identical on every vendor/compiler) — NO RNG, NO
// clock, NO float. `seed` plays the emitterId role, `index` the spawn-index role: distinct (seed,index) pairs
// hash to distinct, replay-stable streams.
inline uint32_t PcgHash(uint32_t seed, uint32_t index) {
    uint32_t h = seed * 2654435761u;               // Knuth multiplicative
    h ^= (index + 0x9E3779B9u + (h << 6) + (h >> 2));
    h += index * 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}

// ----- PcgRand01: a Q16.16 value in [0, kOne) from the TOP 16 bits of the hash ------------------------------
// A pure shift (NO division, NO float): the top 16 bits of the 32-bit hash become the Q16.16 fraction. The
// top16 of a uint32 is in [0, 65535] == [0, kOne), so the result is STRICTLY < kOne (a valid [0,1) sample).
inline fx PcgRand01(uint32_t seed, uint32_t index) {
    return (fx)(PcgHash(seed, index) >> 16);       // in [0, kOne)
}

// ----- PcgRandRange: lo + PcgRand01 * (hi - lo) in Q16.16 ---------------------------------------------------
// lo + fxmul(PcgRand01(...), hi - lo). Since PcgRand01 in [0, kOne) and fxmul is the int64-intermediate
// ((int64)a*b >> kFrac), the result lies in [lo, hi) (so within [lo, hi]) for hi >= lo.
inline fx PcgRandRange(uint32_t seed, uint32_t index, fx lo, fx hi) {
    return lo + fxmul(PcgRand01(seed, index), hi - lo);
}

// ----- PcgUnitDir: a deterministic ~unit direction by indexing the EmitDir table with the hash --------------
// Reuse engine/sim/particles.h::EmitDir VERBATIM (the fixed 13-entry host-Q16.16 direction table, NO trig, NO
// sqrt). The hash selects the table entry (EmitDir does the `% kEmitDirCount`), so the direction is a pure
// function of (seed, index) — deterministic + cross-vendor identical.
inline FxVec3 PcgUnitDir(uint32_t seed, uint32_t index) {
    return hf::sim::particles::EmitDir(PcgHash(seed, index));
}

// ----- PcgStream: a {seed, salt} stream so distinct PCG layers diverge --------------------------------------
// The emitterId-salt pattern from particles: a stream carries a `seed` AND a `salt`; the overloads fold `salt`
// into the hash INPUT so two streams with the same seed but different salt produce DIFFERENT sequences for the
// same index (the future scatter/mask/transform stages each get their own salt -> independent streams). The
// fold is `seed ^ salt` (then PcgHash mixes it thoroughly) — pure uint32, deterministic.
struct PcgStream {
    uint32_t seed = 0;
    uint32_t salt = 0;
};

inline uint32_t PcgHash(const PcgStream& s, uint32_t index) {
    return PcgHash(s.seed ^ s.salt, index);
}
inline fx PcgRand01(const PcgStream& s, uint32_t index) {
    return (fx)(PcgHash(s, index) >> 16);          // in [0, kOne)
}
inline fx PcgRandRange(const PcgStream& s, uint32_t index, fx lo, fx hi) {
    return lo + fxmul(PcgRand01(s, index), hi - lo);
}
inline FxVec3 PcgUnitDir(const PcgStream& s, uint32_t index) {
    return hf::sim::particles::EmitDir(PcgHash(s, index));
}

}  // namespace hf::pcg
