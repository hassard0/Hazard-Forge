#pragma once
// Slice PT1 — Integer fBm heightfield generation (the BEACHHEAD of FLAGSHIP #26: DETERMINISTIC
// PROCEDURAL TERRAIN, hf::terrain). Pure CPU, header-only. NO device, NO backend symbols, NO new RHI,
// NO new shader. Namespace hf::terrain.
//
// The irreducible primitive: a DETERMINISTIC INTEGER HEIGHTFIELD. IntHeight(x,z) returns a Q16.16
// height from a fractal sum (fBm) of integer value-noise octaves, bit-identical CPU<->Vulkan<->Metal
// BY CONSTRUCTION — NO runtime sin/sqrt/floor: the strict-integer twin of the existing FLOAT
// terrain::Height (engine/terrain/heightmap.cpp:26-61, which is FROZEN by the `terrain`/`terrain_stream`
// goldens + terrain_test.cpp and is NOT touched here). This is the additive sibling of the float
// heightmap.h — the strict-integer heightfield the erosion (PT2/PT3) and render (PT4-PT6) slices build on.
//
// WHY IT IS BIT-EXACT (the cross-backend crux): every operation is a pure uint32/int32/int64 integer
// op that wraps + shifts identically on every compiler/vendor. IntHashLattice is a Wang-style avalanche
// on a combined 32-bit key (== heightmap.cpp's HashLattice integer mix, mapped to a Q16.16 fraction).
// IntValueNoise is a smoothstep-faded bilinear blend over that lattice in Q16.16 (the t^2(3-2t) fade
// done in fixed point). IntHeight sums octaves with halving amplitude / doubling frequency. The whole
// field is a pure function of (x,z,octaves,seed) — NO <cmath>, NO float on the bit-exact path.
//
// REUSE MAP: fx/kOne/kFrac/fxmul come from engine/sim/fpx.h (read-only — the Q16.16 toolbox). The
// integer hash shape mirrors engine/terrain/heightmap.cpp:26-32 HashLattice; the smoothstep-bilinear
// shape mirrors heightmap.cpp:36-52 ValueNoise; the fBm octave loop mirrors the standard fractal sum.

#include <cstdint>
#include <vector>

#include "sim/fpx.h"  // fx / kOne / kFrac / fxmul (Q16.16 toolbox), read-only.

namespace hf::terrain {

// Pull the Q16.16 primitives from the fixed-point toolbox (read-only reuse).
using hf::sim::fpx::fx;
using hf::sim::fpx::kOne;
using hf::sim::fpx::kFrac;
using hf::sim::fpx::fxmul;

// IntHashLattice(ix, iz, seed): a deterministic Q16.16 corner value in [0, kOne). The SAME integer hash
// shape as heightmap.cpp's float HashLattice (a Wang-style avalanche on a combined 32-bit key) with the
// seed folded in before mixing; the top 16 bits of the avalanched hash become the Q16.16 fraction (so
// the result is strictly < kOne == 1.0). Pure uint32 arithmetic — wraps mod 2^32 identically on every
// target, so the lattice is bit-identical Windows/Vulkan <-> Apple/Metal. NO float.
inline fx IntHashLattice(int32_t ix, int32_t iz, uint32_t seed) {
    uint32_t h = static_cast<uint32_t>(ix) * 374761393u + static_cast<uint32_t>(iz) * 668265263u;
    h += seed * 2246822519u;                 // fold the seed in (a distinct prime) before the avalanche
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<fx>(h >> 16);         // top 16 bits => Q16.16 fraction in [0, kOne)
}

// IntValueNoise(x, z, seed): smoothstep-faded bilinear value noise over the integer lattice, x/z in
// Q16.16. ix = (int32_t)(x >> kFrac) is an ARITHMETIC right shift == floor (C++20 makes >> on signed
// arithmetic), tx = x - (ix << kFrac) the fractional part in [0, kOne). The smoothstep fade
// sx = t^2(3-2t) is done in Q16.16; the 4 lattice corners are blended bilinearly. Pure integer — the
// strict-integer twin of heightmap.cpp ValueNoise. Returns a value in [0, kOne).
inline fx IntValueNoise(fx x, fx z, uint32_t seed) {
    const int32_t ix = static_cast<int32_t>(x >> kFrac);   // floor(x) (arithmetic shift, C++20)
    const int32_t iz = static_cast<int32_t>(z >> kFrac);   // floor(z)
    const fx tx = x - (static_cast<fx>(ix) << kFrac);      // fractional part in [0, kOne)
    const fx tz = z - (static_cast<fx>(iz) << kFrac);
    // smoothstep s = t*t*(3 - 2t) in Q16.16 (the C1-continuous fade => no creased seams).
    const fx sx = fxmul(fxmul(tx, tx), 3 * kOne - 2 * tx);
    const fx sz = fxmul(fxmul(tz, tz), 3 * kOne - 2 * tz);
    const fx c00 = IntHashLattice(ix,     iz,     seed);
    const fx c10 = IntHashLattice(ix + 1, iz,     seed);
    const fx c01 = IntHashLattice(ix,     iz + 1, seed);
    const fx c11 = IntHashLattice(ix + 1, iz + 1, seed);
    const fx a = c00 + fxmul(c10 - c00, sx);
    const fx b = c01 + fxmul(c11 - c01, sx);
    return a + fxmul(b - a, sz);
}

// IntHeight(x, z, octaves, seed): fractional Brownian motion (fBm) — a fractal sum of value-noise
// octaves with halving amplitude / doubling frequency, x/z in Q16.16. Each octave is offset in the
// noise hash (seed + o) so the layers are decorrelated. octaves <= 0 => 0 (a flat no-op). Pure integer;
// NO <cmath>, NO float. The base amplitude (kOne/2) keeps the summed height well inside the +-32768
// Q16.16 world bound (sum of halving amps < kOne, value-noise < kOne => |h| < kOne).
inline fx IntHeight(fx x, fx z, int octaves, uint32_t seed) {
    fx h = 0;
    fx amp = kOne / 2;            // base amplitude (sum over octaves converges below kOne)
    fx freq = kOne / 4;           // base frequency (1/4 lattice cell per world unit => smooth rolling)
    for (int o = 0; o < octaves; ++o) {
        h += fxmul(amp, IntValueNoise(fxmul(x, freq), fxmul(z, freq), seed + static_cast<uint32_t>(o)));
        amp >>= 1;                // halve amplitude
        freq <<= 1;               // double frequency
    }
    return h;
}

// GenHeightField(seed, n, worldSize, octaves): sample IntHeight at an n x n grid over [0, worldSize)^2
// in Q16.16 (cell (gx,gz) -> x = gx*worldSize/n, z = gz*worldSize/n via an int64 intermediate =>
// deterministic floor division, NO float). Returns the n*n field row-major (index gz*n + gx). Pure
// integer; NO <cmath>, NO float. octaves <= 0 => every cell 0 (the flat no-op).
inline std::vector<fx> GenHeightField(uint32_t seed, int n, fx worldSize, int octaves) {
    std::vector<fx> field;
    if (n <= 0) return field;
    field.resize(static_cast<size_t>(n) * static_cast<size_t>(n));
    for (int gz = 0; gz < n; ++gz) {
        const fx z = static_cast<fx>((static_cast<int64_t>(gz) * static_cast<int64_t>(worldSize)) /
                                     static_cast<int64_t>(n));
        for (int gx = 0; gx < n; ++gx) {
            const fx x = static_cast<fx>((static_cast<int64_t>(gx) * static_cast<int64_t>(worldSize)) /
                                         static_cast<int64_t>(n));
            field[static_cast<size_t>(gz) * static_cast<size_t>(n) + static_cast<size_t>(gx)] =
                IntHeight(x, z, octaves, seed);
        }
    }
    return field;
}

}  // namespace hf::terrain
