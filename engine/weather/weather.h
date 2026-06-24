#pragma once
// Slice WE1 — Deterministic integer DRIFTING CLOUD-DENSITY field (the BEACHHEAD of FLAGSHIP #27:
// DETERMINISTIC DYNAMIC WEATHER, hf::weather). Pure CPU, header-only. NO device, NO backend symbols,
// NO new RHI, NO new shader. Namespace hf::weather.
//
// The irreducible primitive: a DETERMINISTIC INTEGER CLOUD-DENSITY field. IntCloudDensity(x,z,frame,...)
// returns a Q16.16 density in [0,kOne] from an integer fBm value-noise field, ADVECTED by an integer
// wind offset that grows with the frame counter (clouds drift +X), bit-identical CPU<->Vulkan<->Metal
// BY CONSTRUCTION — NO runtime sin/frac(sin())/float (unlike the existing FLOAT engine/render/clouds.h,
// which is the WE4 float RENDER bridge, NOT the strict-integer data source). The moat: clouds drift as a
// pure function of (seed, frame), so two netcode peers see the byte-identical cloudscape (UE5 volumetric
// clouds are float/clock-driven). This is the NEW header the WEATHER flagship's later slices build on.
//
// WHY IT IS BIT-EXACT (the cross-backend crux): every operation is a pure int32/int64 integer op (the
// integer noise basis is REUSED read-only from terrain/procterrain.h: IntValueNoise = a smoothstep-faded
// bilinear blend over a Wang-avalanche hash lattice in Q16.16). The fBm octave sum is divided by a
// COMPILE-TIME-derivable octave-amplitude bound so the normalized field is a Q16.16 in [0,kOne); the
// drift offset is (int64)kCloudDriftRate*frame (frame is the ONLY time input — NO clock, NO RNG). The
// coverage carve is a threshold-subtract + integer rescale. The whole field is a pure function of
// (x,z,frame,seed,coverage,octaves) — NO <cmath>, NO float on the bit-exact path.
//
// REUSE MAP: fx/kOne/kFrac/fxmul/fxdiv come from engine/sim/fpx.h (read-only — the Q16.16 toolbox).
// terrain::IntValueNoise is the integer value-noise basis (read-only — engine/terrain/procterrain.h).

#include <cstdint>
#include <vector>

#include "sim/fpx.h"                // fx / kOne / kFrac / fxmul / fxdiv (Q16.16 toolbox), read-only.
#include "terrain/procterrain.h"   // terrain::IntValueNoise — the integer value-noise basis, read-only.

namespace hf::weather {

// Pull the Q16.16 primitives from the fixed-point toolbox (read-only reuse).
using hf::sim::fpx::fx;
using hf::sim::fpx::kOne;
using hf::sim::fpx::kFrac;
using hf::sim::fpx::fxmul;
using hf::sim::fpx::fxdiv;

// kCloudDriftRate: the deterministic per-frame drift speed — clouds translate kCloudDriftRate world units
// in +X per frame (a small Q16.16 offset). frame is the ONLY time input; driftX = kCloudDriftRate*frame.
inline constexpr fx kCloudDriftRate = kOne / 4;   // 0.25 world units / frame (Q16.16)

// IntCloudDensity(x, z, frame, seed, coverage, octaves): a drifted fBm value-noise carved by coverage into
// a Q16.16 cloud density in [0, kOne]. Pure integer — NO <cmath>, NO float, NO clock/RNG.
//
//  * driftX = (fx)((int64_t)kCloudDriftRate * frame): a Q16.16 offset growing with frame (the wind advection
//    — pure integer). The noise is sampled at (x + driftX, z) so the cloudscape translates +X each frame.
//  * nrm: the fBm at (x+driftX, z), NORMALIZED to [0, kOne). The octave sum uses base amplitude kOne/2 with
//    halving amplitude (== the terrain IntHeight shape); the running amplitude bound (sum of the amplitudes
//    actually used) divides the sum so nrm = sum/bound is a Q16.16 in [0, kOne) (value-noise is in [0,kOne)
//    so sum/bound < kOne). octaves <= 0 -> nrm = 0 (a flat field).
//  * Coverage carve: threshold = kOne - coverage; d = nrm - threshold (clamped >= 0); then rescale
//    d = (coverage > 0) ? clamp(fxdiv(d, coverage), 0, kOne) : 0. So coverage = kOne -> density == nrm
//    (full); coverage = 0 -> threshold = kOne -> d = nrm - kOne <= 0 -> density 0 EVERYWHERE (the no-op).
inline fx IntCloudDensity(fx x, fx z, uint32_t frame, uint32_t seed, fx coverage, int octaves) {
    if (octaves <= 0) return 0;   // flat field -> nrm 0 -> density 0

    // The deterministic wind advection: a Q16.16 X offset growing with the frame counter (pure integer).
    const fx driftX = static_cast<fx>(static_cast<int64_t>(kCloudDriftRate) * static_cast<int64_t>(frame));
    const fx sx = x + driftX;     // Q16.16 wraps identically on every target (int32 add)

    // The integer fBm octave sum (the terrain IntHeight shape: base amp kOne/2, halving amp / doubling freq,
    // each octave offset in the seed so layers decorrelate). The running amplitude bound normalizes the sum
    // into [0, kOne) (value-noise in [0,kOne) => sum < bound => sum/bound < kOne). Pure integer.
    fx sum   = 0;
    fx bound = 0;
    fx amp   = kOne / 2;          // base amplitude
    fx freq  = kOne / 4;          // base frequency (1/4 lattice cell per world unit => smooth rolling)
    for (int o = 0; o < octaves; ++o) {
        sum   += fxmul(amp, hf::terrain::IntValueNoise(fxmul(sx, freq), fxmul(z, freq),
                                                       seed + static_cast<uint32_t>(o)));
        bound += amp;             // the max the octave could contribute (value-noise < kOne)
        amp  >>= 1;               // halve amplitude
        freq <<= 1;               // double frequency
    }
    // nrm = sum / bound in [0, kOne) (bound > 0 since octaves >= 1). fxdiv: (sum << kFrac) / bound.
    const fx nrm = fxdiv(sum, bound);

    // Coverage carve: subtract the threshold, clamp >= 0, rescale by coverage into [0, kOne].
    const fx threshold = kOne - coverage;
    fx d = nrm - threshold;
    if (d < 0) d = 0;
    if (coverage > 0) {
        fx density = fxdiv(d, coverage);
        if (density < 0)    density = 0;
        if (density > kOne) density = kOne;
        return density;
    }
    return 0;                     // coverage 0 -> the no-op (clear sky)
}

// GenCloudSlice(seed, n, worldSize, frame, coverage, octaves): sample IntCloudDensity over an n x n
// horizontal slab cut over [0, worldSize)^2 in Q16.16 (cell (gx,gz) -> x = gx*worldSize/n, z =
// gz*worldSize/n via an int64 intermediate => deterministic floor division, NO float; the GenHeightField
// shape). Returns the n*n field row-major (index gz*n + gx). Pure integer; NO <cmath>, NO float.
inline std::vector<fx> GenCloudSlice(uint32_t seed, int n, fx worldSize, uint32_t frame,
                                     fx coverage, int octaves) {
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
                IntCloudDensity(x, z, frame, seed, coverage, octaves);
        }
    }
    return field;
}

}  // namespace hf::weather
