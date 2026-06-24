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
#include "pcg/pcg.h"               // pcg::PcgRandRange — the deterministic seeded scatter, read-only (Slice WE2).
#include "foliage/foliage.h"       // foliage::kFoliageWind16 — the committed full-wave sine LUT, read-only (Slice WE3).

namespace hf::weather {

// Pull the Q16.16 primitives from the fixed-point toolbox (read-only reuse).
using hf::sim::fpx::fx;
using hf::sim::fpx::kOne;
using hf::sim::fpx::kFrac;
using hf::sim::fpx::fxmul;
using hf::sim::fpx::fxdiv;
using hf::sim::fpx::FxVec3;   // Slice WE2: the Q16.16 3-vector (a precipitation drop position).
using hf::sim::fpx::FxToFloat; // Slice WE4: the single Q16.16->float crossing (the render bridge), read-only.

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

// ===== Slice WE2 — Deterministic integer PRECIPITATION (rain/snow) field (2nd slice of FLAGSHIP #27) ========
// Rain/snow as a DETERMINISTIC INTEGER STREAK FIELD: each drop's position is a PURE FUNCTION of (seed, frame).
// The drop falls and WRAPS with NO accumulator state, so two netcode peers see the byte-identical rain (UE5
// particle precipitation is float/LCG/clock-driven — non-portable). Pure CPU, header-only, APPEND-ONLY (WE1
// above + fpx.h + pcg.h read-only). NO <cmath>, NO float, NO clock, NO RNG, NO accumulator state.
//
// WHY IT IS BIT-EXACT (the cross-backend crux): the XZ scatter + the start phase are deterministic seeded
// pcg::PcgRandRange draws (the SAME integer avalanche hash WE2 reuses read-only from pcg.h — a pure function of
// (seed, index)). The fall is a POSITIVE integer modulo: yRaw = y0 - fallSpeed*frame (int64, frame the ONLY time
// input), wrapped into [0, columnH) by ((yRaw % columnH) + columnH) % columnH. Every op is int32/int64 integer —
// deterministic + identical on every compiler/vendor.

// ----- PrecipField: the precipitation column descriptor ------------------------------------------------------
// `count` drops scattered over an `areaW` x `areaD` XZ patch, falling through a vertical column of height
// `columnH` at `fallSpeed` Q16.16 world units per frame. seed picks the deterministic scatter + start phases.
struct PrecipField {
    uint32_t seed      = 0;
    int      count     = 0;
    fx       areaW     = kOne;   // XZ scatter width  (X in [0, areaW))
    fx       areaD     = kOne;   // XZ scatter depth  (Z in [0, areaD))
    fx       columnH   = kOne;   // vertical column height (Y wraps in [0, columnH))
    fx       fallSpeed = kOne;   // per-frame fall (Q16.16 world units)
};

// ----- PrecipDrop: drop i's world position at `frame` (pure integer) -----------------------------------------
// x/z = a FIXED deterministic XZ scatter (pcg::PcgRandRange over (seed, i*3+{0,2})); y0 = the deterministic START
// phase (pcg::PcgRandRange over (seed, i*3+1) in [0, columnH)). The Y at `frame`: yRaw = y0 - fallSpeed*frame
// (int64 — the drop falls fallSpeed/frame), then wrap into [0, columnH) with a POSITIVE integer modulo so the
// drop that exits the bottom re-enters at the top — frame the ONLY time input, NO accumulator. Pure integer.
inline FxVec3 PrecipDrop(const PrecipField& p, uint32_t i, uint32_t frame) {
    const fx x  = hf::pcg::PcgRandRange(p.seed, i * 3u + 0u, 0, p.areaW);
    const fx z  = hf::pcg::PcgRandRange(p.seed, i * 3u + 2u, 0, p.areaD);
    const fx y0 = hf::pcg::PcgRandRange(p.seed, i * 3u + 1u, 0, p.columnH);
    const int64_t yRaw = static_cast<int64_t>(y0) - static_cast<int64_t>(p.fallSpeed) * static_cast<int64_t>(frame);
    // POSITIVE integer modulo wrap into [0, columnH) (columnH > 0 assumed for a real column; guard div-by-zero).
    const int64_t H = static_cast<int64_t>(p.columnH);
    const fx y = (H > 0) ? static_cast<fx>(((yRaw % H) + H) % H) : 0;
    return FxVec3{ x, y, z };
}

// ----- GenPrecip: the `count` drops at `frame`, in fixed index order -----------------------------------------
// PrecipDrop(p, i, frame) for i in [0, count). count <= 0 -> empty (the no-op control). Pure integer.
inline std::vector<FxVec3> GenPrecip(const PrecipField& p, uint32_t frame) {
    std::vector<FxVec3> out;
    if (p.count <= 0) return out;                 // no-op control
    out.reserve(static_cast<size_t>(p.count));
    for (uint32_t i = 0; i < static_cast<uint32_t>(p.count); ++i) out.push_back(PrecipDrop(p, i, frame));
    return out;
}

// ===== Slice WE3 — Deterministic integer TIME-OF-DAY sun + sky ramp (3rd slice of FLAGSHIP #27) ===============
// The day cycle as a DETERMINISTIC INTEGER function of the frame counter: SunSky(frame) returns a Q16.16 sun
// direction (the sun sweeps an arc across the sky as the frame advances) + a sky-color ramp (midnight -> dawn ->
// noon -> dusk RGB), bit-identical CPU<->Vulkan<->Metal BY CONSTRUCTION — NO runtime trig (the baked-sine-LUT
// discipline). The sun/sky are a PURE FUNCTION of the frame, so two netcode peers see the byte-identical golden
// hour (UE5 time-of-day is float/clock-driven). APPEND-ONLY (WE1/WE2 above + fpx.h/pcg.h/foliage.h read-only).
// NO <cmath>, NO float, NO clock, NO RNG.
//
// WHY IT IS BIT-EXACT (the cross-backend crux): the sun elevation/azimuth come from the COMMITTED full-wave sine
// LUT foliage::kFoliageWind16[256] (reused read-only — kFoliageWind16[i] == round(32767*sin(2*pi*i/256))) indexed
// by an integer day-phase, so there is NO runtime sin/cos. The sky color is a per-channel integer lerp (fxmul)
// across 4 committed Q16.16 RGB key colors. Every op is int32/int64 integer — deterministic + identical on every
// compiler/vendor. frame is the ONLY time input (NO clock, NO RNG).

// kDayFrames: the number of frames in one full day cycle (the period of the sun arc + the sky ramp).
inline constexpr uint32_t kDayFrames = 1440;

// kSkyRamp[4]: the committed sky-color ramp keys (Q16.16 RGB in [0,kOne]) at day phase 0/1/2/3 quarters —
//   [0] midnight (dark blue), [1] dawn (warm orange), [2] noon (sky blue), [3] dusk (warm orange ~= dawn).
// The exact hues are art; the determinism is the point. Clean Q16.16 literals (kOne == 1.0).
static const FxVec3 kSkyRamp[4] = {
    { kOne / 16,     kOne / 12,     kOne / 6  },   // midnight — dark blue
    { kOne * 8 / 10, kOne * 4 / 10, kOne * 3 / 10 },// dawn     — warm orange
    { kOne * 4 / 10, kOne * 6 / 10, kOne       },   // noon     — sky blue
    { kOne * 8 / 10, kOne * 4 / 10, kOne * 3 / 10 },// dusk     — warm orange (like dawn)
};

// SunSkyState: the time-of-day result at a frame — the Q16.16 sun direction (a unit-ish arc vector), the Q16.16
// sky color (RGB in [0,kOne]), and the sun elevation (the Y component of sunDir — the sine over the day, + near
// noon, - at night).
struct SunSkyState {
    FxVec3 sunDir;     // Q16.16 sun direction { cosE, sinE, 0 } — the east->west arc across the day.
    FxVec3 skyColor;   // Q16.16 RGB in [0,kOne] — the midnight->dawn->noon->dusk ramp lerped at this phase.
    fx     sunElev;    // Q16.16 sun elevation == sunDir.y (the sine over the day: + near noon, - at night).
};

// SunSky(frame): the deterministic integer time-of-day at `frame`. Pure integer — NO <cmath>, NO float, NO
// clock/RNG. phase = frame % kDayFrames maps the frame into one day; tabIdx maps the phase to a 0..255 LUT index.
//   * Sun direction from the LUT (a quarter-cycle phase shift so the ELEVATION troughs at midnight (phase 0)
//     and PEAKS at noon (phase 1/2) — the day's natural arc): elevIdx = (tabIdx - 64) & 255, so sinE =
//     kFoliageWind16[elevIdx] is -32767 at midnight, 0 at dawn, +32767 at noon, 0 at dusk (the elevation sine
//     over the day). cosE = kFoliageWind16[tabIdx] (== the quarter-shifted cosine, the east->west azimuth
//     sweep). sunDir = { cosE, sinE, 0 } (a Q16.16 direction arcing across the sky); sunElev = sinE. NOTE the
//     LUT samples are int16 in [-32767, 32767]; we keep them as Q16.16 fx directly (a ~0.5-scale direction —
//     direction, not a unit vector — which is fine: callers normalize / only the arc + sign matter).
//   * Sky color by lerping kSkyRamp across the day: seg = phase*4/kDayFrames (0..3), t = the Q16.16 fractional
//     position within the segment, skyColor = per-channel lerp(kSkyRamp[seg], kSkyRamp[(seg+1)&3], t) via fxmul
//     (wrapping back to midnight after dusk). All Q16.16.
inline SunSkyState SunSky(uint32_t frame) {
    const uint32_t phase  = frame % kDayFrames;
    const uint32_t tabIdx = static_cast<uint32_t>((static_cast<uint64_t>(phase) * 256u) / kDayFrames) & 255u;

    // Sun direction from the committed full-wave sine LUT (read-only) — NO runtime trig. The elevation is the
    // sine phase-shifted a quarter cycle so it troughs at midnight (phase 0) and peaks at noon (phase 1/2).
    const uint32_t elevIdx = (tabIdx + 192u) & 255u;                                  // (tabIdx - 64) & 255
    const fx sinE = static_cast<fx>(hf::foliage::kFoliageWind16[elevIdx]);             // elevation sine
    const fx cosE = static_cast<fx>(hf::foliage::kFoliageWind16[tabIdx]);              // azimuth cosine (1/4 shift)

    SunSkyState s;
    s.sunDir  = FxVec3{ cosE, sinE, 0 };
    s.sunElev = sinE;

    // Sky color: lerp the kSkyRamp keys across the day. seg in [0,3], t = the Q16.16 fraction within the segment.
    const uint32_t segLen = kDayFrames / 4u;                       // frames per quarter-day segment
    const uint32_t seg    = (phase * 4u) / kDayFrames;             // which segment 0..3 (== phase / segLen)
    const uint32_t inSeg  = phase - seg * segLen;                  // frames into the segment [0, segLen)
    // t = inSeg / segLen as a Q16.16 fraction (pure integer floor division via int64): t in [0, kOne).
    const fx t = static_cast<fx>((static_cast<int64_t>(inSeg) * static_cast<int64_t>(kOne)) /
                                 static_cast<int64_t>(segLen));
    const FxVec3& a = kSkyRamp[seg & 3u];
    const FxVec3& b = kSkyRamp[(seg + 1u) & 3u];                   // wraps dusk -> midnight
    // Per-channel lerp: a + (b - a)*t (Q16.16 via fxmul). (b-a) may be negative -> fxmul handles signed Q16.16.
    s.skyColor = FxVec3{
        a.x + fxmul(b.x - a.x, t),
        a.y + fxmul(b.y - a.y, t),
        a.z + fxmul(b.z - a.z, t),
    };
    return s;
}

// ===== Slice WE4 — the FLOAT RENDER BRIDGE (4th slice of FLAGSHIP #27; the arc's FIRST float crossing) =========
// The render bridge crosses the bit-exact INTEGER weather data (WE1 drift + WE3 sun) into the FLOAT clouds raymarch
// (engine/render/clouds.h / clouds.frag) ONCE, via the single host FxToFloat conversion. These tiny helpers are the
// ONLY float code the flagship adds: both backends compute the SAME integer (kCloudDriftRate*frame, SunSky(frame))
// then the SAME FxToFloat, so the cloud DRIFT + the cloud LIGHTING are deterministic cross-platform (the FO5/PT4
// single-float-crossing precedent). WE1-WE3 stay strict-integer; this is APPEND-ONLY and READ-ONLY over them.

// CloudDriftTime(frame): the float advection time the existing clouds shader consumes (cprm.time), derived from the
// WE1 INTEGER drift offset (kCloudDriftRate*frame, the SAME int64 multiply IntCloudDensity uses for driftX) — so the
// cloudscape advects by the bit-exact integer drift at `frame`, NOT a clock. One FxToFloat per backend => identical.
inline float CloudDriftTime(uint32_t frame) {
    return hf::sim::fpx::FxToFloat(static_cast<fx>(static_cast<int64_t>(kCloudDriftRate) *
                                                   static_cast<int64_t>(frame)));
}

// FxToFloat3(v): convert a Q16.16 FxVec3 (e.g. SunSky(frame).sunDir / .skyColor) into a float xyz for the render
// path (the sun light direction + the sky-color tint feeding the existing CloudParams). Three FxToFloat crossings.
struct Float3 { float x, y, z; };
inline Float3 FxToFloat3(const FxVec3& v) {
    return Float3{ hf::sim::fpx::FxToFloat(v.x), hf::sim::fpx::FxToFloat(v.y), hf::sim::fpx::FxToFloat(v.z) };
}

}  // namespace hf::weather
