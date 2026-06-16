#pragma once
// Slice CW — Auto-Exposure (histogram eye adaptation) math — pure CPU (header-only, no device, no
// backend symbols). Namespace hf::render::autoexp. Mirrors dof.h / gtao.h / froxel.h: a tiny shared-
// math header ABOVE the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions
// of "vk"/"MTL" anywhere in this slice's above-seam files are seam-discipline doc comments + the
// [[vk::binding]] HLSL decorations). The two auto-exposure compute shaders
// (autoexposure_histogram.comp.hlsl / autoexposure_reduce.comp.hlsl) copy Luminance / LumToBin /
// AverageLuminance / ExposureFromAverage VERBATIM, so tests/auto_exposure_test.cpp exercises the EXACT
// math the GPU histogram + reduce run — which is what makes the adaptationEnabled=false render
// byte-identical to the standard fixed-exposure tonemap AND bit-identical cross-backend.
//
// THE TECHNIQUE (luminance-histogram auto-exposure / "eye adaptation"; the modern Frostbite/UE4
// camera): the camera should expose so a "middle-grey" scene maps to a fixed display brightness, like
// the eye adapting to a bright sky or a dark room. Three passes operate on the HDR scene color:
//   1. HISTOGRAM (one thread per pixel): read the HDR scene color, compute its Rec.709 Luminance,
//      map it to a log2-luminance BIN via LumToBin, and atomicAdd(histogram[bin], 1) into a
//      `bins`-entry INTEGER SSBO cleared to 0. INTEGER counts -> commutative+associative addition ->
//      the final per-bin count is ORDER-INDEPENDENT -> bit-deterministic (no float atomics).
//   2. REDUCE (one thread): sum the (deterministic) histogram into the weighted AverageLuminance
//      (excluding bin 0 = black, the standard), compute the target exposure via ExposureFromAverage
//      (the key-value / middle-grey formula), and write the single `exposure` float to a 1-entry SSBO.
//      Gated by adaptationEnabled: when FALSE it writes the FIXED reference E0 (the engine's existing
//      default tonemap exposure) instead.
//   3. APPLY (the tonemap_autoexp fragment variant): read `exposure` from the SSBO and apply it to the
//      HDR color BEFORE the tonemap curve — otherwise IDENTICAL to the existing post.frag tonemap.
//
// THE ADAPTATION-OFF NO-OP PROOF (what makes this golden-safe — like CS density=0==scene,
// CR radius=0==no-AO, CT maxDist=0==no-contact): the tonemap applies `exposure` to the HDR color
// before the curve. With adaptationEnabled == false the reduce writes the FIXED E0 the default tonemap
// already uses, so tonemap_autoexp(E0) produces the EXACTLY the same output as the default fixed-
// exposure post.frag. So the showcase renders adaptationEnabled=false and asserts it is BYTE-IDENTICAL
// (SHA) to the engine's standard fixed-exposure render of the same scene (no constant bias, no exposure
// drift) — proving the histogram/reduce/apply plumbing is a pure pass-through when adaptation is off —
// then renders adaptationEnabled=true as the golden. The proof is the SAME tonemap_autoexp shader at
// E0 vs the standard render (backend-portable), NOT a comparison across different shaders.
//
// Pure, deterministic functions: no RNG, no time, fixed bins/keyValue. Single-frame INSTANT adaptation
// (no temporal exposure history) so the golden is stable + two runs are byte-identical.

#include "math/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace hf::render::autoexp {

// Rec.709 relative luminance of a linear RGB color: 0.2126 r + 0.7152 g + 0.0722 b (the standard luma
// weights; green dominates because the eye is most sensitive to it). Pure white -> 1, black -> 0, pure
// green -> 0.7152. MIRRORED VERBATIM in autoexposure_histogram.comp (so the GPU bins the EXACT same
// luminance the unit test pins).
inline float Luminance(const math::Vec3& rgb) {
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

// Map a luminance to a histogram BIN in [0, bins-1] via log2-luminance binning. The luminance range
// [2^minLogLum, 2^(minLogLum+logLumRange)] is partitioned into `bins` equal LOG2 intervals:
//   t = (log2(lum) - minLogLum) / logLumRange   in [0,1] across the range
//   bin = clamp(floor(t * bins), 0, bins-1)
// lum <= 0 (true black, or below the 2^minLogLum floor) -> bin 0 (the standard "black" bin, which the
// average EXCLUDES). Documented choice: log2 binning gives the histogram a perceptually-even spread of
// luminances (a stop of brightness == a fixed bin span), the standard for auto-exposure. MIRRORED
// VERBATIM in autoexposure_histogram.comp.
inline int LumToBin(float lum, float minLogLum, float logLumRange, int bins) {
    if (bins < 1) bins = 1;
    if (lum <= 0.0f) return 0;                          // black / non-positive -> the black bin
    float logLum = std::log2(lum);
    float t = (logLum - minLogLum) / logLumRange;       // 0 at the floor, 1 at the ceiling
    if (t < 0.0f) return 0;                             // below the floor -> the black bin
    if (t > 1.0f) t = 1.0f;                             // above the ceiling -> the top bin
    int bin = (int)std::floor(t * (float)bins);
    if (bin < 0) bin = 0;
    if (bin > bins - 1) bin = bins - 1;
    return bin;
}

// The inverse of LumToBin: the representative luminance of a bin (its CENTER). The bin spans the log2
// interval [minLogLum + (bin/bins)*logLumRange, minLogLum + ((bin+1)/bins)*logLumRange]; its center is
// at (bin+0.5)/bins. BinToLum(bin) = 2^(minLogLum + ((bin+0.5)/bins) * logLumRange). Unit-tested:
// BinToLum(LumToBin(lum)) ~= lum within one bin width, and monotone increasing in bin. MIRRORED
// VERBATIM in autoexposure_reduce.comp (the reduce weights each bin by its center luminance).
inline float BinToLum(int bin, float minLogLum, float logLumRange, int bins) {
    if (bins < 1) bins = 1;
    if (bin < 0) bin = 0;
    if (bin > bins - 1) bin = bins - 1;
    float t = ((float)bin + 0.5f) / (float)bins;        // bin center in [0,1]
    float logLum = minLogLum + t * logLumRange;
    return std::exp2(logLum);
}

// The weighted average luminance of the histogram: Sum over the NON-ZERO bins of (binCenterLum * count)
// divided by the contributing pixel count. The BLACK BIN (bin 0) is EXCLUDED from BOTH the weighted sum
// AND the divisor (the standard: true-black / below-floor pixels — sky punch-through, letterboxing,
// unlit background — would otherwise drag the average toward 0 and blow the exposure up; excluding them
// makes the metering follow the LIT content). If every contributing pixel is black (all weight in bin
// 0), there is nothing to meter -> return a sensible non-zero FLOOR (the bin-0 center luminance) so the
// downstream exposure is finite (never a divide-by-zero). MIRRORED VERBATIM in autoexposure_reduce.comp.
//
// (`totalPixels` is accepted to match the GPU reduce's signature — the contributing count is recomputed
// from the histogram so the average is a pure function of the bins; totalPixels is informational.)
inline float AverageLuminance(const uint32_t* histogram, int bins, float minLogLum, float logLumRange,
                              uint32_t totalPixels) {
    (void)totalPixels;
    if (bins < 1) bins = 1;
    double weightedSum = 0.0;   // Sum binCenterLum * count over bins 1..bins-1
    double countSum    = 0.0;   // Sum count        over bins 1..bins-1 (the contributing pixels)
    for (int b = 1; b < bins; ++b) {
        double c = (double)histogram[b];
        if (c <= 0.0) continue;
        weightedSum += (double)BinToLum(b, minLogLum, logLumRange, bins) * c;
        countSum    += c;
    }
    if (countSum <= 0.0) {
        // Everything is in the black bin -> no lit content to meter. Return the floor so the exposure
        // is finite (the ExposureFromAverage eps also guards this, but the floor keeps it meaningful).
        return BinToLum(0, minLogLum, logLumRange, bins);
    }
    return (float)(weightedSum / countSum);
}

// The target exposure from the average luminance via the standard KEY-VALUE ("middle grey") formula:
//   exposure = keyValue / max(avgLum, eps)
// keyValue ~ 0.18 is the classic 18%-grey card the camera exposes the average scene luminance toward.
// HIGHER average luminance -> LOWER exposure (a bright scene darkens, like the eye squinting); LOWER
// average -> HIGHER exposure (a dark scene brightens, the eye opening up). The eps floors the divisor so
// a near-black scene yields a large-but-finite exposure (never a divide-by-zero / Inf). Monotone
// DECREASING in avgLum. MIRRORED VERBATIM in autoexposure_reduce.comp.
inline float ExposureFromAverage(float avgLum, float keyValue) {
    const float eps = 1e-4f;
    float denom = (avgLum > eps) ? avgLum : eps;
    return keyValue / denom;
}

}  // namespace hf::render::autoexp
