#pragma once
// Slice IK1 — Deterministic IK Control-Rig: THE FIXED-POINT ANGLE LUT (the beachhead of FLAGSHIP #32).
// engine/anim/ik.h — hf::anim::ik. A host-baked Q16.16 angle LUT (acos / sin / cos / atan2) computed in
// DOUBLE at build time and looked up with PURE INTEGER ops on the hot path: zero runtime transcendentals,
// bit-identical CPU/Vulkan/Metal. This is the engine's "zero runtime transcendental" doctrine
// (joint.h's host-cos cone-limit constants) GENERALIZED from per-scene constants to a swept table, the
// engine/sim/fluid.h::BuildKernelTable mold (host double eval -> round-to-nearest snap -> integer table ->
// integer index + fixed-point lerp). No solver yet — this is the transcendental primitive every later IK
// slice (IK2 two-bone law-of-cosines, IK3 FABRIK look-at) indexes.
//
// READ-ONLY #includes of the byte-frozen headers (this header MODIFIES none of them):
//   * sim/fpx.h    — fx / Q16.16 / kOne / kFrac / fxmul / fxdiv (the fixed-point substrate).
//   * sim/joint.h  — the cone-limit host-cos/sin CONSTANT precedent IK1 generalizes to a table.
//   * anim/skeleton.h + anim/animation.h — the rig types later IK slices solve over (carried here so the
//     IK module composes with the anim pillar; IK1 itself only needs the fx scalar from fpx.h).
//
// ===================================================================================================
// THE int32 / MSL-NATIVE SPLIT (the make-or-break design call, documented per the spec):
//   * acos / sin / cos  -> the index math + the fixed-point lerp stay PURE INT32 (NO int64), so the GPU
//     shader shaders/ik_angle.comp lowers to MSL and runs on BOTH backends (a REAL GPU==CPU memcmp, the
//     strongest determinism tier — the FPX2 int32-broadphase / fract_classify lesson). These three are
//     IK2's critical path, so they get the strongest proof.
//   * atan2 -> needs the y/x ratio via fxdiv (int64), exactly the op glslc cannot lower to MSL (the
//     engine's int64/glslc Metal lesson). So FxAtan2Lut is the CPU/int64 part: it is NOT in the
//     MSL-native sweep shader; it is proven by the CPU test (quadrant correctness) + is available for
//     IK3's look-at on the CPU / the Vulkan side. The acos/sin/cos sweep alone rides the GPU shader.
// ===================================================================================================
//
// FIDELITY CAVEAT (the fluid.h BINNED-LUT caveat shape): a host-snapped table + linear lerp is a
// DETERMINISTIC step/lerp APPROXIMATION of the analytic transcendental. The claim is DETERMINISM +
// cross-platform bit-identity + a documented within-LSB-band accuracy vs the analytic angle, NOT
// "more accurate than std::acos". kAcosBins=256 / kTrigBins=256 are golden-stable resolution choices.

#include "anim/skeleton.h"
#include "anim/animation.h"
#include "sim/fpx.h"
#include "sim/joint.h"

#include <array>
#include <cmath>     // BUILD-TIME ONLY: std::acos/std::sin/std::cos/std::atan in the host LUT builders.
#include <cstdint>

namespace hf::anim::ik {

// Pull the Q16.16 substrate from fpx.h (READ-ONLY). fx is a Q16.16 int32 scalar; kOne == 1.0; kFrac == 16.
using fx = hf::sim::fpx::fx;
inline constexpr int kFrac = hf::sim::fpx::kFrac;   // 16
inline constexpr fx  kOne  = hf::sim::fpx::kOne;    // 65536

// ---- LUT resolutions (golden-stable; NOT tuned for accuracy — the fluid.h kKernelBins doctrine) -------
// acos: 1024 bins over the cosine domain [-1, 1] (1025 table entries). 1024 (not 256) because acos has a
// sqrt-singularity slope at x=+/-1 where linear interp is worst; 1024 bins shrinks the endpoint band to a
// few-hundred-LSB while the index math stays int32-safe ((x+kOne)*1024 <= 2^17*1024 = 2^27 < INT32_MAX).
inline constexpr int kAcosBins = 1024;
inline constexpr int kTrigBins = 256;   // sin/cos: 256 bins over [0, 2pi) (257 table entries)
inline constexpr int kAtanBins = 256;   // atan: 256 bins over the ratio domain [0, 1] (257 table entries)

// ---- Host-snapped Q16.16 constants (round-to-nearest, ties away from zero — the cloth.h restLen snap) --
// kPi / kHalfPi / kTwoPi as Q16.16. 2*kOne == 1<<17 is EXACT, used by the acos power-of-two index trick.
inline constexpr fx kPi     = (fx)205887;   // round(pi      * 65536) = 205887.42 -> 205887
inline constexpr fx kHalfPi = (fx)102944;   // round(pi/2    * 65536) = 102943.71 -> 102944
inline constexpr fx kTwoPi  = (fx)411775;   // round(2*pi    * 65536) = 411774.84 -> 411775

// snap(v): round-to-nearest Q16.16 of a host double v (ties away from zero). The fluid.h::BuildKernelTable
// snap, the ONLY place double is used — at BUILD time, in the constexpr-eligible host table builders.
inline constexpr fx Snap(double v) {
    const double s = v * (double)kOne;
    return (fx)(s + (s < 0 ? -0.5 : 0.5));
}

// ===================================================================================================
// The LUT tables (host-baked ONCE). std::array<fx, bins+1>; entry i = the analytic value at the bin's
// lower domain edge, snapped to Q16.16. bins+1 entries so the linear lerp can read table[i] + table[i+1].
// Built lazily-once via a function-local static (header-only, no ODR duplication, no per-call float).
// ===================================================================================================

using AcosTable = std::array<fx, kAcosBins + 1>;
using TrigTable = std::array<fx, kTrigBins + 1>;
using AtanTable = std::array<fx, kAtanBins + 1>;

// BuildAcosLut(): acos over cosine x in [-1, 1]. Entry i is at x_i = -1 + 2*i/bins; value = acos(x_i) in
// radians, snapped to Q16.16. acos is MONOTONE NON-INCREASING in x, so the table is non-increasing in i
// BY CONSTRUCTION (entry 0 = acos(-1) = pi; entry bins = acos(1) = 0). The fluid.h BuildKernelTable mold.
inline AcosTable BuildAcosLut() {
    AcosTable t{};
    for (int i = 0; i <= kAcosBins; ++i) {
        const double x = -1.0 + 2.0 * (double)i / (double)kAcosBins;  // x_i in [-1, 1]
        const double xc = x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x);      // clamp (i==bins -> exactly 1.0)
        t[(size_t)i] = Snap(std::acos(xc));                           // radians, snapped Q16.16
    }
    return t;
}

// BuildSinLut(): sin over theta in [0, 2pi). Entry i is at theta_i = 2pi*i/bins; value = sin(theta_i),
// snapped to Q16.16 (range [-1, 1] -> [-kOne, kOne]). Periodic: entry bins == entry 0 == sin(2pi) == 0.
inline TrigTable BuildSinLut() {
    TrigTable t{};
    const double kPiD = 3.14159265358979323846;
    for (int i = 0; i <= kTrigBins; ++i) {
        const double th = 2.0 * kPiD * (double)i / (double)kTrigBins;
        t[(size_t)i] = Snap(std::sin(th));
    }
    return t;
}

// BuildCosLut(): cos over theta in [0, 2pi). Entry i at theta_i = 2pi*i/bins; value = cos(theta_i), Q16.16.
inline TrigTable BuildCosLut() {
    TrigTable t{};
    const double kPiD = 3.14159265358979323846;
    for (int i = 0; i <= kTrigBins; ++i) {
        const double th = 2.0 * kPiD * (double)i / (double)kTrigBins;
        t[(size_t)i] = Snap(std::cos(th));
    }
    return t;
}

// BuildAtanLut(): atan over the ratio r in [0, 1]. Entry i at r_i = i/bins (a Q16.16 fraction); value =
// atan(r_i) in radians [0, pi/4], snapped Q16.16. The base table for the FxAtan2Lut standard reduction.
inline AtanTable BuildAtanLut() {
    AtanTable t{};
    for (int i = 0; i <= kAtanBins; ++i) {
        const double r = (double)i / (double)kAtanBins;  // r_i in [0, 1]
        t[(size_t)i] = Snap(std::atan(r));
    }
    return t;
}

// The shared single-instance tables (built once at first use; header-only inline -> one definition).
inline const AcosTable& AcosLut() { static const AcosTable t = BuildAcosLut(); return t; }
inline const TrigTable& SinLut()  { static const TrigTable t = BuildSinLut();  return t; }
inline const TrigTable& CosLut()  { static const TrigTable t = BuildCosLut();  return t; }
inline const AtanTable& AtanLut() { static const AtanTable t = BuildAtanLut(); return t; }

// ===================================================================================================
// The PURE-INTEGER lookups. Each: clamp/reduce the input to the domain with int32 ops, compute the bin
// index + a Q16.16 sub-bin fraction with int32 ops, and return the deterministic fixed-point lerp
// table[i] + (((table[i+1]-table[i]) * frac) >> kFrac). Identical bits CPU <-> shader.
// ===================================================================================================

// FxAcosLut(x): x a Q16.16 COSINE in [-kOne, kOne] -> the angle in Q16.16 RADIANS [0, kPi].
// INDEX (pure int32): the cosine domain width is 2*kOne == 1<<17 (a power of two), so
//   scaled = (x + kOne) * kAcosBins      // x+kOne in [0, 2*kOne]; *256 <= 2^17*256 = 2^25 -> int32-safe
//   i      = scaled >> 17                // the bin (>> log2(2*kOne)); x==kOne -> 256 -> clamp to bins-1
//   frac   = (scaled & 0x1FFFF) >> 1     // the low 17 sub-bin bits rescaled to Q16.16 [0, kOne)
// LERP: pure int32 (the adjacent-bin acos diff <= ~8186, * frac < INT32_MAX).
inline fx FxAcosLut(fx x) {
    if (x <= -kOne) return AcosLut()[0];            // acos(-1) = pi (the clamped low end)
    if (x >=  kOne) return AcosLut()[kAcosBins];    // acos(1)  = 0  (the clamped high end)
    const int32_t scaled = (x + kOne) * kAcosBins;  // int32-safe (<= 2^25)
    int32_t i = scaled >> 17;                       // bin in [0, kAcosBins)
    if (i >= kAcosBins) i = kAcosBins - 1;          // guard the x==kOne boundary
    const int32_t frac = (scaled & 0x1FFFF) >> 1;   // Q16.16 fraction in [0, kOne)
    const AcosTable& T = AcosLut();
    const int32_t a = T[(size_t)i];
    const int32_t b = T[(size_t)i + 1];
    return (fx)(a + (((b - a) * frac) >> kFrac));    // deterministic int32 fixed-point lerp
}

// FxReducePhaseBin(theta, bins, &outI, &outFrac): reduce theta (Q16.16 radians) into [0, 2pi) and split
// into a bin index + a Q16.16 sub-bin fraction over `bins` bins of [0, 2pi). PURE INT32 (no int64):
//   * modulo-reduce theta into [0, kTwoPi) with int32 % (+ kTwoPi if negative).
//   * prod = theta * bins  (<= kTwoPi*256 ~= 1.05e8 -> int32-safe); i = prod / kTwoPi; rem = prod % kTwoPi.
//   * frac = floor(rem * kOne / kTwoPi), computed in TWO base-256 steps so rem*kOne never overflows int32
//     (rem<<8 <= ~1.05e8 fits; the nested base-256 floor-div reconstructs floor(rem*65536/kTwoPi) EXACTLY).
inline void FxReducePhaseBin(fx theta, int bins, int32_t* outI, int32_t* outFrac) {
    int32_t th = theta % kTwoPi;                    // int32 modulo (radians)
    if (th < 0) th += kTwoPi;                        // fold negatives into [0, kTwoPi)
    const int32_t prod = th * bins;                  // int32-safe (<= kTwoPi*256)
    int32_t i = prod / kTwoPi;                        // bin in [0, bins]
    if (i >= bins) i = bins - 1;                      // guard the th-near-kTwoPi boundary
    const int32_t rem = prod - i * kTwoPi;           // in [0, kTwoPi)
    // frac = floor(rem * 65536 / kTwoPi) via nested base-256 (avoids the rem<<16 int32 overflow):
    const int32_t hiNum = rem << 8;                  // rem<<8 <= ~1.05e8 -> int32-safe
    const int32_t hi    = hiNum / kTwoPi;            // high byte of the fraction (0..255)
    const int32_t rem2  = hiNum - hi * kTwoPi;       // remainder in [0, kTwoPi)
    const int32_t lo    = (rem2 << 8) / kTwoPi;      // low byte of the fraction (0..255); rem2<<8 int32-safe
    *outI    = i;
    *outFrac = (hi << 8) | lo;                        // Q16.16 fraction in [0, kOne)
}

// FxSinLut(theta): theta a Q16.16 angle in radians -> sin(theta) in Q16.16 [-kOne, kOne]. Pure int32.
inline fx FxSinLut(fx theta) {
    int32_t i, frac;
    FxReducePhaseBin(theta, kTrigBins, &i, &frac);
    const TrigTable& T = SinLut();
    const int32_t a = T[(size_t)i];
    const int32_t b = T[(size_t)i + 1];
    return (fx)(a + (((b - a) * frac) >> kFrac));    // adjacent diff <= ~1608, * frac int32-safe
}

// FxCosLut(theta): theta a Q16.16 angle in radians -> cos(theta) in Q16.16 [-kOne, kOne]. Pure int32.
inline fx FxCosLut(fx theta) {
    int32_t i, frac;
    FxReducePhaseBin(theta, kTrigBins, &i, &frac);
    const TrigTable& T = CosLut();
    const int32_t a = T[(size_t)i];
    const int32_t b = T[(size_t)i + 1];
    return (fx)(a + (((b - a) * frac) >> kFrac));
}

// FxAtanLut(r): r a Q16.16 ratio in [0, kOne] -> atan(r) in Q16.16 radians [0, pi/4]. Pure int32:
// r is already a Q16.16 fraction of the [0,1] domain, so scaled = r * kAtanBins (<= kOne*256 = 2^24,
// int32-safe); i = scaled >> 16; frac = scaled & 0xFFFF (already Q16.16). (Internal helper for FxAtan2Lut.)
inline fx FxAtanLut(fx r) {
    if (r <= 0)    return AtanLut()[0];               // atan(0) = 0
    if (r >= kOne) return AtanLut()[kAtanBins];       // atan(1) = pi/4
    const int32_t scaled = r * kAtanBins;             // int32-safe (<= 2^24)
    int32_t i = scaled >> 16;                          // bin in [0, kAtanBins)
    if (i >= kAtanBins) i = kAtanBins - 1;
    const int32_t frac = scaled & 0xFFFF;             // Q16.16 fraction in [0, kOne)
    const AtanTable& T = AtanLut();
    const int32_t a = T[(size_t)i];
    const int32_t b = T[(size_t)i + 1];
    return (fx)(a + (((b - a) * frac) >> kFrac));
}

// FxAtan2Lut(y, x): the full-quadrant angle of (x, y) in Q16.16 radians, in (-pi, pi]. THE CPU/int64 PART
// (NOT in the MSL-native ik_angle.comp): the standard reduction uses the y/x ratio via fpx::fxdiv, which
// is int64 (the op glslc cannot lower to MSL). Reduce |y/x| (or |x/y|) into [0,1], look it up in the
// FxAtanLut [0,pi/4] table, then place into the correct quadrant by the integer signs of y/x.
//   * x==0 && y==0 -> 0 (degenerate; a deterministic answer, no division).
//   * |y| <= |x|: a = atan(|y|/|x|);   else: a = pi/2 - atan(|x|/|y|)  (keep the ratio in [0,1] for accuracy).
//   * quadrant: x>=0 -> +/-a by sign(y); x<0 -> +/-(pi - a) by sign(y).
inline fx FxAtan2Lut(fx y, fx x) {
    if (x == 0 && y == 0) return 0;                   // degenerate -> deterministic 0
    const fx ax = x < 0 ? (fx)(-x) : x;
    const fx ay = y < 0 ? (fx)(-y) : y;
    fx a;                                              // the in-[0, pi/2] reference angle
    if (ay <= ax) {
        a = FxAtanLut(hf::sim::fpx::fxdiv(ay, ax));   // |y/x| in [0,1] -> atan in [0, pi/4]
    } else {
        a = (fx)(kHalfPi - FxAtanLut(hf::sim::fpx::fxdiv(ax, ay)));  // pi/2 - atan(|x/y|) in (pi/4, pi/2]
    }
    fx r;
    if (x >= 0) r = a;                                 // quadrants I/IV: angle in [0, pi/2]
    else        r = (fx)(kPi - a);                     // quadrants II/III: angle in (pi/2, pi]
    if (y < 0)  r = (fx)(-r);                           // mirror below the x-axis -> (-pi, 0)
    return r;
}

} // namespace hf::anim::ik
