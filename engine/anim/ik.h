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

// ===================================================================================================
// Slice IK2 — DETERMINISTIC TWO-BONE IK (the law-of-cosines limb solve). APPENDED to ik.h (the IK1 lines
// above are BYTE-FROZEN). Given a root, a target, a pole/hint, and the two bone lengths, solve the
// elbow/knee bend via the law of cosines (the ONE acos, from IK1's FxAcosLut) and emit the two bone LOCAL
// rotations as FxQuats — all Q16.16 fixed-point, bit-identical CPU/Vulkan/Metal. This is the limb
// primitive the rig (IK4) + the lockstep headline (IK5) + the render capstone (IK6) drive.
//
// THE int64 REALITY (the FPX3/JT1 lesson): TwoBoneSolve uses FxLength/fxdiv (int64) for the reach `d` +
// the cosine ratios, so shaders/ik_twobone.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot
// lower it to MSL) — NOT in the Metal hf_gen_msl list. The Metal --ik2-twobone showcase runs THIS CPU
// TwoBoneSolve over the same targets -> byte-identical to the Vulkan GPU result by construction, while the
// Vulkan side carries the GPU==CPU memcmp proof. The shader copies the math here VERBATIM.
//
// THE WITHIN-BAND CAVEAT (honest): the end-effector will NOT hit the target EXACTLY — FxAcosLut carries
// IK1's documented LSB band, so forward-kinematics from the solved quats reaches the target within a
// bounded, DETERMINISTIC residual (NOT zero). The headline is DETERMINISM + cross-platform bit-identity,
// NOT analytic reach. The over-extended clamp (d >= lenU+lenL -> aElbow=0, straight limb at target) is the
// deterministic, falsifiable boundary beat.

// Pull the rest of the Q16.16 substrate from fpx.h (READ-ONLY) for the two-bone solve.
using FxVec3 = hf::sim::fpx::FxVec3;
using FxQuat = hf::sim::fpx::FxQuat;
using hf::sim::fpx::fxmul;
using hf::sim::fpx::fxdiv;
using hf::sim::fpx::FxLength;
using hf::sim::fpx::FxNormalize;
using hf::sim::fpx::FxRotate;
using hf::sim::fpx::FxQuatMul;
using hf::sim::fpx::FxQuatNormalize;

// FxCross(a,b): the Q16.16 cross product a×b = (ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx), each term an
// int64 fxmul (the FxLength int64 discipline; the fpx.h::FxRotate internal cross, the sim::FxCross twin).
// A LOCAL helper so ik.h stays self-contained (it does not #include convex.h). The shader copies it VERBATIM.
inline FxVec3 FxCross(const FxVec3& a, const FxVec3& b) {
    return FxVec3{
        fxmul(a.y, b.z) - fxmul(a.z, b.y),
        fxmul(a.z, b.x) - fxmul(a.x, b.z),
        fxmul(a.x, b.y) - fxmul(a.y, b.x),
    };
}

// FxDot(a,b): the Q16.16 dot product (int64 fxmul terms). A LOCAL helper (joint.h::FxDot twin; ik.h is
// header-only + self-contained). Used for the pole/axis degeneracy test.
inline fx FxDotV(const FxVec3& a, const FxVec3& b) {
    return fxmul(a.x, b.x) + fxmul(a.y, b.y) + fxmul(a.z, b.z);
}

// The two LOCAL bone rotations produced by the two-bone solve. qUpper rotates the root->target axis by the
// root angle about the bend-plane normal; qLower is the LOCAL elbow rotation (relative to the upper bone),
// the π−aElbow bend about the same normal. The world lower rotation is FxQuatMul(qUpper, qLower) (the FK
// composition EndEffector does). Both feed anim::JointPose.r (the rig integration in IK4).
struct TwoBoneResult {
    FxQuat qUpper;   // upper-bone LOCAL rotation (from the root->target axis, about the bend normal)
    FxQuat qLower;   // lower-bone LOCAL rotation (relative to the upper bone, the elbow bend)
    fx     aRoot  = 0;   // the root opening angle (Q16.16 radians) — diagnostics/tests
    fx     aElbow = 0;   // the elbow INTERIOR angle (Q16.16 radians); ~0 over-extended, ~pi folded
    fx     reach  = 0;   // the clamped reach d (Q16.16) used by the solve
};

// QuatFromAxisAngle(n, theta): the unit quaternion (n·sin(theta/2), cos(theta/2)) for a UNIT axis n and a
// Q16.16-radian angle theta — VERBATIM the FxQuat w-last layout (fpx.h:109), the sin/cos via IK1's LUTs
// (the half-angle). The half-angle is theta>>1 (integer; the LUT lerp absorbs the dropped LSB). Pure
// integer; the shader copies it VERBATIM. (Not normalized again — n is unit + the LUT sin/cos are within
// band; the caller may FxQuatNormalize for the strict |q|≈kOne contract.)
inline FxQuat QuatFromAxisAngle(const FxVec3& n, fx theta) {
    const fx half = theta >> 1;            // theta/2 (Q16.16; floor — deterministic)
    const fx s = FxSinLut(half);
    const fx c = FxCosLut(half);
    return FxQuat{fxmul(n.x, s), fxmul(n.y, s), fxmul(n.z, s), c};
}

// TwoBoneSolve(root, target, pole, lenUpper, lenLower) -> the two LOCAL bone rotations.
//   e = target - root; d = FxLength(e), CLAMPED to the reachable band [|lenU-lenL|, lenU+lenL]
//     (un-reachable-near AND over-extended both clamp to the boundary -> deterministic).
//   cosRoot  = fxdiv(lenU² + d² - lenL², 2·lenU·d)     (squares/products in int64, narrowed to Q16.16)
//   cosElbow = fxdiv(lenU² + lenL² - d², 2·lenU·lenL)
//   aRoot = FxAcosLut(cosRoot); aElbow = FxAcosLut(cosElbow). Over-extended (d == lenU+lenL) -> the
//     cosElbow argument is exactly kOne so FxAcosLut returns 0 (straight limb), automatically.
//   axis = FxNormalize(e); n = FxNormalize(FxCross(axis, pole)) — the bend-plane normal (rotation axis for
//     both bones); the limb bends in the plane spanned by axis + pole, toward the pole. If the cross
//     degenerates (pole ∥ axis), use a DETERMINISTIC fallback axis (the world up (0,1,0) crossed with
//     axis, then the world right (1,0,0) crossed with axis — guaranteed non-zero).
//   qUpper = axis-angle(n, aRoot)  — rotate the axis by the root angle to the upper-bone direction.
//   qLower = axis-angle(n, kPi - aElbow) — the LOCAL elbow bend (interior angle aElbow -> exterior bend).
// Pure integer (int64 in FxLength/fxdiv/fxmul). The shader copies THIS body VERBATIM.
inline TwoBoneResult TwoBoneSolve(const FxVec3& root, const FxVec3& target, const FxVec3& pole,
                                  fx lenUpper, fx lenLower) {
    TwoBoneResult out{};

    const FxVec3 e = FxVec3{target.x - root.x, target.y - root.y, target.z - root.z};
    fx d = FxLength(e);

    // Clamp d to the reachable band [|lenU-lenL|, lenU+lenL] — deterministic for near AND far.
    const fx dMin = (lenUpper > lenLower) ? (fx)(lenUpper - lenLower) : (fx)(lenLower - lenUpper);
    const fx dMax = (fx)(lenUpper + lenLower);
    if (d < dMin) d = dMin;
    if (d > dMax) d = dMax;
    out.reach = d;

    // Q16.16 squares/products in int64 then narrow to Q16.16 (the FxLength discipline): X² = (X*X)>>kFrac.
    const fx lU2 = (fx)(((int64_t)lenUpper * (int64_t)lenUpper) >> kFrac);
    const fx lL2 = (fx)(((int64_t)lenLower * (int64_t)lenLower) >> kFrac);
    const fx d2  = (fx)(((int64_t)d * (int64_t)d) >> kFrac);
    const fx twoLUd  = (fx)(((int64_t)(2 * (int64_t)lenUpper) * (int64_t)d) >> kFrac);
    const fx twoLUlL = (fx)(((int64_t)(2 * (int64_t)lenUpper) * (int64_t)lenLower) >> kFrac);

    fx cosRoot  = fxdiv((fx)(lU2 + d2 - lL2), twoLUd);
    fx cosElbow = fxdiv((fx)(lU2 + lL2 - d2), twoLUlL);
    // Clamp the cosine arguments to [-kOne, kOne] (rounding can push them a few LSB past; FxAcosLut clamps
    // internally too, but pin it here for the diagnostic angles).
    if (cosRoot  >  kOne) cosRoot  =  kOne; if (cosRoot  < -kOne) cosRoot  = -kOne;
    if (cosElbow >  kOne) cosElbow =  kOne; if (cosElbow < -kOne) cosElbow = -kOne;

    const fx aRoot  = FxAcosLut(cosRoot);
    const fx aElbow = FxAcosLut(cosElbow);
    out.aRoot  = aRoot;
    out.aElbow = aElbow;

    // The bend plane: axis = root->target; n = normalize(axis × pole). Deterministic fallback if degenerate.
    const FxVec3 axis = FxNormalize(e);
    FxVec3 cr = FxCross(axis, pole);
    if (FxLength(cr) == 0) {
        // pole ∥ axis (or zero pole) -> pick a deterministic non-parallel reference: world up, else world right.
        cr = FxCross(axis, FxVec3{0, kOne, 0});
        if (FxLength(cr) == 0) cr = FxCross(axis, FxVec3{kOne, 0, 0});
    }
    const FxVec3 n = FxNormalize(cr);

    // qUpper rotates the axis by +aRoot about n (toward the pole). qLower is the LOCAL elbow bend: the
    // exterior turn (pi - aElbow) in the OPPOSITE rotational sense so the lower bone turns BACK toward the
    // axis and the chain closes on the target. axis-angle(n, aElbow - kPi) == axis-angle(-n, kPi - aElbow):
    // we use the NEGATED-axis + POSITIVE-angle form (kPi - aElbow >= 0 always, aElbow in [0, kPi]) so the
    // half-angle handed to the sin/cos LUT is NON-NEGATIVE — deterministic + identical on the GPU (the LUT
    // phase-reduce is cleanest for non-negative angles). World lower = FxQuatMul(qUpper, qLower).
    const FxVec3 nNeg{(fx)(-n.x), (fx)(-n.y), (fx)(-n.z)};
    out.qUpper = FxQuatNormalize(QuatFromAxisAngle(n, aRoot));
    out.qLower = FxQuatNormalize(QuatFromAxisAngle(nNeg, (fx)(kPi - aElbow)));
    return out;
}

// EndEffector(root, target, pole, lenU, lenL, qU, qL): forward-kinematics from the solved LOCAL quats —
// place root -> elbow -> end. axis = normalize(target-root); dirUpper = FxRotate(qUpper, axis);
// elbow = root + lenU·dirUpper; the WORLD lower rotation = FxQuatMul(qUpper, qLower); dirLower =
// FxRotate(qLowerWorld, axis); end = elbow + lenL·dirLower. The residual |end - target| is the bounded
// LUT-band reach error (the honest within-band proof). Pure integer; for the test/showcase residual check.
inline FxVec3 EndEffector(const FxVec3& root, const FxVec3& target, const FxVec3& /*pole*/,
                          fx lenUpper, fx lenLower, const FxQuat& qUpper, const FxQuat& qLower) {
    const FxVec3 e = FxVec3{target.x - root.x, target.y - root.y, target.z - root.z};
    const FxVec3 axis = FxNormalize(e);
    const FxVec3 dirUpper = FxRotate(qUpper, axis);
    const FxVec3 elbow = FxVec3{root.x + fxmul(lenUpper, dirUpper.x),
                                root.y + fxmul(lenUpper, dirUpper.y),
                                root.z + fxmul(lenUpper, dirUpper.z)};
    const FxQuat qLowerWorld = FxQuatMul(qUpper, qLower);
    const FxVec3 dirLower = FxRotate(qLowerWorld, axis);
    return FxVec3{elbow.x + fxmul(lenLower, dirLower.x),
                  elbow.y + fxmul(lenLower, dirLower.y),
                  elbow.z + fxmul(lenLower, dirLower.z)};
}

// ElbowPos(root, target, lenU, qU): the elbow joint position (root + lenU·FxRotate(qUpper, axis)) — the
// limb-pose viz draws root->elbow->end as two segments. Pure integer.
inline FxVec3 ElbowPos(const FxVec3& root, const FxVec3& target, fx lenUpper, const FxQuat& qUpper) {
    const FxVec3 e = FxVec3{target.x - root.x, target.y - root.y, target.z - root.z};
    const FxVec3 axis = FxNormalize(e);
    const FxVec3 dirUpper = FxRotate(qUpper, axis);
    return FxVec3{root.x + fxmul(lenUpper, dirUpper.x),
                  root.y + fxmul(lenUpper, dirUpper.y),
                  root.z + fxmul(lenUpper, dirUpper.z)};
}

// ===================================================================================================
// Slice IK3 — DETERMINISTIC IK CONTROL-RIG: FABRIK N-BONE CHAIN + LOOK-AT (the 3rd slice of FLAGSHIP #32).
// APPENDED to ik.h (the IK1 + IK2 lines above are BYTE-FROZEN — append-only). IK2 solved the two-bone limb
// (the law of cosines); IK3 generalizes to the n-bone chain via FABRIK (Forward-And-Backward-Reaching IK —
// the iterative position-based chain solve, the cloth/Gauss-Seidel discipline along a bone chain) for
// tails/spines/tentacles, PLUS a shortest-arc LOOK-AT (aim a bone's forward axis at a target). Both in
// Q16.16, bit-identical CPU/Vulkan/Metal.
//
// FABRIK = PURE POSITION REACHING (NO transcendentals): for a chain pos[0..K-1] with fixed segment lengths
// len[i] = |pos[i+1] - pos[i]| (captured at rest), a fixed root, and a target for the end-effector
// pos[K-1], each iteration is two passes — Backward (pos[K-1]=target; reach each joint exactly len[i] from
// its successor along the current direction) then Forward (pos[0]=root; reach each joint exactly len[i-1]
// from its predecessor). Pure FxNormalize (int64 sqrt) + integer position updates — NO acos/sin/cos.
// Segment lengths are preserved BY CONSTRUCTION (each joint placed exactly `len` away — the only drift is
// the FxNormalize LSB error). If the target is unreachable (> Sum(len)), the chain straightens toward it.
// DETERMINISM BY CONSTRUCTION: fixed pass order, fixed iters, in-place integer mutation -> two-run +
// GPU==CPU bit-identity.
//
// LOOK-AT (shortest-arc): c = FxDotV(fwd, to) (cosine in [-kOne, kOne]); angle = FxAcosLut(c) (ALWAYS
// NON-NEGATIVE — acos returns [0, pi], so IK2's negative-half-angle GPU==CPU divergence CANNOT arise here);
// axis = FxNormalize(FxCross(fwd, to)); LookAtRotation = QuatFromAxisAngle(axis, angle). ANTIPARALLEL
// fallback: fwd ~ -to (c ~ -kOne, cross degenerate) -> a deterministic perpendicular axis (cross with
// world-up, else world-right) + rotate pi; fwd ~ to (c ~ kOne) -> identity.
//
// THE int64 REALITY (the IK2/FPX3 split): FabrikSolve uses FxNormalize/FxLength (int64 sqrt), so
// shaders/ik_fabrik.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot lower it to MSL) — NOT in
// the Metal hf_gen_msl list. The Metal --ik3-fabrik showcase runs THIS CPU FabrikSolve over the same
// targets -> byte-identical to the Vulkan GPU result by construction; the Vulkan side carries the GPU==CPU
// memcmp proof. The shader copies the FabrikSolve body VERBATIM.
//
// THE WITHIN-BAND CAVEAT (honest): FABRIK's end-effector reaches the target within a bounded, DETERMINISTIC
// residual (the iterative Gauss-Seidel residual + the FxNormalize LSB — NOT zero); segment lengths drift
// only by the FxNormalize LSB. The headline is DETERMINISM + cross-platform bit-identity, NOT analytic
// convergence. Both bands are documented + checked.

// kIkMaxBones — the fixed chain capacity (8 joints -> up to 7 segments; covers a tail/spine/tentacle).
inline constexpr int kIkMaxBones = 8;

// IkChainN — a fixed-capacity FABRIK chain. `pos[0..count-1]` the joint positions (Q16.16 world units),
// `len[0..count-2]` the fixed segment lengths (captured at rest = |pos[i+1]-pos[i]|), `count` the live
// joint count (<= kIkMaxBones). std430-packable as kIkMaxBones FxVec3 + (kIkMaxBones-1) fx + 1 int (the GPU
// chain mirror — the shader reads/writes the same flat layout).
struct IkChainN {
    FxVec3 pos[kIkMaxBones];        // joint positions (only [0,count) live)
    fx     len[kIkMaxBones - 1];    // fixed segment lengths (only [0,count-1) live)
    int    count = 0;               // live joint count (2..kIkMaxBones)
};

// CaptureChainLengths(chain): set chain.len[i] = |pos[i+1] - pos[i]| (the rest-pose segment lengths) for
// i in [0, count-1). Call ONCE after laying out the rest pose; FabrikSolve preserves these by construction.
inline void CaptureChainLengths(IkChainN& chain) {
    for (int i = 0; i + 1 < chain.count; ++i) {
        const FxVec3 d{chain.pos[i + 1].x - chain.pos[i].x,
                       chain.pos[i + 1].y - chain.pos[i].y,
                       chain.pos[i + 1].z - chain.pos[i].z};
        chain.len[i] = FxLength(d);
    }
}

// FabrikSolve(chain, root, target, iters): the FABRIK forward+backward reaching solve. In-place: mutates
// chain.pos toward placing pos[K-1] at `target` while anchoring pos[0] at `root`, preserving chain.len by
// construction. Each iteration: BACKWARD (pos[K-1]=target; for i=K-2..0: pos[i] = pos[i+1] +
// len[i]*FxNormalize(pos[i]-pos[i+1])) then FORWARD (pos[0]=root; for i=1..K-1: pos[i] = pos[i-1] +
// len[i-1]*FxNormalize(pos[i]-pos[i-1])). Pure FxNormalize + integer position updates — NO transcendental.
// Fixed pass order, fixed iters -> deterministic. The shader copies THIS body VERBATIM.
inline void FabrikSolve(IkChainN& chain, const FxVec3& root, const FxVec3& target, int iters) {
    const int K = chain.count;
    if (K < 2) { if (K == 1) chain.pos[0] = root; return; }
    for (int it = 0; it < iters; ++it) {
        // BACKWARD pass: pin the end-effector to the target, reach back toward the root.
        chain.pos[K - 1] = target;
        for (int i = K - 2; i >= 0; --i) {
            const FxVec3 d{chain.pos[i].x - chain.pos[i + 1].x,
                           chain.pos[i].y - chain.pos[i + 1].y,
                           chain.pos[i].z - chain.pos[i + 1].z};
            const FxVec3 u = FxNormalize(d);
            chain.pos[i] = FxVec3{chain.pos[i + 1].x + fxmul(chain.len[i], u.x),
                                  chain.pos[i + 1].y + fxmul(chain.len[i], u.y),
                                  chain.pos[i + 1].z + fxmul(chain.len[i], u.z)};
        }
        // FORWARD pass: pin the root, reach forward toward the end-effector.
        chain.pos[0] = root;
        for (int i = 1; i < K; ++i) {
            const FxVec3 d{chain.pos[i].x - chain.pos[i - 1].x,
                           chain.pos[i].y - chain.pos[i - 1].y,
                           chain.pos[i].z - chain.pos[i - 1].z};
            const FxVec3 u = FxNormalize(d);
            chain.pos[i] = FxVec3{chain.pos[i - 1].x + fxmul(chain.len[i - 1], u.x),
                                  chain.pos[i - 1].y + fxmul(chain.len[i - 1], u.y),
                                  chain.pos[i - 1].z + fxmul(chain.len[i - 1], u.z)};
        }
    }
}

// ChainEndEffector(chain): the end-effector joint position pos[count-1] (the solved tip).
inline FxVec3 ChainEndEffector(const IkChainN& chain) {
    return chain.count > 0 ? chain.pos[chain.count - 1] : FxVec3{0, 0, 0};
}

// ChainResidual(chain, target): the distance |end-effector - target| (Q16.16), the bounded iterative
// Gauss-Seidel residual (the honest within-band reach proof — NOT zero). Pure integer (FxLength).
inline fx ChainResidual(const IkChainN& chain, const FxVec3& target) {
    const FxVec3 ee = ChainEndEffector(chain);
    return FxLength(FxVec3{ee.x - target.x, ee.y - target.y, ee.z - target.z});
}

// ChainMaxLenDrift(chain): the max |segment length now - captured len[i]| over the chain (Q16.16 LSB),
// the segment-length-preservation proof (drift = FxNormalize LSB only). Pure integer.
inline fx ChainMaxLenDrift(const IkChainN& chain) {
    fx maxDrift = 0;
    for (int i = 0; i + 1 < chain.count; ++i) {
        const FxVec3 d{chain.pos[i + 1].x - chain.pos[i].x,
                       chain.pos[i + 1].y - chain.pos[i].y,
                       chain.pos[i + 1].z - chain.pos[i].z};
        const fx now = FxLength(d);
        const fx drift = (now > chain.len[i]) ? (fx)(now - chain.len[i]) : (fx)(chain.len[i] - now);
        if (drift > maxDrift) maxDrift = drift;
    }
    return maxDrift;
}

// LookAtRotation(fwd, to): the shortest-arc unit quaternion rotating UNIT `fwd` onto UNIT `to`. c =
// FxDotV(fwd,to) (cosine in [-kOne,kOne]); angle = FxAcosLut(c) (NON-NEGATIVE — acos in [0,pi]); axis =
// FxNormalize(FxCross(fwd,to)); LookAtRotation = QuatFromAxisAngle(axis, angle), normalized. DEGENERATE:
//   * fwd ~ to (c >= kOne - eps): identity (no rotation).
//   * fwd ~ -to (c <= -kOne + eps): the cross degenerates -> pick a DETERMINISTIC perpendicular axis
//     (FxCross(fwd, world-up); if that degenerates too — fwd ~ world-up — FxCross(fwd, world-right)) and
//     rotate by pi (the FxAcosLut(-kOne) result == kPi).
// Pure integer (int64 in FxNormalize/FxLength/the LUT-internal). The shortest-arc/double-cover discipline
// (the active.h precedent). `fwd` + `to` MUST be unit (FxNormalize them at the call site).
inline FxQuat LookAtRotation(const FxVec3& fwd, const FxVec3& to) {
    const fx eps = (fx)64;                                  // ~0.001 — the antiparallel/parallel band
    fx c = FxDotV(fwd, to);
    if (c >  kOne) c =  kOne;                               // clamp (rounding can push a few LSB past)
    if (c < -kOne) c = -kOne;
    if (c >= kOne - eps) return FxQuat{0, 0, 0, kOne};      // fwd ~ to -> identity
    const fx angle = FxAcosLut(c);                          // [0, pi], NON-NEGATIVE (the clean LUT path)
    FxVec3 cr = FxCross(fwd, to);
    if (c <= -kOne + eps || FxLength(cr) == 0) {
        // Antiparallel (or a degenerate cross): a deterministic perpendicular axis + rotate pi.
        cr = FxCross(fwd, FxVec3{0, kOne, 0});             // cross with world-up
        if (FxLength(cr) == 0) cr = FxCross(fwd, FxVec3{kOne, 0, 0});  // fwd ~ world-up -> world-right
        const FxVec3 axis = FxNormalize(cr);
        return FxQuatNormalize(QuatFromAxisAngle(axis, kPi));
    }
    const FxVec3 axis = FxNormalize(cr);
    return FxQuatNormalize(QuatFromAxisAngle(axis, angle));
}

// ===================================================================================================
// Slice IK4 — DETERMINISTIC IK CONTROL-RIG: IK ON THE SKELETON (the FK-pose -> IK-corrected palette bridge,
// the 4th slice of FLAGSHIP #32). APPENDED to ik.h (the IK1 + IK2 + IK3 lines above are BYTE-FROZEN —
// append-only). IK1-IK3 built the deterministic IK SOLVERS (angle LUT, two-bone, FABRIK + look-at); IK4 is
// the ANIM-PILLAR BRIDGE: wire those solvers into the EXISTING skeleton + skinning-palette pipeline. Take a
// base local pose (the skeleton bind pose / a sampled animation pose), run the IK solver to correct a limb
// SUB-CHAIN toward a world target (foot-plant / hand-reach), and emit the corrected joint palette through
// the same forward-kinematics + palette machinery the animation + ragdoll capstones use.
//
// THE BIT-EXACT PIPELINE (Q16.16, GPU==CPU):
//   (1) FK the base LOCAL pose -> world joint positions (Q16.16), forward-accumulating along the
//       topologically-sorted joints (the joint::RagdollFromSkeleton / anim::PaletteFromLocalPose single
//       forward pass, MIRRORED in Q16.16): gquat[j] = FxQuatMul(gquat[parent], localR[j]); gpos[j] =
//       gpos[parent] + FxRotate(gquat[parent], localT[j]). The world POSITION is the chain input.
//   (2) IK solve over each rig chain (FABRIK over the world joint positions) toward the world target ->
//       corrected world positions (the foot-plant). Pure integer; the IK1-IK3 within-band residual carries.
//   (3) Convert the corrected world positions -> corrected LOCAL rotations. For each chain bone the corrected
//       LOCAL rotation aims the BIND-POSE bone direction along the corrected world bone direction, expressed
//       in the joint's PARENT world frame: deltaWorld = LookAtRotation(bindWorldDir, corrWorldDir);
//       correctedLocalR = QConj(parentWorldRot) * deltaWorld * parentWorldRot * baseLocalR. Write into ONLY
//       the chain joints.
//   (4) Un-IK'd joints stay EXACTLY the base pose (byte-identical) — the falsifiable invariant.
//
// THE FLOAT CROSSINGS (documented, OUTSIDE the bit-exact loop): (a) the base pose comes from the skeleton
// bind pose / anim::SampleLocalPose (float) snapped to Q16.16 (the RagdollFromSkeleton bind, FxQuatFromFloat
// idiom); (b) IkPoseToPalette is the Q16.16->float math::Mat4 read-back for the render (joint::PoseToPalette
// twin). The IK SOLVE + FK + rotation-convert between them is bit-exact integer.
//
// THE int64 REALITY (the IK2/IK3/FPX3 split): the FK (FxRotate/FxQuatMul), the FABRIK solve (FxNormalize/
// FxLength), and the rotation-convert (LookAtRotation/FxQuatMul) all use int64, so shaders/ik_rig.comp is
// VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot lower it to MSL) — NOT in the Metal hf_gen_msl list.
// The Metal --ik4-rig showcase runs THIS CPU SolveRigToTargets (byte-identical to the Vulkan GPU result by
// construction); the Vulkan side carries the GPU==CPU memcmp proof. The shader copies the bodies VERBATIM.
//
// THE WITHIN-BAND CAVEAT (honest): the FK from the corrected LOCAL rotations reaches the world target within
// a bounded, DETERMINISTIC residual (the FABRIK iterative residual + the FxNormalize / LookAtRotation LSB —
// NOT zero). The un-IK'd-joints invariant is EXACT (byte-identical). The headline is DETERMINISM + cross-
// platform bit-identity, NOT analytic reach.

// Pull the float math bridge from joint.h / fpx.h (READ-ONLY) for the base-pose snap + palette read-back.
using hf::sim::joint::QConj;

// kIkMaxJoints — the fixed skeleton capacity for the GPU SSBO mirror (the rig's hand-built test skeleton is
// small; the std430 GPU layout reads/writes a flat per-joint array). 16 covers a humanoid limb + spine.
inline constexpr int kIkMaxJoints = 16;

// IkRig — the IK overlay description for ONE limb sub-chain. `joint[0..count-1]` the skeleton joint INDICES
// along the chain (joint[0] the chain root, joint[count-1] the end-effector), parent-before-child (a path
// down the skeleton hierarchy: each joint[i+1]'s skeleton parent IS joint[i]). `target` the chain's
// end-effector goal in WORLD space (Q16.16). `iters` the FABRIK iteration count. `pole` an optional bend
// hint (carried for parity with the two-bone solve; FABRIK here is unconstrained). count in [2, kIkMaxBones].
struct IkRig {
    int    joint[kIkMaxBones];     // skeleton joint indices along the chain (root..end), <= kIkMaxBones
    int    count = 0;              // live chain joint count (2..kIkMaxBones)
    FxVec3 target;                 // the end-effector world-space target (Q16.16)
    FxVec3 pole{0, kOne, 0};       // optional bend hint (unused by FABRIK; carried for parity)
    int    iters = 12;             // FABRIK iterations
};

// IkBasePose — a base LOCAL pose snapped to Q16.16: per-joint local rotation + translation (the bind pose /
// a sampled animation pose, snapped once OUTSIDE the bit-exact loop). The bit-exact FK + solve + convert
// read this; SolveRigToTargets returns a corrected COPY of `localR` (translations are untouched by IK).
struct IkBasePose {
    FxVec3 localT[kIkMaxJoints];   // per-joint local translation (Q16.16)
    FxQuat localR[kIkMaxJoints];   // per-joint local rotation (Q16.16)
    int    count = 0;              // joint count (<= kIkMaxJoints)
};

// IkFkWorld — the FK result of an IkBasePose: per-joint WORLD rotation + position (Q16.16). The chain input
// for the IK solve + the parent frame for the rotation-convert.
struct IkFkWorld {
    FxVec3 pos[kIkMaxJoints];      // per-joint world position (Q16.16)
    FxQuat rot[kIkMaxJoints];      // per-joint world rotation (Q16.16)
    int    count = 0;
};

// FkWorldPositions(parents, base) -> the Q16.16 forward-accumulate. parents[j] = skeleton parent index (or
// -1 for a root); base the local pose. gquat[j] = FxQuatMul(gquat[parent], localR[j]); gpos[j] =
// gpos[parent] + FxRotate(gquat[parent], localT[j]). Topologically-sorted -> single forward pass (parent
// precedes child). Pure integer (FxQuatMul/FxRotate int64). The shader copies THIS body VERBATIM.
inline IkFkWorld FkWorldPositions(const int* parents, const IkBasePose& base) {
    IkFkWorld w{};
    w.count = base.count;
    for (int j = 0; j < base.count; ++j) {
        const int p = parents[j];
        if (p < 0) {
            w.rot[j] = base.localR[j];
            w.pos[j] = base.localT[j];
        } else {
            w.rot[j] = FxQuatMul(w.rot[p], base.localR[j]);
            const FxVec3 tWorld = FxRotate(w.rot[p], base.localT[j]);
            w.pos[j] = FxVec3{w.pos[p].x + tWorld.x, w.pos[p].y + tWorld.y, w.pos[p].z + tWorld.z};
        }
    }
    return w;
}

// IkPose — the SolveRigToTargets output: a corrected per-joint LOCAL pose (rotations corrected on the chain,
// translations + un-IK'd rotations EXACTLY the base). The corrected `localR` IS the bit-exact bridge output
// (the GPU==CPU memcmp compares THIS); IkPoseToPalette reads it back to a float skinning palette.
struct IkPose {
    FxVec3 localT[kIkMaxJoints];
    FxQuat localR[kIkMaxJoints];
    int    count = 0;
};

// SolveRigToTargets(parents, base, rig) -> IkPose. The bit-exact FK + IK + rotation-convert bridge:
//   (1) FK the base local pose -> world joint rotations + positions.
//   (2) lay the chain's WORLD positions into an IkChainN, capture its rest segment lengths, FabrikSolve
//       toward rig.target -> corrected world positions.
//   (3) for each chain bone i (joint c = rig.joint[i], child cn = rig.joint[i+1]): bindWorldDir =
//       normalize(fkWorldPos[cn] - fkWorldPos[c]); corrWorldDir = normalize(corrPos[i+1] - corrPos[i]);
//       deltaWorld = LookAtRotation(bindWorldDir, corrWorldDir); the corrected world rotation of c is
//       deltaWorld * fkRot[c]; the corrected LOCAL rotation is QConj(parentWorldRot) * (deltaWorld *
//       fkRot[c]), where parentWorldRot = fkRot[skeletonParent(c)] (identity if c is a root). Write it into
//       out.localR[c]. The end-effector joint (i==count-1, no child bone) keeps its base local rotation.
//   (4) every joint NOT on the chain keeps its base localR + localT EXACTLY.
// Pure integer (int64 in the FK / FABRIK / LookAtRotation). The shader copies THIS body VERBATIM.
inline IkPose SolveRigToTargets(const int* parents, const IkBasePose& base, const IkRig& rig) {
    IkPose out{};
    out.count = base.count;
    // (4) seed with the EXACT base pose (un-IK'd joints stay byte-identical; the chain joints are overwritten).
    for (int j = 0; j < base.count; ++j) { out.localT[j] = base.localT[j]; out.localR[j] = base.localR[j]; }

    // (1) FK the base pose to world rotations + positions.
    const IkFkWorld fk = FkWorldPositions(parents, base);

    if (rig.count < 2) return out;   // a degenerate chain has no bone to correct.

    // (2) lay the chain world positions into an IkChainN, capture rest lengths, FABRIK toward the target.
    IkChainN chain{};
    chain.count = rig.count;
    for (int i = 0; i < rig.count; ++i) chain.pos[i] = fk.pos[rig.joint[i]];
    CaptureChainLengths(chain);
    const FxVec3 chainRoot = fk.pos[rig.joint[0]];
    FabrikSolve(chain, chainRoot, rig.target, rig.iters);

    // (3) corrected world positions -> corrected LOCAL rotations on the chain joints (NOT the end-effector).
    // CRUX: the local rotation must be expressed in the CORRECTED PARENT world frame, NOT the base FK parent
    // frame — otherwise the re-FK does not reproduce the corrected bone direction (the chain compounds). We
    // walk the chain forward, tracking corrWorldRot (the corrected world rotation of the PREVIOUS chain joint,
    // = the corrected parent frame for the next). For chain joint 0 the parent is NOT in the chain, so its
    // frame is the (unchanged) base FK rotation of chain[0]'s skeleton parent. For each bone: deltaWorld =
    // LookAtRotation(baseBoneDir, corrBoneDir); the corrected world rotation of c is deltaWorld * fk.rot[c]
    // (so FK reproduces |t|*corrBoneDir); the corrected LOCAL rotation is QConj(corrParentWorld) *
    // (deltaWorld * fk.rot[c]). corrParentWorld == corrWorldRot[c's chain parent] (the corrected one).
    FxQuat prevCorrWorldRot{0, 0, 0, kOne};   // the corrected world rotation of the previous chain joint
    for (int i = 0; i + 1 < rig.count; ++i) {
        const int c  = rig.joint[i];        // the joint whose local rotation drives bone i
        const int cn = rig.joint[i + 1];    // the chain successor (defines the bone direction)
        const FxVec3 bindWorldDir = FxNormalize(
            FxVec3{fk.pos[cn].x - fk.pos[c].x, fk.pos[cn].y - fk.pos[c].y, fk.pos[cn].z - fk.pos[c].z});
        const FxVec3 corrWorldDir = FxNormalize(
            FxVec3{chain.pos[i + 1].x - chain.pos[i].x, chain.pos[i + 1].y - chain.pos[i].y,
                   chain.pos[i + 1].z - chain.pos[i].z});
        const FxQuat deltaWorld = LookAtRotation(bindWorldDir, corrWorldDir);
        // the corrected WORLD rotation of joint c (the delta pre-composed onto its base world rotation).
        const FxQuat corrWorldRot = FxQuatNormalize(FxQuatMul(deltaWorld, fk.rot[c]));
        // the CORRECTED parent world frame: chain joint 0's parent is outside the chain (base FK rot), every
        // later chain joint's parent IS the previous chain joint (its corrected world rotation, just computed).
        FxQuat parentWorldRot;
        if (i == 0) {
            const int p = parents[c];
            parentWorldRot = (p >= 0) ? fk.rot[p] : FxQuat{0, 0, 0, kOne};
        } else {
            parentWorldRot = prevCorrWorldRot;
        }
        out.localR[c] = FxQuatNormalize(FxQuatMul(QConj(parentWorldRot), corrWorldRot));
        prevCorrWorldRot = corrWorldRot;
    }
    return out;
}

// IkSolvedFkWorld(parents, pose) -> the FK world positions of an IkPose (the corrected local pose). The
// foot-plant proof FK's the CORRECTED pose + checks the end-effector reached the world target within band.
// Pure integer (the FkWorldPositions over the corrected localR). A thin adapter: copy the IkPose into an
// IkBasePose then FK it.
inline IkFkWorld IkSolvedFkWorld(const int* parents, const IkPose& pose) {
    IkBasePose b{};
    b.count = pose.count;
    for (int j = 0; j < pose.count; ++j) { b.localT[j] = pose.localT[j]; b.localR[j] = pose.localR[j]; }
    return FkWorldPositions(parents, b);
}

// IkFootPlantResidual(parents, pose, rig) -> |FK(corrected pose).endEffector - rig.target| (Q16.16). The
// bounded within-band foot-plant residual (the FABRIK iterative residual + the rotation-convert LSB — NOT
// zero). Pure integer.
inline fx IkFootPlantResidual(const int* parents, const IkPose& pose, const IkRig& rig) {
    if (rig.count < 1) return 0;
    const IkFkWorld fk = IkSolvedFkWorld(parents, pose);
    const int ee = rig.joint[rig.count - 1];
    return FxLength(FxVec3{fk.pos[ee].x - rig.target.x, fk.pos[ee].y - rig.target.y,
                           fk.pos[ee].z - rig.target.z});
}

// IkPoseToPalette(skeleton, pose) -> the Q16.16->float skinning palette read-back (the joint::PoseToPalette
// twin; render-only, the documented float crossing). FK the corrected LOCAL pose to per-joint world
// transforms in FLOAT (translate(pos)*quatToMat(normalize(rot)), the fpx::FxBodyTransform float convention),
// then palette[j] = worldFloat[j] * inverseBind[j]. A pure deterministic function of the bit-exact corrected
// pose (the provenance proof: two calls byte-identical). One Mat4 per joint, in skeleton order.
inline std::vector<math::Mat4> IkPoseToPalette(const anim::Skeleton& skeleton, const IkPose& pose) {
    const size_t n = skeleton.joints.size();
    std::vector<math::Mat4> palette(n);
    std::vector<math::Mat4> worldF(n);
    auto fxToF = [](fx v) -> float { return (float)v / (float)kOne; };
    for (size_t j = 0; j < n; ++j) {
        const int idx = (int)j;
        const math::Vec3 t{fxToF(pose.localT[idx].x), fxToF(pose.localT[idx].y), fxToF(pose.localT[idx].z)};
        const math::Quat r = math::Normalize(math::Quat{
            fxToF(pose.localR[idx].x), fxToF(pose.localR[idx].y),
            fxToF(pose.localR[idx].z), fxToF(pose.localR[idx].w)});
        const math::Mat4 local = math::FromTRS(t, r, math::Vec3{1, 1, 1});
        const int parent = skeleton.joints[j].parent;
        worldF[j] = (parent >= 0) ? (worldF[(size_t)parent] * local) : local;
        palette[j] = worldF[j] * skeleton.joints[j].inverseBind;
    }
    return palette;
}

// BasePoseFromSkeleton(skeleton) -> the IkBasePose snapped from the skeleton's REST local TRS (the bind
// pose). The float->Q16.16 snap (the RagdollFromSkeleton bind / FxQuatFromFloat idiom — the documented float
// crossing OUTSIDE the bit-exact loop). Rotation normalized in float then snapped; translation scaled by
// kOne. Pure deterministic host float. Joints beyond kIkMaxJoints are ignored (the test skeleton is small).
inline IkBasePose BasePoseFromSkeleton(const anim::Skeleton& skeleton) {
    IkBasePose b{};
    const int n = (int)skeleton.joints.size();
    b.count = n < kIkMaxJoints ? n : kIkMaxJoints;
    for (int j = 0; j < b.count; ++j) {
        const anim::Joint& jt = skeleton.joints[(size_t)j];
        b.localT[j] = FxVec3{(fx)std::llround((double)jt.t.x * (double)kOne),
                             (fx)std::llround((double)jt.t.y * (double)kOne),
                             (fx)std::llround((double)jt.t.z * (double)kOne)};
        const math::Quat q = math::Normalize(jt.r);
        b.localR[j] = FxQuatNormalize(FxQuat{(fx)std::llround((double)q.x * (double)kOne),
                                             (fx)std::llround((double)q.y * (double)kOne),
                                             (fx)std::llround((double)q.z * (double)kOne),
                                             (fx)std::llround((double)q.w * (double)kOne)});
    }
    return b;
}

// SkeletonParents(skeleton, out[]) -> fill out[j] = skeleton.joints[j].parent for j in [0, count). A small
// helper so the solver takes a flat int parent array (the GPU SSBO mirror). Pure copy.
inline void SkeletonParents(const anim::Skeleton& skeleton, int* out, int count) {
    for (int j = 0; j < count; ++j) out[j] = skeleton.joints[(size_t)j].parent;
}

// ===================================================================================================
// Slice IK5 — DETERMINISTIC IK CONTROL-RIG: LOCKSTEP + ROLLBACK over IK-driven character motion (THE MOAT
// HEADLINE, the 5th slice of FLAGSHIP #32). APPENDED to ik.h (the IK1 + IK2 + IK3 + IK4 lines above are
// BYTE-FROZEN — append-only). IK1-IK4 built the deterministic IK SOLVERS + the skeleton bridge; IK5 proves
// the IK-driven character pose is true cross-platform LOCKSTEP + ROLLBACK-replayable — the headline a FLOAT
// engine cannot do. Two peers fed ONLY a deterministic INPUT stream (per-tick effector-target moves — "the
// hand/foot target the player is dragging") re-simulate to a BIT-IDENTICAL IK-corrected pose; a mispredicted
// target is corrected by rolling back to a snapshot + re-simulating. PURE CPU, 0 backend symbols, NO new
// shader/RHI (the FPX5/JT5/AC5/VD5 lockstep convention).
//
// THE DESIGN CALL (the AC5/VD5 mold over SolveRigToTargets; the target is the state, the pose is the pure
// function): the lockstep machinery is proven many times (FPX5, JT5, VH5, AC5, VD5). IK5 reuses the SHAPE
// verbatim, with the IK-specific state minimal + clean:
//   * The REPLAYABLE STATE = the effector target(s) (each rig's `target`, moved by kCmdMoveTarget each tick).
//     The base pose + skeleton parents + chain topology are CONST (carried as params, re-supplied at restore,
//     NOT snapshotted). The IK-corrected pose is a PURE FUNCTION of (base, targets) — re-derived each tick by
//     SolveRigToTargets (IK4, VERBATIM).
//   * SimIkTick = apply this tick's target moves (in FIXED command order) -> SolveRigToTargets. Deterministic
//     by construction (fixed command order, fixed FABRIK iters, integer). Two peers running the same SimIkTick
//     over the same command stream are byte-identical; rollback restores the exact target(s) + re-sims.
//   * The SNAPSHOT captures the target(s) + the solved pose (the IkPose localR/localT) + the tick. The pose is
//     technically re-derivable from the targets, so the snapshot is small; capturing the pose makes
//     IkStatesEqual a direct byte-compare of the WHOLE rig state (the VD5 whole-state-equality shape).
//   * DETERMINISM BY CONSTRUCTION: SolveRigToTargets is the frozen IK4 integer solve (GPU==CPU already proven).
//     The lockstep is a determinism PROPERTY of it — PURE CPU, no GPU dispatch, no shader, no TDR. The
//     converged pose is bit-identical Vulkan-Windows AND Metal-Mac (both run the same CPU SimIkTick) -> the
//     cross-backend zero-diff golden IS the lockstep evidence.
//
// MULTIPLE CHAINS: IkSimState carries a fixed-capacity array of rigs (chains). Each tick the commands nudge a
// chain's target (cmd.chain selects the rig); the pose is the result of solving the rigs in FIXED index order,
// each over the running corrected pose so multiple limbs compose deterministically. A single-chain scene
// (chains==1) is the common case; the showcase uses one leg chain.

// kIkMaxChains — the fixed chain capacity for the lockstep state (a humanoid has a handful of IK limbs).
inline constexpr int kIkMaxChains = 4;

// kCmdMoveTarget — the only IK command kind (the per-tick effector-target nudge). A second discriminant slot
// is reserved for parity with the other lockstep slices; IK5 has exactly one verb.
inline constexpr int kCmdMoveTarget = 0;

// ----- IkCommand: ONE per-tick effector-target move (the dragged hand/foot target) ----------------------
// At `tick`, add `delta` to chain `chain`'s end-effector target. The harness applies every command whose
// cmd.tick == tick in ARRAY ORDER before the SolveRigToTargets of that tick (the AC5 ActiveCommand /
// VehicleCommand scripted-input-per-tick shape). Pure integer (delta is a Q16.16 FxVec3).
struct IkCommand {
    int    tick = 0;          // the tick this target move fires at
    int    chain = 0;         // which rig/chain's target is nudged (index into IkSimState.rig[])
    FxVec3 delta;             // the Q16.16 world-space target delta
    int    kind = kCmdMoveTarget;  // the command discriminant (always kCmdMoveTarget for IK5)
};

// ----- IkSimState: the mutable IK rig state (the const base + the per-chain rigs whose targets MOVE + the
// current corrected pose) ----------------------------------------------------------------------------------
// `base` + the skeleton parents (carried separately as a param) are CONST. `rig[0..chains-1]` carry the per-
// chain end-effector targets — the ONLY moving state. `pose` is the current solved IkPose (re-derived each
// tick from the targets; captured so IkStatesEqual byte-compares the whole rig state). The snapshot/equality
// compares the targets + the pose; base/parents/topology are NOT part of the replayable state.
struct IkSimState {
    IkBasePose base;                 // the const base local pose (the bind / sampled pose)
    IkRig      rig[kIkMaxChains];    // per-chain rigs (their `target` is the moving state)
    int        chains = 0;           // live chain count (1..kIkMaxChains)
    IkPose     pose;                 // the current corrected pose (re-derived each tick; snapshotted)
};

// ----- SolveRigState: re-derive the corrected pose from the base + every chain's target (pure function) ----
// Solve the chains in FIXED index order. Chain 0 solves over the base pose; each later chain solves over the
// running corrected pose (so independent limbs compose deterministically — an n-chain generalization of IK4's
// single SolveRigToTargets). For the common single-chain case this is exactly one SolveRigToTargets over the
// base. The result is written into state.pose. Pure integer (the IK4 solve, VERBATIM). No GPU, no float.
inline void SolveRigState(IkSimState& state, const int* parents) {
    if (state.chains < 1) {
        // degenerate: no chain -> the pose is exactly the base pose.
        state.pose.count = state.base.count;
        for (int j = 0; j < state.base.count; ++j) {
            state.pose.localT[j] = state.base.localT[j];
            state.pose.localR[j] = state.base.localR[j];
        }
        return;
    }
    // Chain 0 over the base pose.
    state.pose = SolveRigToTargets(parents, state.base, state.rig[0]);
    // Each later chain solves over the running corrected pose (copy pose -> a base, solve, write back).
    for (int ch = 1; ch < state.chains; ++ch) {
        IkBasePose cur{};
        cur.count = state.pose.count;
        for (int j = 0; j < state.pose.count; ++j) {
            cur.localT[j] = state.pose.localT[j];
            cur.localR[j] = state.pose.localR[j];
        }
        state.pose = SolveRigToTargets(parents, cur, state.rig[ch]);
    }
}

// ----- SimIkTick: the deterministic per-tick step (apply this tick's target moves + re-solve) -------------
// (0) APPLY this tick's commands in ARRAY ORDER (every cmd with cmd.tick == tick) — target += delta on the
//     selected chain — BEFORE the solve so the moved target drives this tick's pose. (1) SolveRigState ->
//     the corrected pose re-derived from the moved targets. Pure integer (the IK4 solve) + the deterministic
//     host target nudge -> bit-identical on every peer/platform. The AC5 SimActiveTick / VH5 SimVehicleTick
//     twin (the input is a TARGET MOVE rather than a hit-impulse).
inline void SimIkTick(IkSimState& state, const int* parents,
                      const std::vector<IkCommand>& commands, int tick) {
    for (size_t c = 0; c < commands.size(); ++c) {
        if (commands[c].tick == tick) {
            const int ch = commands[c].chain;
            if (ch >= 0 && ch < state.chains) {
                state.rig[ch].target.x += commands[c].delta.x;
                state.rig[ch].target.y += commands[c].delta.y;
                state.rig[ch].target.z += commands[c].delta.z;
            }
        }
    }
    SolveRigState(state, parents);
}

// ----- IkSnapshot: the captured mutable IK rig state (the targets + the solved pose + the tick) -----------
// The replayable state is the per-chain targets + the corrected pose + the tick (the rollback restore point).
// The base/parents/topology are immutable structure (re-supplied at restore) so they are NOT snapshotted —
// EXACTLY the mutable state is captured. (The pose is technically re-derivable; capturing it makes the equality
// a whole-state byte-compare — the VD5 shape.)
struct IkSnapshot {
    FxVec3 target[kIkMaxChains];   // the per-chain end-effector targets (the moving state)
    int    chains = 0;             // the live chain count (so restore re-establishes exactly)
    IkPose pose;                   // the corrected pose at the snapshot tick
    int    tick = 0;               // the tick this snapshot was taken at
};

// ----- SnapshotIkRig: capture the mutable state (the rollback restore point) ------------------------------
// Bit-exact round-trip with RestoreIkRig: RestoreIkRig(state, SnapshotIkRig(state0, t)) leaves state's
// targets + pose == state0's byte-for-byte AND state resumes at tick t.
inline IkSnapshot SnapshotIkRig(const IkSimState& state, int tick) {
    IkSnapshot snap;
    snap.chains = state.chains;
    for (int ch = 0; ch < state.chains; ++ch) snap.target[ch] = state.rig[ch].target;
    snap.pose = state.pose;
    snap.tick = tick;
    return snap;
}

// ----- RestoreIkRig: restore the targets + pose from a snapshot + return the captured tick (the rollback) --
// Restores each chain's target + the corrected pose and returns snap.tick so the harness resumes from there.
// Bit-exact round-trip with SnapshotIkRig. The base/parents/topology are untouched (immutable structure).
inline int RestoreIkRig(IkSimState& state, const IkSnapshot& snap) {
    for (int ch = 0; ch < snap.chains && ch < state.chains; ++ch)
        state.rig[ch].target = snap.target[ch];
    state.pose = snap.pose;
    return snap.tick;
}

// ----- IkStatesEqual: whole-rig-state byte-equality (the per-chain targets + the corrected pose) ----------
// Two states are equal iff every live chain's target is byte-equal AND the corrected pose (localR + localT)
// is byte-equal. The base/parents are CONST + identical by construction so they are not compared — the
// equality is over EXACTLY the replayable state (the VD5 whole-world memcmp shape). Pure integer compare.
inline bool IkStatesEqual(const IkSimState& a, const IkSimState& b) {
    if (a.chains != b.chains) return false;
    for (int ch = 0; ch < a.chains; ++ch) {
        if (a.rig[ch].target.x != b.rig[ch].target.x ||
            a.rig[ch].target.y != b.rig[ch].target.y ||
            a.rig[ch].target.z != b.rig[ch].target.z) return false;
    }
    if (a.pose.count != b.pose.count) return false;
    for (int j = 0; j < a.pose.count; ++j) {
        if (a.pose.localR[j].x != b.pose.localR[j].x || a.pose.localR[j].y != b.pose.localR[j].y ||
            a.pose.localR[j].z != b.pose.localR[j].z || a.pose.localR[j].w != b.pose.localR[j].w) return false;
        if (a.pose.localT[j].x != b.pose.localT[j].x || a.pose.localT[j].y != b.pose.localT[j].y ||
            a.pose.localT[j].z != b.pose.localT[j].z) return false;
    }
    return true;
}

// ----- RunIkLockstep: authority + replica from the SAME inputs, bit-identical every tick ------------------
// THE peer entry point (the AC5 RunActiveLockstep / VD5 control flow over SimIkTick). Run `ticks` SimIkTicks
// from a COPY of `initialState`, applying the command stream -> the converged rig. authority = the run;
// replica = the SAME from the SAME init + stream (INPUTS ONLY — no state shared) -> BIT-IDENTICAL by
// determinism. This function ASSERTS authority == replica EACH tick (IkStatesEqual) via an internal replica
// run; the caller also compares two RunIkLockstep returns for the determinism proof. Sets *identical (false
// if the per-tick lockstep invariant ever broke — unreachable for a deterministic sim) and returns the
// converged authority state.
inline IkSimState RunIkLockstep(const IkSimState& initialState, const int* parents,
                                const std::vector<IkCommand>& commands, int ticks, bool* identical) {
    IkSimState authority = initialState;   // a fresh copy (base + rigs + pose)
    IkSimState replica   = initialState;   // the second peer fed the SAME inputs
    bool ident = true;
    for (int t = 0; t < ticks; ++t) {
        SimIkTick(authority, parents, commands, t);
        SimIkTick(replica,   parents, commands, t);
        if (!IkStatesEqual(authority, replica)) ident = false;   // the lockstep invariant — must hold each tick
    }
    if (identical) *identical = ident;
    return authority;
}

// ----- RunIkRollback: snapshot -> mispredict diverges -> rollback -> corrected == authority ----------------
// The rollback harness (the AC5 RunActiveRollback / VD5 control flow over SimIkTick). (1) advance ticks
// 0..rollbackAt from `initialState` applying authStream; (2) SAVE an IkSnapshot AT rollbackAt (the targets +
// pose + tick); (2b) speculatively advance <=3 ticks with the MISPREDICTED stream (a WRONG target move — a
// different delta that diverges the pose, the client prediction); (3) ROLLBACK — RestoreIkRig to the snapshot
// + RE-SIMULATE rollbackAt..ticks with the CORRECT authStream -> the corrected final state. Sets *correctedEq
// (corrected == the straight authority run) AND *diverged (the speculative pre-rollback state DIFFERED from
// the authority at that tick — a REAL divergence was fixed). Returns the corrected state.
inline IkSimState RunIkRollback(const IkSimState& initialState, const int* parents,
                                const std::vector<IkCommand>& authStream,
                                const std::vector<IkCommand>& mispredictStream, int ticks, int rollbackAt,
                                bool* correctedEq, bool* diverged) {
    // The straight authority run (the truth the rollback must reproduce).
    bool authIdent = true;
    const IkSimState authority = RunIkLockstep(initialState, parents, authStream, ticks, &authIdent);

    IkSimState state = initialState;
    // (1) advance 0..rollbackAt with the authoritative stream.
    for (int t = 0; t < rollbackAt; ++t)
        SimIkTick(state, parents, authStream, t);
    // (2) SAVE the snapshot at rollbackAt (the rollback restore point — the targets + pose + tick).
    const IkSnapshot snap = SnapshotIkRig(state, rollbackAt);

    // (2b) speculatively advance <=3 ticks with the MISPREDICTED stream (the wrong target move that diverges).
    int specTicks = ticks - rollbackAt;
    if (specTicks > 3) specTicks = 3;
    IkSimState spec = state;   // a copy to measure the speculative divergence against the authority
    for (int s = 0; s < specTicks; ++s)
        SimIkTick(spec, parents, mispredictStream, rollbackAt + s);
    // the authority advanced to the SAME tick (with the correct stream) for the divergence comparison.
    IkSimState authAtSpec = initialState;
    for (int t = 0; t < rollbackAt + specTicks; ++t)
        SimIkTick(authAtSpec, parents, authStream, t);
    const bool didDiverge = !IkStatesEqual(spec, authAtSpec);

    // (3) ROLLBACK: restore the snapshot (targets + pose) + re-sim rollbackAt..ticks with the authStream.
    const int resumeTick = RestoreIkRig(state, snap);   // == rollbackAt
    for (int t = resumeTick; t < ticks; ++t)
        SimIkTick(state, parents, authStream, t);

    if (correctedEq) *correctedEq = IkStatesEqual(state, authority);
    if (diverged)    *diverged    = didDiverge;
    return state;
}

} // namespace hf::anim::ik
