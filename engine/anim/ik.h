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

} // namespace hf::anim::ik
