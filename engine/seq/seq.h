#pragma once
// Slice SEQ-S1 — Scalar keyframe track + integer lerp (Issue #25, the BEACHHEAD of a DETERMINISTIC
// CINEMATIC SEQUENCER runtime). The timeline-evaluation core a sequencer editor would drive (the
// visual editor GUI is OUT OF SCOPE). THE MOAT: UE5 Sequencer is float to the bone — float keyframe
// values, float curve interpolation, FQuat::Slerp via acos/sin, float playback timing — so two
// machines sampling the same sequence at the same time diverge in the low bits. A timeline whose
// evaluation is BIT-IDENTICAL cross-platform + lockstep/replay/scrub-able is the moat extension UE5
// lacks — the sibling of the deterministic-Blueprints flow.h.
//
// S1 establishes the substrate: a Q16.16 scalar keyframe track + integer keyframe interpolation (the
// float that breaks UE5, rebuilt in fixed point). Pure-CPU INTEGER — every sample is fxmul/fxdiv on
// int32 in Q16.16 (the engine's fpx toolbox), NO float, NO <cmath>, NO clock/RNG. The golden is a
// hard-pinned net::DigestBytes over a sampled sweep, proven identical Windows/MSVC + Mac/clang via a
// standalone clang compile — NO render-bake, the cheapest cross-platform proof.
//
// SELF-CONTAINED ON PURPOSE: this header includes ONLY <cstddef>/<cstdint>/<vector> plus sim/fpx.h
// (the Q16.16 toolbox: fx/kOne/fxmul/fxdiv) and net/session.h (hf::net::DigestBytes). NO <algorithm>
// (the binary search is a hand-written integer loop), NO <cmath>/float on the bit-exact path, NO
// <unordered_*>/<map>/<random>/<functional>/std::hash. It compiles STANDALONE under
// `clang++ -std=c++20 -I engine -I tests tests/seq_test.cpp` (fpx.h pulls only math/math.h + std,
// session.h pulls only std). It does NOT include or reuse anim/animation.h — that is the FLOAT
// keyframe path (the contrast), left untouched + golden-invariant; seq rebuilds interpolation in
// Q16.16 from scratch.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sim/fpx.h"      // the Q16.16 toolbox: fx (int32 Q16.16) / kOne / fxmul / fxdiv
#include "net/session.h"  // hf::net::DigestBytes (FNV-1a-64 over raw bytes)

namespace hf::seq {

// Bring the Q16.16 fixed-point toolbox into hf::seq (the exact names/namespace fpx.h defines).
using hf::sim::fpx::fx;      // Q16.16 fixed-point scalar (int32_t)
using hf::sim::fpx::kOne;    // 1.0 in Q16.16 (65536)
using hf::sim::fpx::fxmul;   // (int64)a*b >> 16  (deterministic arithmetic shift)
using hf::sim::fpx::fxdiv;   // ((int64)a << 16) / b  (the engine's guarded fixed divide; b==0 -> 0)

// The keyframe interpolation mode. S2 APPENDS curve modes (EaseInOutSine, etc.) — DO NOT renumber
// S1's two (the golden + any serialized track pin the enum values).
enum class Easing : uint32_t {
    Step          = 0,   // S1 — frozen: hold the lower key value
    Linear        = 1,   // S1 — frozen: identity ease, t01 used directly
    EaseInOutSine = 2,   // smooth S-curve (the showcase ease) — (1 - cos(pi*t))/2
    EaseInQuad    = 3,   // accelerate from rest — t*t
    EaseOutQuad   = 4,   // decelerate to rest — 1 - (1-t)*(1-t)
};

// A scalar keyframe track: parallel (times, values) arrays in Q16.16. The INVARIANT: `times` is
// STRICTLY ASCENDING (sorted keys, no duplicates) and values.size() == times.size().
struct ScalarTrack {
    std::vector<fx> times;                  // Q16.16 seconds, STRICTLY ASCENDING
    std::vector<fx> values;                 // Q16.16 keyframe values; values.size() == times.size()
    Easing          easing = Easing::Linear;
};

// ============================ S2 — easing-curve LUTs (host-baked, INTEGER) =========================
// The float acos/sin that breaks UE5 Sequencer, rebuilt as a host-baked Q16.16 LUT — ZERO runtime
// transcendentals (the ik.h BuildSinLut / FxSinLut precedent). A fixed-resolution table over the unit
// interval [0,kOne], read by a scaled-index + fractional-lerp. kEaseBins+1 entries (incl. the t=1
// endpoint); entry[i] = ease(i/kEaseBins) in Q16.16, with entry[0]==0 and entry[kEaseBins]==kOne.
constexpr int kEaseBins  = 256;       // table has kEaseBins+1 entries (incl. the t=1 endpoint)
constexpr int kEaseShift = 8;         // kEaseBins == 1<<kEaseShift

// EaseInOutSine table = (1 - cos(pi*t))/2 — needs a transcendental at BAKE time, so it is a COMMITTED
// integer literal array (bake-and-paste) to keep this header <cmath>-free (HARD constraint). The bytes
// ARE the spec; a pinned digest (seq_test) catches any regeneration. Generated ONCE with a throwaway
// program: entry[i] = round((1 - cos(pi*i/256))/2 * 65536), i in [0,256]. DO NOT recompute here.
inline const std::vector<fx>& SineEaseTable() {
    static const std::vector<fx> tbl = {
        0, 2, 10, 22, 39, 62, 89, 121, 158, 200, 246, 298,
        355, 416, 482, 554, 630, 710, 796, 887, 982, 1082, 1187, 1297,
        1411, 1530, 1654, 1782, 1915, 2053, 2196, 2343, 2494, 2650, 2811, 2976,
        3146, 3320, 3499, 3682, 3869, 4061, 4257, 4457, 4662, 4871, 5084, 5301,
        5522, 5748, 5977, 6211, 6448, 6690, 6935, 7185, 7438, 7695, 7956, 8220,
        8489, 8760, 9036, 9315, 9598, 9884, 10173, 10466, 10762, 11062, 11365, 11671,
        11980, 12293, 12608, 12927, 13248, 13573, 13900, 14230, 14563, 14899, 15237, 15578,
        15922, 16268, 16617, 16968, 17321, 17677, 18035, 18395, 18758, 19122, 19489, 19858,
        20228, 20601, 20975, 21351, 21729, 22108, 22489, 22872, 23256, 23641, 24028, 24417,
        24806, 25197, 25588, 25981, 26375, 26770, 27166, 27563, 27960, 28358, 28757, 29156,
        29556, 29957, 30357, 30759, 31160, 31562, 31964, 32366, 32768, 33170, 33572, 33974,
        34376, 34777, 35179, 35579, 35980, 36380, 36779, 37178, 37576, 37973, 38370, 38766,
        39161, 39555, 39948, 40339, 40730, 41119, 41508, 41895, 42280, 42664, 43047, 43428,
        43807, 44185, 44561, 44935, 45308, 45678, 46047, 46414, 46778, 47141, 47501, 47859,
        48215, 48568, 48919, 49268, 49614, 49958, 50299, 50637, 50973, 51306, 51636, 51963,
        52288, 52609, 52928, 53243, 53556, 53865, 54171, 54474, 54774, 55070, 55363, 55652,
        55938, 56221, 56500, 56776, 57047, 57316, 57580, 57841, 58098, 58351, 58601, 58846,
        59088, 59325, 59559, 59788, 60014, 60235, 60452, 60665, 60874, 61079, 61279, 61475,
        61667, 61854, 62037, 62216, 62390, 62560, 62725, 62886, 63042, 63193, 63340, 63483,
        63621, 63754, 63882, 64006, 64125, 64239, 64349, 64454, 64554, 64649, 64740, 64826,
        64906, 64982, 65054, 65120, 65181, 65238, 65290, 65336, 65378, 65415, 65447, 65474,
        65497, 65514, 65526, 65534, 65536,
    };
    return tbl;
}

// The QUADRATIC tables are EXACT integer (no transcendental) — built once at static-init via fxmul.
// t for bin i is i/kEaseBins in Q16.16 = (fx)((int64)i * kOne / kEaseBins) (NOT fxdiv — the numerator
// is a plain int, not Q16.16). For i==kEaseBins, t==kOne exactly.
inline std::vector<fx> BuildQuadInLut() {                 // EaseInQuad(t) = t*t
    std::vector<fx> tbl(kEaseBins + 1);
    for (int i = 0; i <= kEaseBins; ++i) {
        const fx t = (fx)((int64_t)i * (int64_t)kOne / (int64_t)kEaseBins);
        tbl[i] = fxmul(t, t);
    }
    return tbl;
}
inline std::vector<fx> BuildQuadOutLut() {                // EaseOutQuad(t) = kOne - (kOne-t)*(kOne-t)
    std::vector<fx> tbl(kEaseBins + 1);
    for (int i = 0; i <= kEaseBins; ++i) {
        const fx t = (fx)((int64_t)i * (int64_t)kOne / (int64_t)kEaseBins);
        tbl[i] = kOne - fxmul(kOne - t, kOne - t);
    }
    return tbl;
}
inline const std::vector<fx>& QuadInTable()  { static const std::vector<fx> t = BuildQuadInLut();  return t; }
inline const std::vector<fx>& QuadOutTable() { static const std::vector<fx> t = BuildQuadOutLut(); return t; }

// Ease(e, t01): the LUT read (FxSinLut mold). t01 assumed in [0,kOne]; clamps defensively. Step/Linear
// short-circuit (Step returns 0 — the lower-key weight; Linear is the identity). The eased modes do a
// scaled-index + fractional-lerp between adjacent table entries. Endpoints are EXACT (tables pin
// entry[0]=0, entry[kEaseBins]=kOne) so Ease(*,0)==0 and Ease(*,kOne)==kOne.
inline fx Ease(Easing e, fx t01) {
    if (t01 <= 0)    return 0;
    if (t01 >= kOne) return kOne;
    if (e == Easing::Step)   return 0;     // (SampleScalar already short-circuits Step before calling)
    if (e == Easing::Linear) return t01;   // identity
    const std::vector<fx>& tbl = (e == Easing::EaseInOutSine) ? SineEaseTable()
                               : (e == Easing::EaseInQuad)    ? QuadInTable()
                                                              : QuadOutTable();
    const int64_t scaled = (int64_t)t01 * kEaseBins;   // Q16.16 * int = Q16.16 scaled index.fraction
    const int     i      = (int)(scaled >> 16);        // bin index in [0, kEaseBins)
    const fx      frac   = (fx)(scaled & 0xFFFF);      // Q16.16 fraction within the bin
    return tbl[i] + fxmul(frac, tbl[i + 1] - tbl[i]);  // lerp between adjacent table entries
}

// FindSegment(tr, t): a HAND-WRITTEN integer binary search (NO <algorithm>) for the segment index k
// such that times[k] <= t < times[k+1]. Clamp: t <= times.front() -> 0; t >= times.back() ->
// times.size()-1 (the last key index; sampling there holds the last value). Pure integer compares.
// The empty-track case is handled by the caller (SampleScalar returns 0 before calling this).
inline std::size_t FindSegment(const ScalarTrack& tr, fx t) {
    const std::size_t n = tr.times.size();
    // Clamp to the endpoints (also covers n==1: lo==hi==0).
    if (t <= tr.times[0])     return 0;
    if (t >= tr.times[n - 1]) return n - 1;
    // Binary search for the greatest k with times[k] <= t (and, since t < times.back(), k < n-1).
    std::size_t lo = 0, hi = n - 1;          // answer is in [lo, hi]
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo + 1) / 2;   // ceil-mid so lo advances (no infinite loop)
        if (tr.times[mid] <= t) lo = mid;                  // times[mid] <= t -> k could be mid
        else                    hi = mid - 1;              // times[mid] > t  -> k < mid
    }
    return lo;
}

// SampleScalar(tr, t): the beachhead sample — the deterministic integer keyframe interpolation.
//   - empty times -> 0 (deterministic).
//   - clamp t to [times.front(), times.back()].
//   - k = FindSegment(tr, t). If k is the last key OR easing == Step -> return values[k] (hold).
//   - den = times[k+1] - times[k]; t01 = (den == 0) ? 0 : fxdiv(t - times[k], den)  (a Q16.16 in
//     [0, kOne]; the coincident-key guard — the strictly-ascending invariant means den > 0).
//   - return values[k] + fxmul(t01, values[k+1] - values[k])  — the load-bearing a + t*(b-a) lerp.
// S1 handles Step + Linear only; Linear uses t01 directly as the eased parameter. S2 inserts
// `Ease(easing, t01)` in place of the bare t01 below — a ONE-LINE change (the structure is kept so).
inline fx SampleScalar(const ScalarTrack& tr, fx t) {
    const std::size_t n = tr.times.size();
    if (n == 0) return 0;                                  // empty track -> deterministic 0
    // Clamp t to the keyed range so a sample before/after the track holds the end value.
    if (t < tr.times[0])     t = tr.times[0];
    if (t > tr.times[n - 1]) t = tr.times[n - 1];

    const std::size_t k = FindSegment(tr, t);
    if (k == n - 1 || tr.easing == Easing::Step) return tr.values[k];   // hold the key value

    const fx den = tr.times[k + 1] - tr.times[k];
    const fx t01 = (den == 0) ? 0 : fxdiv(t - tr.times[k], den);        // Q16.16 in [0, kOne]
    // The integer lerp a + w*(b-a). S2: w = Ease(tr.easing, t01) — a NO-OP for Linear
    // (Ease(Linear,t)==t, so the S1 digest is bit-identical), Step already short-circuited above.
    const fx w = Ease(tr.easing, t01);
    return tr.values[k] + fxmul(w, tr.values[k + 1] - tr.values[k]);
}

// SampleSweep(tr, dt, n): sample tr at n fixed ticks t = (fx)((int64)i * dt) for i in [0, n) -> a
// byte-stable std::vector<fx> (the digest input). The i*dt product is formed in int64 to avoid
// overflow at large tick counts, then cast back to fx.
inline std::vector<fx> SampleSweep(const ScalarTrack& tr, fx dt, uint32_t n) {
    std::vector<fx> out;
    out.reserve((std::size_t)n);
    for (uint32_t i = 0; i < n; ++i) {
        const fx t = (fx)((int64_t)i * (int64_t)dt);
        out.push_back(SampleScalar(tr, t));
    }
    return out;
}

// DigestTrack(sweep): the pinned-golden currency — FNV-1a-64 over the contiguous int32 sweep bytes.
// Byte-stable (a vector<int32> is contiguous), so two equal sweeps hash IDENTICALLY and a single
// changed sample changes the digest.
inline uint64_t DigestTrack(const std::vector<fx>& sweep) {
    return hf::net::DigestBytes(sweep.data(), sweep.size() * sizeof(fx));
}

// MakeShowcaseTrack(): the FIXED golden fixture — a 4-key track exercising a rising, a falling, and a
// through-zero segment. times = {0, 1, 2, 3} seconds, values = {0, 1, -0.5, 2}, Linear. Keep it
// FIXED forever — the pinned golden hashes its sweep.
inline ScalarTrack MakeShowcaseTrack() {
    ScalarTrack tr;
    tr.times  = {0, kOne, 2 * kOne, 3 * kOne};
    tr.values = {0, kOne, -kOne / 2, 2 * kOne};
    tr.easing = Easing::Linear;
    return tr;
}

// ============================ S2 — multi-track Sequence (the value-bus) ============================
// A timeline is N channels sampled at ONE time -> a value-bus. Each channel is an independent
// ScalarTrack; SampleSequence reads them all at time t into a contiguous fx bus (byte-stable digest
// input), and SampleSequenceSweep sweeps the whole bus over n ticks.
struct Sequence {
    std::vector<ScalarTrack> tracks;        // channel c = tracks[c]
};

// SampleSequence(seq, t): the value-bus at time t — one fx per channel, in channel order.
inline std::vector<fx> SampleSequence(const Sequence& seq, fx t) {
    std::vector<fx> bus;
    bus.reserve(seq.tracks.size());
    for (const auto& tr : seq.tracks) bus.push_back(SampleScalar(tr, t));
    return bus;
}

// SampleSequenceSweep(seq, dt, n): n ticks at t = (fx)((int64)i*dt); each tick appends the whole bus ->
// a contiguous (n * tracks) fx vector (the byte-stable digest input). The i*dt product is int64 to
// avoid overflow at large tick counts, then cast to fx (matching SampleSweep).
inline std::vector<fx> SampleSequenceSweep(const Sequence& seq, fx dt, uint32_t n) {
    std::vector<fx> out;
    out.reserve((std::size_t)n * seq.tracks.size());
    for (uint32_t i = 0; i < n; ++i) {
        const fx t = (fx)((int64_t)i * (int64_t)dt);
        for (const auto& tr : seq.tracks) out.push_back(SampleScalar(tr, t));
    }
    return out;
}

// MakeShowcaseSequence(): the FIXED golden fixture — 3 channels exercising the new easings. Channel 0
// is the S1 MakeShowcaseTrack() VERBATIM (Linear — anchors S1 invariance); channel 1 reuses its times
// with EaseInOutSine and values {0, kOne, 0, kOne}; channel 2 reuses its times with EaseInQuad and
// values {0, 2*kOne, kOne, 0}. Keep FIXED forever — the pinned goldens hash its sweep.
inline Sequence MakeShowcaseSequence() {
    Sequence seq;
    const ScalarTrack ch0 = MakeShowcaseTrack();    // S1 verbatim (Linear)

    ScalarTrack ch1;
    ch1.times  = ch0.times;
    ch1.values = {0, kOne, 0, kOne};
    ch1.easing = Easing::EaseInOutSine;

    ScalarTrack ch2;
    ch2.times  = ch0.times;
    ch2.values = {0, 2 * kOne, kOne, 0};
    ch2.easing = Easing::EaseInQuad;

    seq.tracks = {ch0, ch1, ch2};
    return seq;
}

}  // namespace hf::seq
