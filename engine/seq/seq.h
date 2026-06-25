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
// these two (the golden + any serialized track pin the enum values).
enum class Easing : uint32_t { Step = 0, Linear = 1 };

// A scalar keyframe track: parallel (times, values) arrays in Q16.16. The INVARIANT: `times` is
// STRICTLY ASCENDING (sorted keys, no duplicates) and values.size() == times.size().
struct ScalarTrack {
    std::vector<fx> times;                  // Q16.16 seconds, STRICTLY ASCENDING
    std::vector<fx> values;                 // Q16.16 keyframe values; values.size() == times.size()
    Easing          easing = Easing::Linear;
};

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
    // The integer lerp a + t*(b-a). (S2: replace the bare t01 here with Ease(tr.easing, t01).)
    return tr.values[k] + fxmul(t01, tr.values[k + 1] - tr.values[k]);
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

}  // namespace hf::seq
