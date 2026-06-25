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
#include "flow/flow.h"    // S3: flow::EventRecord + flow::DigestEvents (the deterministic event-trace currency)

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

// ============================ S3 — event track (discrete timeline events) ==========================
// The OTHER half of a real timeline: discrete events that FIRE at exact tick boundaries as the playhead
// sweeps a time interval (a cutscene triggers a sound / a Blueprint pulse / a gameplay flag on frame N).
// The moat: the fired event SET is bit-identical cross-platform AND it composes with flow.h's
// deterministic-VM — a sequence event feeds a flow::Graph input channel and the flow trace is itself
// deterministic. UE5 Sequencer event tracks fire on FLOAT playback timing (two machines fire on
// different frames); ours fires on INTEGER ticks, identically, replayably. APPEND-ONLY — S1+S2 untouched.

// The event track: parallel-SoA (like ScalarTrack). `times` is STRICTLY ASCENDING (sorted, no dupes);
// eventIds[i] is the id fired at times[i]; payloads is OPTIONAL (size==times.size() OR empty -> all 0).
struct EventTrack {
    std::vector<fx>       times;     // Q16.16 seconds, STRICTLY ASCENDING (the invariant — sorted, no dupes)
    std::vector<uint32_t> eventIds;  // the event id fired at times[i]; eventIds.size()==times.size()
    std::vector<fx>       payloads;  // OPTIONAL Q16.16 payload per event; payloads.size()==times.size() OR 0
};

// SampleEvents(tr, tPrev, t): fire every event whose time is in the HALF-OPEN window [tPrev, t) — so a
// tick boundary fires EXACTLY ONCE across consecutive sweeps (the standard sampler-window convention).
// A hand integer scan in ascending index order (NO <algorithm>); each fired event maps to a
// flow::EventRecord. flow::EventRecord has exactly two fields: `uint32_t eventId` and `flow::Reg payload`
// (Reg is int32, == our fx) — NO tick/order field, so ordering is purely emission (ascending-time) order.
// We populate eventId = eventIds[i] and payload = (flow::Reg)(payloads.empty() ? 0 : payloads[i]).
// Guard: t <= tPrev -> empty (no negative/empty window fires).
inline std::vector<flow::EventRecord> SampleEvents(const EventTrack& tr, fx tPrev, fx t) {
    std::vector<flow::EventRecord> out;
    if (t <= tPrev) return out;                          // empty/negative window -> fires nothing
    const std::size_t n = tr.times.size();
    const bool hasPayload = !tr.payloads.empty();
    for (std::size_t i = 0; i < n; ++i) {                // ascending index scan (times sorted) = ascending time
        const fx ti = tr.times[i];
        if (ti < tPrev) continue;                        // before the window -> skip
        if (ti >= t)    break;                            // at/after the window end -> done (sorted, none remain)
        const fx pay = hasPayload ? tr.payloads[i] : (fx)0;
        out.push_back(flow::EventRecord{ tr.eventIds[i], (flow::Reg)pay });
    }
    return out;
}

// SampleEventSweep(tr, dt, n): drive the track across n fixed ticks -> ONE contiguous fired trace. For i
// in [0,n) the window is [i*dt, (i+1)*dt) (int64 products to avoid overflow), all fired EventRecords
// accumulated in order. Every event in [0, n*dt) fires exactly once, in ascending time order.
inline std::vector<flow::EventRecord> SampleEventSweep(const EventTrack& tr, fx dt, uint32_t n) {
    std::vector<flow::EventRecord> out;
    for (uint32_t i = 0; i < n; ++i) {
        const fx a = (fx)((int64_t)i * (int64_t)dt);          // window start  i*dt
        const fx b = (fx)((int64_t)(i + 1) * (int64_t)dt);    // window end   (i+1)*dt (half-open)
        const std::vector<flow::EventRecord> fired = SampleEvents(tr, a, b);
        for (const flow::EventRecord& e : fired) out.push_back(e);
    }
    return out;
}

// MakeShowcaseEvents(): the FIXED golden fixture — 5 events at fixed ticks with distinct ids + payloads.
// Note eventIds repeats 20 (proves the id is NOT a key — emission order is by time). Keep FIXED forever:
// the pinned golden hashes its fired trace.
inline EventTrack MakeShowcaseEvents() {
    EventTrack tr;
    tr.times    = {kOne / 2, kOne, 3 * kOne / 2, 2 * kOne, 5 * kOne / 2};  // 0.5s .. 2.5s
    tr.eventIds = {10, 20, 30, 20, 40};                                    // repeated 20 — id is not a key
    tr.payloads = {kOne, -kOne, kOne / 4, 2 * kOne, 0};
    return tr;
}

// ============================ S4 — transform / rotation track (THE Q16.16 ROTATION CRUX) ===========
// The HEADLINE: a transform track (translation + rotation + scale) sampled at a tick into an integer
// FxTransform — where the rotation is the float that breaks UE5 Sequencer (FQuat::Slerp via acos/sin,
// float playback timing → two machines diverge in the low bits) rebuilt as a DETERMINISTIC Q16.16
// integer NLERP: zero runtime transcendentals, bit-identical cross-platform. v1 is normalized-lerp
// (nlerp), NOT true constant-velocity slerp — the documented fidelity tradeoff (LUT-slerp via ik.h's
// FxAcosLut is the later upgrade; nlerp is correct, deterministic, and standard for cutscenes).
// APPEND-ONLY — S1–S3 types/semantics untouched + golden-invariant. NO new include: this reuses fpx.h's
// quaternion/vector substrate (already pulled in by "sim/fpx.h"). STILL NO <cmath>/<algorithm>/float.

// Reuse the fpx quaternion/vector substrate (read-only).
using hf::sim::fpx::FxVec3;          // {fx x,y,z}
using hf::sim::fpx::FxQuat;          // {fx x,y,z,w}  (w defaults to kOne = identity)
using hf::sim::fpx::FxQuatMul;       // Hamilton product (int64 fxmul terms) — pulled in for completeness
using hf::sim::fpx::FxQuatNormalize; // integer unit-normalize via FxISqrt (NO <cmath>)
using hf::sim::fpx::FxISqrt;         // integer floor-sqrt on int64 (for the ~unit length proof)

// FxTransform — the integer twin of a TRS pose (translation + unit-quat rotation + scale).
struct FxTransform {
    FxVec3 t;                            // translation (Q16.16 world units)
    FxQuat r = FxQuat{0, 0, 0, kOne};    // rotation (unit quaternion, identity default)
    FxVec3 s = FxVec3{kOne, kOne, kOne}; // scale (Q16.16, identity default)
};

// Vec3Track — three independent scalar tracks sampled into an FxVec3 (translation/scale channels).
struct Vec3Track { ScalarTrack x, y, z; };
inline FxVec3 SampleVec3(const Vec3Track& tr, fx t) {
    return FxVec3{ SampleScalar(tr.x, t), SampleScalar(tr.y, t), SampleScalar(tr.z, t) };
}

// RotationTrack — keyframe unit quaternions + integer nlerp (THE CRUX). The INVARIANT: `times` is
// STRICTLY ASCENDING (sorted, no dupes) and keys.size() == times.size().
struct RotationTrack {
    std::vector<fx>     times;   // Q16.16 seconds, STRICTLY ASCENDING
    std::vector<FxQuat> keys;    // unit quaternions; keys.size() == times.size()
};

// SampleRotation(tr, t): the deterministic integer quaternion sample.
//   - empty -> identity {0,0,0,kOne}. Single key -> keys[0].
//   - clamp t to [times.front(), times.back()]; at/past the last key -> keys.back().
//   - find segment k (the S1 FindSegment integer binary-search pattern, inline over tr.times — NO
//     <algorithm>); den = times[k+1]-times[k]; t01 = (den==0)?0:fxdiv(t-times[k], den) in [0,kOne].
//   - SHORTEST-ARC FLIP: int64 dot of qa,qb (Q32.32); if dot<0 negate every component of qb (the
//     double-cover convention — without it a 0->large-angle interp takes the long way around).
//   - INTEGER NLERP: m.c = qa.c + fxmul(t01, qb.c-qa.c) per component, then FxQuatNormalize(m). NO
//     transcendental, NO acos/sin — that is the determinism win. (nlerp is NOT constant angular
//     velocity; LUT-slerp via FxAcosLut is the future fidelity slice.)
inline FxQuat SampleRotation(const RotationTrack& tr, fx t) {
    const std::size_t n = tr.times.size();
    if (n == 0) return FxQuat{0, 0, 0, kOne};   // empty -> identity
    if (n == 1) return tr.keys[0];              // single key -> hold it

    // Clamp t to the keyed range.
    if (t < tr.times[0])     t = tr.times[0];
    if (t > tr.times[n - 1]) t = tr.times[n - 1];

    // FindSegment over tr.times (inline — the S1 hand-written integer binary search, NO <algorithm>).
    std::size_t k;
    if (t <= tr.times[0])          k = 0;
    else if (t >= tr.times[n - 1]) k = n - 1;
    else {
        std::size_t lo = 0, hi = n - 1;
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo + 1) / 2;   // ceil-mid so lo advances
            if (tr.times[mid] <= t) lo = mid;
            else                    hi = mid - 1;
        }
        k = lo;
    }
    if (k == n - 1) return tr.keys[k];          // at/past the last key -> hold it (normalize-free hold)

    const fx den = tr.times[k + 1] - tr.times[k];
    const fx t01 = (den == 0) ? 0 : fxdiv(t - tr.times[k], den);   // Q16.16 in [0, kOne]

    FxQuat qa = tr.keys[k];
    FxQuat qb = tr.keys[k + 1];

    // SHORTEST-ARC FLIP — int64 dot (Q32.32). If negative, negate qb (quaternions double-cover SO(3)).
    const int64_t dot = (int64_t)qa.x * (int64_t)qb.x + (int64_t)qa.y * (int64_t)qb.y
                      + (int64_t)qa.z * (int64_t)qb.z + (int64_t)qa.w * (int64_t)qb.w;
    if (dot < 0) { qb.x = -qb.x; qb.y = -qb.y; qb.z = -qb.z; qb.w = -qb.w; }

    // INTEGER NLERP — component-wise a + t01*(b-a), then integer normalize.
    FxQuat m{
        qa.x + fxmul(t01, qb.x - qa.x),
        qa.y + fxmul(t01, qb.y - qa.y),
        qa.z + fxmul(t01, qb.z - qa.z),
        qa.w + fxmul(t01, qb.w - qa.w),
    };
    return FxQuatNormalize(m);
}

// TransformTrack — the TRS bundle: translation Vec3Track + RotationTrack + scale Vec3Track.
struct TransformTrack { Vec3Track translation; RotationTrack rotation; Vec3Track scale; };
inline FxTransform SampleTransform(const TransformTrack& tr, fx t) {
    return FxTransform{ SampleVec3(tr.translation, t), SampleRotation(tr.rotation, t), SampleVec3(tr.scale, t) };
}

// SampleTransformSweep(tr, dt, n): sample tr at n fixed ticks t=(fx)((int64)i*dt) -> a contiguous fx
// buffer (10 fx per tick: t.xyz, r.xyzw, s.xyz). FxTransform is 10 fx but the STRUCT may carry padding —
// NEVER DigestBytes a vector of structs. Serialize FIELD-BY-FIELD (the replay.h / S1 discipline) so the
// digest input is contiguous + byte-stable. The i*dt product is int64 to avoid overflow.
inline std::vector<fx> SampleTransformSweep(const TransformTrack& tr, fx dt, uint32_t n) {
    std::vector<fx> out;
    out.reserve((std::size_t)n * 10u);
    for (uint32_t i = 0; i < n; ++i) {
        const fx t = (fx)((int64_t)i * (int64_t)dt);
        const FxTransform x = SampleTransform(tr, t);
        out.push_back(x.t.x); out.push_back(x.t.y); out.push_back(x.t.z);
        out.push_back(x.r.x); out.push_back(x.r.y); out.push_back(x.r.z); out.push_back(x.r.w);
        out.push_back(x.s.x); out.push_back(x.s.y); out.push_back(x.s.z);
    }
    return out;   // contiguous fx -> DigestTrack() (net::DigestBytes) is padding-safe + byte-stable
}

// MakeShowcaseRotation(): the FIXED golden fixture — 3 unit quaternions about the +Y axis at 0°, 90°,
// 180° as COMMITTED Q16.16 literals (the 180° crossing exercises the shortest-arc flip — the dot of
// q(90°) and q(180°) about Y stays positive, but a direct q(0°)->q(180°) interp is where the
// double-cover flip matters; the S4 shortest-arc test builds that pair explicitly). The literals were
// baked ONCE with a throwaway program computing {y=sin(θ/2), w=cos(θ/2)} * 65536, snapped to Q16.16,
// pasted here, and the generator DELETED — seq.h stays <cmath>-free:
//   0°   -> {0,     0, 0, 65536}     (cos0=1,             sin0=0)
//   90°  -> {0, 46341, 0, 46341}     (cos45=sin45=0.70710678... * 65536 = 46340.95 -> 46341)
//   180° -> {0, 65536, 0,     0}     (cos90=0,            sin90=1)
// Each is ≈ unit (FxQuatNormalize-stable: |{0,46341,0,46341}|=65536±1). Keep FIXED forever.
inline RotationTrack MakeShowcaseRotation() {
    RotationTrack tr;
    tr.times = {0, kOne, 2 * kOne};                         // 0s, 1s, 2s
    tr.keys  = { FxQuat{0,     0, 0, kOne},                 // 0°   about +Y (identity)
                 FxQuat{0, 46341, 0, 46341},                // 90°  about +Y
                 FxQuat{0, kOne,  0, 0} };                  // 180° about +Y  (kOne == 65536)
    return tr;
}

// MakeShowcaseTransform(): the FIXED golden fixture — translation = a Vec3Track (distinct simple ramps
// per axis), rotation = MakeShowcaseRotation(), scale = a gentle Vec3Track (constant unit on x/z, a small
// ramp on y). Keep FIXED forever — the pinned golden hashes its transform sweep.
inline TransformTrack MakeShowcaseTransform() {
    TransformTrack tr;

    // Translation: x rises 0->2, y is a small dip-and-rise, z falls 0->-1, all over [0,2]s (Linear).
    tr.translation.x.times  = {0, kOne, 2 * kOne};
    tr.translation.x.values = {0, kOne, 2 * kOne};
    tr.translation.x.easing = Easing::Linear;
    tr.translation.y.times  = {0, kOne, 2 * kOne};
    tr.translation.y.values = {0, kOne / 2, 0};
    tr.translation.y.easing = Easing::Linear;
    tr.translation.z.times  = {0, kOne, 2 * kOne};
    tr.translation.z.values = {0, -kOne / 2, -kOne};
    tr.translation.z.easing = Easing::Linear;

    tr.rotation = MakeShowcaseRotation();

    // Scale: gentle — x/z held at unit, y a small ramp 1.0 -> 1.5 -> 1.0 (a subtle pulse).
    tr.scale.x.times  = {0, 2 * kOne};
    tr.scale.x.values = {kOne, kOne};
    tr.scale.x.easing = Easing::Linear;
    tr.scale.y.times  = {0, kOne, 2 * kOne};
    tr.scale.y.values = {kOne, kOne + kOne / 2, kOne};
    tr.scale.y.easing = Easing::Linear;
    tr.scale.z.times  = {0, 2 * kOne};
    tr.scale.z.values = {kOne, kOne};
    tr.scale.z.easing = Easing::Linear;

    return tr;
}

// ============================ S5 — lockstep / replay / SCRUB via net::Session (THE MOAT) =============
// THE HEADLINE: wrap timeline evaluation as a net::Session StepFn so the whole sequence becomes
// lockstep-replayable, desync-detectable, and SCRUB-able — and prove the moat property UE5 Sequencer's
// float playback timing CANNOT do: seeking to tick S then playing forward is BIT-IDENTICAL to playing
// from tick 0 (a deterministic timeline scrub). The playhead is a FLAT, value-copyable World
// (snapshot-complete by construction — net::Session's value-copy captures the WHOLE state); the step is
// integer add + SampleSequence; the digest is hand-LE fx (NEVER DigestBytes the struct — the bus vector
// has a heap pointer). APPEND-ONLY — S1–S4 untouched + golden-invariant. NO new include: net/session.h
// is already pulled in (S1). STILL NO <cmath>/<algorithm>/float/<random>/clock/std::hash.

// SeqPlayhead — the net::Session World: a FLAT, value-copyable struct. `time` is the TRUE state (the
// current Q16.16 timeline position); `bus` is DERIVED from `time` each step (the readable outputs sampled
// at `time`). Flat + value-copyable → net::Session's value-copy snapshot captures the WHOLE state by
// construction (the verdict.h completeness argument).
struct SeqPlayhead {
    fx              time = 0;   // current Q16.16 timeline position (the TRUE state)
    std::vector<fx> bus;        // the value-bus sampled at `time` (derived outputs)
};

// StepPlayhead — the deterministic transition net::Session drives. Advance the playhead by the SUM of
// this tick's inputs (delta in Q16.16 seconds), then resample the bus. `inputs` is net::Session's per-tick
// input vector: {delta} in plain lockstep, {local, remote} in rollback (StepPredicted/ConfirmRemote pass a
// TWO-element vector) — so we SUM in fixed order (one elem lockstep, two rollback).
inline void StepPlayhead(const Sequence& seq, SeqPlayhead& w, const std::vector<fx>& inputs, uint32_t /*tick*/) {
    fx delta = 0;
    for (fx d : inputs) delta += d;          // sum (1 elem lockstep, 2 elems rollback) — fixed order
    w.time += delta;
    w.bus = SampleSequence(seq, w.time);     // resample the multi-track value-bus at the new time
}

// DigestPlayhead — hand-LE digest of the playhead: serialize time (1 fx) then the bus (N fx) into a
// contiguous fx buffer -> DigestTrack (S1's net::DigestBytes over the int32 bytes). Padding-safe +
// byte-stable. NEVER DigestBytes a SeqPlayhead — the `bus` vector has a heap pointer + size; serialize the
// VALUES (the S4 field-by-field discipline).
inline uint64_t DigestPlayhead(const SeqPlayhead& w) {
    std::vector<fx> buf;
    buf.reserve(w.bus.size() + 1u);
    buf.push_back(w.time);
    for (fx v : w.bus) buf.push_back(v);
    return DigestTrack(buf);                 // == net::DigestBytes(buf.data(), buf.size()*sizeof(fx))
}

// MakeShowcasePlayheadSeq() — the FIXED S5 fixture: reuse MakeShowcaseSequence() (the S2 3-channel
// sequence, mixed easings) so the bus is non-trivial. Keep FIXED forever — the S5 goldens hash its playback.
inline Sequence MakeShowcasePlayheadSeq() {
    return MakeShowcaseSequence();
}

}  // namespace hf::seq
