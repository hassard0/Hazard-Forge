#pragma once
// Slice AL1 — DETERMINISTIC ANIMATION LAYERING (clip notifies/events + additive poses + layered slot
// blending + montages; hf::anim::layer). The next-tier parity gap on the animation pillar: the anim stack
// ships clips + sampling (animation.h), the FSM (state_machine.h), blend spaces (AN1), retargeting (AN2),
// IK (ik.h) and motion matching (MM1) — but NO LAYERING tier. UE5 authors this as Anim Notifies, Additive
// / Layered Blend-Per-Bone anim, and Montages (the "attack combo with hit windows" + "wave on the upper
// body while walking" primitives). AL1 builds all four deterministically, COMPOSING the existing anim
// primitives read-only. Namespace hf::anim::layer, header-only, PURE CPU, NO device/backend symbols, NO
// RNG, NO clock, NO new shader. animation.h / retarget.h / skeleton.h / seq/seq.h / game/gameplay_tags.h
// are #included READ-ONLY (byte-untouched); any new helper lives HERE.
//
// THE INTEGER DOMAIN (the AN2 discipline, reused wholesale): AL1 operates on retarget::FxJointPose — a
// Q16.16 LOCAL pose (integer unit-quat rotation + integer translation). The ONE float boundary is the
// clip sample: animation.h SampleLocalPose (float) -> retarget::QuantizePose (llround, the MM1/AN2
// QuantizeFx). EVERYTHING downstream — the additive quat compose, the identity-nlerp, the layered
// per-bone blend, the montage blend curve, the forward-kinematics for the viz, and every digest — is
// INTEGER (int32 state, int64 intermediates), so two peers / two compilers agree bit-for-bit BY
// CONSTRUCTION (the AN2 guarantee). Fixture clips use exact binary-fraction keys sampled AT keyframe
// times (translation swings + exact-unit (+-1/2,+-1/2,+-1/2,+-1/2)-family rotations), so the QuantizeFx
// inputs are themselves cross-compiler exact. Rotation NLERP uses a pure-integer normalize (FxISqrt),
// which is deterministic across compilers (integer sqrt, no transcendental).
//
// (1) NOTIFIES (the seq.h event-track sibling). A NotifyTrack is notifies {tick, notifyId, payload}
// attached to a clip/montage. SampleNotifies fires every notify whose tick lies in the HALF-OPEN window
// (prevTick, curTick] — EXCLUSIVE-low / INCLUSIVE-high. This is the MIRROR of seq::SampleEvents' [tPrev,
// t) (inclusive-low / exclusive-high): seq fires at the window START, anim notifies fire at the window
// END, so "the tick you ADVANCE ONTO fires" — the intuitive gameplay-frame semantics (a notify AT
// prevTick does NOT re-fire; it fired on the step that landed there; a notify AT curTick fires now). A
// tick-by-tick advance (prev,cur]=(t-1,t] fires each notify EXACTLY ONCE as the playhead passes it. The
// track is LOOP-AWARE (loopLen > 0): a notify at local n has absolute occurrences n, n+L, n+2L, ...; it
// fires once per occurrence inside the window (loop wrap fires correctly). Firing order is ascending
// (absoluteTick, notifyIndex) — deterministic. A notify can carry a gameplay-event id (see the bridge).
//
// (2) ADDITIVE POSES. An additive animation is a clip stored as a DELTA from a reference pose. ApplyAdditive
// composes it onto a base pose per bone at weight w (Q16.16 in [0, kOne]):
//     result.r = base.r (x) NlerpQuat(identity, delta.r, w)     [THE PINNED COMPOSE ORDER — base on the LEFT]
//     result.t = base.t + w * delta.t                            [translations add, scaled]
// NlerpQuat(identity, delta.r, w) is the fixed-point identity-slerp (nlerp with the shortest-arc flip +
// integer normalize; the seq.h S4 nlerp discipline). Hamilton product a (x) b == "apply b, then a", so
// base.r (x) weightedDelta applies the additive delta in the base bone's local frame — the standard
// LOCAL-SPACE additive (aim-offset / breathing / lean). HONESTY: nlerp is NON-associative and NOT
// constant-velocity slerp; the pinned ORDER (base-left, delta-right) is the determinism contract, not an
// idealized blend. w <= 0 SHORT-CIRCUITS to base (bit-exact); w >= kOne composes base (x) delta directly
// (no normalize rounding at the endpoints). Deviation is ZERO for an identity delta (any w) and grows with
// the delta's angle.
//
// (3) LAYERED SLOT BLENDING (layered-blend-per-bone). An AnimLayerStack is a base pose + ordered layers,
// each {slot pose, weight, optional per-bone MASK}. EvaluateLayers folds the layers over the base IN
// ARRAY ORDER; for layer L, bone j, the effective weight is fxmul(L.weight, mask[j]) (mask empty => all
// kOne), and the bone is BlendBone(current, L.slot[j], eff) — nlerp the rotation (shortest-arc), lerp the
// translation. The per-bone mask is a scalar WEIGHT per bone (NOT full IK): an upper-body mask (1 on the
// arm/spine bones, 0 on the legs) makes the layer affect ONLY the masked bones — "play a wave on the
// upper body while the legs keep walking". eff <= 0 keeps the current bone (a weight-0 layer / all-zero
// mask == base, bit-exact); eff >= kOne takes the slot bone. The fold ORDER is the determinism contract
// (order matters — the last layer wins where masks overlap).
//
// (4) MONTAGES (the "attack combo with hit windows"). A Montage is a list of sections {clip, lengthTicks,
// blendInTicks, blendOutTicks, local NotifyTrack} laid out as a CONCATENATED tick timeline. A MontagePlayer
// holds one global montage tick. StepMontage advances the tick and fires every section's notifies whose
// GLOBAL tick (sectionStart + local) lands in (prevTick, curTick]. EvaluateMontage samples the active
// section's clip (the ONE float boundary) and computes the blend WEIGHT = min(rampIn, rampOut): a linear
// ramp 0->kOne over the first blendInTicks and kOne->0 over the last blendOutTicks (LINEAR, documented —
// eased montage blends are a future refinement). JumpToSection sets the global tick to a section's start
// (the deterministic combo transition). A montage plays as a SLOT LAYER: (montage slot pose, weight) feeds
// EvaluateLayers (#3), so it composes with the base + additive. A single section with no blend == the raw
// clip (weight kOne everywhere).
//
// (5) THE GAMEPLAY BRIDGE (notify -> GT1/GAS1 tagged effect). A notify carrying a gameplay-event id routes,
// through a THIN ADAPTER (NOT deep GAS integration — the caller maps notifyId -> a tags::TagCommand and
// runs it BEFORE the GAS step, the commands-before-step contract), to tags::TryActivateTagged, applying a
// tagged effect. RunBridgeScenario proves the notify->effect timing is EXACT: an attack montage's "hit"
// notify at tick T issues a damage ability, and the target's health drops on exactly the step for tick T.
//
// (6) IDENTITY (all pinned): additive w==0 == base bit-exact; an identity-delta additive == base (any w);
// a layer with weight 0 or an all-zero mask == base; a montage single section with no blend == the raw
// clip; a notify at an exact tick fires exactly once.
//
// REUSE MAP: retarget.h FxQuat/FxQuatMul/FxQuatConj/FxQuatIdentity/FxQuatRotate (the integer quat algebra),
// FxJointPose/FxJointModel, QuantizePose/BindPose (the float boundary), ForwardKinematics (the FK for the
// viz), DigestFxPose/DigestModel (the digest currency) — ALL read-only. animation.h SampleLocalPose (the
// clip sample) read-only. seq/seq.h SampleEvents (the half-open convention cited, not reinvented; notifies
// are the integer-tick sibling). game/gameplay_tags.h TryActivateTagged/StepTagged (the bridge target)
// read-only. mm:: fx/kOne/kFrac/QuantizeFx/fxmul/FxV3 + Fnv1a64Word (the anim family's Q16.16 currency).
// NO new RHI, NO shader, NO device.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "anim/animation.h"        // READ-ONLY: JointPose / SampleLocalPose (the clip sample)
#include "anim/retarget.h"         // READ-ONLY: FxQuat algebra + FxJointPose/FxJointModel + FK + QuantizePose + digests
#include "anim/skeleton.h"         // READ-ONLY: Skeleton / Joint
#include "seq/seq.h"               // READ-ONLY: SampleEvents (the half-open convention this mirrors — cited)
#include "game/gameplay_tags.h"    // READ-ONLY: TryActivateTagged / StepTagged (the notify->tagged-effect bridge)

namespace hf::anim {
namespace layer {

using mm::fx;
using mm::FxV3;
using mm::fxmul;
using mm::kFrac;
using mm::kOne;
using mm::QuantizeFx;                 // THE float->integer boundary (llround — the MM1/AN2 discipline)

using retarget::FxJointModel;
using retarget::FxJointPose;
using retarget::FxQuat;
using retarget::FxQuatConj;
using retarget::FxQuatIdentity;
using retarget::FxQuatMul;

// ===================== Integer quaternion nlerp (pure-integer normalize, no transcendental) ===========

// FxISqrt: floor integer square root of a non-negative int64 (Newton iteration — deterministic, no
// <cmath>). Used by the quaternion normalize below; converges in a few steps for our < 2^40 inputs.
inline int64_t FxISqrt(int64_t v) {
    if (v <= 0) return 0;
    int64_t x = v, y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + v / x) / 2; }
    return x;
}

// FxQuatNormalize: re-unitize a Q16.16 quaternion via the integer sqrt. |q|^2 is accumulated in Q16.16
// (each term (q.i^2 >> kFrac)); |q| in Q16.16 == FxISqrt(|q|^2 * kOne); each component is (q.i * kOne)/|q|.
// The identity {0,0,0,kOne} normalizes to itself EXACTLY. Degenerate (zero) -> identity.
inline FxQuat FxQuatNormalize(const FxQuat& q) {
    int64_t s = 0;
    s += ((int64_t)q.x * (int64_t)q.x) >> kFrac;
    s += ((int64_t)q.y * (int64_t)q.y) >> kFrac;
    s += ((int64_t)q.z * (int64_t)q.z) >> kFrac;
    s += ((int64_t)q.w * (int64_t)q.w) >> kFrac;
    if (s <= 0) return FxQuatIdentity();
    const int64_t len = FxISqrt(s * (int64_t)kOne);       // |q| in Q16.16
    if (len <= 0) return FxQuatIdentity();
    FxQuat r;
    r.x = (fx)(((int64_t)q.x * (int64_t)kOne) / len);
    r.y = (fx)(((int64_t)q.y * (int64_t)kOne) / len);
    r.z = (fx)(((int64_t)q.z * (int64_t)kOne) / len);
    r.w = (fx)(((int64_t)q.w * (int64_t)kOne) / len);
    return r;
}

// NlerpQuat(a, b, w): the deterministic integer nlerp from a to b at Q16.16 weight w in [0, kOne]. The
// seq.h S4 discipline: shortest-arc flip (int64 dot; if < 0 negate b — quaternions double-cover SO(3)),
// component-wise a + w*(b-a), then FxQuatNormalize. w <= 0 returns `a` and w >= kOne returns `b` DIRECTLY
// (no normalize rounding at the endpoints — the identity-at-endpoint contract). nlerp is NOT constant-
// angular-velocity slerp; it is deterministic + standard for layering.
inline FxQuat NlerpQuat(const FxQuat& a, FxQuat b, fx w) {
    if (w <= 0)    return a;
    if (w >= kOne) return b;
    const int64_t dot = (int64_t)a.x * b.x + (int64_t)a.y * b.y + (int64_t)a.z * b.z + (int64_t)a.w * b.w;
    if (dot < 0) { b.x = -b.x; b.y = -b.y; b.z = -b.z; b.w = -b.w; }
    FxQuat m{
        a.x + fxmul(w, b.x - a.x),
        a.y + fxmul(w, b.y - a.y),
        a.z + fxmul(w, b.z - a.z),
        a.w + fxmul(w, b.w - a.w),
    };
    return FxQuatNormalize(m);
}

// ===================== (1) Notifies (the seq.h event-track sibling, integer ticks) ====================

// One notify: a local tick offset, a notify id (a gameplay-event id for the bridge), and an optional
// Q16.16 payload.
struct Notify {
    int32_t  tick     = 0;   // local tick offset within the clip/section
    uint32_t notifyId = 0;   // the fired id (== a gameplay-event id for the bridge)
    fx       payload  = 0;   // optional Q16.16 payload
};

// A notify track attached to a clip/section. loopLen > 0 makes the track LOOP-AWARE (notify local ticks
// are in [0, loopLen); occurrences repeat every loopLen); loopLen <= 0 is a one-shot track.
struct NotifyTrack {
    std::vector<Notify> notifies;
    int32_t             loopLen = 0;   // <= 0 => non-looping
};

// One fired notify: the ABSOLUTE tick it fired at (post loop-unwrap), the id, and the payload.
struct FiredNotify {
    int64_t  tick     = 0;
    uint32_t notifyId = 0;
    fx       payload  = 0;
};

namespace detail {
// FloorDiv: floor of a/b for b > 0 (handles negative a — the loop-unwrap arithmetic).
inline int64_t FloorDiv(int64_t a, int64_t b) {
    return (a >= 0) ? (a / b) : -(((-a) + b - 1) / b);
}
}  // namespace detail

// SampleNotifies(track, prevTick, curTick): fire every notify whose (loop-unwrapped) absolute tick is in
// the HALF-OPEN window (prevTick, curTick] (EXCLUSIVE-low / INCLUSIVE-high — the mirror of seq::SampleEvents
// [tPrev, t); see the banner). Empty/negative window (curTick <= prevTick) fires nothing. Fired in
// ascending (absoluteTick, notifyIndex) order — deterministic.
inline std::vector<FiredNotify> SampleNotifies(const NotifyTrack& tr, int64_t prevTick, int64_t curTick) {
    std::vector<FiredNotify> out;
    if (curTick <= prevTick) return out;                                  // empty/negative window
    // Collect (absoluteTick, notifyIndex) then sort — general (a big window may cross several loops).
    std::vector<std::pair<int64_t, int32_t>> fires;
    const int32_t L = tr.loopLen;
    for (size_t i = 0; i < tr.notifies.size(); ++i) {
        const int64_t n = (int64_t)tr.notifies[i].tick;
        if (L <= 0) {
            if (prevTick < n && n <= curTick) fires.emplace_back(n, (int32_t)i);
        } else {
            // absolute occurrences n + k*L, k >= 0, inside (prevTick, curTick].
            int64_t k = detail::FloorDiv(prevTick - n, L) + 1;            // first occurrence strictly > prev
            if (k < 0) k = 0;
            for (;; ++k) {
                const int64_t a = n + k * (int64_t)L;
                if (a > curTick) break;
                fires.emplace_back(a, (int32_t)i);
            }
        }
    }
    std::sort(fires.begin(), fires.end(), [](const std::pair<int64_t, int32_t>& p,
                                             const std::pair<int64_t, int32_t>& q) {
        return (p.first != q.first) ? (p.first < q.first) : (p.second < q.second);
    });
    out.reserve(fires.size());
    for (const auto& f : fires) {
        const Notify& nf = tr.notifies[(size_t)f.second];
        out.push_back(FiredNotify{f.first, nf.notifyId, nf.payload});
    }
    return out;
}

// ===================== (2) Additive poses ============================================================

// ApplyAdditiveBone: compose one additive DELTA bone onto a base bone at weight w (the pinned order —
// base on the LEFT). w <= 0 returns base bit-exact.
inline FxJointPose ApplyAdditiveBone(const FxJointPose& base, const FxJointPose& delta, fx w) {
    if (w <= 0) return base;
    FxJointPose out;
    const FxQuat wd = NlerpQuat(FxQuatIdentity(), delta.r, w);   // identity-slerp toward the delta
    out.r = FxQuatMul(base.r, wd);                               // base (x) weightedDelta (LOCAL-SPACE additive)
    out.t = FxV3{base.t.x + fxmul(w, delta.t.x),
                 base.t.y + fxmul(w, delta.t.y),
                 base.t.z + fxmul(w, delta.t.z)};
    return out;
}

// ApplyAdditive: compose a whole additive DELTA pose onto a base pose at weight w. Bones past the shorter
// length pass through unchanged.
inline std::vector<FxJointPose> ApplyAdditive(const std::vector<FxJointPose>& base,
                                              const std::vector<FxJointPose>& delta, fx w) {
    std::vector<FxJointPose> out = base;
    const size_t n = (base.size() < delta.size()) ? base.size() : delta.size();
    for (size_t j = 0; j < n; ++j) out[j] = ApplyAdditiveBone(base[j], delta[j], w);
    return out;
}

// ===================== (3) Layered slot blending (layered-blend-per-bone) =============================

// BlendBone(a, b, w): per-bone blend of two fixed-point bones at weight w — nlerp the rotation (shortest-
// arc), lerp the translation. w <= 0 returns a bit-exact; w >= kOne returns b bit-exact.
inline FxJointPose BlendBone(const FxJointPose& a, const FxJointPose& b, fx w) {
    if (w <= 0) return a;
    if (w >= kOne) return b;
    FxJointPose out;
    out.r = NlerpQuat(a.r, b.r, w);
    out.t = FxV3{a.t.x + fxmul(w, b.t.x - a.t.x),
                 a.t.y + fxmul(w, b.t.y - a.t.y),
                 a.t.z + fxmul(w, b.t.z - a.t.z)};
    return out;
}

// One layer: a slot pose, a scalar weight, and an OPTIONAL per-bone mask (per-bone Q16.16 weight in
// [0, kOne]; empty => all kOne — the whole skeleton). The effective per-bone weight is
// fxmul(weight, mask[j]).
struct Layer {
    std::vector<FxJointPose> pose;            // the slot pose
    fx                       weight = kOne;   // the layer weight
    std::vector<fx>          mask;            // per-bone weight; empty => all kOne
};

// The layer stack: a base pose + ordered layers folded over it in array order.
struct AnimLayerStack {
    std::vector<FxJointPose> base;
    std::vector<Layer>       layers;
};

// EvaluateLayers: fold the layers over the base IN ARRAY ORDER (the determinism contract). For each layer,
// each bone blends by fxmul(layer.weight, mask[j]) (mask empty => kOne). A weight-0 layer / all-zero mask
// leaves the base unchanged (bit-exact).
inline std::vector<FxJointPose> EvaluateLayers(const AnimLayerStack& stack) {
    std::vector<FxJointPose> cur = stack.base;
    for (const Layer& L : stack.layers) {
        const size_t n = (cur.size() < L.pose.size()) ? cur.size() : L.pose.size();
        for (size_t j = 0; j < n; ++j) {
            const fx m = (L.mask.empty()) ? kOne : (j < L.mask.size() ? L.mask[j] : 0);
            const fx eff = fxmul(L.weight, m);
            cur[j] = BlendBone(cur[j], L.pose[j], eff);
        }
    }
    return cur;
}

// ===================== (4) Montages ==================================================================

// One montage section: a clip index (into a caller vector<Animation>), its length in ticks, the blend-in
// / blend-out ramp lengths, and a section-LOCAL notify track.
struct MontageSection {
    int32_t     clip          = 0;
    int32_t     lengthTicks   = 1;
    int32_t     blendInTicks  = 0;
    int32_t     blendOutTicks = 0;
    NotifyTrack notifies;                 // section-local ticks (loopLen typically 0)
};

// A montage: an ordered list of sections laid out as a CONCATENATED tick timeline + the clip-time mapping
// (localTick -> clip seconds = localTick * secondsPerTick; the fixture discipline keeps samples on
// keyframes for cross-compiler exactness).
struct Montage {
    std::vector<MontageSection> sections;
    float                       secondsPerTick = 1.0f;
};

// The montage player: one global tick over the concatenated section timeline. `done` latches when the tick
// reaches the total length.
struct MontagePlayer {
    int32_t tick = 0;
    bool    done = false;
};

// MontageTotalTicks: the summed section lengths (the concatenated timeline length).
inline int32_t MontageTotalTicks(const Montage& m) {
    int32_t total = 0;
    for (const MontageSection& s : m.sections) total += (s.lengthTicks > 0 ? s.lengthTicks : 0);
    return total;
}

// SectionStart: the global tick a section begins at (sum of prior lengths).
inline int32_t SectionStart(const Montage& m, int32_t idx) {
    int32_t start = 0;
    for (int32_t i = 0; i < idx && (size_t)i < m.sections.size(); ++i)
        start += (m.sections[(size_t)i].lengthTicks > 0 ? m.sections[(size_t)i].lengthTicks : 0);
    return start;
}

// LocateSection: map a global tick to (sectionIndex, localTick). A tick past the end clamps to the last
// section's final local tick; an empty montage -> {-1, 0}.
inline void LocateSection(const Montage& m, int32_t tick, int32_t& outSection, int32_t& outLocal) {
    outSection = -1; outLocal = 0;
    if (m.sections.empty()) return;
    if (tick < 0) tick = 0;
    int32_t start = 0;
    for (size_t i = 0; i < m.sections.size(); ++i) {
        const int32_t len = (m.sections[i].lengthTicks > 0 ? m.sections[i].lengthTicks : 0);
        if (tick < start + len) { outSection = (int32_t)i; outLocal = tick - start; return; }
        start += len;
    }
    outSection = (int32_t)m.sections.size() - 1;                        // clamp to the last section
    const int32_t lastLen = (m.sections.back().lengthTicks > 0 ? m.sections.back().lengthTicks : 1);
    outLocal = lastLen - 1;
}

// MontageBlendWeight: the LINEAR ramp for a section at local tick — min(rampIn, rampOut). rampIn is
// 0 -> kOne over the first blendInTicks; rampOut is kOne -> 0 over the last blendOutTicks; kOne in the
// middle. blendIn/out == 0 => that side is instantly full (a no-blend section is kOne everywhere -> the
// raw clip).
inline fx MontageBlendWeight(const MontageSection& s, int32_t local) {
    if (local < 0) local = 0;
    const int32_t len = (s.lengthTicks > 0 ? s.lengthTicks : 1);
    if (local >= len) local = len - 1;
    fx wIn = kOne;
    if (s.blendInTicks > 0 && local < s.blendInTicks)
        wIn = (fx)(((int64_t)local * (int64_t)kOne) / (int64_t)s.blendInTicks);
    fx wOut = kOne;
    if (s.blendOutTicks > 0) {
        const int32_t fromEnd = len - 1 - local;                       // ticks remaining to the end
        if (fromEnd < s.blendOutTicks)
            wOut = (fx)(((int64_t)fromEnd * (int64_t)kOne) / (int64_t)s.blendOutTicks);
    }
    return (wIn < wOut) ? wIn : wOut;
}

// EvaluateMontage: sample the active section's clip (the ONE float boundary -> QuantizePose) and compute
// the blend weight. `clips` is the caller's animation pool; `sk` the target skeleton. Out of range / empty
// -> the skeleton bind pose at weight 0.
inline void EvaluateMontage(const Montage& m, const MontagePlayer& p, const std::vector<Animation>& clips,
                            const Skeleton& sk, std::vector<FxJointPose>& outPose, fx& outWeight) {
    int32_t si = -1, local = 0;
    LocateSection(m, p.tick, si, local);
    if (si < 0) { outPose = retarget::BindPose(sk); outWeight = 0; return; }
    const MontageSection& s = m.sections[(size_t)si];
    if (s.clip < 0 || (size_t)s.clip >= clips.size()) { outPose = retarget::BindPose(sk); outWeight = 0; return; }
    const float t = (float)local * m.secondsPerTick;
    const std::vector<JointPose> fp = SampleLocalPose(sk, clips[(size_t)s.clip], t);
    outPose = retarget::QuantizePose(fp);
    outWeight = MontageBlendWeight(s, local);
}

// StepMontage: advance the player by `dt` ticks and return every section's notifies whose GLOBAL tick
// (SectionStart + local) lands in the half-open window (prevTick, curTick]. Latches `done` at the end.
inline std::vector<FiredNotify> StepMontage(const Montage& m, MontagePlayer& p, int32_t dt) {
    const int64_t prev = (int64_t)p.tick;
    int64_t cur = prev + (dt > 0 ? dt : 0);
    const int32_t total = MontageTotalTicks(m);
    if (total > 0 && cur >= total) { cur = total; p.done = true; }
    p.tick = (int32_t)cur;
    std::vector<FiredNotify> out;
    for (size_t i = 0; i < m.sections.size(); ++i) {
        const int32_t start = SectionStart(m, (int32_t)i);
        const std::vector<FiredNotify> local = SampleNotifies(m.sections[i].notifies, prev - start, cur - start);
        for (const FiredNotify& f : local)
            out.push_back(FiredNotify{f.tick + start, f.notifyId, f.payload});   // section-local -> global
    }
    std::sort(out.begin(), out.end(), [](const FiredNotify& a, const FiredNotify& b) {
        return (a.tick != b.tick) ? (a.tick < b.tick) : (a.notifyId < b.notifyId);
    });
    return out;
}

// JumpToSection: land the player at a section's start tick (the deterministic combo transition). Returns
// the landing global tick.
inline int32_t JumpToSection(const Montage& m, MontagePlayer& p, int32_t idx) {
    if (idx < 0 || (size_t)idx >= m.sections.size()) return p.tick;
    p.tick = SectionStart(m, idx);
    p.done = false;
    return p.tick;
}

// ===================== Digests (reuse the AN2 currency) ==============================================

using retarget::DigestFxPose;         // FNV over a fixed-point local pose (t.xyz + r.xyzw)
using retarget::DigestModel;          // FNV over a model-space FK result (pos.xyz)
using retarget::FxPoseEqual;          // field-exact pose equality (the identity proofs)

// DigestFired: fold a fired-notify list (tick + id + payload) into an FNV digest (the notify-trace currency).
inline uint64_t DigestFired(const std::vector<FiredNotify>& fired) {
    uint64_t h = 14695981039346656037ull;
    for (const FiredNotify& f : fired) {
        h = mm::detail::Fnv1a64Word(h, (uint32_t)(f.tick & 0xffffffff));
        h = mm::detail::Fnv1a64Word(h, (uint32_t)((uint64_t)f.tick >> 32));
        h = mm::detail::Fnv1a64Word(h, f.notifyId);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)f.payload);
    }
    return h;
}

// ===================== (5) The gameplay bridge (notify -> GT1/GAS1 tagged effect) =====================
namespace tags = hf::game::tags;
namespace gas  = hf::game::gas;

// A notify->ability binding: when a notify with `notifyId` fires, request `abilityId` from `caster` at
// `target` (a tags::TagCommand). The THIN adapter — NOT deep GAS integration.
struct NotifyAbilityBind {
    uint32_t              notifyId  = 0;
    uint32_t              abilityId = 0;
    tags::EntityId        caster    = 0;
    tags::EntityId        target    = 0;
};

// FindNotifyBind: the ability bound to a notify id, or nullptr.
inline const NotifyAbilityBind* FindNotifyBind(const std::vector<NotifyAbilityBind>& binds, uint32_t id) {
    for (const NotifyAbilityBind& b : binds) if (b.notifyId == id) return &b;
    return nullptr;
}

// ===================== The AL1 showcase skeleton + fixtures ===========================================
// A 9-bone stick-figure character (identity rest rotations; binary-fraction translations). Legs walk
// (base clip, translation swings), the right arm attacks (montage, exact-unit rotations), the spine leans
// (additive, an exact-unit rotation delta).
//   0 root  1 hipL  2 kneeL  3 hipR  4 kneeR  5 spine  6 shoulderR  7 elbowR  8 head
inline constexpr int kAl1Bones = 9;

inline Skeleton MakeAl1Skeleton() {
    Skeleton sk;
    Joint root;                                                            // 0
    Joint hipL;  hipL.parent = 0;  hipL.t  = math::Vec3{-0.25f, 0.0f, 0.0f};   // 1
    Joint kneeL; kneeL.parent = 1; kneeL.t = math::Vec3{ 0.0f, -0.5f, 0.0f};   // 2
    Joint hipR;  hipR.parent = 0;  hipR.t  = math::Vec3{ 0.25f, 0.0f, 0.0f};   // 3
    Joint kneeR; kneeR.parent = 3; kneeR.t = math::Vec3{ 0.0f, -0.5f, 0.0f};   // 4
    Joint spine; spine.parent = 0; spine.t = math::Vec3{ 0.0f, 0.5f, 0.0f};    // 5
    Joint shldR; shldR.parent = 5; shldR.t = math::Vec3{ 0.25f, 0.25f, 0.0f};  // 6
    Joint elbR;  elbR.parent = 6;  elbR.t  = math::Vec3{ 0.375f, 0.0f, 0.0f};  // 7
    Joint head;  head.parent = 5;  head.t  = math::Vec3{ 0.0f, 0.375f, 0.0f};  // 8
    sk.joints = {root, hipL, kneeL, hipR, kneeR, spine, shldR, elbR, head};
    return sk;
}

// The exact-unit stylized swing quaternion (the only binary-fraction non-axis unit quats — AN2's choice).
inline FxQuat Al1SwingQuat() { return FxQuat{kOne / 2, kOne / 2, kOne / 2, kOne / 2}; }

// The BASE locomotion clip: root eases +z; the knees (feet) swing +-0.25 in z in antiphase (a 4-tick
// half-period). Translation-only -> EXACT under sampling. Keyed at integer seconds 0..4.
inline Animation MakeAl1BaseClip() {
    Animation a;
    a.name = "al1_base";
    a.duration = 4.0f;
    Channel root; root.jointIndex = 0; root.path = Channel::Path::Translation;
    root.times  = {0.0f, 4.0f};
    root.values = {0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 1.0f};
    a.channels.push_back(root);
    Channel kl; kl.jointIndex = 2; kl.path = Channel::Path::Translation;
    kl.times  = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    kl.values = {0.0f, -0.5f, -0.25f,   0.0f, -0.5f, 0.25f,
                 0.0f, -0.5f, -0.25f,   0.0f, -0.5f, 0.25f,   0.0f, -0.5f, -0.25f};
    a.channels.push_back(kl);
    Channel kr; kr.jointIndex = 4; kr.path = Channel::Path::Translation;
    kr.times  = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    kr.values = {0.0f, -0.5f, 0.25f,   0.0f, -0.5f, -0.25f,
                 0.0f, -0.5f, 0.25f,   0.0f, -0.5f, -0.25f,   0.0f, -0.5f, 0.25f};
    a.channels.push_back(kr);
    return a;
}

// The montage WINDUP clip: the right shoulder holds identity, elbow holds identity (arm cocked back). One
// key -> constant. Keyed rotation identity (exact unit).
inline Animation MakeAl1WindupClip() {
    Animation a;
    a.name = "al1_windup";
    a.duration = 4.0f;
    Channel sh; sh.jointIndex = 6; sh.path = Channel::Path::Rotation;
    sh.times = {0.0f}; sh.values = {0.0f, 0.0f, 0.0f, 1.0f};
    a.channels.push_back(sh);
    return a;
}

// The montage STRIKE clip: the right shoulder swings identity -> qSwing -> identity (the arm arcs). Exact-
// unit keys at integer seconds sampled AT those times -> cross-compiler exact.
inline Animation MakeAl1StrikeClip() {
    Animation a;
    a.name = "al1_strike";
    a.duration = 4.0f;
    Channel sh; sh.jointIndex = 6; sh.path = Channel::Path::Rotation;
    sh.times  = {0.0f, 2.0f, 4.0f};
    sh.values = {0.0f, 0.0f, 0.0f, 1.0f,   0.5f, 0.5f, 0.5f, 0.5f,   0.0f, 0.0f, 0.0f, 1.0f};
    a.channels.push_back(sh);
    return a;
}

// The AL1 clip pool (indices: 0 base, 1 windup, 2 strike).
inline std::vector<Animation> MakeAl1Clips() {
    return std::vector<Animation>{MakeAl1BaseClip(), MakeAl1WindupClip(), MakeAl1StrikeClip()};
}

inline constexpr int32_t kAl1ClipBase   = 0;
inline constexpr int32_t kAl1ClipWindup = 1;
inline constexpr int32_t kAl1ClipStrike = 2;

// The notify ids (gameplay-event ids for the bridge).
inline constexpr uint32_t kNotifyFootstep = 100u;   // per-loop footstep (base locomotion)
inline constexpr uint32_t kNotifyHit      = 200u;   // the montage's "hit window" -> the damage ability

// The upper-body mask: kOne on spine(5)/shoulderR(6)/elbowR(7)/head(8), 0 on root+legs. The montage arm
// swing affects ONLY the upper body while the legs keep walking.
inline std::vector<fx> MakeUpperBodyMask() {
    std::vector<fx> mask((size_t)kAl1Bones, 0);
    mask[5] = kOne; mask[6] = kOne; mask[7] = kOne; mask[8] = kOne;
    return mask;
}

// The additive LEAN delta: identity everywhere except a spine rotation delta (qSwing) — a stylized exact-
// unit lean applied at a modest weight. Bones with an identity delta pass through == base.
inline std::vector<FxJointPose> MakeAl1LeanDelta() {
    std::vector<FxJointPose> d((size_t)kAl1Bones);   // all identity rot + zero translation
    d[5].r = Al1SwingQuat();                         // spine leans
    return d;
}

inline constexpr fx      kAl1LeanWeight = kOne / 4;   // the additive lean weight (stylized)
inline constexpr int32_t kAl1WindupLen  = 8;          // montage section A length (ticks)
inline constexpr int32_t kAl1StrikeLen  = 8;          // montage section B length (ticks)
inline constexpr int32_t kAl1HitLocal   = 2;          // the "hit" notify's local tick in the strike section
inline constexpr int32_t kAl1Steps      = 20;         // the scripted shot length (ticks)

// MakeAl1Montage: a 2-section attack montage. Section A "windup" (blend-in 4), section B "strike"
// (blend-out 4) carrying the "hit" notify at local tick kAl1HitLocal + a footstep-like notify. The clip
// time steps 0.5 s/tick so the 8-tick strike sweeps the 4 s strike clip once (samples land on the exact
// keyframes 0/2/4 at local ticks 0/4/8 — the exact-fixture discipline).
inline Montage MakeAl1Montage() {
    Montage m;
    m.secondsPerTick = 0.5f;
    MontageSection windup;
    windup.clip = kAl1ClipWindup; windup.lengthTicks = kAl1WindupLen; windup.blendInTicks = 4;
    MontageSection strike;
    strike.clip = kAl1ClipStrike; strike.lengthTicks = kAl1StrikeLen; strike.blendOutTicks = 4;
    strike.notifies.notifies = { Notify{kAl1HitLocal, kNotifyHit, 0} };
    m.sections = {windup, strike};
    return m;
}

// The hit-notify global tick (windup length + the strike-local hit tick) — pinned.
inline int32_t Al1HitGlobalTick() { return kAl1WindupLen + kAl1HitLocal; }

// ===================== The bridge world (GT1/GAS1) ===================================================
// A minimal 2-entity tagged world: the ATTACKER (id 1) and the TARGET (id 2, health 100). One damage
// ability (kAbHit): instant -25 health on the target. No tag gates — the point is the notify->effect
// TIMING, not the gating (GT1's skirmish covers the gating). The bridge routes the montage "hit" notify to
// this ability on the exact tick it fires.
inline constexpr uint32_t kAbHit       = 1u;
inline constexpr uint32_t kEffHitDamage = 10u;
inline constexpr fx       kAl1HitDamage = 25 * kOne;
inline constexpr fx       kAl1TargetHp  = 100 * kOne;

inline gas::AbilityKit MakeAl1Kit() {
    gas::KitBuilder b;
    b.Ability(kAbHit, gas::kAttrMana, 0, 0u)
     .Effect(kEffHitDamage, gas::kAttrHealth, gas::kOpAdd, -kAl1HitDamage, gas::kDurInstant, 0u,
             gas::kStackIgnore, 1u, 0u, gas::kTargetOther);
    return b.Build();
}

inline tags::TaggedWorld MakeAl1BridgeWorld() {
    tags::TaggedWorld tw;
    const fx attackerBases[gas::kAttrCount] = {
        kAl1TargetHp, kAl1TargetHp, 100 * kOne, 100 * kOne, 4 * kOne, 10 * kOne, 5 * kOne, 0 };
    const fx targetBases[gas::kAttrCount] = {
        kAl1TargetHp, kAl1TargetHp, 100 * kOne, 100 * kOne, 4 * kOne, 10 * kOne, 5 * kOne, 0 };
    tags::TagContainer none;
    tags::SpawnTagged(tw, attackerBases, none);   // attacker = 1
    tags::SpawnTagged(tw, targetBases, none);     // target   = 2
    return tw;
}

// ===================== The shared shot scenario (--al1-layer-shot, both backends) =====================
// The AN2 header-shared-scenario pattern: RunAl1ShotScenario + RenderAl1Shot are the ONE implementation
// both backends call — strict-zero cross-backend BY CONSTRUCTION (pure-CPU integer scenario; the raster
// consumes only integers). Per tick: sample the base locomotion, step the montage (fire notifies), build
// the layer stack (base -> additive lean -> upper-body montage slot), route the "hit" notify to the GT1/
// GAS1 damage ability BEFORE the GAS step (commands-before-step), and record the FK joints + the target
// health. The character walks (legs) while the right arm winds up + strikes (montage on the upper body)
// with a spine lean (additive), and the target's health drops on the exact hit tick.

// One recorded frame of the shot.
struct Al1ShotFrame {
    std::vector<FxJointModel> model;       // FK of the final layered pose (the stick figure)
    fx                        targetHp = 0;// the target's current health after this tick's GAS step
    uint8_t                   footstep = 0;// a footstep notify fired this tick
    uint8_t                   hit = 0;     // the hit notify fired this tick
    fx                        montageWeight = 0;
    int32_t                   montageTick = 0;
};

struct Al1ShotRun {
    std::vector<Al1ShotFrame> frames;
    int32_t   bones    = 0;
    int32_t   layers   = 0;
    int32_t   notifies = 0;   // total notifies fired across the shot
    int32_t   sections = 0;
    int32_t   hitTick  = -1;  // the tick the hit notify fired (== Al1HitGlobalTick when it lands in range)
    uint64_t  poseDigest   = 0;
    uint64_t  healthDigest = 0;
    uint64_t  notifyDigest = 0;
    uint64_t  digest       = 0;
};

// RunAl1ShotScenario: the pure function both backends call.
inline Al1ShotRun RunAl1ShotScenario() {
    Al1ShotRun run;
    const Skeleton sk = MakeAl1Skeleton();
    const std::vector<Animation> clips = MakeAl1Clips();
    const Montage montage = MakeAl1Montage();
    const std::vector<fx> upperMask = MakeUpperBodyMask();
    const std::vector<FxJointPose> lean = MakeAl1LeanDelta();

    // The base-locomotion footstep track (loop-aware: a footstep every 2 ticks of the 4-tick loop).
    NotifyTrack footTrack;
    footTrack.loopLen = 4;
    footTrack.notifies = { Notify{0, kNotifyFootstep, 0}, Notify{2, kNotifyFootstep, 0} };

    // The bridge world + the notify->ability binding (attacker 1 hits target 2 on the "hit" notify).
    const gas::AbilityKit kit = MakeAl1Kit();
    const tags::TagRegistry reg;                       // no tags needed — empty registry
    const tags::TagRules rules;                        // no gates
    tags::TaggedWorld tw = MakeAl1BridgeWorld();
    const std::vector<NotifyAbilityBind> binds = {
        NotifyAbilityBind{kNotifyHit, kAbHit, /*caster*/1u, /*target*/2u} };

    MontagePlayer mp;
    run.bones = kAl1Bones;
    run.layers = 2;                                    // additive lean + montage slot
    run.sections = (int32_t)montage.sections.size();
    run.frames.reserve((size_t)kAl1Steps);

    uint64_t hp = 14695981039346656037ull, hh = hp, hn = hp;
    for (int32_t t = 0; t < kAl1Steps; ++t) {
        // (1) base locomotion pose at this tick (sample base clip at t seconds; 1 tick == 1 s here).
        const std::vector<JointPose> baseF = SampleLocalPose(sk, clips[(size_t)kAl1ClipBase], (float)t);
        const std::vector<FxJointPose> baseQ = retarget::QuantizePose(baseF);

        // (2) footstep notifies over (t-1, t].
        const std::vector<FiredNotify> footFired = SampleNotifies(footTrack, (int64_t)t - 1, (int64_t)t);

        // (3) step the montage by 1 tick; fire its notifies (incl. the hit window).
        const std::vector<FiredNotify> monFired = StepMontage(montage, mp, 1);

        // (4) build the layer stack: base -> additive lean -> upper-body montage slot.
        std::vector<FxJointPose> withLean = ApplyAdditive(baseQ, lean, kAl1LeanWeight);
        std::vector<FxJointPose> slotPose; fx slotWeight = 0;
        EvaluateMontage(montage, mp, clips, sk, slotPose, slotWeight);
        AnimLayerStack stack;
        stack.base = withLean;
        Layer arm; arm.pose = slotPose; arm.weight = slotWeight; arm.mask = upperMask;
        stack.layers = {arm};
        const std::vector<FxJointPose> finalPose = EvaluateLayers(stack);

        // (5) route the hit notify to the tagged ability BEFORE the GAS step (commands-before-step).
        Al1ShotFrame fr;
        fr.montageWeight = slotWeight;
        fr.montageTick = mp.tick;
        for (const FiredNotify& f : footFired) if (f.notifyId == kNotifyFootstep) fr.footstep = 1;
        for (const FiredNotify& f : monFired) {
            if (f.notifyId == kNotifyHit) { fr.hit = 1; run.hitTick = t; }
            const NotifyAbilityBind* b = FindNotifyBind(binds, f.notifyId);
            if (b) tags::TryActivateTagged(tw, kit, rules, reg, b->caster, b->abilityId, b->target);
        }
        // The GAS step (this tick's committed activation resolves; the target's health updates).
        gas::StepAbilities(tw.gas);
        const int ti = gas::FindEntity(tw.gas, 2u);
        fr.targetHp = (ti >= 0) ? tw.gas.entities[(size_t)ti].attrs.current[gas::kAttrHealth] : 0;

        // (6) FK the final pose for the viz.
        fr.model = retarget::ForwardKinematics(sk, finalPose);

        // Fold digests.
        const uint64_t dp = DigestFxPose(finalPose);
        hp = mm::detail::Fnv1a64Word(hp, (uint32_t)(dp & 0xffffffffu));
        hp = mm::detail::Fnv1a64Word(hp, (uint32_t)(dp >> 32));
        hh = mm::detail::Fnv1a64Word(hh, (uint32_t)fr.targetHp);
        for (const FiredNotify& f : footFired) { hn = mm::detail::Fnv1a64Word(hn, f.notifyId); }
        for (const FiredNotify& f : monFired)  { hn = mm::detail::Fnv1a64Word(hn, f.notifyId);
                                                 hn = mm::detail::Fnv1a64Word(hn, (uint32_t)f.tick); }
        run.notifies += (int32_t)footFired.size() + (int32_t)monFired.size();
        run.frames.push_back(std::move(fr));
    }
    run.poseDigest = hp;
    run.healthDigest = hh;
    run.notifyDigest = hn;
    uint64_t h = 14695981039346656037ull;
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hp & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hp >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hh & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hh >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hn & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hn >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)run.hitTick);
    run.digest = h;
    return run;
}

// RenderAl1Shot: the PURE-INTEGER raster both backends call — strict-zero cross-backend BY CONSTRUCTION.
// Top: the stick figure over several frames (legs walking + the right arm winding up/striking + the spine
// lean), older frames dimmer. Middle: a per-tick timeline strip with footstep (sage) + hit (red) notify
// markers and the montage-weight bar. Bottom: the target health bar reacting at the hit tick. 520x420
// BGRA8, integer DDA/discs only.
inline void RenderAl1Shot(const Al1ShotRun& run, std::vector<uint8_t>& bgra, uint32_t& outW, uint32_t& outH) {
    const int W = 520, H = 420;
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {   // deep slate ground
        bgra[p * 4 + 0] = 24; bgra[p * 4 + 1] = 18; bgra[p * 4 + 2] = 14; bgra[p * 4 + 3] = 255;
    }
    auto putPx = [&](int ix, int iy, uint8_t r, uint8_t g, uint8_t b) {
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) return;
        uint8_t* dst = &bgra[((size_t)iy * W + ix) * 4];
        dst[0] = b; dst[1] = g; dst[2] = r; dst[3] = 255;
    };
    auto line = [&](int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
        const int dx = x1 - x0, dy = y1 - y0;
        int steps = dx >= 0 ? dx : -dx;
        const int ady = dy >= 0 ? dy : -dy;
        if (ady > steps) steps = ady;
        if (steps == 0) { putPx(x0, y0, r, g, b); return; }
        for (int i = 0; i <= steps; ++i) {
            const int px = x0 + (int)(((int64_t)dx * i * 2 + steps) / (2 * (int64_t)steps));
            const int py = y0 + (int)(((int64_t)dy * i * 2 + steps) / (2 * (int64_t)steps));
            putPx(px, py, r, g, b);
        }
    };
    auto disc = [&](int cx, int cy, int rr, uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -rr; dy <= rr; ++dy)
            for (int dx = -rr; dx <= rr; ++dx)
                if (dx * dx + dy * dy <= rr * rr) putPx(cx + dx, cy + dy, r, g, b);
    };
    const Skeleton sk = MakeAl1Skeleton();
    const int nF = (int)run.frames.size();
    // (1) The stick figure over frames. Model (x,y) -> screen; +z sheared right so the stride reads.
    const int kScale = 90, kBaseY = 210;
    auto project = [&](const FxJointModel& j, int centerX, int frame, int& sx, int& sy) {
        sx = centerX + (int)(((int64_t)j.pos.x * kScale) >> kFrac) + (int)(((int64_t)j.pos.z * 20) >> kFrac)
             + frame * 6;
        sy = kBaseY - (int)(((int64_t)j.pos.y * kScale) >> kFrac);
    };
    auto shade = [&](int frame, uint8_t base) -> uint8_t {
        const int lo = 70;
        const int v = lo + (base - lo) * (frame + 1) / (nF > 0 ? nF : 1);
        return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    };
    for (int f = 0; f < nF; ++f) {
        const Al1ShotFrame& fr = run.frames[(size_t)f];
        const uint8_t r = shade(f, 232), g = shade(f, 190), b = shade(f, 96);   // amber, older dimmer
        // hit frames flash red.
        const uint8_t rr = fr.hit ? 240 : r, gg = fr.hit ? 90 : g, bb = fr.hit ? 90 : b;
        for (size_t j = 0; j < fr.model.size() && j < sk.joints.size(); ++j) {
            int sx, sy; project(fr.model[j], 150, f, sx, sy);
            const int parent = sk.joints[j].parent;
            if (parent >= 0 && (size_t)parent < fr.model.size()) {
                int px, py; project(fr.model[(size_t)parent], 150, f, px, py);
                line(px, py, sx, sy, rr, gg, bb);
            }
            disc(sx, sy, 1, rr, gg, bb);
        }
    }
    // (2) The per-tick timeline strip: py [250, 300). Each tick a 20-px column. Footstep marker (sage) at
    // the base, hit marker (red) at the top; the montage-weight bar in cyan.
    const int stripBase = 300, stripH = 44;
    for (int f = 0; f < nF; ++f) {
        const Al1ShotFrame& fr = run.frames[(size_t)f];
        const int x0 = 20 + f * 24;
        const int wh = (int)(((int64_t)fr.montageWeight * stripH) >> kFrac);
        for (int yy = 0; yy < wh; ++yy)
            for (int xx = 0; xx < 6; ++xx) putPx(x0 + xx, stripBase - 1 - yy, 96, 214, 214);   // cyan weight
        if (fr.footstep) disc(x0 + 3, stripBase + 6, 2, 140, 200, 120);                         // sage footstep
        if (fr.hit)      disc(x0 + 3, stripBase - stripH - 6, 3, 240, 84, 84);                   // red hit
    }
    // (3) The target health bar: py [340, 400). A red-to-green bar per tick (health / max).
    const int hpBase = 400, hpH = 54, hpMax = (int)(kAl1TargetHp >> kFrac);
    for (int f = 0; f < nF; ++f) {
        const Al1ShotFrame& fr = run.frames[(size_t)f];
        const int x0 = 20 + f * 24;
        const int hpv = (int)(fr.targetHp >> kFrac);
        const int hh = (hpMax > 0) ? (hpv * hpH / hpMax) : 0;
        for (int yy = 0; yy < hh; ++yy)
            for (int xx = 0; xx < 6; ++xx) putPx(x0 + xx, hpBase - 1 - yy, 120, 214, 130);       // green health
        // a thin red cap where health was lost (below full).
        if (hpv < hpMax)
            for (int xx = 0; xx < 6; ++xx) putPx(x0 + xx, hpBase - hh - 2, 226, 84, 84);
    }
}

}  // namespace layer
}  // namespace hf::anim
