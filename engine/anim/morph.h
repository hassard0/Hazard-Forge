#pragma once
// Slice MP1 — DETERMINISTIC MORPH TARGETS / BLEND SHAPES (per-vertex morph deltas + weighted blend +
// animated weight tracks, composing with skinning). The next-tier parity gap #5: morph targets / blend
// shapes were explicitly SKIPPED — gltf_loader.cpp (~L792) has a "YAGNI: morph" that drops glTF `weights`
// animation channels, and CAPABILITIES.md confirms morph targets are not built. UE5 ships morph targets
// (facial animation, blend shapes); MP1 adds them DETERMINISTICALLY: represent morph-target vertex deltas,
// blend them by weights, drive the weights from an animation track, and COMPOSE with the existing skinning
// (morph BEFORE skin — the standard order). Namespace hf::anim::morph, header-only, pure CPU, NO device /
// backend symbols, NO RNG, NO clock, NO new shader. anim/animation.h + anim/motion_match.h are #included
// READ-ONLY (byte-untouched) for the scalar-keyframe sampling pattern + the Q16.16 currency (mm::fx / kOne /
// kFrac / fxmul / QuantizeFx / FxV3 + mm::detail::Fnv1a64Word — the same integer bits AN1/AN2/SK1 pin).
//
// LOADER DECISION (documented, honest): gltf_loader.cpp is LEFT BYTE-UNTOUCHED (its "YAGNI: morph" stays).
// MP1 is a NEW header composing the anim stack read-only, NOT a loader extension — the cgltf import is
// device-coupled and the SK1 precedent is a DEVICE-FREE authored-literal import through a real reshaping
// step. So the "import" here reshapes a glTF-STYLE authored accessor description (GltfMorphAsset: base
// positions + per-target flat delta accessors + a mesh default weights[] + a `weights` animation track — the
// exact shape a glTF primitive.targets[] + mesh.weights + a weights-channel carries) into a MorphSet +
// WeightTrack via BuildMorphSet/BuildWeightTrack. HONEST CAVEAT: the asset is a HAND-AUTHORED accessor
// literal (Mp1MorphAsset), NOT parsed from a binary .glb on disk — the SK1 authored-literal discipline; a
// full cgltf morph-accessor read is a follow-up that would additively extend gltf_loader.cpp.
//
// DETERMINISM DECISION (documented, honest): scene::Vertex positions are FLOAT. Two blend paths are provided:
//   * ApplyMorph   — the FLOAT authoring blend (deformed[v] = base[v] + Σ_t weight[t]*delta_t[v], targets in
//                    ASCENDING fixed accumulation order). Natural for the render vertex format.
//   * ApplyMorphFx — the INTEGER Q16.16 blend (base + deltas + weights quantized ONCE at the QuantizeFx
//                    boundary, then acc += fxmul(weightQ, deltaQ) in the SAME ascending order). This is the
//                    PINNED determinism path — pure int32 state / int64 intermediates, bit-identical across
//                    compilers / peers BY CONSTRUCTION (the AN2/SK1 discipline; no FMA / rounding sensitivity
//                    past the boundary). All authored deltas/weights are exact binary fractions, so the two
//                    paths AGREE and the QuantizeFx inputs are themselves cross-compiler exact. The digests +
//                    the compose proof + the showcase all read the INTEGER path.
//
// COMPOSE ORDER (PINNED — the standard, proven by mp1's compose test): base -> ApplyMorph(weights) ->
// skin(palette) -> final. Morph deforms the BASE (bind-space) position BEFORE skinning transforms it. This
// ORDER MATTERS whenever a bone carries a rotation: skin(base+delta) != skin(base)+delta (the delta must be
// rotated with the vertex). SkinPointFx composes with the SK1/AN2 integer FK palette (retarget::FxJointModel
// + FxQuatRotate), so a rigged+morphed vertex is bit-exact end to end.
//
// NORMAL DELTAS: optional per-target normal deltas blend the SAME way (base_n + Σ w*delta_n). HONEST CAVEAT:
// the blended normal is NOT re-normalized here (a post-blend renormalize is only APPROXIMATE for large
// weights and would introduce a sqrt — deferred; the render lit path renormalizes in-shader as it already
// does). MP1 pins the RAW blended normal delta sum.
//
// SCOPE (v1) — documented deferrals: authored-accessor import (no binary .glb cgltf read); POSITION (+
// optional NORMAL) deltas only (no TANGENT deltas); LINEAR weight-track interpolation (glTF morph weights are
// LINEAR/STEP; no CUBICSPLINE tangents); a single MorphSet per mesh; no sparse-accessor delta storage (dense
// per-vertex deltas). The float engine render path is untouched — MP1 is additive.

#include <cstdint>
#include <string>
#include <vector>

#include "anim/animation.h"     // READ-ONLY: the clip/keyframe model (the scalar-keyframe sampling pattern)
#include "anim/motion_match.h"  // READ-ONLY: mm::fx/kOne/kFrac/fxmul/QuantizeFx/FxV3 + Fnv1a64Word (Q16.16 currency)
#include "math/math.h"          // math::Vec3 (the authoring delta type — matches scene::Vertex float positions)

namespace hf::anim {
namespace morph {

using mm::fx;
using mm::FxV3;
using mm::fxmul;
using mm::kFrac;
using mm::kOne;
using mm::QuantizeFx;   // THE float->integer boundary (llround, round-half-away — the MM1/AN2/SK1 discipline)

// ===================== Morph representation ==========================================================

// A morph target (blend shape): per-vertex POSITION deltas (+ optional per-vertex NORMAL deltas), a name.
// Deltas are in the mesh's coordinate space (added to the base vertex). normalDeltas is empty == none.
struct MorphTarget {
    std::string             name;
    std::vector<math::Vec3> positionDeltas;   // per base vertex (size == base vertex count)
    std::vector<math::Vec3> normalDeltas;      // optional; empty == no normal deltas
};

// A mesh's morph set: the base (neutral) vertices, the targets, and the current/default weights (one per
// target). weights.size() == targets.size(); a target's contribution is weight[t] * delta_t.
struct MorphSet {
    std::vector<math::Vec3> basePositions;    // neutral mesh positions
    std::vector<math::Vec3> baseNormals;       // optional neutral normals (empty == none)
    std::vector<MorphTarget> targets;
    std::vector<float>       weights;          // current/default weight per target (parallel to targets)
};

// A morph WEIGHT animation track: a scalar-per-target keyframe stream over time (the glTF `weights` channel
// — one output accessor of `targetCount` scalars per keyframe time). LINEAR interpolation (v1 scope).
//   * times  — keyframe timestamps in seconds (sorted ascending), K keys.
//   * values — packed K*targetCount weights: values[k*targetCount + t] is target t's weight at key k.
struct WeightTrack {
    std::vector<float> times;
    std::vector<float> values;
    int                targetCount = 0;
};

// ===================== glTF-style authored asset (the import source) ==================================

// A glTF-style authored morph description — the exact accessor shape a glTF primitive.targets[] +
// mesh.weights + a `weights` animation channel carries, hand-authored (the SK1 authored-literal discipline;
// no binary .glb). basePositions is a flat xyz array (vCount*3); targetDeltas[t] is target t's flat xyz
// delta accessor (vCount*3); defaultWeights is the mesh default weights[]; the weight track is the glTF
// weights-channel input (weightTimes, K) + output (weightValues, K*targetCount).
struct GltfMorphAsset {
    std::vector<float>              basePositions;   // vCount*3
    std::vector<float>              baseNormals;      // optional vCount*3 (empty == none)
    std::vector<std::vector<float>> targetDeltas;    // [target][vCount*3]
    std::vector<std::vector<float>> targetNormalDeltas;  // optional [target][vCount*3] (empty == none)
    std::vector<std::string>        targetNames;
    std::vector<float>              defaultWeights;  // per target
    std::vector<float>              weightTimes;     // K keyframe times (seconds)
    std::vector<float>              weightValues;    // K*targetCount (glTF weights output accessor)
};

// BuildMorphSet: reshape the flat glTF-style accessors into a MorphSet (the "import" — mirrors how a loader
// reshapes primitive.targets[] accessors into per-vertex delta arrays). Deterministic, pure copy.
inline MorphSet BuildMorphSet(const GltfMorphAsset& a) {
    MorphSet ms;
    const size_t vCount = a.basePositions.size() / 3;
    ms.basePositions.resize(vCount);
    for (size_t v = 0; v < vCount; ++v)
        ms.basePositions[v] = math::Vec3{a.basePositions[v * 3 + 0], a.basePositions[v * 3 + 1],
                                         a.basePositions[v * 3 + 2]};
    if (a.baseNormals.size() == vCount * 3) {
        ms.baseNormals.resize(vCount);
        for (size_t v = 0; v < vCount; ++v)
            ms.baseNormals[v] = math::Vec3{a.baseNormals[v * 3 + 0], a.baseNormals[v * 3 + 1],
                                           a.baseNormals[v * 3 + 2]};
    }
    const size_t nT = a.targetDeltas.size();
    ms.targets.resize(nT);
    for (size_t t = 0; t < nT; ++t) {
        MorphTarget& mt = ms.targets[t];
        mt.name = (t < a.targetNames.size()) ? a.targetNames[t] : std::string();
        const std::vector<float>& d = a.targetDeltas[t];
        const size_t dv = d.size() / 3;
        mt.positionDeltas.resize(dv);
        for (size_t v = 0; v < dv; ++v)
            mt.positionDeltas[v] = math::Vec3{d[v * 3 + 0], d[v * 3 + 1], d[v * 3 + 2]};
        if (t < a.targetNormalDeltas.size() && a.targetNormalDeltas[t].size() == vCount * 3) {
            const std::vector<float>& n = a.targetNormalDeltas[t];
            mt.normalDeltas.resize(vCount);
            for (size_t v = 0; v < vCount; ++v)
                mt.normalDeltas[v] = math::Vec3{n[v * 3 + 0], n[v * 3 + 1], n[v * 3 + 2]};
        }
    }
    ms.weights = a.defaultWeights;
    if (ms.weights.size() != nT) ms.weights.assign(nT, 0.0f);   // default all-zero (neutral)
    return ms;
}

// BuildWeightTrack: reshape the glTF weights-channel accessors (input times + output K*targetCount) into a
// WeightTrack. targetCount is inferred from targetDeltas (the authored morph target count).
inline WeightTrack BuildWeightTrack(const GltfMorphAsset& a) {
    WeightTrack wt;
    wt.targetCount = (int)a.targetDeltas.size();
    wt.times = a.weightTimes;
    wt.values = a.weightValues;
    return wt;
}

// ===================== Weighted blend =================================================================

// ApplyMorph (FLOAT): deformed[v] = base[v] + Σ_t weight[t]*delta_t[v], over targets in ASCENDING index
// order (the fixed accumulation order). A target with weight 0 contributes nothing; a missing/short delta
// array leaves that vertex unchanged for that target. deformed.size() == base.size().
inline std::vector<math::Vec3> ApplyMorph(const std::vector<math::Vec3>& base,
                                          const std::vector<MorphTarget>& targets,
                                          const std::vector<float>& weights) {
    std::vector<math::Vec3> out = base;
    const size_t nT = targets.size();
    for (size_t t = 0; t < nT; ++t) {                 // ascending target order — the pinned accumulation
        const float w = (t < weights.size()) ? weights[t] : 0.0f;
        if (w == 0.0f) continue;
        const std::vector<math::Vec3>& d = targets[t].positionDeltas;
        const size_t n = d.size() < out.size() ? d.size() : out.size();
        for (size_t v = 0; v < n; ++v) {
            out[v].x += w * d[v].x;
            out[v].y += w * d[v].y;
            out[v].z += w * d[v].z;
        }
    }
    return out;
}

// ApplyMorphNormals (FLOAT): the RAW blended normal (base_n + Σ w*delta_n) — NOT re-normalized (banner
// caveat). Empty if the set carries no base normals.
inline std::vector<math::Vec3> ApplyMorphNormals(const std::vector<math::Vec3>& baseNormals,
                                                 const std::vector<MorphTarget>& targets,
                                                 const std::vector<float>& weights) {
    std::vector<math::Vec3> out = baseNormals;
    if (out.empty()) return out;
    const size_t nT = targets.size();
    for (size_t t = 0; t < nT; ++t) {
        const float w = (t < weights.size()) ? weights[t] : 0.0f;
        if (w == 0.0f || targets[t].normalDeltas.empty()) continue;
        const std::vector<math::Vec3>& d = targets[t].normalDeltas;
        const size_t n = d.size() < out.size() ? d.size() : out.size();
        for (size_t v = 0; v < n; ++v) {
            out[v].x += w * d[v].x;
            out[v].y += w * d[v].y;
            out[v].z += w * d[v].z;
        }
    }
    return out;
}

// QuantizeVerts: snap a float vertex stream to Q16.16 (the ONE float->integer boundary, per vertex).
inline std::vector<FxV3> QuantizeVerts(const std::vector<math::Vec3>& v) {
    std::vector<FxV3> out(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        out[i] = FxV3{QuantizeFx(v[i].x), QuantizeFx(v[i].y), QuantizeFx(v[i].z)};
    return out;
}

// ApplyMorphFx (INTEGER Q16.16 — the PINNED determinism path): the base + every target's deltas + the
// weights are quantized ONCE at the QuantizeFx boundary; then deformed[v] = baseQ[v] + Σ_t fxmul(weightQ[t],
// deltaQ_t[v]) in ASCENDING target order (int64 intermediates in fxmul; int32 accumulation). Bit-identical
// across compilers/peers BY CONSTRUCTION. weightsQ are already-quantized Q16.16 weights (see SampleMorphWeights
// -> QuantizeFx, or QuantizeFx of a MorphSet.weights entry).
inline std::vector<FxV3> ApplyMorphFx(const std::vector<FxV3>& baseQ,
                                      const std::vector<MorphTarget>& targets,
                                      const std::vector<fx>& weightsQ) {
    std::vector<FxV3> out = baseQ;
    const size_t nT = targets.size();
    for (size_t t = 0; t < nT; ++t) {                 // ascending target order — matches the float path
        const fx w = (t < weightsQ.size()) ? weightsQ[t] : 0;
        if (w == 0) continue;
        const std::vector<math::Vec3>& d = targets[t].positionDeltas;
        const size_t n = d.size() < out.size() ? d.size() : out.size();
        for (size_t v = 0; v < n; ++v) {
            out[v].x += fxmul(w, QuantizeFx(d[v].x));
            out[v].y += fxmul(w, QuantizeFx(d[v].y));
            out[v].z += fxmul(w, QuantizeFx(d[v].z));
        }
    }
    return out;
}

// ===================== Weight animation ==============================================================

// SampleMorphWeights: sample the weight track at `time` (seconds) -> one weight per target, LINEAR-
// interpolated between the bracketing keyframes (clamped to the first/last key outside the range). Mirrors
// animation.cpp's FindKey discipline. Returns targetCount weights (all 0 if the track is empty).
inline std::vector<float> SampleMorphWeights(const WeightTrack& track, float time) {
    const int tc = track.targetCount;
    std::vector<float> out((size_t)(tc > 0 ? tc : 0), 0.0f);
    const size_t K = track.times.size();
    if (K == 0 || tc <= 0) return out;
    auto keyVals = [&](size_t k, int t) -> float {
        const size_t idx = k * (size_t)tc + (size_t)t;
        return idx < track.values.size() ? track.values[idx] : 0.0f;
    };
    if (K == 1 || time <= track.times.front()) {
        for (int t = 0; t < tc; ++t) out[(size_t)t] = keyVals(0, t);
        return out;
    }
    if (time >= track.times.back()) {
        for (int t = 0; t < tc; ++t) out[(size_t)t] = keyVals(K - 1, t);
        return out;
    }
    // first key strictly greater than time -> bracket [i0, i1]
    size_t i1 = 1;
    while (i1 < K && track.times[i1] <= time) ++i1;
    const size_t i0 = i1 - 1;
    const float span = track.times[i1] - track.times[i0];
    const float frac = span > 0.0f ? (time - track.times[i0]) / span : 0.0f;
    for (int t = 0; t < tc; ++t) {
        const float a = keyVals(i0, t), b = keyVals(i1, t);
        out[(size_t)t] = a + (b - a) * frac;
    }
    return out;
}

// SampleMorphWeightsFx: the Q16.16 twin — sample (float) then quantize each weight at the ONE boundary. The
// weights feed ApplyMorphFx (the integer blend). Kept as a distinct call so the boundary is explicit.
inline std::vector<fx> SampleMorphWeightsFx(const WeightTrack& track, float time) {
    const std::vector<float> w = SampleMorphWeights(track, time);
    std::vector<fx> out(w.size());
    for (size_t i = 0; i < w.size(); ++i) out[i] = QuantizeFx(w[i]);
    return out;
}

// ===================== Digests (the pinned-golden currency, FNV-1a-64) ================================

// DigestVertsFx: fold a Q16.16 vertex stream (x,y,z per vertex) through the MM1 FNV primitive.
inline uint64_t DigestVertsFx(const std::vector<FxV3>& v) {
    uint64_t h = 14695981039346656037ull;
    for (const FxV3& p : v) {
        h = mm::detail::Fnv1a64Word(h, (uint32_t)p.x);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)p.y);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)p.z);
    }
    return h;
}

// DigestMorphSet: fold the imported morph set (base positions, per-target name + deltas, default weights) —
// every float snapped to Q16.16 so the digest is cross-compiler exact for exactly-representable authored data.
inline uint64_t DigestMorphSet(const MorphSet& ms) {
    uint64_t h = 14695981039346656037ull;
    auto putF = [&](float f) { h = mm::detail::Fnv1a64Word(h, (uint32_t)QuantizeFx(f)); };
    auto putI = [&](uint32_t w) { h = mm::detail::Fnv1a64Word(h, w); };
    putI((uint32_t)ms.basePositions.size());
    for (const math::Vec3& p : ms.basePositions) { putF(p.x); putF(p.y); putF(p.z); }
    putI((uint32_t)ms.targets.size());
    for (const MorphTarget& t : ms.targets) {
        for (char c : t.name) putI((uint32_t)(unsigned char)c);
        putI(0xffffffffu);
        putI((uint32_t)t.positionDeltas.size());
        for (const math::Vec3& d : t.positionDeltas) { putF(d.x); putF(d.y); putF(d.z); }
        putI((uint32_t)t.normalDeltas.size());
        for (const math::Vec3& d : t.normalDeltas) { putF(d.x); putF(d.y); putF(d.z); }
    }
    putI((uint32_t)ms.weights.size());
    for (float w : ms.weights) putF(w);
    return h;
}

// DigestWeightTrack: fold the weight track (times + K*targetCount values).
inline uint64_t DigestWeightTrack(const WeightTrack& wt) {
    uint64_t h = 14695981039346656037ull;
    h = mm::detail::Fnv1a64Word(h, (uint32_t)wt.targetCount);
    h = mm::detail::Fnv1a64Word(h, (uint32_t)wt.times.size());
    for (float t : wt.times)  h = mm::detail::Fnv1a64Word(h, (uint32_t)QuantizeFx(t));
    for (float v : wt.values) h = mm::detail::Fnv1a64Word(h, (uint32_t)QuantizeFx(v));
    return h;
}

}  // namespace morph
}  // namespace hf::anim
