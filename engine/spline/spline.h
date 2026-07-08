#pragma once
// Slice SP1 — FIRST-CLASS DETERMINISTIC SPLINES (eval + arc-length + spline-driven scatter/mesh/camera).
// UE5's spline component drives roads, fences, camera rails and PCG paths — all of it FLOAT (two machines
// evaluating the same spline diverge in the low bits). SP1 rebuilds the whole stack in Q16.16 INTEGER:
// a uniform Catmull-Rom evaluator + analytic tangent, a fixed-K chord-table arc-length reparameterization,
// spline-scatter (the open PC1 roadmap item — instances {pos, yaw-from-tangent} that FEED the existing
// hf::pcg transform/prune stages), a spline-swept mesh strip (the road primitive, engine mesh layout via
// the ONE float render bridge), and a spline camera track. Everything on the core path is bit-exact
// cross-platform BY CONSTRUCTION (pure int32/int64 arithmetic — NO float, NO <cmath>, NO runtime
// transcendentals, NO clock/RNG beyond the seeded pcg hash).
//
// STATELESS ON PURPOSE — NO LOCKSTEP HARNESS: unlike fpx/cloth/fluid/grain, a spline has NO evolving
// state. Eval/EvalByDistance/ScatterAlongSpline/SweepStrip/CameraAlongSpline are PURE FUNCTIONS of
// (control points, parameters); there is nothing to snapshot, roll back, or feed a command stream — two
// netcode peers holding the same control points already agree bit-for-bit on every derived value. The
// lockstep story is inherited by the CONSUMERS (a camera distance driven by a seq.h ScalarTrack is
// lockstep-replayable because seq.h is; a scattered field pruned by pcg.h is a pure function of the seed).
//
// THE PARAMETERIZATION (v1, DOCUMENTED CHOICE): UNIFORM Catmull-Rom with the standard basis. Centripetal
// Catmull-Rom (the loop/cusp-free variant) parameterizes knots by sqrt(chord length) — a per-knot
// fractional power that is float-hostile and would drag sqrt-of-sqrt integer plumbing through every
// evaluation. Uniform CR keeps the basis matrix EXACT small-integer (all coefficients in {-1,0,2,-5,4,3})
// so evaluation is int64 Horner with a single >>16 per multiply — bit-exact everywhere. THE v1 CAVEAT
// (honesty over green): uniform CR can OVERSHOOT/LOOP/CUSP when control-point spacing is very uneven.
// Concrete example: points {(0,0,0), (0.1,0,0), (10,0,0.1), (10.1,0,0)} — the tiny first span next to the
// huge second span makes segment 1 swing far outside the hull (the classic uniform-CR loop). Author
// splines with roughly-even control spacing (the showcase does); centripetal-via-LUT is the future
// fidelity slice. The arc-length table (SP1-B) does NOT fix uneven parameterization — it fixes uneven
// SPEED along the curve, which is the property scatter/sweep/camera actually need.
//
// ENDPOINT CONVENTION (pinned): open splines MIRROR the end neighbors — segment 0 uses the phantom
// P0 = 2*points[0] - points[1], the last segment uses P3 = 2*points[n-1] - points[n-2]. This is what
// makes the 2-POINT SPLINE AN EXACT INTEGER LERP (the mirrored phantoms zero the t^2/t^3 coefficients:
// c2 = c3 = 0 exactly — see the derivation at EvalAxis), the SP1 degeneracy identity. Closed splines
// wrap modulo n (segment i spans points[i] -> points[(i+1)%n], n segments), and Eval(end)==Eval(0)
// bit-exact because Catmull-Rom interpolates its knots EXACTLY in this integer form (t=0 -> 2*P1>>1,
// t=kOne -> 2*P2>>1 — both exact; the basis rows sum to {0,2,0,0}/{0,0,2,0}).
//
// OVERFLOW BUDGET (documented, pinned): control-point coordinates must satisfy |coord| <= 2^29
// (+-8192 world units in Q16.16). Then the mirrored phantoms are <= 3*2^29 < 2^31 (int64-safe; they are
// FORMED in int64, never stored in an fx), the basis coefficients are <= 18*2^29 < 2^34, the Horner
// accumulator is <= ~2^35, and every (acc * t) product with t <= kOne = 2^16 is <= ~2^51 — comfortably
// inside int64. Results land back in fx range because the curve stays within a small multiple of the
// control hull. The arc-length table stores CUMULATIVE length in fx, so total spline length must also be
// < 2^31 Q16.16 (~32768 wu) — vastly beyond any authored road.
//
// REUSE MAP: fpx.h (fx/kOne/kFrac/fxmul/fxdiv/FxVec3/FxAdd/FxSub/FxScale/FxISqrt/FxLength/FxNormalize/
// FxQuat) read-only; pcg.h (PcgStream/PcgRandRange, PcgInstance — the scatter output currency so
// PruneOverlaps/BuildInstances compose UNCHANGED) read-only; net/session.h DigestBytes (the digest
// currency); scene/vertex.h (the FLOAT mesh bridge output shape ONLY — the PCG6 one-float-crossing
// precedent). NO new RHI, NO shader, NO device — pure-CPU header, both showcase backends run the
// IDENTICAL scenario + raster below (the WV1 header-shared-scenario zero-copy-drift pattern).

#include <cstdint>
#include <vector>

#include "net/session.h"    // hf::net::DigestBytes (FNV-1a-64 over raw bytes) — the digest currency
#include "pcg/pcg.h"        // PcgStream/PcgRandRange + PcgInstance (the scatter feeds pcg transform/prune)
#include "scene/vertex.h"   // scene::Vertex (the FLOAT mesh-bridge output shape ONLY — SP1-D render bridge)
#include "sim/fpx.h"        // the Q16.16 toolbox (read-only)

namespace hf::spline {

using hf::sim::fpx::fx;
using hf::sim::fpx::kOne;
using hf::sim::fpx::kFrac;
using hf::sim::fpx::fxmul;
using hf::sim::fpx::fxdiv;
using hf::sim::fpx::FxVec3;
using hf::sim::fpx::FxAdd;
using hf::sim::fpx::FxSub;
using hf::sim::fpx::FxScale;
using hf::sim::fpx::FxISqrt;
using hf::sim::fpx::FxLength;
using hf::sim::fpx::FxNormalize;
using hf::sim::fpx::FxQuat;

// ===================== SP1-A — the eval core (uniform Catmull-Rom, integer Horner) =====================

// The spline: ordered Q16.16 control points + the open/closed flag. Plain data (no behaviour) — every
// operation below is a free function of (spline, params), the stateless discipline.
struct Spline {
    std::vector<FxVec3> points;
    bool                closed = false;
};

// SegmentCount: open n-point spline has n-1 segments (segment i spans points[i]->points[i+1]);
// closed has n (the last wraps back to points[0]). n < 2 -> 0 (degenerate; Eval returns points[0]/zero).
inline int SegmentCount(const Spline& s) {
    const int n = (int)s.points.size();
    if (n < 2) return 0;
    return s.closed ? n : n - 1;
}

// SegControl: the four int64 control values per axis for segment `seg` (P[0..3][axis 0..2]).
// Closed: indices wrap modulo n. Open: interior neighbors are the real points; the END NEIGHBORS ARE
// MIRRORED (P0 = 2*points[0]-points[1] for segment 0; P3 = 2*points[n-1]-points[n-2] for the last) —
// formed IN int64 (a mirrored phantom can exceed int32 at the |coord|<=2^29 budget edge; it is never
// stored in an fx). The mirror is the load-bearing convention: it makes the 2-point spline an exact lerp.
inline void SegControl(const Spline& s, int seg, int64_t P[4][3]) {
    const int n = (int)s.points.size();
    auto put = [&](int slot, const FxVec3& p) {
        P[slot][0] = (int64_t)p.x; P[slot][1] = (int64_t)p.y; P[slot][2] = (int64_t)p.z;
    };
    if (s.closed) {
        const int i0 = ((seg - 1) % n + n) % n;
        put(0, s.points[i0]);
        put(1, s.points[seg % n]);
        put(2, s.points[(seg + 1) % n]);
        put(3, s.points[(seg + 2) % n]);
        return;
    }
    put(1, s.points[seg]);
    put(2, s.points[seg + 1]);
    if (seg == 0) {  // mirrored start phantom: 2*P1 - P2 (int64 — may exceed int32)
        for (int a = 0; a < 3; ++a) P[0][a] = 2 * P[1][a] - P[2][a];
    } else {
        put(0, s.points[seg - 1]);
    }
    if (seg == n - 2) {  // mirrored end phantom: 2*P2 - P1
        for (int a = 0; a < 3; ++a) P[3][a] = 2 * P[2][a] - P[1][a];
    } else {
        put(3, s.points[seg + 2]);
    }
}

// EvalAxis: uniform Catmull-Rom on one axis, int64 Horner. Standard basis:
//   C(t) = 0.5 * ( 2*P1 + (P2-P0)*t + (2*P0-5*P1+4*P2-P3)*t^2 + (-P0+3*P1-3*P2+P3)*t^3 )
// Horner in Q16.16: acc = c3; acc = c2 + (acc*t >> 16); acc = c1 + (acc*t >> 16);
// acc = c0 + (acc*t >> 16); result = acc >> 1 (the 0.5, an arithmetic shift — pinned floor convention).
// EXACT KNOT INTERPOLATION: t=0 -> (2*P1)>>1 == P1 exactly; t=kOne -> every (acc*kOne)>>16 == acc, and
// c0+c1+c2+c3 collapses to 2*P2 (P0: -1+2-1=0, P1: 2-5+3=0, P2: 1+4-3=2, P3: -1+1=0) -> P2 exactly.
// EXACT 2-POINT LERP (with the mirrored phantoms P0=2A-B, P3=2B-A over span A->B):
//   c1 = B-(2A-B) = 2(B-A);  c2 = 2(2A-B)-5A+4B-(2B-A) = 0;  c3 = -(2A-B)+3A-3B+(2B-A) = 0
//   -> C(t) = (2A + 2(B-A)*t)/2 == A + (B-A)*t, and the integer shifts compose exactly:
//   (2A + ((2D*t)>>16))>>1 == A + ((D*t)>>16) (floor-division composition on two's-complement).
inline fx EvalAxis(const int64_t P[4], fx t) {
    const int64_t c0 = 2 * P[1];
    const int64_t c1 = P[2] - P[0];
    const int64_t c2 = 2 * P[0] - 5 * P[1] + 4 * P[2] - P[3];
    const int64_t c3 = -P[0] + 3 * P[1] - 3 * P[2] + P[3];
    int64_t acc = c3;
    acc = c2 + ((acc * (int64_t)t) >> kFrac);
    acc = c1 + ((acc * (int64_t)t) >> kFrac);
    acc = c0 + ((acc * (int64_t)t) >> kFrac);
    return (fx)(acc >> 1);
}

// TangentAxis: the analytic derivative dC/dt = 0.5 * (c1 + 2*c2*t + 3*c3*t^2), int64 Horner (same
// coefficient set, same >>16 discipline, same >>1). Q16.16 world-units per unit-t (UN-normalized).
inline fx TangentAxis(const int64_t P[4], fx t) {
    const int64_t c1 = P[2] - P[0];
    const int64_t c2 = 2 * P[0] - 5 * P[1] + 4 * P[2] - P[3];
    const int64_t c3 = -P[0] + 3 * P[1] - 3 * P[2] + P[3];
    int64_t acc = 3 * c3;
    acc = 2 * c2 + ((acc * (int64_t)t) >> kFrac);
    acc = c1 + ((acc * (int64_t)t) >> kFrac);
    return (fx)(acc >> 1);
}

// Eval(spline, segment, tQ): position on segment `segment` at local parameter tQ in [0,kOne] Q16.16.
// segment clamped to [0, SegmentCount-1], t clamped to [0,kOne] (deterministic out-of-range behaviour).
// n<2 -> points[0] or zero (degenerate guard).
inline FxVec3 Eval(const Spline& s, int segment, fx t) {
    const int segs = SegmentCount(s);
    if (segs == 0) return s.points.empty() ? FxVec3{0, 0, 0} : s.points[0];
    if (segment < 0) segment = 0;
    if (segment >= segs) segment = segs - 1;
    if (t < 0) t = 0;
    if (t > kOne) t = kOne;
    int64_t P[4][3];
    SegControl(s, segment, P);
    int64_t Px[4] = {P[0][0], P[1][0], P[2][0], P[3][0]};
    int64_t Py[4] = {P[0][1], P[1][1], P[2][1], P[3][1]};
    int64_t Pz[4] = {P[0][2], P[1][2], P[2][2], P[3][2]};
    return FxVec3{EvalAxis(Px, t), EvalAxis(Py, t), EvalAxis(Pz, t)};
}

// Tangent(spline, segment, tQ): the analytic (UN-normalized) derivative at (segment, t). Same clamps.
// On the 2-point line the tangent is EXACTLY (B-A), constant (c2=c3=0 -> (2(B-A))>>1).
inline FxVec3 Tangent(const Spline& s, int segment, fx t) {
    const int segs = SegmentCount(s);
    if (segs == 0) return FxVec3{0, 0, 0};
    if (segment < 0) segment = 0;
    if (segment >= segs) segment = segs - 1;
    if (t < 0) t = 0;
    if (t > kOne) t = kOne;
    int64_t P[4][3];
    SegControl(s, segment, P);
    int64_t Px[4] = {P[0][0], P[1][0], P[2][0], P[3][0]};
    int64_t Py[4] = {P[0][1], P[1][1], P[2][1], P[3][1]};
    int64_t Pz[4] = {P[0][2], P[1][2], P[2][2], P[3][2]};
    return FxVec3{TangentAxis(Px, t), TangentAxis(Py, t), TangentAxis(Pz, t)};
}

// ============== SP1-B — arc-length reparameterization (fixed-K chord table, integer sqrt) ==============

// The pinned convention: each segment is subdivided into kArcSamplesPerSeg = 16 UNIFORM-t chords; chord
// lengths come from FxLength (the fpx int64 integer sqrt); the table stores the CUMULATIVE length at every
// chord boundary (segCount*16 + 1 entries, cum[0] = 0, cum.back() = the total). EvalByDistance inverts the
// table by a deterministic PIECEWISE-LINEAR search (the seq.h FindSegment hand-written integer binary
// search — NO <algorithm>): distance s lands in chord k, the in-chord fraction is fxdiv, and the global
// parameter is reassembled exactly. The table is a QUANTIZATION (16 chords/segment): distances are
// polyline distances, not true arc lengths — the pinned, documented tradeoff (K is a constant, not a
// tunable, so every platform builds the identical table).
inline constexpr int kArcSamplesPerSeg = 16;

struct ArcTable {
    std::vector<fx> cum;   // cumulative chord length; size == SegmentCount*kArcSamplesPerSeg + 1
};

inline ArcTable BuildArcTable(const Spline& s) {
    ArcTable tab;
    const int segs = SegmentCount(s);
    if (segs == 0) { tab.cum.push_back(0); return tab; }
    tab.cum.reserve((size_t)segs * kArcSamplesPerSeg + 1);
    tab.cum.push_back(0);
    fx total = 0;
    FxVec3 prev = Eval(s, 0, 0);
    for (int seg = 0; seg < segs; ++seg) {
        for (int j = 1; j <= kArcSamplesPerSeg; ++j) {
            const fx t = (fx)(((int64_t)j << kFrac) / kArcSamplesPerSeg);   // exact kOne at j==K
            const FxVec3 p = Eval(s, seg, t);
            total += FxLength(FxSub(p, prev));                              // int64 integer sqrt chord
            tab.cum.push_back(total);
            prev = p;
        }
    }
    return tab;
}

inline fx ArcTotal(const ArcTable& t) { return t.cum.empty() ? 0 : t.cum.back(); }

// SplineParam: a (segment, t) pair — the currency between the arc table and the eval core.
struct SplineParam {
    int seg = 0;
    fx  t   = 0;
};

// ArcToParam(tab, sQ): invert the chord table at distance sQ (clamped to [0, total]). Hand-written
// integer binary search for the greatest chord k with cum[k] <= s (k < last), then the in-chord fraction
// t01 = fxdiv(s - cum[k], cum[k+1] - cum[k]) (zero-length chord -> 0, the deterministic guard), then the
// global parameter: seg = k / K, tQ = ((j << 16) + t01) / K with j = k % K (exact at chord boundaries).
// Monotonic by construction (cum is non-decreasing).
inline SplineParam ArcToParam(const ArcTable& tab, fx s) {
    SplineParam out;
    const size_t n = tab.cum.size();
    if (n < 2) return out;
    if (s <= 0) return out;
    if (s >= tab.cum[n - 1]) {
        out.seg = (int)((n - 2) / (size_t)kArcSamplesPerSeg);
        out.t   = (fx)(((((int64_t)((n - 2) % (size_t)kArcSamplesPerSeg)) << kFrac) + kOne) /
                       kArcSamplesPerSeg);
        return out;
    }
    // Greatest k with cum[k] <= s (s < cum.back() here, so k < n-1). The seq.h ceil-mid search.
    size_t lo = 0, hi = n - 2;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo + 1) / 2;
        if (tab.cum[mid] <= s) lo = mid;
        else                   hi = mid - 1;
    }
    const fx den = tab.cum[lo + 1] - tab.cum[lo];
    const fx t01 = (den == 0) ? 0 : fxdiv(s - tab.cum[lo], den);            // Q16.16 in [0, kOne]
    out.seg = (int)(lo / (size_t)kArcSamplesPerSeg);
    const int64_t j = (int64_t)(lo % (size_t)kArcSamplesPerSeg);
    out.t = (fx)(((j << kFrac) + (int64_t)t01) / kArcSamplesPerSeg);
    return out;
}

// EvalByDistance / TangentByDistance: the arc-length-true samplers every consumer below uses.
inline FxVec3 EvalByDistance(const Spline& s, const ArcTable& tab, fx dist) {
    const SplineParam p = ArcToParam(tab, dist);
    return Eval(s, p.seg, p.t);
}
inline FxVec3 TangentByDistance(const Spline& s, const ArcTable& tab, fx dist) {
    const SplineParam p = ArcToParam(tab, dist);
    return Tangent(s, p.seg, p.t);
}

// ================== SP1-C — spline-scatter (the open PC1 roadmap item; feeds hf::pcg) ==================

// FxCrossQ: integer cross product, each component an fxmul (int64 intermediate, >>16). Q16.16 in/out.
inline FxVec3 FxCrossQ(const FxVec3& a, const FxVec3& b) {
    return FxVec3{fxmul(a.y, b.z) - fxmul(a.z, b.y),
                  fxmul(a.z, b.x) - fxmul(a.x, b.z),
                  fxmul(a.x, b.y) - fxmul(a.y, b.x)};
}

// YawFromTangent: a yaw quaternion about +Y that rotates +Z onto the tangent's XZ direction — from the
// INTEGER HALF-ANGLE IDENTITY, no trig, no LUT: with the normalized XZ direction (dx, dz),
//   cos(theta) = dz, sin(theta) = dx
//   w = sqrt((kOne + dz)/2),  y = sign(dx) * sqrt((kOne - dz)/2)   (both via FxISqrt on Q32.32)
// Identity checks: dz=kOne -> {0,0,0,kOne} exactly; dz=-kOne (a 180-degree yaw) -> {0,kOne,0,0} exactly.
// The output is unit to +-1 LSB (integer sqrt floor), matching the kPcgYaw16 tolerance discipline.
// Degenerate (zero XZ tangent) -> identity.
inline FxQuat YawFromTangent(const FxVec3& tangent) {
    const FxVec3 flat{tangent.x, 0, tangent.z};
    const fx len = FxLength(flat);
    if (len == 0) return FxQuat{0, 0, 0, kOne};
    const fx dx = fxdiv(flat.x, len);
    fx dz = fxdiv(flat.z, len);
    if (dz > kOne)  dz = kOne;    // integer-normalize can land 1 LSB outside [-kOne, kOne]
    if (dz < -kOne) dz = -kOne;
    const fx hc = (kOne + dz) >> 1;                        // cos^2(theta/2) in Q16.16
    const fx hs = (kOne - dz) >> 1;                        // sin^2(theta/2) in Q16.16
    const fx w  = (fx)FxISqrt((int64_t)hc << kFrac);       // sqrt(Q16.16) via Q32.32 -> Q16.16
    fx y        = (fx)FxISqrt((int64_t)hs << kFrac);
    if (dx < 0) y = -y;                                    // sign(sin(theta)) = sign(dx); dx==0 -> +
    return FxQuat{0, y, 0, w};
}

// SideDir: the unit lateral direction at a point — normalize(tangent x up) with up = +Y, which for a
// tangent (tx,ty,tz) is the XZ perpendicular (-tz, 0, tx) renormalized. Degenerate (vertical tangent) ->
// FxNormalize's fixed fallback (0,kOne,0) — deterministic.
inline FxVec3 SideDir(const FxVec3& tangentDir) {
    return FxNormalize(FxVec3{-tangentDir.z, 0, tangentDir.x});
}

// ScatterAlongSpline: instances at fixed arc-length intervals — i-th instance at distance s = i*spacingQ
// for i in [0, total/spacing] — each offset LATERALLY (normal to the tangent) by offsetQ plus a seeded
// jitter draw PcgRandRange(stream, i, -jitterQ, +jitterQ) (jitterQ==0 -> exactly offsetQ, the no-op
// control: hi-lo==0 makes the draw 0 without branching). Output is pcg::PcgInstance{pos, yaw-from-tangent,
// scale=kOne} — the EXISTING pcg currency, so pcg::PruneOverlaps / pcg-driven render bridges compose
// UNCHANGED (the PC1 composition, not a reimplementation). spacingQ<=0 or an empty table -> empty.
inline std::vector<pcg::PcgInstance> ScatterAlongSpline(const Spline& sp, const ArcTable& tab,
                                                        fx spacingQ, fx offsetQ, fx jitterQ,
                                                        const pcg::PcgStream& stream) {
    std::vector<pcg::PcgInstance> out;
    if (spacingQ <= 0) return out;
    const fx total = ArcTotal(tab);
    if (total <= 0) return out;
    const uint32_t count = (uint32_t)(total / spacingQ) + 1u;   // s = 0, spacing, ..., <= total
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const fx s = (fx)((int64_t)i * (int64_t)spacingQ);
        const SplineParam p = ArcToParam(tab, s);
        FxVec3 pos = Eval(sp, p.seg, p.t);
        const FxVec3 tanDir = FxNormalize(Tangent(sp, p.seg, p.t));
        const FxVec3 side = SideDir(tanDir);
        const fx lat = offsetQ + pcg::PcgRandRange(stream, i, -jitterQ, jitterQ);
        pos = FxAdd(pos, FxScale(side, lat));
        out.push_back(pcg::PcgInstance{pos, YawFromTangent(tanDir), kOne});
    }
    return out;
}

// ===================== SP1-D — the spline-swept mesh strip (the road primitive) ========================

// The integer strip: per arc sample a LEFT and RIGHT vertex at pos -/+ side*(width/2), side =
// normalize(tangent x up). positions/normals/tangents are Q16.16 (bit-exact, digestable); indices are the
// pinned two-triangles-per-span winding. samples = positions.size()/2. The FLOAT crossing to the engine
// mesh layout is StripToMeshVertices below (the PCG6 render-bridge precedent).
struct SweptStrip {
    std::vector<FxVec3>   positions;   // 2 per sample: [2k] = left, [2k+1] = right
    std::vector<FxVec3>   normals;     // 2 per sample (== normalize(side x tangent); up for planar splines)
    std::vector<FxVec3>   tangents;    // 2 per sample (the normalized run direction)
    std::vector<uint32_t> indices;     // 6 per span: (a,b,c) (c,b,d) with a=2i b=2i+1 c=2i+2 d=2i+3 (wrapped if closed)
};

// SweepStrip(spline, widthQ, segmentsPerSpan): samples every segment at segmentsPerSpan uniform-t steps
// (open: plus the final t=kOne sample -> segCount*sps + 1 samples; closed: segCount*sps samples, spans
// wrap). At each sample: pos = Eval, tanDir = normalize(Tangent), side = SideDir(tanDir), halfW =
// widthQ>>1; left = pos - side*halfW, right = pos + side*halfW; normal = normalize(side x tangent) —
// EXACTLY (0,kOne,0) for a planar XZ spline. Winding is PINNED: span i emits (a,b,c) then (c,b,d) with
// a=2i, b=2i+1, c=2(i+1), d=2(i+1)+1 (mod vertex count when closed) — consistent orientation, no
// degenerate index triples by construction. COLLINEAR IDENTITY: collinear control points give an exactly
// straight strip (the tangent is exactly the line direction on every sample of an axis-aligned line, so
// left/right rails are exactly parallel lines). widthQ<=0 or segmentsPerSpan<1 or no segments -> empty.
inline SweptStrip SweepStrip(const Spline& sp, fx widthQ, int segmentsPerSpan) {
    SweptStrip strip;
    const int segs = SegmentCount(sp);
    if (segs == 0 || widthQ <= 0 || segmentsPerSpan < 1) return strip;
    const int sampleCount = segs * segmentsPerSpan + (sp.closed ? 0 : 1);
    const fx halfW = widthQ >> 1;
    strip.positions.reserve((size_t)sampleCount * 2);
    strip.normals.reserve((size_t)sampleCount * 2);
    strip.tangents.reserve((size_t)sampleCount * 2);
    for (int k = 0; k < sampleCount; ++k) {
        int seg = k / segmentsPerSpan;
        int j   = k % segmentsPerSpan;
        if (seg >= segs) { seg = segs - 1; j = segmentsPerSpan; }   // the open-spline final t=kOne sample
        const fx t = (fx)(((int64_t)j << kFrac) / segmentsPerSpan);
        const FxVec3 pos    = Eval(sp, seg, t);
        const FxVec3 tanDir = FxNormalize(Tangent(sp, seg, t));
        const FxVec3 side   = SideDir(tanDir);
        const FxVec3 nrm    = FxNormalize(FxCrossQ(side, tanDir));
        strip.positions.push_back(FxSub(pos, FxScale(side, halfW)));   // left
        strip.positions.push_back(FxAdd(pos, FxScale(side, halfW)));   // right
        strip.normals.push_back(nrm);
        strip.normals.push_back(nrm);
        strip.tangents.push_back(tanDir);
        strip.tangents.push_back(tanDir);
    }
    const int spanCount = sp.closed ? sampleCount : sampleCount - 1;
    strip.indices.reserve((size_t)spanCount * 6);
    for (int i = 0; i < spanCount; ++i) {
        const uint32_t a = (uint32_t)(2 * i);
        const uint32_t b = a + 1u;
        const uint32_t c = (uint32_t)(2 * ((i + 1) % sampleCount));
        const uint32_t d = c + 1u;
        strip.indices.push_back(a); strip.indices.push_back(b); strip.indices.push_back(c);
        strip.indices.push_back(c); strip.indices.push_back(b); strip.indices.push_back(d);
    }
    return strip;
}

// StripToMeshVertices — THE ONE FLOAT CROSSING of SP1 (the PCG6 PcgToRenderInstances precedent): convert
// the bit-exact integer strip into scene::Vertex (pos/color/uv/normal/tangent, the layout
// scene::MeshVertexLayout describes — stride 56, the shape SC3's glTF loader/meshlet consumers eat) for
// the EXISTING lit pipeline. uv: u = 0 (left rail) / 1 (right rail), v = the sample index (tiles per
// span). Deterministic host float (FxToFloat is a single divide by kOne). Render-only — NOT used by any
// bit-exact path. Pair with strip.indices UNCHANGED.
inline std::vector<scene::Vertex> StripToMeshVertices(const SweptStrip& strip, float r, float g, float b) {
    using hf::sim::fpx::FxToFloat;
    std::vector<scene::Vertex> out;
    out.reserve(strip.positions.size());
    for (size_t i = 0; i < strip.positions.size(); ++i) {
        scene::Vertex v{};
        v.pos[0] = FxToFloat(strip.positions[i].x);
        v.pos[1] = FxToFloat(strip.positions[i].y);
        v.pos[2] = FxToFloat(strip.positions[i].z);
        v.color[0] = r; v.color[1] = g; v.color[2] = b;
        v.uv[0] = (i & 1) ? 1.0f : 0.0f;
        v.uv[1] = (float)(i / 2);
        v.normal[0] = FxToFloat(strip.normals[i].x);
        v.normal[1] = FxToFloat(strip.normals[i].y);
        v.normal[2] = FxToFloat(strip.normals[i].z);
        v.tangent[0] = FxToFloat(strip.tangents[i].x);
        v.tangent[1] = FxToFloat(strip.tangents[i].y);
        v.tangent[2] = FxToFloat(strip.tangents[i].z);
        out.push_back(v);
    }
    return out;
}

// ========================== SP1-E — the spline camera track (rail camera) ==============================

// CameraAlongSpline(spline, tab, sQ): the rail-camera pose at arc distance sQ — pos on the curve,
// forward = the normalized tangent. Bit-exact integer; the float crossing to a view matrix is the
// caller's FxToFloat + math::Mat4::LookAt(pos, pos+forward, up) (render-only). EASING COMPOSES VIA seq.h:
// drive sQ from a seq::ScalarTrack (SampleScalar(track, tick) -> the eased distance) — the seq.h
// TransformTrack sibling, and because seq.h is lockstep-replayable the whole camera rail inherits the
// netcode story with ZERO new machinery (the stateless-consumer argument at the top of this header).
struct SplineCamera {
    FxVec3 pos;
    FxVec3 forward;   // normalized tangent (FxNormalize's (0,kOne,0) fallback on a degenerate tangent)
};

inline SplineCamera CameraAlongSpline(const Spline& sp, const ArcTable& tab, fx s) {
    return SplineCamera{EvalByDistance(sp, tab, s), FxNormalize(TangentByDistance(sp, tab, s))};
}

// ===================== SP1 showcase — the shared scenario + raster (WV1 pattern) =======================
// Header-local so BOTH showcase backends (Vulkan --sp1-road-shot / Metal --sp1-road) run the IDENTICAL
// bytes with ZERO copy drift: the fixed S-curve spline, the swept road strip, fence posts scattered along
// both edges (pruned via pcg — the composition proof), camera-track samples, one digest, one raster.

// The FIXED 6-point S-curve (XZ plane, y=0, roughly-even spacing — the documented uniform-CR authoring
// discipline). Keep FIXED forever — the pinned test digests + the golden hash it.
inline Spline MakeShowcaseSpline() {
    Spline s;
    s.closed = false;
    s.points = {
        FxVec3{-6 * kOne, 0, -3 * kOne},
        FxVec3{-4 * kOne, 0,  2 * kOne},
        FxVec3{-1 * kOne, 0, -2 * kOne},
        FxVec3{ 2 * kOne, 0,  3 * kOne},
        FxVec3{ 5 * kOne, 0, -1 * kOne},
        FxVec3{ 7 * kOne, 0,  3 * kOne},
    };
    return s;
}

inline constexpr fx  kShotRoadWidth   = kOne;               // 1.0 wu road (apex curvature bound)
inline constexpr int kShotSegsPerSpan = 8;                 // strip samples per spline segment
inline constexpr fx  kShotPostSpacing = kOne;              // a fence post every 1 wu of arc length
inline constexpr fx  kShotPostOffset  = 3 * kOne / 4;      // 0.75 wu lateral (0.25 outside the 0.5 half-width)
inline constexpr fx  kShotPostJitter  = kOne / 8;          // +-0.125 wu seeded lateral jitter
inline constexpr int kShotCamCount    = 9;                 // camera-track markers at i*total/8

struct SplineShotRun {
    Spline                        spline;
    ArcTable                      tab;
    SweptStrip                    strip;
    std::vector<pcg::PcgInstance> postsL, postsR;   // fence posts, both edges (post-prune)
    std::vector<SplineCamera>     cams;
    fx                            totalLen = 0;
    uint64_t                      digest   = 0;     // FNV-1a-64 over the field-serialized scenario
};

// RunSplineShotScenario: the pure function both backends call. Scatter composes with pcg::PruneOverlaps
// (radius kOne/8 — jitter can pinch neighbors, the prune keeps the canonical survivors) — the PC1
// compose-don't-reimplement proof. Digest is FIELD-BY-FIELD serialized fx/int32 (NEVER DigestBytes a
// struct — the seq.h S4 padding discipline).
inline SplineShotRun RunSplineShotScenario() {
    SplineShotRun run;
    run.spline = MakeShowcaseSpline();
    run.tab    = BuildArcTable(run.spline);
    run.totalLen = ArcTotal(run.tab);
    run.strip  = SweepStrip(run.spline, kShotRoadWidth, kShotSegsPerSpan);
    const pcg::PcgStream streamL{77u, 1u};
    const pcg::PcgStream streamR{77u, 2u};
    run.postsL = pcg::PruneOverlaps(
        ScatterAlongSpline(run.spline, run.tab, kShotPostSpacing, -kShotPostOffset, kShotPostJitter, streamL),
        kOne / 8);
    run.postsR = pcg::PruneOverlaps(
        ScatterAlongSpline(run.spline, run.tab, kShotPostSpacing, kShotPostOffset, kShotPostJitter, streamR),
        kOne / 8);
    run.cams.reserve((size_t)kShotCamCount);
    for (int i = 0; i < kShotCamCount; ++i) {
        const fx s = (fx)(((int64_t)i * (int64_t)run.totalLen) / (kShotCamCount - 1));
        run.cams.push_back(CameraAlongSpline(run.spline, run.tab, s));
    }
    // Field-by-field digest buffer (fx == int32; indices reinterpreted as int32 bytes).
    std::vector<fx> buf;
    buf.reserve(run.tab.cum.size() + run.strip.positions.size() * 3 + run.strip.indices.size() +
                (run.postsL.size() + run.postsR.size()) * 5 + run.cams.size() * 6 + 2);
    buf.push_back(run.totalLen);
    for (fx c : run.tab.cum) buf.push_back(c);
    for (const FxVec3& p : run.strip.positions) { buf.push_back(p.x); buf.push_back(p.y); buf.push_back(p.z); }
    for (uint32_t ix : run.strip.indices) buf.push_back((fx)ix);
    for (const pcg::PcgInstance& in : run.postsL) {
        buf.push_back(in.pos.x); buf.push_back(in.pos.y); buf.push_back(in.pos.z);
        buf.push_back(in.orient.y); buf.push_back(in.orient.w);
    }
    for (const pcg::PcgInstance& in : run.postsR) {
        buf.push_back(in.pos.x); buf.push_back(in.pos.y); buf.push_back(in.pos.z);
        buf.push_back(in.orient.y); buf.push_back(in.orient.w);
    }
    for (const SplineCamera& c : run.cams) {
        buf.push_back(c.pos.x); buf.push_back(c.pos.y); buf.push_back(c.pos.z);
        buf.push_back(c.forward.x); buf.push_back(c.forward.y); buf.push_back(c.forward.z);
    }
    buf.push_back((fx)run.strip.indices.size());
    run.digest = hf::net::DigestBytes(buf.data(), buf.size() * sizeof(fx));
    return run;
}

// RenderSplineShot: the PURE-INTEGER top-down raster (XZ plane; x -> px, z -> py) both backends call —
// strict-zero cross-backend BY CONSTRUCTION. World window x in [-8, 9], z in [-4.6, 5.4] at 40 px/wu ->
// 680x400 BGRA8. The road strip triangles are filled by an int64 edge-function rasterizer at integer
// pixel centers; rails/tangent ticks are integer DDA lines; posts/control points/camera markers are
// integer discs/squares/crosses. Fixed constant colors.
inline void RenderSplineShot(const SplineShotRun& run, std::vector<uint8_t>& bgra, uint32_t& outW,
                             uint32_t& outH) {
    const int W = 680, H = 400, kScale = 40;
    const fx xMin = -8 * kOne;
    const fx zMin = -(4 * kOne + kOne * 6 / 10);         // -4.6 wu
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {         // dark moss ground
        bgra[p * 4 + 0] = 14; bgra[p * 4 + 1] = 26; bgra[p * 4 + 2] = 16; bgra[p * 4 + 3] = 255;
    }
    auto putPx = [&](int ix, int iy, uint8_t r, uint8_t g, uint8_t b) {
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) return;
        uint8_t* dst = &bgra[((size_t)iy * W + ix) * 4];
        dst[0] = b; dst[1] = g; dst[2] = r; dst[3] = 255;
    };
    auto mapX = [&](fx x) { return (int)(((int64_t)(x - xMin) * kScale) >> kFrac); };
    auto mapY = [&](fx z) { return H - 1 - (int)(((int64_t)(z - zMin) * kScale) >> kFrac); };
    auto disc = [&](int cx, int cy, int rr, uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -rr; dy <= rr; ++dy)
            for (int dx = -rr; dx <= rr; ++dx)
                if (dx * dx + dy * dy <= rr * rr) putPx(cx + dx, cy + dy, r, g, b);
    };
    auto line = [&](int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
        const int dx = x1 - x0, dy = y1 - y0;
        int steps = dx >= 0 ? dx : -dx;
        const int ady = dy >= 0 ? dy : -dy;
        if (ady > steps) steps = ady;
        if (steps == 0) { putPx(x0, y0, r, g, b); return; }
        for (int i = 0; i <= steps; ++i) {               // integer DDA (rounded interpolation)
            const int px = x0 + (int)(((int64_t)dx * i * 2 + steps) / (2 * (int64_t)steps));
            const int py = y0 + (int)(((int64_t)dy * i * 2 + steps) / (2 * (int64_t)steps));
            putPx(px, py, r, g, b);
        }
    };
    // (1) The road strip: fill every triangle via int64 edge functions at pixel centers.
    auto edge = [](int64_t ax, int64_t ay, int64_t bx, int64_t by, int64_t px, int64_t py) {
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    };
    const size_t triCount = run.strip.indices.size() / 3;
    for (size_t t = 0; t < triCount; ++t) {
        const FxVec3& A = run.strip.positions[run.strip.indices[t * 3 + 0]];
        const FxVec3& B = run.strip.positions[run.strip.indices[t * 3 + 1]];
        const FxVec3& C = run.strip.positions[run.strip.indices[t * 3 + 2]];
        int ax = mapX(A.x), ay = mapY(A.z), bx = mapX(B.x), by = mapY(B.z), cx = mapX(C.x), cy = mapY(C.z);
        const int64_t area = edge(ax, ay, bx, by, cx, cy);
        if (area == 0) continue;
        if (area < 0) { int tx = bx, ty = by; bx = cx; by = cy; cx = tx; cy = ty; }   // canonical CCW
        int lox = ax < bx ? ax : bx; lox = lox < cx ? lox : cx; if (lox < 0) lox = 0;
        int hix = ax > bx ? ax : bx; hix = hix > cx ? hix : cx; if (hix >= W) hix = W - 1;
        int loy = ay < by ? ay : by; loy = loy < cy ? loy : cy; if (loy < 0) loy = 0;
        int hiy = ay > by ? ay : by; hiy = hiy > cy ? hiy : cy; if (hiy >= H) hiy = H - 1;
        for (int py = loy; py <= hiy; ++py)
            for (int px = lox; px <= hix; ++px)
                if (edge(ax, ay, bx, by, px, py) >= 0 && edge(bx, by, cx, cy, px, py) >= 0 &&
                    edge(cx, cy, ax, ay, px, py) >= 0)
                    putPx(px, py, 72, 74, 78);                                        // asphalt gray
    }
    // (2) The rails (left warm / right cool edge polylines) over the fill.
    const size_t samples = run.strip.positions.size() / 2;
    for (size_t k = 0; k + 1 < samples; ++k) {
        const FxVec3& l0 = run.strip.positions[k * 2],     &l1 = run.strip.positions[(k + 1) * 2];
        const FxVec3& r0 = run.strip.positions[k * 2 + 1], &r1 = run.strip.positions[(k + 1) * 2 + 1];
        line(mapX(l0.x), mapY(l0.z), mapX(l1.x), mapY(l1.z), 214, 196, 120);
        line(mapX(r0.x), mapY(r0.z), mapX(r1.x), mapY(r1.z), 120, 176, 214);
    }
    // (3) Fence posts: left warm-yellow, right sage-green discs.
    for (const pcg::PcgInstance& in : run.postsL) disc(mapX(in.pos.x), mapY(in.pos.z), 3, 226, 188, 84);
    for (const pcg::PcgInstance& in : run.postsR) disc(mapX(in.pos.x), mapY(in.pos.z), 3, 140, 200, 120);
    // (4) Control points: orange squares.
    for (const FxVec3& p : run.spline.points) {
        const int cx = mapX(p.x), cy = mapY(p.z);
        for (int dy = -3; dy <= 3; ++dy)
            for (int dx = -3; dx <= 3; ++dx) putPx(cx + dx, cy + dy, 232, 128, 58);
    }
    // (5) Camera-track markers: cyan crosses + a forward tick (0.5 wu along the tangent).
    for (const SplineCamera& c : run.cams) {
        const int cx = mapX(c.pos.x), cy = mapY(c.pos.z);
        for (int d = -4; d <= 4; ++d) { putPx(cx + d, cy, 96, 214, 214); putPx(cx, cy + d, 96, 214, 214); }
        const FxVec3 tip = FxAdd(c.pos, FxScale(c.forward, kOne / 2));
        line(cx, cy, mapX(tip.x), mapY(tip.z), 200, 240, 240);
    }
}

}  // namespace hf::spline
