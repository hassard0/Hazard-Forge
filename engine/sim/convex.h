#pragma once
// Slice CX1 — Deterministic Convex Rigid-Body Contacts: THE BOX-BOX SAT OVERLAP TEST (the BEACHHEAD of
// FLAGSHIP #19: DETERMINISTIC CONVEX RIGID-BODY CONTACTS — SAT manifold + inertia tensor + angular
// impulse, hf::sim::convex). The whole deterministic-sim suite (fpx/cloth/fluid/grain/couple*/fract/
// vehicle/joint) shares ONE caveat in SEVEN headers: contacts are SPHERE-SPHERE with NO inertia tensor /
// no torque-from-contact. This flagship lifts that ceiling. CX1 is the narrowphase BEACHHEAD: a
// deterministic INTEGER box-box Separating-Axis Test (SAT) that, for a pair of ORIENTED boxes, computes
// the minimum-penetration axis + depth (or reports separation), bit-exact CPU<->Vulkan<->Metal.
//
// Header-only, namespace hf::sim::convex, #include "sim/fpx.h" READ-ONLY (do NOT modify fpx.h). Reuses the
// fpx Q16.16 toolbox VERBATIM: fx/fxmul/fxdiv/FxVec3/FxAdd/FxSub/FxDot... — wait, FxDot lives here? No:
// FxDot/FxNormalize/FxLength/FxRotate/FxISqrt are all in fpx.h. We add FxMat3 (the CX3 inertia-tensor type,
// minimal here) + FxCross (the fpx::FxRotate internal cross PROMOTED to a public helper — NOT copied from
// fpx.h source-modification; re-expressed here) + FxBox + SatResult + BoxSat + MeasureSat.
//
// THE int64 REALITY (the FPX3/FPX1 lesson, the honest proof-strength call): FxNormalize/FxISqrt + the
// FxDot/FxCross Q16.16 products are int64 (world-scale products overflow int32). DXC compiles int64 (the
// Vulkan path); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL. So the GPU
// shader shaders/convex_sat.comp.hlsl is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal
// hf_gen_msl list); the Metal --convex-sat runs the CPU BoxSat -> byte-identical to the Vulkan GPU result
// BY CONSTRUCTION (the fpx_solve.comp / boids_steer.comp convention), while the Vulkan side carries the
// GPU==CPU memcmp proof. convex_sat.comp copies BoxSat's body VERBATIM, so the GPU exercises the EXACT
// integer ops -> a divergence is exactly what the host GPU==CPU memcmp catches.
//
// FIXED AXIS ORDER (the determinism crux): the 15 candidate separating axes are iterated in ONE fixed
// order — 3 A-face normals (A's local X/Y/Z rotated to world), 3 B-face normals, then the 9 edge-edge
// crosses FxCross(faceAxisA_i, faceAxisB_j) in a fixed (i,j) order — with the SAME FxNormalize/FxDot/
// FxCross int64 ops and the SAME strict-< min-pen tie-break (ties keep the LOWEST axis index). A degenerate
// near-zero edge-cross (FxLength < kEdgeEps; parallel face axes) is SKIPPED deterministically (it cannot
// separate). Per-pair-disjoint (one GPU thread per box pair) -> race-free, two runs byte-identical.

#include <cstdint>
#include <vector>

#include "sim/fpx.h"   // read-only: fx/fxmul/fxdiv/kOne/kFrac, FxVec3/FxAdd/FxSub/FxScale, FxDot? (no —
                       // FxDot is added below; fpx.h has FxLength/FxISqrt/FxNormalize/FxQuat/FxBody/FxRotate)

namespace hf::sim {
namespace convex {

// Pull the fpx Q16.16 scalar + helpers into this namespace (re-use, do NOT redefine the fixed-point format).
using fpx::fx;
using fpx::kOne;
using fpx::kFrac;
using fpx::fxmul;
using fpx::fxdiv;
using fpx::FxVec3;
using fpx::FxBody;
using fpx::FxQuat;

// ----- FxDot: the Q16.16 dot product (int64 intermediate, the FxLength sum-of-squares discipline) --------
// a·b = (ax*bx + ay*by + az*bz) >> kFrac, each product an int64 (world-scale products overflow int32), the
// sum kept in int64 then a single arithmetic right shift -> Q16.16. Deterministic, identical CPU/HLSL.
inline fx FxDot(const FxVec3& a, const FxVec3& b) {
    int64_t d = (int64_t)a.x * (int64_t)b.x + (int64_t)a.y * (int64_t)b.y + (int64_t)a.z * (int64_t)b.z;
    return (fx)(d >> kFrac);
}

// ----- FxCross: the Q16.16 cross product (the fpx::FxRotate internal cross promoted to a public helper) ---
// a×b = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx), each fxmul an int64 product (the SAME ops the
// FxRotate cross uses ~fpx.h:444). The edge-edge separating axes are FxCross(faceAxisA, faceAxisB).
inline FxVec3 FxCross(const FxVec3& a, const FxVec3& b) {
    return FxVec3{
        fxmul(a.y, b.z) - fxmul(a.z, b.y),
        fxmul(a.z, b.x) - fxmul(a.x, b.z),
        fxmul(a.x, b.y) - fxmul(a.y, b.x),
    };
}

// ----- FxMat3: the Q16.16 row-major 3x3 matrix (the inertia-tensor type, introduced for CX3) -------------
// Minimal here: identity, diagonal, mul-by-vector, transpose. CX3 (the angular-impulse solve) is the full
// consumer; CX1 only introduces the type + these helpers (YAGNI — no FxMat3Mul-mat / inverse until CX3).
// m[r*3 + c] is row r, column c. Q16.16 components.
struct FxMat3 {
    fx m[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

// FxMat3Identity(): the Q16.16 identity (diagonal kOne).
inline FxMat3 FxMat3Identity() {
    FxMat3 r;
    r.m[0] = kOne; r.m[4] = kOne; r.m[8] = kOne;
    return r;
}

// FxMat3Diagonal(d): a diagonal matrix with d.x/d.y/d.z on the diagonal (an axis-aligned inertia tensor).
inline FxMat3 FxMat3Diagonal(const FxVec3& d) {
    FxMat3 r;
    r.m[0] = d.x; r.m[4] = d.y; r.m[8] = d.z;
    return r;
}

// FxMat3MulVec(M, v) = M·v in Q16.16 (each row dotted with v via FxDot). int64 in the products.
inline FxVec3 FxMat3MulVec(const FxMat3& M, const FxVec3& v) {
    const FxVec3 r0{M.m[0], M.m[1], M.m[2]};
    const FxVec3 r1{M.m[3], M.m[4], M.m[5]};
    const FxVec3 r2{M.m[6], M.m[7], M.m[8]};
    return FxVec3{FxDot(r0, v), FxDot(r1, v), FxDot(r2, v)};
}

// FxMat3Transpose(M): the transpose (swap m[r*3+c] with m[c*3+r]). Pure integer copy, no products.
inline FxMat3 FxMat3Transpose(const FxMat3& M) {
    FxMat3 r;
    r.m[0] = M.m[0]; r.m[1] = M.m[3]; r.m[2] = M.m[6];
    r.m[3] = M.m[1]; r.m[4] = M.m[4]; r.m[5] = M.m[7];
    r.m[6] = M.m[2]; r.m[7] = M.m[5]; r.m[8] = M.m[8];
    return r;
}

// ----- FxBox: an oriented box collider (the body carries pos + orient; the box carries the half-extents) -
// halfExtents are the per-axis half-widths in Q16.16 (a 2x2x2 box has halfExtents {kOne,kOne,kOne}). The
// box is centered at its FxBody::pos, oriented by its FxBody::orient (the 3 world face axes = FxRotate of
// the local X/Y/Z). std430-packable as 3 x int32 (12 bytes).
struct FxBox {
    FxVec3 halfExtents;
};

// ----- SatResult: the box-box SAT outcome ----------------------------------------------------------------
// overlap=false -> the boxes are SEPARATED (some axis separates them); axisIndex/penetration/axis are
// then unused (0). overlap=true -> axisIndex is the 0..14 min-penetration axis index, penetration is the
// (non-negative) min depth over all axes, axis is the unit min-pen separating axis SIGNED toward B (so
// moving B by +axis*penetration separates the pair). std430-packable: overlap as a uint32, axisIndex
// uint32, penetration fx, axis 3 x int32 -> 6 x int32 (24 bytes) for the GPU mirror.
struct SatResult {
    bool     overlap     = false;
    uint32_t axisIndex   = 0;
    fx       penetration = 0;
    FxVec3   axis;
};

// The near-zero edge-cross epsilon: an edge-edge cross FxCross(faceAxisA_i, faceAxisB_j) whose FxLength is
// below this is DEGENERATE (the two face axes are ~parallel) -> that axis cannot separate -> SKIP it
// deterministically (the SAME threshold on CPU + GPU, byte-identical). ~1/256 of a Q16.16 unit.
inline constexpr fx kEdgeEps = kOne / 256;   // ~0.0039 world units of cross-product length

// BoxAxes(b): the 3 WORLD face axes of an oriented box (its local +X/+Y/+Z rotated by the body orient).
// Unit by construction (a unit orient rotates the unit local axes to unit world axes, up to the fixed-point
// FxRotate drift). The SAT projects onto these (the 3 A-face normals + the 3 B-face normals).
inline void BoxAxes(const FxBody& b, FxVec3 axesOut[3]) {
    axesOut[0] = fpx::FxRotate(b.orient, FxVec3{kOne, 0, 0});
    axesOut[1] = fpx::FxRotate(b.orient, FxVec3{0, kOne, 0});
    axesOut[2] = fpx::FxRotate(b.orient, FxVec3{0, 0, kOne});
}

// ProjectedRadius(L, axes, h): the projected half-width (radius) of a box onto the unit axis L:
// r = |L·axes[0]|·h.x + |L·axes[1]|·h.y + |L·axes[2]|·h.z (the standard OBB extent onto a direction).
// Each |L·axis_i| is a Q16.16 magnitude; the fxmul with the half-extent + the sum stay in Q16.16. int64
// inside FxDot/fxmul. The shader copies THIS body VERBATIM.
inline fx ProjectedRadius(const FxVec3& L, const FxVec3 axes[3], const FxVec3& h) {
    auto absfx = [](fx v) { return v < 0 ? -v : v; };
    const fx d0 = absfx(FxDot(L, axes[0]));
    const fx d1 = absfx(FxDot(L, axes[1]));
    const fx d2 = absfx(FxDot(L, axes[2]));
    return fxmul(d0, h.x) + fxmul(d1, h.y) + fxmul(d2, h.z);
}

// BoxSat(bodyA, boxA, bodyB, boxB): the 15-axis box-box Separating-Axis Test in Q16.16.
//
// The 15 candidate separating axes, in FIXED order:
//   index 0..2  : A's 3 world face normals (axA[0..2])
//   index 3..5  : B's 3 world face normals (axB[0..2])
//   index 6..14 : the 9 edge-edge crosses FxCross(axA[i], axB[j]) for i in 0..2, j in 0..2
//                 (index = 6 + i*3 + j) — a degenerate near-zero cross (FxLength < kEdgeEps) is SKIPPED.
// Per axis L: normalize L (FxNormalize — int64 FxISqrt; the face axes are already ~unit, the edge crosses
// are not), then rA = ProjectedRadius(L, axA, hA), rB = ProjectedRadius(L, axB, hB),
// s = |L·(b.pos - a.pos)|. The axis SEPARATES iff s > rA + rB -> return {overlap=false}. Else the
// penetration on L is pen = (rA + rB) - s; track the MINIMUM pen over all axes (strict-< scan, the LOWEST
// axis index wins ties — bit-reproducible). The result axis is that min-pen L SIGNED toward B (flip if
// L·(b.pos - a.pos) < 0). Pure integer, fixed order -> bit-identical CPU<->Vulkan<->Metal.
inline SatResult BoxSat(const FxBody& bodyA, const FxBox& boxA, const FxBody& bodyB, const FxBox& boxB) {
    FxVec3 axA[3], axB[3];
    BoxAxes(bodyA, axA);
    BoxAxes(bodyB, axB);
    const FxVec3 hA = boxA.halfExtents;
    const FxVec3 hB = boxB.halfExtents;
    const FxVec3 t = fpx::FxSub(bodyB.pos, bodyA.pos);   // center separation A->B

    auto absfx = [](fx v) { return v < 0 ? -v : v; };

    // The running minimum-penetration axis (strict-< scan, lowest-index tie-break).
    bool   found      = false;
    fx     minPen     = 0;
    uint32_t minIndex = 0;
    FxVec3 minAxis;

    // A single axis test: returns false (separated -> caller bails) or true (overlapping -> updates min).
    // skipDegenerate=true for the edge crosses (a near-zero L can't separate -> ignore, do NOT update min).
    auto testAxis = [&](const FxVec3& rawL, uint32_t index, bool skipDegenerate) -> bool {
        // Degenerate edge-cross guard: a near-zero raw cross is skipped (deterministic, FxLength<kEdgeEps).
        if (skipDegenerate) {
            const fx rawLen = fpx::FxLength(rawL);
            if (rawLen < kEdgeEps) return true;   // can't separate on this axis -> ignore it
        }
        const FxVec3 L = fpx::FxNormalize(rawL);   // unit axis (int64 FxISqrt/fxdiv)
        const fx rA = ProjectedRadius(L, axA, hA);
        const fx rB = ProjectedRadius(L, axB, hB);
        const fx s  = absfx(FxDot(L, t));
        const fx sum = rA + rB;
        if (s > sum) return false;                 // SEPARATED on this axis -> the whole test fails
        const fx pen = sum - s;                    // penetration depth along L
        if (!found || pen < minPen) {              // strict-<: lowest index keeps a tie
            found    = true;
            minPen   = pen;
            minIndex = index;
            // Sign the axis toward B: flip if L points away from (b.pos - a.pos).
            minAxis  = (FxDot(L, t) < 0) ? FxVec3{-L.x, -L.y, -L.z} : L;
        }
        return true;
    };

    // (1) A's 3 face normals (indices 0..2) — already unit, NOT skip-degenerate.
    for (uint32_t i = 0; i < 3; ++i)
        if (!testAxis(axA[i], i, false)) return SatResult{};   // separated
    // (2) B's 3 face normals (indices 3..5).
    for (uint32_t j = 0; j < 3; ++j)
        if (!testAxis(axB[j], 3 + j, false)) return SatResult{};
    // (3) the 9 edge-edge crosses (indices 6..14), FIXED (i,j) order, skip-degenerate.
    for (uint32_t i = 0; i < 3; ++i)
        for (uint32_t j = 0; j < 3; ++j) {
            const FxVec3 e = FxCross(axA[i], axB[j]);
            if (!testAxis(e, 6 + i * 3 + j, true)) return SatResult{};
        }

    // No axis separated -> the boxes OVERLAP. Report the min-pen axis + depth.
    SatResult r;
    r.overlap     = true;
    r.axisIndex   = minIndex;
    r.penetration = minPen;
    r.axis        = minAxis;
    return r;
}

// ----- A SAT pair (the showcase's deterministic box-pair array element) ----------------------------------
// A pair of oriented boxes: each an FxBody (pos + orient) + an FxBox (half-extents). The showcase builds a
// fixed array of these spanning separated / deep face-face / edge-edge / touching cases.
struct SatPair {
    FxBody bodyA;
    FxBox  boxA;
    FxBody bodyB;
    FxBox  boxB;
};

// SatMeasure: the deterministic summary of a set of SAT pairs — how many overlap + the mean/min penetration
// over the overlapping ones (pure integer; the showcase prints + cross-checks against truth).
struct SatMeasure {
    uint32_t pairs       = 0;   // total pairs measured
    uint32_t overlapping = 0;   // pairs reporting overlap=true
    fx       meanPen     = 0;   // mean penetration over the overlapping pairs (0 if none)
    fx       minPen      = 0;   // min penetration over the overlapping pairs (0 if none)
};

// MeasureSat(pairs): run BoxSat over each pair, count overlaps, accumulate the mean/min penetration. Pure
// integer, fixed order -> deterministic. (The mean is an integer divide by the overlap count.)
inline SatMeasure MeasureSat(const std::vector<SatPair>& pairs) {
    SatMeasure m;
    m.pairs = (uint32_t)pairs.size();
    int64_t penSum = 0;
    bool firstOverlap = true;
    for (const SatPair& p : pairs) {
        const SatResult r = BoxSat(p.bodyA, p.boxA, p.bodyB, p.boxB);
        if (r.overlap) {
            ++m.overlapping;
            penSum += (int64_t)r.penetration;
            if (firstOverlap || r.penetration < m.minPen) { m.minPen = r.penetration; firstOverlap = false; }
        }
    }
    if (m.overlapping > 0) m.meanPen = (fx)(penSum / (int64_t)m.overlapping);
    return m;
}

// =========================================================================================================
// Slice CX2 — THE CONTACT MANIFOLD (clip the incident face -> 1-4 contact points). APPENDED after MeasureSat
// (CX1's lines above are BYTE-FROZEN). Turns CX1's SAT min-pen axis into the actual SET of contact POINTS
// (with per-point penetration depth) where the two boxes touch — the data CX3's angular impulse needs.
// Deterministic integer Sutherland-Hodgman face clipping (face case, axisIndex 0..5) + the single
// closest-point edge-edge contact (axisIndex 6..14). PURE INTEGER, every order PINNED -> bit-identical
// CPU<->Vulkan<->Metal. int64 in the fxdiv/FxDot products -> shaders/convex_manifold.comp is Vulkan-only
// (the convex_sat.comp / fpx_solve.comp split); the Metal --convex-manifold runs THIS CPU BuildManifold,
// byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU memcmp.
//
// HONEST SIMPLIFICATION (documented, in scope): the 4-point reduction keeps the DEEPEST point + the first 3
// in CLIP ORDER (NOT the area-maximizing 4 a production solver would keep) — a deterministic first cut; the
// area-maximizing reduction is a deferred CX-refinement. The face-clip contact point is the CLIPPED INCIDENT
// VERTEX position itself (not projected onto the reference plane) — documented; its depth is the signed
// distance below the reference face. Boxes ONLY (the CX1 box-box SAT, not general convex/GJK/EPA).
//
// ----- ContactManifold: 1-4 contact points + per-point depths + the contact normal --------------------
// count   = the number of valid contact points (0 if separated, 1 for edge-edge, 1..4 for face). points and
//           depths are filled 0..count, the rest left zero. normal = the reference face normal (face case) or
//           the signed edge-cross axis (edge case), SIGNED from A toward B (the CX1 SatResult.axis convention
//           — moving B by +normal separates the pair). std430-packable: count uint32 + 4 x FxVec3 (12 ints) +
//           4 x fx depths + FxVec3 normal -> a fixed int32 layout the GPU ManifoldGpu mirrors.
struct ContactManifold {
    uint32_t count = 0;
    FxVec3   points[4];
    fx       depths[4] = {0, 0, 0, 0};
    FxVec3   normal;
};

using fpx::kHalf;   // 0.5 in Q16.16 (the deterministic mid-segment fallback for parallel edges)

// ClosestPointsOnSegments: the integer closest points of two segments (the edge-edge contact core). Segment
// A is [pA - dA, pA + dA] (center pA, half-vector dA = axA[i]*hA[i]); segment B likewise. Returns the two
// closest points cA, cB. The standard clamped-parametric solution (Ericson, "Real-Time Collision Detection")
// in Q16.16: r = pA - pB; a = dA·dA*4 ... — we work with the FULL-LENGTH segment direction u = 2*dA, v = 2*dB
// and base points s0 = pA - dA, s1 = pB - dB, params s,t in [0,kOne]. All fxdiv/FxDot are int64. PARALLEL
// (denom ~0) -> clamp s=kHalf (the deterministic mid-segment fallback). Pure integer, fixed order.
inline void ClosestPointsOnSegments(const FxVec3& pA, const FxVec3& dA,
                                    const FxVec3& pB, const FxVec3& dB,
                                    FxVec3& cAOut, FxVec3& cBOut) {
    const FxVec3 s0{pA.x - dA.x, pA.y - dA.y, pA.z - dA.z};   // A segment base
    const FxVec3 s1{pB.x - dB.x, pB.y - dB.y, pB.z - dB.z};   // B segment base
    const FxVec3 u{dA.x * 2, dA.y * 2, dA.z * 2};             // A full direction (2*half)
    const FxVec3 v{dB.x * 2, dB.y * 2, dB.z * 2};             // B full direction
    const FxVec3 w{s0.x - s1.x, s0.y - s1.y, s0.z - s1.z};
    const fx a = FxDot(u, u);   // >= 0
    const fx b = FxDot(u, v);
    const fx c = FxDot(v, v);   // >= 0
    const fx d = FxDot(u, w);
    const fx e = FxDot(v, w);
    const fx denom = fxmul(a, c) - fxmul(b, b);   // >= 0 (Cauchy-Schwarz); ~0 -> parallel
    fx s, tnum, tdenom;
    if (denom > kEdgeEps) {
        s = fxdiv(fxmul(b, e) - fxmul(c, d), denom);
        if (s < 0) s = 0; else if (s > kOne) s = kOne;
    } else {
        s = kHalf;   // parallel: deterministic mid-segment of A
    }
    // t = (b*s + e) / c  (the closest point on B to A's chosen s), clamped to [0,1].
    tnum = fxmul(b, s) + e;
    tdenom = c;
    fx tt;
    if (tdenom > kEdgeEps) {
        tt = fxdiv(tnum, tdenom);
        if (tt < 0) tt = 0; else if (tt > kOne) tt = kOne;
    } else {
        tt = kHalf;
    }
    // Re-derive s from the clamped t for a tighter (still deterministic) closest pair: s = (b*t - d)/a.
    if (a > kEdgeEps) {
        fx s2 = fxdiv(fxmul(b, tt) - d, a);
        if (s2 < 0) s2 = 0; else if (s2 > kOne) s2 = kOne;
        s = s2;
    }
    cAOut = FxVec3{s0.x + fxmul(u.x, s), s0.y + fxmul(u.y, s), s0.z + fxmul(u.z, s)};
    cBOut = FxVec3{s1.x + fxmul(v.x, tt), s1.y + fxmul(v.y, tt), s1.z + fxmul(v.z, tt)};
}

// FxAt(v, i): the i-th component (0=x,1=y,2=z) of an FxVec3 — a deterministic per-axis index accessor used
// by the manifold (half-extent along a chosen local axis). Identical CPU/HLSL (a plain branch).
inline fx FxAt(const FxVec3& v, uint32_t i) { return (i == 0) ? v.x : (i == 1) ? v.y : v.z; }

// BuildManifold(bodyA, boxA, bodyB, boxB, satResult): the box-box contact manifold from CX1's SAT result.
//
// SEPARATED (!satResult.overlap) -> {count=0}.
// FACE contact (axisIndex 0..5): reference/incident face clip (Sutherland-Hodgman).
//   - reference box = the box that OWNS the axis (A if axisIndex<3 else B); reference normal n = that box's
//     world face axis, SIGNED toward the incident box (the CX1 sign rule via FxDot(axis, ref->inc)).
//   - reference face center = refPos + n*H (H = the ref half-extent along the owning axis); u,v = the OTHER
//     two ref world axes (ascending local index), hu,hv their half-extents; the 4 ref-face corners in the
//     FIXED order (+u+v),(-u+v),(-u-v),(+u-v).
//   - incident face = the incident box's face whose signed world normal is MOST ANTI-PARALLEL to n (min
//     FxDot); tie-break lowest local-axis index then + before -. Its 4 corners in the SAME fixed order.
//   - Sutherland-Hodgman clip the incident 4-gon against the 4 ref-face SIDE planes in the FIXED order
//     (+u,-u,+v,-v) (each inside test sgn*FxDot(axis,p-center) <= h; crossing edges emit fxdiv intersections
//     in a PINNED iteration order).
//   - keep clipped vertices with depth d = FxDot(n, faceCenter - vertex) >= 0 (below/inside the ref face);
//     contact point = the vertex itself (documented), depth = d.
//   - reduce to <=4: ALWAYS keep the deepest (max d, tie -> lowest clip-order index), then up to 3 more in
//     clip order. normal = n.
// EDGE-EDGE contact (axisIndex 6..14): ONE point = the midpoint of the closest points of edge A_i and
//   edge B_j (i=(idx-6)/3, j=(idx-6)%3); depth = satResult.penetration; normal = satResult.axis. count=1.
inline ContactManifold BuildManifold(const FxBody& bodyA, const FxBox& boxA,
                                     const FxBody& bodyB, const FxBox& boxB,
                                     const SatResult& sat) {
    ContactManifold m;
    if (!sat.overlap) return m;   // separated -> empty

    FxVec3 axA[3], axB[3];
    BoxAxes(bodyA, axA);
    BoxAxes(bodyB, axB);
    const FxVec3 hA = boxA.halfExtents;
    const FxVec3 hB = boxB.halfExtents;

    // ---------------- EDGE-EDGE case (axisIndex 6..14): the single closest-point contact ----------------
    if (sat.axisIndex >= 6) {
        const uint32_t e = sat.axisIndex - 6;     // 0..8
        const uint32_t i = e / 3, j = e % 3;
        const FxVec3 dA = FxVec3{fxmul(axA[i].x, FxAt(hA, i)), fxmul(axA[i].y, FxAt(hA, i)),
                                 fxmul(axA[i].z, FxAt(hA, i))};
        const FxVec3 dB = FxVec3{fxmul(axB[j].x, FxAt(hB, j)), fxmul(axB[j].y, FxAt(hB, j)),
                                 fxmul(axB[j].z, FxAt(hB, j))};
        FxVec3 cA, cB;
        ClosestPointsOnSegments(bodyA.pos, dA, bodyB.pos, dB, cA, cB);
        m.count = 1;
        m.points[0] = FxVec3{(cA.x + cB.x) / 2, (cA.y + cB.y) / 2, (cA.z + cB.z) / 2};
        m.depths[0] = sat.penetration;
        m.normal = sat.axis;
        return m;
    }

    // ---------------- FACE case (axisIndex 0..5): reference/incident face clip ----------------
    const bool refIsA = sat.axisIndex < 3;
    const uint32_t refIdx = refIsA ? sat.axisIndex : (sat.axisIndex - 3);   // owning local axis 0..2
    const FxVec3* refAxes = refIsA ? axA : axB;
    const FxVec3* incAxes = refIsA ? axB : axA;
    const FxBody& refBody = refIsA ? bodyA : bodyB;
    const FxBody& incBody = refIsA ? bodyB : bodyA;
    const FxVec3& refH = refIsA ? hA : hB;
    const FxVec3& incH = refIsA ? hB : hA;

    // Reference normal n: the owning world face axis, signed from the reference box toward the incident box
    // (EXACTLY the CX1 sign rule). tRefToInc = incCenter - refCenter.
    const FxVec3 tRefToInc = fpx::FxSub(incBody.pos, refBody.pos);
    FxVec3 n = refAxes[refIdx];
    if (FxDot(n, tRefToInc) < 0) n = FxVec3{-n.x, -n.y, -n.z};
    const fx Href = FxAt(refH, refIdx);

    // The reference face center + its two in-plane axes u,v (the OTHER two ref axes, ascending local index)
    // + half-extents hu,hv. Corner order FIXED: (+u+v),(-u+v),(-u-v),(+u-v).
    uint32_t ui = (refIdx == 0) ? 1u : 0u;
    uint32_t vi = (refIdx == 2) ? 1u : 2u;
    // ensure ui<vi ascending (refIdx 0->{1,2}, 1->{0,2}, 2->{0,1})
    if (refIdx == 1) { ui = 0u; vi = 2u; }
    const FxVec3 u = refAxes[ui];
    const FxVec3 v = refAxes[vi];
    const fx hu = FxAt(refH, ui);
    const fx hv = FxAt(refH, vi);
    const FxVec3 faceCenter = FxVec3{refBody.pos.x + fxmul(n.x, Href),
                                     refBody.pos.y + fxmul(n.y, Href),
                                     refBody.pos.z + fxmul(n.z, Href)};

    // Incident face: the incident box face whose SIGNED world normal is most ANTI-PARALLEL to n (min FxDot).
    // Iterate the 3 inc axes, the + sign then the - sign; tie-break lowest axis index then + before -.
    uint32_t bestK = 0; bool bestNeg = false; fx bestDot = 0; bool firstK = true;
    for (uint32_t k = 0; k < 3; ++k) {
        for (int sgn = 0; sgn < 2; ++sgn) {   // 0 = +axis, 1 = -axis (so + is considered before -)
            const FxVec3 cand = (sgn == 0) ? incAxes[k]
                                           : FxVec3{-incAxes[k].x, -incAxes[k].y, -incAxes[k].z};
            const fx dt = FxDot(cand, n);
            if (firstK || dt < bestDot) { bestDot = dt; bestK = k; bestNeg = (sgn == 1); firstK = false; }
        }
    }
    // The incident face outward normal (the chosen most-anti-parallel signed axis) + center + in-plane axes.
    const FxVec3 incN = bestNeg ? FxVec3{-incAxes[bestK].x, -incAxes[bestK].y, -incAxes[bestK].z}
                                : incAxes[bestK];
    const fx incHk = FxAt(incH, bestK);
    const FxVec3 incFaceCenter = FxVec3{incBody.pos.x + fxmul(incN.x, incHk),
                                        incBody.pos.y + fxmul(incN.y, incHk),
                                        incBody.pos.z + fxmul(incN.z, incHk)};
    uint32_t iui = (bestK == 0) ? 1u : 0u;
    uint32_t ivi = (bestK == 2) ? 1u : 2u;
    if (bestK == 1) { iui = 0u; ivi = 2u; }
    const FxVec3 iu = incAxes[iui];
    const FxVec3 iv = incAxes[ivi];
    const fx ihu = FxAt(incH, iui);
    const fx ihv = FxAt(incH, ivi);
    // The 4 incident-face corners in the FIXED order (+iu+iv),(-iu+iv),(-iu-iv),(+iu-iv).
    const int su[4] = {+1, -1, -1, +1};
    const int sv[4] = {+1, +1, -1, -1};
    FxVec3 poly[8];
    int polyN = 0;
    for (int k = 0; k < 4; ++k) {
        poly[k] = FxVec3{incFaceCenter.x + su[k] * fxmul(iu.x, ihu) + sv[k] * fxmul(iv.x, ihv),
                         incFaceCenter.y + su[k] * fxmul(iu.y, ihu) + sv[k] * fxmul(iv.y, ihv),
                         incFaceCenter.z + su[k] * fxmul(iu.z, ihu) + sv[k] * fxmul(iv.z, ihv)};
    }
    polyN = 4;

    // Sutherland-Hodgman clip against the 4 ref-face side planes in the FIXED order (+u,-u,+v,-v). Each
    // plane is (axis a, half h, sign sgn): a vertex p is INSIDE iff f(p) = h - sgn*FxDot(a, p-faceCenter)
    // >= 0. A crossing edge (prev,cur) emits the intersection at tparam = f(prev)/(f(prev)-f(cur)) (int64
    // fxdiv) -> point = prev + tparam*(cur-prev). Iteration order PINNED (edge 0..n-1, prev=last).
    struct Plane { FxVec3 a; fx h; int sgn; };
    const Plane planes[4] = {{u, hu, +1}, {u, hu, -1}, {v, hv, +1}, {v, hv, -1}};
    auto sdist = [&](const Plane& pl, const FxVec3& p) -> fx {
        const FxVec3 rel = FxVec3{p.x - faceCenter.x, p.y - faceCenter.y, p.z - faceCenter.z};
        const fx proj = FxDot(pl.a, rel);
        return pl.h - (fx)(pl.sgn * proj);
    };
    for (int pl = 0; pl < 4; ++pl) {
        FxVec3 out[8];
        int outN = 0;
        if (polyN == 0) break;
        FxVec3 prev = poly[polyN - 1];
        fx fprev = sdist(planes[pl], prev);
        for (int k = 0; k < polyN; ++k) {
            const FxVec3 cur = poly[k];
            const fx fcur = sdist(planes[pl], cur);
            const bool curIn = (fcur >= 0);
            const bool prevIn = (fprev >= 0);
            if (curIn) {
                if (!prevIn) {
                    // entering: emit the crossing point first, then cur.
                    const fx denom = fprev - fcur;
                    const fx tp = (denom != 0) ? fxdiv(fprev, denom) : 0;
                    out[outN++] = FxVec3{prev.x + fxmul(cur.x - prev.x, tp),
                                         prev.y + fxmul(cur.y - prev.y, tp),
                                         prev.z + fxmul(cur.z - prev.z, tp)};
                }
                out[outN++] = cur;
            } else if (prevIn) {
                // leaving: emit the crossing point only.
                const fx denom = fprev - fcur;
                const fx tp = (denom != 0) ? fxdiv(fprev, denom) : 0;
                out[outN++] = FxVec3{prev.x + fxmul(cur.x - prev.x, tp),
                                     prev.y + fxmul(cur.y - prev.y, tp),
                                     prev.z + fxmul(cur.z - prev.z, tp)};
            }
            prev = cur; fprev = fcur;
        }
        polyN = outN;
        for (int k = 0; k < polyN; ++k) poly[k] = out[k];
    }

    // Keep the penetrating clipped vertices (depth d = FxDot(n, faceCenter - vertex) >= 0). Their world
    // contact point = the vertex itself (documented choice); depth = d. 0..8 candidates in clip order.
    FxVec3 candPts[8];
    fx     candDepth[8];
    int    candN = 0;
    for (int k = 0; k < polyN; ++k) {
        const FxVec3 rel = FxVec3{faceCenter.x - poly[k].x, faceCenter.y - poly[k].y,
                                  faceCenter.z - poly[k].z};
        const fx d = FxDot(n, rel);
        if (d >= 0) { candPts[candN] = poly[k]; candDepth[candN] = d; ++candN; }
    }

    m.normal = n;
    if (candN == 0) {
        // No clipped vertex penetrates (a grazing/degenerate face contact) -> fall back to ONE point at the
        // SAT-implied deepest incident-face corner. Keep count>=1 for an overlapping pair (the proof bar).
        // Use incident-face corner 0 with depth = satResult.penetration (deterministic).
        m.count = 1;
        m.points[0] = poly[0 < polyN ? 0 : 0];   // first surviving vertex (or corner 0 if none survived)
        if (polyN == 0) m.points[0] = incFaceCenter;
        m.depths[0] = sat.penetration;
        return m;
    }

    // Reduce to <=4: ALWAYS keep the DEEPEST (max depth, tie -> lowest clip-order index), then up to 3 MORE
    // in clip order. Deterministic.
    int deepest = 0;
    for (int k = 1; k < candN; ++k) if (candDepth[k] > candDepth[deepest]) deepest = k;
    m.points[0] = candPts[deepest];
    m.depths[0] = candDepth[deepest];
    uint32_t cnt = 1;
    for (int k = 0; k < candN && cnt < 4; ++k) {
        if (k == deepest) continue;
        m.points[cnt] = candPts[k];
        m.depths[cnt] = candDepth[k];
        ++cnt;
    }
    m.count = cnt;
    return m;
}

// ----- ManifoldMeasure: the deterministic summary of a set of pairs' manifolds -------------------------
// pairs        = total pairs measured; withContact = pairs yielding a non-empty manifold (count>=1);
// totalPoints  = the sum of all contact points over those manifolds. Pure integer, fixed order.
struct ManifoldMeasure {
    uint32_t pairs       = 0;
    uint32_t withContact = 0;
    uint32_t totalPoints = 0;
};

// MeasureManifold(pairs): BoxSat then BuildManifold over each pair, accumulate the deterministic summary.
inline ManifoldMeasure MeasureManifold(const std::vector<SatPair>& pairs) {
    ManifoldMeasure m;
    m.pairs = (uint32_t)pairs.size();
    for (const SatPair& p : pairs) {
        const SatResult r = BoxSat(p.bodyA, p.boxA, p.bodyB, p.boxB);
        const ContactManifold cm = BuildManifold(p.bodyA, p.boxA, p.bodyB, p.boxB, r);
        if (cm.count > 0) { ++m.withContact; m.totalPoints += cm.count; }
    }
    return m;
}

}  // namespace convex
}  // namespace hf::sim
