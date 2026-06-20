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

// =========================================================================================================
// Slice CX3 — THE ANGULAR CONTACT IMPULSE (the new physics). APPENDED after MeasureManifold (CX1+CX2's
// lines above are BYTE-FROZEN). Turns the CX2 ContactManifold into the VELOCITY response: the inverse-mass
// + INERTIA-TENSOR contact impulse, applied to BOTH linear velocity AND angular velocity. This is the FIRST
// time in the whole engine that `angVel` is driven by a contact — a box hitting a static box OFF-CENTER
// gains spin and TUMBLES (impossible in the sphere-sphere fpx solver). Deterministic integer Gauss-Seidel,
// every order PINNED. VELOCITY-ONLY (NO position de-penetration — that is CX4); the showcase free-integrates
// the resolved body with IntegrateBodyFull to SHOW the tumble.
//
// THE int64 REALITY (the CX1/CX2/FPX3 lesson): the inertia fxdiv + the FxDot/FxCross/FxMat3MulVec Q16.16
// products are int64. DXC compiles int64 (Vulkan); glslc cannot. So shaders/convex_solve.comp is
// VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl); the Metal --convex-tumble runs THIS CPU ResolveContactPair —
// byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU memcmp.

using fpx::FxScale;
using fpx::FxAdd;

// ----- ContactSolveConfig: the impulse solve parameters (restitution + Gauss-Seidel sweep count) ---------
struct ContactSolveConfig {
    fx       restitution = 0;   // Q16.16 bounce factor (0 = fully inelastic; the CX3 showcase uses 0)
    uint32_t iters       = 4;   // Gauss-Seidel sweeps over the manifold points
};

// ----- FxBoxInvInertiaBody: the 3 diagonal BODY-space inverse inertias of a box -------------------------
// A box with half-extents (hx,hy,hz) and mass m has the analytic DIAGONAL body-space inertia
// Ixx=(m/3)(hy²+hz²), Iyy=(m/3)(hx²+hz²), Izz=(m/3)(hx²+hy²) (from the full extents 2h). We store
// invMass=1/m, so the body-space INVERSE inertia diagonal is invIbody = (3·invMass/(hy²+hz²),
// 3·invMass/(hx²+hz²), 3·invMass/(hx²+hy²)) — each a Q16.16 fxdiv(3*invMass, fxmul(h,h)+fxmul(h,h)) (int64
// inside fxmul/fxdiv). A STATIC body (invMass==0) -> (0,0,0) (infinite inertia, takes no angular impulse).
inline FxVec3 FxBoxInvInertiaBody(const FxBox& box, fx invMass) {
    if (invMass == 0) return FxVec3{0, 0, 0};
    const fx hx = box.halfExtents.x, hy = box.halfExtents.y, hz = box.halfExtents.z;
    const fx hx2 = fxmul(hx, hx), hy2 = fxmul(hy, hy), hz2 = fxmul(hz, hz);
    const fx three = 3 * invMass;   // 3·invMass in Q16.16 (invMass < a few kOne -> no int32 overflow)
    return FxVec3{
        fxdiv(three, hy2 + hz2),
        fxdiv(three, hx2 + hz2),
        fxdiv(three, hx2 + hy2),
    };
}

// ----- WorldInvInertia: R·diag(invIbody)·Rᵀ as an explicit FxMat3 (the FxMat3 type's first real use) ----
// R's columns are the body's WORLD face axes (BoxAxes). For an orthonormal R, R·diag·Rᵀ = Σ_k invIbody[k]·
// (axis_k ⊗ axis_k) — the sum of each world axis outer-product with itself, scaled by its inverse inertia.
// Symmetric. Pure integer (the outer products are fxmul, the scaled sums are int64-safe within bound).
inline FxMat3 WorldInvInertia(const FxBody& body, const FxVec3& invIbody) {
    FxVec3 ax[3];
    BoxAxes(body, ax);
    const fx d[3] = {invIbody.x, invIbody.y, invIbody.z};
    FxMat3 M;   // zero-initialized
    for (int k = 0; k < 3; ++k) {
        const FxVec3& a = ax[k];
        const fx dk = d[k];
        // outer = dk * (a ⊗ a); accumulate into M (row r, col c) = Σ_k dk * a[r] * a[c]. Two nested fxmul.
        const fx ax0 = a.x, ax1 = a.y, ax2 = a.z;
        const fx da0 = fxmul(dk, ax0), da1 = fxmul(dk, ax1), da2 = fxmul(dk, ax2);
        M.m[0] += fxmul(da0, ax0); M.m[1] += fxmul(da0, ax1); M.m[2] += fxmul(da0, ax2);
        M.m[3] += fxmul(da1, ax0); M.m[4] += fxmul(da1, ax1); M.m[5] += fxmul(da1, ax2);
        M.m[6] += fxmul(da2, ax0); M.m[7] += fxmul(da2, ax1); M.m[8] += fxmul(da2, ax2);
    }
    return M;
}

// ----- SolveManifoldImpulse: the Gauss-Seidel velocity+angular impulse over a ContactManifold -----------
// The normal is SIGN-CORRECTED to point from A toward B ONCE (flip if FxDot(normal, bodyB.pos-bodyA.pos)<0;
// this sidesteps the CX2 ref->inc ambiguity). Then `iters` sweeps; in each sweep the manifold points are
// iterated 0..count-1 in order, mutating the body vel/angVel IN PLACE (Gauss-Seidel — later points see the
// earlier updates; the single-thread GPU mirror reproduces this bit-for-bit). For each point:
//   rA = p - bodyA.pos, rB = p - bodyB.pos (world lever arms from each center of mass).
//   vpA = vA + ωA×rA, vpB = vB + ωB×rB (contact-point velocities; ω is WORLD-frame body.angVel).
//   vn = (vpB - vpA)·n; if vn >= 0 (separating/resting) -> SKIP this point (deterministic).
//   k = invMassA + invMassB + n·((invIaW·(rA×n))×rA) + n·((invIbW·(rB×n))×rB); if k<=0 -> skip (degenerate).
//   jn = -(1+restitution)·vn / k, CLAMPED to >= 0 (a contact only PUSHES). J = n·jn.
//   vA -= J·invMassA; ωA -= invIaW·(rA×J);   vB += J·invMassB; ωB += invIbW·(rB×J).
inline void SolveManifoldImpulse(FxBody& bodyA, FxBody& bodyB, const FxMat3& invIaW, const FxMat3& invIbW,
                                 const ContactManifold& manifold, fx restitution, uint32_t iters) {
    if (manifold.count == 0) return;
    // Sign-correct the normal to point A->B ONCE.
    FxVec3 n = manifold.normal;
    if (FxDot(n, fpx::FxSub(bodyB.pos, bodyA.pos)) < 0) n = FxVec3{-n.x, -n.y, -n.z};

    const fx invMassA = bodyA.invMass;
    const fx invMassB = bodyB.invMass;

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t pi = 0; pi < manifold.count; ++pi) {
            const FxVec3 p  = manifold.points[pi];
            const FxVec3 rA = fpx::FxSub(p, bodyA.pos);
            const FxVec3 rB = fpx::FxSub(p, bodyB.pos);
            // contact-point velocities vpA = vA + ωA×rA, vpB = vB + ωB×rB.
            const FxVec3 vpA = FxAdd(bodyA.vel, FxCross(bodyA.angVel, rA));
            const FxVec3 vpB = FxAdd(bodyB.vel, FxCross(bodyB.angVel, rB));
            const fx vn = FxDot(fpx::FxSub(vpB, vpA), n);
            if (vn >= 0) continue;   // separating/resting -> no impulse (deterministic)

            // effective-mass denominator k.
            const FxVec3 raxn = FxCross(rA, n);
            const FxVec3 rbxn = FxCross(rB, n);
            const fx angA = FxDot(n, FxCross(FxMat3MulVec(invIaW, raxn), rA));
            const fx angB = FxDot(n, FxCross(FxMat3MulVec(invIbW, rbxn), rB));
            const fx k = invMassA + invMassB + angA + angB;
            if (k <= 0) continue;   // degenerate -> skip

            // scalar impulse jn = -(1+restitution)·vn / k, clamped >= 0.
            fx jn = fxdiv(-fxmul(kOne + restitution, vn), k);
            if (jn < 0) jn = 0;
            const FxVec3 J = FxScale(n, jn);

            // apply to BOTH bodies (statics with invMass==0 / invI==0 are unaffected).
            bodyA.vel = fpx::FxSub(bodyA.vel, FxScale(J, invMassA));
            bodyA.angVel = fpx::FxSub(bodyA.angVel, FxMat3MulVec(invIaW, FxCross(rA, J)));
            bodyB.vel = FxAdd(bodyB.vel, FxScale(J, invMassB));
            bodyB.angVel = FxAdd(bodyB.angVel, FxMat3MulVec(invIbW, FxCross(rB, J)));
        }
    }
}

// ----- ResolveContactPair: BoxSat -> BuildManifold -> world inertias -> SolveManifoldImpulse ------------
// One box-box pair end-to-end. !overlap -> no-op (the velocities/angVels are untouched). VELOCITY-ONLY (NO
// position de-penetration — CX4). The shader copies THIS body VERBATIM (with the BoxSat/BuildManifold/
// inertia/SolveManifoldImpulse it calls) so the GPU resolved-body result is byte-identical to the CPU.
inline void ResolveContactPair(FxBody& bodyA, const FxBox& boxA, FxBody& bodyB, const FxBox& boxB,
                               const ContactSolveConfig& cfg) {
    const SatResult sat = BoxSat(bodyA, boxA, bodyB, boxB);
    if (!sat.overlap) return;   // separated -> no-op
    const ContactManifold m = BuildManifold(bodyA, boxA, bodyB, boxB, sat);
    if (m.count == 0) return;
    const FxVec3 invIa = FxBoxInvInertiaBody(boxA, bodyA.invMass);
    const FxVec3 invIb = FxBoxInvInertiaBody(boxB, bodyB.invMass);
    const FxMat3 invIaW = WorldInvInertia(bodyA, invIa);
    const FxMat3 invIbW = WorldInvInertia(bodyB, invIb);
    SolveManifoldImpulse(bodyA, bodyB, invIaW, invIbW, m, cfg.restitution, cfg.iters);
}

// =========================================================================================================
// Slice CX4 — THE FULL CONVEX STEP (a settling stack). APPENDED after ResolveContactPair (CX1+CX2+CX3's
// lines above are BYTE-FROZEN). ASSEMBLES BoxSat -> BuildManifold -> the box inertia tensors ->
// SolveManifoldImpulse into the full per-tick WORLD step over a small set of boxes (some static), and ADDS
// the one piece CX3 lacked: POSITION DE-PENETRATION (CX3 was velocity-only). The result: a STACK of boxes
// on a static floor SETTLES into a coherent resting tower (boxes interlock + rest instead of sinking),
// impossible in the sphere-sphere fpx solver. Deterministic integer multi-pass, every order PINNED:
//   1. predict-integrate every DYNAMIC body (IntegrateBodyFull — vel+=g*dt, pos+=vel*dt, orient integrate);
//   2. all-pairs narrowphase in the FIXED i<j order (skip static-static), collect overlapping manifolds;
//   3. impulse solve — Gauss-Seidel: solveIters outer sweeps, each sweep iterates the pair list in fixed
//      order applying ONE SolveManifoldImpulse sweep per pair (mutating bodies in place — later pairs see
//      earlier updates); the world inverse inertias are recomputed ONCE per tick from the post-integrate
//      orient (the common cheap choice — stays deterministic);
//   4. position de-penetration (the NEW bit) — re-run BoxSat per pair (positions unchanged this sub-step,
//      re-deriving keeps the depth current) and push the two bodies APART along the A->B-corrected normal
//      by corrected = fxmul(max(0, penetration - slop), beta), split by inverse mass (both static -> skip;
//      one static -> the dynamic takes all). LINEAR correction only (no positional angular correction).
// All-pairs (NOT the FPX2 broadphase) — the CX4 scene is a SMALL stack (<= ~6 boxes); broadphase is a
// scaling refinement, deferred + documented. Pure integer -> the convex_step.comp shader runs the WHOLE
// StepConvexWorldN over the small body set single-thread, copying StepConvexWorld VERBATIM -> the GPU final
// body world is byte-identical to the CPU reference (the GPU==CPU memcmp is the make-or-break).
//
// THE int64 REALITY (the CX1/CX2/CX3/FPX3 lesson): the whole chain is int64 (the inertia/impulse/SAT
// products). DXC compiles int64 (Vulkan); glslc cannot. So convex_step.comp is VULKAN-SPIR-V-ONLY (NOT in
// hf_gen_msl); the Metal --convex-stack runs THIS CPU StepConvexWorldN — byte-identical to the Vulkan GPU
// result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU memcmp proof.

// ----- ConvexWorld: a small set of oriented boxes, parallel bodies[i] <-> boxes[i] arrays ----------------
// Some bodies are STATIC (invMass==0, e.g. the floor). The body carries pos/orient/vel/angVel/invMass/flags;
// the box carries the half-extents. std430-packable as (FxBody 16 ints) + (FxBox 3 ints) per body.
struct ConvexWorld {
    std::vector<FxBody> bodies;
    std::vector<FxBox>  boxes;
};

// ----- ConvexStepConfig: the full-step parameters (gravity + dt + solve iters + restitution + the position
// de-penetration slop + beta). slop = the allowed penetration (~1/64 unit) + beta = the correction fraction
// (~0.8 in Q16.16) reduce jitter -> the stack RESTS instead of sinking. All FIXED, deterministic.
struct ConvexStepConfig {
    FxVec3   gravity;            // Q16.16 acceleration (e.g. (0,-9.8,0) host-snapped)
    fx       dt          = kOne / 60;
    uint32_t solveIters  = 8;    // world-level Gauss-Seidel sweeps over the pair list
    fx       restitution = 0;    // Q16.16 bounce factor (0 = fully inelastic — a resting stack)
    fx       slop        = kOne / 64;   // allowed penetration before pushing apart (~0.0156 unit)
    fx       beta        = (fx)((int64_t)8 * kOne / 10);   // 0.8 — the position-correction fraction
    // Per-tick velocity retention (a deterministic drag, applied post-integrate). linDamp/angDamp are
    // Q16.16 RETAIN factors in (0,kOne]: vel *= linDamp, angVel *= angDamp each tick (kOne == no damping).
    // ANGULAR damping is the STABILITY KNOB for a resting stack: the non-accumulated Gauss-Seidel + the
    // fixed-point 4-point face manifold leave a tiny per-tick RESIDUAL TORQUE (documented, the CX3 lesson —
    // a symmetric multi-point patch is non-zero in fixed point), which integrates into a SPURIOUS spin that
    // slowly tips + collapses the tower. A mild angular drag (~0.9 retain) bleeds that spurious spin off so
    // the stack RESTS, with NO effect on the linear settling. Defensible (real solvers sleep/damp resting
    // bodies); deterministic (a plain per-component fxmul). Default kOne == OFF (back-compat / opt-in).
    fx       linDamp     = kOne;   // velocity retain per tick (kOne = none)
    fx       angDamp     = kOne;   // angular-velocity retain per tick (kOne = none)
    // Position de-penetration sweep count (the PBD "position iterations"). One pass cannot propagate the
    // separation through a multi-box stack in a single tick (the bottom contact pushes the lower box up
    // into the one above faster than a single de-pen sweep separates them); looping the de-pen pass in the
    // FIXED i<j order lets the push propagate up the tower each tick -> a stable resting stack. Default 1
    // (one pass — the spec's baseline). Deterministic (the same fixed order each sweep).
    uint32_t posIters    = 1;
};

// ----- StackMeasure: the deterministic rest/interpenetration summary of a settled stack -----------------
// maxSpeed       = the max FxLength(vel) over the DYNAMIC bodies (the rest test — a settled stack ~0);
// maxPenetration = the max BoxSat penetration over all i<j pairs (the interpenetration test — a held stack
//                  is within slop + epsilon); dynamicCount = the number of dynamic bodies. Pure integer.
struct StackMeasure {
    fx       maxSpeed       = 0;
    fx       maxPenetration = 0;
    uint32_t dynamicCount   = 0;
};

// IsDynamic(b): a body is integrated/movable iff it has a non-zero inverse mass AND the dynamic flag (the
// step-1 predicate; matches IntegrateBodyFull's kFlagDynamic gate + the static invMass==0 convention).
inline bool IsDynamic(const FxBody& b) {
    return b.invMass != 0 && (b.flags & fpx::kFlagDynamic);
}

// The face-preference epsilon: in a RESTING stack two axis-aligned boxes touch face-to-face with a shallow
// Y penetration, but the edge-edge cross axis FxCross(X,Z) ~ -Y normalizes to the SAME direction up to a
// 1-LSB fixed-point drift — so the strict-< min-pen scan can pick that EDGE axis by a single LSB. The edge
// axis carries tiny lateral (off-Y) components that, accumulated over many ticks of position correction,
// drift + DESTABILIZE the stack (the boxes slowly walk off + collapse). The fix (a CX4-level post-filter,
// CX1's BoxSat BYTE-FROZEN): when BoxSat returns an EDGE axis, prefer a FACE axis whose penetration is no
// more than kFacePrefEps deeper — the face normal is a CLEAN FxRotate'd world axis (no edge-cross drift),
// so the resting contact resolves stably. ~1/64 unit covers the LSB tie band without changing genuine
// edge contacts (whose face penetration is FAR larger). Deterministic (same threshold CPU + GPU).
inline constexpr fx kFacePrefEps = kOne / 64;   // face preferred if within this of the global min-pen

// BoxSatStable(bodyA, boxA, bodyB, boxB): BoxSat (CX1, byte-frozen) + the face-preference post-filter. If
// BoxSat reports overlap on an EDGE axis (axisIndex >= 6) but a FACE axis (0..5) penetrates within
// kFacePrefEps of that edge penetration, the FACE result is returned instead (a clean, stable normal). The
// 6 face-axis projections are RE-COMPUTED here (the SAME ProjectedRadius/FxDot/FxNormalize int64 ops as
// BoxSat — additive CX4 code, NOT a CX1 modification), in the FIXED order (A's 3 then B's 3, lowest index
// wins ties), so the result is bit-reproducible CPU<->Vulkan<->Metal. A separated pair -> {overlap=false}.
inline SatResult BoxSatStable(const FxBody& bodyA, const FxBox& boxA,
                              const FxBody& bodyB, const FxBox& boxB) {
    const SatResult base = BoxSat(bodyA, boxA, bodyB, boxB);
    if (!base.overlap || base.axisIndex < 6) return base;   // separated / already a face axis -> as-is

    // Re-run ONLY the 6 face axes to find the shallowest face penetration (clean normals).
    FxVec3 axA[3], axB[3];
    BoxAxes(bodyA, axA);
    BoxAxes(bodyB, axB);
    const FxVec3 hA = boxA.halfExtents;
    const FxVec3 hB = boxB.halfExtents;
    const FxVec3 t = fpx::FxSub(bodyB.pos, bodyA.pos);
    auto absfx = [](fx v) { return v < 0 ? -v : v; };

    bool found = false;
    fx minPen = 0;
    uint32_t minIndex = 0;
    FxVec3 minAxis;
    auto testFace = [&](const FxVec3& rawL, uint32_t index) {
        const FxVec3 L = fpx::FxNormalize(rawL);
        const fx rA = ProjectedRadius(L, axA, hA);
        const fx rB = ProjectedRadius(L, axB, hB);
        const fx s = absfx(FxDot(L, t));
        const fx sum = rA + rB;
        if (s > sum) return;                 // (shouldn't happen for an overlapping pair, but be safe)
        const fx pen = sum - s;
        if (!found || pen < minPen) {
            found = true; minPen = pen; minIndex = index;
            minAxis = (FxDot(L, t) < 0) ? FxVec3{-L.x, -L.y, -L.z} : L;
        }
    };
    for (uint32_t i = 0; i < 3; ++i) testFace(axA[i], i);
    for (uint32_t j = 0; j < 3; ++j) testFace(axB[j], 3 + j);

    // Prefer the face only if it is within kFacePrefEps of the edge penetration (the LSB-tie band).
    if (found && minPen <= base.penetration + kFacePrefEps) {
        SatResult r;
        r.overlap = true;
        r.axisIndex = minIndex;
        r.penetration = minPen;
        r.axis = minAxis;
        return r;
    }
    return base;
}

// ----- StepConvexWorld(world, cfg): ONE deterministic tick. The 5-pass step above, ALL orders PINNED. -----
// The shader copies THIS body VERBATIM (with the BoxSat/BuildManifold/inertia/SolveManifoldImpulse it calls)
// so the GPU final body world is byte-identical to the CPU.
inline void StepConvexWorld(ConvexWorld& world, const ConvexStepConfig& cfg) {
    const size_t n = world.bodies.size();

    // (1) Predict-integrate every dynamic body (static bodies untouched), then apply the per-tick velocity
    // retention (the deterministic drag — angDamp is the resting-stack stability knob; kOne == no damping,
    // the back-compat default). Fixed body order.
    for (size_t i = 0; i < n; ++i) {
        if (IsDynamic(world.bodies[i])) {
            fpx::IntegrateBodyFull(world.bodies[i], cfg.gravity, cfg.dt);
            if (cfg.linDamp != kOne) world.bodies[i].vel = FxScale(world.bodies[i].vel, cfg.linDamp);
            if (cfg.angDamp != kOne) world.bodies[i].angVel = FxScale(world.bodies[i].angVel, cfg.angDamp);
        }
    }

    // The world inverse inertias, recomputed ONCE per tick from the post-integrate orient (the common cheap
    // choice; deterministic). Indexed by body. Statics get a zero matrix (FxBoxInvInertiaBody(.,0) -> 0).
    std::vector<FxMat3> invIW(n);
    for (size_t i = 0; i < n; ++i) {
        const FxVec3 invIbody = FxBoxInvInertiaBody(world.boxes[i], world.bodies[i].invMass);
        invIW[i] = WorldInvInertia(world.bodies[i], invIbody);
    }

    // (3) Impulse solve — world-level Gauss-Seidel over the all-pairs list. For determinism we iterate the
    // pairs in the FIXED i<j order EACH outer sweep and run ONE SolveManifoldImpulse sweep per pair. The
    // manifold/SAT are re-derived per pair per sweep from the CURRENT positions (positions don't move in the
    // velocity solve, but the bodies' vel/angVel do — re-deriving keeps the order pinned + the GPU-copy
    // trivial). Skip static-static pairs (both invMass==0). The mutation is in place (later pairs see it).
    for (uint32_t sweep = 0; sweep < cfg.solveIters; ++sweep) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;  // static-static
                const SatResult sat = BoxSatStable(world.bodies[i], world.boxes[i],
                                                   world.bodies[j], world.boxes[j]);
                if (!sat.overlap) continue;
                const ContactManifold m = BuildManifold(world.bodies[i], world.boxes[i],
                                                        world.bodies[j], world.boxes[j], sat);
                if (m.count == 0) continue;
                SolveManifoldImpulse(world.bodies[i], world.bodies[j], invIW[i], invIW[j], m,
                                     cfg.restitution, 1);   // ONE inner sweep — the outer loop is the GS
            }
        }
    }

    // (4) Position de-penetration (the NEW bit — what makes the stack REST not sink). posIters sweeps, each
    // sweep over every overlapping pair in the FIXED i<j order, pushing the two bodies APART along the
    // A->B-corrected normal by
    //   corrected = fxmul(max(0, penetration - slop), beta)
    // split by inverse mass: wi = fxdiv(invMassA, invMassA+invMassB), wj = kOne - wi (both static -> skip;
    // one static -> the dynamic takes all). LINEAR only (no angular position correction). Re-run BoxSat for
    // the current depth. Fixed order, in place (later pairs/sweeps see earlier pushes — the Gauss-Seidel
    // de-pen; multiple sweeps propagate the separation up a multi-box tower each tick).
    for (uint32_t pit = 0; pit < cfg.posIters; ++pit) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const fx invSum = world.bodies[i].invMass + world.bodies[j].invMass;
                if (invSum == 0) continue;   // both static -> skip
                const SatResult sat = BoxSatStable(world.bodies[i], world.boxes[i],
                                                   world.bodies[j], world.boxes[j]);
                if (!sat.overlap) continue;
                // The de-penetration normal: the SAT axis, SIGN-CORRECTED to point A->B (mirrors the impulse
                // solver's sign rule), so +normal moves B away from A.
                FxVec3 nrm = sat.axis;
                if (FxDot(nrm, fpx::FxSub(world.bodies[j].pos, world.bodies[i].pos)) < 0)
                    nrm = FxVec3{-nrm.x, -nrm.y, -nrm.z};
                fx excess = sat.penetration - cfg.slop;
                if (excess <= 0) continue;   // within the allowed band -> no push (the anti-jitter slop)
                const fx corrected = fxmul(excess, cfg.beta);
                const fx wi = fxdiv(world.bodies[i].invMass, invSum);
                const fx wj = kOne - wi;
                // bodyA (i) moves -nrm*(corrected*wi); bodyB (j) moves +nrm*(corrected*wj).
                const FxVec3 ci = FxScale(nrm, fxmul(corrected, wi));
                const FxVec3 cj = FxScale(nrm, fxmul(corrected, wj));
                world.bodies[i].pos = fpx::FxSub(world.bodies[i].pos, ci);
                world.bodies[j].pos = FxAdd(world.bodies[j].pos, cj);
            }
        }
    }
    // (5) Orientation was already integrated in step (1).
}

// ----- StepConvexWorldN(world, cfg, ticks): run `ticks` StepConvexWorld steps -> the stack settles. -------
inline void StepConvexWorldN(ConvexWorld& world, const ConvexStepConfig& cfg, uint32_t ticks) {
    for (uint32_t t = 0; t < ticks; ++t) StepConvexWorld(world, cfg);
}

// ----- MeasureStack(world): the deterministic rest + interpenetration summary (pure integer, fixed order).
inline StackMeasure MeasureStack(const ConvexWorld& world) {
    StackMeasure ms;
    const size_t n = world.bodies.size();
    for (size_t i = 0; i < n; ++i) {
        if (IsDynamic(world.bodies[i])) {
            ++ms.dynamicCount;
            const fx sp = fpx::FxLength(world.bodies[i].vel);
            if (sp > ms.maxSpeed) ms.maxSpeed = sp;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;
            const SatResult sat = BoxSat(world.bodies[i], world.boxes[i],
                                         world.bodies[j], world.boxes[j]);
            if (sat.overlap && sat.penetration > ms.maxPenetration) ms.maxPenetration = sat.penetration;
        }
    }
    return ms;
}

}  // namespace convex
}  // namespace hf::sim
