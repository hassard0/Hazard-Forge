#pragma once
// Slice GJ1 — General Convex-Hull Contacts: THE HULL + SUPPORT FUNCTION (the BEACHHEAD of FLAGSHIP #22:
// DETERMINISTIC GENERAL CONVEX-HULL CONTACTS via integer GJK + EPA, hf::sim::gjk). The whole convex/
// friction/persist contact stack (#19-21) is BOX-ONLY (box-box SAT, convex.h::BoxSat). This flagship
// generalizes the ONE box-only component — the narrowphase — to arbitrary convex polyhedra. GJ1 builds the
// single primitive GJK and EPA are both built on: the SUPPORT FUNCTION — the world-space vertex of a convex
// hull farthest along a direction — plus the Minkowski-difference (configuration-space-obstacle) support
// that is GJK/EPA's only geometry call.
//
// Header-only, namespace hf::sim::gjk, #include "sim/convex.h" READ-ONLY (which transitively gives the fpx
// Q16.16 toolbox: fx/fxmul/FxVec3/FxAdd/FxSub/FxDot/FxRotate/FxBody/FxQuat). convex.h / fric.h / persist.h /
// fpx.h stay BYTE-FROZEN — this header REUSES their helpers, it does NOT redefine the fixed-point format.
//
// THE int64 REALITY (the CX1/FPX3 lesson, the honest proof-strength call): the support's world transform
// (FxRotate of the direction in + the vertex out) uses the int64 Q16.16 products (fxmul overflows int32 at
// world scale). DXC compiles int64 (the Vulkan path); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT
// parse int64_t in HLSL. So shaders/gjk_support.comp.hlsl is VULKAN-SPIR-V-ONLY (in the Vulkan compile list,
// NOT in the Metal hf_gen_msl list); the Metal --gjk-support runs the CPU Support path -> byte-identical to
// the Vulkan GPU result BY CONSTRUCTION (the convex_sat.comp / fpx_solve.comp split), while the Vulkan side
// carries the GPU==CPU memcmp proof. gjk_support.comp copies Support's body VERBATIM so the GPU exercises the
// EXACT integer ops -> a divergence is exactly what the host GPU==CPU memcmp catches.
//
// DETERMINISM CRUX: SupportLocal scans the verts in FIXED index order [0,count) with a STRICT-GREATER update,
// so ties keep the LOWEST index (the convex.h:28 / SAT min-pen tie-break idiom). The argmax COMPARE ranks the
// int64 FxDot extents but the result is the vertex index — pure, reproducible, identical CPU/GPU.

#include <cstdint>

#include "sim/convex.h"   // read-only: convex::FxVec3/FxDot/FxSub/FxAdd/FxScale + convex::fx + the fpx
                          // toolbox (fpx::FxBody/FxQuat/FxRotate/fxmul). FxHull/Support* are NEW, here.

namespace hf::sim {
namespace gjk {

// Pull the convex/fpx Q16.16 helpers into this namespace (REUSE, do NOT redefine the fixed-point format).
using convex::fx;
using convex::kOne;
using convex::kFrac;
using convex::FxVec3;
using convex::FxDot;     // the Q16.16 dot (int64 intermediate) — ranks the support
using fpx::FxBody;
using fpx::FxQuat;
using fpx::FxRotate;     // rotate a vector by a unit quaternion (int64 fxmul)
using fpx::FxAdd;
using fpx::FxSub;

// The fixed compile-time vertex cap. GPU buffers are fixed-size; 20 verts covers tetra/octa/cube/wedge/
// dodecahedron — the documented ceiling, identical CPU/GPU. (No sibling constant precedent exists — convex.h
// is box-only with no vertex array — so 20 is the chosen ceiling per the spec.)
constexpr uint32_t kMaxHullVerts = 20;

// ----- FxHull: a convex hull as its LOCAL-space vertices (body-relative; the hull is immutable/shared like a
// box's half-extents). count <= kMaxHullVerts. Faces are NOT needed for GJ1 — the support function is purely
// over vertices (faces arrive in a later slice). std430-packable as kMaxHullVerts x int3 + a uint count.
struct FxHull {
    FxVec3   verts[kMaxHullVerts];
    uint32_t count = 0;
};

// FxQuatConjugate(q): the conjugate {-x,-y,-z,w} of a quaternion. For a UNIT quaternion this is its inverse
// (the rotation by -theta). No existing conjugate/inverse-rotate helper lives in fpx.h (grep clean), so it is
// expressed inline here WITHOUT modifying fpx.h. Rotating a direction by the conjugate maps it from world
// into the body's local frame. Pure integer negate — no products.
inline FxQuat FxQuatConjugate(const FxQuat& q) {
    return FxQuat{-q.x, -q.y, -q.z, q.w};
}

// FxNeg(v): component-wise negation of a vector (the Minkowski -dir). Pure integer negate, no products. No
// FxNeg helper exists in fpx.h/convex.h (grep clean), so it is expressed inline here.
inline FxVec3 FxNeg(const FxVec3& v) {
    return FxVec3{-v.x, -v.y, -v.z};
}

// ----- SupportLocal(hull, dir): the LOCAL-space vertex of the hull with the maximum FxDot(vert, dir) over
// i in [0, count). Scanned in FIXED index order with a STRICT-GREATER update so ties keep the LOWEST index
// (the convex.h min-pen tie-break idiom). The argmax COMPARE is on the int64 FxDot extents; the RESULT is the
// vertex. Deterministic, pure integer. count==0 -> the origin (a deterministic empty-hull fallback). The
// shader copies THIS body VERBATIM.
inline FxVec3 SupportLocal(const FxHull& hull, const FxVec3& dir) {
    if (hull.count == 0) return FxVec3{0, 0, 0};
    uint32_t best = 0;
    fx bestDot = FxDot(hull.verts[0], dir);
    for (uint32_t i = 1; i < hull.count; ++i) {
        const fx d = FxDot(hull.verts[i], dir);
        if (d > bestDot) {            // STRICT-greater -> ties keep the LOWEST index
            bestDot = d;
            best = i;
        }
    }
    return hull.verts[best];
}

// ----- Support(hull, body, dir): the WORLD-space support — the world vertex of the (body-placed) hull
// farthest along the WORLD direction dir. Rotate dir into the body's local frame (by the CONJUGATE of
// body.orient — rotating ONE vector in is fewer ops + identical result vs rotating every vertex out), call
// SupportLocal, then map that local vertex back to world: FxRotate(body.orient, v) + body.pos. The result is
// bit-identical on CPU and the GPU shader (the int64 FxRotate/FxDot ops match). The shader copies THIS body
// VERBATIM.
inline FxVec3 Support(const FxHull& hull, const FxBody& body, const FxVec3& dir) {
    const FxVec3 localDir = FxRotate(FxQuatConjugate(body.orient), dir);
    const FxVec3 localV = SupportLocal(hull, localDir);
    const FxVec3 worldV = FxRotate(body.orient, localV);
    return FxAdd(worldV, body.pos);
}

// ----- SupportMinkowski(hullA, bodyA, hullB, bodyB, dir): the support of the Minkowski difference A (-) B
// (the configuration-space obstacle) = Support_A(dir) - Support_B(-dir). The SINGLE geometry call GJK
// (simplex evolution) and EPA (polytope expansion) make. Deterministic, bit-identical CPU/GPU.
inline FxVec3 SupportMinkowski(const FxHull& hullA, const FxBody& bodyA,
                               const FxHull& hullB, const FxBody& bodyB,
                               const FxVec3& dir) {
    const FxVec3 sa = Support(hullA, bodyA, dir);
    const FxVec3 sb = Support(hullB, bodyB, FxNeg(dir));
    return FxSub(sa, sb);
}

// ----- HullMeasure: a deterministic summary the showcase/test asserts. Over a fixed (hull, body) set x a
// fixed direction set: the number of queries, the SUM of the returned WORLD support vertices, and the SUM of
// the support EXTENTS (FxDot(support, dir), the farthest-along projection). A PURE function of the inputs ->
// two MeasureSupport calls over the same inputs are byte-equal.
struct HullMeasure {
    uint32_t queries   = 0;   // # of (hull, dir) support queries
    FxVec3   supportSum;      // Σ of the returned world support vertices
    fx       extentSum = 0;   // Σ of FxDot(support, dir) — the support extents
};

// MeasureSupport(hulls, bodies, count, dirs, dirCount): run Support over every (hull i, dir d) pair in FIXED
// (i, d) order, accumulating the summary. PURE — no clock/RNG; the SAME inputs give a byte-equal HullMeasure
// every call. hulls[i] is placed by bodies[i].
inline HullMeasure MeasureSupport(const FxHull* hulls, const FxBody* bodies, uint32_t count,
                                  const FxVec3* dirs, uint32_t dirCount) {
    HullMeasure m;
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t d = 0; d < dirCount; ++d) {
            const FxVec3 s = Support(hulls[i], bodies[i], dirs[d]);
            m.supportSum = FxAdd(m.supportSum, s);
            m.extentSum += FxDot(s, dirs[d]);
            ++m.queries;
        }
    }
    return m;
}

// ===== Canonical hull builders (deterministic integer constructions) ================================
// A small set of canonical convex test hulls in LOCAL space, built from EXACT Q16.16 integer coordinates
// (no <cmath>, no float). They exercise tetra/box/octa/wedge vertex counts; all <= kMaxHullVerts.

// FromInt(v): the exact Q16.16 of an integer.
inline constexpr fx FromInt(int v) { return (fx)((int64_t)v << kFrac); }

// MakeTetra(h): a regular-ish tetrahedron with 4 verts at the alternate cube corners scaled by h (Q16.16):
// (+h,+h,+h), (+h,-h,-h), (-h,+h,-h), (-h,-h,+h). A deterministic 4-vert convex hull.
inline FxHull MakeTetra(fx h) {
    FxHull hull;
    hull.verts[0] = FxVec3{ h,  h,  h};
    hull.verts[1] = FxVec3{ h, -h, -h};
    hull.verts[2] = FxVec3{-h,  h, -h};
    hull.verts[3] = FxVec3{-h, -h,  h};
    hull.count = 4;
    return hull;
}

// MakeBox(hx, hy, hz): the 8 corners of an axis-aligned box with the given half-extents (Q16.16). The corner
// order is the fixed sign sweep (x outer, y middle, z inner) -> deterministic.
inline FxHull MakeBox(fx hx, fx hy, fx hz) {
    FxHull hull;
    uint32_t n = 0;
    const fx sx[2] = {-hx, hx}, sy[2] = {-hy, hy}, sz[2] = {-hz, hz};
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz)
                hull.verts[n++] = FxVec3{sx[ix], sy[iy], sz[iz]};
    hull.count = n;
    return hull;
}

// MakeOcta(r): a regular octahedron — the 6 axis poles at +-r (Q16.16) on each axis. A deterministic 6-vert
// convex hull. Order: +x,-x,+y,-y,+z,-z.
inline FxHull MakeOcta(fx r) {
    FxHull hull;
    hull.verts[0] = FxVec3{ r, 0, 0};
    hull.verts[1] = FxVec3{-r, 0, 0};
    hull.verts[2] = FxVec3{0,  r, 0};
    hull.verts[3] = FxVec3{0, -r, 0};
    hull.verts[4] = FxVec3{0, 0,  r};
    hull.verts[5] = FxVec3{0, 0, -r};
    hull.count = 6;
    return hull;
}

// MakeWedge(hx, hy, hz): a triangular prism (6 verts) — a box with the +x/+y top edge collapsed to a ridge
// (a right-triangle cross-section in xy extruded along z). Deterministic 6-vert convex hull:
//   base rectangle at y=-hy: (+-hx, -hy, +-hz); ridge at x=-hx, y=+hy: (-hx, +hy, +-hz).
inline FxHull MakeWedge(fx hx, fx hy, fx hz) {
    FxHull hull;
    hull.verts[0] = FxVec3{ hx, -hy,  hz};
    hull.verts[1] = FxVec3{ hx, -hy, -hz};
    hull.verts[2] = FxVec3{-hx, -hy,  hz};
    hull.verts[3] = FxVec3{-hx, -hy, -hz};
    hull.verts[4] = FxVec3{-hx,  hy,  hz};
    hull.verts[5] = FxVec3{-hx,  hy, -hz};
    hull.count = 6;
    return hull;
}

// =========================================================================================================
// Slice GJ2 — General Convex-Hull Contacts: THE GJK ALGORITHM (overlap + closest distance). APPENDED after
// MakeWedge (GJ1's lines above are BYTE-FROZEN). Builds the Gilbert-Johnson-Keerthi simplex evolution that,
// fed ONLY SupportMinkowski queries, decides whether two convex hulls OVERLAP and (when separated) returns
// their closest-point distance + the witness points on each hull. Pure integer Q16.16, a FIXED iteration
// bound + integer tie-breaks -> the simplex two runs evolve is byte-identical (the float-engine divergence
// killer). gjk_distance.comp copies Gjk's body VERBATIM (int64 -> Vulkan-only; Metal runs the CPU Gjk).
//
// THE int64 REALITY (the GJ1 / CX1 lesson): the sub-distance barycentrics (fxdiv) + the FxDot/FxCross
// Q16.16 products are int64. DXC compiles int64 (the Vulkan path); glslc CANNOT parse int64 in HLSL, so
// shaders/gjk_distance.comp.hlsl is VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl); the Metal --gjk-distance runs
// the CPU Gjk -> byte-identical BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU memcmp.
//
// DETERMINISM CRUX (spelled out): (a) FIXED iteration bound kGjkMaxIter — never an unbounded while;
// (b) integer sub-distance: the closest feature of the current simplex to the origin is chosen by FxDot/
// FxCross sign tests, ALL integer, ties -> the LOWEST-index feature (the convex.h:28 / SupportLocal idiom);
// (c) the barycentric weights use fxdiv in a PINNED order; (d) a FIXED initial search direction
// (bodyB.pos - bodyA.pos, fallback +X); (e) a DUPLICATE-SUPPORT guard — a CSO point already in the simplex
// (exact integer equality) terminates as separated (no progress), preventing cycling. Pure integer, fixed
// order -> identical CPU/GPU.

using convex::fxmul;       // the Q16.16 product (int64 intermediate)
using convex::fxdiv;       // the Q16.16 truncating divide (int64 intermediate)
using convex::FxCross;     // the Q16.16 cross (int64 intermediate) — the triangle/tetra normals + region tests
using convex::FxScale;     // component-wise fxmul by a scalar
using fpx::FxLength;       // the Q16.16 vector length (int64 FxISqrt) — the separation distance

// The FIXED iteration bound. GJK converges in a handful of iterations for convex hulls of <= kMaxHullVerts;
// 32 is a generous, deterministic ceiling that the loop NEVER exceeds (no unbounded while).
constexpr uint32_t kGjkMaxIter = 32;

// ----- Simplex: up to 4 CSO points + their witness points on A and B (for closest-point recovery) --------
// pts[i] is a Minkowski-difference (configuration-space-obstacle) point; csoA[i] = Support_A(d) and
// csoB[i] = Support_B(-d) the witness vertices on each hull that produced it. When GJK reduces to the closest
// feature, the SAME barycentric weights that express the closest CSO point combine csoA/csoB into the witness
// points on each hull. count in [0,4]. std430-packable as 12 x int3 + a uint.
struct Simplex {
    FxVec3   pts[4];
    FxVec3   csoA[4];
    FxVec3   csoB[4];
    uint32_t count = 0;
};

// ----- GjkResult: the GJK outcome ------------------------------------------------------------------------
// overlap=true  -> the hulls intersect (the origin is enclosed by the terminal simplex); separation/closest*
//                  are not meaningful (GJ3 EPA computes the penetration from `simplex`).
// overlap=false -> separation = the closest CSO point (origin -> closest point of A-B), its FxLength = the
//                  closest distance; closestA/closestB = the witness points on each hull (the barycentric
//                  combine of csoA/csoB). witnessFeature packs the terminal simplex's count + size for a
//                  deterministic feature id (the ContactKey raw material for later slices). iterations = the
//                  number of GJK iterations taken (<= kGjkMaxIter). std430-packable below.
struct GjkResult {
    uint32_t overlap        = 0;   // 0/1 (a uint for the std430 GPU mirror — bool packs as 4 bytes here)
    FxVec3   separation;           // origin -> closest CSO point (the closest-distance vector)
    FxVec3   closestA;             // witness point on hull A
    FxVec3   closestB;             // witness point on hull B
    Simplex  simplex;              // the terminal simplex (GJ3 EPA seeds from it)
    uint32_t witnessFeature = 0;   // a deterministic terminal-simplex feature id (count in low bits)
    uint32_t iterations     = 0;   // GJK iterations taken
};

// FxVec3 exact equality (the duplicate-support guard — a repeated CSO point terminates as separated). Pure
// integer compares, identical CPU/GPU.
inline bool FxVec3Eq(const FxVec3& a, const FxVec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// SubResult: the output of a sub-distance step — the new search direction toward the origin (dir), the
// retained simplex SIZE (how many of the input points survive, 1..3), the survivor indices (idx, fixed
// order), and the barycentric weights of the closest point over the survivors (Q16.16, summing to ~kOne).
// containsOrigin is set true ONLY by the tetra case when the origin is enclosed.
struct SubResult {
    FxVec3   dir;                 // search direction = origin - closestPoint (toward the origin)
    uint32_t keep[4] = {0,0,0,0}; // the retained input indices in FIXED order
    fx       w[4]    = {0,0,0,0}; // barycentric weights of the closest point over keep[0..size)
    uint32_t size    = 0;         // 1..3 retained (4 only transiently for the tetra-enclosing case)
    bool     containsOrigin = false;
};

// DoSimplex2(a,b): the closest feature of the segment [b -> a] to the origin (a is the LAST-added point, the
// Ericson convention). Region test: t = dot(AO, AB)/dot(AB,AB), clamped [0,1]. t<=0 -> vertex a (the newest);
// t>=1 -> would step back toward b (kept only if AB points away). FIXED tie-break: a degenerate AB (length
// below kEdgeEps) -> keep vertex a. The search dir = -(closest point). The weights distribute over {a,b}.
inline SubResult DoSimplex2(const FxVec3& a, const FxVec3& b) {
    SubResult r;
    const FxVec3 ab = FxSub(b, a);
    const FxVec3 ao = FxNeg(a);                 // origin - a
    const fx abLen = FxLength(ab);
    if (abLen < convex::kEdgeEps) {             // degenerate edge -> the newest vertex a (lowest-index rule)
        r.size = 1; r.keep[0] = 0; r.w[0] = kOne; r.dir = ao;
        return r;
    }
    const fx d1 = FxDot(ao, ab);                // projection of AO onto AB
    if (d1 <= 0) {                              // before a -> closest is the vertex a alone
        r.size = 1; r.keep[0] = 0; r.w[0] = kOne; r.dir = ao;
        return r;
    }
    const fx d2 = FxDot(ab, ab);                // |AB|^2
    if (d1 >= d2) {                             // beyond b -> closest is the vertex b alone
        r.size = 1; r.keep[0] = 1; r.w[0] = kOne; r.dir = FxNeg(b);
        return r;
    }
    // interior of the edge: closest = a + t*AB, t = d1/d2. Weights (1-t) on a, t on b.
    const fx t = fxdiv(d1, d2);
    const FxVec3 closest = FxAdd(a, FxScale(ab, t));
    r.size = 2; r.keep[0] = 0; r.keep[1] = 1; r.w[0] = kOne - t; r.w[1] = t;
    r.dir = FxNeg(closest);
    return r;
}

// DoSimplex3(a,b,c): the closest feature of the triangle (a is the newest vertex) to the origin. The Ericson
// ClosestPtPointTriangle region machine, integer: the 7 Voronoi regions (3 vertices, 3 edges, the face).
// FIXED tie-breaks (the convex.h lowest-index idiom): edge/vertex regions resolve by the standard <=/>=
// boundary tests in a PINNED order; a degenerate (near-zero) triangle normal -> fall back to the closest
// edge. Weights are the face barycentrics (u,v,w) for the face region, the edge formula for an edge, kOne for
// a vertex. The local indices map keep -> {a=0,b=1,c=2}.
inline SubResult DoSimplex3(const FxVec3& a, const FxVec3& b, const FxVec3& c) {
    SubResult r;
    const FxVec3 ab = FxSub(b, a);
    const FxVec3 ac = FxSub(c, a);
    const FxVec3 ao = FxNeg(a);
    // Vertex region a: outside both AB and AC at a.
    const fx d1 = FxDot(ab, ao);
    const fx d2 = FxDot(ac, ao);
    if (d1 <= 0 && d2 <= 0) { r.size = 1; r.keep[0] = 0; r.w[0] = kOne; r.dir = ao; return r; }
    // Vertex region b.
    const FxVec3 bo = FxNeg(b);
    const fx d3 = FxDot(ab, bo);
    const fx d4 = FxDot(ac, bo);
    if (d3 >= 0 && d4 <= d3) { r.size = 1; r.keep[0] = 1; r.w[0] = kOne; r.dir = bo; return r; }
    // Edge region AB: closest on AB.
    const fx vc = fxmul(d1, d4) - fxmul(d3, d2);
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const fx denom = d1 - d3;
        const fx t = (denom > convex::kEdgeEps || denom < -convex::kEdgeEps) ? fxdiv(d1, denom) : 0;
        const FxVec3 closest = FxAdd(a, FxScale(ab, t));
        r.size = 2; r.keep[0] = 0; r.keep[1] = 1; r.w[0] = kOne - t; r.w[1] = t; r.dir = FxNeg(closest);
        return r;
    }
    // Vertex region c.
    const FxVec3 co = FxNeg(c);
    const fx d5 = FxDot(ab, co);
    const fx d6 = FxDot(ac, co);
    if (d6 >= 0 && d5 <= d6) { r.size = 1; r.keep[0] = 2; r.w[0] = kOne; r.dir = co; return r; }
    // Edge region AC: closest on AC.
    const fx vb = fxmul(d5, d2) - fxmul(d1, d6);
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const fx denom = d2 - d6;
        const fx t = (denom > convex::kEdgeEps || denom < -convex::kEdgeEps) ? fxdiv(d2, denom) : 0;
        const FxVec3 closest = FxAdd(a, FxScale(ac, t));
        r.size = 2; r.keep[0] = 0; r.keep[1] = 2; r.w[0] = kOne - t; r.w[1] = t; r.dir = FxNeg(closest);
        return r;
    }
    // Edge region BC.
    const fx va = fxmul(d3, d6) - fxmul(d5, d4);
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        const fx denom = (d4 - d3) + (d5 - d6);
        const fx t = (denom > convex::kEdgeEps || denom < -convex::kEdgeEps) ? fxdiv(d4 - d3, denom) : 0;
        const FxVec3 bc = FxSub(c, b);
        const FxVec3 closest = FxAdd(b, FxScale(bc, t));
        r.size = 2; r.keep[0] = 1; r.keep[1] = 2; r.w[0] = kOne - t; r.w[1] = t; r.dir = FxNeg(closest);
        return r;
    }
    // Face region: closest = a + u*AB + v*AC via barycentrics (va,vb,vc). The face normal direction toward
    // the origin determines the search dir; the closest point's weights are the face barycentrics.
    const fx denom = va + vb + vc;
    fx u, v;   // weights on b (=v) and c (=w via 1-u-v); the Ericson (u=va,v=vb,w=vc)/denom
    if (denom > convex::kEdgeEps || denom < -convex::kEdgeEps) {
        v = fxdiv(vb, denom);
        const fx w = fxdiv(vc, denom);
        u = kOne - v - w;
        r.w[0] = u; r.w[1] = v; r.w[2] = w;
    } else {
        // degenerate triangle -> distribute deterministically over the 3 verts (the fallback closest is a).
        r.w[0] = kOne; r.w[1] = 0; r.w[2] = 0;
        v = 0;
    }
    const FxVec3 closest = FxAdd(FxAdd(FxScale(a, r.w[0]), FxScale(b, r.w[1])), FxScale(c, r.w[2]));
    r.size = 3; r.keep[0] = 0; r.keep[1] = 1; r.keep[2] = 2;
    r.dir = FxNeg(closest);
    (void)u; (void)v;
    return r;
}

// FxTriple(a,b,c) = a · (b × c), the scalar triple product (signed volume * 6) in Q16.16. int64 inside.
inline fx FxTriple(const FxVec3& a, const FxVec3& b, const FxVec3& c) { return FxDot(a, FxCross(b, c)); }

// DoSimplex4(a,b,c,d): the closest feature of the tetra (a is the newest) to the origin, OR origin enclosed.
// The standard 4-face point-in-tetra test with the face normals consistently oriented toward the 4th vertex.
// For each of the 3 faces incident to the newest vertex a (abc, acd, adb), if the origin is on the OUTSIDE,
// recurse into that triangle (the closest must be on it). If the origin is inside all -> overlap (enclosed).
// FIXED face order abc, acd, adb (the newest-vertex faces) with a consistent winding -> deterministic.
inline SubResult DoSimplex4(const FxVec3& a, const FxVec3& b, const FxVec3& c, const FxVec3& d) {
    // Orient: the reference is the 4th vertex (the one NOT on the face). A point p is OUTSIDE face (x,y,z)
    // away from the opposite vertex o iff sign(triple(p-x, y-x, z-x)) opposes sign(triple(o-x, y-x, z-x)).
    const FxVec3 ao = FxNeg(a);
    // Face ABC (opposite vertex d). Normal abc = (b-a)x(c-a).
    const FxVec3 ab = FxSub(b, a), ac = FxSub(c, a), ad = FxSub(d, a);
    const FxVec3 nABC = FxCross(ab, ac);
    const FxVec3 nACD = FxCross(ac, ad);
    const FxVec3 nADB = FxCross(ad, ab);
    // Sign of the opposite vertex relative to each face (through a): d for ABC, b for ACD, c for ADB.
    const fx sdABC = FxDot(nABC, ad);   // d side of ABC
    const fx sdACD = FxDot(nACD, ab);   // b side of ACD
    const fx sdADB = FxDot(nADB, ac);   // c side of ADB
    // Origin side of each face (through a): dot(n, AO).
    const fx soABC = FxDot(nABC, ao);
    const fx soACD = FxDot(nACD, ao);
    const fx soADB = FxDot(nADB, ao);
    // The origin is OUTSIDE a face iff it lies on the opposite side from the reference vertex (signs differ).
    auto outside = [](fx sRef, fx sOrigin) -> bool {
        // outside if the origin's side is strictly opposite the reference vertex's side.
        if (sRef > 0) return sOrigin < 0;
        if (sRef < 0) return sOrigin > 0;
        return sOrigin != 0;   // degenerate face (sRef==0): treat any non-coplanar origin as outside
    };
    // FIXED order: ABC, then ACD, then ADB. The first face the origin is outside of wins; recurse into its
    // triangle (newest vertex a stays first to preserve the DoSimplex3 a-is-newest convention).
    if (outside(sdABC, soABC)) {
        SubResult t = DoSimplex3(a, b, c);
        // remap t.keep (0=a,1=b,2=c) -> tetra-local (0=a,1=b,2=c) — identity here.
        return t;
    }
    if (outside(sdACD, soACD)) {
        SubResult t = DoSimplex3(a, c, d);
        for (uint32_t i = 0; i < t.size; ++i) {
            const uint32_t loc = t.keep[i];                  // 0=a,1=c,2=d
            t.keep[i] = (loc == 0) ? 0u : (loc == 1) ? 2u : 3u;
        }
        return t;
    }
    if (outside(sdADB, soADB)) {
        SubResult t = DoSimplex3(a, d, b);
        for (uint32_t i = 0; i < t.size; ++i) {
            const uint32_t loc = t.keep[i];                  // 0=a,1=d,2=b
            t.keep[i] = (loc == 0) ? 0u : (loc == 1) ? 3u : 1u;
        }
        return t;
    }
    // Inside all three newest-vertex faces -> the origin is enclosed by the tetra -> OVERLAP.
    SubResult r;
    r.containsOrigin = true;
    r.size = 4; r.keep[0] = 0; r.keep[1] = 1; r.keep[2] = 2; r.keep[3] = 3;
    r.dir = FxVec3{0, 0, 0};
    return r;
}

// Gjk(hullA, bodyA, hullB, bodyB): the GJK main loop. Returns overlap + (separated) the closest distance +
// witness points. The simplex is stored newest-FIRST inside the loop's working arrays; on each iteration the
// sub-distance reduces it to the feature closest to the origin and yields the next search direction. The
// barycentric weights of that closest feature combine the surviving csoA/csoB into the witness points.
inline GjkResult Gjk(const FxHull& hullA, const FxBody& bodyA, const FxHull& hullB, const FxBody& bodyB) {
    GjkResult res;

    // (d) FIXED initial search direction: bodyB.pos - bodyA.pos, fallback +X if zero.
    FxVec3 dir = FxSub(bodyB.pos, bodyA.pos);
    if (dir.x == 0 && dir.y == 0 && dir.z == 0) dir = FxVec3{kOne, 0, 0};

    // Working simplex (newest at [0]): the CSO points + their A/B witnesses.
    FxVec3 sp[4], swA[4], swB[4];
    uint32_t n = 0;

    fx wts[4] = {0, 0, 0, 0};       // the closest-feature barycentric weights (parallel to sp after reduce)
    uint32_t keepCount = 1;          // how many simplex points the closest feature retains

    // Seed the simplex with the first support; the closest point of a 1-simplex is the vertex itself.
    {
        const FxVec3 a  = SupportMinkowski(hullA, bodyA, hullB, bodyB, dir);
        sp[0] = a;
        swA[0] = Support(hullA, bodyA, dir);
        swB[0] = Support(hullB, bodyB, FxNeg(dir));
        n = 1;
        wts[0] = kOne;
    }
    FxVec3 closest = sp[0];          // the closest point of the CURRENT simplex to the origin

    // The progress epsilon: a new support whose advance toward the origin (along dir = -closest) over the
    // current closest point is below this is "no progress" -> SEPARATED. ~kEdgeEps in the dot scale; FIXED,
    // identical CPU/GPU. Guards against fixed-point stall at the closest feature.
    const fx kProgressEps = convex::kEdgeEps;

    bool overlap = false;
    uint32_t iter = 0;
    for (; iter < kGjkMaxIter; ++iter) {
        // (b) origin enclosed / on the feature: the closest point reached the origin (within kProgressEps) ->
        // OVERLAP. A near-origin closest point means the simplex feature passes through the origin (a touching
        // or penetrating pair); for a SEPARATED pair the closest distance is bounded well above this integer
        // epsilon (the scene gaps are ~1 world unit), so the band cleanly classifies. Using the length here
        // (vs exact zero) also resolves the degenerate case where the only supports lie on a near-origin edge
        // and the perpendicular search direction collapses (the classic 2-simplex-through-origin stall).
        if (FxLength(closest) <= kProgressEps) { overlap = true; break; }

        // Search toward the origin from the current closest point. NORMALIZE the direction to ~unit length
        // (kOne) so the support query + the advance test run at a CONSISTENT scale even when the closest
        // feature sits a tiny distance from the origin (a near-origin edge would otherwise collapse dir to a
        // sub-unit vector and stall the integer progress test). Support's argmax is scale-invariant; the
        // normalize is the SAME int64 FxISqrt/fxdiv op CPU + GPU. The advance test is then in world units.
        dir = FxNeg(closest);
        const FxVec3 sdir = fpx::FxNormalize(dir);

        // Add the new support along the normalized direction.
        const FxVec3 a  = SupportMinkowski(hullA, bodyA, hullB, bodyB, sdir);
        const FxVec3 wA = Support(hullA, bodyA, sdir);
        const FxVec3 wB = Support(hullB, bodyB, FxNeg(sdir));

        // (e) duplicate-support guard: a already in the simplex (exact) -> no progress -> SEPARATED.
        bool dup = false;
        for (uint32_t i = 0; i < n; ++i) if (FxVec3Eq(a, sp[i])) { dup = true; break; }
        if (dup) { overlap = false; break; }

        // (a) the canonical GJK progress test (against the UNIT direction so the scale is world units): the
        // new support must advance PAST the current closest point toward the origin. advance = FxDot(a,sdir) -
        // FxDot(closest,sdir). If it does not exceed kProgressEps the origin is unreachable beyond the current
        // feature -> SEPARATED. FIXED integer compare, consistent CPU/GPU.
        const fx advance = FxDot(a, sdir) - FxDot(closest, sdir);
        if (advance <= kProgressEps) { overlap = false; break; }

        // Push a as the NEWEST point (shift the others down).
        for (uint32_t i = n; i > 0; --i) { sp[i] = sp[i-1]; swA[i] = swA[i-1]; swB[i] = swB[i-1]; }
        sp[0] = a; swA[0] = wA; swB[0] = wB; ++n;

        // Sub-distance: reduce the simplex to the feature closest to the origin.
        SubResult sub;
        if (n == 2)      sub = DoSimplex2(sp[0], sp[1]);
        else if (n == 3) sub = DoSimplex3(sp[0], sp[1], sp[2]);
        else             sub = DoSimplex4(sp[0], sp[1], sp[2], sp[3]);

        if (sub.containsOrigin) { overlap = true;
            keepCount = sub.size; for (uint32_t i = 0; i < 4; ++i) wts[i] = sub.w[i];
            break;   // keep the full tetra as the terminal simplex (already in sp[0..n))
        }

        // Compact the simplex to the survivors (FIXED keep order) + recompute the closest point from the
        // surviving CSO points + the sub-distance weights.
        FxVec3 nsp[4], nswA[4], nswB[4]; fx nw[4] = {0,0,0,0};
        for (uint32_t i = 0; i < sub.size; ++i) {
            const uint32_t k = sub.keep[i];
            nsp[i] = sp[k]; nswA[i] = swA[k]; nswB[i] = swB[k]; nw[i] = sub.w[i];
        }
        FxVec3 newClosest{0, 0, 0};
        for (uint32_t i = 0; i < sub.size; ++i) {
            sp[i] = nsp[i]; swA[i] = nswA[i]; swB[i] = nswB[i]; wts[i] = nw[i];
            newClosest = FxAdd(newClosest, FxScale(nsp[i], nw[i]));
        }
        n = sub.size;
        keepCount = sub.size;
        closest = newClosest;
    }

    // Recover the witness points: closestA/closestB = the barycentric combine of the surviving csoA/csoB.
    // For overlap, the closest points are not meaningful (EPA's job) but we still fill them deterministically
    // from the terminal simplex's weights so the result is fully defined + byte-reproducible.
    if (!overlap) {
        // If the loop ended without a reduce having run this round (separated on the first probe), the
        // weights/keepCount reflect the last reduce; for n==1 the single vertex carries weight kOne.
        if (keepCount == 0) { keepCount = (n > 0 ? 1u : 0u); if (keepCount == 1) wts[0] = kOne; }
    }
    FxVec3 cA{0,0,0}, cB{0,0,0}, closestCso{0,0,0};
    uint32_t kc = (keepCount == 0 ? n : keepCount);
    if (kc > 4) kc = 4;
    // Renormalize/guard: if all weights are zero (defensive), put full weight on the newest survivor.
    fx wsum = 0; for (uint32_t i = 0; i < kc; ++i) wsum += wts[i];
    if (wsum == 0 && kc > 0) { wts[0] = kOne; }
    for (uint32_t i = 0; i < kc; ++i) {
        cA = FxAdd(cA, FxScale(swA[i], wts[i]));
        cB = FxAdd(cB, FxScale(swB[i], wts[i]));
        closestCso = FxAdd(closestCso, FxScale(sp[i], wts[i]));
    }

    res.overlap = overlap ? 1u : 0u;
    res.separation = closestCso;     // origin -> closest CSO point
    res.closestA = cA;
    res.closestB = cB;
    res.iterations = iter;
    res.simplex.count = n;
    for (uint32_t i = 0; i < n && i < 4; ++i) {
        res.simplex.pts[i] = sp[i]; res.simplex.csoA[i] = swA[i]; res.simplex.csoB[i] = swB[i];
    }
    res.witnessFeature = n;          // the terminal simplex size (a deterministic feature id)
    return res;
}

// ----- GjkMeasure: a deterministic summary the showcase/test asserts. Over a fixed set of (hull,body) PAIRS:
// the number of pairs, how many OVERLAP, how many SEPARATED, and the SUM of the separated closest distances
// (FxLength(separation)). A PURE function -> two MeasureGjk calls over the same inputs are byte-equal.
struct GjkMeasure {
    uint32_t pairs       = 0;   // # of hull-pair queries
    uint32_t overlapping = 0;   // pairs reporting overlap
    uint32_t separated   = 0;   // pairs reporting separation
    fx       distSum     = 0;   // Σ of the separated closest distances
};

// A GJK query pair (the showcase's deterministic hull-pair array element).
struct GjkPair {
    FxHull hullA; FxBody bodyA;
    FxHull hullB; FxBody bodyB;
};

// MeasureGjk(pairs, count): run Gjk over each pair in FIXED order, accumulate the summary. PURE — no clock/
// RNG; the SAME inputs give a byte-equal GjkMeasure every call.
inline GjkMeasure MeasureGjk(const GjkPair* pairs, uint32_t count) {
    GjkMeasure m;
    m.pairs = count;
    for (uint32_t i = 0; i < count; ++i) {
        const GjkResult r = Gjk(pairs[i].hullA, pairs[i].bodyA, pairs[i].hullB, pairs[i].bodyB);
        if (r.overlap) ++m.overlapping;
        else { ++m.separated; m.distSum += FxLength(r.separation); }
    }
    return m;
}

}  // namespace gjk
}  // namespace hf::sim
