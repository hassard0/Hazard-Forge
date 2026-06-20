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

}  // namespace gjk
}  // namespace hf::sim
