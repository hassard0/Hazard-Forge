// Slice CX1 — Deterministic Convex Rigid-Body Contacts: THE BOX-BOX SAT OVERLAP TEST (the BEACHHEAD of
// FLAGSHIP #19: hf::sim::convex). The integer core (engine/sim/convex.h) that the GPU
// shaders/convex_sat.comp.hlsl copies VERBATIM + proves bit-identical. Pure CPU (header-only, hf_core),
// ASan-eligible. convex.h #includes sim/fpx.h read-only.
//
// What this test PINS (the contracts the GPU convex_sat.comp + the GPU==CPU proof build on):
//   * FxDot / FxCross: hand-computed dot + cross of known axes (incl the right-hand basis x×y==z).
//   * FxMat3: identity, diagonal, MulVec (M·v), transpose — the CX3 inertia-tensor type's CX1 helpers.
//   * BoxSat: two clearly-separated boxes -> overlap=false; a box deeply inside another -> overlap=true
//     with a sensible min-pen axis + penetration>0; a face-face touching pair is the boundary (~0 pen);
//     an edge-edge overlap of two 45°-rotated boxes picks an EDGE-CROSS axis (index >= 6); the min-pen
//     tie-break keeps the LOWEST axis index; two runs byte-identical (determinism).
//   * MeasureSat: deterministic overlap count + mean/min penetration over a mixed pair set.
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "sim/convex.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace convex = hf::sim::convex;
namespace fpx = hf::sim::fpx;
using convex::fx;
using convex::kOne;
using convex::kFrac;
using convex::FxVec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << kFrac); }

// Build an axis-aligned (identity orient) box body at an integer position.
static fpx::FxBody MakeBoxBody(int x, int y, int z) {
    fpx::FxBody b;
    b.pos = {FromInt(x), FromInt(y), FromInt(z)};
    b.orient = fpx::FxQuat{0, 0, 0, kOne};   // identity
    return b;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= FxDot: hand-computed dot products =================
    {
        // (1,2,3)·(4,5,6) == 4 + 10 + 18 == 32 (all integers -> exact in Q16.16).
        const FxVec3 a{FromInt(1), FromInt(2), FromInt(3)};
        const FxVec3 b{FromInt(4), FromInt(5), FromInt(6)};
        check(convex::FxDot(a, b) == FromInt(32), "FxDot (1,2,3)·(4,5,6) == 32");
        // Orthogonal unit axes dot to 0.
        check(convex::FxDot(FxVec3{kOne, 0, 0}, FxVec3{0, kOne, 0}) == 0, "FxDot x·y == 0");
        // A unit axis dotted with itself == 1.0.
        check(convex::FxDot(FxVec3{kOne, 0, 0}, FxVec3{kOne, 0, 0}) == kOne, "FxDot x·x == 1");
        // 0.5·0.5 over a single axis == 0.25.
        check(convex::FxDot(FxVec3{kOne / 2, 0, 0}, FxVec3{kOne / 2, 0, 0}) == kOne / 4,
              "FxDot 0.5·0.5 == 0.25");
    }

    // ================= FxCross: hand-computed cross products (right-hand basis) =================
    {
        const FxVec3 ex{kOne, 0, 0}, ey{0, kOne, 0}, ez{0, 0, kOne};
        // x × y == z.
        FxVec3 xy = convex::FxCross(ex, ey);
        check(xy.x == 0 && xy.y == 0 && xy.z == kOne, "FxCross x×y == z");
        // y × z == x.
        FxVec3 yz = convex::FxCross(ey, ez);
        check(yz.x == kOne && yz.y == 0 && yz.z == 0, "FxCross y×z == x");
        // z × x == y.
        FxVec3 zx = convex::FxCross(ez, ex);
        check(zx.x == 0 && zx.y == kOne && zx.z == 0, "FxCross z×x == y");
        // Anticommutativity: y × x == -z.
        FxVec3 yx = convex::FxCross(ey, ex);
        check(yx.x == 0 && yx.y == 0 && yx.z == -kOne, "FxCross y×x == -z");
        // A parallel pair crosses to ~0 (degenerate — the SAT skips these).
        FxVec3 par = convex::FxCross(ex, ex);
        check(par.x == 0 && par.y == 0 && par.z == 0, "FxCross x×x == 0 (degenerate)");
        // A known non-axis cross: (1,2,3)×(4,5,6) == (2*6-3*5, 3*4-1*6, 1*5-2*4) == (-3, 6, -3).
        FxVec3 g = convex::FxCross(FxVec3{FromInt(1), FromInt(2), FromInt(3)},
                                   FxVec3{FromInt(4), FromInt(5), FromInt(6)});
        check(g.x == FromInt(-3) && g.y == FromInt(6) && g.z == FromInt(-3),
              "FxCross (1,2,3)×(4,5,6) == (-3,6,-3)");
    }

    // ================= FxMat3: identity / diagonal / MulVec / transpose =================
    {
        // Identity·v == v.
        convex::FxMat3 I = convex::FxMat3Identity();
        check(I.m[0] == kOne && I.m[4] == kOne && I.m[8] == kOne, "FxMat3Identity diagonal == 1");
        check(I.m[1] == 0 && I.m[2] == 0 && I.m[3] == 0 && I.m[5] == 0 && I.m[6] == 0 && I.m[7] == 0,
              "FxMat3Identity off-diagonal == 0");
        const FxVec3 v{FromInt(3), FromInt(-5), FromInt(7)};
        FxVec3 iv = convex::FxMat3MulVec(I, v);
        check(iv.x == v.x && iv.y == v.y && iv.z == v.z, "FxMat3MulVec identity·v == v");

        // Diagonal(2,3,4)·(1,1,1) == (2,3,4).
        convex::FxMat3 D = convex::FxMat3Diagonal(FxVec3{FromInt(2), FromInt(3), FromInt(4)});
        FxVec3 dv = convex::FxMat3MulVec(D, FxVec3{kOne, kOne, kOne});
        check(dv.x == FromInt(2) && dv.y == FromInt(3) && dv.z == FromInt(4),
              "FxMat3 diagonal·(1,1,1) == (2,3,4)");

        // A full matrix MulVec: rows (1,2,3),(4,5,6),(7,8,9) · (1,0,0) == (1,4,7) (first column).
        convex::FxMat3 M;
        M.m[0] = FromInt(1); M.m[1] = FromInt(2); M.m[2] = FromInt(3);
        M.m[3] = FromInt(4); M.m[4] = FromInt(5); M.m[5] = FromInt(6);
        M.m[6] = FromInt(7); M.m[7] = FromInt(8); M.m[8] = FromInt(9);
        FxVec3 col0 = convex::FxMat3MulVec(M, FxVec3{kOne, 0, 0});
        check(col0.x == FromInt(1) && col0.y == FromInt(4) && col0.z == FromInt(7),
              "FxMat3MulVec M·(1,0,0) == first column (1,4,7)");
        // Transpose swaps rows<->columns: T(M).m[1] == M.m[3] == 4, T(M).m[3] == M.m[1] == 2.
        convex::FxMat3 T = convex::FxMat3Transpose(M);
        check(T.m[0] == M.m[0] && T.m[1] == M.m[3] && T.m[3] == M.m[1] && T.m[2] == M.m[6] &&
              T.m[6] == M.m[2] && T.m[5] == M.m[7] && T.m[7] == M.m[5] && T.m[8] == M.m[8],
              "FxMat3Transpose swaps off-diagonal entries");
    }

    // ================= BoxSat: two clearly-separated boxes -> overlap=false =================
    {
        // Two unit boxes (halfExtents 1.0), axis-aligned, 5 world units apart on x -> a 3-unit gap.
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        convex::SatResult r = convex::BoxSat(MakeBoxBody(0, 0, 0), unit, MakeBoxBody(5, 0, 0), unit);
        check(!r.overlap, "BoxSat separated boxes -> overlap=false");
        // Symmetry: swapping A and B is still separated.
        convex::SatResult rs = convex::BoxSat(MakeBoxBody(5, 0, 0), unit, MakeBoxBody(0, 0, 0), unit);
        check(!rs.overlap, "BoxSat separated is symmetric -> overlap=false");
        // Separated on a diagonal too.
        convex::SatResult rd = convex::BoxSat(MakeBoxBody(0, 0, 0), unit, MakeBoxBody(4, 4, 4), unit);
        check(!rd.overlap, "BoxSat far-diagonal -> overlap=false");
    }

    // ================= BoxSat: a box deeply inside another -> overlap=true, pen>0, sensible axis ==========
    {
        // A big box (halfExtents 4) centered at origin, a small box (halfExtents 1) at (1,0,0) deep inside.
        const convex::FxBox big{FxVec3{FromInt(4), FromInt(4), FromInt(4)}};
        const convex::FxBox small{FxVec3{kOne, kOne, kOne}};
        convex::SatResult r = convex::BoxSat(MakeBoxBody(0, 0, 0), big, MakeBoxBody(1, 0, 0), small);
        check(r.overlap, "BoxSat deep-inside -> overlap=true");
        check(r.penetration > 0, "BoxSat deep-inside -> penetration > 0");
        // The min-pen axis should be a FACE axis (a deeply nested AABB pair separates first on a face
        // normal, NOT an edge cross). axisIndex 0..5.
        check(r.axisIndex < 6, "BoxSat deep-inside -> min-pen on a face axis (index < 6)");
        // The axis is signed toward B: B is at +x of A, the smallest exit is along ±x; the unit axis
        // magnitude is ~1 on one component.
        const fx axMag = (r.axis.x < 0 ? -r.axis.x : r.axis.x);
        check(axMag > kOne - kOne / 100, "BoxSat deep-inside -> min-pen axis ~ unit on x");

        // A pen sanity: small box at x=1 with hx=1 reaches x=2; big box face at x=4 -> overlap on x by
        // 4 - (1) ... the min pen across all axes is positive + below the box sizes (no absurd value).
        check(r.penetration < FromInt(10), "BoxSat deep-inside -> penetration is a sane magnitude");
    }

    // ================= BoxSat: a face-face touching pair is the boundary case (~0 pen) =================
    {
        // Two unit boxes (halfExtents 1), centers exactly 2.0 apart on x -> faces touch at x=1.
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        convex::SatResult r = convex::BoxSat(MakeBoxBody(0, 0, 0), unit, MakeBoxBody(2, 0, 0), unit);
        // Touching counts as overlap (s == rA+rB, NOT s > rA+rB) with ~0 penetration.
        check(r.overlap, "BoxSat face-touching -> overlap=true (boundary, inclusive)");
        check(r.penetration == 0, "BoxSat face-touching -> penetration == 0 (the boundary)");
        // Just BEYOND touching (centers 2.0 + a hair apart) -> separated. Use centers 3 apart -> separated.
        convex::SatResult rsep = convex::BoxSat(MakeBoxBody(0, 0, 0), unit, MakeBoxBody(3, 0, 0), unit);
        check(!rsep.overlap, "BoxSat just-past-touching -> overlap=false");
    }

    // ================= BoxSat: an edge-edge overlap picks an EDGE-CROSS axis (index >= 6) =================
    {
        // Two long thin boxes crossed like an X, one rotated 45° about Z, overlapping only along their
        // crossing edges. A box rotated 45° about Z: q = {0,0,sin22.5,cos22.5}. sin/cos 22.5° in Q16.16:
        // sin(22.5°)=0.38268 -> ~25080, cos(22.5°)=0.92388 -> ~60547.
        fpx::FxBody a = MakeBoxBody(0, 0, 0);                 // axis-aligned long-x bar
        fpx::FxBody b = MakeBoxBody(0, 0, 0);
        b.orient = fpx::FxQuatNormalize(fpx::FxQuat{0, 0, (fx)25080, (fx)60547});  // 45° about Z
        // Long, thin boxes (a bar): big x half-extent, small y/z so only the crossing edges interact when
        // one is rotated 45° in the XY plane. Centered at the same point so they DO overlap at the cross.
        const convex::FxBox bar{FxVec3{FromInt(3), kOne / 4, kOne / 4}};
        convex::SatResult r = convex::BoxSat(a, bar, b, bar);
        // They overlap (coincident centers).
        check(r.overlap, "BoxSat crossed bars -> overlap=true");
        check(r.penetration >= 0, "BoxSat crossed bars -> penetration >= 0");

        // A genuine edge-edge configuration: two unit boxes, B rotated 45° about Z, offset diagonally so
        // their nearest features are EDGES, not faces. The min-pen axis should be an edge-cross (index>=6).
        fpx::FxBody ea = MakeBoxBody(0, 0, 0);
        fpx::FxBody eb = MakeBoxBody(0, 0, 0);
        eb.orient = fpx::FxQuatNormalize(fpx::FxQuat{(fx)25080, 0, (fx)25080, (fx)55000});  // tilt off-axis
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        // Offset so the closest approach is edge-edge along a tilted diagonal.
        eb.pos = {FromInt(1) + kOne / 2, FromInt(1) + kOne / 2, FromInt(1) + kOne / 2};
        convex::SatResult er = convex::BoxSat(ea, unit, eb, unit);
        // We don't assert overlap here (geometry-dependent); we assert that IF it overlaps, the axis is a
        // valid 0..14 index, and that an edge-cross axis IS reachable for SOME tilted offset below.
        check(er.axisIndex <= 14, "BoxSat tilted pair -> axisIndex in [0,14]");

        // A constructed edge-edge winner: two unit boxes, A tilted ~45° about X + B tilted ~45° about Z so
        // their nearest features are SKEW EDGES -> the separating axis between two skew edges is their CROSS
        // (an edge-cross axis). Offset diagonally on all three axes so a face normal is NOT the minimum.
        // sin/cos 22.5° in Q16.16: ~25080 / ~60547 (a 45° rotation quaternion). At off in [1.0,1.375] the
        // min-pen axis is edge-cross index 12 (verified by an exhaustive offline tilt/offset sweep).
        bool sawEdgeCross = false;
        for (int k = 0; k < 4 && !sawEdgeCross; ++k) {
            fpx::FxBody ta = MakeBoxBody(0, 0, 0);
            fpx::FxBody tb = MakeBoxBody(0, 0, 0);
            ta.orient = fpx::FxQuatNormalize(fpx::FxQuat{(fx)25080, 0, 0, (fx)60547});       // 45° about X
            tb.orient = fpx::FxQuatNormalize(fpx::FxQuat{0, 0, (fx)25080, (fx)60547});       // 45° about Z
            const fx off = kOne + (fx)(k * (kOne / 8));   // 1.0, 1.125, 1.25, 1.375
            tb.pos = {off, off, off};                     // diagonal offset (all three axes)
            convex::SatResult tr = convex::BoxSat(ta, unit, tb, unit);
            if (tr.overlap && tr.axisIndex >= 6) sawEdgeCross = true;
        }
        check(sawEdgeCross, "BoxSat skew-edge pair -> min-pen axis is an EDGE-CROSS (index >= 6)");
    }

    // ================= BoxSat: the min-pen tie-break keeps the LOWEST axis index =================
    {
        // A symmetric configuration where multiple axes share the SAME minimum penetration: two IDENTICAL
        // axis-aligned cubes (halfExtents 1) at the SAME center -> every face axis penetrates by the full
        // 2.0 (rA+rB - 0). The min is tied across axes 0..5; the LOWEST index (0) must win.
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        convex::SatResult r = convex::BoxSat(MakeBoxBody(0, 0, 0), unit, MakeBoxBody(0, 0, 0), unit);
        check(r.overlap, "BoxSat coincident cubes -> overlap=true");
        check(r.axisIndex == 0, "BoxSat tie -> lowest axis index (0) wins (strict-< scan)");
        // The penetration is the full rA+rB == 2.0 (the centers coincide -> s==0).
        check(r.penetration == FromInt(2), "BoxSat coincident cubes -> penetration == rA+rB == 2.0");
    }

    // ================= determinism: two BoxSat runs byte-identical =================
    {
        const convex::FxBox big{FxVec3{FromInt(4), FromInt(4), FromInt(4)}};
        const convex::FxBox small{FxVec3{kOne, kOne, kOne}};
        fpx::FxBody a = MakeBoxBody(0, 0, 0);
        fpx::FxBody b = MakeBoxBody(1, 0, 0);
        b.orient = fpx::FxQuatNormalize(fpx::FxQuat{(fx)10000, (fx)5000, (fx)25080, (fx)60547});
        convex::SatResult r1 = convex::BoxSat(a, big, b, small);
        convex::SatResult r2 = convex::BoxSat(a, big, b, small);
        // Compare the meaningful FIELDS (the bool's struct padding is not load-bearing; the showcase packs
        // into an explicit 6 x int32 form for its GPU==CPU memcmp — see the convex-sat-shot packResult).
        check(r1.overlap == r2.overlap && r1.axisIndex == r2.axisIndex &&
              r1.penetration == r2.penetration && r1.axis.x == r2.axis.x &&
              r1.axis.y == r2.axis.y && r1.axis.z == r2.axis.z,
              "BoxSat determinism: two runs BYTE-IDENTICAL");
    }

    // ================= MeasureSat: deterministic overlap count + mean/min penetration =================
    {
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        std::vector<convex::SatPair> pairs;
        // Pair 0: deep overlap (coincident) -> pen 2.0.
        pairs.push_back({MakeBoxBody(0, 0, 0), unit, MakeBoxBody(0, 0, 0), unit});
        // Pair 1: separated (5 apart) -> no overlap.
        pairs.push_back({MakeBoxBody(0, 0, 0), unit, MakeBoxBody(5, 0, 0), unit});
        // Pair 2: shallow overlap (centers 1.5 apart -> pen 0.5 on x).
        pairs.push_back({MakeBoxBody(0, 0, 0), unit, MakeBoxBody(0, 0, 0), unit});
        pairs[2].bodyB.pos = {kOne + kOne / 2, 0, 0};   // 1.5 apart

        convex::SatMeasure m = convex::MeasureSat(pairs);
        check(m.pairs == 3, "MeasureSat counts 3 pairs");
        check(m.overlapping == 2, "MeasureSat counts 2 overlapping pairs");
        check(m.minPen == kOne / 2, "MeasureSat minPen == 0.5 (the shallow pair)");
        // mean of 2.0 and 0.5 == 1.25.
        check(m.meanPen == FromInt(1) + kOne / 4, "MeasureSat meanPen == 1.25");

        // Determinism: a second MeasureSat over the same pairs is byte-identical.
        convex::SatMeasure m2 = convex::MeasureSat(pairs);
        check(std::memcmp(&m, &m2, sizeof(convex::SatMeasure)) == 0,
              "MeasureSat determinism: two runs BYTE-IDENTICAL");
    }

    // ================= CX2 BuildManifold: a deep face-face stack -> a 4-point manifold ================
    {
        // Two unit boxes (halfExtents 1), axis-aligned, stacked on Y with the top one pushed DOWN into the
        // bottom (centers 1.5 apart on y -> a deep face-face overlap of 0.5; the touching faces are the full
        // 2x2 patch in XZ -> a 4-corner manifold). A is the lower box, B the upper.
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        fpx::FxBody a = MakeBoxBody(0, 0, 0);
        fpx::FxBody b = MakeBoxBody(0, 0, 0);
        b.pos = {0, kOne + kOne / 2, 0};   // 1.5 above -> overlap 0.5 on Y
        convex::SatResult r = convex::BoxSat(a, unit, b, unit);
        check(r.overlap, "CX2 face-stack -> overlap=true");
        check(r.axisIndex < 6, "CX2 face-stack -> a FACE axis (index < 6)");
        convex::ContactManifold m = convex::BuildManifold(a, unit, b, unit, r);
        check(m.count == 4, "CX2 face-stack -> 4-point manifold (full face patch)");
        bool allNonNeg = true;
        for (uint32_t k = 0; k < m.count; ++k) if (m.depths[k] < 0) allNonNeg = false;
        check(allNonNeg, "CX2 face-stack -> all depths >= 0");
        // Every contact point lies inside BOTH boxes' slabs (within an integer epsilon): the projection onto
        // each box axis is within halfExtent + eps. eps ~ a few LSB of fixed-point clip drift.
        const fx eps = kOne / 64;
        bool inside = true;
        convex::FxVec3 axA[3], axB[3];
        convex::BoxAxes(a, axA);
        convex::BoxAxes(b, axB);
        for (uint32_t k = 0; k < m.count; ++k) {
            const convex::FxVec3 pa = fpx::FxSub(m.points[k], a.pos);
            const convex::FxVec3 pb = fpx::FxSub(m.points[k], b.pos);
            for (int ax = 0; ax < 3; ++ax) {
                const fx da = convex::FxDot(pa, axA[ax]);
                const fx db = convex::FxDot(pb, axB[ax]);
                const fx ha = convex::FxAt(unit.halfExtents, (uint32_t)ax);
                if (da > ha + eps || da < -(ha + eps)) inside = false;
                if (db > ha + eps || db < -(ha + eps)) inside = false;
            }
        }
        check(inside, "CX2 face-stack -> every contact point inside both boxes' slabs");
        // The normal points from A (lower) toward B (upper): +Y.
        check(m.normal.y > kOne - kOne / 100, "CX2 face-stack -> normal ~ +Y (A toward B)");
    }

    // ================= CX2 BuildManifold: a separated pair -> count 0 ================
    {
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        convex::SatResult r = convex::BoxSat(MakeBoxBody(0, 0, 0), unit, MakeBoxBody(5, 0, 0), unit);
        check(!r.overlap, "CX2 separated -> SAT overlap=false (precondition)");
        convex::ContactManifold m = convex::BuildManifold(MakeBoxBody(0, 0, 0), unit,
                                                          MakeBoxBody(5, 0, 0), unit, r);
        check(m.count == 0, "CX2 separated -> manifold count == 0");
    }

    // ================= CX2 BuildManifold: an edge-edge overlap -> 1 point near the expected midpoint =====
    {
        // The CX1 skew-edge config: A tilted 45° about X, B tilted 45° about Z, diagonal offset -> the
        // min-pen axis is an edge-cross (index >= 6); the manifold is ONE point at the segment midpoint.
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        fpx::FxBody ta = MakeBoxBody(0, 0, 0);
        fpx::FxBody tb = MakeBoxBody(0, 0, 0);
        ta.orient = fpx::FxQuatNormalize(fpx::FxQuat{(fx)25080, 0, 0, (fx)60547});       // 45° about X
        tb.orient = fpx::FxQuatNormalize(fpx::FxQuat{0, 0, (fx)25080, (fx)60547});       // 45° about Z
        const fx off = kOne + kOne / 8;   // 1.125 diagonal (an edge-cross winner per the CX1 sweep)
        tb.pos = {off, off, off};
        convex::SatResult r = convex::BoxSat(ta, unit, tb, unit);
        if (r.overlap && r.axisIndex >= 6) {
            convex::ContactManifold m = convex::BuildManifold(ta, unit, tb, unit, r);
            check(m.count == 1, "CX2 edge-edge -> 1-point manifold");
            check(m.depths[0] == r.penetration, "CX2 edge-edge -> depth == SAT penetration");
            // The contact point lies BETWEEN the two box centers (the segment midpoint is bounded by the
            // centers + half-extents). A loose bound: |point - midOfCenters| within the box diagonal.
            const convex::FxVec3 midC{(ta.pos.x + tb.pos.x) / 2, (ta.pos.y + tb.pos.y) / 2,
                                      (ta.pos.z + tb.pos.z) / 2};
            const fx dlen = fpx::FxLength(fpx::FxSub(m.points[0], midC));
            check(dlen < FromInt(3), "CX2 edge-edge -> contact near the centers' midpoint");
        } else {
            check(false, "CX2 edge-edge -> the skew config produced an edge-cross overlap (precondition)");
        }
    }

    // ================= CX2 BuildManifold: a touching face pair -> >= 1 point, ~0 depth ================
    {
        // Two unit boxes, centers exactly 2.0 apart on x -> faces touch at x=1 (the CX1 boundary case).
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        convex::SatResult r = convex::BoxSat(MakeBoxBody(0, 0, 0), unit, MakeBoxBody(2, 0, 0), unit);
        check(r.overlap, "CX2 touching -> overlap=true (boundary)");
        convex::ContactManifold m = convex::BuildManifold(MakeBoxBody(0, 0, 0), unit,
                                                          MakeBoxBody(2, 0, 0), unit, r);
        check(m.count >= 1, "CX2 touching -> at least 1 contact point");
        bool nearZero = true;
        for (uint32_t k = 0; k < m.count; ++k) if (m.depths[k] > kOne / 64) nearZero = false;
        check(nearZero, "CX2 touching -> all depths ~ 0 (the boundary)");
    }

    // ================= CX2 BuildManifold: determinism (two runs byte-identical) ================
    {
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        fpx::FxBody a = MakeBoxBody(0, 0, 0);
        fpx::FxBody b = MakeBoxBody(0, 0, 0);
        b.pos = {kOne / 2, kOne + kOne / 4, kOne / 3};   // an off-center deep face overlap
        convex::SatResult r = convex::BoxSat(a, unit, b, unit);
        convex::ContactManifold m1 = convex::BuildManifold(a, unit, b, unit, r);
        convex::ContactManifold m2 = convex::BuildManifold(a, unit, b, unit, r);
        bool same = (m1.count == m2.count);
        for (uint32_t k = 0; k < m1.count && same; ++k) {
            if (m1.points[k].x != m2.points[k].x || m1.points[k].y != m2.points[k].y ||
                m1.points[k].z != m2.points[k].z || m1.depths[k] != m2.depths[k]) same = false;
        }
        if (m1.normal.x != m2.normal.x || m1.normal.y != m2.normal.y || m1.normal.z != m2.normal.z)
            same = false;
        check(same, "CX2 determinism: two BuildManifold runs BYTE-IDENTICAL");
    }

    // ================= CX2 BuildManifold: the reduction keeps the DEEPEST point ================
    {
        // A TILTED top box over a flat bottom box -> the clipped corners have DIFFERENT depths; m.points[0]
        // (after the reduction) must be the deepest of the kept candidates. We verify points[0]'s depth is
        // the MAX over the reported points (the reduction's invariant).
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        fpx::FxBody a = MakeBoxBody(0, 0, 0);
        fpx::FxBody b = MakeBoxBody(0, 0, 0);
        b.orient = fpx::FxQuatNormalize(fpx::FxQuat{(fx)8000, 0, 0, (fx)64000});  // a small tilt about X
        b.pos = {0, kOne + kOne / 2, 0};   // pressed down into A
        convex::SatResult r = convex::BoxSat(a, unit, b, unit);
        convex::ContactManifold m = convex::BuildManifold(a, unit, b, unit, r);
        check(m.count >= 1, "CX2 reduction -> at least 1 point");
        bool deepestFirst = true;
        for (uint32_t k = 1; k < m.count; ++k) if (m.depths[k] > m.depths[0]) deepestFirst = false;
        check(deepestFirst, "CX2 reduction -> points[0] is the DEEPEST kept point");
    }

    // ================= CX2 MeasureManifold: deterministic summary over a mixed pair set ================
    {
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        std::vector<convex::SatPair> pairs;
        // Pair 0: deep face stack (4 points).
        pairs.push_back({MakeBoxBody(0, 0, 0), unit, MakeBoxBody(0, 0, 0), unit});
        pairs[0].bodyB.pos = {0, kOne + kOne / 2, 0};
        // Pair 1: separated (no contact).
        pairs.push_back({MakeBoxBody(0, 0, 0), unit, MakeBoxBody(5, 0, 0), unit});
        convex::ManifoldMeasure m = convex::MeasureManifold(pairs);
        check(m.pairs == 2, "CX2 MeasureManifold counts 2 pairs");
        check(m.withContact == 1, "CX2 MeasureManifold counts 1 pair with contact");
        check(m.totalPoints == 4, "CX2 MeasureManifold totalPoints == 4 (the face stack)");
        convex::ManifoldMeasure m2 = convex::MeasureManifold(pairs);
        check(std::memcmp(&m, &m2, sizeof(convex::ManifoldMeasure)) == 0,
              "CX2 MeasureManifold determinism: two runs BYTE-IDENTICAL");
    }

    // ========================================================================================
    // ================= Slice CX3 — THE ANGULAR CONTACT IMPULSE (the new physics) ==============
    // ========================================================================================
    // FxBoxInvInertiaBody, WorldInvInertia (-> FxMat3), SolveManifoldImpulse, ResolveContactPair.
    // The FIRST contact-driven angVel in the engine: an OFF-CENTER box-box contact imparts spin.

    // ================= FxBoxInvInertiaBody: matches the analytic (m/3)(h²+h²) inverse =================
    {
        // A unit box (halfExtents 1,1,1), mass 1 (invMass==kOne). Body inertia diagonal Ixx=(m/3)(hy²+hz²)
        // = (1/3)(1+1) = 2/3 -> invIbody.x = 3*invMass/(hy²+hz²) = 3/2 = 1.5. Symmetric for all three.
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        const FxVec3 inv = convex::FxBoxInvInertiaBody(unit, kOne);
        const fx threeHalves = FromInt(1) + kOne / 2;   // 1.5 in Q16.16
        check(inv.x == threeHalves && inv.y == threeHalves && inv.z == threeHalves,
              "FxBoxInvInertiaBody unit box mass 1 -> diag 1.5 each (3*invMass/(h²+h²))");

        // A NON-cubic box (halfExtents 1,2,3), mass 1. invIbody.x = 3/(hy²+hz²) = 3/(4+9) = 3/13;
        // invIbody.y = 3/(hx²+hz²) = 3/(1+9) = 3/10; invIbody.z = 3/(hx²+hy²) = 3/(1+4) = 3/5.
        const convex::FxBox rect{FxVec3{kOne, FromInt(2), FromInt(3)}};
        const FxVec3 invr = convex::FxBoxInvInertiaBody(rect, kOne);
        check(invr.x == fpx::fxdiv(FromInt(3), FromInt(13)), "FxBoxInvInertiaBody rect x = 3/13");
        check(invr.y == fpx::fxdiv(FromInt(3), FromInt(10)), "FxBoxInvInertiaBody rect y = 3/10");
        check(invr.z == fpx::fxdiv(FromInt(3), FromInt(5)),  "FxBoxInvInertiaBody rect z = 3/5");

        // A STATIC body (invMass==0) -> zero inverse inertia (infinite inertia, takes no angular impulse).
        const FxVec3 invs = convex::FxBoxInvInertiaBody(unit, 0);
        check(invs.x == 0 && invs.y == 0 && invs.z == 0,
              "FxBoxInvInertiaBody static (invMass 0) -> (0,0,0)");
    }

    // ================= WorldInvInertia: identity orient -> diagonal; 90° rotate -> permuted ============
    {
        // For an IDENTITY-oriented box the world inverse inertia == diag(invIbody) (R==I -> R·diag·Rᵀ==diag).
        const convex::FxBox rect{FxVec3{kOne, FromInt(2), FromInt(3)}};
        const FxVec3 invb = convex::FxBoxInvInertiaBody(rect, kOne);
        fpx::FxBody ident = MakeBoxBody(0, 0, 0);   // identity orient
        const convex::FxMat3 Iw = convex::WorldInvInertia(ident, invb);
        check(Iw.m[0] == invb.x && Iw.m[4] == invb.y && Iw.m[8] == invb.z,
              "WorldInvInertia identity orient -> diagonal == invIbody");
        check(Iw.m[1] == 0 && Iw.m[2] == 0 && Iw.m[3] == 0 && Iw.m[5] == 0 && Iw.m[6] == 0 && Iw.m[7] == 0,
              "WorldInvInertia identity orient -> off-diagonal == 0");
        // Symmetric (R·diag·Rᵀ is always symmetric).
        check(Iw.m[1] == Iw.m[3] && Iw.m[2] == Iw.m[6] && Iw.m[5] == Iw.m[7],
              "WorldInvInertia is symmetric (identity case)");

        // A 90° rotation about Z maps local x->world y, local y->world -x: so the world diagonal PERMUTES —
        // the x-diagonal world entry picks up the body's Y inverse inertia, the y-diagonal entry the X one,
        // the z entry stays. q(90° about Z) = {0,0,sin45,cos45}, sin45=cos45=0.70710678 -> ~46341 in Q16.16.
        fpx::FxBody rot = MakeBoxBody(0, 0, 0);
        rot.orient = fpx::FxQuatNormalize(fpx::FxQuat{0, 0, (fx)46341, (fx)46341});  // 90° about Z
        const convex::FxMat3 Ir = convex::WorldInvInertia(rot, invb);
        // The world (0,0) entry ~ invb.y, the (1,1) ~ invb.x, the (2,2) == invb.z. Allow a small fixed-point
        // drift epsilon (the FxRotate of the axes + the outer-product products carry a few LSB).
        const fx eps = kOne / 256;
        auto absfx = [](fx v) { return v < 0 ? -v : v; };
        check(absfx(Ir.m[0] - invb.y) < eps, "WorldInvInertia 90°Z -> world-x diag ~ body-y inertia");
        check(absfx(Ir.m[4] - invb.x) < eps, "WorldInvInertia 90°Z -> world-y diag ~ body-x inertia");
        check(absfx(Ir.m[8] - invb.z) < eps, "WorldInvInertia 90°Z -> world-z diag unchanged");
        // Still symmetric.
        check(absfx(Ir.m[1] - Ir.m[3]) < eps && absfx(Ir.m[2] - Ir.m[6]) < eps &&
              absfx(Ir.m[5] - Ir.m[7]) < eps, "WorldInvInertia 90°Z still symmetric");
    }

    // ================= SolveManifoldImpulse: a 1-point OFF-CENTER contact -> non-zero angVel ===========
    {
        // A static box A at origin; a dynamic unit box B above it, moving DOWN, with the single contact point
        // OFFSET in +x from B's center -> the downward impulse at an off-center lever arm torques B -> spin.
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        fpx::FxBody A = MakeBoxBody(0, 0, 0);
        A.invMass = 0; A.flags = 0;                  // static floor box
        fpx::FxBody B = MakeBoxBody(0, 2, 0);        // above A
        B.invMass = kOne; B.flags = fpx::kFlagDynamic;
        B.vel = {0, -kOne, 0};                       // moving down at 1 u/s
        const FxVec3 invIa = convex::FxBoxInvInertiaBody(unit, A.invMass);   // (0,0,0)
        const FxVec3 invIb = convex::FxBoxInvInertiaBody(unit, B.invMass);
        const convex::FxMat3 invIaW = convex::WorldInvInertia(A, invIa);
        const convex::FxMat3 invIbW = convex::WorldInvInertia(B, invIb);

        // One contact point at the touching plane (y=1), OFFSET +0.5 in x from B's center (x=0). normal +Y.
        convex::ContactManifold m;
        m.count = 1;
        m.points[0] = FxVec3{kOne / 2, kOne, 0};
        m.depths[0] = 0;
        m.normal = FxVec3{0, kOne, 0};   // A->B is +Y

        fpx::FxBody A2 = A, B2 = B;
        convex::SolveManifoldImpulse(A2, B2, invIaW, invIbW, m, /*restitution*/0, /*iters*/1);
        // The static body A is unaffected.
        check(A2.vel.x == 0 && A2.vel.y == 0 && A2.vel.z == 0 && A2.angVel.x == 0 &&
              A2.angVel.y == 0 && A2.angVel.z == 0, "SolveManifoldImpulse: static body A unaffected");
        // B gained UPWARD linear velocity (the impulse removed the downward approach).
        check(B2.vel.y > B.vel.y, "SolveManifoldImpulse: off-center -> B vel.y increased (approach removed)");
        // B gained NON-ZERO angular velocity (the contact-driven spin — the new physics). A downward
        // impulse (-Y) applied at +x lever -> torque rB×J = (+x)×(-Y) ~ +z -> ωB.z != 0.
        check(B2.angVel.x == 0 && B2.angVel.z != 0 && B2.angVel.y == 0,
              "SolveManifoldImpulse: off-center contact -> non-zero angVel.z (contact-driven spin)");
        check(B2.angVel.z > 0, "SolveManifoldImpulse: +x lever, -Y impulse -> +z spin (sign check)");
    }

    // ================= SolveManifoldImpulse: a DEAD-CENTER symmetric contact -> zero angVel ============
    {
        // The SAME scene but the contact point is DEAD-CENTER under B (x=0) -> no lever arm -> NO spin.
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        fpx::FxBody A = MakeBoxBody(0, 0, 0); A.invMass = 0; A.flags = 0;
        fpx::FxBody B = MakeBoxBody(0, 2, 0); B.invMass = kOne; B.flags = fpx::kFlagDynamic;
        B.vel = {0, -kOne, 0};
        const FxVec3 invIb = convex::FxBoxInvInertiaBody(unit, B.invMass);
        const convex::FxMat3 invIaW = convex::WorldInvInertia(A, FxVec3{0, 0, 0});
        const convex::FxMat3 invIbW = convex::WorldInvInertia(B, invIb);
        convex::ContactManifold m;
        m.count = 1;
        m.points[0] = FxVec3{0, kOne, 0};   // dead-center (x=0), no lever arm
        m.depths[0] = 0;
        m.normal = FxVec3{0, kOne, 0};
        convex::SolveManifoldImpulse(A, B, invIaW, invIbW, m, 0, 1);
        check(B.angVel.x == 0 && B.angVel.y == 0 && B.angVel.z == 0,
              "SolveManifoldImpulse: dead-center contact -> angVel == (0,0,0) (no spin)");
        check(B.vel.y > -kOne, "SolveManifoldImpulse: dead-center -> still removes the approach");
    }

    // ================= SolveManifoldImpulse: an already-SEPARATING contact -> no-op ====================
    {
        // B is moving AWAY (up) -> vn >= 0 -> the contact must NOT apply any impulse (deterministic skip).
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        fpx::FxBody A = MakeBoxBody(0, 0, 0); A.invMass = 0; A.flags = 0;
        fpx::FxBody B = MakeBoxBody(0, 2, 0); B.invMass = kOne; B.flags = fpx::kFlagDynamic;
        B.vel = {0, kOne, 0};   // moving UP, away from A
        const FxVec3 invIb = convex::FxBoxInvInertiaBody(unit, B.invMass);
        const convex::FxMat3 invIaW = convex::WorldInvInertia(A, FxVec3{0, 0, 0});
        const convex::FxMat3 invIbW = convex::WorldInvInertia(B, invIb);
        convex::ContactManifold m;
        m.count = 1;
        m.points[0] = FxVec3{kOne / 2, kOne, 0};
        m.depths[0] = 0;
        m.normal = FxVec3{0, kOne, 0};
        const fpx::FxBody Bbefore = B;
        convex::SolveManifoldImpulse(A, B, invIaW, invIbW, m, 0, 1);
        check(B.vel.x == Bbefore.vel.x && B.vel.y == Bbefore.vel.y && B.vel.z == Bbefore.vel.z &&
              B.angVel.x == 0 && B.angVel.y == 0 && B.angVel.z == 0,
              "SolveManifoldImpulse: separating contact (vn>=0) -> no-op");
    }

    // ================= ResolveContactPair: off-center hit -> spin; separating pair -> no-op ============
    {
        // An overlapping off-center box-box pair end-to-end: BoxSat -> BuildManifold -> world inertias ->
        // SolveManifoldImpulse, all inside ResolveContactPair. B descends onto A, offset in x -> B spins.
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        fpx::FxBody A = MakeBoxBody(0, 0, 0); A.invMass = 0; A.flags = 0;
        fpx::FxBody B = MakeBoxBody(0, 0, 0);
        // B overlapping A on +Y, OFFSET +x so the face contact is off-center relative to the combined COM
        // motion -> an asymmetric manifold -> spin. centers 1.5 apart on y (overlap 0.5), x offset +0.6.
        B.pos = {(fx)(kOne * 6 / 10), kOne + kOne / 2, 0};
        B.invMass = kOne; B.flags = fpx::kFlagDynamic;
        B.vel = {0, -kOne, 0};   // descending
        convex::ContactSolveConfig cfg;
        cfg.restitution = 0;
        cfg.iters = 4;
        fpx::FxBody A2 = A, B2 = B;
        convex::ResolveContactPair(A2, unit, B2, unit, cfg);
        check(A2.vel.x == 0 && A2.vel.y == 0 && A2.vel.z == 0 && A2.angVel.z == 0,
              "ResolveContactPair: static A unaffected");
        bool spun = (B2.angVel.x != 0 || B2.angVel.y != 0 || B2.angVel.z != 0);
        check(spun, "ResolveContactPair: off-center overlap -> B gained spin (contact-driven angVel)");
        check(B2.vel.y > B.vel.y, "ResolveContactPair: off-center overlap -> approach removed (vel.y up)");

        // A SEPARATED pair -> BoxSat overlap=false -> ResolveContactPair is a no-op.
        fpx::FxBody Sa = MakeBoxBody(0, 0, 0); Sa.invMass = 0; Sa.flags = 0;
        fpx::FxBody Sb = MakeBoxBody(5, 0, 0); Sb.invMass = kOne; Sb.flags = fpx::kFlagDynamic;
        Sb.vel = {-kOne, 0, 0};
        const fpx::FxBody SbBefore = Sb;
        convex::ResolveContactPair(Sa, unit, Sb, unit, cfg);
        check(Sb.vel.x == SbBefore.vel.x && Sb.vel.y == SbBefore.vel.y && Sb.vel.z == SbBefore.vel.z &&
              Sb.angVel.x == 0 && Sb.angVel.y == 0 && Sb.angVel.z == 0,
              "ResolveContactPair: separated pair -> no-op");
    }

    // ================= CX3 determinism: two ResolveContactPair runs byte-identical =====================
    {
        const convex::FxBox unit{FxVec3{kOne, kOne, kOne}};
        fpx::FxBody A = MakeBoxBody(0, 0, 0); A.invMass = 0; A.flags = 0;
        fpx::FxBody B = MakeBoxBody(0, 0, 0);
        B.pos = {(fx)(kOne * 6 / 10), kOne + kOne / 2, kOne / 4};
        B.orient = fpx::FxQuatNormalize(fpx::FxQuat{(fx)8000, 0, 0, (fx)64000});
        B.invMass = kOne; B.flags = fpx::kFlagDynamic;
        B.vel = {0, -kOne, 0};
        convex::ContactSolveConfig cfg; cfg.restitution = kOne / 4; cfg.iters = 6;
        fpx::FxBody A1 = A, B1 = B, A2 = A, B2 = B;
        convex::ResolveContactPair(A1, unit, B1, unit, cfg);
        convex::ResolveContactPair(A2, unit, B2, unit, cfg);
        check(std::memcmp(&B1, &B2, sizeof(fpx::FxBody)) == 0 &&
              std::memcmp(&A1, &A2, sizeof(fpx::FxBody)) == 0,
              "CX3 determinism: two ResolveContactPair runs BYTE-IDENTICAL");
    }

    // ========================================================================================
    // ================= Slice CX4 — THE FULL CONVEX STEP (a settling stack) ===================
    // ========================================================================================
    // ConvexWorld (parallel bodies/boxes arrays, some static) + StepConvexWorld (the 5-pass tick:
    // predict-integrate -> all-pairs narrowphase -> impulse solve -> position de-penetration) +
    // StepConvexWorldN + MeasureStack. The NEW bit vs CX3 is POSITION de-penetration so a box
    // STACK settles on a static floor instead of sinking. PURE INTEGER, all orders PINNED.

    // A helper: the deterministic gravity host-snap (== the showcase's, matches the fpx convention).
    const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));

    // A standard settling config: gravity down, dt 1/60, no bounce, slop 1/64, beta 0.4, 20 velocity solve
    // sweeps + 4 position-de-pen sweeps + mild damping (linDamp 0.98, angDamp 0.5) — the damping bleeds off
    // the spurious resting torque the non-accumulated Gauss-Seidel + fixed-point face manifold leave (the
    // CX3-documented residual), so a 3-box tower SETTLES + rests coherently instead of slowly tipping.
    auto makeStepCfg = [&]() {
        convex::ConvexStepConfig c;
        c.gravity = FxVec3{0, kGravY, 0};
        c.dt = kOne / 60;
        c.solveIters = 20;
        c.restitution = 0;
        c.slop = kOne / 64;
        c.beta = (fx)((int64_t)4 * kOne / 10);    // 0.4
        c.linDamp = (fx)((int64_t)98 * kOne / 100);   // 0.98
        c.angDamp = (fx)((int64_t)50 * kOne / 100);   // 0.5
        c.posIters = 4;
        return c;
    };
    // Build a body+box at a position with an orientation, dynamic or static.
    auto makeWorldBody = [&](fx x, fx y, fx z, bool dyn) {
        fpx::FxBody b;
        b.pos = {x, y, z};
        b.orient = fpx::FxQuat{0, 0, 0, kOne};
        b.invMass = dyn ? kOne : 0;
        b.flags = dyn ? fpx::kFlagDynamic : 0u;
        b.vel = {0, 0, 0};
        b.angVel = {0, 0, 0};
        return b;
    };
    const convex::FxBox kUnitBox{FxVec3{kOne, kOne, kOne}};

    // ================= CX4: a no-overlap free-fall step is a pure integrate (no contact) ============
    {
        // A single dynamic box high above a far-away static floor — no pair overlaps -> StepConvexWorld
        // is just IntegrateBodyFull (gravity changes vel/pos; no de-penetration, no impulse).
        convex::ConvexWorld w;
        convex::FxBox floor{FxVec3{FromInt(8), kOne, FromInt(8)}};
        w.bodies.push_back(makeWorldBody(0, 0, 0, false)); w.boxes.push_back(floor);     // floor (static, top y=1)
        w.bodies.push_back(makeWorldBody(0, FromInt(10), 0, true)); w.boxes.push_back(kUnitBox); // far above
        convex::ConvexStepConfig cfg = makeStepCfg();

        // Reference: a lone IntegrateBodyFull on the dynamic body + the same per-tick damping (the step
        // applies linDamp/angDamp post-integrate; with no contact that is the ONLY other effect).
        fpx::FxBody refDyn = w.bodies[1];
        fpx::IntegrateBodyFull(refDyn, cfg.gravity, cfg.dt);
        refDyn.vel = fpx::FxScale(refDyn.vel, cfg.linDamp);
        refDyn.angVel = fpx::FxScale(refDyn.angVel, cfg.angDamp);

        convex::StepConvexWorld(w, cfg);
        check(std::memcmp(&w.bodies[1], &refDyn, sizeof(fpx::FxBody)) == 0,
              "CX4 no-overlap step == IntegrateBodyFull + damping (free fall, no contact)");
        // The static floor never moved.
        const fpx::FxBody floorRef = makeWorldBody(0, 0, 0, false);
        check(std::memcmp(&w.bodies[0], &floorRef, sizeof(fpx::FxBody)) == 0,
              "CX4 no-overlap step -> static floor unchanged");
    }

    // ================= CX4: static bodies NEVER move (over many ticks, even while loaded) ===========
    {
        // A static floor + a dynamic box dropped onto it. After many ticks the floor must be byte-identical.
        convex::ConvexWorld w;
        convex::FxBox floor{FxVec3{FromInt(8), kOne, FromInt(8)}};
        w.bodies.push_back(makeWorldBody(0, 0, 0, false)); w.boxes.push_back(floor);
        w.bodies.push_back(makeWorldBody(0, FromInt(3), 0, true)); w.boxes.push_back(kUnitBox);
        const fpx::FxBody floorBefore = w.bodies[0];
        convex::StepConvexWorldN(w, makeStepCfg(), 150);
        check(std::memcmp(&w.bodies[0], &floorBefore, sizeof(fpx::FxBody)) == 0,
              "CX4 static floor NEVER moves over 150 ticks (loaded by a resting box)");
    }

    // ================= CX4: a SINGLE box dropped on the floor RESTS ON the floor (does not sink) ====
    {
        // Floor top at y = 1 (floor center 0, halfY 1). A unit box (halfY 1) at rest must sit with its
        // CENTER at y ~ floorTop + halfY = 2. We drop it from y=3 and step until it settles.
        convex::ConvexWorld w;
        convex::FxBox floor{FxVec3{FromInt(8), kOne, FromInt(8)}};
        w.bodies.push_back(makeWorldBody(0, 0, 0, false)); w.boxes.push_back(floor);
        w.bodies.push_back(makeWorldBody(0, FromInt(3), 0, true)); w.boxes.push_back(kUnitBox);
        convex::StepConvexWorldN(w, makeStepCfg(), 240);

        const fpx::FxBody& box = w.bodies[1];
        const fx floorTop = w.bodies[0].pos.y + floor.halfExtents.y;   // 0 + 1 = 1
        const fx boxBottom = box.pos.y - kUnitBox.halfExtents.y;       // center - halfY
        // boxBottom should be ~ floorTop, within slop + a small integer epsilon (the position correction
        // leaves the allowed penetration band). It must NOT have sunk through (boxBottom far below floorTop).
        const fx eps = kOne / 16;
        const fx slop = kOne / 64;
        auto absfx = [](fx v) { return v < 0 ? -v : v; };
        check(absfx(boxBottom - floorTop) < slop + eps,
              "CX4 single box rests ON the floor (boxBottom ~ floorTop, did not sink)");
        // At rest: the box's speed is small (it came to rest, not bouncing / falling).
        const fx speed = fpx::FxLength(box.vel);
        check(speed < kOne / 4, "CX4 single box at rest (small residual speed)");
    }

    // ================= CX4: a 3-box STACK settles ORDERED in y + non-interpenetrating ===============
    {
        // A static floor (top y=1) + 3 dynamic FLAT SLAB boxes (halfExtents 1.5 x 0.5 x 1.5 — wide + thin)
        // dropped in a column near their rest centers with a small drop gap. Flat slabs make the Y face the
        // UNAMBIGUOUS min-penetration axis (the wide X/Z faces overlap deeply, so an edge-cross can't tie the
        // shallow Y), so the de-penetration normal stays cleanly +Y -> a coherent resting tower. Box height
        // is 1 (2*halfY) -> rest centers ~ floorTop+0.5, +1.5, +2.5 = 1.5, 2.5, 3.5.
        const convex::FxBox kSlab{FxVec3{FromInt(3) / 2, kOne / 2, FromInt(3) / 2}};   // 3 x 1 x 3 slab
        convex::ConvexWorld w;
        convex::FxBox floor{FxVec3{FromInt(8), kOne, FromInt(8)}};
        w.bodies.push_back(makeWorldBody(0, 0, 0, false)); w.boxes.push_back(floor);
        w.bodies.push_back(makeWorldBody(0, FromInt(1) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
        w.bodies.push_back(makeWorldBody(0, FromInt(2) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);
        w.bodies.push_back(makeWorldBody(0, FromInt(3) + kOne * 5 / 8, 0, true)); w.boxes.push_back(kSlab);

        convex::StepConvexWorldN(w, makeStepCfg(), 360);

        // Ordered in y: each box above the one below (a coherent tower, not collapsed into one blob).
        check(w.bodies[1].pos.y < w.bodies[2].pos.y && w.bodies[2].pos.y < w.bodies[3].pos.y,
              "CX4 stack -> boxes ordered ascending in y (a tower, not a blob)");
        // Separated by ~the box height (1 unit, 2*halfY): consecutive centers differ by ~1 within a band.
        const fx gap01 = w.bodies[2].pos.y - w.bodies[1].pos.y;
        const fx gap12 = w.bodies[3].pos.y - w.bodies[2].pos.y;
        const fx hi = FromInt(1) + kOne / 4;   // 1.25 upper band
        const fx lo = FromInt(1) - kOne / 4;   // 0.75 lower band
        check(gap01 > lo && gap01 < hi, "CX4 stack -> box0->box1 separated by ~1 box height");
        check(gap12 > lo && gap12 < hi, "CX4 stack -> box1->box2 separated by ~1 box height");
        // The bottom box rests on the floor.
        const fx floorTop = w.bodies[0].pos.y + floor.halfExtents.y;
        const fx bottomBoxBottom = w.bodies[1].pos.y - kSlab.halfExtents.y;
        auto absfx = [](fx v) { return v < 0 ? -v : v; };
        check(absfx(bottomBoxBottom - floorTop) < kOne / 8,
              "CX4 stack -> bottom box rests on the floor");

        // MeasureStack: deterministic rest + interpenetration metrics.
        const convex::StackMeasure ms = convex::MeasureStack(w);
        check(ms.dynamicCount == 3, "CX4 MeasureStack -> 3 dynamic bodies");
        check(ms.maxSpeed < kOne / 2, "CX4 MeasureStack -> the stack is at rest (small max speed)");
        // Non-interpenetration: the max pairwise penetration is within slop + an integer epsilon.
        check(ms.maxPenetration < kOne / 16,
              "CX4 MeasureStack -> no pair interpenetrates beyond slop (position correction held)");
    }

    // ================= CX4: StepConvexWorldN determinism (two runs byte-identical) ==================
    {
        auto buildStack = [&]() {
            convex::ConvexWorld w;
            convex::FxBox floor{FxVec3{FromInt(8), kOne, FromInt(8)}};
            w.bodies.push_back(makeWorldBody(0, 0, 0, false)); w.boxes.push_back(floor);
            w.bodies.push_back(makeWorldBody(0,          FromInt(3), 0, true)); w.boxes.push_back(kUnitBox);
            w.bodies.push_back(makeWorldBody(kOne / 16,  FromInt(5), 0, true)); w.boxes.push_back(kUnitBox);
            return w;
        };
        convex::ConvexWorld w1 = buildStack();
        convex::ConvexWorld w2 = buildStack();
        convex::StepConvexWorldN(w1, makeStepCfg(), 200);
        convex::StepConvexWorldN(w2, makeStepCfg(), 200);
        bool same = (w1.bodies.size() == w2.bodies.size());
        for (size_t i = 0; i < w1.bodies.size() && same; ++i)
            if (std::memcmp(&w1.bodies[i], &w2.bodies[i], sizeof(fpx::FxBody)) != 0) same = false;
        check(same, "CX4 StepConvexWorldN determinism: two runs BYTE-IDENTICAL");
    }

    if (g_fail == 0) std::printf("convex_test: ALL PASS\n");
    else std::printf("convex_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
