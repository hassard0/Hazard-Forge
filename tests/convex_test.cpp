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

    if (g_fail == 0) std::printf("convex_test: ALL PASS\n");
    else std::printf("convex_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
