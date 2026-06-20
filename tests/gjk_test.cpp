// Slice GJ1 — General Convex-Hull Contacts: THE HULL + SUPPORT FUNCTION (the BEACHHEAD of FLAGSHIP #22:
// hf::sim::gjk). The integer core (engine/sim/gjk.h) that the GPU shaders/gjk_support.comp.hlsl copies
// VERBATIM + proves bit-identical. Pure CPU (header-only, hf_core), ASan-eligible. gjk.h #includes
// sim/convex.h read-only (which transitively gives fpx.h).
//
// What this test PINS (the contracts the GPU gjk_support.comp + the GPU==CPU proof build on):
//   * SupportLocal: returns the maximal vertex (vs brute force) over the canonical hulls + a direction
//     sweep; ties resolve to the LOWEST index (the strict-greater scan).
//   * Support: matches a HAND-COMPUTED world support for a rotated body (the conjugate-rotate-in path).
//   * SupportMinkowski: == Support_A(dir) - Support_B(-dir) for a sample pair.
//   * MeasureSupport: a PURE function (two calls byte-equal).
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "sim/gjk.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace gjk = hf::sim::gjk;
namespace convex = hf::sim::convex;
namespace fpx = hf::sim::fpx;
using gjk::fx;
using gjk::kOne;
using gjk::kFrac;
using gjk::FxVec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static fx FromInt(int v) { return (fx)((int64_t)v << kFrac); }

// A body at an integer position with a given orientation (default identity).
static fpx::FxBody MakeBody(int x, int y, int z, fpx::FxQuat q = fpx::FxQuat{0, 0, 0, kOne}) {
    fpx::FxBody b;
    b.pos = {FromInt(x), FromInt(y), FromInt(z)};
    b.orient = q;
    return b;
}

// Brute-force the maximal-extent vertex INDEX of a LOCAL hull along a LOCAL dir (the reference SupportLocal).
static uint32_t BruteSupportLocalIndex(const gjk::FxHull& hull, const FxVec3& dir) {
    uint32_t best = 0;
    fx bestDot = convex::FxDot(hull.verts[0], dir);
    for (uint32_t i = 1; i < hull.count; ++i) {
        const fx d = convex::FxDot(hull.verts[i], dir);
        if (d > bestDot) { bestDot = d; best = i; }
    }
    return best;
}

int main() {
    HF_TEST_MAIN_INIT();

    // The canonical hull set (unit-scale).
    const gjk::FxHull tetra = gjk::MakeTetra(kOne);
    const gjk::FxHull box   = gjk::MakeBox(kOne, kOne, kOne);
    const gjk::FxHull octa  = gjk::MakeOcta(kOne);
    const gjk::FxHull wedge = gjk::MakeWedge(kOne, kOne, kOne);
    const gjk::FxHull hulls[4] = {tetra, box, octa, wedge};
    const char* hullNames[4] = {"tetra", "box", "octa", "wedge"};

    // ================= hull builders: vertex counts + cap =================
    {
        check(tetra.count == 4, "MakeTetra has 4 verts");
        check(box.count == 8, "MakeBox has 8 verts");
        check(octa.count == 6, "MakeOcta has 6 verts");
        check(wedge.count == 6, "MakeWedge has 6 verts");
        check(box.count <= gjk::kMaxHullVerts, "box verts <= kMaxHullVerts");
        check(gjk::kMaxHullVerts == 20, "kMaxHullVerts == 20 (the documented ceiling)");
    }

    // A fixed direction sweep: the 6 axes + a few diagonals (Q16.16, unnormalized — argmax is scale-free).
    std::vector<FxVec3> dirs;
    dirs.push_back({ FromInt(1), 0, 0});
    dirs.push_back({-FromInt(1), 0, 0});
    dirs.push_back({0,  FromInt(1), 0});
    dirs.push_back({0, -FromInt(1), 0});
    dirs.push_back({0, 0,  FromInt(1)});
    dirs.push_back({0, 0, -FromInt(1)});
    dirs.push_back({ FromInt(1),  FromInt(1),  FromInt(1)});
    dirs.push_back({-FromInt(1),  FromInt(1), -FromInt(1)});
    dirs.push_back({ FromInt(2), -FromInt(1),  FromInt(3)});
    dirs.push_back({-FromInt(3),  FromInt(2), -FromInt(1)});

    // ================= SupportLocal: maximal vs brute force =================
    {
        bool maximalAll = true;
        for (int h = 0; h < 4; ++h) {
            for (const FxVec3& d : dirs) {
                const FxVec3 s = gjk::SupportLocal(hulls[h], d);
                const fx got = convex::FxDot(s, d);
                // No vertex may have a strictly greater extent than the returned support.
                for (uint32_t i = 0; i < hulls[h].count; ++i) {
                    if (convex::FxDot(hulls[h].verts[i], d) > got) maximalAll = false;
                }
            }
        }
        check(maximalAll, "SupportLocal is maximal over all canonical hulls x dirs (brute force)");
    }

    // ================= SupportLocal: hand-checked + lowest-index tie-break =================
    {
        // Box: +x dir -> a +x-side corner (x == +kOne). The FIXED sign sweep (x outer) makes verts[4..7]
        // the +x corners; the strict-greater scan keeps the LOWEST tied index -> verts[4] == (+,-,-).
        const FxVec3 sx = gjk::SupportLocal(box, FxVec3{FromInt(1), 0, 0});
        check(sx.x == kOne, "box SupportLocal(+x) has x == +1");
        const uint32_t tieIdx = BruteSupportLocalIndex(box, FxVec3{FromInt(1), 0, 0});
        check(tieIdx == 4, "box +x tie resolves to lowest index 4 (the sign sweep)");
        // The returned vertex equals verts[4] exactly.
        check(sx.x == box.verts[4].x && sx.y == box.verts[4].y && sx.z == box.verts[4].z,
              "box SupportLocal(+x) == verts[4] (lowest-index tie)");

        // Octa: +y dir -> the +y pole (verts[2]).
        const FxVec3 sy = gjk::SupportLocal(octa, FxVec3{0, FromInt(1), 0});
        check(sy.x == 0 && sy.y == kOne && sy.z == 0, "octa SupportLocal(+y) == (0,+1,0) pole");

        // Tetra: +x+y+z diagonal -> verts[0] = (+,+,+).
        const FxVec3 st = gjk::SupportLocal(tetra, FxVec3{FromInt(1), FromInt(1), FromInt(1)});
        check(st.x == kOne && st.y == kOne && st.z == kOne, "tetra SupportLocal(+++) == verts[0]");
    }

    // ================= Support: hand-computed world support for a rotated body =================
    {
        // A box rotated 90 deg about Z: local +x -> world +y. Quaternion about Z by 90 deg is
        // {0,0,sin45,cos45}; in Q16.16 sin45=cos45 ~ 0.70710678 -> use the host-snapped normalized quat.
        // sin/cos 45 deg in Q16.16: 0.70710678 * 65536 ~ 46341.
        const fx s = (fx)46341, c = (fx)46341;
        const fpx::FxQuat qz90 = fpx::FxQuatNormalize(fpx::FxQuat{0, 0, s, c});
        const fpx::FxBody body = MakeBody(10, 0, 0, qz90);
        // World support along +x: the support is the box corner whose WORLD position is farthest in +x.
        // After a +90deg-Z rotation, local +x maps to world +y and local -y maps to world +x, so the
        // farthest-+x corner is local (x=+,y=-1,...) family. Rather than hand-derive the exact tie, assert
        // the world support equals FxRotate(orient, SupportLocal(localDir)) + pos for the SAME path, and
        // that it lies on the +x side of the body center (world x > pos.x).
        const FxVec3 dir{FromInt(1), 0, 0};
        const FxVec3 sup = gjk::Support(box, body, dir);

        // Reconstruct via the documented path (conjugate-rotate the dir in).
        const fpx::FxQuat conj = fpx::FxQuat{-qz90.x, -qz90.y, -qz90.z, qz90.w};
        const FxVec3 localDir = fpx::FxRotate(conj, dir);
        const FxVec3 localV = gjk::SupportLocal(box, localDir);
        const FxVec3 expect = convex::FxAdd(fpx::FxRotate(qz90, localV), body.pos);
        check(sup.x == expect.x && sup.y == expect.y && sup.z == expect.z,
              "Support matches the conjugate-rotate-in world path for a rotated body");
        // The support is on the +x side of the body center.
        check(sup.x > body.pos.x, "rotated-body Support(+x) world x > center x");

        // Sanity for an IDENTITY body: Support(box, dir) extent == the box half-extent reach + pos.
        const fpx::FxBody idBody = MakeBody(5, 0, 0);
        const FxVec3 sup2 = gjk::Support(box, idBody, dir);
        // Farthest +x corner of a unit box at x=5 -> world x = 5 + 1 = 6.
        check(sup2.x == FromInt(6), "identity-body Support(+x) world x == pos.x + halfExtent");
    }

    // ================= SupportMinkowski: == Support_A(dir) - Support_B(-dir) =================
    {
        const fpx::FxBody bodyA = MakeBody(0, 0, 0);
        const fpx::FxBody bodyB = MakeBody(3, 1, 0,
            fpx::FxQuatNormalize(fpx::FxQuat{(fx)25080, 0, 0, (fx)60547}));  // ~45deg about X
        for (const FxVec3& d : dirs) {
            const FxVec3 mink = gjk::SupportMinkowski(box, bodyA, octa, bodyB, d);
            const FxVec3 sa = gjk::Support(box, bodyA, d);
            const FxVec3 sb = gjk::Support(octa, bodyB, FxVec3{-d.x, -d.y, -d.z});
            const FxVec3 expect = gjk::FxSub(sa, sb);
            check(mink.x == expect.x && mink.y == expect.y && mink.z == expect.z,
                  "SupportMinkowski == Support_A(dir) - Support_B(-dir)");
        }
    }

    // ================= MeasureSupport: a PURE function (two calls byte-equal) =================
    {
        std::vector<fpx::FxBody> bodies;
        for (int i = 0; i < 4; ++i) bodies.push_back(MakeBody(i * 3 - 4, 0, i));
        // Rotate a couple to exercise FxRotate.
        bodies[1].orient = fpx::FxQuatNormalize(fpx::FxQuat{(fx)25080, 0, 0, (fx)60547});
        bodies[3].orient = fpx::FxQuatNormalize(fpx::FxQuat{0, 0, (fx)25080, (fx)60547});

        const gjk::HullMeasure m1 = gjk::MeasureSupport(hulls, bodies.data(), 4, dirs.data(),
                                                        (uint32_t)dirs.size());
        const gjk::HullMeasure m2 = gjk::MeasureSupport(hulls, bodies.data(), 4, dirs.data(),
                                                        (uint32_t)dirs.size());
        check(std::memcmp(&m1, &m2, sizeof(gjk::HullMeasure)) == 0,
              "MeasureSupport is a pure function (two calls byte-equal)");
        check(m1.queries == 4u * (uint32_t)dirs.size(), "MeasureSupport query count == hulls x dirs");
    }

    // ============================================================================================
    // Slice GJ2 — the GJK algorithm (overlap + closest distance). APPENDED (GJ1 cases above unchanged).
    // ============================================================================================

    // An independent brute-force min-vertex-distance / overlap reference. For SEPARATED hulls the brute-force
    // closest-vertex distance is an UPPER bound on the true (feature-feature) closest distance; we use it for
    // a coarse separation classification + an order-of-magnitude band check. Overlap is independently checked
    // by a SAT-style separating axis over the face normals + edge crosses (boxes) — but since the canonical
    // hulls are general convex, we use the GJK overlap + a containment cross-check: a separated pair has the
    // origin strictly outside the Minkowski difference (some direction has SupportMinkowski·dir > 0 AND
    // SupportMinkowski(-dir)·(-dir) > 0 ... ) — simplest robust independent reference: for the analytic box
    // cases we know the truth; for the general set we assert overlap agrees with a fine support-sampling test.
    auto bodyAt = [&](int x, int y, int z, fpx::FxQuat q = fpx::FxQuat{0,0,0,kOne}) {
        return MakeBody(x, y, z, q);
    };

    // Independent overlap reference by support-sampling: the hulls OVERLAP iff for EVERY sampled direction in
    // a dense fixed set, the Minkowski-difference support has a non-negative extent (the origin is not
    // separable by any sampled axis). With a dense axis set this is a reliable classifier for the well-
    // separated / clearly-overlapping scene (the touching boundary is the documented gray zone).
    auto refOverlap = [&](const gjk::FxHull& hA, const fpx::FxBody& bA,
                          const gjk::FxHull& hB, const fpx::FxBody& bB) -> bool {
        // Dense direction set: all (i,j,k) in [-2,2]^3 except origin -> 124 axes.
        for (int ix = -2; ix <= 2; ++ix)
            for (int iy = -2; iy <= 2; ++iy)
                for (int iz = -2; iz <= 2; ++iz) {
                    if (ix == 0 && iy == 0 && iz == 0) continue;
                    const FxVec3 d{FromInt(ix), FromInt(iy), FromInt(iz)};
                    const FxVec3 s = gjk::SupportMinkowski(hA, bA, hB, bB, d);
                    if (convex::FxDot(s, d) < 0) return false;   // this axis separates -> not overlapping
                }
        return true;
    };

    // Brute-force min vertex-to-vertex distance of the two WORLD hulls (an UPPER bound on the true distance).
    auto bruteVertDist = [&](const gjk::FxHull& hA, const fpx::FxBody& bA,
                             const gjk::FxHull& hB, const fpx::FxBody& bB) -> fx {
        fx best = 0; bool first = true;
        for (uint32_t i = 0; i < hA.count; ++i) {
            const FxVec3 wa = convex::FxAdd(fpx::FxRotate(bA.orient, hA.verts[i]), bA.pos);
            for (uint32_t j = 0; j < hB.count; ++j) {
                const FxVec3 wb = convex::FxAdd(fpx::FxRotate(bB.orient, hB.verts[j]), bB.pos);
                const fx d = fpx::FxLength(fpx::FxSub(wa, wb));
                if (first || d < best) { best = d; first = false; }
            }
        }
        return best;
    };

    // The fixed GJK hull-pair scene: SEPARATED, nearly-TOUCHING, SHALLOW-OVERLAP, a couple rotated.
    std::vector<gjk::GjkPair> gpairs;
    auto addG = [&](gjk::FxHull hA, fpx::FxBody bA, gjk::FxHull hB, fpx::FxBody bB) {
        gpairs.push_back({hA, bA, hB, bB});
    };
    const fpx::FxQuat qX45 = fpx::FxQuatNormalize(fpx::FxQuat{(fx)25080, 0, 0, (fx)60547});
    const fpx::FxQuat qZ45 = fpx::FxQuatNormalize(fpx::FxQuat{0, 0, (fx)25080, (fx)60547});
    // 0: two unit boxes offset along X by 3 -> gap exactly 1.0 (the analytic case).
    addG(gjk::MakeBox(kOne,kOne,kOne), bodyAt(0,0,0), gjk::MakeBox(kOne,kOne,kOne), bodyAt(3,0,0));
    // 1: clearly separated tetra vs octa, far on X.
    addG(gjk::MakeTetra(kOne), bodyAt(-6,0,0), gjk::MakeOcta(kOne), bodyAt(2,0,0));
    // 2: shallow overlap — two unit boxes offset by 1 (penetrating by 1).
    addG(gjk::MakeBox(kOne,kOne,kOne), bodyAt(0,4,0), gjk::MakeBox(kOne,kOne,kOne), bodyAt(1,4,0));
    // 3: deep overlap — coincident-ish box vs octa.
    addG(gjk::MakeBox(kOne,kOne,kOne), bodyAt(0,-4,0), gjk::MakeOcta(kOne), bodyAt(0,-4,0));
    // 4: rotated box (45 X) separated from an octa on Z.
    addG(gjk::MakeBox(kOne,kOne,kOne), bodyAt(5,0,-5,qX45), gjk::MakeOcta(kOne), bodyAt(5,0,0));
    // 5: rotated wedge (45 Z) deeply overlapping a coincident box (clear overlap, off the touching boundary).
    addG(gjk::MakeWedge(kOne,kOne,kOne), bodyAt(-5,0,5,qZ45), gjk::MakeBox(kOne,kOne,kOne), bodyAt(-5,0,5));

    // ================= GJK overlap boolean matches the independent reference =================
    {
        bool allAgree = true;
        for (const gjk::GjkPair& p : gpairs) {
            const gjk::GjkResult r = gjk::Gjk(p.hullA, p.bodyA, p.hullB, p.bodyB);
            const bool ref = refOverlap(p.hullA, p.bodyA, p.hullB, p.bodyB);
            if ((r.overlap != 0u) != ref) allAgree = false;
        }
        check(allAgree, "Gjk overlap boolean matches the independent support-sampling reference");
    }

    // ================= analytic case: two unit boxes 3 apart on X -> gap 1.0 within band =================
    {
        const gjk::GjkResult r = gjk::Gjk(gjk::MakeBox(kOne,kOne,kOne), bodyAt(0,0,0),
                                          gjk::MakeBox(kOne,kOne,kOne), bodyAt(3,0,0));
        check(r.overlap == 0u, "boxes 3 apart on X are SEPARATED");
        const fx dist = fpx::FxLength(r.separation);
        // Expected gap = 3 - 1 - 1 = 1.0. Band: within 1/8 unit of kOne (fixed-point + sampling slack).
        const fx band = kOne / 8;
        check(dist > kOne - band && dist < kOne + band, "boxes 3-apart closest distance ~= 1.0 within band");
        // The witness points lie on the facing faces: closestA.x ~= +1 (A's +X face), closestB.x ~= +2.
        check(r.closestA.x > kOne - band && r.closestA.x < kOne + band, "witness A on the +X face of box A");
        check(r.closestB.x > FromInt(2) - band && r.closestB.x < FromInt(2) + band,
              "witness B on the -X face of box B (x ~= 2)");
    }

    // ================= separated closest distance within band of the brute-force upper bound =================
    {
        bool allInBand = true;
        for (const gjk::GjkPair& p : gpairs) {
            const gjk::GjkResult r = gjk::Gjk(p.hullA, p.bodyA, p.hullB, p.bodyB);
            if (r.overlap != 0u) continue;
            const fx gjkDist = fpx::FxLength(r.separation);
            const fx upper = bruteVertDist(p.hullA, p.bodyA, p.hullB, p.bodyB);
            // The true (feature) distance is <= the brute vertex distance; GJK must be <= upper + a small
            // fixed-point slack, and non-negative.
            if (gjkDist < 0) allInBand = false;
            if (gjkDist > upper + kOne / 16) allInBand = false;
        }
        check(allInBand, "Gjk separated closest distance is within band (<= brute vertex distance)");
    }

    // ================= Gjk is deterministic (two calls byte-equal) =================
    {
        bool allDet = true;
        for (const gjk::GjkPair& p : gpairs) {
            const gjk::GjkResult a = gjk::Gjk(p.hullA, p.bodyA, p.hullB, p.bodyB);
            const gjk::GjkResult b = gjk::Gjk(p.hullA, p.bodyA, p.hullB, p.bodyB);
            if (std::memcmp(&a, &b, sizeof(gjk::GjkResult)) != 0) allDet = false;
        }
        check(allDet, "Gjk is deterministic (two calls byte-equal for every pair)");
    }

    // ================= terminal simplex valid (count 1-4, points are genuine SupportMinkowski results) ======
    {
        bool allValid = true;
        for (const gjk::GjkPair& p : gpairs) {
            const gjk::GjkResult r = gjk::Gjk(p.hullA, p.bodyA, p.hullB, p.bodyB);
            if (r.simplex.count < 1 || r.simplex.count > 4) allValid = false;
            // Each simplex CSO point must equal csoA - csoB (a genuine Minkowski-difference point).
            for (uint32_t i = 0; i < r.simplex.count; ++i) {
                const FxVec3 expect = fpx::FxSub(r.simplex.csoA[i], r.simplex.csoB[i]);
                if (!(r.simplex.pts[i].x == expect.x && r.simplex.pts[i].y == expect.y &&
                      r.simplex.pts[i].z == expect.z)) allValid = false;
            }
            if (r.iterations > gjk::kGjkMaxIter) allValid = false;
        }
        check(allValid, "Gjk terminal simplex valid (count 1-4, CSO points genuine, iters bounded)");
    }

    // ================= MeasureGjk is a PURE function (two calls byte-equal) =================
    {
        const gjk::GjkMeasure m1 = gjk::MeasureGjk(gpairs.data(), (uint32_t)gpairs.size());
        const gjk::GjkMeasure m2 = gjk::MeasureGjk(gpairs.data(), (uint32_t)gpairs.size());
        check(std::memcmp(&m1, &m2, sizeof(gjk::GjkMeasure)) == 0,
              "MeasureGjk is a pure function (two calls byte-equal)");
        check(m1.pairs == (uint32_t)gpairs.size(), "MeasureGjk pair count == scene size");
        check(m1.overlapping + m1.separated == m1.pairs, "MeasureGjk overlap+separated == pairs");
    }

    (void)hullNames;
    if (g_fail == 0) std::printf("gjk_test: ALL PASS\n");
    else std::printf("gjk_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
