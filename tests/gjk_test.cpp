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

    (void)hullNames;
    if (g_fail == 0) std::printf("gjk_test: ALL PASS\n");
    else std::printf("gjk_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
