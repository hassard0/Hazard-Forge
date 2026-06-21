// Slice MF1 — Hull Narrowphase Hardening: HULL FACE TOPOLOGY (the new primitive, the BEACHHEAD of FLAGSHIP
// #25: DETERMINISTIC HULL NARROWPHASE HARDENING, hf::sim::manifold). The pure-CPU integer core
// (engine/sim/manifold.h) — the per-hull POLYGON FACE table + the reference/incident face selectors + the
// render-only float face soup. manifold.h #includes sim/ccd.h read-only (transitively gjk/broad/convex/fpx);
// gjk.h and ALL sibling sim headers are BYTE-FROZEN.
//
// What this test PINS (the contracts MF2/MF3/MF4 build on; the 4 proofs the showcase also asserts):
//   * (1) FACE COUNTS: BuildCanonicalFaces returns exactly {tetra:4, box:6, octa:8, wedge:5}.
//   * (2) OUTWARD WINDING: every face of every canonical hull is outward-wound (FxDot(faceNormalWorld,
//     faceCentroidWorld - hullCentroidWorld) > 0); the min over all faces is > 0.
//   * VALIDITY: every face's vertex indices are < hull.count and within the per-face vertCount.
//   * (3) SELECTION purity + correctness: SupportFace/IncidentFace over a FIXED (hull,dir) battery are byte-
//     equal across two runs; for a box SupportFace(+X) is the +X face and IncidentFace of its normal is -X.
//   * (4) RENDER provenance: FacesToRenderInstances is a pure function (two calls byte-equal).
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/manifold.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace manifold = hf::sim::manifold;
namespace gjk = hf::sim::gjk;
namespace convex = hf::sim::convex;
namespace fpx = hf::sim::fpx;
using manifold::fx;
using manifold::kOne;
using manifold::FxVec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static fx FromInt(int v) { return gjk::FromInt(v); }

// An identity-orientation body at the origin (the canonical hulls placed at world origin).
static fpx::FxBody MakeBody(fx px, fx py, fx pz) {
    fpx::FxBody b;
    b.pos = {px, py, pz};
    b.orient = {0, 0, 0, kOne};   // identity quaternion {0,0,0,1}
    return b;
}

int main() {
    HF_TEST_MAIN_INIT();

    const gjk::FxHull tetra = gjk::MakeTetra(kOne);
    const gjk::FxHull box   = gjk::MakeBox(kOne, kOne, kOne);
    const gjk::FxHull octa  = gjk::MakeOcta(kOne);
    const gjk::FxHull wedge = gjk::MakeWedge(kOne, kOne, kOne);

    const manifold::FxHullFaces fTetra = manifold::BuildCanonicalFaces(tetra);
    const manifold::FxHullFaces fBox   = manifold::BuildCanonicalFaces(box);
    const manifold::FxHullFaces fOcta  = manifold::BuildCanonicalFaces(octa);
    const manifold::FxHullFaces fWedge = manifold::BuildCanonicalFaces(wedge);

    // === (1) FACE COUNTS — exactly {tetra:4, box:6, octa:8, wedge:5}. ===
    check(fTetra.faceCount == 4u, "MF1 counts: tetra has 4 faces");
    check(fBox.faceCount   == 6u, "MF1 counts: box has 6 faces");
    check(fOcta.faceCount  == 8u, "MF1 counts: octa has 8 faces");
    check(fWedge.faceCount == 5u, "MF1 counts: wedge has 5 faces");
    // A non-canonical hull (count not in {4,6,8}) -> 0 faces (documented YAGNI).
    {
        gjk::FxHull empty; empty.count = 0;
        check(manifold::BuildCanonicalFaces(empty).faceCount == 0u, "MF1 counts: empty hull -> 0 faces");
    }

    // Helper: assert every face of a hull is outward-wound + has valid indices, returning the min outward dot.
    struct Batch { const gjk::FxHull* hull; const manifold::FxHullFaces* faces; const char* name;
                   uint32_t expectVerts; };  // expectVerts: 3 tri, 4 quad, 0 mixed
    const fpx::FxBody body = MakeBody(0, 0, 0);
    auto auditHull = [&](const gjk::FxHull& hull, const manifold::FxHullFaces& faces, const char* name) -> fx {
        fx minDot = 0; bool first = true;
        const FxVec3 hc = manifold::HullCentroidWorld(hull, body);
        for (uint32_t f = 0; f < faces.faceCount; ++f) {
            // VALIDITY: vertCount in {3,4}; each index < hull.count; indices distinct within the face.
            const uint32_t vc = faces.vertCount[f];
            check(vc == 3u || vc == 4u, name);
            for (uint32_t k = 0; k < vc; ++k) {
                check(faces.vertIdx[f][k] < hull.count, name);
                for (uint32_t k2 = k + 1; k2 < vc; ++k2)
                    check(faces.vertIdx[f][k] != faces.vertIdx[f][k2], name);
            }
            // (2) OUTWARD: FxDot(normal, faceCentroid - hullCentroid) > 0.
            const FxVec3 nrm = manifold::FaceNormalWorld(hull, faces, body, f);
            const FxVec3 fc  = manifold::FaceCentroidWorld(hull, faces, body, f);
            const fx d = convex::FxDot(nrm, fpx::FxSub(fc, hc));
            check(d > 0, name);
            if (first || d < minDot) { minDot = d; first = false; }
        }
        return minDot;
    };

    // === (2) OUTWARD WINDING — every face of every canonical hull, min outward dot > 0. ===
    fx minDot = auditHull(tetra, fTetra, "MF1 winding: tetra faces outward + valid");
    fx d2;
    d2 = auditHull(box,   fBox,   "MF1 winding: box faces outward + valid");   if (d2 < minDot) minDot = d2;
    d2 = auditHull(octa,  fOcta,  "MF1 winding: octa faces outward + valid");  if (d2 < minDot) minDot = d2;
    d2 = auditHull(wedge, fWedge, "MF1 winding: wedge faces outward + valid"); if (d2 < minDot) minDot = d2;
    check(minDot > 0, "MF1 winding: min outward dot over ALL canonical-hull faces > 0");

    // === (3) SELECTION purity + correctness ===
    // A FIXED (hull, dir) battery: the four canonical hulls x the six axis directions. SupportFace +
    // IncidentFace of each returned face's normal, summed -> a deterministic scalar two-run-byte-equal.
    const FxVec3 dirs[6] = {{kOne, 0, 0}, {-kOne, 0, 0}, {0, kOne, 0},
                            {0, -kOne, 0}, {0, 0, kOne}, {0, 0, -kOne}};
    const gjk::FxHull* hulls[4] = {&tetra, &box, &octa, &wedge};
    const manifold::FxHullFaces* facesArr[4] = {&fTetra, &fBox, &fOcta, &fWedge};
    auto supportSum = [&]() -> uint64_t {
        uint64_t sum = 0;
        for (int hi = 0; hi < 4; ++hi) {
            for (int di = 0; di < 6; ++di) {
                const uint32_t sf = manifold::SupportFace(*hulls[hi], *facesArr[hi], body, dirs[di]);
                const FxVec3 nrm = manifold::FaceNormalWorld(*hulls[hi], *facesArr[hi], body, sf);
                const uint32_t inc = manifold::IncidentFace(*hulls[hi], *facesArr[hi], body, nrm);
                sum += (uint64_t)(sf + 1u) * 131u + (uint64_t)(inc + 1u) * 17u;
            }
        }
        return sum;
    };
    const uint64_t s1 = supportSum();
    const uint64_t s2 = supportSum();
    check(s1 == s2, "MF1 support: SupportFace/IncidentFace battery is two-run BYTE-EQUAL (pure)");

    // CORRECTNESS: for a box, SupportFace(+X) returns the +X face, IncidentFace of its normal returns -X.
    {
        const uint32_t plusX = manifold::SupportFace(box, fBox, body, FxVec3{kOne, 0, 0});
        const FxVec3 nX = manifold::FaceNormalWorld(box, fBox, body, plusX);
        // The +X face's outward normal points +X (its x component is the strictly-largest).
        check(nX.x > 0 && nX.x > (nX.y < 0 ? -nX.y : nX.y) && nX.x > (nX.z < 0 ? -nX.z : nX.z),
              "MF1 support: box SupportFace(+X) normal points +X");
        const uint32_t minusX = manifold::IncidentFace(box, fBox, body, nX);
        const FxVec3 nNeg = manifold::FaceNormalWorld(box, fBox, body, minusX);
        check(minusX != plusX, "MF1 support: box incident face of +X is a DIFFERENT face");
        check(nNeg.x < 0 && (-nNeg.x) > (nNeg.y < 0 ? -nNeg.y : nNeg.y) &&
              (-nNeg.x) > (nNeg.z < 0 ? -nNeg.z : nNeg.z),
              "MF1 support: box IncidentFace(+X normal) is the -X face (opposing)");
        // The opposing relation: the incident face's normal is anti-parallel to the reference (dot < 0).
        check(convex::FxDot(nX, nNeg) < 0, "MF1 support: +X and incident -X normals oppose (dot < 0)");
    }

    // === (4) RENDER provenance — FacesToRenderInstances is a PURE function (two calls byte-equal). ===
    {
        gjk::HullWorld world;
        world.hulls = {tetra, box, octa, wedge};
        world.bodies = {MakeBody(FromInt(-6), 0, 0), MakeBody(FromInt(-2), 0, 0),
                        MakeBody(FromInt(2), 0, 0), MakeBody(FromInt(6), 0, 0)};
        const gjk::HullRenderMesh soupA = manifold::FacesToRenderInstances(world);
        const gjk::HullRenderMesh soupB = manifold::FacesToRenderInstances(world);
        check(gjk::HullRenderMeshEqual(soupA, soupB),
              "MF1 render: FacesToRenderInstances two calls BYTE-EQUAL (pure function)");
        check(soupA.triangles > 0u, "MF1 render: the four-hull scene produced triangles");
        check(soupA.verts.size() == (size_t)soupA.triangles * 3u, "MF1 render: verts.size() == triangles*3");
        // tetra 4 tris + box 6 quads(12 tris) + octa 8 tris + wedge (2 tris + 3 quads=6 tris -> 8 tris)
        // = 4 + 12 + 8 + 8 = 32 tris.
        check(soupA.triangles == 32u, "MF1 render: triangle count == 32 (tetra4 + box12 + octa8 + wedge8)");
        // The render REFLECTS the sim: a moved hull renders a different soup.
        gjk::HullWorld moved = world;
        moved.bodies[1].pos.x += FromInt(1);
        check(!gjk::HullRenderMeshEqual(soupA, manifold::FacesToRenderInstances(moved)),
              "MF1 render: a moved hull renders a DIFFERENT soup (render reflects pose)");
    }

    if (g_fail == 0) std::printf("manifold_test: ALL PASS\n");
    else std::printf("manifold_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
