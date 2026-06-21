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

    // =====================================================================================================
    // Slice MF2 — SUTHERLAND-HODGMAN FACE CLIP -> THE MULTI-POINT MANIFOLD. The MF1 lines above are frozen;
    // these tests PIN the MF2 contracts the showcase also asserts:
    //   * (1) the teeter fix: HullManifoldFromEpa counts {boxOnBox:4, tetraOnFace:3, edgeOnFace:2}.
    //   * (2) points valid + below the ref face: every kept point depth >= 0; count <= 4.
    //   * (3) determinism / purity: the manifold POD over the battery is byte-equal across two runs.
    //   * (4) consistency: the manifold normal agrees with the EPA seed normal (FxDot > 0).
    //   * (5) a degenerate/edge case falls back to count==1 (never 0 for an overlapping pair).
    // =====================================================================================================
    {
        auto makeBodyQ = [&](fx px, fx py, fx pz) -> fpx::FxBody { return MakeBody(px, py, pz); };

        // ---- The three canonical contacts (the SAME scene the showcase builds). ----
        // (a) box flat on box -> a unit box resting face-down on a static unit box, slightly overlapping in Y
        //     -> the quad-on-quad clip = 4 corner contacts.
        const gjk::FxHull boxA = gjk::MakeBox(kOne, kOne, kOne);
        const gjk::FxHull boxB = gjk::MakeBox(kOne, kOne, kOne);
        const fx overlap = kOne / 8;                 // a small penetration (0.125)
        const fpx::FxBody boxStatic = makeBodyQ(0, 0, 0);
        const fpx::FxBody boxTop = makeBodyQ(0, FromInt(2) - overlap, 0);   // 2*halfExtent=2, minus overlap

        // (b) tetra face-down on a box -> a tetra with a FLAT triangular base at y=-1 (the bottom face) + an
        //     apex at y=+1. BuildCanonicalFaces treats any 4-vert hull as a tetra (the face groupings are
        //     fixed; FaceNormalWorld canonicalizes outward), so the flat base face = the reference clip ->
        //     the box's top face clipped to the triangle = 3 contacts. Local base verts form a triangle.
        gjk::FxHull tetraH;
        tetraH.verts[0] = FxVec3{0, kOne, 0};                       // apex (up)
        tetraH.verts[1] = FxVec3{FromInt(1), -kOne, FromInt(1)};    // base corner
        tetraH.verts[2] = FxVec3{FromInt(1), -kOne, -FromInt(1)};   // base corner
        tetraH.verts[3] = FxVec3{-FromInt(1), -kOne, 0};            // base corner
        tetraH.count = 4;
        const fpx::FxBody tetraBody = makeBodyQ(0, FromInt(2) - overlap, 0);   // base y=-1 -> world y=+1-overlap

        // (c) edge-on-face -> a box rotated 45deg about Z so an EDGE points down onto the box top. The
        //     down-facing incident face is a side quad; clipped against the box top it survives as a 2-point
        //     edge contact (the box-corner-edge straddling the top face). Quaternion for 45deg about Z:
        //     (0,0,sin(22.5),cos(22.5)). sin(22.5deg)=0.38268, cos=0.92388 in Q16.16.
        const gjk::FxHull octaH = gjk::MakeBox(kOne, kOne, kOne);
        // y_center = (+1 - overlap) + sqrt(2) so the rotated box's lowest edge dips `overlap` below the top.
        fpx::FxBody octaBody = makeBodyQ(0, (fx)((1.0 - 0.125 + 1.41421356) * 65536.0), 0);
        octaBody.orient = {0, 0, (fx)(0.38268343f * 65536.0f), (fx)(0.92387953f * 65536.0f)};
        // The box rotated 45deg about Z has its lowest point a corner-edge at y = -sqrt(2) ~ -1.414; drop so
        // it overlaps the static box's top (+1): place center so the lowest edge dips ~overlap below +1.

        // Run gjk+epa+HullManifoldFromEpa for a pair, returning the manifold.
        auto manifoldFor = [&](const gjk::FxHull& hA, const fpx::FxBody& bA,
                               const gjk::FxHull& hB, const fpx::FxBody& bB) -> convex::ContactManifold {
            const gjk::GjkResult g = gjk::Gjk(hA, bA, hB, bB);
            if (!g.overlap) { convex::ContactManifold z; return z; }
            const gjk::EpaResult e = gjk::Epa(hA, bA, hB, bB, g.simplex);
            return manifold::HullManifoldFromEpa(hA, bA, hB, bB, e);
        };

        const convex::ContactManifold mBoxOnBox = manifoldFor(boxA, boxStatic, boxB, boxTop);
        const convex::ContactManifold mTetra    = manifoldFor(boxA, boxStatic, tetraH, tetraBody);
        const convex::ContactManifold mEdge     = manifoldFor(boxA, boxStatic, octaH, octaBody);

        // === (1) THE TEETER FIX — counts {boxOnBox:4, tetraOnFace:3, edgeOnFace:2}. ===
        check(mBoxOnBox.count == 4u, "MF2 clip: box-on-box manifold has 4 contact points");
        check(mTetra.count    == 3u, "MF2 clip: tetra-face-down manifold has 3 contact points");
        check(mEdge.count     == 2u, "MF2 clip: edge-on-face manifold has 2 contact points");

        // === (2) POINTS VALID + BELOW THE REF FACE — every kept depth >= 0; count <= 4. ===
        auto depthsOk = [&](const convex::ContactManifold& m) -> bool {
            if (m.count == 0u || m.count > 4u) return false;
            for (uint32_t k = 0; k < m.count; ++k) if (m.depths[k] < 0) return false;
            return true;
        };
        check(depthsOk(mBoxOnBox), "MF2 depths: box-on-box all points depth >= 0, count <= 4");
        check(depthsOk(mTetra),    "MF2 depths: tetra all points depth >= 0, count <= 4");
        check(depthsOk(mEdge),     "MF2 depths: edge all points depth >= 0, count <= 4");

        // === (4) CONSISTENCY — the manifold normal agrees with the EPA seed normal (FxDot > 0). ===
        auto normalAgrees = [&](const gjk::FxHull& hA, const fpx::FxBody& bA,
                                const gjk::FxHull& hB, const fpx::FxBody& bB,
                                const convex::ContactManifold& m) -> bool {
            const gjk::GjkResult g = gjk::Gjk(hA, bA, hB, bB);
            const gjk::EpaResult e = gjk::Epa(hA, bA, hB, bB, g.simplex);
            return convex::FxDot(m.normal, e.normal) > 0;
        };
        check(normalAgrees(boxA, boxStatic, boxB, boxTop, mBoxOnBox),
              "MF2 normal: box-on-box manifold normal == EPA normal (dot > 0)");
        check(normalAgrees(boxA, boxStatic, tetraH, tetraBody, mTetra),
              "MF2 normal: tetra manifold normal == EPA normal (dot > 0)");
        check(normalAgrees(boxA, boxStatic, octaH, octaBody, mEdge),
              "MF2 normal: edge manifold normal == EPA normal (dot > 0)");

        // === (3) DETERMINISM / PURITY — the manifold POD over the battery is byte-equal across two runs. ===
        auto manifoldSum = [&]() -> uint64_t {
            const convex::ContactManifold battery[3] = {
                manifoldFor(boxA, boxStatic, boxB, boxTop),
                manifoldFor(boxA, boxStatic, tetraH, tetraBody),
                manifoldFor(boxA, boxStatic, octaH, octaBody),
            };
            uint64_t sum = 0;
            for (int i = 0; i < 3; ++i) {
                const convex::ContactManifold& m = battery[i];
                sum = sum * 1000003ull + m.count;
                for (uint32_t k = 0; k < 4; ++k) {
                    sum = sum * 1000003ull + (uint64_t)(uint32_t)m.points[k].x;
                    sum = sum * 1000003ull + (uint64_t)(uint32_t)m.points[k].y;
                    sum = sum * 1000003ull + (uint64_t)(uint32_t)m.points[k].z;
                    sum = sum * 1000003ull + (uint64_t)(uint32_t)m.depths[k];
                }
                sum = sum * 1000003ull + (uint64_t)(uint32_t)m.normal.x;
                sum = sum * 1000003ull + (uint64_t)(uint32_t)m.normal.y;
                sum = sum * 1000003ull + (uint64_t)(uint32_t)m.normal.z;
            }
            return sum;
        };
        const uint64_t ms1 = manifoldSum();
        const uint64_t ms2 = manifoldSum();
        check(ms1 == ms2, "MF2 purity: HullManifoldFromEpa battery is two-run BYTE-EQUAL (pure)");

        // Also assert the manifold POD memcmp byte-equality directly (the strict guard).
        {
            const convex::ContactManifold a = manifoldFor(boxA, boxStatic, boxB, boxTop);
            const convex::ContactManifold b = manifoldFor(boxA, boxStatic, boxB, boxTop);
            check(std::memcmp(&a, &b, sizeof(convex::ContactManifold)) == 0,
                  "MF2 purity: box-on-box ContactManifold POD memcmp byte-equal across two runs");
        }

        // === (5) DEGENERATE/EDGE FALLBACK — an overlapping pair whose faces don't clip cleanly never gives
        //     count 0. Two deeply-coincident boxes (centers nearly the same) still yield count >= 1. ===
        {
            const fpx::FxBody coincident = makeBodyQ(kOne / 16, kOne / 16, kOne / 16);  // tiny offset overlap
            const convex::ContactManifold mc = manifoldFor(boxA, boxStatic, boxB, coincident);
            check(mc.count >= 1u, "MF2 fallback: a deeply-overlapping pair never yields count 0");
        }

        // =================================================================================================
        // Slice MF3 — THE MULTI-POINT MANIFOLD GPU SHADER. The CPU test of the NAMED hardened entry
        // manifold::HullContactMulti (the gjk::HullContact counterpart the GPU shader mirrors VERBATIM). The
        // GPU==CPU bit-identity is the showcase's job (--mf3-manifold-shot); this test PINS the CPU entry:
        //   * HullContactMulti over the SAME battery returns counts {4,3,2}.
        //   * it equals HullManifold/HullManifoldFromEpa (the SAME result — it IS the same call chain).
        //   * a separated pair -> count 0 (gjk::HullContact's empty-manifold contract).
        //   * two runs are byte-equal (pure).
        // =================================================================================================
        {
            const convex::ContactManifold cmBoxOnBox = manifold::HullContactMulti(boxStatic, boxA, boxTop, boxB);
            const convex::ContactManifold cmTetra    = manifold::HullContactMulti(boxStatic, boxA, tetraBody, tetraH);
            const convex::ContactManifold cmEdge     = manifold::HullContactMulti(boxStatic, boxA, octaBody, octaH);

            // (1) THE COUNTS {boxOnBox:4, tetraOnFace:3, edgeOnFace:2} (the MF2 multi-point manifold on GPU-entry).
            check(cmBoxOnBox.count == 4u, "MF3 HullContactMulti: box-on-box manifold has 4 contact points");
            check(cmTetra.count    == 3u, "MF3 HullContactMulti: tetra-face-down manifold has 3 contact points");
            check(cmEdge.count     == 2u, "MF3 HullContactMulti: edge-on-face manifold has 2 contact points");

            // (2) it equals HullManifold (the convenience wrapper) AND HullManifoldFromEpa (the MF2 core) —
            //     HullContactMulti IS the same call chain, so the ContactManifold POD memcmp byte-equals.
            const convex::ContactManifold hm = manifold::HullManifold(boxStatic, boxA, boxTop, boxB);
            check(std::memcmp(&cmBoxOnBox, &hm, sizeof(convex::ContactManifold)) == 0,
                  "MF3 HullContactMulti == HullManifold (same call chain, POD memcmp byte-equal)");
            const convex::ContactManifold fromEpa = manifoldFor(boxA, boxStatic, boxB, boxTop);
            // manifoldFor uses A=boxStatic-as-hullA order matching HullContactMulti(boxStatic,boxA,...).
            check(std::memcmp(&cmBoxOnBox, &fromEpa, sizeof(convex::ContactManifold)) == 0,
                  "MF3 HullContactMulti == HullManifoldFromEpa (same result)");

            // (3) a SEPARATED pair -> count 0 (the gjk::HullContact empty-manifold contract).
            {
                const fpx::FxBody farAway = makeBodyQ(FromInt(20), FromInt(20), FromInt(20));
                const convex::ContactManifold mc = manifold::HullContactMulti(boxStatic, boxA, farAway, boxB);
                check(mc.count == 0u, "MF3 HullContactMulti: a separated pair -> count 0");
            }

            // (4) PURITY — the manifold POD over the battery is byte-equal across two runs.
            auto multiSum = [&]() -> uint64_t {
                const convex::ContactManifold battery[3] = {
                    manifold::HullContactMulti(boxStatic, boxA, boxTop, boxB),
                    manifold::HullContactMulti(boxStatic, boxA, tetraBody, tetraH),
                    manifold::HullContactMulti(boxStatic, boxA, octaBody, octaH),
                };
                uint64_t sum = 0;
                for (int i = 0; i < 3; ++i) {
                    const convex::ContactManifold& m = battery[i];
                    sum = sum * 1000003ull + m.count;
                    for (uint32_t k = 0; k < 4; ++k) {
                        sum = sum * 1000003ull + (uint64_t)(uint32_t)m.points[k].x;
                        sum = sum * 1000003ull + (uint64_t)(uint32_t)m.points[k].y;
                        sum = sum * 1000003ull + (uint64_t)(uint32_t)m.points[k].z;
                        sum = sum * 1000003ull + (uint64_t)(uint32_t)m.depths[k];
                    }
                    sum = sum * 1000003ull + (uint64_t)(uint32_t)m.normal.x;
                    sum = sum * 1000003ull + (uint64_t)(uint32_t)m.normal.y;
                    sum = sum * 1000003ull + (uint64_t)(uint32_t)m.normal.z;
                }
                return sum;
            };
            check(multiSum() == multiSum(), "MF3 HullContactMulti: battery is two-run BYTE-EQUAL (pure)");
        }
    }

    // =================================================================================================
    // Slice MF4 — FULL INERTIA + THE HARDENED RESTACKED-STABILITY STEP. Pins:
    //   * (1) THE CUBE CROSS-CHECK: FxMat3SymInverse(FxHullInertiaBodyFull(MakeBox)) == gjk::FxHullInvInertiaBody
    //     (the full signed-tetra tensor reduces to the AABB diagonal for a cube; off-diagonals zero).
    //   * (2) FxMat3SymInverse round-trips a known symmetric matrix (M·M⁻¹ ≈ I within tol).
    //   * (3) THE HEADLINE: a tilted box dropped flat onto a static box SETTLES under StepHullWorldHardenedN
    //     (maxAngVel below a fixed band) while the SAME scene under the frozen gjk::StepHullWorldN TEETERS.
    //   * (4) a tumbling tetra's trajectory is two-run byte-equal (deterministic).
    //   * (5) the hardened step is deterministic (two runs byte-equal).
    // =================================================================================================
    {
        // (1) THE CUBE CROSS-CHECK.
        const gjk::FxHull cube = gjk::MakeBox(kOne, kOne, kOne);
        const manifold::FxHullFaces cubeFaces = manifold::BuildCanonicalFaces(cube);
        const convex::FxMat3 full = manifold::FxHullInertiaBodyFull(cube, cubeFaces, kOne);
        const convex::FxVec3 diag = gjk::FxHullInvInertiaBody(cube, kOne);
        const fx fdiag[3] = {full.m[0], full.m[4], full.m[8]};
        const fx ddiag[3] = {diag.x, diag.y, diag.z};
        fx maxErr = 0;
        for (int k = 0; k < 3; ++k) { fx e = fdiag[k] - ddiag[k]; if (e < 0) e = -e; if (e > maxErr) maxErr = e; }
        fx maxOff = 0;
        const fx off[3] = {full.m[1], full.m[2], full.m[5]};
        for (int k = 0; k < 3; ++k) { fx e = off[k]; if (e < 0) e = -e; if (e > maxOff) maxOff = e; }
        const fx tol = kOne / 64;
        check(maxErr <= tol, "MF4 cube cross-check: full inertia diagonal == AABB diagonal within tol");
        check(maxOff <= tol, "MF4 cube cross-check: full inertia off-diagonals ~zero for a cube");

        // A static body -> the zero matrix.
        const convex::FxMat3 staticI = manifold::FxHullInertiaBodyFull(cube, cubeFaces, 0);
        bool allZero = true;
        for (int k = 0; k < 9; ++k) if (staticI.m[k] != 0) allZero = false;
        check(allZero, "MF4 FxHullInertiaBodyFull: a static body (invMass==0) -> zero matrix");

        // (2) FxMat3SymInverse round-trips a known symmetric matrix (M·M⁻¹ ≈ I).
        convex::FxMat3 S;
        S.m[0] = 2 * kOne; S.m[4] = 3 * kOne; S.m[8] = 4 * kOne;
        S.m[1] = S.m[3] = kOne / 2; S.m[2] = S.m[6] = kOne / 4; S.m[5] = S.m[7] = kOne / 3;
        const convex::FxMat3 Si = manifold::FxMat3SymInverse(S);
        // P = S·Si (manual fixed-point matmul).
        convex::FxMat3 P;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                int64_t s = 0;
                for (int k = 0; k < 3; ++k) s += ((int64_t)S.m[r * 3 + k] * (int64_t)Si.m[k * 3 + c]) >> 16;
                P.m[r * 3 + c] = (fx)s;
            }
        const fx idTol = kOne / 256;   // ~0.004
        bool roundtrip = true;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                const fx want = (r == c) ? kOne : 0;
                fx e = P.m[r * 3 + c] - want; if (e < 0) e = -e;
                if (e > idTol) roundtrip = false;
            }
        check(roundtrip, "MF4 FxMat3SymInverse: M*Minv ~ identity (round-trip within tol)");

        // The hardened-step config (== the showcases). angDamp OFF so the teeter is real.
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        convex::ConvexStepConfig cfg;
        cfg.gravity     = convex::FxVec3{0, kGravY, 0};
        cfg.dt          = kOne / 60;
        cfg.solveIters  = 24;
        cfg.restitution = 0;
        cfg.slop        = kOne / 64;
        cfg.beta        = (fx)((int64_t)2 * kOne / 10);
        cfg.linDamp     = (fx)((int64_t)95 * kOne / 100);
        cfg.angDamp     = kOne;
        cfg.posIters    = 2;
        const uint32_t K = 300u;

        auto makeDyn = [&](fx x, fx y, fx z, const fpx::FxQuat& q) {
            fpx::FxBody b; b.pos = {x, y, z}; b.orient = q; b.invMass = kOne; b.flags = fpx::kFlagDynamic;
            b.vel = {0, 0, 0}; b.angVel = {0, 0, 0}; return b;
        };
        auto makeStat = [&](fx x, fx y, fx z) {
            fpx::FxBody b; b.pos = {x, y, z}; b.orient = {0, 0, 0, kOne}; b.invMass = 0; b.flags = 0u;
            b.vel = {0, 0, 0}; b.angVel = {0, 0, 0}; return b;
        };
        const fpx::FxQuat tilt{0, 0, (fx)(0.024997 * (double)kOne), (fx)(0.999688 * (double)kOne)};
        auto buildStack = [&]() {
            gjk::HullWorld w;
            w.bodies.push_back(makeStat(0, 0, 0));                      w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
            w.bodies.push_back(makeDyn(0, (fx)(2.3 * (double)kOne), 0, tilt)); w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
            return w;
        };
        auto maxAngVel = [&](const gjk::HullWorld& w) -> fx {
            fx m = 0;
            for (const auto& b : w.bodies) if (convex::IsDynamic(b)) { fx a = fpx::FxLength(b.angVel); if (a > m) m = a; }
            return m;
        };
        auto maxSpeed = [&](const gjk::HullWorld& w) -> fx {
            fx m = 0;
            for (const auto& b : w.bodies) if (convex::IsDynamic(b)) { fx a = fpx::FxLength(b.vel); if (a > m) m = a; }
            return m;
        };

        // (3) THE HEADLINE — hardened settles, frozen teeters.
        gjk::HullWorld hard = buildStack(); manifold::StepHullWorldHardenedN(hard, cfg, K);
        gjk::HullWorld froz = buildStack(); gjk::StepHullWorldN(froz, cfg, K);
        const fx band = (fx)((int64_t)5 * kOne / 100);   // 0.05
        check(maxAngVel(hard) < band && maxSpeed(hard) < band,
              "MF4 headline: the hardened step SETTLES the tilted box (maxAngVel/maxSpeed below band)");
        check(maxAngVel(froz) >= band,
              "MF4 headline: the frozen single-point step TEETERS the SAME scene (maxAngVel above band)");

        // (5) the hardened step is deterministic (two runs byte-equal).
        gjk::HullWorld a = buildStack(); manifold::StepHullWorldHardenedN(a, cfg, K);
        gjk::HullWorld b = buildStack(); manifold::StepHullWorldHardenedN(b, cfg, K);
        bool det = (a.bodies.size() == b.bodies.size());
        for (size_t i = 0; det && i < a.bodies.size(); ++i)
            if (std::memcmp(&a.bodies[i], &b.bodies[i], sizeof(fpx::FxBody)) != 0) det = false;
        check(det, "MF4 StepHullWorldHardenedN: two runs BYTE-IDENTICAL (deterministic)");

        // (4) a tumbling tetra's trajectory is two-run byte-equal (spin + gravity, deterministic).
        auto buildTumble = [&]() {
            gjk::HullWorld w;
            w.bodies.push_back(makeStat(0, 0, 0));   w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
            fpx::FxBody t = makeDyn(0, (fx)(3.0 * (double)kOne), 0, fpx::FxQuat{0, 0, 0, kOne});
            t.angVel = {kOne / 2, kOne / 3, 0};   // a tumble
            w.bodies.push_back(t);                   w.hulls.push_back(gjk::MakeTetra(kOne));
            return w;
        };
        gjk::HullWorld t1 = buildTumble(); manifold::StepHullWorldHardenedN(t1, cfg, 120u);
        gjk::HullWorld t2 = buildTumble(); manifold::StepHullWorldHardenedN(t2, cfg, 120u);
        bool tumbleEq = (t1.bodies.size() == t2.bodies.size());
        for (size_t i = 0; tumbleEq && i < t1.bodies.size(); ++i)
            if (std::memcmp(&t1.bodies[i], &t2.bodies[i], sizeof(fpx::FxBody)) != 0) tumbleEq = false;
        check(tumbleEq, "MF4 tumbling tetra: trajectory is two-run BYTE-EQUAL (deterministic)");
    }

    // =================================================================================================
    // Slice MF5 — LOCKSTEP + ROLLBACK over the hardened stack (the NETCODE HEADLINE). Pins (PURE CPU):
    //   * (1) LOCKSTEP: RunHullLockstepHardened authority==replica BIT-IDENTICAL (inputs-only re-sim).
    //   * (2) DETERMINISM: two full RunHullLockstepHardened runs byte-identical (+ snapshot round-trip).
    //   * (3) ROLLBACK: RunHullRollbackHardened corrected==authority BIT-EXACT.
    //   * (4) MISPREDICT REAL: the speculative pre-rollback state genuinely DIVERGED from the authority.
    //   * (5) THE COMMAND STREAM MOVED THE STACK NON-TRIVIALLY (the hardened step + re-manifold actually
    //     exercised — the converged world differs from the no-command settle; not a frozen no-op).
    // =================================================================================================
    {
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        convex::ConvexStepConfig cfg;
        cfg.gravity     = convex::FxVec3{0, kGravY, 0};
        cfg.dt          = kOne / 60;
        cfg.solveIters  = 24;
        cfg.restitution = 0;
        cfg.slop        = kOne / 64;
        cfg.beta        = (fx)((int64_t)2 * kOne / 10);
        cfg.linDamp     = (fx)((int64_t)95 * kOne / 100);
        cfg.angDamp     = kOne;
        cfg.posIters    = 2;
        const uint32_t kTicks      = 240u;
        const uint32_t kRollbackAt = 30u;

        auto makeBody = [&](fx x, fx y, fx z, bool dyn, const fpx::FxQuat& q) {
            fpx::FxBody b; b.pos = {x, y, z}; b.orient = q;
            b.invMass = dyn ? kOne : 0; b.flags = dyn ? fpx::kFlagDynamic : 0u;
            b.vel = {0, 0, 0}; b.angVel = {0, 0, 0}; return b;
        };
        const fpx::FxQuat kIdentity{0, 0, 0, kOne};
        const fpx::FxQuat kTilt{0, 0, (fx)(0.024997 * (double)kOne), (fx)(0.999688 * (double)kOne)};
        auto buildScene = [&]() {
            gjk::HullWorld w;
            w.bodies.push_back(makeBody(0, 0, 0, false, kIdentity));                  w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
            w.bodies.push_back(makeBody(0, (fx)(2.3 * (double)kOne), 0, true, kTilt)); w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
            w.bodies.push_back(makeBody(0, (fx)(4.5 * (double)kOne), 0, true, kIdentity)); w.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));
            return w;
        };
        const gjk::HullWorld kInit = buildScene();

        const std::vector<convex::ConvexCommand> authStream = {
            convex::ConvexCommand{6u,  convex::kConvexCmdAddImpulse, 2u, convex::FxVec3{2 * kOne, 0, 0}},
            convex::ConvexCommand{10u, convex::kConvexCmdAddImpulse, 1u, convex::FxVec3{kOne, 0, 0}},
            convex::ConvexCommand{14u, convex::kConvexCmdSetAngVel,  2u, convex::FxVec3{0, 0, kOne / 2}},
        };
        std::vector<convex::ConvexCommand> mispredictStream = authStream;
        mispredictStream.push_back(convex::ConvexCommand{kRollbackAt, convex::kConvexCmdAddImpulse, 2u,
                                                         convex::FxVec3{30 * kOne, 0, 0}});

        // (1) LOCKSTEP.
        bool lockstepIdentical = false;
        const gjk::HullWorld authority =
            manifold::RunHullLockstepHardened(kInit, cfg, authStream, kTicks, &lockstepIdentical);
        const gjk::HullWorld replica = manifold::RunHullLockstepHardened(kInit, cfg, authStream, kTicks);
        check(lockstepIdentical && gjk::HullBodiesEqual(authority.bodies, replica.bodies),
              "MF5 RunHullLockstepHardened: authority==replica BIT-IDENTICAL (inputs-only re-sim)");

        // (2) DETERMINISM (+ snapshot round-trip).
        const gjk::HullWorld authority2 = manifold::RunHullLockstepHardened(kInit, cfg, authStream, kTicks);
        check(gjk::HullBodiesEqual(authority2.bodies, authority.bodies),
              "MF5 RunHullLockstepHardened: two runs BYTE-IDENTICAL (deterministic)");
        {
            gjk::HullWorld w = manifold::RunHullLockstepHardened(kInit, cfg, authStream, kRollbackAt);
            const gjk::HullSnapshot snap = gjk::SnapshotHull(w, kRollbackAt);
            manifold::SimHullTickHardened(w, cfg, authStream, kRollbackAt);   // mutate
            gjk::RestoreHull(w, snap);
            check(gjk::HullBodiesEqual(w.bodies, snap.bodies),
                  "MF5 SnapshotHull/RestoreHull: snapshot round-trip BIT-EXACT");
        }

        // (3) ROLLBACK + (4) MISPREDICT REAL.
        bool rollbackCorrected = false, mispredictDiverged = false;
        const gjk::HullWorld rolledBack =
            manifold::RunHullRollbackHardened(kInit, cfg, authStream, mispredictStream, kTicks, kRollbackAt,
                                              &rollbackCorrected, &mispredictDiverged);
        check(rollbackCorrected && gjk::HullBodiesEqual(rolledBack.bodies, authority.bodies),
              "MF5 RunHullRollbackHardened: corrected==authority BIT-EXACT");
        check(mispredictDiverged,
              "MF5 RunHullRollbackHardened: mispredict diverged before rollback (real divergence corrected)");

        // (5) THE COMMAND STREAM MOVED THE STACK NON-TRIVIALLY — the commanded converged world differs from
        // the no-command settle (the hardened step + re-manifold are actually exercised, not a frozen no-op).
        const std::vector<convex::ConvexCommand> noCmds;
        const gjk::HullWorld settled = manifold::RunHullLockstepHardened(kInit, cfg, noCmds, kTicks);
        check(!gjk::HullBodiesEqual(authority.bodies, settled.bodies),
              "MF5 command stream MOVED the stack non-trivially (commanded world != no-command settle)");
    }

    if (g_fail == 0) std::printf("manifold_test: ALL PASS\n");
    else std::printf("manifold_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
