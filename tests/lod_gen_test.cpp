// Slice LOD1 — DETERMINISTIC AUTOMATIC LOD GENERATION (Track-S S6): integer quadric-error-metric
// edge-collapse decimation (engine/render/lod_gen.h). Namespace hf::render::vg.
//
// What this test PINS (the cross-platform pinned-golden discipline — the SAME integer digests must
// come out of MSVC, Windows clang, and Mac clang):
//   * QuantizeWeld on SphereGeometry(48,32): welded V/F counts + dropped pole-degenerates + the input
//     digest; the welded sphere is a CLOSED manifold (every edge on exactly 2 faces, V-E+F == 2).
//   * DETERMINISM: DecimateMesh twice -> byte-identical (qpos/indices/srcVert memcmp) + the pinned
//     FNV-1a-64 digests (net::DigestBytes over V,T,qpos,indices).
//   * CORRECTNESS: LOD1 (~50%) + LOD2 (~25%) triangle counts pinned (and within 1 of the target);
//     NO lattice-degenerate triangles (integer cross != 0); NO out-of-range indices; EULER sanity
//     (result still closed, V-E+F == 2); result AABB inside the input AABB (+float-dequant eps) and
//     within a pinned epsilon of it per axis (the shape did not collapse).
//   * QUALITY: max committed quadric error + max accumulated displacement pinned per LOD (the honest
//     integer quality numbers).
//   * INTEGRATION: BuildAutoLods -> BuildAutoLodMeshes feeds the EXISTING byte-untouched cluster_lod
//     BuildLodMeshes/SelectLod path: near -> LOD0, very-far -> LOD2, distance-monotonic sweep spans
//     all 3 LODs, clamp — the same near/far expectations cluster_lod_test pins for hand LODs.
//   * NON-SPHERE fixtures: an OPEN 9x9 grid (boundary verts all survive EXACTLY — v1 forbids boundary
//     collapses — corners included; interior decimated) and the welded CUBE (corners are features:
//     pinned digest + validity + Euler on whatever the guards honestly allow).
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests. No GPU, no RNG, no clock.
#include "render/lod_gen.h"
#include "render/cluster_lod.h"
#include "render/meshlet.h"
#include "scene/mesh.h"
#include "scene/vertex.h"
#include "math/math.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vg = hf::render::vg;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// ---------------- pinned goldens (filled from the first verified run; integer-only) ----------------
static constexpr uint32_t kPinSphereWeldV      = 1490;
static constexpr uint32_t kPinSphereWeldF      = 2976;
static constexpr uint32_t kPinSphereWeldDrop   = 96;    // the 2*48 pole-fan degenerates
static constexpr uint64_t kPinSphereDigest0    = 0xeade209d8e61838full;
static constexpr uint32_t kPinSphereLod1Tris   = 1488;
static constexpr uint32_t kPinSphereLod2Tris   = 744;
static constexpr uint64_t kPinSphereDigest1    = 0xbedec704837c71f4ull;
static constexpr uint64_t kPinSphereDigest2    = 0x1a9821628ab77e80ull;
static constexpr uint64_t kPinSphereMaxQErr1   = 172325955ull;
static constexpr uint64_t kPinSphereMaxQErr2   = 1124451217ull;
static constexpr uint32_t kPinSphereMaxDisp1   = 233;
static constexpr uint32_t kPinSphereMaxDisp2   = 282;
static constexpr uint32_t kPinGridLodTris      = 64;
static constexpr uint64_t kPinGridDigest       = 0x6055345bb4463690ull;
static constexpr uint32_t kPinCubeLodTris      = 6;
static constexpr uint64_t kPinCubeDigest       = 0xaf4f3c97f26bd28full;

// ---------------- mesh validity helpers (all integer) -----------------------------------------------
struct LatticeCheck {
    bool idxOk = true;        // every index < V
    bool nonDegenerate = true;// every face has a non-zero integer cross product (area > 0)
    bool closed = true;       // every edge on exactly 2 faces
    int64_t euler = 0;        // V - E + F
};
static LatticeCheck CheckLattice(const vg::LodGenMesh& m) {
    LatticeCheck r;
    const uint32_t V = (uint32_t)(m.qpos.size() / 3);
    const uint32_t F = (uint32_t)(m.indices.size() / 3);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edges;
    for (uint32_t f = 0; f < F; ++f) {
        const uint32_t i[3] = {m.indices[3 * f], m.indices[3 * f + 1], m.indices[3 * f + 2]};
        for (int k = 0; k < 3; ++k) {
            if (i[k] >= V) { r.idxOk = false; return r; }
            uint32_t a = i[k], b = i[(k + 1) % 3];
            if (a > b) std::swap(a, b);
            edges[{a, b}]++;
        }
        vg::LodQuadric q;
        if (!vg::LodFaceQuadric(&m.qpos[3 * i[0]], &m.qpos[3 * i[1]], &m.qpos[3 * i[2]], q))
            r.nonDegenerate = false;
    }
    for (const auto& e : edges)
        if (e.second != 2) r.closed = false;
    r.euler = (int64_t)V - (int64_t)edges.size() + (int64_t)F;
    return r;
}

static bool SameLodGenMesh(const vg::LodGenMesh& a, const vg::LodGenMesh& b) {
    return a.qpos.size() == b.qpos.size() && a.indices.size() == b.indices.size() &&
           a.srcVert.size() == b.srcVert.size() &&
           (a.qpos.empty() ||
            std::memcmp(a.qpos.data(), b.qpos.data(), a.qpos.size() * sizeof(int32_t)) == 0) &&
           (a.indices.empty() ||
            std::memcmp(a.indices.data(), b.indices.data(), a.indices.size() * sizeof(uint32_t)) == 0) &&
           (a.srcVert.empty() ||
            std::memcmp(a.srcVert.data(), b.srcVert.data(), a.srcVert.size() * sizeof(uint32_t)) == 0);
}

// Float AABB of a MeshGeometry (I/O-boundary check only; never feeds a decision).
static void GeoAabb(const scene::MeshGeometry& g, float lo[3], float hi[3]) {
    for (int k = 0; k < 3; ++k) { lo[k] = 1e30f; hi[k] = -1e30f; }
    for (const scene::Vertex& v : g.verts)
        for (int k = 0; k < 3; ++k) {
            if (v.pos[k] < lo[k]) lo[k] = v.pos[k];
            if (v.pos[k] > hi[k]) hi[k] = v.pos[k];
        }
}

// An OPEN 9x9 unit-grid plane fixture (boundaries + corners; y == 0).
static scene::MeshGeometry GridGeometry(uint32_t n) {
    scene::MeshGeometry g;
    for (uint32_t j = 0; j <= n; ++j)
        for (uint32_t i = 0; i <= n; ++i) {
            scene::Vertex v{};
            v.pos[0] = (float)i; v.pos[1] = 0.0f; v.pos[2] = (float)j;
            v.color[0] = v.color[1] = v.color[2] = 0.8f;
            v.uv[0] = (float)i / (float)n; v.uv[1] = (float)j / (float)n;
            v.normal[1] = 1.0f; v.tangent[0] = 1.0f;
            g.verts.push_back(v);
        }
    const uint32_t stride = n + 1;
    for (uint32_t j = 0; j < n; ++j)
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t a = j * stride + i, b = a + 1, c = a + stride, d = c + 1;
            g.indices.insert(g.indices.end(), {a, c, b, b, c, d});
        }
    return g;
}

int main() {
    HF_TEST_MAIN_INIT();
    using math::Mat4; using math::Vec3;

    // ================= QuantizeWeld: the sphere becomes a CLOSED welded manifold =================
    const scene::MeshGeometry sphere = scene::SphereGeometry(48, 32);
    const std::span<const scene::Vertex> sv(sphere.verts.data(), sphere.verts.size());
    const std::span<const uint32_t> si(sphere.indices.data(), sphere.indices.size());
    const vg::LodGenMesh q0 = vg::QuantizeWeld(sv, si);
    const uint32_t weldV = (uint32_t)(q0.qpos.size() / 3);
    const uint32_t weldF = (uint32_t)(q0.indices.size() / 3);
    std::printf("sphere weld: V=%u F=%u dropped=%u digest0=0x%016llx\n",
                weldV, weldF, q0.weldDropped, (unsigned long long)vg::LodGenDigest(q0));
    check(weldV == kPinSphereWeldV, "welded sphere V pinned");
    check(weldF == kPinSphereWeldF, "welded sphere F pinned");
    check(q0.weldDropped == kPinSphereWeldDrop, "welded sphere dropped pole-degenerates pinned");
    check(vg::LodGenDigest(q0) == kPinSphereDigest0, "welded sphere digest pinned");
    {
        const LatticeCheck c = CheckLattice(q0);
        check(c.idxOk && c.nonDegenerate, "welded sphere valid (indices in range, no degenerates)");
        check(c.closed && c.euler == 2, "welded sphere CLOSED manifold (every edge 2 faces, V-E+F==2)");
    }

    // ================= DETERMINISM: two decimations byte-identical + pinned digests ==============
    vg::DecimateStats s1{}, s1b{}, s2{};
    const vg::LodGenMesh q1  = vg::DecimateMesh(q0, weldF / 2, &s1);
    const vg::LodGenMesh q1b = vg::DecimateMesh(q0, weldF / 2, &s1b);
    const vg::LodGenMesh q2  = vg::DecimateMesh(q1, weldF / 4, &s2);
    check(SameLodGenMesh(q1, q1b), "determinism: two DecimateMesh runs BYTE-IDENTICAL");
    check(s1.collapses == s1b.collapses && s1.maxQuadricError == s1b.maxQuadricError &&
          s1.maxDispLattice == s1b.maxDispLattice, "determinism: two runs identical stats");
    const uint32_t tris1 = (uint32_t)(q1.indices.size() / 3);
    const uint32_t tris2 = (uint32_t)(q2.indices.size() / 3);
    std::printf("sphere lods: %u -> %u -> %u tris (targets %u/%u) collapses=%u/%u\n",
                weldF, tris1, tris2, weldF / 2, weldF / 4, s1.collapses, s2.collapses);
    std::printf("sphere digests: d1=0x%016llx d2=0x%016llx\n",
                (unsigned long long)vg::LodGenDigest(q1), (unsigned long long)vg::LodGenDigest(q2));
    std::printf("sphere quality: maxQErr1=%llu maxQErr2=%llu maxDisp1=%u maxDisp2=%u\n",
                (unsigned long long)s1.maxQuadricError, (unsigned long long)s2.maxQuadricError,
                s1.maxDispLattice, s2.maxDispLattice);
    check(vg::LodGenDigest(q1) == kPinSphereDigest1, "LOD1 digest pinned (the cross-platform golden)");
    check(vg::LodGenDigest(q2) == kPinSphereDigest2, "LOD2 digest pinned (the cross-platform golden)");

    // ================= CORRECTNESS: counts, validity, Euler, AABB =================================
    check(tris1 == kPinSphereLod1Tris, "LOD1 triangle count pinned");
    check(tris2 == kPinSphereLod2Tris, "LOD2 triangle count pinned");
    check(tris1 <= weldF / 2 && tris1 + 1 >= weldF / 2, "LOD1 within 1 of the ~50% target");
    check(tris2 <= weldF / 4 && tris2 + 1 >= weldF / 4, "LOD2 within 1 of the ~25% target");
    {
        const LatticeCheck c1 = CheckLattice(q1);
        check(c1.idxOk, "LOD1 indices in range");
        check(c1.nonDegenerate, "LOD1 has NO degenerate triangles (integer area > 0)");
        check(c1.closed && c1.euler == 2, "LOD1 still a CLOSED manifold (Euler V-E+F==2)");
        const LatticeCheck c2 = CheckLattice(q2);
        check(c2.idxOk, "LOD2 indices in range");
        check(c2.nonDegenerate, "LOD2 has NO degenerate triangles (integer area > 0)");
        check(c2.closed && c2.euler == 2, "LOD2 still a CLOSED manifold (Euler V-E+F==2)");
    }
    {
        float inLo[3], inHi[3];
        GeoAabb(sphere, inLo, inHi);
        const float kOuterEps = 1e-4f;  // float dequant slack (lattice points stay inside the frame)
        const float kShapeEps = 0.05f;  // 10% of the 0.5 radius: the shape did not collapse
        const scene::MeshGeometry g1 = vg::ToMeshGeometry(q1, sv);
        const scene::MeshGeometry g2 = vg::ToMeshGeometry(q2, sv);
        for (const scene::MeshGeometry* g : {&g1, &g2}) {
            float lo[3], hi[3];
            GeoAabb(*g, lo, hi);
            bool inside = true, close = true;
            for (int k = 0; k < 3; ++k) {
                if (lo[k] < inLo[k] - kOuterEps || hi[k] > inHi[k] + kOuterEps) inside = false;
                if (lo[k] > inLo[k] + kShapeEps || hi[k] < inHi[k] - kShapeEps) close = false;
            }
            check(inside, "decimated AABB inside the input AABB (+dequant eps)");
            check(close, "decimated AABB within the pinned epsilon of the input AABB");
        }
    }

    // ================= QUALITY pins (the honest integer numbers) =================================
    check(s1.maxQuadricError == kPinSphereMaxQErr1, "LOD1 max quadric error pinned");
    check(s2.maxQuadricError == kPinSphereMaxQErr2, "LOD2 max quadric error pinned");
    check(s1.maxDispLattice == kPinSphereMaxDisp1, "LOD1 max displacement pinned");
    check(s2.maxDispLattice == kPinSphereMaxDisp2, "LOD2 max displacement pinned");

    // ================= INTEGRATION: generated LODs through the EXISTING SelectLod =================
    {
        const vg::AutoLods al = vg::BuildAutoLods(sphere);
        check(al.triCount[0] == weldF && al.triCount[1] == tris1 && al.triCount[2] == tris2,
              "BuildAutoLods counts match the standalone chain");
        check(al.digest[1] == kPinSphereDigest1 && al.digest[2] == kPinSphereDigest2,
              "BuildAutoLods digests match the standalone chain");
        check(al.geometricError[0] == 0.0f, "generated LOD0 geometricError == 0 (full detail)");
        check(al.geometricError[1] > 0.0f && al.geometricError[2] > al.geometricError[1],
              "generated errors strictly increase with coarseness (the SelectLod contract)");
        std::printf("auto errors: e1=%.6f e2=%.6f\n", al.geometricError[1], al.geometricError[2]);

        const vg::AutoLods al2 = vg::BuildAutoLods(sphere);
        check(al.digest == al2.digest, "determinism: two BuildAutoLods -> identical digests");

        vg::LodMeshes lm = vg::BuildAutoLodMeshes(al);
        // Cluster ranges: contiguous, exact cover, coarser LODs have fewer clusters (DS decomposition
        // of the generated meshes) — the same structural checks cluster_lod_test makes on hand LODs.
        uint32_t expectFirst = 0, totalClusters = 0;
        bool contiguous = true;
        for (uint32_t n = 0; n < vg::kNumLods; ++n) {
            if (lm.lods[n].firstCluster != expectFirst || lm.lods[n].clusterCount == 0)
                contiguous = false;
            expectFirst += lm.lods[n].clusterCount;
            totalClusters += lm.lods[n].clusterCount;
        }
        check(contiguous, "generated LOD cluster ranges contiguous + non-empty");
        check(totalClusters == (uint32_t)lm.combined.meshlets.size(),
              "generated LOD ranges cover every combined cluster exactly once");
        check(lm.lods[0].clusterCount >= lm.lods[1].clusterCount &&
              lm.lods[1].clusterCount >= lm.lods[2].clusterCount,
              "coarser generated LODs have no more clusters");

        // SelectLod: the SAME near/far/monotonic expectations the hand-authored test pins.
        const float fovY = 1.0471976f;  // 60 deg
        const int   H = 720;
        const float projScale = vg::ProjectionScaleForScreenError(fovY, H);
        Mat4 view = Mat4::LookAt({0, 0, 0}, {0, 0, -1}, {0, 1, 0});
        std::array<float, vg::kNumLods> errs{};
        for (uint32_t n = 0; n < vg::kNumLods; ++n) errs[n] = lm.lods[n].geometricError;
        std::span<const float> es(errs.data(), errs.size());
        const float threshold = 1.0f, scale = 1.0f;

        check(vg::SelectLod(es, Vec3{0, 0, -2.0f}, view, projScale, threshold, scale, false) == 0u,
              "SelectLod near -> LOD0 on GENERATED LODs");
        check(vg::SelectLod(es, Vec3{0, 0, -100000.0f}, view, projScale, threshold, scale, false)
                  == vg::kNumLods - 1u,
              "SelectLod very-far -> coarsest GENERATED LOD");
        check(vg::SelectLod(es, Vec3{0, 0, -100000.0f}, view, projScale, threshold, scale, true) == 0u,
              "SelectLod forceLod0 -> LOD0 on generated LODs");
        bool monotone = true, clamped = true;
        uint32_t prev = 0, distinctSeen = 0;
        bool seen[vg::kNumLods] = {false, false, false};
        for (int s = 1; s <= 4000; ++s) {
            const float z = -(float)s * 0.5f;
            const uint32_t lod = vg::SelectLod(es, Vec3{0, 0, z}, view, projScale, threshold, scale,
                                               false);
            if (lod < prev) monotone = false;
            if (lod >= vg::kNumLods) clamped = false;
            prev = lod;
            if (!seen[lod]) { seen[lod] = true; ++distinctSeen; }
        }
        check(monotone, "SelectLod distance-monotonic on GENERATED LODs (farther never finer)");
        check(clamped, "SelectLod clamped to [0, kNumLods-1] on generated LODs");
        check(distinctSeen == vg::kNumLods,
              "SelectLod spans all 3 GENERATED LODs across the increasing-distance sweep");
    }

    // ================= NON-SPHERE fixture 1: the OPEN grid (boundaries + corners survive) =========
    {
        const scene::MeshGeometry grid = GridGeometry(8);  // 81 verts, 128 tris, open
        const std::span<const scene::Vertex> gv(grid.verts.data(), grid.verts.size());
        const vg::LodGenMesh gq = vg::QuantizeWeld(
            gv, std::span<const uint32_t>(grid.indices.data(), grid.indices.size()));
        check((uint32_t)(gq.qpos.size() / 3) == 81 && (uint32_t)(gq.indices.size() / 3) == 128 &&
              gq.weldDropped == 0, "grid quantizes 1:1 (no welds, no drops)");
        vg::DecimateStats gs{};
        const vg::LodGenMesh gd = vg::DecimateMesh(gq, 64, &gs);
        const uint32_t gTris = (uint32_t)(gd.indices.size() / 3);
        std::printf("grid: 128 -> %u tris collapses=%u digest=0x%016llx maxQErr=%llu\n",
                    gTris, gs.collapses, (unsigned long long)vg::LodGenDigest(gd),
                    (unsigned long long)gs.maxQuadricError);
        check(gTris == kPinGridLodTris, "grid decimated triangle count pinned");
        check(gTris < 128, "grid interior actually decimated");
        check(vg::LodGenDigest(gd) == kPinGridDigest, "grid digest pinned");
        {
            const LatticeCheck c = CheckLattice(gd);
            check(c.idxOk && c.nonDegenerate,
                  "grid result valid (indices in range, no degenerates)");
        }
        // Boundary preservation: EVERY input boundary lattice point survives exactly (v1 forbids
        // boundary collapses), corners included — corners survive BETTER than interior verts.
        std::map<std::array<int32_t, 3>, int> outPts;
        for (size_t i = 0; i + 2 < gd.qpos.size(); i += 3)
            outPts[{gd.qpos[i], gd.qpos[i + 1], gd.qpos[i + 2]}] = 1;
        uint32_t bdryIn = 0, bdryKept = 0;
        bool cornersKept = true;
        for (uint32_t j = 0; j <= 8; ++j)
            for (uint32_t i = 0; i <= 8; ++i) {
                if (i != 0 && i != 8 && j != 0 && j != 8) continue;  // boundary ring only
                const uint32_t vi = j * 9 + i;
                const std::array<int32_t, 3> p{gq.qpos[3 * vi], gq.qpos[3 * vi + 1],
                                               gq.qpos[3 * vi + 2]};
                ++bdryIn;
                if (outPts.count(p)) ++bdryKept;
                else if ((i == 0 || i == 8) && (j == 0 || j == 8)) cornersKept = false;
            }
        check(bdryKept == bdryIn, "ALL grid boundary vertices survive EXACTLY (v1 boundary rule)");
        check(cornersKept, "all 4 grid corners survive");
        check((uint32_t)(gd.qpos.size() / 3) < 81, "grid interior vertices were removed");
    }

    // ================= NON-SPHERE fixture 2: the welded cube (feature corners) ====================
    {
        const scene::MeshGeometry cube = scene::CubeGeometry();
        const std::span<const scene::Vertex> cv(cube.verts.data(), cube.verts.size());
        const vg::LodGenMesh cq = vg::QuantizeWeld(
            cv, std::span<const uint32_t>(cube.indices.data(), cube.indices.size()));
        check((uint32_t)(cq.qpos.size() / 3) == 8 && (uint32_t)(cq.indices.size() / 3) == 12,
              "cube welds to 8 corners / 12 triangles");
        vg::DecimateStats cs{};
        const vg::LodGenMesh cd = vg::DecimateMesh(cq, 6, &cs);
        const uint32_t cTris = (uint32_t)(cd.indices.size() / 3);
        std::printf("cube: 12 -> %u tris collapses=%u digest=0x%016llx maxQErr=%llu\n",
                    cTris, cs.collapses, (unsigned long long)vg::LodGenDigest(cd),
                    (unsigned long long)cs.maxQuadricError);
        check(cTris == kPinCubeLodTris, "cube decimated triangle count pinned (honest guard result)");
        check(vg::LodGenDigest(cd) == kPinCubeDigest, "cube digest pinned");
        const LatticeCheck c = CheckLattice(cd);
        check(c.idxOk && c.nonDegenerate, "cube result valid (indices in range, no degenerates)");
        check(c.closed && c.euler == 2, "cube result still a CLOSED manifold (V-E+F==2)");
    }

    if (g_fail == 0) { std::printf("lod_gen_test OK\n"); return 0; }
    std::printf("lod_gen_test: %d failures\n", g_fail);
    return 1;
}
