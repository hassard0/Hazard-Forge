// Slice DS — Virtual-Geometry Slice 1: meshlet / cluster decomposition. Pure CPU (header-only, no
// device, no backend symbols). Mirrors engine/render/meshlet.h, the SAME BuildMeshlets the
// --meshlet-viz showcase uses. Namespace hf::render::vg.
//
// What this test PINS (the contracts the whole Nanite-style arc — DT cull, DU Hi-Z, DV LOD — builds on):
//   * PARTITION COMPLETENESS: the multiset of all (reordered) cluster triangles' index-triples ==
//     the original index multiset (every triangle covered exactly once, none invented/dropped);
//     Sum(triCount) == T; each triCount <= kMaxTrisPerCluster and only the LAST cluster may be partial.
//   * DETERMINISM: two builds memcmp the meshlets array AND the indices array bit-identical.
//   * CONSERVATIVE BOUNDS: every referenced vertex lies inside [boundMin,boundMax] AND within
//     boundRadius of boundCenter (the contract DT/DU rely on — never under-bound).
//   * DEGENERATE/EMPTY: empty -> 0; 1-tri -> 1 cluster; a non-multiple-of-3 length ignores the
//     trailing 1-2 indices (T = len/3).
//   * MortonCode10 round-trip + hashColor determinism/spread.
//   * STRESS: Cube + several Sphere tessellations -> completeness + determinism + bounds hold.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/meshlet.h"
#include "scene/mesh.h"
#include "scene/vertex.h"
#include "math/math.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vg = hf::render::vg;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A sorted-triple key so two index-triples that name the SAME triangle (same 3 vertices in the same
// winding order) compare equal regardless of which corner is listed first. The decomposition copies
// the triangle's three indices VERBATIM (no rotation), so we keep the ordered triple — but sorting the
// whole list of triples lets us compare the two multisets.
struct Tri { uint32_t a, b, c; };
static bool TriLess(const Tri& x, const Tri& y) {
    if (x.a != y.a) return x.a < y.a;
    if (x.b != y.b) return x.b < y.b;
    return x.c < y.c;
}
static std::vector<Tri> TrisOf(const std::vector<uint32_t>& idx) {
    std::vector<Tri> tris;
    tris.reserve(idx.size() / 3);
    for (size_t t = 0; t + 2 < idx.size() + 1 && (t + 3) <= idx.size(); t += 3)
        tris.push_back({idx[t], idx[t + 1], idx[t + 2]});
    return tris;
}

// Verify the full DS contract for one input mesh; `label` tags failures.
static void VerifySet(const std::vector<scene::Vertex>& verts,
                      const std::vector<uint32_t>& indices, const char* label) {
    char buf[160];
    const uint32_t T = (uint32_t)(indices.size() / 3);

    vg::MeshletSet ms = vg::BuildMeshlets(verts, indices);

    // --- Sum(triCount) == T; reordered index count == 3T. ---
    uint32_t sumTri = 0;
    for (const auto& m : ms.meshlets) sumTri += m.triCount;
    std::snprintf(buf, sizeof(buf), "%s: Sum(triCount) == T (%u)", label, T);
    check(sumTri == T, buf);
    std::snprintf(buf, sizeof(buf), "%s: indices.size() == 3T", label);
    check(ms.indices.size() == (size_t)T * 3, buf);

    // --- triCount <= kMaxTrisPerCluster; only the LAST cluster may be partial; triOffset contiguous. ---
    uint32_t expectOffset = 0;
    bool sizeOk = true, offsetOk = true, partialOk = true;
    for (size_t i = 0; i < ms.meshlets.size(); ++i) {
        const auto& m = ms.meshlets[i];
        if (m.triCount == 0 || m.triCount > vg::kMaxTrisPerCluster) sizeOk = false;
        if (m.triOffset != expectOffset) offsetOk = false;
        if (i + 1 < ms.meshlets.size() && m.triCount != vg::kMaxTrisPerCluster) partialOk = false;
        expectOffset += m.triCount;
    }
    std::snprintf(buf, sizeof(buf), "%s: every triCount in (0,128]", label); check(sizeOk, buf);
    std::snprintf(buf, sizeof(buf), "%s: triOffset is contiguous", label);    check(offsetOk, buf);
    std::snprintf(buf, sizeof(buf), "%s: only the last cluster is partial", label); check(partialOk, buf);

    // --- Partition completeness: the reordered triangle multiset == the original. ---
    std::vector<Tri> orig = TrisOf(indices);
    std::vector<Tri> reord = TrisOf(ms.indices);
    std::sort(orig.begin(), orig.end(), TriLess);
    std::sort(reord.begin(), reord.end(), TriLess);
    bool sameMultiset = (orig.size() == reord.size());
    if (sameMultiset)
        for (size_t i = 0; i < orig.size(); ++i)
            if (orig[i].a != reord[i].a || orig[i].b != reord[i].b || orig[i].c != reord[i].c) {
                sameMultiset = false; break;
            }
    std::snprintf(buf, sizeof(buf), "%s: triangle multiset preserved (every tri once)", label);
    check(sameMultiset, buf);

    // --- Conservative bounds: every referenced vertex inside the AABB AND within the radius. ---
    bool boundsOk = true;
    const float kEps = 1e-4f;
    for (const auto& m : ms.meshlets) {
        for (uint32_t t = 0; t < m.triCount; ++t)
            for (int e = 0; e < 3; ++e) {
                uint32_t idx = ms.indices[3 * (m.triOffset + t) + e];
                const scene::Vertex& v = verts[idx];
                math::Vec3 p{v.pos[0], v.pos[1], v.pos[2]};
                if (p.x < m.boundMin.x - kEps || p.x > m.boundMax.x + kEps ||
                    p.y < m.boundMin.y - kEps || p.y > m.boundMax.y + kEps ||
                    p.z < m.boundMin.z - kEps || p.z > m.boundMax.z + kEps) boundsOk = false;
                math::Vec3 d = p - m.boundCenter;
                if (math::length(d) > m.boundRadius + kEps) boundsOk = false;
            }
        // boundCenter must be the AABB midpoint.
        math::Vec3 mid = (m.boundMin + m.boundMax) * 0.5f;
        if (std::fabs(mid.x - m.boundCenter.x) > kEps || std::fabs(mid.y - m.boundCenter.y) > kEps ||
            std::fabs(mid.z - m.boundCenter.z) > kEps) boundsOk = false;
    }
    std::snprintf(buf, sizeof(buf), "%s: conservative bounds hold (verts in AABB + radius)", label);
    check(boundsOk, buf);

    // --- Determinism: a second build is byte-identical (meshlets array AND indices array). ---
    vg::MeshletSet ms2 = vg::BuildMeshlets(verts, indices);
    bool detMesh = (ms.meshlets.size() == ms2.meshlets.size()) &&
                   (ms.meshlets.empty() ||
                    std::memcmp(ms.meshlets.data(), ms2.meshlets.data(),
                                ms.meshlets.size() * sizeof(vg::Meshlet)) == 0);
    bool detIdx = (ms.indices.size() == ms2.indices.size()) &&
                  (ms.indices.empty() ||
                   std::memcmp(ms.indices.data(), ms2.indices.data(),
                               ms.indices.size() * sizeof(uint32_t)) == 0);
    std::snprintf(buf, sizeof(buf), "%s: two builds BYTE-IDENTICAL (meshlets)", label); check(detMesh, buf);
    std::snprintf(buf, sizeof(buf), "%s: two builds BYTE-IDENTICAL (indices)", label);  check(detIdx, buf);
}

int main() {
    HF_TEST_MAIN_INIT();

    // --- MortonCode10 round-trip: recover the 3 lanes from the interleaved code. ---
    {
        bool rtOk = true;
        auto compact = [](uint32_t code) {  // inverse of the 3-bit-stride spread, lane 0
            uint32_t v = code & 0x09249249u;
            v = (v | (v >> 2))  & 0x030C30C3u;
            v = (v | (v >> 4))  & 0x0300F00Fu;
            v = (v | (v >> 8))  & 0x030000FFu;
            v = (v | (v >> 16)) & 0x000003FFu;
            return v;
        };
        uint32_t samples[] = {0, 1, 2, 511, 512, 1023, 7, 1000, 333, 1022};
        for (uint32_t x : samples)
            for (uint32_t y : samples)
                for (uint32_t z : samples) {
                    uint32_t code = vg::MortonCode10(x, y, z);
                    if (compact(code) != x || compact(code >> 1) != y || compact(code >> 2) != z)
                        rtOk = false;
                }
        check(rtOk, "MortonCode10 interleave round-trips all 3 lanes");
        // Monotone-ish sanity: a pure-x ramp keeps codes strictly increasing.
        bool mono = true;
        for (uint32_t x = 1; x <= 1023; ++x)
            if (vg::MortonCode10(x, 0, 0) <= vg::MortonCode10(x - 1, 0, 0)) mono = false;
        check(mono, "MortonCode10 is monotone along a single-axis ramp");
    }

    // --- hashColor: deterministic + spread (distinct adjacent indices -> distinct colors). ---
    {
        bool det = true;
        for (uint32_t i = 0; i < 256; ++i) {
            math::Vec3 a = vg::hashColor(i), b = vg::hashColor(i);
            if (a.x != b.x || a.y != b.y || a.z != b.z) det = false;
            if (a.x < 0.0f || a.x > 1.0f || a.y < 0.0f || a.y > 1.0f || a.z < 0.0f || a.z > 1.0f)
                det = false;
        }
        check(det, "hashColor is deterministic + in [0,1]^3");
        // Spread: over 64 adjacent indices, every consecutive pair differs visibly (no flat runs).
        int distinctAdjacent = 0;
        for (uint32_t i = 1; i < 64; ++i) {
            math::Vec3 a = vg::hashColor(i - 1), b = vg::hashColor(i);
            if (std::fabs(a.x - b.x) + std::fabs(a.y - b.y) + std::fabs(a.z - b.z) > 0.1f)
                ++distinctAdjacent;
        }
        check(distinctAdjacent == 63, "hashColor: all 63 adjacent index pairs are visibly distinct");
    }

    // --- Degenerate/empty no-ops. ---
    {
        std::vector<scene::Vertex> noVerts;
        std::vector<uint32_t> noIdx;
        vg::MeshletSet empty = vg::BuildMeshlets(noVerts, noIdx);
        check(empty.meshlets.empty() && empty.indices.empty(), "empty input -> 0 meshlets, 0 indices");

        // 1-triangle mesh -> exactly 1 cluster, triCount 1.
        std::vector<scene::Vertex> tri(3);
        tri[0].pos[0] = 0; tri[0].pos[1] = 0; tri[0].pos[2] = 0;
        tri[1].pos[0] = 1; tri[1].pos[1] = 0; tri[1].pos[2] = 0;
        tri[2].pos[0] = 0; tri[2].pos[1] = 1; tri[2].pos[2] = 0;
        std::vector<uint32_t> triIdx = {0, 1, 2};
        vg::MeshletSet one = vg::BuildMeshlets(tri, triIdx);
        check(one.meshlets.size() == 1 && one.meshlets[0].triCount == 1 && one.indices.size() == 3,
              "1-triangle mesh -> exactly 1 cluster (triCount 1)");

        // Non-multiple-of-3 length: the trailing 1-2 indices are ignored (T = len/3).
        std::vector<uint32_t> trailing = {0, 1, 2, 0, 1};  // 5 indices -> T = 1
        vg::MeshletSet trail = vg::BuildMeshlets(tri, trailing);
        check(trail.meshlets.size() == 1 && trail.meshlets[0].triCount == 1 && trail.indices.size() == 3,
              "non-multiple-of-3 length: trailing indices ignored (T = len/3)");
    }

    // --- Stress: Cube + several Sphere tessellations (incl. one > 128 tris so multiple clusters). ---
    {
        scene::MeshGeometry cube = scene::CubeGeometry();
        VerifySet(cube.verts, cube.indices, "Cube");

        // (8,6) sphere: a small mesh (likely a single partial cluster). (24,16) default. (48,32) the
        // showcase tessellation (thousands of tris -> many clusters, exercises the partial last).
        const uint32_t segs[] = {8, 24, 48};
        const uint32_t rings[] = {6, 16, 32};
        for (int k = 0; k < 3; ++k) {
            scene::MeshGeometry s = scene::SphereGeometry(segs[k], rings[k]);
            char lbl[48];
            std::snprintf(lbl, sizeof(lbl), "Sphere(%u,%u)", segs[k], rings[k]);
            VerifySet(s.verts, s.indices, lbl);
        }

        // The showcase mesh specifically: assert it produces MORE THAN ONE cluster (the interesting
        // multi-cluster path) and a sensible count.
        scene::MeshGeometry showcase = scene::SphereGeometry(48, 32);
        vg::MeshletSet ms = vg::BuildMeshlets(showcase.verts, showcase.indices);
        uint32_t T = (uint32_t)(showcase.indices.size() / 3);
        uint32_t expectClusters = (T + vg::kMaxTrisPerCluster - 1) / vg::kMaxTrisPerCluster;
        check(ms.meshlets.size() > 1, "Sphere(48,32) decomposes into >1 cluster");
        check(ms.meshlets.size() == expectClusters, "Sphere(48,32) cluster count == ceil(T/128)");
    }

    if (g_fail == 0) { std::printf("meshlet_test OK\n"); return 0; }
    std::printf("meshlet_test: %d failures\n", g_fail);
    return 1;
}
