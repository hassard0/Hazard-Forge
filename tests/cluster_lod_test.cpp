// Slice DV — Virtual-Geometry Slice 4: discrete cluster-LOD selection by projected SCREEN-SPACE ERROR.
// Pure CPU (header-only, no device, no backend symbols). Mirrors engine/render/cluster_lod.h, the SAME
// BuildLodMeshes + SelectLod + ProjectionScaleForScreenError the --cluster-lod-shot (Vulkan) /
// --cluster-lod (Metal) showcases use, and the SAME SelectLod (squared form) shaders/cluster_lod_select.comp
// copies VERBATIM. Namespace hf::render::vg.
//
// What this test PINS:
//   * BuildLodMeshes: 3 LODs, each a valid cluster range into the combined MeshletSet; LOD0 error 0;
//     coarser LODs larger error + fewer clusters; the combined ranges cover every cluster contiguously +
//     exactly; the combined index buffer covers all LODs' triangles.
//   * SelectLod: near instance -> LOD0; far -> coarser; forceLod0 / errorScale==0 -> LOD0 always; distance
//     monotonicity (farther never finer); threshold boundaries (squared form) select correctly; clamp [0,2].
//   * ProjectionScaleForScreenError == screenH/(2 tan(fovY/2)).
//   * DETERMINISM: two builds + two selects byte-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/cluster_lod.h"
#include "render/meshlet.h"
#include "scene/mesh.h"
#include "scene/vertex.h"
#include "math/math.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vg = hf::render::vg;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// The three pre-baked tessellations (finest -> coarsest), matching the showcase.
static const std::array<std::pair<uint32_t, uint32_t>, vg::kNumLods> kTess = {{
    {48, 32},  // LOD0 (full detail)
    {24, 16},  // LOD1
    {12, 8},   // LOD2 (coarse)
}};

static vg::LodMeshes BuildShowcaseLods() {
    std::array<scene::MeshGeometry, vg::kNumLods> geos;
    for (uint32_t n = 0; n < vg::kNumLods; ++n)
        geos[n] = scene::SphereGeometry(kTess[n].first, kTess[n].second);
    return vg::BuildLodMeshes(geos, kTess);
}

int main() {
    HF_TEST_MAIN_INIT();
    using math::Mat4; using math::Vec3;

    vg::LodMeshes lm = BuildShowcaseLods();
    std::array<float, vg::kNumLods> errs{};
    for (uint32_t n = 0; n < vg::kNumLods; ++n) errs[n] = lm.lods[n].geometricError;

    // ================= BuildLodMeshes: ranges, errors, contiguity, coverage =================
    {
        check(lm.lods[0].geometricError == 0.0f, "LOD0 geometricError == 0 (full detail)");
        check(lm.lods[1].geometricError > lm.lods[0].geometricError,
              "LOD1 error > LOD0 error");
        check(lm.lods[2].geometricError > lm.lods[1].geometricError,
              "LOD2 error > LOD1 error (coarser = larger error)");
        check(lm.lods[0].clusterCount > lm.lods[1].clusterCount,
              "LOD0 has more clusters than LOD1");
        check(lm.lods[1].clusterCount > lm.lods[2].clusterCount,
              "LOD1 has more clusters than LOD2 (coarser = fewer clusters)");

        // Contiguous, exact-cover cluster ranges over the combined MeshletSet.
        uint32_t expectFirst = 0;
        uint32_t totalClusters = 0;
        bool contiguous = true;
        for (uint32_t n = 0; n < vg::kNumLods; ++n) {
            if (lm.lods[n].firstCluster != expectFirst) contiguous = false;
            if (lm.lods[n].clusterCount == 0) contiguous = false;
            expectFirst += lm.lods[n].clusterCount;
            totalClusters += lm.lods[n].clusterCount;
        }
        check(contiguous, "LOD cluster ranges are contiguous + non-empty (firstCluster chains)");
        check(totalClusters == (uint32_t)lm.combined.meshlets.size(),
              "LOD ranges cover EVERY combined cluster exactly once");

        // The combined index buffer covers all LODs' triangles: each LOD's index count == 3*sum(triCount)
        // over its clusters, and triOffsets index into the combined buffer in range.
        bool slicesInRange = true;
        uint32_t coveredTris = 0;
        for (const vg::Meshlet& m : lm.combined.meshlets) {
            if ((size_t)(m.triOffset + m.triCount) * 3 > lm.combined.indices.size()) slicesInRange = false;
            coveredTris += m.triCount;
        }
        check(slicesInRange, "every combined cluster's triOffset/triCount lies within the combined indices");
        check(coveredTris * 3u == (uint32_t)lm.combined.indices.size(),
              "combined clusters cover the combined index buffer exactly (Sum(triCount)*3 == |indices|)");

        // Every combined index references a valid vertex in the shared vertex array.
        bool idxOk = true;
        for (uint32_t idx : lm.combined.indices)
            if (idx >= lm.verts.size()) idxOk = false;
        check(idxOk, "every combined index references a valid shared vertex");
    }

    // ================= ProjectionScaleForScreenError ==================================================
    {
        const float fovY = 1.0471976f;  // 60 deg
        const int   H = 720;
        float ps = vg::ProjectionScaleForScreenError(fovY, H);
        float ref = (float)H / (2.0f * std::tan(fovY * 0.5f));
        check(ps == ref, "ProjectionScaleForScreenError == screenH/(2 tan(fovY/2)) (bit-exact)");
        check(ps > 0.0f, "projection scale positive");
    }

    // ================= SelectLod: near->LOD0, far->coarser, forceLod0, monotonicity, clamp ============
    {
        const float fovY = 1.0471976f;  // 60 deg
        const int   H = 720;
        const float projScale = vg::ProjectionScaleForScreenError(fovY, H);
        // A camera at +Z looking down -Z (view = LookAt). Instances march away along -Z.
        Mat4 view = Mat4::LookAt({0, 0, 0}, {0, 0, -1}, {0, 1, 0});
        std::span<const float> es(errs.data(), errs.size());

        const float threshold = 1.0f;   // 1-pixel screen-error budget
        const float scale     = 1.0f;

        // NEAR instance -> LOD0 (full detail).
        uint32_t lodNear = vg::SelectLod(es, Vec3{0, 0, -2.0f}, view, projScale, threshold, scale, false);
        check(lodNear == 0u, "SelectLod near -> LOD0 (full detail)");

        // VERY FAR instance -> the coarsest LOD (LOD2).
        uint32_t lodFar = vg::SelectLod(es, Vec3{0, 0, -100000.0f}, view, projScale, threshold, scale, false);
        check(lodFar == vg::kNumLods - 1u, "SelectLod very-far -> coarsest LOD");

        // forceLod0 -> LOD0 regardless of distance (the disabled path).
        uint32_t lodForce = vg::SelectLod(es, Vec3{0, 0, -100000.0f}, view, projScale, threshold, scale, true);
        check(lodForce == 0u, "SelectLod forceLod0 -> LOD0 always");
        // errorScale == 0 -> LOD0 regardless (the other disabled-path spelling).
        uint32_t lodScale0 = vg::SelectLod(es, Vec3{0, 0, -100000.0f}, view, projScale, threshold, 0.0f, false);
        check(lodScale0 == 0u, "SelectLod errorScale==0 -> LOD0 always");

        // DISTANCE MONOTONICITY: scan distance increasing; the selected LOD must be non-decreasing (farther
        // never selects a FINER LOD).
        bool monotone = true;
        uint32_t prev = 0;
        uint32_t distinctSeen = 0;
        bool seen[vg::kNumLods] = {false, false, false};
        for (int s = 1; s <= 4000; ++s) {
            float z = -(float)s * 0.5f;
            uint32_t lod = vg::SelectLod(es, Vec3{0, 0, z}, view, projScale, threshold, scale, false);
            if (lod < prev) monotone = false;
            prev = lod;
            if (!seen[lod]) { seen[lod] = true; ++distinctSeen; }
        }
        check(monotone, "SelectLod is distance-monotonic (farther never finer)");
        check(distinctSeen == vg::kNumLods, "SelectLod spans all 3 LODs across an increasing-distance sweep");

        // THRESHOLD BOUNDARY (squared form): at distance d, LOD n is acceptable iff geometricError[n]*projScale
        // <= allowed*d, i.e. d >= geometricError[n]*projScale/allowed. Pick LOD1's exact boundary distance and
        // check the selection flips from <LOD1 just-inside to >=LOD1 just-outside.
        const float allowed = threshold * scale;
        const float dBoundary = errs[1] * projScale / allowed;  // distance where LOD1 becomes acceptable
        // Just NEARER than the boundary -> LOD1 NOT yet acceptable -> LOD0.
        uint32_t lodInside = vg::SelectLod(es, Vec3{0, 0, -(dBoundary * 0.99f)}, view, projScale,
                                           threshold, scale, false);
        check(lodInside == 0u, "SelectLod just-inside LOD1 boundary -> LOD0");
        // Just FARTHER than the boundary -> LOD1 acceptable (LOD2 not yet) -> LOD1.
        uint32_t lodOutside = vg::SelectLod(es, Vec3{0, 0, -(dBoundary * 1.01f)}, view, projScale,
                                            threshold, scale, false);
        check(lodOutside >= 1u, "SelectLod just-outside LOD1 boundary -> >= LOD1");

        // CLAMP: the result is always in [0, kNumLods-1] over an extreme sweep (already covered, assert here).
        bool clamped = true;
        for (int s = 0; s < 200; ++s) {
            float z = -std::pow(10.0f, (float)s * 0.1f - 5.0f);  // 1e-5 .. 1e15
            uint32_t lod = vg::SelectLod(es, Vec3{0, 0, z}, view, projScale, threshold, scale, false);
            if (lod >= vg::kNumLods) clamped = false;
        }
        check(clamped, "SelectLod result always clamped to [0, kNumLods-1]");

        // DETERMINISM: a second build + the same selects are byte-identical.
        vg::LodMeshes lm2 = BuildShowcaseLods();
        bool buildDet = (lm.combined.meshlets.size() == lm2.combined.meshlets.size()) &&
                        (lm.combined.indices.size() == lm2.combined.indices.size()) &&
                        (std::memcmp(lm.lods.data(), lm2.lods.data(),
                                     sizeof(vg::LodLevel) * vg::kNumLods) == 0) &&
                        (lm.combined.meshlets.empty() ||
                         std::memcmp(lm.combined.meshlets.data(), lm2.combined.meshlets.data(),
                                     lm.combined.meshlets.size() * sizeof(vg::Meshlet)) == 0) &&
                        (lm.combined.indices.empty() ||
                         std::memcmp(lm.combined.indices.data(), lm2.combined.indices.data(),
                                     lm.combined.indices.size() * sizeof(uint32_t)) == 0);
        check(buildDet, "determinism: two BuildLodMeshes BYTE-IDENTICAL");

        bool selDet = true;
        for (int s = 1; s <= 500; ++s) {
            float z = -(float)s * 0.5f;
            uint32_t a = vg::SelectLod(es, Vec3{0, 0, z}, view, projScale, threshold, scale, false);
            uint32_t b = vg::SelectLod(es, Vec3{0, 0, z}, view, projScale, threshold, scale, false);
            if (a != b) selDet = false;
        }
        check(selDet, "determinism: SelectLod two runs identical");
    }

    if (g_fail == 0) { std::printf("cluster_lod_test OK\n"); return 0; }
    std::printf("cluster_lod_test: %d failures\n", g_fail);
    return 1;
}
