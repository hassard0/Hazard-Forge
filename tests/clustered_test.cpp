// Slice AG — clustered / Forward+ lighting. Pure CPU math: cluster-grid index round-trips,
// light-sphere -> cluster assignment, and clusters/lightIndices/offsets consistency. No device,
// ASan-eligible (links hf_core). Mirrors the math the --clustered-shot showcase and
// lit_clustered.frag.hlsl use.
#include "render/clustered.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace cl = hf::render::clustered;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();
    // A standard perspective camera: 60deg fovY, 16:9, znear 0.1, zfar 80.
    const float fovY = 1.0471975512f;  // 60 deg
    const float W = 1280.0f, H = 720.0f;
    const float aspect = W / H;
    const float znear = 0.1f, zfar = 80.0f;
    Mat4 proj = Mat4::Perspective(fovY, aspect, znear, zfar);
    cl::Grid g = cl::MakeGrid(proj, znear, zfar, W, H, 16, 9, 24);

    check(g.clusterCount() == 16 * 9 * 24, "cluster count = CX*CY*CZ");

    // ---- z-slice is monotonic + exponential, spans exactly [znear, zfar]. ----
    {
        check(std::fabs(cl::SliceZ(g, 0) - znear) < 1e-4f, "SliceZ(0) == znear");
        check(std::fabs(cl::SliceZ(g, g.cz) - zfar) < 1e-3f, "SliceZ(CZ) == zfar");
        for (int k = 0; k < g.cz; ++k)
            check(cl::SliceZ(g, k) < cl::SliceZ(g, k + 1), "SliceZ strictly increasing");
        // Near slices thinner than far slices (exponential).
        float first = cl::SliceZ(g, 1) - cl::SliceZ(g, 0);
        float last = cl::SliceZ(g, g.cz) - cl::SliceZ(g, g.cz - 1);
        check(last > first * 5.0f, "exponential: far slices much deeper than near");
    }

    // ---- SliceForViewZ is the inverse of SliceZ: a depth sampled inside slice k maps back to k. ----
    {
        for (int k = 0; k < g.cz; ++k) {
            float zmid = 0.5f * (cl::SliceZ(g, k) + cl::SliceZ(g, k + 1));
            check(cl::SliceForViewZ(g, zmid) == k, "SliceForViewZ(mid of slice k) == k");
        }
        check(cl::SliceForViewZ(g, znear * 0.5f) == 0, "viewZ < znear clamps to slice 0");
        check(cl::SliceForViewZ(g, zfar * 2.0f) == g.cz - 1, "viewZ > zfar clamps to slice CZ-1");
    }

    // ---- ClusterForViewPos: a point whose AABB-containing cluster we know round-trips, and the
    //      point lies INSIDE that cluster's AABB. ----
    {
        // Center of the frustum at a mid depth -> central tile.
        Vec3 center{0.0f, 0.0f, -10.0f};   // straight ahead, 10 units in front
        int ci = cl::ClusterForViewPos(g, center);
        check(ci >= 0 && ci < g.clusterCount(), "center cluster index in range");
        // Reconstruct (cx,cy,cz) and verify the AABB contains the point.
        int cz = ci / (g.cx * g.cy);
        int rem = ci % (g.cx * g.cy);
        int cy = rem / g.cx;
        int cx = rem % g.cx;
        check(cx == g.cx / 2 || cx == g.cx / 2 - 1, "center maps to a middle column");
        check(cy == g.cy / 2 || cy == g.cy / 2 - 1, "center maps to a middle row");
        Vec3 bmin, bmax;
        cl::ClusterAABB(g, cx, cy, cz, bmin, bmax);
        check(center.x >= bmin.x - 1e-3f && center.x <= bmax.x + 1e-3f, "point x in AABB");
        check(center.y >= bmin.y - 1e-3f && center.y <= bmax.y + 1e-3f, "point y in AABB");
        check(center.z >= bmin.z - 1e-3f && center.z <= bmax.z + 1e-3f, "point z in AABB");

        // A point off to the upper-left-far maps to a DIFFERENT, plausibly-cornered cluster.
        Vec3 ul{-6.0f, 3.0f, -40.0f};
        int ui = cl::ClusterForViewPos(g, ul);
        check(ui >= 0 && ui != ci, "off-axis far point -> different cluster");
        int ucz = ui / (g.cx * g.cy);
        check(ucz > cz, "farther point -> deeper z-slice");
    }

    // ---- A behind-camera point has no cluster. ----
    check(cl::ClusterForViewPos(g, Vec3{0, 0, 5.0f}) == -1, "point behind camera -> no cluster");

    // ---- Light-sphere -> cluster assignment: a small light overlaps the cluster containing its
    //      center, and does NOT overlap a far-away cluster. ----
    {
        cl::Light L{};
        L.viewPos = {0.0f, 0.0f, -10.0f};
        L.radius = 1.5f;
        L.color = {1, 1, 1};
        L.intensity = 1.0f;

        int home = cl::ClusterForViewPos(g, L.viewPos);
        int hcz = home / (g.cx * g.cy);
        int hrem = home % (g.cx * g.cy);
        int hcy = hrem / g.cx;
        int hcx = hrem % g.cx;
        check(cl::LightOverlapsCluster(g, L, hcx, hcy, hcz), "light overlaps its home cluster");

        // A cluster on the opposite side of the screen + a far z-slice must NOT be hit.
        check(!cl::LightOverlapsCluster(g, L, 0, 0, g.cz - 1),
              "small light does not reach the far corner cluster");
        check(!cl::LightOverlapsCluster(g, L, g.cx - 1, g.cy - 1, 0),
              "small light does not reach the opposite near corner");

        // The set of clusters the light hits must include every cluster whose AABB the sphere
        // intersects. Spot-check: shift one slice over in z but same tile near the center -> the
        // 1.5-radius sphere should still reach the immediately-adjacent z slice if it's close.
        int adjacentZ = hcz + 1 < g.cz ? hcz + 1 : hcz - 1;
        Vec3 bmin, bmax;
        cl::ClusterAABB(g, hcx, hcy, adjacentZ, bmin, bmax);
        bool reaches = cl::SqDistPointAABB(L.viewPos, bmin, bmax) <= L.radius * L.radius;
        check(cl::LightOverlapsCluster(g, L, hcx, hcy, adjacentZ) == reaches,
              "adjacent-z overlap matches the direct sphere/AABB test");
    }

    // ---- A wide light hits MANY clusters; a tiny light hits FEW. Monotone in radius. ----
    {
        cl::Light small{}; small.viewPos = {0, 0, -10}; small.radius = 0.5f;
        small.color = {1,1,1}; small.intensity = 1;
        cl::Light big = small; big.radius = 8.0f;
        int nSmall = 0, nBig = 0;
        for (int z = 0; z < g.cz; ++z)
            for (int y = 0; y < g.cy; ++y)
                for (int x = 0; x < g.cx; ++x) {
                    if (cl::LightOverlapsCluster(g, small, x, y, z)) ++nSmall;
                    if (cl::LightOverlapsCluster(g, big, x, y, z)) ++nBig;
                }
        check(nSmall >= 1, "small light hits at least its home cluster");
        check(nBig > nSmall, "bigger radius hits strictly more clusters");
    }

    // ---- Full BuildClusters: offsets are the prefix sum, sum(count) == lightIndices.size(),
    //      and every listed light actually overlaps the cluster it's listed under. ----
    {
        std::vector<cl::Light> lights;
        // A deterministic spray of 64 lights across the frustum.
        for (int i = 0; i < 64; ++i) {
            cl::Light L{};
            float fx = ((i % 8) - 3.5f) * 1.8f;
            float fy = ((i / 8) - 3.5f) * 1.0f;
            float fz = -(6.0f + (i % 5) * 6.0f);
            L.viewPos = {fx, fy, fz};
            L.radius = 2.5f;
            L.color = {1.0f, 0.5f, 0.2f};
            L.intensity = 1.0f;
            lights.push_back(L);
        }
        cl::ClusterBuffers cb = cl::BuildClusters(g, lights);
        check((int)cb.clusters.size() == g.clusterCount(), "clusters array sized CX*CY*CZ");
        check(cb.lights.size() == lights.size(), "one GpuLight per input light");

        uint32_t sumCount = 0, expectOffset = 0;
        bool offsetsOk = true, overlapsOk = true;
        for (size_t c = 0; c < cb.clusters.size(); ++c) {
            if (cb.clusters[c].offset != expectOffset) offsetsOk = false;
            expectOffset += cb.clusters[c].count;
            sumCount += cb.clusters[c].count;
            // Decode cluster coords for the overlap re-check.
            int cz = (int)(c / (g.cx * g.cy));
            int rem = (int)(c % (g.cx * g.cy));
            int cy = rem / g.cx;
            int cx = rem % g.cx;
            for (uint32_t k = 0; k < cb.clusters[c].count; ++k) {
                uint32_t li = cb.lightIndices[cb.clusters[c].offset + k];
                if (!cl::LightOverlapsCluster(g, lights[li], cx, cy, cz)) overlapsOk = false;
            }
        }
        check(offsetsOk, "offsets == running prefix sum of counts");
        check(sumCount == (uint32_t)cb.lightIndices.size(), "sum(count) == lightIndices.size()");
        check(overlapsOk, "every listed light overlaps its cluster's AABB");
        check(sumCount > 0, "at least some lights were assigned");
    }

    if (g_fail == 0) std::printf("clustered_test: all checks passed\n");
    else std::printf("clustered_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
