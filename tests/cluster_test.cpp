// Slice CL — Clustered Light Culling (Forward+). Pure CPU math (no device, ASan-eligible, links
// hf_core). Exercises the SAME render/cluster.h assignment the --clustered-lights-shot showcase +
// shaders/cluster_assign.comp.hlsl + shaders/lit_clustered_cl.frag.hlsl mirror. Covers:
//   * SphereAABBIntersect: inside / touching face,edge,corner / outside.
//   * ClusterViewAABB: the grid tiles the frustum (adjacent clusters share a face; near < far; the
//     grid covers the screen).
//   * AssignLights correctness + the SAFETY PROPERTY: a brute-force per-cluster sphere/AABB check has
//     ZERO false-negatives vs the assignment (no contributing cluster is ever missed).
//   * Real culling: maxPerCluster < totalLights, every list a subset of all lights.
//   * Determinism: same inputs -> same lists, same order.
#include "render/cluster.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

using namespace hf::math;
namespace cl = hf::render::cluster;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    // ===================== SphereAABBIntersect: inside / touching / outside =====================
    {
        Vec3 mn{-1, -1, -1}, mx{1, 1, 1};
        // Center inside -> intersects (any positive radius).
        check(cl::SphereAABBIntersect(Vec3{0, 0, 0}, 0.1f, mn, mx), "sphere centered inside intersects");
        // Far outside, small radius -> no.
        check(!cl::SphereAABBIntersect(Vec3{5, 0, 0}, 1.0f, mn, mx), "far sphere does not intersect");
        // Just touching a FACE: center at x=2, radius exactly 1 -> closest point (1,0,0), dist==1==r.
        check(cl::SphereAABBIntersect(Vec3{2, 0, 0}, 1.0f, mn, mx), "sphere touching a face (dist==r) intersects");
        check(!cl::SphereAABBIntersect(Vec3{2, 0, 0}, 0.999f, mn, mx), "sphere just short of a face misses");
        // Touching an EDGE: center past the +x,+y edge; closest point is the edge corner line.
        // center (2,2,0): closest = (1,1,0), dist = sqrt(2). radius sqrt(2) touches, less misses.
        check(cl::SphereAABBIntersect(Vec3{2, 2, 0}, std::sqrt(2.0f) + 1e-4f, mn, mx), "sphere touching an edge intersects");
        check(!cl::SphereAABBIntersect(Vec3{2, 2, 0}, std::sqrt(2.0f) - 1e-3f, mn, mx), "sphere short of an edge misses");
        // Touching a CORNER: center (2,2,2), closest = (1,1,1), dist = sqrt(3).
        check(cl::SphereAABBIntersect(Vec3{2, 2, 2}, std::sqrt(3.0f) + 1e-4f, mn, mx), "sphere touching a corner intersects");
        check(!cl::SphereAABBIntersect(Vec3{2, 2, 2}, std::sqrt(3.0f) - 1e-3f, mn, mx), "sphere short of a corner misses");
    }

    // ===================== Cluster grid + camera =====================
    const float fovY = 1.0471975512f;  // 60 deg
    const int   SW = 1280, SH = 720;
    const float aspect = (float)SW / (float)SH;
    const float zNear = 0.5f, zFar = 90.0f;
    Mat4 proj = Mat4::Perspective(fovY, aspect, zNear, zFar);
    Mat4 invProj = proj.Inverse();
    cl::ClusterGrid g; g.dimX = 16; g.dimY = 9; g.dimZ = 24; g.zNear = zNear; g.zFar = zFar;

    check(g.clusterCount() == 16 * 9 * 24, "cluster count = dimX*dimY*dimZ");

    // ---- Exponential Z slicing: monotone, spans [zNear,zFar], near thinner than far. ----
    {
        check(std::fabs(cl::SliceZ(g, 0) - zNear) < 1e-4f, "SliceZ(0) == zNear");
        check(std::fabs(cl::SliceZ(g, g.dimZ) - zFar) < 1e-2f, "SliceZ(dimZ) == zFar");
        for (int k = 0; k < g.dimZ; ++k)
            check(cl::SliceZ(g, k) < cl::SliceZ(g, k + 1), "SliceZ strictly increasing");
        float first = cl::SliceZ(g, 1) - cl::SliceZ(g, 0);
        float last  = cl::SliceZ(g, g.dimZ) - cl::SliceZ(g, g.dimZ - 1);
        check(last > first * 5.0f, "exponential: far slices much deeper than near");
        for (int k = 0; k < g.dimZ; ++k) {
            float zmid = 0.5f * (cl::SliceZ(g, k) + cl::SliceZ(g, k + 1));
            check(cl::SliceForViewZ(g, zmid) == k, "SliceForViewZ(mid of slice k) == k");
        }
    }

    // ---- ClusterViewAABB tiling: near slice closer than far; adjacent clusters share a face; the
    //      grid's union covers the central view ray. ----
    {
        // A central cluster: its AABB straddles x=0,y=0 and is in front of the camera (z<0).
        int ccx = g.dimX / 2, ccy = g.dimY / 2, ccz = 8;
        Vec3 amn, amx;
        cl::ClusterViewAABB(g, ccx, ccy, ccz, invProj, SW, SH, amn, amx);
        check(amx.z < 0.0f && amn.z < 0.0f, "cluster AABB is in front of camera (view z < 0)");
        check(amn.z < amx.z, "AABB min.z (far) < max.z (near)");
        check(std::fabs(-amx.z - cl::SliceZ(g, ccz)) < 1e-2f, "AABB near plane == SliceZ(cz)");
        check(std::fabs(-amn.z - cl::SliceZ(g, ccz + 1)) < 1e-2f, "AABB far plane == SliceZ(cz+1)");

        // Adjacent in X tile the frustum: cluster (cx+1)'s AABB starts at/inside cluster (cx)'s right
        // edge and extends further right (conservative depth-span AABBs overlap at the shared frustum
        // face rather than meeting at an exact plane). No GAP: bmn.x <= amx.x and bmx.x > amx.x.
        Vec3 bmn, bmx;
        cl::ClusterViewAABB(g, ccx + 1, ccy, ccz, invProj, SW, SH, bmn, bmx);
        check(bmn.x <= amx.x + 1e-4f, "adjacent-X clusters tile with no gap (cx+1.min.x <= cx.max.x)");
        check(bmx.x > amx.x, "adjacent-X cluster extends further right");
        check(std::fabs(amn.z - bmn.z) < 1e-4f && std::fabs(amx.z - bmx.z) < 1e-4f,
              "adjacent-X clusters share the same z-slice bounds");

        // The near plane is physically nearer (smaller |z|) than the far plane within a slice.
        check(-amx.z < -amn.z, "near slice plane is nearer than far slice plane");

        // A deeper z-slice cluster is farther away.
        Vec3 dmn, dmx;
        cl::ClusterViewAABB(g, ccx, ccy, ccz + 4, invProj, SW, SH, dmn, dmx);
        check(-dmx.z > -amx.z, "deeper z-slice cluster is farther from the camera");

        // X extent grows with depth (perspective): far AABB is wider in x than near.
        Vec3 nearMn, nearMx, farMn, farMx;
        cl::ClusterViewAABB(g, g.dimX - 1, ccy, 1,  invProj, SW, SH, nearMn, nearMx);
        cl::ClusterViewAABB(g, g.dimX - 1, ccy, 20, invProj, SW, SH, farMn,  farMx);
        check((farMx.x - farMn.x) > (nearMx.x - nearMn.x), "perspective: far cluster wider than near");
    }

    // ===================== AssignLights: correctness + the no-false-negative SAFETY property ======
    // Camera at origin looking down -Z (view == identity rotation). Lights in WORLD space; AssignLights
    // transforms them by `view`. With view == LookAt(eye=origin, center=(0,0,-1)) the world == view.
    Mat4 view = Mat4::LookAt(Vec3{0, 0, 0}, Vec3{0, 0, -1}, Vec3{0, 1, 0});

    // A deterministic spray of point lights spread across the frustum at varied depths/radii.
    std::vector<cl::PointLight> lights;
    const int kN = 80;
    for (int i = 0; i < kN; ++i) {
        cl::PointLight L{};
        float fx = ((i % 8) - 3.5f) * 2.2f;
        float fy = ((i / 8) % 8 - 3.5f) * 1.4f;
        float fz = -(4.0f + (float)((i * 7) % 11) * 5.0f);   // 4..54 in front
        L.posWorld = {fx, fy, fz};
        L.radius   = 2.0f + (float)((i * 3) % 5) * 0.8f;       // 2.0 .. 5.2
        L.color    = {1.0f, 0.5f, 0.25f};
        L.intensity = 1.0f;
        lights.push_back(L);
    }

    std::vector<std::vector<uint32_t>> perCluster;
    cl::AssignLights(g, proj, view, SW, SH, std::span<const cl::PointLight>(lights), perCluster);
    check((int)perCluster.size() == g.clusterCount(), "perCluster sized dimX*dimY*dimZ");

    // Re-derive view-space positions for the independent brute-force re-check.
    std::vector<Vec3> vpos(lights.size());
    for (size_t li = 0; li < lights.size(); ++li) vpos[li] = MulPoint(view, lights[li].posWorld);

    // (a) Every listed light's view-sphere ACTUALLY intersects its cluster's AABB (no false POSITIVES),
    //     lists are ascending (ordered), every index valid, every list a subset of all lights.
    // (b) The SAFETY PROPERTY (no false NEGATIVES): for EVERY (cluster, light) pair, the light is in
    //     the cluster's list IFF its sphere intersects the cluster AABB. A brute-force double loop
    //     proves the assignment misses NOTHING a contributing light would touch.
    bool listedOk = true, orderedOk = true, subsetOk = true, parityOk = true;
    int assignedTotal = 0;
    size_t maxPerCluster = 0;
    for (int cz = 0; cz < g.dimZ; ++cz)
        for (int cy = 0; cy < g.dimY; ++cy)
            for (int cx = 0; cx < g.dimX; ++cx) {
                int idx = g.flatIndex(cx, cy, cz);
                Vec3 amn, amx;
                cl::ClusterViewAABB(g, cx, cy, cz, invProj, SW, SH, amn, amx);
                const std::vector<uint32_t>& list = perCluster[(size_t)idx];
                assignedTotal += (int)list.size();
                maxPerCluster = std::max(maxPerCluster, list.size());

                // membership set for this cluster (for the parity test).
                std::vector<char> inList(lights.size(), 0);
                uint32_t prev = 0; bool first = true;
                for (uint32_t li : list) {
                    if (li >= lights.size()) { subsetOk = false; continue; }
                    inList[li] = 1;
                    if (!first && li <= prev) orderedOk = false;  // strictly ascending
                    prev = li; first = false;
                    if (!cl::SphereAABBIntersect(vpos[li], lights[li].radius, amn, amx)) listedOk = false;
                }
                // Brute-force parity: list membership IFF sphere/AABB intersection.
                for (size_t li = 0; li < lights.size(); ++li) {
                    bool truth = cl::SphereAABBIntersect(vpos[li], lights[li].radius, amn, amx);
                    if (truth != (inList[li] != 0)) parityOk = false;
                }
            }
    check(listedOk, "every listed light actually overlaps its cluster (no false positives)");
    check(orderedOk, "per-cluster lists are ascending light index (ordered, matches GPU fill)");
    check(subsetOk, "every listed index is a valid light (subset of all lights)");
    check(parityOk, "SAFETY: assignment == brute-force sphere/AABB for every (cluster,light) (no false negatives)");
    check(assignedTotal > 0, "some lights were assigned");

    // ---- Each light lands in >= 1 cluster (it is in front of the camera, within the frustum). ----
    {
        std::vector<int> hitCount(lights.size(), 0);
        for (const auto& list : perCluster)
            for (uint32_t li : list) ++hitCount[li];
        bool allPlaced = true;
        for (int h : hitCount) if (h < 1) allPlaced = false;
        check(allPlaced, "every (in-frustum) light is assigned to at least one cluster");
    }

    // ---- Radius monotonicity: a tiny-radius light hits FEWER clusters than a huge-radius one. ----
    {
        std::vector<cl::PointLight> tiny = {cl::PointLight{Vec3{0, 0, -12}, 0.6f, Vec3{1, 1, 1}, 1.0f}};
        std::vector<cl::PointLight> huge = {cl::PointLight{Vec3{0, 0, -12}, 14.0f, Vec3{1, 1, 1}, 1.0f}};
        std::vector<std::vector<uint32_t>> pcTiny, pcHuge;
        cl::AssignLights(g, proj, view, SW, SH, std::span<const cl::PointLight>(tiny), pcTiny);
        cl::AssignLights(g, proj, view, SW, SH, std::span<const cl::PointLight>(huge), pcHuge);
        int nTiny = 0, nHuge = 0;
        for (const auto& l : pcTiny) nTiny += (int)l.size();
        for (const auto& l : pcHuge) nHuge += (int)l.size();
        check(nTiny >= 1, "tiny light hits at least its home cluster");
        check(nHuge > nTiny, "huge-radius light hits strictly more clusters");
    }

    // ===================== Real culling: maxPerCluster < totalLights =====================
    check((int)maxPerCluster < kN, "real culling: no cluster sees all lights (maxPerCluster < totalLights)");
    check(maxPerCluster >= 1, "at least one cluster has a light");

    // ===================== Determinism: same inputs -> identical lists + order =====================
    {
        std::vector<std::vector<uint32_t>> pc2;
        cl::AssignLights(g, proj, view, SW, SH, std::span<const cl::PointLight>(lights), pc2);
        bool same = (pc2.size() == perCluster.size());
        for (size_t c = 0; same && c < pc2.size(); ++c) {
            if (pc2[c].size() != perCluster[c].size()) { same = false; break; }
            for (size_t k = 0; k < pc2[c].size(); ++k)
                if (pc2[c][k] != perCluster[c][k]) { same = false; break; }
        }
        check(same, "determinism: re-running AssignLights yields identical lists + order");
    }

    if (g_fail == 0) std::printf("cluster_test: all checks passed (assigned total %d, maxPerCluster %zu)\n",
                                 assignedTotal, maxPerCluster);
    else std::printf("cluster_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
