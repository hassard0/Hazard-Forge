// Slice DU — Virtual-Geometry Slice 3: GPU per-CLUSTER Hi-Z OCCLUSION cull. Pure CPU (header-only, no
// device, no backend symbols). Mirrors engine/render/cluster_cull.h::CullClusterInstancesHiZ — the SAME
// frustum-cull (DT) PLUS the Hi-Z occlusion test (over each cluster's bounding-sphere AABB) the
// --cluster-hiz-shot (Vulkan) / --cluster-hiz (Metal) showcases + shaders/cluster_hiz_cull.comp.hlsl run.
// Namespace hf::render::vg.
//
// What this test PINS (the contracts the GPU compute cluster_hiz_cull.comp.hlsl must match byte-for-byte):
//   * occlusionEnabled=false == CullClusterInstances (DT) BYTE-IDENTICAL (the disabled-path guarantee:
//     same survivors, same order, same MdiCommand fields).
//   * A cluster whose bounding-sphere AABB is FULLY BEHIND the Hi-Z depth -> DROPPED; a cluster in FRONT
//     of the depth -> KEPT. Conservative: a cluster straddling the near plane or partly off-screen -> KEPT
//     (IsOccluded returns false).
//   * occlusion survivor count <= frustum-only count; the dropped set is a SUBSET of the frustum-surviving
//     set (occlusion only removes frustum survivors).
//   * DETERMINISM: two culls byte-identical. MdiCommand fields preserved (indexCount==triCount*3,
//     firstIndex==triOffset*3, instanceCount==1, vertexOffset==0, firstInstance==source index).
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/cluster_cull.h"
#include "render/frustum.h"
#include "render/hiz.h"
#include "render/mdi.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vg  = hf::render::vg;
namespace fr  = hf::render::frustum;
namespace hz  = hf::render::hiz;
namespace mdi = hf::render::mdi;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Build a Hi-Z from a flat depth buffer where a near OCCLUDER covers the FULL screen at depth `occZ`
// (everything else stays at the far plane 1.0). With a screen-filling occluder, ANY cluster whose nearest
// NDC z is strictly > occZ (fully behind the occluder) and whose screen rect is fully on-screen is occluded.
static void MakeFullScreenOccluder(int w, int h, float occZ, std::vector<hz::HiZMip>& mips) {
    std::vector<float> depth((size_t)w * (size_t)h, occZ);
    hz::BuildHiZ(depth.data(), w, h, mips);
}

int main() {
    HF_TEST_MAIN_INIT();
    using math::Mat4; using math::Vec3;

    // A camera looking down -Z at the origin (Vulkan-clip view-proj feeds both the frustum + the AABB
    // projection in IsOccluded). Reverse-free [0,1] depth: 0 near .. 1 far.
    const int W = 256, H = 256;
    const float aspect = (float)W / (float)H;
    Mat4 view = Mat4::LookAt({0, 0, 5}, {0, 0, -1}, {0, 1, 0});
    Mat4 proj = Mat4::Perspective(1.0471976f /*60deg*/, aspect, 0.5f, 100.0f);
    Mat4 vp = proj * view;
    fr::Frustum f = fr::FromViewProj(vp);

    // A helper that returns the NDC z of a world point (for choosing occluder depths between clusters).
    auto ndcZ = [&](const Vec3& p) { float w = 0; return math::MulPointDivide(vp, p, w).z; };

    // ================= occlusionEnabled=false == CullClusterInstances (DT) BYTE-IDENTICAL ================
    {
        // A handful of cluster-instances spread across the view (some in, some out of frustum).
        std::vector<vg::ClusterInstance> cis;
        auto add = [&](Vec3 c, float r, uint32_t triOff, uint32_t triCnt, uint32_t inst) {
            vg::ClusterInstance ci{};
            ci.triOffset = triOff; ci.triCount = triCnt;
            ci.worldCenter = c; ci.worldRadius = r; ci.instanceIndex = inst;
            cis.push_back(ci);
        };
        add({ 0.0f, 0.0f, -2.0f}, 0.5f, 0,  10, 0);
        add({ 1.5f, 0.0f, -3.0f}, 0.4f, 10, 12, 1);
        add({-1.0f, 0.5f, -4.0f}, 0.6f, 22, 8,  2);
        add({ 50.0f, 0.0f, -3.0f}, 0.4f, 30, 5, 3);   // far to the side -> outside frustum
        add({ 0.0f, 0.0f, 30.0f}, 0.5f, 35, 7, 4);    // behind the camera -> outside frustum

        std::span<const vg::ClusterInstance> span(cis.data(), cis.size());

        // An empty Hi-Z (no mips) — IsOccluded returns false for everything anyway, but occlusionEnabled=false
        // must bypass the Hi-Z ENTIRELY and equal DT regardless of the mips. Use a real (full-screen near)
        // Hi-Z to make the bypass meaningful.
        std::vector<hz::HiZMip> mips;
        MakeFullScreenOccluder(W, H, 0.01f, mips);  // a VERY near occluder (would occlude all if enabled)
        std::span<const hz::HiZMip> mipSpan(mips.data(), mips.size());

        std::vector<mdi::MdiCommand> dt   = vg::CullClusterInstances(span, f);
        std::vector<mdi::MdiCommand> off  = vg::CullClusterInstancesHiZ(span, f, vp, W, H, mipSpan, /*occ=*/false);
        bool identical = (dt.size() == off.size()) &&
                         (dt.empty() || std::memcmp(dt.data(), off.data(),
                                          dt.size() * sizeof(mdi::MdiCommand)) == 0);
        check(identical, "occlusionEnabled=false == CullClusterInstances (DT) BYTE-IDENTICAL");
        check(vg::SurvivorClusterCountHiZ(span, f, vp, W, H, mipSpan, false) ==
              vg::SurvivorClusterCount(span, f), "SurvivorClusterCountHiZ(off) == SurvivorClusterCount");
        check(dt.size() > 0, "disabled-path: a non-empty frustum-survivor set exists");
    }

    // ================= fully-behind -> DROPPED; in-front -> KEPT (conservative occlusion) ================
    {
        // Two clusters in frustum, on-axis, fully on-screen:
        //   FRONT cluster at z=-1.5 (near), BACK cluster at z=-6 (far). Place a full-screen occluder at a
        //   depth BETWEEN them, so the BACK cluster is fully behind it (occluded) and the FRONT is in front
        //   (kept). Both small enough that their screen rect stays on-screen.
        Vec3 frontC{0.0f, 0.0f, -1.5f}; float frontR = 0.25f;
        Vec3 backC {0.0f, 0.0f, -6.0f}; float backR  = 0.25f;

        vg::ClusterInstance front{}; front.triOffset = 0; front.triCount = 10;
        front.worldCenter = frontC; front.worldRadius = frontR; front.instanceIndex = 0;
        vg::ClusterInstance back{};  back.triOffset = 10; back.triCount = 12;
        back.worldCenter = backC; back.worldRadius = backR; back.instanceIndex = 1;
        std::vector<vg::ClusterInstance> cis = {front, back};
        std::span<const vg::ClusterInstance> span(cis.data(), cis.size());

        // The occluder depth must be > the FRONT cluster's farthest corner AND < the BACK cluster's nearest
        // corner so exactly the back cluster is occluded. The front AABB nearest corner z, back AABB nearest:
        float frontNear = ndcZ({frontC.x, frontC.y, frontC.z + frontR});  // nearest corner (largest z toward cam? )
        float backNear  = ndcZ({backC.x,  backC.y,  backC.z  + backR});
        // In [0,1] depth, nearer == smaller z. The front cluster's whole AABB has SMALLER z than the back's.
        // Choose occZ strictly greater than the front AABB's FARTHEST z and strictly less than the back's
        // NEAREST z. Farthest of front == z at frontC.z - frontR; nearest of back == z at backC.z + backR.
        float frontFar = ndcZ({frontC.x, frontC.y, frontC.z - frontR});
        (void)frontNear;
        check(frontFar < backNear, "front cluster fully nearer than back cluster (scene sanity)");
        float occZ = 0.5f * (frontFar + backNear);

        std::vector<hz::HiZMip> mips;
        MakeFullScreenOccluder(W, H, occZ, mips);
        std::span<const hz::HiZMip> mipSpan(mips.data(), mips.size());

        // Direct IsOccluded checks over the bounding-sphere AABBs.
        Vec3 fmn, fmx, bmn, bmx;
        vg::ClusterAabb(front, fmn, fmx);
        vg::ClusterAabb(back,  bmn, bmx);
        check(!hz::IsOccluded(fmn, fmx, vp, W, H, mipSpan), "front cluster KEPT (in front of occluder)");
        check( hz::IsOccluded(bmn, bmx, vp, W, H, mipSpan), "back cluster OCCLUDED (fully behind occluder)");

        // Through the cull: occlusion-on drops the back cluster only.
        std::vector<mdi::MdiCommand> on  = vg::CullClusterInstancesHiZ(span, f, vp, W, H, mipSpan, true);
        std::vector<mdi::MdiCommand> off = vg::CullClusterInstancesHiZ(span, f, vp, W, H, mipSpan, false);
        check(off.size() == 2, "frustum-only: both clusters survive frustum");
        check(on.size() == 1, "occlusion-on: exactly the back cluster dropped");
        check(on.size() == 1 && on[0].firstInstance == 0u, "the SURVIVOR is the FRONT cluster (source idx 0)");
        // MdiCommand fields of the survivor preserved.
        check(on.size() == 1 && on[0].indexCount == front.triCount * 3u &&
              on[0].firstIndex == front.triOffset * 3u && on[0].instanceCount == 1u &&
              on[0].vertexOffset == 0u, "survivor MdiCommand fields preserved");

        // Survivor count <= frustum-only; dropped set subset of frustum-survivors.
        check(on.size() <= off.size(), "occlusion survivor count <= frustum-only count");
        // every survivor index appears in the frustum-only set.
        bool subset = true;
        for (const auto& c : on) {
            bool found = false;
            for (const auto& d : off) if (d.firstInstance == c.firstInstance) found = true;
            if (!found) subset = false;
        }
        check(subset, "occlusion survivors are a SUBSET of frustum survivors");

        // DETERMINISM.
        std::vector<mdi::MdiCommand> on2 = vg::CullClusterInstancesHiZ(span, f, vp, W, H, mipSpan, true);
        bool det = (on.size() == on2.size()) &&
                   (on.empty() || std::memcmp(on.data(), on2.data(),
                                     on.size() * sizeof(mdi::MdiCommand)) == 0);
        check(det, "determinism: two occlusion culls BYTE-IDENTICAL");
    }

    // ================= CONSERVATIVE: near-plane-straddle / off-screen -> KEPT (never false-cull) =========
    {
        // A full-screen VERY near occluder (occZ tiny) — would occlude anything fully behind it. But a
        // cluster STRADDLING the near plane (a corner with clip w<=0) or whose rect leaves the screen must
        // be KEPT.
        std::vector<hz::HiZMip> mips;
        MakeFullScreenOccluder(W, H, 0.02f, mips);
        std::span<const hz::HiZMip> mipSpan(mips.data(), mips.size());

        // Straddling the near plane: center just in front of the camera, radius reaching behind it.
        vg::ClusterInstance straddle{};
        straddle.worldCenter = {0.0f, 0.0f, 4.6f};  // near plane is at z = 5 - 0.5 = 4.5
        straddle.worldRadius = 1.0f;                // AABB spans z in [3.6, 5.6] -> crosses the near plane
        Vec3 smn, smx; vg::ClusterAabb(straddle, smn, smx);
        check(!hz::IsOccluded(smn, smx, vp, W, H, mipSpan),
              "near-plane-straddling cluster KEPT (conservative, clip w<=0)");

        // Off-screen: a cluster whose rect runs off the right edge -> KEPT.
        vg::ClusterInstance offscreen{};
        offscreen.worldCenter = {6.0f, 0.0f, -3.0f};  // far to the right
        offscreen.worldRadius = 1.0f;
        Vec3 omn, omx; vg::ClusterAabb(offscreen, omn, omx);
        check(!hz::IsOccluded(omn, omx, vp, W, H, mipSpan),
              "off-screen cluster KEPT (conservative, partially off-screen)");

        // Empty Hi-Z -> nothing to test against -> KEPT.
        std::span<const hz::HiZMip> emptySpan;
        vg::ClusterInstance any{};
        any.worldCenter = {0.0f, 0.0f, -6.0f}; any.worldRadius = 0.25f;
        Vec3 amn, amx; vg::ClusterAabb(any, amn, amx);
        check(!hz::IsOccluded(amn, amx, vp, W, H, emptySpan), "empty Hi-Z -> KEPT (nothing to test against)");
    }

    if (g_fail == 0) { std::printf("cluster_hiz_test OK\n"); return 0; }
    std::printf("cluster_hiz_test: %d failures\n", g_fail);
    return 1;
}
