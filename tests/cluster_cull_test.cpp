// Slice DT — Virtual-Geometry Slice 2: GPU per-cluster frustum cull -> indirect cluster draw. Pure CPU
// (header-only, no device, no backend symbols). Mirrors engine/render/cluster_cull.h, the SAME
// BuildClusterInstances + CullClusterInstances the --cluster-cull-shot (Vulkan) / --cluster-cull (Metal)
// showcases use. Namespace hf::render::vg.
//
// What this test PINS (the contracts the GPU compute cluster_cull.comp.hlsl must match byte-for-byte):
//   * BuildClusterInstances ORDER: M instances x K clusters -> M*K records, instance-major / cluster-minor;
//     each record's triOffset/triCount == the meshlet's; instanceIndex == the owning instance.
//   * CONSERVATIVE WORLD RADIUS: each world radius == localRadius * uniform-scale (>= the local radius for
//     scale>=1), i.e. InstanceWorldSphere applied — never under-bounds (the cull never drops a visible
//     cluster).
//   * CullClusterInstances all-survive (frustum contains all) -> M*K commands in source order;
//     none-survive (frustum behind everything) -> 0; half-cut -> EXACTLY the frustum.h::SphereOutside
//     subset IN SOURCE ORDER (independent brute-force reference).
//   * MdiCommand fields: indexCount == triCount*3, firstIndex == triOffset*3, instanceCount == 1,
//     vertexOffset == 0, firstInstance == the SOURCE cluster-instance index.
//   * DETERMINISM: two builds + two culls byte-identical.
//   * CONSERVATIVE (straddling KEPT): a cluster whose sphere straddles a plane is kept (not culled).
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/cluster_cull.h"
#include "render/frustum.h"
#include "render/gpu_cull.h"
#include "render/meshlet.h"
#include "scene/mesh.h"
#include "scene/vertex.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vg = hf::render::vg;
namespace fr = hf::render::frustum;
namespace mdi = hf::render::mdi;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A deterministic small instance grid of column-major model matrices (translate + uniform scale).
static std::vector<math::Mat4> MakeInstances(int n, float spacing, float scale) {
    using math::Mat4; using math::Vec3;
    std::vector<Mat4> out;
    const float half = 0.5f * (float)(n - 1) * spacing;
    for (int gx = 0; gx < n; ++gx) {
        float x = (float)gx * spacing - half;
        out.push_back(Mat4::Translate({x, 0.0f, -6.0f}) * Mat4::Scale({scale, scale, scale}));
    }
    return out;
}

// Independent brute-force survivor reference (NOT via cluster_cull.h): walk source order, keep
// !SphereOutside. Returns the kept SOURCE indices, in order.
static std::vector<uint32_t> RefSurvivors(std::span<const vg::ClusterInstance> cis, const fr::Frustum& f) {
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < (uint32_t)cis.size(); ++i)
        if (!fr::SphereOutside(f, cis[i].worldCenter, cis[i].worldRadius)) out.push_back(i);
    return out;
}

int main() {
    HF_TEST_MAIN_INIT();
    using math::Mat4; using math::Vec3;

    // --- The shared DS decomposition (the SAME mesh the showcase uses). ---
    scene::MeshGeometry geo = scene::SphereGeometry(48, 32);
    vg::MeshletSet ms = vg::BuildMeshlets(geo.verts, geo.indices);
    const uint32_t K = (uint32_t)ms.meshlets.size();
    check(K > 1, "Sphere(48,32) decomposes into >1 cluster");

    // ================= BuildClusterInstances: order + slice + conservative world radius =================
    {
        const int N = 4;
        const float scale = 1.5f;
        std::vector<Mat4> models = MakeInstances(N, 3.0f, scale);
        std::vector<vg::ClusterInstance> cis = vg::BuildClusterInstances(models, ms);

        check(cis.size() == (size_t)N * K, "BuildClusterInstances: M*K records");

        bool orderOk = true, sliceOk = true, radiusOk = true, instOk = true;
        for (uint32_t i = 0; i < (uint32_t)N; ++i) {
            for (uint32_t k = 0; k < K; ++k) {
                const vg::ClusterInstance& ci = cis[(size_t)i * K + k];
                const vg::Meshlet& m = ms.meshlets[k];
                if (ci.triOffset != m.triOffset || ci.triCount != m.triCount) sliceOk = false;
                if (ci.instanceIndex != i) instOk = false;
                // Reference world sphere from InstanceWorldSphere directly.
                Vec3 wc; float wr;
                render::gpu_cull::InstanceWorldSphere(models[i].m, m.boundCenter, m.boundRadius, wc, wr);
                if (std::fabs(ci.worldCenter.x - wc.x) > 1e-6f ||
                    std::fabs(ci.worldCenter.y - wc.y) > 1e-6f ||
                    std::fabs(ci.worldCenter.z - wc.z) > 1e-6f ||
                    std::fabs(ci.worldRadius - wr) > 1e-6f) orderOk = false;
                // Conservative: the world radius is the local radius scaled by the uniform scale (>= local
                // for scale >= 1), never under-bounding.
                if (ci.worldRadius + 1e-5f < m.boundRadius * scale) radiusOk = false;
            }
        }
        check(sliceOk, "BuildClusterInstances: triOffset/triCount == meshlet slice");
        check(instOk, "BuildClusterInstances: instanceIndex == owning instance (instance-major order)");
        check(orderOk, "BuildClusterInstances: worldCenter/Radius == InstanceWorldSphere");
        check(radiusOk, "BuildClusterInstances: world radius conservative (>= localRadius*scale)");
    }

    // ================= CullClusterInstances: all-survive / none / half-cut == frustum.h subset =========
    {
        const int N = 4;
        std::vector<Mat4> models = MakeInstances(N, 3.0f, 1.0f);
        std::vector<vg::ClusterInstance> cis = vg::BuildClusterInstances(models, ms);

        // --- All-survive: a wide frustum that contains the whole grid. The grid sits at z=-6 across x in
        // [-4.5,4.5]; a wide FOV camera at the origin looking down -Z contains it all. ---
        {
            Mat4 view = Mat4::LookAt({0, 0, 4}, {0, 0, -6}, {0, 1, 0});
            Mat4 proj = Mat4::Perspective(1.7453293f /*100deg*/, 1.7777f, 0.1f, 100.0f);
            fr::Frustum f = fr::FromViewProj(proj * view);
            std::vector<mdi::MdiCommand> cmds = vg::CullClusterInstances(cis, f);
            uint32_t cnt = vg::SurvivorClusterCount(cis, f);
            std::vector<uint32_t> ref = RefSurvivors(cis, f);
            check(cmds.size() == (size_t)N * K, "CullClusterInstances: all-survive -> M*K commands");
            check(cnt == (uint32_t)N * K, "SurvivorClusterCount: all-survive -> M*K");
            check(ref.size() == cmds.size(), "all-survive: brute-force ref count matches");
            // firstInstance carries the SOURCE index in source order (== identity here).
            bool srcOrder = true;
            for (uint32_t j = 0; j < (uint32_t)cmds.size(); ++j)
                if (cmds[j].firstInstance != ref[j]) srcOrder = false;
            check(srcOrder, "all-survive: commands in SOURCE order (firstInstance == source index)");
        }

        // --- None-survive: a frustum looking AWAY from the grid (the grid is fully behind the near plane).
        {
            Mat4 view = Mat4::LookAt({0, 0, 4}, {0, 0, 20}, {0, 1, 0});  // looking +Z, grid at -6 is behind
            Mat4 proj = Mat4::Perspective(0.7853982f /*45deg*/, 1.7777f, 0.1f, 100.0f);
            fr::Frustum f = fr::FromViewProj(proj * view);
            std::vector<mdi::MdiCommand> cmds = vg::CullClusterInstances(cis, f);
            check(cmds.empty(), "CullClusterInstances: none-survive -> 0 commands");
            check(vg::SurvivorClusterCount(cis, f) == 0, "SurvivorClusterCount: none-survive -> 0");
        }

        // --- Half-cut: a NARROW frustum that sees only the central instances; the wings fall outside. The
        // survivor SET + ORDER must equal the independent frustum.h brute-force reference. ---
        {
            Mat4 view = Mat4::LookAt({0, 0, 2}, {0, 0, -6}, {0, 1, 0});
            Mat4 proj = Mat4::Perspective(0.5235988f /*30deg*/, 1.7777f, 0.5f, 50.0f);
            fr::Frustum f = fr::FromViewProj(proj * view);
            std::vector<mdi::MdiCommand> cmds = vg::CullClusterInstances(cis, f);
            std::vector<uint32_t> ref = RefSurvivors(cis, f);
            check(cmds.size() == ref.size(), "half-cut: command count == frustum.h subset count");
            check(cmds.size() > 0 && cmds.size() < cis.size(),
                  "half-cut: a STRICT non-empty subset survives (a real cull)");
            bool match = (cmds.size() == ref.size());
            for (uint32_t j = 0; match && j < (uint32_t)cmds.size(); ++j) {
                const vg::ClusterInstance& src = cis[ref[j]];
                if (cmds[j].firstInstance != ref[j]) match = false;
                if (cmds[j].indexCount != src.triCount * 3u) match = false;
                if (cmds[j].firstIndex != src.triOffset * 3u) match = false;
                if (cmds[j].instanceCount != 1u || cmds[j].vertexOffset != 0u) match = false;
            }
            check(match, "half-cut: commands == frustum.h subset IN SOURCE ORDER, MdiCommand fields exact");

            // --- DETERMINISM: a second build + cull is byte-identical. ---
            std::vector<vg::ClusterInstance> cis2 = vg::BuildClusterInstances(models, ms);
            std::vector<mdi::MdiCommand> cmds2 = vg::CullClusterInstances(cis2, f);
            bool det = (cis.size() == cis2.size()) && (cmds.size() == cmds2.size()) &&
                       (cis.empty() || std::memcmp(cis.data(), cis2.data(),
                                          cis.size() * sizeof(vg::ClusterInstance)) == 0) &&
                       (cmds.empty() || std::memcmp(cmds.data(), cmds2.data(),
                                          cmds.size() * sizeof(mdi::MdiCommand)) == 0);
            check(det, "determinism: two builds + two culls BYTE-IDENTICAL");
        }
    }

    // ================= MdiCommand field exactness over a known single cluster ==========================
    {
        std::vector<Mat4> models = {Mat4::Translate({0, 0, -6})};  // one instance at the origin row
        std::vector<vg::ClusterInstance> cis = vg::BuildClusterInstances(models, ms);
        check(cis.size() == K, "single instance -> K cluster-instances");
        // A frustum containing all -> every command's fields derive from its meshlet exactly.
        Mat4 view = Mat4::LookAt({0, 0, 4}, {0, 0, -6}, {0, 1, 0});
        Mat4 proj = Mat4::Perspective(1.7453293f, 1.7777f, 0.1f, 100.0f);
        fr::Frustum f = fr::FromViewProj(proj * view);
        std::vector<mdi::MdiCommand> cmds = vg::CullClusterInstances(cis, f);
        bool fieldsOk = (cmds.size() == K);
        for (uint32_t j = 0; fieldsOk && j < K; ++j) {
            const vg::Meshlet& m = ms.meshlets[j];
            if (cmds[j].indexCount != m.triCount * 3u) fieldsOk = false;
            if (cmds[j].firstIndex != m.triOffset * 3u) fieldsOk = false;
            if (cmds[j].firstInstance != j) fieldsOk = false;
        }
        check(fieldsOk, "MdiCommand: indexCount==triCount*3, firstIndex==triOffset*3, firstInstance==source");
    }

    // ================= CONSERVATIVE: a cluster whose sphere STRADDLES a plane is KEPT ===================
    {
        // One cluster-instance whose world sphere straddles the near plane (center just behind near, big
        // radius reaching across). Build it directly so we control the geometry.
        vg::ClusterInstance ci{};
        ci.triOffset = 7; ci.triCount = 11;
        ci.worldCenter = Vec3{0, 0, -0.05f};  // just behind a near plane at z ~ -0.1
        ci.worldRadius = 1.0f;                // straddles
        ci.instanceIndex = 0;
        std::vector<vg::ClusterInstance> one = {ci};
        Mat4 view = Mat4::LookAt({0, 0, 1}, {0, 0, -6}, {0, 1, 0});
        Mat4 proj = Mat4::Perspective(1.0471976f, 1.7777f, 0.1f, 50.0f);
        fr::Frustum f = fr::FromViewProj(proj * view);
        bool outside = fr::SphereOutside(f, ci.worldCenter, ci.worldRadius);
        std::vector<mdi::MdiCommand> cmds = vg::CullClusterInstances(one, f);
        // The straddling sphere must NOT be fully outside, so it is kept.
        check(!outside, "straddling sphere is NOT fully outside (conservative predicate)");
        check(cmds.size() == 1 && cmds[0].firstInstance == 0u, "straddling cluster KEPT (conservative)");
    }

    if (g_fail == 0) { std::printf("cluster_cull_test OK\n"); return 0; }
    std::printf("cluster_cull_test: %d failures\n", g_fail);
    return 1;
}
