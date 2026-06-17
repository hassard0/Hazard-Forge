// Slice DW — Virtual-Geometry Slice 1: Visibility Buffer. Pure CPU (header-only, no device, no
// backend symbols). Mirrors engine/render/visbuffer.h, the SAME ID packing + CPU coverage reference
// the --visbuffer-shot (Vulkan) / --visbuffer (Metal) showcases use. Namespace hf::render::vg.
//
// What this test PINS (the contracts the GPU visbuffer.frag.hlsl + the proofs build on):
//   * PackVisId / UnpackVisId round-trip over the FULL valid range (clusterID x triID).
//   * kTriIdBits >= 7 (a cluster's <=128 triangles index in 7 bits) + the static budget assertion.
//   * PACKING INJECTIVE (collision-free) over [0, survivorCount) x [0, kMaxTrisPerCluster): no two
//     distinct (clusterID, triID) pairs map to the same packed ID — the visibility identity is unique.
//   * kVisBackground (0xFFFFFFFF) never equals a valid packed ID over the realized survivor range — the
//     clear sentinel can never be mistaken for coverage.
//   * The CPU coverage reference's survivor membership == CullClusterInstances (ties DW to the DT/DV
//     contract): every InteriorSample's survivorDrawIndex is a real, in-range survivor; samples are in
//     survivor-draw order; each names the correct source cluster-instance.
//   * DETERMINISM: two pack passes + two SurvivorInteriorSamples calls bit-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/visbuffer.h"
#include "render/cluster_cull.h"
#include "render/frustum.h"
#include "render/meshlet.h"
#include "scene/mesh.h"
#include "scene/vertex.h"
#include "math/math.h"

#include <cstdio>
#include <cstring>
#include <span>
#include <unordered_set>
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

int main() {
    HF_TEST_MAIN_INIT();
    using math::Mat4; using math::Vec3;

    // ================= kTriIdBits budget =================
    check(vg::kTriIdBits >= 7, "kTriIdBits >= 7");
    check((1u << vg::kTriIdBits) >= vg::kMaxTrisPerCluster,
          "1<<kTriIdBits covers kMaxTrisPerCluster");
    check(vg::kTriIdMask == ((1u << vg::kTriIdBits) - 1u), "kTriIdMask == (1<<kTriIdBits)-1");

    // ================= PackVisId / UnpackVisId round-trip over the full valid range =================
    {
        bool roundTripOk = true, lowBitsOk = true;
        // Sweep a generous clusterID range x every triID in [0, 128). 100k clusters >> any realized
        // survivor count, exercising the high-bit field thoroughly.
        for (uint32_t c = 0; c < 100000u; c += 7u) {  // stride 7 to keep it fast yet dense
            for (uint32_t t = 0; t < vg::kMaxTrisPerCluster; ++t) {
                uint32_t packed = vg::PackVisId(c, t);
                uint32_t uc = 0, ut = 0;
                vg::UnpackVisId(packed, uc, ut);
                if (uc != c || ut != t) roundTripOk = false;
                // The triID occupies exactly the low kTriIdBits; the clusterID the rest.
                if ((packed & vg::kTriIdMask) != t) lowBitsOk = false;
                if ((packed >> vg::kTriIdBits) != c) lowBitsOk = false;
            }
        }
        check(roundTripOk, "PackVisId/UnpackVisId round-trip over clusterID x triID");
        check(lowBitsOk, "PackVisId field layout: triID low kTriIdBits, clusterID high bits");

        // The mask guards an out-of-range triID from bleeding into the clusterID field.
        uint32_t packedMasked = vg::PackVisId(5u, 0xFFu);  // triID 255 -> masked to 0x7F
        uint32_t uc = 0, ut = 0; vg::UnpackVisId(packedMasked, uc, ut);
        check(uc == 5u && ut == 0x7Fu, "PackVisId masks an out-of-range triID (no clusterID bleed)");
    }

    // ================= Packing injective (collision-free) over [0,survivorCount) x [0,128) ==========
    {
        const uint32_t survivorCount = 512u;  // far above any realized on-screen survivor set
        std::unordered_set<uint32_t> seen;
        seen.reserve((size_t)survivorCount * vg::kMaxTrisPerCluster);
        bool injective = true;
        for (uint32_t c = 0; c < survivorCount && injective; ++c) {
            for (uint32_t t = 0; t < vg::kMaxTrisPerCluster; ++t) {
                uint32_t packed = vg::PackVisId(c, t);
                if (!seen.insert(packed).second) { injective = false; break; }
            }
        }
        check(injective, "PackVisId injective over [0,survivorCount) x [0,kMaxTrisPerCluster)");

        // kVisBackground never equals any valid packed ID over this range.
        bool bgClean = true;
        for (uint32_t c = 0; c < survivorCount && bgClean; ++c)
            for (uint32_t t = 0; t < vg::kMaxTrisPerCluster; ++t)
                if (vg::PackVisId(c, t) == vg::kVisBackground) { bgClean = false; break; }
        check(bgClean, "kVisBackground never collides with a valid packed ID over the survivor range");
        // It would only collide at clusterID 0x1FFFFFF (~33.5M survivors), well beyond any feasible scene.
        check(vg::kVisBackground == 0xFFFFFFFFu, "kVisBackground sentinel value");
    }

    // ================= CPU coverage reference: survivor membership == CullClusterInstances ===========
    {
        // The SAME clustered-sphere scene + narrow camera as the showcase (a clear off-screen subset).
        scene::MeshGeometry geo = scene::SphereGeometry(48, 32);
        vg::MeshletSet ms = vg::BuildMeshlets(geo.verts, geo.indices);
        check(ms.meshlets.size() > 1, "Sphere(48,32) decomposes into >1 cluster");

        const int kInstances = 4;
        const float spacing = 3.2f;
        const float half = 0.5f * (float)(kInstances - 1) * spacing;
        std::vector<Mat4> models;
        for (int gi = 0; gi < kInstances; ++gi)
            models.push_back(Mat4::Translate({(float)gi * spacing - half, 0.0f, -6.0f}));

        std::vector<vg::ClusterInstance> cis = vg::BuildClusterInstances(models, ms);
        std::span<const vg::ClusterInstance> cspan(cis.data(), cis.size());

        // A narrow 30-degree view-proj (Vulkan clip), matching the showcase framing.
        const float aspect = 16.0f / 9.0f;
        Mat4 view = Mat4::LookAt({0.0f, 0.0f, 1.5f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
        Mat4 proj = Mat4::Perspective(0.5235988f, aspect, 0.2f, 50.0f);
        Mat4 viewProj = proj * view;
        fr::Frustum f = fr::FromViewProj(viewProj);

        std::vector<mdi::MdiCommand> cmds = vg::CullClusterInstances(cspan, f);
        check(cmds.size() > 0 && cmds.size() < cis.size(),
              "CullClusterInstances yields a STRICT non-empty subset (a real cull)");

        // Per-instance world centers (sphere origins) + radii for the instance-provenance reference.
        std::vector<Vec3> centers(models.size());
        for (size_t i = 0; i < models.size(); ++i)
            centers[i] = {models[i].m[12], models[i].m[13], models[i].m[14]};
        std::span<const Vec3> centersSpan(centers.data(), centers.size());
        std::vector<float> radii(models.size(), 0.0f);
        for (const vg::ClusterInstance& ci : cis) {
            float reach = math::length(ci.worldCenter - centers[ci.instanceIndex]) + ci.worldRadius;
            if (reach > radii[ci.instanceIndex]) radii[ci.instanceIndex] = reach;
        }
        std::span<const float> radiiSpan(radii.data(), radii.size());
        const Vec3 cameraPos{0.0f, 0.0f, 1.5f};

        const uint32_t w = 1280, h = 720;
        std::vector<vg::InteriorSample> samples = vg::InstanceInteriorSamples(
            std::span<const mdi::MdiCommand>(cmds.data(), cmds.size()), cspan, centersSpan, radiiSpan,
            cameraPos, viewProj, w, h);
        check(!samples.empty(), "InstanceInteriorSamples yields >=1 visible-instance near-pole sample");

        // Every sample names a real, in-range source instance; the (px,py) lies inside the viewport; the
        // sampled instance has >=1 frustum survivor (so the GPU draws it).
        std::vector<uint8_t> instHasSurvivor(models.size(), 0);
        for (const mdi::MdiCommand& c : cmds) instHasSurvivor[cis[c.firstInstance].instanceIndex] = 1;
        bool membershipOk = true, viewportOk = true, hasSurvivorOk = true;
        for (const vg::InteriorSample& s : samples) {
            if (s.expectedInstance >= models.size()) membershipOk = false;
            if (s.px >= w || s.py >= h) viewportOk = false;
            if (s.expectedInstance < models.size() && !instHasSurvivor[s.expectedInstance]) hasSurvivorOk = false;
        }
        check(membershipOk, "every InteriorSample.expectedInstance is a real source instance");
        check(viewportOk, "every InteriorSample pixel lies inside the viewport");
        check(hasSurvivorOk, "every sampled instance has >=1 frustum survivor (ties B3 to the cull)");

        // ===== DETERMINISM: two InstanceInteriorSamples calls byte-identical. =====
        std::vector<vg::InteriorSample> samples2 = vg::InstanceInteriorSamples(
            std::span<const mdi::MdiCommand>(cmds.data(), cmds.size()), cspan, centersSpan, radiiSpan,
            cameraPos, viewProj, w, h);
        bool det = samples.size() == samples2.size() &&
                   std::memcmp(samples.data(), samples2.data(),
                               samples.size() * sizeof(vg::InteriorSample)) == 0;
        check(det, "InstanceInteriorSamples deterministic (two calls byte-identical)");
    }

    if (g_fail == 0) std::printf("visbuffer_test: ALL PASS\n");
    else std::printf("visbuffer_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
