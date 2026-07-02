// Slice SC3 — SPONZA THROUGH THE VIRTUAL-GEOMETRY STACK: the pure-CPU composition header
// engine/render/sc3_stack.h (meshlet decomposition -> per-cluster frustum cull -> auto-QEM LOD over
// REAL glTF content), driven end-to-end through the device-free asset::LoadGltfSceneCpu seam.
//
// TWO TIERS (the asset-gated discipline):
//   * ALWAYS-ON: the two biggest COMMITTED assets (assets/models/DamagedHelmet.glb — one dense
//     closed-ish mesh — and assets/models/CesiumMilkTruck.glb — a multi-mesh multi-instance node
//     hierarchy) run the FULL pipeline: at-scale completeness (every triangle exactly once, per
//     mesh), decomposition determinism (two runs, identical digests), cull sanity (an enclosing
//     camera keeps a nonzero subset with ZERO soundness violations; a camera pointed AWAY culls
//     everything), and LOD validity (monotone non-increasing tri counts, zero invalid output
//     triangles, non-decreasing conservative errors, deterministic digests).
//   * SPONZA-GATED: the REAL at-scale proof over the FETCHED (gitignored) Khronos PBR Sponza —
//     103 meshes / 262k heterogeneous triangles — from the SC1 hero camera. The asset is never
//     committed, so when it is absent this tier SKIPS GRACEFULLY with a clear message (exit 0; the
//     suite has no committed asset of hero scale, so a hard requirement would break clean clones).
//     The full Sponza proof also runs unconditionally in the --sc3-stack-shot showcase goldens.
//
// Pure C++ (hf_core: gltf_loader.cpp carries the cgltf implementation), no RHI/device, ASan-eligible.
#include "asset/gltf_loader.h"
#include "math/math.h"
#include "render/frustum.h"
#include "render/sc3_stack.h"
#include "scene/vertex.h"

#include <cstdio>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vg = hf::render::vg;
namespace fr = hf::render::frustum;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Adapt a device-free CpuScene to the SC3 input views (non-owning; `scene` must outlive them).
static void Adapt(const asset::CpuScene& scene,
                  std::vector<vg::Sc3Mesh>& meshes, std::vector<vg::Sc3Instance>& instances) {
    meshes.clear(); instances.clear();
    meshes.reserve(scene.meshes.size());
    for (const auto& m : scene.meshes)
        meshes.push_back({std::span<const scene::Vertex>(m.verts.data(), m.verts.size()),
                          std::span<const uint32_t>(m.indices.data(), m.indices.size())});
    instances.reserve(scene.instances.size());
    for (const auto& inst : scene.instances)
        instances.push_back({inst.meshIndex, inst.world});
}

// Run the full SC3 pipeline over one loaded scene and verify every at-scale invariant. `label` tags
// failures; `topN` is the LOD breadth. Returns the decomposition digest (for cross-run pins by the
// caller if wanted).
static uint64_t VerifyScene(const asset::CpuScene& scene, const char* label, uint32_t topN) {
    char buf[192];
    std::vector<vg::Sc3Mesh> meshes;
    std::vector<vg::Sc3Instance> instances;
    Adapt(scene, meshes, instances);

    std::snprintf(buf, sizeof(buf), "%s: loaded at least one mesh + instance", label);
    check(!meshes.empty() && !instances.empty(), buf);

    // --- 1. Decomposition: completeness + validity on EVERY mesh; determinism across two runs. ---
    vg::Sc3Decomposition dec = vg::Sc3DecomposeAll(meshes);
    vg::Sc3Decomposition dec2 = vg::Sc3DecomposeAll(meshes);
    std::snprintf(buf, sizeof(buf), "%s: decomposition COMPLETE on all %u meshes",
                  label, (uint32_t)meshes.size());
    check(dec.incompleteMeshes == 0, buf);
    std::snprintf(buf, sizeof(buf), "%s: no invalid (out-of-range-index) meshes", label);
    check(dec.invalidMeshes == 0, buf);
    std::snprintf(buf, sizeof(buf), "%s: every cluster <= kMaxTrisPerCluster", label);
    check(dec.maxTrisPerCluster <= vg::kMaxTrisPerCluster && dec.maxTrisPerCluster > 0, buf);
    std::snprintf(buf, sizeof(buf), "%s: two decompositions IDENTICAL digests", label);
    check(dec.digest == dec2.digest && dec.digest != 0, buf);
    {
        // Cluster count sanity: Sum over meshes of ceil(T/128).
        uint32_t expect = 0;
        for (const vg::Sc3Mesh& m : meshes) {
            const uint32_t T = (uint32_t)(m.indices.size() / 3);
            expect += (T + vg::kMaxTrisPerCluster - 1) / vg::kMaxTrisPerCluster;
        }
        std::snprintf(buf, sizeof(buf), "%s: totalClusters == Sum(ceil(T/128))", label);
        check(dec.totalClusters == expect, buf);
    }
    std::printf("[%s] meshes=%u totalTris=%llu clusters=%u maxTris=%u degenerate=%llu "
                "boundaryEdges=%llu nonManifoldEdges=%llu digest=0x%016llx\n",
                label, (uint32_t)meshes.size(), (unsigned long long)dec.totalTris,
                dec.totalClusters, dec.maxTrisPerCluster,
                (unsigned long long)dec.degenerateTris, (unsigned long long)dec.boundaryEdges,
                (unsigned long long)dec.nonManifoldEdges, (unsigned long long)dec.digest);

    // --- 2. Cull sanity + SOUNDNESS from an interior-ish enclosing camera. ---
    const math::Vec3 c{0.5f * (scene.bbMin[0] + scene.bbMax[0]),
                       0.5f * (scene.bbMin[1] + scene.bbMax[1]),
                       0.5f * (scene.bbMin[2] + scene.bbMax[2])};
    const math::Vec3 ext{scene.bbMax[0] - scene.bbMin[0], scene.bbMax[1] - scene.bbMin[1],
                         scene.bbMax[2] - scene.bbMin[2]};
    const float diag = math::length(ext) > 0.0f ? math::length(ext) : 1.0f;
    const math::Vec3 eye = c + math::normalize(math::Vec3{1.0f, 0.6f, 1.0f}) * (0.9f * diag);
    {
        math::Mat4 view = math::Mat4::LookAt(eye, c, {0, 1, 0});
        math::Mat4 proj = math::Mat4::Perspective(1.0f, 16.0f / 9.0f, 0.05f, 20.0f * diag);
        fr::Frustum f = fr::FromViewProj(proj * view);
        vg::Sc3CullResult cull = vg::Sc3BuildAndCull(dec, instances, f);
        std::snprintf(buf, sizeof(buf), "%s: enclosing camera keeps a nonzero survivor set", label);
        check(cull.survivorCount > 0 && cull.survivorCount <= cull.total, buf);
        std::snprintf(buf, sizeof(buf), "%s: culled + survivors == total", label);
        check(cull.culledCount + cull.survivorCount == cull.total, buf);
        // Survivor MdiCommands reference their source records verbatim.
        bool cmdsOk = true;
        for (const auto& cmd : cull.survivors) {
            if (cmd.firstInstance >= cull.total) { cmdsOk = false; break; }
            const vg::ClusterInstance& ci = cull.records[cmd.firstInstance];
            if (cmd.indexCount != ci.triCount * 3u || cmd.firstIndex != ci.triOffset * 3u ||
                cmd.instanceCount != 1u || cmd.vertexOffset != 0u) { cmdsOk = false; break; }
        }
        std::snprintf(buf, sizeof(buf), "%s: survivor MdiCommands mirror their records", label);
        check(cmdsOk, buf);
        vg::Sc3Soundness s = vg::Sc3CheckCullSoundness(meshes, dec, instances, cull, f);
        std::snprintf(buf, sizeof(buf), "%s: cull soundness — ZERO witness violations", label);
        check(s.violations == 0, buf);
        std::snprintf(buf, sizeof(buf), "%s: the enclosing camera sees at least one witness", label);
        check(s.witnessClusters > 0, buf);
        std::printf("[%s] enclosing cull: total=%u survivors=%u culled=%u witnesses=%llu "
                    "violations=%llu\n", label, cull.total, cull.survivorCount, cull.culledCount,
                    (unsigned long long)s.witnessClusters, (unsigned long long)s.violations);
    }
    // A camera FAR away pointed AWAY from the scene culls every cluster (everything far behind
    // the near plane; the 10x-diag offset dwarfs any cluster bounding radius).
    {
        const math::Vec3 farEye = c + math::normalize(math::Vec3{1.0f, 0.6f, 1.0f}) * (10.0f * diag);
        const math::Vec3 awayAt = farEye + (farEye - c);  // look away from the scene
        math::Mat4 view = math::Mat4::LookAt(farEye, awayAt, {0, 1, 0});
        math::Mat4 proj = math::Mat4::Perspective(1.0f, 16.0f / 9.0f, 0.05f, 20.0f * diag);
        fr::Frustum f = fr::FromViewProj(proj * view);
        vg::Sc3CullResult cull = vg::Sc3BuildAndCull(dec, instances, f);
        std::snprintf(buf, sizeof(buf), "%s: a look-away camera culls EVERY cluster", label);
        check(cull.survivorCount == 0 && cull.culledCount == cull.total, buf);
    }

    // --- 3. Auto-LOD on the top-N meshes: validity + monotonicity + determinism; honest hit/miss. ---
    {
        std::vector<vg::Sc3LodEntry> lods = vg::Sc3BuildTopLods(meshes, topN);
        std::vector<vg::Sc3LodEntry> lods2 = vg::Sc3BuildTopLods(meshes, topN);
        std::snprintf(buf, sizeof(buf), "%s: LOD ran on min(topN, meshCount) meshes", label);
        check(lods.size() == std::min<size_t>(topN, meshes.size()), buf);
        bool detOk = (lods.size() == lods2.size());
        bool monoOk = true, validOk = true, errOk = true, weldOk = true;
        for (size_t i = 0; i < lods.size(); ++i) {
            const auto& e = lods[i];
            if (detOk && (e.digest != lods2[i].digest || e.triCount != lods2[i].triCount))
                detOk = false;
            if (!(e.triCount[1] <= e.triCount[0] && e.triCount[2] <= e.triCount[1])) monoOk = false;
            if (e.invalidTris != 0) validOk = false;
            if (!(e.geometricError[1] >= 0.0f && e.geometricError[2] >= e.geometricError[1]))
                errOk = false;
            if (e.triCount[0] + e.weldDropped != e.rawTris) weldOk = false;
            std::printf("[%s] lod[%zu] mesh=%u raw=%u welded=%u -> %u (hit50=%d) -> %u (hit25=%d) "
                        "err=%.6f/%.6f\n", label, i, e.meshIndex, e.rawTris, e.triCount[0],
                        e.triCount[1], (int)e.hit50, e.triCount[2], (int)e.hit25,
                        e.geometricError[1], e.geometricError[2]);
        }
        std::snprintf(buf, sizeof(buf), "%s: LOD tri counts monotone non-increasing", label);
        check(monoOk, buf);
        std::snprintf(buf, sizeof(buf), "%s: LOD output has ZERO invalid triangles", label);
        check(validOk, buf);
        std::snprintf(buf, sizeof(buf), "%s: LOD conservative errors non-decreasing", label);
        check(errOk, buf);
        std::snprintf(buf, sizeof(buf), "%s: welded + weldDropped == raw tris", label);
        check(weldOk, buf);
        std::snprintf(buf, sizeof(buf), "%s: two LOD builds IDENTICAL (digests + counts)", label);
        check(detOk, buf);
        // hit50/hit25 are HONEST reports, not assertions: boundary-preserving QEM v1 legally stalls
        // above target on boundary-dominated open meshes (a real finding, printed above).
    }
    return dec.digest;
}

int main() {
    HF_TEST_MAIN_INIT();

    // --- ALWAYS-ON tier: the committed reference assets through the full pipeline. ---
    try {
        asset::CpuScene helmet = asset::LoadGltfSceneCpu(HF_SC3_HELMET_GLB);
        VerifyScene(helmet, "DamagedHelmet", 1);
    } catch (const std::exception& e) {
        std::printf("FAIL: DamagedHelmet load threw: %s\n", e.what());
        ++g_fail;
    }
    try {
        asset::CpuScene truck = asset::LoadGltfSceneCpu(HF_SC3_TRUCK_GLB);
        VerifyScene(truck, "CesiumMilkTruck", 2);
        // The truck's node hierarchy re-references the wheels mesh: dedup means FEWER unique
        // meshes than instances (the multi-instance path the cull's instance-major records need).
        check(truck.instances.size() > truck.meshes.size(),
              "CesiumMilkTruck: instances > unique meshes (shared-mesh dedup held)");
    } catch (const std::exception& e) {
        std::printf("FAIL: CesiumMilkTruck load threw: %s\n", e.what());
        ++g_fail;
    }

    // --- SPONZA-GATED tier: the REAL at-scale proof (skips gracefully when not fetched). ---
    std::FILE* probe = std::fopen(HF_SC3_SPONZA_GLTF, "rb");
    if (!probe) {
        std::printf("SKIP: Sponza not found at '%s' — the at-scale tier needs the fetched asset "
                    "(run assets/reference/fetch_reference_assets.ps1 -Sponza). The committed-asset "
                    "tier above still ran the full pipeline.\n", HF_SC3_SPONZA_GLTF);
    } else {
        std::fclose(probe);
        try {
            asset::CpuScene sponza = asset::LoadGltfSceneCpu(HF_SC3_SPONZA_GLTF);
            check(sponza.meshes.size() >= 50, "Sponza: at least 50 unique meshes (hero scale)");
            uint64_t tris = 0;
            for (const auto& m : sponza.meshes) tris += m.indices.size() / 3;
            check(tris > 100000, "Sponza: more than 100k triangles (hero scale)");
            VerifyScene(sponza, "Sponza", 10);

            // The SC1 HERO CAMERA cull (the pinned deterministic view the showcase renders):
            // interior, so a real subset must be culled AND a real subset must survive.
            std::vector<vg::Sc3Mesh> meshes;
            std::vector<vg::Sc3Instance> instances;
            Adapt(sponza, meshes, instances);
            vg::Sc3Decomposition dec = vg::Sc3DecomposeAll(meshes);
            math::Mat4 view = math::Mat4::LookAt({8.2f, 1.7f, 0.0f}, {-9.5f, 3.4f, 0.0f}, {0, 1, 0});
            math::Mat4 proj = math::Mat4::Perspective(1.22173048f, 1280.0f / 720.0f, 0.1f, 100.0f);
            fr::Frustum hero = fr::FromViewProj(proj * view);
            vg::Sc3CullResult cull = vg::Sc3BuildAndCull(dec, instances, hero);
            check(cull.survivorCount > 0 && cull.survivorCount < cull.total,
                  "Sponza: the SC1 hero camera culls a STRICT nonzero subset");
            vg::Sc3Soundness s = vg::Sc3CheckCullSoundness(meshes, dec, instances, cull, hero);
            check(s.violations == 0, "Sponza: hero-camera cull soundness — ZERO violations");
            check(s.witnessClusters > 0, "Sponza: hero camera sees witnesses");
            std::printf("[Sponza] hero cull: total=%u survivors=%u culled=%u witnesses=%llu\n",
                        cull.total, cull.survivorCount, cull.culledCount,
                        (unsigned long long)s.witnessClusters);
        } catch (const std::exception& e) {
            std::printf("FAIL: Sponza load threw: %s\n", e.what());
            ++g_fail;
        }
    }

    if (g_fail == 0) { std::printf("sc3_stack_test OK\n"); return 0; }
    std::printf("sc3_stack_test: %d failures\n", g_fail);
    return 1;
}
