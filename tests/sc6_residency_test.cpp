// Slice SC6 — TEXTURE RESIDENCY: VT page feedback driving a streaming budget on real content
// (engine/render/sc6_residency.h — the pure-CPU bridge between the proven vt.h page-table math and
// the scene/streaming.h budget+hysteresis discipline, applied to texture pages).
//
// TWO TIERS (the sc3_stack asset-gated discipline):
//   * ALWAYS-ON (synthetic, asset-free): StepResidency SEMANTICS on a tiny hand-built page space
//     (deterministic (mip,tex,y,x) load ordering, the load budget, K-frame hysteresis, the
//     oldest-un-requested-first eviction order + globalId tie-break, the pool cap), then the
//     CANONICAL textured-ground-plane fly-over: the pinned {peakResident, loads, evicts,
//     convergedAt, traceDigest} artifacts, two-run determinism, the BUDGET-IS-LOAD-BEARING contrast
//     (an unthrottled run loads more pages in one frame than the whole budget; the budgeted run
//     still CONVERGES), the HYSTERESIS thrash contrast (a flickering view: K>=2 loads each page
//     once; the K=1 baseline reloads it every other frame), and the pool cap invariant.
//   * SPONZA-GATED: the REAL-content proof — the FETCHED Khronos PBR Sponza's 69-texture set mapped
//     into page space (each texture its own 4-mip pyramid), per-frame object-space VT feedback from
//     the SC1 hero camera DOLLIED down the nave, the full budgeted residency trace with pinned
//     {peak resident, total loads, total evictions, trace digest} + two-run determinism. SKIPS
//     GRACEFULLY when the fetched (gitignored) asset is absent.
//
// The synthetic tier (and sc6_residency.h itself) is standalone-clang-compilable:
//   clang++ -std=c++20 -I engine -I tests tests/sc6_residency_test.cpp
// (the Sponza tier needs hf_core's cgltf/gltf_loader and is compiled only when HF_SC6_SPONZA_GLTF
// is defined — the CMake build defines it; the bare clang probe does not.)
#include "render/sc6_residency.h"

#include <cstdio>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

#if defined(HF_SC6_SPONZA_GLTF)
#include "asset/gltf_loader.h"
#include "cgltf/cgltf.h"   // declarations only; the implementation lives in hf_core (gltf_loader.cpp)
#endif

using namespace hf;
namespace sc6 = hf::render::sc6;
namespace vt = hf::render::vt;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// ---------------------------------------------------------------------------------------------------
// 1. StepResidency SEMANTICS on a tiny page space: 1 texture, 2 mips, vpps0=2 -> mip0 has 4 pages
//    (global ids 0..3, row-major), mip1 has 1 page (global id 4). 5 pages total.
// ---------------------------------------------------------------------------------------------------
static void TestStepSemantics() {
    vt::VtTexture tiny;
    tiny.mipLevels = 2; tiny.pageSize = 16; tiny.virtualPagesPerSideMip0 = 2;
    const sc6::Sc6PageSpace space = sc6::MakeSc6PageSpace(1, tiny);
    check(space.totalPages == 5, "tiny space: 4 + 1 = 5 pages");

    // --- Load ordering + budget: 4 requests in SCRAMBLED order, budget 2 -> the two FINEST-mip
    // lowest-(y,x) pages load first; the rest wait. ---
    {
        sc6::Sc6ResidencyConfig cfg;
        cfg.loadBudgetPerFrame = 2; cfg.evictBudgetPerFrame = 8;
        cfg.hysteresisFrames = 3; cfg.poolCapacity = 8;
        sc6::Sc6ResidencyState st;
        std::vector<sc6::Sc6PageRequest> req = {
            {0, 1, 0, 0},   // mip1 -> global id 4 (coarsest: LAST priority)
            {0, 0, 1, 1},   // mip0 (1,1) -> id 3
            {0, 0, 0, 0},   // mip0 (0,0) -> id 0
            {0, 0, 1, 0},   // mip0 (1,0) -> id 1
        };
        sc6::Sc6StepResult r0 = sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(req), space, cfg);
        check(r0.requested == 4, "step: 4 deduped requests");
        check(r0.loaded.size() == 2 && r0.loaded[0] == 0 && r0.loaded[1] == 1,
              "step: budget 2 loads exactly the (mip asc, y, x) head: ids {0, 1}");
        check(r0.missingAfter == 2, "step: 2 requested pages still missing after the budget");
        sc6::Sc6StepResult r1 = sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(req), space, cfg);
        check(r1.loaded.size() == 2 && r1.loaded[0] == 3 && r1.loaded[1] == 4,
              "step: frame 2 loads the tail {3, 4} (mip0 before mip1)");
        check(r1.missingAfter == 0 && r1.residentCount == 4, "step: all requested resident by frame 2");
    }

    // --- Hysteresis: a page loaded then un-requested is evicted EXACTLY after K un-requested
    // frames, not before. ---
    {
        sc6::Sc6ResidencyConfig cfg;
        cfg.loadBudgetPerFrame = 8; cfg.evictBudgetPerFrame = 8;
        cfg.hysteresisFrames = 3; cfg.poolCapacity = 8;
        sc6::Sc6ResidencyState st;
        std::vector<sc6::Sc6PageRequest> req = {{0, 0, 0, 0}};
        std::vector<sc6::Sc6PageRequest> none;
        sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(req), space, cfg);   // frame 0: load
        sc6::Sc6StepResult f1 = sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(none), space, cfg);
        sc6::Sc6StepResult f2 = sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(none), space, cfg);
        check(f1.evicted.empty() && f2.evicted.empty(), "hysteresis: frames 1-2 un-requested (< K=3) keep the page");
        sc6::Sc6StepResult f3 = sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(none), space, cfg);
        check(f3.evicted.size() == 1 && f3.evicted[0] == 0, "hysteresis: frame 3 (== K) evicts the page");
    }

    // --- Eviction order: oldest-un-requested first, globalId tie-break; the evict budget throttles. ---
    {
        sc6::Sc6ResidencyConfig cfg;
        cfg.loadBudgetPerFrame = 8; cfg.evictBudgetPerFrame = 1;
        cfg.hysteresisFrames = 2; cfg.poolCapacity = 8;
        sc6::Sc6ResidencyState st;
        std::vector<sc6::Sc6PageRequest> reqA = {{0, 0, 0, 0}};                 // id 0
        std::vector<sc6::Sc6PageRequest> reqAB = {{0, 0, 0, 0}, {0, 0, 1, 0}};  // ids 0, 1
        std::vector<sc6::Sc6PageRequest> none;
        sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(reqAB), space, cfg);  // f0: load 0+1
        sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(reqA), space, cfg);   // f1: touch 0 only
        sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(none), space, cfg);   // f2: 1 is 2 stale
        // At f2: page 1 last requested f0 (delta 2 == K) -> evictable; page 0 last f1 (delta 1) -> kept.
        // But the budget is 1, so exactly one eviction and it must be the OLDEST (page 1).
        sc6::Sc6StepResult f3 = sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(none), space, cfg);
        // f2 already evicted page 1 (delta hit K at f2). f3 evicts page 0 (delta 2 at f3).
        check(f3.evicted.size() == 1 && f3.evicted[0] == 0, "evict order: the older page went first, then this one");
        // Same-age tie-break: load 2 pages together, let both expire, budget 1 -> LOWER global id first.
        sc6::Sc6ResidencyState st2;
        sc6::StepResidency(st2, std::span<const sc6::Sc6PageRequest>(reqAB), space, cfg); // f0: load 0+1
        sc6::StepResidency(st2, std::span<const sc6::Sc6PageRequest>(none), space, cfg);  // f1
        sc6::Sc6StepResult g2 = sc6::StepResidency(st2, std::span<const sc6::Sc6PageRequest>(none), space, cfg);
        check(g2.evicted.size() == 1 && g2.evicted[0] == 0, "evict tie-break: equal age -> ascending global id");
    }

    // --- Pool cap: 5 requested, cap 3 -> the resident set NEVER exceeds 3, loads stall. ---
    {
        sc6::Sc6ResidencyConfig cfg;
        cfg.loadBudgetPerFrame = 8; cfg.evictBudgetPerFrame = 8;
        cfg.hysteresisFrames = 3; cfg.poolCapacity = 3;
        sc6::Sc6ResidencyState st;
        std::vector<sc6::Sc6PageRequest> req = {
            {0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 1, 1}, {0, 1, 0, 0}};
        for (int f = 0; f < 6; ++f) {
            sc6::Sc6StepResult r = sc6::StepResidency(st, std::span<const sc6::Sc6PageRequest>(req), space, cfg);
            check(r.residentCount <= 3, "pool cap: resident set never exceeds the cap");
        }
        check((int)st.resident.size() == 3, "pool cap: the pool sits full");
        check(st.totalLoads == 3, "pool cap: exactly cap loads happened (requested pages wait)");
    }
}

// ---------------------------------------------------------------------------------------------------
// 2. The CANONICAL synthetic fly-over: pins + proofs (shared verbatim with both showcases).
// ---------------------------------------------------------------------------------------------------
static void TestCanonicalSynthetic() {
    const sc6::Sc6CanonicalRun run = sc6::Sc6RunCanonicalShowcase();

    std::printf("[synthetic] pages=%d peakResident=%d loads=%lld evicts=%lld convergedAt=%d "
                "maxLoadedInFrame=%d digest=0x%016llx\n",
                run.stats.pages, run.stats.peakResident,
                (long long)run.stats.totalLoads, (long long)run.stats.totalEvicts,
                run.stats.convergedAt, run.stats.maxLoadedInFrame,
                (unsigned long long)run.stats.traceDigest);
    std::printf("[synthetic] unbounded: peakResident=%d maxLoadedInFrame=%d loads=%lld\n",
                run.unbounded.peakResident, run.unbounded.maxLoadedInFrame,
                (long long)run.unbounded.totalLoads);
    std::printf("[synthetic] thrash: flickerPages=%d K=%d loads=%lld evicts=%lld | K=1 loads=%lld "
                "evicts=%lld\n",
                run.thrash.flickerPages, run.rcfg.hysteresisFrames,
                (long long)run.thrash.loadsHyst, (long long)run.thrash.evictsHyst,
                (long long)run.thrash.loadsBase, (long long)run.thrash.evictsBase);

    // --- The pins (the cross-platform integer artifacts all three call sites assert). ---
    check(run.stats.pages == sc6::kSc6ExpectedPages, "synthetic: page-space size pinned (1360)");
    check(run.stats.peakResident == sc6::kSc6ExpectedPeakResident, "synthetic: peakResident PINNED");
    check(run.stats.totalLoads == sc6::kSc6ExpectedLoads, "synthetic: totalLoads PINNED");
    check(run.stats.totalEvicts == sc6::kSc6ExpectedEvicts, "synthetic: totalEvicts PINNED");
    check(run.stats.convergedAt == sc6::kSc6ExpectedConvergedAt, "synthetic: convergedAt PINNED");
    check(run.stats.traceDigest == sc6::kSc6ExpectedTraceDigest, "synthetic: traceDigest PINNED");

    // --- Two-run determinism (independent state, same feedback). ---
    check(run.stats.traceDigest == run.statsRepeat.traceDigest &&
              run.stats.totalLoads == run.statsRepeat.totalLoads &&
              run.stats.totalEvicts == run.statsRepeat.totalEvicts &&
              run.stats.peakResident == run.statsRepeat.peakResident,
          "synthetic: two full runs BYTE-IDENTICAL (trace digest + stats)");

    // --- BUDGET IS LOAD-BEARING: the unthrottled run loads more pages in a single frame than the
    // whole per-frame budget; the budgeted run never exceeds it AND still converges. ---
    check(run.unbounded.maxLoadedInFrame > run.rcfg.loadBudgetPerFrame,
          "budget: the unthrottled run exceeds the per-frame budget in one frame (a real throttle)");
    check(run.stats.maxLoadedInFrame <= run.rcfg.loadBudgetPerFrame,
          "budget: the budgeted run NEVER loads more than the budget in a frame");
    check(run.stats.convergedAt >= 0, "budget: the budgeted run CONVERGES (all requested resident)");
    check(run.unbounded.convergedAt >= 0 && run.unbounded.convergedAt <= run.stats.convergedAt,
          "budget: the unthrottled run converges no later than the budgeted one");

    // --- HYSTERESIS: the flickering view does NOT thrash under the canonical K; the K=1 baseline
    // reloads every flicker (the contrast that proves the band is load-bearing). ---
    check(run.thrash.loadsHyst == (int64_t)run.thrash.flickerPages,
          "hysteresis: K>=2 loads each flickering page EXACTLY ONCE");
    check(run.thrash.evictsHyst == 0, "hysteresis: K>=2 never evicts a flickering page");
    check(run.thrash.loadsBase > 10 * run.thrash.loadsHyst,
          "hysteresis: the K=1 baseline thrashes (>10x the loads)");
    check(run.thrash.evictsBase > 0, "hysteresis: the K=1 baseline evicts flickering pages");

    // --- Pool cap invariant. ---
    check(run.stats.capRespected, "cap: resident set never exceeded poolCapacity on any frame");
    check(run.stats.peakResident <= run.rcfg.poolCapacity, "cap: peakResident <= poolCapacity");

    // --- The heatmap capture is coherent: a loaded-now page is resident; state sets are nonempty at
    // the mid-path frame (the showcase renders a real picture, not a blank grid). ---
    int resident = 0, loadedNow = 0, ever = 0;
    for (size_t i = 0; i < run.capture.resident.size(); ++i) {
        resident += run.capture.resident[i];
        loadedNow += run.capture.loadedNow[i];
        ever += run.capture.everRequested[i];
        if (run.capture.loadedNow[i]) check(run.capture.resident[i] == 1, "capture: loaded-now implies resident");
    }
    check(resident > 0 && ever >= resident, "capture: mid-path frame has a live resident set");
    std::printf("[synthetic] capture@%d: resident=%d loadedNow=%d everRequested=%d\n",
                sc6::kSc6ShotFrame, resident, loadedNow, ever);
}

// ---------------------------------------------------------------------------------------------------
// 3. SPONZA-GATED tier: the 69-texture real-content residency trace (skips without the asset).
// ---------------------------------------------------------------------------------------------------
#if defined(HF_SC6_SPONZA_GLTF)
// PINNED artifacts of the Sponza hero-dolly run (from the verified run; deterministic integers).
// VERIFIED MSVC x64 == clang x64 (default -ffp-contract=on). HONEST CAVEAT (the sc5_foliage float
// precedent): a forced `-mfma -ffp-contract=fast` stress build DOES shift these (262k triangle
// densities sit near mip thresholds; cross-statement fusion flips a few), so these are
// x64 / fp-contract=on pins — the standard-conforming contraction mode every supported toolchain
// defaults to. The cross-platform showcase golden uses ONLY the synthetic tier (which is verified
// bit-identical even under forced fusion), so the goldens are unaffected either way.
static constexpr int32_t  kSponzaPeakResident = 557;
static constexpr int64_t  kSponzaLoads        = 1547;
static constexpr int64_t  kSponzaEvicts       = 1220;
static constexpr int32_t  kSponzaConvergedAt  = 34;
static constexpr uint64_t kSponzaTraceDigest  = 0xf0c7132a95bbe343ull;

static void TestSponzaTier() {
    std::FILE* probe = std::fopen(HF_SC6_SPONZA_GLTF, "rb");
    if (!probe) {
        std::printf("SKIP: Sponza not found at '%s' — the real-content tier needs the fetched asset "
                    "(run assets/reference/fetch_reference_assets.ps1 -Sponza). The synthetic tier "
                    "above still ran the full residency pipeline.\n", HF_SC6_SPONZA_GLTF);
        return;
    }
    std::fclose(probe);

    try {
        // --- Geometry + instances through the device-free SC3 seam. ---
        asset::CpuScene sponza = asset::LoadGltfSceneCpu(HF_SC6_SPONZA_GLTF);
        check(sponza.meshes.size() >= 50, "Sponza: hero-scale unique-mesh count");

        // --- The REAL texture set: parse the same glTF with cgltf (declarations here; the
        // implementation is hf_core's gltf_loader.cpp) and replicate LoadGltfSceneCpu's
        // first-reference dedup walk to map each unique CpuScene mesh -> its material's base-color
        // texture index. The walk order is IDENTICAL by construction (same WalkHierarchy over the
        // same node graph, same first-reference mesh dedup), and the parallel-count check below
        // falsifies any drift. ---
        cgltf_options opts = {};
        cgltf_data* data = nullptr;
        check(cgltf_parse_file(&opts, HF_SC6_SPONZA_GLTF, &data) == cgltf_result_success && data,
              "Sponza: cgltf parse for the texture table");
        if (!data) return;

        const int32_t textureCount = (int32_t)data->textures_count;
        std::printf("[Sponza] textures=%d materials=%d meshes(glTF)=%d\n",
                    textureCount, (int)data->materials_count, (int)data->meshes_count);
        check(textureCount >= 60, "Sponza: the real multi-texture set (69 expected)");

        std::vector<int32_t> texOfUnique;   // parallel to sponza.meshes (first-reference order)
        {
            std::vector<asset::SceneNode> nodes(data->nodes_count);
            for (cgltf_size i = 0; i < data->nodes_count; ++i)
                for (cgltf_size c = 0; c < data->nodes[i].children_count; ++c) {
                    ptrdiff_t ci = data->nodes[i].children[c] - data->nodes;
                    if (ci >= 0 && (cgltf_size)ci < data->nodes_count)
                        nodes[i].children.push_back((int)ci);
                }
            const cgltf_scene* scene = data->scene ? data->scene
                                                   : (data->scenes_count > 0 ? &data->scenes[0] : nullptr);
            check(scene != nullptr, "Sponza: glTF has a default scene");
            std::vector<int> roots;
            if (scene)
                for (cgltf_size i = 0; i < scene->nodes_count; ++i) {
                    ptrdiff_t ni = scene->nodes[i] - data->nodes;
                    if (ni >= 0 && (cgltf_size)ni < data->nodes_count) roots.push_back((int)ni);
                }
            std::vector<char> meshSeen(data->meshes_count, 0);
            asset::WalkHierarchy(nodes, roots, [&](int nodeIdx, const math::Mat4&) {
                const cgltf_node& nd = data->nodes[nodeIdx];
                if (!nd.mesh) return;
                ptrdiff_t mi = nd.mesh - data->meshes;
                if (mi < 0 || (cgltf_size)mi >= data->meshes_count) return;
                if (meshSeen[(size_t)mi]) return;
                meshSeen[(size_t)mi] = 1;
                const cgltf_mesh& mesh = data->meshes[(size_t)mi];
                for (cgltf_size p = 0; p < mesh.primitives_count; ++p) {
                    const cgltf_material* mat = mesh.primitives[p].material;
                    int32_t tex = -1;
                    if (mat && mat->has_pbr_metallic_roughness &&
                        mat->pbr_metallic_roughness.base_color_texture.texture)
                        tex = (int32_t)(mat->pbr_metallic_roughness.base_color_texture.texture -
                                        data->textures);
                    texOfUnique.push_back(tex);
                }
            });
        }
        check(texOfUnique.size() == sponza.meshes.size(),
              "Sponza: the replicated dedup walk matches LoadGltfSceneCpu's unique-mesh count");

        // --- Page space: every real texture its own 4-mip pyramid (8 vpps0 * 128 px = a 1024²
        // virtual texture per texture — the Sponza source resolution class). ---
        vt::VtTexture proto;
        proto.mipLevels = 4; proto.pageSize = 128; proto.virtualPagesPerSideMip0 = 8;
        const sc6::Sc6PageSpace space = sc6::MakeSc6PageSpace(textureCount, proto);
        std::printf("[Sponza] page space: %d textures x %d pages = %d global pages\n",
                    textureCount, proto.pageCount(), space.totalPages);

        // --- Adapt the scene to the mesh-feedback views. ---
        std::vector<sc6::Sc6MeshView> meshes;
        meshes.reserve(sponza.meshes.size());
        for (const auto& m : sponza.meshes)
            meshes.push_back({std::span<const scene::Vertex>(m.verts.data(), m.verts.size()),
                              std::span<const uint32_t>(m.indices.data(), m.indices.size())});
        std::vector<sc6::Sc6MeshInstance> instances;
        instances.reserve(sponza.instances.size());
        int32_t untextured = 0;
        for (const auto& inst : sponza.instances) {
            const int32_t tex = (inst.meshIndex < texOfUnique.size()) ? texOfUnique[inst.meshIndex] : -1;
            if (tex < 0) ++untextured;
            instances.push_back({inst.meshIndex, tex, inst.world});
        }
        std::printf("[Sponza] instances=%d (untextured skipped: %d)\n",
                    (int)instances.size(), untextured);

        // --- THE SC1 HERO CAMERA, DOLLIED DOWN THE NAVE: eye slides {8.2,1.7,0} -> {-4.0,1.7,0}
        // toward the fixed SC1 target {-9.5,3.4,0} over 48 frames + 16 held frames (the convergence
        // tail). tan(fovY/2) is HOST-BAKED (fovY = 1.22173048 rad = 70deg -> tan(35deg)); the
        // view-proj is built with the header's split-statement helpers (the fp-contraction guard). ---
        const int kPathFrames = 48, kHoldFrames = 16;
        const float kTanHalfFov = 0.7002075382f;
        const float center[3] = {-9.5f, 3.4f, 0.0f};
        std::vector<std::vector<sc6::Sc6PageRequest>> perFrame;
        perFrame.reserve((size_t)(kPathFrames + kHoldFrames));
        for (int f = 0; f < kPathFrames + kHoldFrames; ++f) {
            const int fc = f < kPathFrames ? f : kPathFrames - 1;
            const float t = (float)fc / (float)(kPathFrames - 1);
            float dx = -4.0f - 8.2f;
            float mx = dx * t;
            float ex = 8.2f + mx;
            const float eye[3] = {ex, 1.7f, 0.0f};
            float view[16], proj[16], vp[16];
            sc6::Sc6LookAt(eye, center, view);
            sc6::Sc6Perspective(kTanHalfFov, 1280.0f / 720.0f, 0.1f, 100.0f, proj);
            sc6::Sc6MulMat4(proj, view, vp);
            perFrame.push_back(sc6::Sc6MeshFeedback(
                std::span<const sc6::Sc6MeshView>(meshes.data(), meshes.size()),
                std::span<const sc6::Sc6MeshInstance>(instances.data(), instances.size()),
                vp, 1280.0f, 720.0f, space));
        }

        // --- The budgeted residency trace over the REAL feedback. ---
        sc6::Sc6ResidencyConfig cfg;
        cfg.loadBudgetPerFrame = 96;
        cfg.evictBudgetPerFrame = 96;
        cfg.hysteresisFrames = 6;
        cfg.poolCapacity = 1024;
        const sc6::Sc6RunStats stats = sc6::Sc6RunTrace(perFrame, space, cfg);
        const sc6::Sc6RunStats stats2 = sc6::Sc6RunTrace(perFrame, space, cfg);

        std::printf("[Sponza] residency: peakResident=%d loads=%lld evicts=%lld convergedAt=%d "
                    "maxLoadedInFrame=%d digest=0x%016llx\n",
                    stats.peakResident, (long long)stats.totalLoads, (long long)stats.totalEvicts,
                    stats.convergedAt, stats.maxLoadedInFrame,
                    (unsigned long long)stats.traceDigest);

        check(stats.traceDigest == stats2.traceDigest && stats.totalLoads == stats2.totalLoads,
              "Sponza: two full residency runs BYTE-IDENTICAL");
        check(stats.capRespected && stats.peakResident <= cfg.poolCapacity,
              "Sponza: pool cap never exceeded");
        check(stats.maxLoadedInFrame <= cfg.loadBudgetPerFrame,
              "Sponza: the load budget held every frame");
        check(stats.convergedAt >= 0, "Sponza: the budgeted run converges during the hold tail");
        check(stats.totalLoads > 0 && stats.totalEvicts > 0,
              "Sponza: the dolly both loads AND evicts (a live working set, not a static view)");

        // --- The pins (the real-content artifacts). ---
        check(stats.peakResident == kSponzaPeakResident, "Sponza: peakResident PINNED");
        check(stats.totalLoads == kSponzaLoads, "Sponza: totalLoads PINNED");
        check(stats.totalEvicts == kSponzaEvicts, "Sponza: totalEvicts PINNED");
        check(stats.convergedAt == kSponzaConvergedAt, "Sponza: convergedAt PINNED");
        check(stats.traceDigest == kSponzaTraceDigest, "Sponza: traceDigest PINNED");

        cgltf_free(data);
    } catch (const std::exception& e) {
        std::printf("FAIL: Sponza tier threw: %s\n", e.what());
        ++g_fail;
    }
}
#endif  // HF_SC6_SPONZA_GLTF

int main() {
    HF_TEST_MAIN_INIT();

    TestStepSemantics();
    TestCanonicalSynthetic();
#if defined(HF_SC6_SPONZA_GLTF)
    TestSponzaTier();
#else
    std::printf("NOTE: standalone build (no HF_SC6_SPONZA_GLTF) — the Sponza tier is compiled out.\n");
#endif

    if (g_fail == 0) { std::printf("sc6_residency_test OK\n"); return 0; }
    std::printf("sc6_residency_test: %d failures\n", g_fail);
    return 1;
}
