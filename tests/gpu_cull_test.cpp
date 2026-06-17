// Slice AR — GPU-driven culling. Pure CPU mirror of the SAME ordered-compaction + frustum-sphere
// logic shaders/cull.comp.hlsl runs on the GPU. Namespace hf::render::gpu_cull. This pins the
// DETERMINISM CONTRACT without needing a device: survivors must be emitted in SOURCE-INDEX ORDER
// (the single-workgroup prefix-sum compaction), and the compacted survivor list + count must match
// an INDEPENDENT brute-force reference IN ORDER.
//
// The crux this test PINS: the shader uses an ordered scan (NOT an unordered atomicAdd append) so
// the survivor buffer the indirect draw consumes is byte-stable across runs/backends — that is what
// makes the gpu_cull.png golden reproducible. An out-of-order compaction would pass a count-only
// check but reorder the per-instance stream and silently break the image. So we assert ORDER, not
// just the count.
//
// We also build a 1024-instance grid + a known camera/frustum (the SAME shape the --gpu-cull-shot
// showcase uses) and confirm the mirror's count matches a brute-force loop, AND that the surviving
// model matrices appear in strictly increasing source order.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/gpu_cull.h"
#include "render/frustum.h"
#include "scene/instance_grid.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace fr = hf::render::frustum;
namespace gc = hf::render::gpu_cull;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Flatten an InstanceData grid into the 16-floats-per-instance layout the mirror/shader consume.
static std::vector<float> FlattenGrid(const std::vector<hf::scene::InstanceData>& g) {
    std::vector<float> out;
    out.reserve(g.size() * 16);
    for (const auto& inst : g)
        for (int k = 0; k < 16; ++k) out.push_back(inst.model[k]);
    return out;
}

int main() {
    HF_TEST_MAIN_INIT();
    // ---- A unit-cube local bound (matches the showcase grid of unit cubes). The unit cube spans
    // [-0.5,0.5]^3, so localCenter = origin and localRadius = |(0.5,0.5,0.5)| = sqrt(0.75). ----
    const Vec3 localCenter{0.0f, 0.0f, 0.0f};
    const float localRadius = std::sqrt(0.75f);

    // ---- Build the SAME 32x32 = 1024 instance grid the showcase uses (one workgroup). ----
    const uint32_t kGridN = 32;
    std::vector<hf::scene::InstanceData> grid =
        hf::scene::BuildInstanceGrid(kGridN, /*spacing=*/1.3f, /*scale=*/0.45f);
    const uint32_t kInstanceCount = (uint32_t)grid.size();
    check(kInstanceCount == 1024, "grid is exactly 1024 instances (one workgroup)");
    std::vector<float> models = FlattenGrid(grid);

    // ---- A narrow render camera that sees only a central subset of the wide grid. View-proj built
    // the same way the showcase does (Vulkan clip), then the six Gribb-Hartmann planes. ----
    const float aspect = 16.0f / 9.0f;
    Mat4 view = Mat4::LookAt(Vec3{0, 6, 14}, Vec3{0, 0, 0}, Vec3{0, 1, 0});
    Mat4 proj = Mat4::Perspective(0.6108652f /*35deg*/, aspect, 0.5f, 60.0f);
    Mat4 vp = proj * view;
    fr::Frustum f = fr::FromViewProj(vp);

    // ---- The mirror's ordered compaction. ----
    std::vector<float> survivors;
    uint32_t count = gc::CompactSurvivors(models, kInstanceCount, f, localCenter, localRadius,
                                          survivors);

    // ---- Independent brute-force reference: walk every instance in source order, compute its world
    // sphere with raw math (NOT reusing gpu_cull.h's helper), test with frustum.h, and append the
    // source index of each survivor. The reference cannot share a compaction bug with the code under
    // test because it stores INDICES and re-derives the world sphere by hand. ----
    std::vector<uint32_t> refIndices;
    for (uint32_t i = 0; i < kInstanceCount; ++i) {
        const float* m = models.data() + (size_t)i * 16;
        Mat4 model;
        for (int k = 0; k < 16; ++k) model.m[k] = m[k];
        Vec3 center = MulPoint(model, localCenter);
        float col0Len = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
        float radius = localRadius * col0Len;
        if (!fr::SphereOutside(f, center, radius)) refIndices.push_back(i);
    }

    // ---- Count equality. ----
    check(count == (uint32_t)refIndices.size(), "mirror survivor count == brute-force reference");
    check(count == survivors.size() / 16, "returned count == emitted matrix count");
    check(count > 0 && count < kInstanceCount,
          "non-trivial cull: some kept, some culled (camera sees a SUBSET)");

    // ---- ORDER equality: survivor[j] must be exactly the model of refIndices[j], AND refIndices
    // must be strictly increasing (source order). This is the determinism contract. ----
    bool ordered = true, matrixMatch = true;
    for (size_t j = 0; j + 1 < refIndices.size(); ++j)
        if (refIndices[j] >= refIndices[j + 1]) ordered = false;
    for (size_t j = 0; j < refIndices.size() && j * 16 < survivors.size(); ++j) {
        const float* src = models.data() + (size_t)refIndices[j] * 16;
        for (int k = 0; k < 16; ++k)
            if (survivors[j * 16 + k] != src[k]) matrixMatch = false;
    }
    check(ordered, "brute-force survivor indices are strictly increasing (source order)");
    check(matrixMatch, "compacted survivor stream == source models in source-index order");

    std::printf("gpu_cull grid partition: %u survivors of %u (camera subset)\n",
                count, kInstanceCount);

    // ---- SurvivorCount must agree with CompactSurvivors (same scan, count-only path). ----
    check(gc::SurvivorCount(models, kInstanceCount, f, localCenter, localRadius) == count,
          "SurvivorCount agrees with CompactSurvivors");

    // ---- Randomized stress: many random frustums over the same grid; the ordered compaction must
    // ALWAYS equal a freshly-built brute-force in-order index list (never reorders, never drops or
    // duplicates a survivor). Catches any prefix-sum off-by-one in the mirror. ----
    {
        std::mt19937 rng(0xC0FFEEu);
        std::uniform_real_distribution<float> ex(-20.0f, 20.0f);
        std::uniform_real_distribution<float> ey(2.0f, 18.0f);
        std::uniform_real_distribution<float> ez(6.0f, 26.0f);
        std::uniform_real_distribution<float> uf(0.4f, 1.2f);
        int trials = 0, mismatches = 0, allCulled = 0, allKept = 0;
        for (int t = 0; t < 200; ++t) {
            Vec3 eye{ex(rng), ey(rng), ez(rng)};
            Mat4 v = Mat4::LookAt(eye, Vec3{ex(rng) * 0.1f, 0, 0}, Vec3{0, 1, 0});
            Mat4 p = Mat4::Perspective(uf(rng), aspect, 0.4f, 80.0f);
            fr::Frustum ff = fr::FromViewProj(p * v);

            std::vector<float> surv;
            uint32_t c = gc::CompactSurvivors(models, kInstanceCount, ff, localCenter, localRadius,
                                              surv);
            std::vector<uint32_t> ref;
            for (uint32_t i = 0; i < kInstanceCount; ++i) {
                const float* m = models.data() + (size_t)i * 16;
                Mat4 model;
                for (int k = 0; k < 16; ++k) model.m[k] = m[k];
                Vec3 center = MulPoint(model, localCenter);
                float col0Len = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
                if (!fr::SphereOutside(ff, center, localRadius * col0Len)) ref.push_back(i);
            }
            ++trials;
            bool ok = (c == ref.size()) && (surv.size() == ref.size() * 16);
            for (size_t j = 0; ok && j < ref.size(); ++j) {
                const float* src = models.data() + (size_t)ref[j] * 16;
                for (int k = 0; k < 16; ++k)
                    if (surv[j * 16 + k] != src[k]) ok = false;
            }
            if (!ok) ++mismatches;
            if (c == 0) ++allCulled;
            if (c == kInstanceCount) ++allKept;
        }
        check(mismatches == 0, "random frustums: ordered compaction always equals in-order reference");
        std::printf("random stress: %d trials, %d mismatches, %d all-culled, %d all-kept\n",
                    trials, mismatches, allCulled, allKept);
    }

    if (g_fail == 0) { std::printf("gpu_cull_test: all checks passed\n"); return 0; }
    std::printf("gpu_cull_test: %d FAILURES\n", g_fail);
    return 1;
}
