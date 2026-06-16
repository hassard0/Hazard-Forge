// Slice CD — Fully-GPU-driven-CULLED pass (compute-cull -> MDI + bindless). Pure-CPU test of the
// cull+compact MIRROR (engine/render/gpu_culled.h, namespace hf::render::gpuculled) that the GPU
// compute pass (shaders/gpudriven_cull.comp.hlsl) implements: a compute shader frustum-culls the FULL
// per-draw list (each object: model + world bounding sphere + material + texIndex), COMPACTS the
// survivors (ordered single-workgroup prefix sum -> SOURCE-INDEX order, the AR determinism trick) into
// the OUTPUT GpuDrivenPerDraw SSBO, and writes the MDI command's drawCount = survivor count.
//
// This pins, WITHOUT a GPU, the three contracts the render-invariance proof rests on:
//   * Cull+compact reference: for a known object set + a known frustum, the cull+compact yields EXACTLY
//     the in-frustum survivors, compacted in source-index order; the count matches an INDEPENDENT
//     brute-force frustum.h reference; 0 false-culls (the conservative sphere test never drops a sphere
//     that reaches the frustum interior).
//   * Per-draw carry: each survivor's compacted GpuDrivenPerDraw carries the RIGHT model + material +
//     texIndex (the compaction preserves the per-draw record, parallel to the AR survivor-stream test).
//   * Determinism: two independent builds -> bit-identical compacted per-draw buffer + drawCount.
//
// The frustum/sphere math is the SAME render::frustum + render::gpu_cull::InstanceWorldSphere the
// already-proven AR slice uses, so this slice cannot diverge from the validated cull logic. The
// GPU-culled-image == CPU-frustum-culled-bound-image byte-identical proof lives in --gpucull-draw-shot;
// this test pins the CPU layout the fully-GPU-driven-culled path reads.
//
// Pure C++ (hf_core), ASan-eligible like the other render-data tests.
#include "render/gpu_culled.h"
#include "render/gpu_driven.h"
#include "render/frustum.h"
#include "render/gpu_cull.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace hf::math;
namespace fr  = hf::render::frustum;
namespace gc  = hf::render::gpu_cull;
namespace gcd = hf::render::gpuculled;
namespace gd  = hf::render::gpudriven;
namespace mdi = hf::render::mdi;
namespace rhi = hf::rhi;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Fabricate N distinct opaque ITexture* handles (never dereferenced — opaque keys, the same contract
// render::bindless relies on).
static rhi::ITexture* FakeTex(size_t i) {
    static unsigned char storage[64];
    return reinterpret_cast<rhi::ITexture*>(&storage[i]);
}

// The unit-cube local bound used throughout (cube spans [-0.5,0.5]^3 -> center origin, radius
// |(0.5,0.5,0.5)| = sqrt(0.75)). The same bound the AR cull + the showcase use.
static const Vec3  kLocalCenter{0.0f, 0.0f, 0.0f};
static const float kLocalRadius = 0.86602540378f;  // sqrt(0.75)

// Build a deterministic N-object scene: a WIDE grid so a narrow camera sees only a central subset
// (a REAL cull). Each object carries a model matrix + per-object material tint + one of `kPalette`
// textures (cycled, so dedup is exercised). Mirrors the --gpucull-draw-shot scene shape.
static const int    kGrid    = 12;       // 12x12 = 144 objects (>100, per spec)
static const size_t kPalette = 5;

static std::vector<gcd::CulledObject> BuildScene(uint32_t cubeIdx, uint32_t cubeVtx, uint32_t sphIdx,
                                                 const std::vector<rhi::ITexture*>& palette) {
    std::vector<gcd::CulledObject> objs;
    for (int gz = 0; gz < kGrid; ++gz) {
        for (int gx = 0; gx < kGrid; ++gx) {
            int idx = gz * kGrid + gx;
            float fx = (float)(gx - kGrid / 2) * 2.6f + 1.3f;
            float fz = (float)(gz - kGrid / 2) * 2.6f + 1.3f;
            float bob = 0.35f * std::sin((float)idx * 0.7f);
            Mat4 m = Mat4::Translate({fx, 0.7f + bob, fz}) * Mat4::Scale({0.45f, 0.45f, 0.45f});
            gcd::CulledObject ob{};
            for (int k = 0; k < 16; ++k) ob.model[k] = m.m[k];
            float tint = 0.35f + 0.5f * (float)((idx * 37) % 100) / 100.0f;
            ob.material[0] = 0.0f; ob.material[1] = tint; ob.material[2] = 0.0f; ob.material[3] = 0.0f;
            const bool isSphere = (idx & 1) != 0;
            ob.indexCount   = isSphere ? sphIdx  : cubeIdx;
            ob.firstIndex   = isSphere ? cubeIdx : 0u;
            ob.vertexOffset = isSphere ? cubeVtx : 0u;
            ob.texIndex     = (uint32_t)((size_t)idx % palette.size());
            ob.localCenter  = kLocalCenter;
            ob.localRadius  = kLocalRadius;
            objs.push_back(ob);
        }
    }
    return objs;
}

int main() {
    const uint32_t kCubeIdx = 36, kCubeVtx = 24, kSphIdx = 2880;
    std::vector<rhi::ITexture*> palette;
    for (size_t i = 0; i < kPalette; ++i) palette.push_back(FakeTex(i));

    std::vector<gcd::CulledObject> objs = BuildScene(kCubeIdx, kCubeVtx, kSphIdx, palette);
    const uint32_t N = (uint32_t)objs.size();
    check(N == 144, "scene is exactly 144 objects (12x12 grid, >100 per spec)");

    // ---- A narrow render camera that sees only a central subset of the wide grid (Vulkan clip), then
    //      the six Gribb-Hartmann planes — the SAME extraction the AR cull + render path use. ----
    const float aspect = 16.0f / 9.0f;
    Mat4 view = Mat4::LookAt(Vec3{0, 11, 17}, Vec3{0, 0, 0}, Vec3{0, 1, 0});
    Mat4 proj = Mat4::Perspective(0.6108652f /*35deg*/, aspect, 0.5f, 80.0f);
    Mat4 vp   = proj * view;
    fr::Frustum f = fr::FromViewProj(vp);

    // ---- The cull+compact under test. Produces the compacted survivor MDI commands + GpuDrivenPerDraw
    //      records (model+material+texIndex) + the survivor drawCount. ----
    gcd::CulledBatch batch = gcd::CullAndCompact(objs, f);

    // =====================================================================================
    // 1) CULL+COMPACT REFERENCE: an INDEPENDENT brute-force frustum.h pass over the SAME objects in
    //    source order; the survivor INDEX list is the reference. The code under test cannot share a
    //    compaction bug because the reference stores indices + re-derives the world sphere by hand.
    // =====================================================================================
    std::vector<uint32_t> refIdx;
    for (uint32_t i = 0; i < N; ++i) {
        const gcd::CulledObject& o = objs[i];
        Mat4 model; for (int k = 0; k < 16; ++k) model.m[k] = o.model[k];
        Vec3 center = MulPoint(model, o.localCenter);
        float col0Len = std::sqrt(o.model[0]*o.model[0] + o.model[1]*o.model[1] + o.model[2]*o.model[2]);
        float radius = o.localRadius * col0Len;
        if (!fr::SphereOutside(f, center, radius)) refIdx.push_back(i);
    }
    check(batch.drawCount == (uint32_t)refIdx.size(),
          "GPU-culled drawCount == independent brute-force frustum.h survivor count");
    check(batch.commands.size() == batch.drawCount, "one MDI command per survivor");
    check(batch.perDraw.size()  == batch.drawCount, "one per-draw record per survivor");
    check(batch.drawCount > 0 && batch.drawCount < N,
          "non-trivial cull: some kept, some culled (camera sees a SUBSET)");

    // ---- 0 false-culls: the cross-check against gpu_cull::SurvivorCount (the AR count reference over
    //      the same flattened models) must agree exactly — conservative sphere test, never drops a
    //      visible sphere. ----
    std::vector<float> flatModels;
    flatModels.reserve((size_t)N * 16);
    for (const auto& o : objs) for (int k = 0; k < 16; ++k) flatModels.push_back(o.model[k]);
    uint32_t arCount = gc::SurvivorCount(flatModels, N, f, kLocalCenter, kLocalRadius);
    check(batch.drawCount == arCount, "drawCount == AR gpu_cull::SurvivorCount (same frustum-sphere logic)");

    // =====================================================================================
    // 2) ORDER + PER-DRAW CARRY: survivor j must be exactly object refIdx[j] (source-index order), and
    //    its compacted GpuDrivenPerDraw must carry that object's model+material+texIndex, and its MDI
    //    command its index slice. This is the determinism contract + the per-draw-data preservation.
    // =====================================================================================
    bool ordered = true, carryOk = true, cmdOk = true;
    for (size_t j = 0; j + 1 < refIdx.size(); ++j)
        if (refIdx[j] >= refIdx[j + 1]) ordered = false;
    for (size_t j = 0; j < refIdx.size(); ++j) {
        const gcd::CulledObject& src = objs[refIdx[j]];
        const gd::GpuDrivenPerDraw& pd = batch.perDraw[j];
        for (int k = 0; k < 16; ++k) if (pd.model[k]    != src.model[k])    carryOk = false;
        for (int k = 0; k < 4;  ++k) if (pd.material[k] != src.material[k]) carryOk = false;
        if (pd.texIndex != src.texIndex) carryOk = false;
        const mdi::MdiCommand& c = batch.commands[j];
        if (c.indexCount    != src.indexCount)   cmdOk = false;
        if (c.instanceCount != 1u)               cmdOk = false;
        if (c.firstIndex    != src.firstIndex)   cmdOk = false;
        if (c.vertexOffset  != src.vertexOffset) cmdOk = false;
        if (c.firstInstance != 0u)               cmdOk = false;
    }
    check(ordered, "survivor indices are strictly increasing (source-index order — ordered compaction)");
    check(carryOk, "each survivor's per-draw carries the right model+material+texIndex");
    check(cmdOk,   "each survivor's MDI command == that object's index slice (instanceCount=1, firstInstance=0)");

    // =====================================================================================
    // 3) DETERMINISM: a second independent build is BIT-IDENTICAL (commands + per-draw + drawCount).
    // =====================================================================================
    gcd::CulledBatch batch2 = gcd::CullAndCompact(objs, f);
    check(batch2.drawCount == batch.drawCount, "two builds -> identical drawCount");
    bool sameCmds = batch.commands.size() == batch2.commands.size() &&
                    std::memcmp(batch.commands.data(), batch2.commands.data(),
                                batch.commands.size() * sizeof(mdi::MdiCommand)) == 0;
    bool samePerDraw = batch.perDraw.size() == batch2.perDraw.size() &&
                       std::memcmp(batch.perDraw.data(), batch2.perDraw.data(),
                                   batch.perDraw.size() * sizeof(gd::GpuDrivenPerDraw)) == 0;
    check(sameCmds, "two builds -> bit-identical compacted MDI command buffer");
    check(samePerDraw, "two builds -> bit-identical compacted per-draw buffer (incl. texIndex)");

    std::printf("gpu-culled cull+compact: %u survivors of %u (camera subset), per-draw carried, deterministic\n",
                batch.drawCount, N);

    // =====================================================================================
    // 4) RANDOMIZED STRESS: many random frustums; the ordered cull+compact must ALWAYS equal a fresh
    //    brute-force in-order survivor list (model+material+texIndex), never reorder/drop/duplicate.
    // =====================================================================================
    {
        std::mt19937 rng(0xCDC0FFEEu);
        std::uniform_real_distribution<float> ex(-20.0f, 20.0f), ey(2.0f, 18.0f), ez(6.0f, 26.0f);
        std::uniform_real_distribution<float> uf(0.4f, 1.2f);
        int trials = 0, mismatches = 0, allCulled = 0, allKept = 0;
        for (int t = 0; t < 200; ++t) {
            Vec3 eye{ex(rng), ey(rng), ez(rng)};
            Mat4 v = Mat4::LookAt(eye, Vec3{ex(rng) * 0.1f, 0, 0}, Vec3{0, 1, 0});
            Mat4 p = Mat4::Perspective(uf(rng), aspect, 0.4f, 90.0f);
            fr::Frustum ff = fr::FromViewProj(p * v);

            gcd::CulledBatch b = gcd::CullAndCompact(objs, ff);
            std::vector<uint32_t> ref;
            for (uint32_t i = 0; i < N; ++i) {
                const gcd::CulledObject& o = objs[i];
                Mat4 model; for (int k = 0; k < 16; ++k) model.m[k] = o.model[k];
                Vec3 center = MulPoint(model, o.localCenter);
                float col0Len = std::sqrt(o.model[0]*o.model[0] + o.model[1]*o.model[1] + o.model[2]*o.model[2]);
                if (!fr::SphereOutside(ff, center, o.localRadius * col0Len)) ref.push_back(i);
            }
            ++trials;
            bool ok = (b.drawCount == ref.size()) && (b.perDraw.size() == ref.size());
            for (size_t j = 0; ok && j < ref.size(); ++j) {
                const gcd::CulledObject& src = objs[ref[j]];
                const gd::GpuDrivenPerDraw& pd = b.perDraw[j];
                for (int k = 0; k < 16; ++k) if (pd.model[k] != src.model[k]) ok = false;
                if (pd.texIndex != src.texIndex) ok = false;
                if (b.commands[j].indexCount != src.indexCount) ok = false;
            }
            if (!ok) ++mismatches;
            if (b.drawCount == 0) ++allCulled;
            if (b.drawCount == N) ++allKept;
        }
        check(mismatches == 0, "random frustums: ordered cull+compact always equals in-order reference");
        std::printf("random stress: %d trials, %d mismatches, %d all-culled, %d all-kept\n",
                    trials, mismatches, allCulled, allKept);
    }

    if (g_fail == 0) { std::printf("gpu_culled_test: all checks passed\n"); return 0; }
    std::printf("gpu_culled_test: %d FAILURES\n", g_fail);
    return 1;
}
