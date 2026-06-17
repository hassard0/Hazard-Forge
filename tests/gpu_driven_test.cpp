// Slice CB — Fully-GPU-driven pass (MDI + bindless capstone). Pure-CPU test of the COMBINED builder
// (engine/render/gpu_driven.h, namespace hf::render::gpudriven) that composes the BM multi-draw-indirect
// command + per-draw layout (render::mdi) with the BZ bindless texture-index table (render::bindless):
//
//   * Combined layout: for N objects the builder emits N MDI commands (BM layout: command i carries
//     object i's {indexCount, firstIndex, vertexOffset}, instanceCount==1, firstInstance==0;
//     drawCount()==N) AND N GpuDrivenPerDraw entries, each carrying object i's model matrix + material
//     + texIndex, where texIndex == render::bindless::Intern of that object's texture in the shared
//     table. command i and perDraw[i] describe the SAME object.
//   * Index consistency: distinct textures -> distinct texIndex; identical textures -> the SAME texIndex
//     (dedup); every object's texIndex is in range [0, table.size()).
//   * Determinism: two independent builds of the same scene -> BIT-IDENTICAL command + per-draw +
//     bindless-table arrays (memcmp over the raw bytes / pointer compare over the table).
//
// `ITexture*` is an OPAQUE handle here (never dereferenced) — the same pure-CPU contract render::mdi
// (model bytes) and render::bindless (pointer interning) rely on; fabricated distinct pointers stand in
// for texture handles. The render-invariance proof (GPU-driven image == per-object bound image) lives in
// the --gpudriven-shot showcase; this test pins the CPU layout the fully-GPU-driven path reads.
//
// Pure C++ (hf_core), ASan-eligible like the other render-data tests.
#include "render/gpu_driven.h"
#include "render/mdi.h"
#include "render/bindless.h"
#include "math/math.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace gd = hf::render::gpudriven;
namespace mdi = hf::render::mdi;
namespace bl = hf::render::bindless;
namespace rhi = hf::rhi;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Fabricate N distinct opaque ITexture* handles (never dereferenced — treated as opaque keys).
static rhi::ITexture* FakeTex(size_t i) {
    static unsigned char storage[64];
    return reinterpret_cast<rhi::ITexture*>(&storage[i]);
}

// Build a deterministic N-object scene (10x10 grid of alternating cube/sphere — the SAME shape as the
// --gpudriven-shot showcase). Each object references one of two meshes by its index-buffer slice, has a
// per-object model matrix + material, and ONE of `kPaletteSize` textures chosen deterministically (so
// distinct textures AND shared textures are both exercised — dedup matters).
static const int kGrid = 10;
static const size_t kPaletteSize = 5;

static std::vector<gd::GpuDrivenObject> BuildScene(uint32_t cubeIndexCount, uint32_t cubeVtxCount,
                                                   uint32_t sphereIndexCount,
                                                   const std::vector<rhi::ITexture*>& palette) {
    std::vector<gd::GpuDrivenObject> objs;
    for (int gz = 0; gz < kGrid; ++gz) {
        for (int gx = 0; gx < kGrid; ++gx) {
            int idx = gz * kGrid + gx;
            float fx = (float)(gx - kGrid / 2) * 1.5f + 0.75f;
            float fz = (float)(gz - kGrid / 2) * 1.5f + 0.75f;
            float bob = 0.35f * std::sin((float)idx * 0.7f);
            Mat4 m = Mat4::Translate({fx, 0.7f + bob, fz}) * Mat4::Scale({0.45f, 0.45f, 0.45f});
            gd::GpuDrivenObject ob{};
            for (int k = 0; k < 16; ++k) ob.model[k] = m.m[k];
            float tint = 0.35f + 0.5f * (float)((idx * 37) % 100) / 100.0f;
            ob.material[0] = 0.0f; ob.material[1] = tint; ob.material[2] = 0.0f; ob.material[3] = 0.0f;
            // Alternating cube/sphere, each its own slice of a shared/combined index buffer.
            if (idx & 1) {
                ob.indexCount   = sphereIndexCount;
                ob.firstIndex   = cubeIndexCount;     // spheres live AFTER the cube indices
                ob.vertexOffset = cubeVtxCount;       // spheres start after the cube verts
            } else {
                ob.indexCount   = cubeIndexCount;
                ob.firstIndex   = 0;
                ob.vertexOffset = 0;
            }
            // Texture choice cycles the palette so several distinct textures appear AND many objects
            // share each texture (dedup -> the bindless table is exactly the unique set).
            ob.texture = palette[(size_t)idx % palette.size()];
            objs.push_back(ob);
        }
    }
    return objs;
}

int main() {
    HF_TEST_MAIN_INIT();
    const uint32_t kCubeIdx = 36, kCubeVtx = 24, kSphereIdx = 2880;
    std::vector<rhi::ITexture*> palette;
    for (size_t i = 0; i < kPaletteSize; ++i) palette.push_back(FakeTex(i));

    std::vector<gd::GpuDrivenObject> objs = BuildScene(kCubeIdx, kCubeVtx, kSphereIdx, palette);
    const uint32_t N = (uint32_t)objs.size();
    check(N == 100, "scene is exactly 100 objects (10x10 grid)");

    // ---- Build the combined GPU-driven batch (MDI commands + per-draw w/ texIndex + bindless table). ----
    gd::GpuDrivenBatch batch = gd::BuildBatch(objs);

    // ---- Command count == N (drawCount). ----
    check(batch.drawCount() == N, "drawCount == N objects");
    check(batch.commands.size() == N, "one MDI command per object");
    check(batch.perDraw.size() == N, "one per-draw record per object");

    // ---- Struct sizes: command byte-compatible with VkDrawIndexedIndirectCommand; per-draw is the
    //      BM mat4+float4 PLUS a uint texIndex (+pad to a 16-byte multiple, std430-friendly). ----
    check(sizeof(mdi::MdiCommand) == 20, "MdiCommand is 20 bytes (VkDrawIndexedIndirectCommand)");
    check(sizeof(gd::GpuDrivenPerDraw) == 96, "GpuDrivenPerDraw is 96 bytes (mat4 + float4 + uint + pad)");
    check(offsetof(gd::GpuDrivenPerDraw, model) == 0, "GpuDrivenPerDraw.model at offset 0");
    check(offsetof(gd::GpuDrivenPerDraw, material) == 64, "GpuDrivenPerDraw.material at offset 64");
    check(offsetof(gd::GpuDrivenPerDraw, texIndex) == 80, "GpuDrivenPerDraw.texIndex at offset 80");

    // ---- Per-command layout (BM contract): command i == object i's draw args, instanceCount=1,
    //      firstInstance=0 (per-draw data indexed by gl_DrawID, not firstInstance). ----
    bool cmdOk = true;
    for (uint32_t i = 0; i < N; ++i) {
        const mdi::MdiCommand& c = batch.commands[i];
        const gd::GpuDrivenObject& o = objs[i];
        if (c.indexCount    != o.indexCount)    cmdOk = false;
        if (c.instanceCount != 1u)              cmdOk = false;
        if (c.firstIndex    != o.firstIndex)    cmdOk = false;
        if (c.vertexOffset  != o.vertexOffset)  cmdOk = false;
        if (c.firstInstance != 0u)              cmdOk = false;
    }
    check(cmdOk, "each command i == object i's {indexCount, firstIndex, vertexOffset}, instanceCount=1, firstInstance=0");

    // ---- Per-draw packing: perDraw[i] == object i's model matrix + material + the CORRECT texIndex
    //      (== the bindless table's index for that object's texture). ----
    bool perDrawOk = true;
    bool texIndexOk = true;
    for (uint32_t i = 0; i < N; ++i) {
        const gd::GpuDrivenPerDraw& pd = batch.perDraw[i];
        const gd::GpuDrivenObject& o = objs[i];
        for (int k = 0; k < 16; ++k) if (pd.model[k] != o.model[k]) perDrawOk = false;
        for (int k = 0; k < 4;  ++k) if (pd.material[k] != o.material[k]) perDrawOk = false;
        // texIndex is exactly the index this object's texture occupies in the bindless table.
        if (batch.table.indexOf.find(o.texture) == batch.table.indexOf.end()) texIndexOk = false;
        else if (pd.texIndex != batch.table.indexOf.at(o.texture)) texIndexOk = false;
        // In range.
        if (pd.texIndex >= batch.table.size()) texIndexOk = false;
    }
    check(perDrawOk, "perDraw[i] == object i's model matrix + material (round-trip)");
    check(texIndexOk, "perDraw[i].texIndex == bindless index of object i's texture, in range");

    // ---- Index consistency: the bindless table is exactly the unique-texture set (dedup), in
    //      first-insertion order; distinct textures -> distinct indices. ----
    check(batch.table.size() == kPaletteSize, "bindless table covers exactly the unique textures (dedup)");
    bool distinctOk = true;
    for (size_t a = 0; a < palette.size(); ++a)
        for (size_t b = a + 1; b < palette.size(); ++b)
            if (batch.table.indexOf.at(palette[a]) == batch.table.indexOf.at(palette[b]))
                distinctOk = false;
    check(distinctOk, "distinct textures -> distinct texIndex");

    // Objects sharing a texture share an index; objects with different textures differ. Object 0 and
    // object kPaletteSize use the same palette entry (idx % palette.size()), so SAME index.
    check(batch.perDraw[0].texIndex == batch.perDraw[kPaletteSize].texIndex,
          "two objects with the same texture resolve to the SAME texIndex (dedup)");
    check(batch.perDraw[0].texIndex != batch.perDraw[1].texIndex,
          "two objects with distinct textures resolve to DIFFERENT texIndex");

    // ---- Determinism: a second independent build is BIT-IDENTICAL (commands + per-draw + table). ----
    gd::GpuDrivenBatch batch2 = gd::BuildBatch(objs);
    bool sameCmds = batch.commands.size() == batch2.commands.size() &&
                    std::memcmp(batch.commands.data(), batch2.commands.data(),
                                batch.commands.size() * sizeof(mdi::MdiCommand)) == 0;
    bool samePerDraw = batch.perDraw.size() == batch2.perDraw.size() &&
                       std::memcmp(batch.perDraw.data(), batch2.perDraw.data(),
                                   batch.perDraw.size() * sizeof(gd::GpuDrivenPerDraw)) == 0;
    bool sameTable = batch.table.textures.size() == batch2.table.textures.size();
    for (size_t i = 0; sameTable && i < batch.table.textures.size(); ++i)
        if (batch.table.textures[i] != batch2.table.textures[i]) sameTable = false;
    check(sameCmds, "two builds -> bit-identical command buffer");
    check(samePerDraw, "two builds -> bit-identical per-draw buffer (incl. texIndex)");
    check(sameTable, "two builds -> identical bindless texture table");

    // ---- The command buffer's raw bytes are exactly N * 20 (tight packing, no padding). ----
    check(batch.commands.size() * sizeof(mdi::MdiCommand) == (size_t)N * 20,
          "command buffer is tightly packed N*20 bytes");

    std::printf("gpu-driven batch: %u objects -> %u MDI commands (1 draw) + %u bindless textures (1 bind), "
                "%zu per-draw records\n",
                N, batch.drawCount(), batch.table.size(), batch.perDraw.size());

    if (g_fail == 0) { std::printf("gpu_driven_test: all checks passed\n"); return 0; }
    std::printf("gpu_driven_test: %d FAILURES\n", g_fail);
    return 1;
}
