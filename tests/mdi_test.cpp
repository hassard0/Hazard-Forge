// Slice BM — GPU multi-draw-indirect batching. Pure-CPU test of the MDI command + per-draw BUILDER
// (engine/render/mdi.h, namespace hf::render::mdi). No GPU, no backend symbols — this pins the
// data-layout contract the Vulkan MDI path (one vkCmdDrawIndexedIndirect(drawCount=N)) consumes:
//
//   * Command layout: for N objects the builder emits N commands; command i carries object i's
//     {indexCount, firstIndex, vertexOffset} and instanceCount==1, firstInstance==0 (per-draw data
//     is indexed by gl_DrawID, not firstInstance). drawCount()==N. The 5x u32 record is byte-compatible
//     with VkDrawIndexedIndirectCommand (validated by sizeof/offsetof static_asserts in the header).
//   * Per-draw packing: PerDraw[i] holds object i's model matrix (column-major, 16 floats) + its
//     float4 material, round-tripped from a known object.
//   * Determinism: two independent builds of the same scene produce BIT-IDENTICAL command + per-draw
//     buffers (memcmp over the raw bytes).
//
// The render-invariance proof (MDI image == per-object reference image) lives in the --mdi-shot
// showcase; this test pins the CPU layout the GPU path reads.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/mdi.h"
#include "scene/instance_grid.h"
#include "math/math.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace hf::math;
namespace mdi = hf::render::mdi;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Build a deterministic N-object scene of mixed cube/sphere draws — the SAME shape as the --mdi-shot
// showcase (a 12x12 grid of distinct objects). Each object references one of two meshes (cube/sphere)
// by its index-buffer slice, has a per-object model matrix, and a per-object material.
static std::vector<mdi::DrawObject> BuildScene(uint32_t cubeIndexCount, uint32_t sphereIndexCount) {
    std::vector<mdi::DrawObject> objs;
    const int kGrid = 12;
    for (int gz = 0; gz < kGrid; ++gz) {
        for (int gx = 0; gx < kGrid; ++gx) {
            int idx = gz * kGrid + gx;
            float fx = (float)(gx - kGrid / 2) * 1.5f + 0.75f;
            float fz = (float)(gz - kGrid / 2) * 1.5f + 0.75f;
            float bob = 0.35f * std::sin((float)idx * 0.7f);
            Mat4 m = Mat4::Translate({fx, 0.7f + bob, fz}) * Mat4::Scale({0.45f, 0.45f, 0.45f});
            mdi::DrawObject ob{};
            for (int k = 0; k < 16; ++k) ob.model[k] = m.m[k];
            float tint = 0.35f + 0.5f * (float)((idx * 37) % 100) / 100.0f;
            ob.material[0] = 0.0f; ob.material[1] = tint; ob.material[2] = 0.0f; ob.material[3] = 0.0f;
            // Alternating cube/sphere; each mesh is its OWN slice of a shared index buffer. We give the
            // spheres a non-zero firstIndex/vertexOffset so the test exercises distinct per-command
            // offsets (not all-zero), mirroring a real combined vertex/index buffer.
            if (idx & 1) {
                ob.indexCount   = sphereIndexCount;
                ob.firstIndex   = cubeIndexCount;     // spheres live AFTER the cube indices
                ob.vertexOffset = 24;                 // cube has 24 verts; spheres start after
            } else {
                ob.indexCount   = cubeIndexCount;
                ob.firstIndex   = 0;
                ob.vertexOffset = 0;
            }
            objs.push_back(ob);
        }
    }
    return objs;
}

int main() {
    const uint32_t kCubeIdx = 36, kSphereIdx = 2880;  // representative, non-equal counts
    std::vector<mdi::DrawObject> objs = BuildScene(kCubeIdx, kSphereIdx);
    const uint32_t N = (uint32_t)objs.size();
    check(N == 144, "scene is exactly 144 objects (12x12 grid)");

    // ---- Build the MDI buffers. ----
    mdi::MdiBatch batch = mdi::BuildBatch(objs);

    // ---- Command count == N (drawCount). ----
    check(batch.drawCount() == N, "drawCount == N objects");
    check(batch.commands.size() == N, "one command per object");
    check(batch.perDraw.size() == N, "one per-draw record per object");

    // ---- The raw command record matches VkDrawIndexedIndirectCommand: 5x u32 = 20 bytes. ----
    check(sizeof(mdi::MdiCommand) == 20, "MdiCommand is 20 bytes (5x u32 == VkDrawIndexedIndirectCommand)");
    // PerDraw: 16-float model + 4-float material = 80 bytes, 16-byte aligned (std430-friendly).
    check(sizeof(mdi::PerDraw) == 80, "PerDraw is 80 bytes (mat4 + float4)");

    // ---- Per-command layout: command i carries object i's draw args, instanceCount==1,
    //      firstInstance==0 (per-draw data indexed by gl_DrawID, not firstInstance). ----
    bool cmdOk = true;
    for (uint32_t i = 0; i < N; ++i) {
        const mdi::MdiCommand& c = batch.commands[i];
        const mdi::DrawObject& o = objs[i];
        if (c.indexCount    != o.indexCount)    cmdOk = false;
        if (c.instanceCount != 1u)              cmdOk = false;
        if (c.firstIndex    != o.firstIndex)    cmdOk = false;
        if (c.vertexOffset  != o.vertexOffset)  cmdOk = false;
        if (c.firstInstance != 0u)              cmdOk = false;
    }
    check(cmdOk, "each command i == object i's {indexCount, firstIndex, vertexOffset}, instanceCount=1, firstInstance=0");

    // ---- A known object round-trips: pick object 1 (a sphere with non-zero offsets). ----
    {
        const mdi::MdiCommand& c1 = batch.commands[1];
        check(c1.indexCount == kSphereIdx, "object 1 (sphere) indexCount == sphere index count");
        check(c1.firstIndex == kCubeIdx, "object 1 firstIndex == cube index count (sphere slice offset)");
        check(c1.vertexOffset == 24, "object 1 vertexOffset == 24 (after cube verts)");
        // object 0 is a cube at the buffer origin.
        const mdi::MdiCommand& c0 = batch.commands[0];
        check(c0.indexCount == kCubeIdx && c0.firstIndex == 0 && c0.vertexOffset == 0,
              "object 0 (cube) is the buffer-origin slice");
    }

    // ---- Per-draw packing: PerDraw[i] holds object i's model matrix + material. ----
    bool perDrawOk = true;
    for (uint32_t i = 0; i < N; ++i) {
        const mdi::PerDraw& pd = batch.perDraw[i];
        const mdi::DrawObject& o = objs[i];
        for (int k = 0; k < 16; ++k) if (pd.model[k] != o.model[k]) perDrawOk = false;
        for (int k = 0; k < 4;  ++k) if (pd.material[k] != o.material[k]) perDrawOk = false;
    }
    check(perDrawOk, "PerDraw[i] == object i's model matrix + material (round-trip)");

    // ---- Determinism: a second independent build is BIT-IDENTICAL. ----
    mdi::MdiBatch batch2 = mdi::BuildBatch(objs);
    bool sameCmds = batch.commands.size() == batch2.commands.size() &&
                    std::memcmp(batch.commands.data(), batch2.commands.data(),
                                batch.commands.size() * sizeof(mdi::MdiCommand)) == 0;
    bool samePerDraw = batch.perDraw.size() == batch2.perDraw.size() &&
                       std::memcmp(batch.perDraw.data(), batch2.perDraw.data(),
                                   batch.perDraw.size() * sizeof(mdi::PerDraw)) == 0;
    check(sameCmds, "two builds -> bit-identical command buffer");
    check(samePerDraw, "two builds -> bit-identical per-draw buffer");

    // ---- The command buffer's raw bytes are exactly N * 20 (tight packing, no padding). ----
    check(batch.commands.size() * sizeof(mdi::MdiCommand) == (size_t)N * 20,
          "command buffer is tightly packed N*20 bytes");

    std::printf("mdi batch: %u objects -> %u commands (1 MDI draw), %zu per-draw records\n",
                N, batch.drawCount(), batch.perDraw.size());

    if (g_fail == 0) { std::printf("mdi_test: all checks passed\n"); return 0; }
    std::printf("mdi_test: %d FAILURES\n", g_fail);
    return 1;
}
