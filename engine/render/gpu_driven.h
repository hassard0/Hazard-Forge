#pragma once
// Slice CB — Fully-GPU-driven pass (MDI + bindless capstone). Pure-CPU BUILDER (header-only, no device,
// no backend symbols) that COMPOSES the two already-proven slices into THE modern GPU-driven renderer:
//
//   * BM (render/mdi.h): the indirect-args buffer of N MdiCommand records that ONE
//     vkCmdDrawIndexedIndirect(drawCount=N) consumes (per-draw data indexed by gl_DrawID).
//   * BZ (render/bindless.h): the texture-index table that fills ONE bindless descriptor array (bound
//     ONCE), each draw sampling its base color by INDEX (gTextures[NonUniformResourceIndex(texIndex)]).
//
// The combined PER-DRAW record (GpuDrivenPerDraw) carries the BM model matrix + material AND the BZ
// texture index: the vertex shader (lit_gpudriven.vert) reads PerDraw[gl_DrawID].{model,material} and
// passes texIndex FLAT to the fragment, which samples gTextures[NonUniformResourceIndex(texIndex)]
// (lit_gpudriven.frag). One DrawIndexedMultiIndirect(N) + one BindBindlessTextures draws an entire
// multi-material scene.
//
// RENDER-INVARIANCE (the capstone crux): because the per-object reference path pushes the SAME model +
// material AND binds the SAME texture (the one this object's texIndex selects in the bindless array),
// the one-MDI-call + one-bindless-bind render is mathematically identical to N per-object per-material
// BOUND draws — byte-identical image. This builder is the single source of truth for the commands, the
// per-draw records, AND the bindless table, so the GPU-driven and bound paths can never diverge.
//
// PURE CPU: model/material are plain floats memcpy'd in order; `ITexture*` is an OPAQUE interning key
// (never dereferenced — render::bindless::Intern), so the whole builder is unit-testable with no GPU
// (tests/gpu_driven_test.cpp). DETERMINISM: a pure function of the input object list — commands/per-draw
// are laid out in order, texIndex assigned in first-insertion order — so two builds are bit-identical.

#include <cstdint>
#include <cstddef>
#include <vector>

#include "render/mdi.h"
#include "render/bindless.h"

namespace hf::rhi { class ITexture; }

namespace hf::render::gpudriven {

// The combined per-draw record the fully-GPU-driven vertex shader reads as PerDraw[gl_DrawID]: the BM
// model matrix (column-major mat4) + float4 material PLUS the BZ texture index. Extends render::mdi's
// PerDraw layout (model at 0, material at 64) with `texIndex` at 80, padded to 96 bytes (a 16-byte
// multiple) so the array is std430-friendly and tightly packed. lit_gpudriven.vert reads model+material,
// passes texIndex flat to lit_gpudriven.frag which samples gTextures[NonUniformResourceIndex(texIndex)].
struct GpuDrivenPerDraw {
    float    model[16];   // column-major mat4 (offsets 0..63) — same as mdi::PerDraw
    float    material[4]; // float4 material (offset 64)        — same as mdi::PerDraw
    uint32_t texIndex;    // bindless array index (offset 80)   — BZ addition
    uint32_t pad[3];      // pad to 96 bytes (16-byte multiple)
};
static_assert(sizeof(GpuDrivenPerDraw) == 96, "GpuDrivenPerDraw must be mat4+float4+uint+pad (96 bytes)");
static_assert(offsetof(GpuDrivenPerDraw, model) == 0,     "GpuDrivenPerDraw layout");
static_assert(offsetof(GpuDrivenPerDraw, material) == 64, "GpuDrivenPerDraw layout");
static_assert(offsetof(GpuDrivenPerDraw, texIndex) == 80, "GpuDrivenPerDraw layout");

// One scene object the builder turns into (command i, perDraw i): its index-buffer slice (BM) + its
// per-draw model matrix + material (BM) + its base-color texture (BZ, an opaque handle resolved to a
// stable bindless index). instanceCount/firstInstance are NOT inputs (the builder fixes them 1/0).
struct GpuDrivenObject {
    uint32_t indexCount   = 0;
    uint32_t firstIndex   = 0;
    uint32_t vertexOffset = 0;
    float    model[16]    = {};   // column-major mat4
    float    material[4]  = {};   // float4 material (metallic, roughness-tint, ...)
    hf::rhi::ITexture* texture = nullptr;  // base-color texture (opaque; interned to a bindless index)
};

// The three packed GPU resources for ONE fully-GPU-driven draw: the BM `commands` (-> BufferUsage::Indirect,
// drawCount()==its size, stride==sizeof(mdi::MdiCommand)), the combined `perDraw` (-> BufferUsage::Storage,
// indexed by gl_DrawID, carrying model+material+texIndex), and the BZ `table` (its `textures` array fills
// the bindless descriptor set IN ORDER; `indexOf` maps each texture to its array slot).
struct GpuDrivenBatch {
    std::vector<mdi::MdiCommand>    commands;  // N indirect-draw commands (BM)
    std::vector<GpuDrivenPerDraw>   perDraw;   // N per-draw records (model+material+texIndex)
    bindless::BindlessTable         table;     // the deduplicated bindless texture set (BZ)
    uint32_t drawCount() const { return (uint32_t)commands.size(); }
};

// Lay out the MDI command + combined per-draw arrays + the bindless table for `objs` IN ORDER. Command i
// gets object i's index slice (BM: instanceCount=1, firstInstance=0). perDraw[i] gets object i's model +
// material + texIndex, where texIndex == bindless::Intern of object i's texture in the shared table (so
// command i, perDraw[i], and the bindless slot all describe the SAME object). Pure function —
// deterministic, no GPU. The Vulkan showcase uploads `commands` to an indirect buffer, `perDraw` to a
// storage buffer, fills the bindless set from `table.textures`, and issues ONE DrawIndexedMultiIndirect +
// ONE BindBindlessTextures.
inline GpuDrivenBatch BuildBatch(const std::vector<GpuDrivenObject>& objs) {
    GpuDrivenBatch out;
    out.commands.reserve(objs.size());
    out.perDraw.reserve(objs.size());
    for (const GpuDrivenObject& o : objs) {
        mdi::MdiCommand c{};
        c.indexCount    = o.indexCount;
        c.instanceCount = 1u;            // one draw per object
        c.firstIndex    = o.firstIndex;
        c.vertexOffset  = o.vertexOffset;
        c.firstInstance = 0u;            // per-draw data indexed by gl_DrawID, not firstInstance
        out.commands.push_back(c);

        GpuDrivenPerDraw pd{};
        for (int k = 0; k < 16; ++k) pd.model[k]    = o.model[k];
        for (int k = 0; k < 4;  ++k) pd.material[k] = o.material[k];
        // Intern this object's texture into the shared table (first-insertion order, dedup) -> its index.
        pd.texIndex = bindless::Intern(out.table, o.texture);
        pd.pad[0] = pd.pad[1] = pd.pad[2] = 0u;
        out.perDraw.push_back(pd);
    }
    return out;
}

}  // namespace hf::render::gpudriven
