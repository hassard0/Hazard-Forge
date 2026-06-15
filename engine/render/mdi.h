#pragma once
// Slice BM — GPU multi-draw-indirect (MDI) batching. Pure-CPU BUILDER (header-only, no device, no
// backend symbols). Lays out, for a scene of N distinct objects, the two GPU buffers a single
// vkCmdDrawIndexedIndirect(drawCount=N) consumes:
//
//   1. An INDIRECT-ARGS buffer of N `MdiCommand` records. The record is the standard 5x u32
//      {indexCount, instanceCount, firstIndex, vertexOffset, firstInstance} — byte-identical to
//      VkDrawIndexedIndirectCommand and MTLDrawIndexedPrimitivesIndirectArguments. instanceCount is
//      always 1 (each object draws once) and firstInstance is 0: per-draw data is indexed by
//      `gl_DrawID` (SPIR-V DrawIndex), NOT firstInstance, so the firstInstance field is free.
//
//   2. A PER-DRAW STORAGE buffer of N `PerDraw` records (the object's column-major model matrix + a
//      float4 material). The MDI vertex shader (shaders/lit_mdi.vert.hlsl) reads `PerDraw[gl_DrawID]`
//      to transform/shade draw i — so command i and PerDraw[i] describe the SAME object.
//
// RENDER-INVARIANCE (the crux): because the per-object reference path pushes the SAME PerDraw[i]
// values (via a push constant) and issues the SAME indexCount/firstIndex/vertexOffset per object, the
// one-MDI-call render is mathematically identical to N per-object draws — byte-identical image. This
// builder is the single source of truth for both arrays, so the two paths can never diverge.
//
// DETERMINISM: the builder is a pure function of the input object list (a plain memcpy of each
// object's fields into the two POD arrays, in order), so two builds of the same scene are
// bit-identical (pinned by tests/mdi_test.cpp).

#include <cstdint>
#include <cstddef>
#include <vector>

namespace hf::render::mdi {

// One indexed indirect-draw command. The field order + types match VkDrawIndexedIndirectCommand
// (and MTLDrawIndexedPrimitivesIndirectArguments) exactly, so the packed array can be uploaded into a
// BufferUsage::Indirect buffer and fed straight to vkCmdDrawIndexedIndirect with stride=sizeof(MdiCommand).
struct MdiCommand {
    uint32_t indexCount;     // indices to draw for this object
    uint32_t instanceCount;  // always 1 (one draw per object; gl_DrawID selects per-draw data)
    uint32_t firstIndex;     // first index in the (shared) index buffer
    uint32_t vertexOffset;   // value added to every index before vertex fetch
    uint32_t firstInstance;  // always 0 (per-draw data is indexed by gl_DrawID, not firstInstance)
};
static_assert(sizeof(MdiCommand) == 20, "MdiCommand must be 5x u32 (VkDrawIndexedIndirectCommand)");
static_assert(offsetof(MdiCommand, indexCount) == 0,  "MdiCommand layout");
static_assert(offsetof(MdiCommand, instanceCount) == 4, "MdiCommand layout");
static_assert(offsetof(MdiCommand, firstIndex) == 8,  "MdiCommand layout");
static_assert(offsetof(MdiCommand, vertexOffset) == 12, "MdiCommand layout");
static_assert(offsetof(MdiCommand, firstInstance) == 16, "MdiCommand layout");

// Per-draw data the MDI vertex shader reads as PerDraw[gl_DrawID]: the object's column-major model
// matrix (16 floats, matching math::Mat4::m + the per-instance mat4 stream byte layout) and a float4
// material (x=metallic, y=roughness-tint, fed to the shared lit fragment exactly like the per-object
// push-constant material). std430-friendly: 16-byte aligned, 80 bytes, tightly packed in the array.
struct PerDraw {
    float model[16];    // column-major mat4 (offsets 0..63)
    float material[4];  // float4 material (offset 64)
};
static_assert(sizeof(PerDraw) == 80, "PerDraw must be mat4 + float4 (80 bytes)");
static_assert(offsetof(PerDraw, model) == 0, "PerDraw layout");
static_assert(offsetof(PerDraw, material) == 64, "PerDraw layout");

// One scene object the builder turns into (command i, PerDraw i). Carries the object's index-buffer
// slice (indexCount/firstIndex/vertexOffset — distinct per mesh in a shared/combined buffer) plus the
// per-draw model matrix + material. instanceCount/firstInstance are NOT inputs: the builder fixes them
// (1 / 0) because each object draws once and per-draw data is gl_DrawID-indexed.
struct DrawObject {
    uint32_t indexCount   = 0;
    uint32_t firstIndex   = 0;
    uint32_t vertexOffset = 0;
    float    model[16]    = {};   // column-major mat4
    float    material[4]  = {};   // float4 material (metallic, roughness-tint, ...)
};

// The two packed GPU buffers for one MDI call. `commands` -> BufferUsage::Indirect (drawCount() == its
// size; stride == sizeof(MdiCommand)); `perDraw` -> BufferUsage::Storage, indexed by gl_DrawID.
struct MdiBatch {
    std::vector<MdiCommand> commands;  // N indirect-draw commands
    std::vector<PerDraw>    perDraw;   // N per-draw records (parallel to commands)
    uint32_t drawCount() const { return (uint32_t)commands.size(); }
};

// Lay out the MDI command + per-draw arrays for `objs` IN ORDER. Command i gets object i's index-buffer
// slice (instanceCount=1, firstInstance=0); PerDraw[i] gets object i's model matrix + material. Pure
// function — deterministic, no GPU. The Vulkan showcase uploads `commands` to an indirect buffer and
// `perDraw` to a storage buffer, then issues ONE vkCmdDrawIndexedIndirect(commands, 0, drawCount(),
// sizeof(MdiCommand)).
inline MdiBatch BuildBatch(const std::vector<DrawObject>& objs) {
    MdiBatch out;
    out.commands.reserve(objs.size());
    out.perDraw.reserve(objs.size());
    for (const DrawObject& o : objs) {
        MdiCommand c{};
        c.indexCount    = o.indexCount;
        c.instanceCount = 1u;            // one draw per object
        c.firstIndex    = o.firstIndex;
        c.vertexOffset  = o.vertexOffset;
        c.firstInstance = 0u;            // per-draw data indexed by gl_DrawID, not firstInstance
        out.commands.push_back(c);

        PerDraw pd{};
        for (int k = 0; k < 16; ++k) pd.model[k]    = o.model[k];
        for (int k = 0; k < 4;  ++k) pd.material[k] = o.material[k];
        out.perDraw.push_back(pd);
    }
    return out;
}

}  // namespace hf::render::mdi
