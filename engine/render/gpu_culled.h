#pragma once
// Slice CD — Fully-GPU-driven-CULLED pass (compute-cull -> MDI + bindless). Pure-CPU cull+compact
// MIRROR (header-only, no device, no backend symbols). This is the AUTHORITATIVE reference for the
// EXACT same ordered-compaction + frustum-sphere logic that shaders/gpudriven_cull.comp.hlsl runs on
// the GPU — the COMPOSITION of the four already-proven slices:
//
//   * AR (render/gpu_cull.h, shaders/cull.comp.hlsl): the frustum-sphere test + the ORDERED
//     single-workgroup prefix-sum compaction (survivors in SOURCE-INDEX order, NOT an unordered
//     atomicAdd append — the determinism trick that makes the golden reproducible). The cull decision
//     here is the IDENTICAL conservative render::frustum::SphereOutside the AR cull + the render path
//     use, on the IDENTICAL world bounding sphere render::gpu_cull::InstanceWorldSphere computes.
//   * BM (render/mdi.h): the compacted survivors become N' MdiCommand records; the survivor count is
//     the MDI drawCount (the GPU compute writes it into the indirect command).
//   * BZ (render/bindless.h): each survivor's texIndex selects its base color in the bindless array.
//   * CB (render/gpu_driven.h): the OUTPUT per-draw record is GpuDrivenPerDraw (model+material+texIndex)
//     — the EXACT 96-byte std430 layout lit_gpudriven.{vert,frag} read as PerDraw[gl_DrawID]. The
//     compute pass writes the SURVIVORS (compacted) into that SSBO, then ONE DrawIndexedMultiIndirect
//     (drawCount = survivor count) renders exactly them with one bindless bind.
//
// So the difference from CB (which laid out ALL objects unconditionally) is: the compute pass FILTERS
// to the in-frustum survivors and writes the GPU-decided drawCount — the GPU decides AND draws.
//
// RENDER-INVARIANCE (the crux): the showcase renders this same survivor set per-object BOUND (CPU
// frustum cull -> draw only the K survivors, each with its texture bound) and asserts the GPU-culled-
// and-drawn image is BYTE-IDENTICAL. Because the survivor SET (the conservative sphere test) + the
// per-draw values are a pure function of the scene + frustum, and the cull math is shared with the
// validated AR reference, the two images cannot diverge.
//
// DETERMINISM: survivors are emitted in source-index order (the ordered scan), texIndex is carried
// verbatim, no time/RNG — two builds are bit-identical (pinned by tests/gpu_culled_test.cpp).
//
// PURE CPU: model/material/texIndex are plain values; the local bounding sphere is plain floats; no
// ITexture* is dereferenced. Unit-testable with no GPU.

#include <cstdint>
#include <vector>

#include "math/math.h"
#include "render/frustum.h"
#include "render/gpu_cull.h"
#include "render/mdi.h"
#include "render/gpu_driven.h"

namespace hf::render::gpuculled {

// One scene object the compute cull consumes: its index-buffer slice (BM) + per-draw model matrix +
// material + bindless texIndex (CB) PLUS its LOCAL bounding sphere (the unit-mesh bound; the world
// sphere is derived from `model` exactly like the AR cull). instanceCount/firstInstance are fixed by
// the builder (1 / 0). This is the FULL per-draw list the GPU frustum-culls + compacts.
struct CulledObject {
    uint32_t indexCount   = 0;
    uint32_t firstIndex   = 0;
    uint32_t vertexOffset = 0;
    float    model[16]    = {};      // column-major mat4
    float    material[4]  = {};      // float4 material (metallic, roughness-tint, ...)
    uint32_t texIndex     = 0;       // bindless array index (BZ)
    math::Vec3 localCenter{0, 0, 0}; // local bounding-sphere center (unit-mesh space)
    float      localRadius = 0.0f;   // local bounding-sphere radius
};

// The compute pass's OUTPUT for ONE fully-GPU-driven-culled draw: the compacted survivor MDI commands
// (-> BufferUsage::Indirect; `drawCount` survivors), the compacted survivor per-draw records (->
// BufferUsage::Storage, indexed by gl_DrawID, carrying model+material+texIndex — the CB layout), and
// the GPU-decided survivor `drawCount` (== the count thread 0 writes into the indirect command, the
// DrawIndexedMultiIndirect drawCount). commands/perDraw are parallel: command j + perDraw[j] are the
// j-th survivor, in source-index order.
struct CulledBatch {
    std::vector<mdi::MdiCommand>            commands;   // survivor indirect-draw commands (compacted)
    std::vector<gpudriven::GpuDrivenPerDraw> perDraw;   // survivor per-draw records (compacted)
    uint32_t                                drawCount = 0;  // GPU-decided survivor count
};

// CULL + ORDERED COMPACT (the mirror of shaders/gpudriven_cull.comp.hlsl). Walk `objs` in SOURCE order;
// for each, derive its world bounding sphere from `model` + its local bound (render::gpu_cull::
// InstanceWorldSphere — the SAME center = model*localCenter, radius = localRadius*|col0| the AR cull
// uses) and test it with the conservative render::frustum::SphereOutside. KEEP each object whose sphere
// is NOT fully outside the frustum, APPENDING its MdiCommand + GpuDrivenPerDraw (model+material+texIndex)
// to the output in source-index order. drawCount == the number of survivors. Pure, deterministic, no GPU.
inline CulledBatch CullAndCompact(const std::vector<CulledObject>& objs, const frustum::Frustum& f) {
    CulledBatch out;
    out.commands.reserve(objs.size());
    out.perDraw.reserve(objs.size());
    for (const CulledObject& o : objs) {
        // World bounding sphere from the model matrix + local bound (shared AR math).
        math::Vec3 center;
        float      radius;
        gpu_cull::InstanceWorldSphere(o.model, o.localCenter, o.localRadius, center, radius);
        if (frustum::SphereOutside(f, center, radius)) continue;  // fully outside -> culled

        // Survivor: emit its MDI command (one draw; per-draw data indexed by gl_DrawID).
        mdi::MdiCommand c{};
        c.indexCount    = o.indexCount;
        c.instanceCount = 1u;
        c.firstIndex    = o.firstIndex;
        c.vertexOffset  = o.vertexOffset;
        c.firstInstance = 0u;
        out.commands.push_back(c);

        // Survivor: emit its combined per-draw record (model + material + texIndex), the CB layout.
        gpudriven::GpuDrivenPerDraw pd{};
        for (int k = 0; k < 16; ++k) pd.model[k]    = o.model[k];
        for (int k = 0; k < 4;  ++k) pd.material[k] = o.material[k];
        pd.texIndex = o.texIndex;
        pd.pad[0] = pd.pad[1] = pd.pad[2] = 0u;
        out.perDraw.push_back(pd);
    }
    out.drawCount = (uint32_t)out.commands.size();
    return out;
}

}  // namespace hf::render::gpuculled
