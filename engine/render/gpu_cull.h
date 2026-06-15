#pragma once
// Slice AR — GPU-driven culling CPU MIRROR. Pure CPU (header-only, no device, no backend symbols).
// This header is the AUTHORITATIVE reference for the EXACT same ordered-compaction + frustum-sphere
// logic that shaders/cull.comp.hlsl implements on the GPU. It is shared by THREE call sites:
//   1. tests/gpu_cull_test.cpp — asserts the ordered compaction matches a brute-force reference IN
//      ORDER (pins the determinism contract without a GPU).
//   2. samples/hello_triangle/main.cpp (--gpu-cull-shot, Vulkan) — computes the CPU reference count
//      the GPU survivor count is asserted against (the exact-count proof).
//   3. metal_headless/visual_test.mm (--gpu-cull, Metal) — same CPU reference on Metal.
//
// DETERMINISM (the crux of the golden): survivors are emitted in SOURCE-INDEX ORDER. The GPU shader
// achieves this with a single-workgroup PREFIX SUM (grid is <=1024 instances = one workgroup), NOT
// an unordered atomicAdd append (whose result order is nondeterministic and would break the image
// golden). This mirror writes survivor[outIdx] where outIdx is the COUNT of earlier survivors — the
// stable scan — so the two orderings are identical by construction.
//
// BOUNDING SPHERE (must match the shader exactly): each instance is placed by a column-major model
// matrix (the InstanceData mat4 stream). Its world bounding sphere is
//     center = model * float4(localCenter, 1)
//     radius = localRadius * length(model column 0)        // uniform-scale: |col0| == scale
// and the cull decision is the CONSERVATIVE bounding-sphere test from engine/render/frustum.h
// (SphereOutside): culled IFF the sphere is fully outside at least one plane. The shader extracts
// the same |col0| scale and uses the same six planes (uploaded as float4 {n.xyz, d}); a divergence
// surfaces loudly in the exact-count proof.

#include "math/math.h"
#include "render/frustum.h"

#include <cstdint>
#include <vector>

namespace hf::render::gpu_cull {

// The world bounding sphere of one instance given its column-major model matrix + the shared local
// bound (localCenter/localRadius of the unit mesh). Pure function; the shader computes the identical
// values from the same four model columns.
inline void InstanceWorldSphere(const float model[16], const math::Vec3& localCenter,
                                float localRadius, math::Vec3& outCenter, float& outRadius) {
    math::Mat4 m;
    for (int k = 0; k < 16; ++k) m.m[k] = model[k];
    outCenter = math::MulPoint(m, localCenter);
    // Uniform-scale assumption (our grids are uniform): the world radius scales by the length of the
    // model's first column (== the uniform scale factor). Matches the shader's length(col0).
    float col0Len = std::sqrt(model[0] * model[0] + model[1] * model[1] + model[2] * model[2]);
    outRadius = localRadius * col0Len;
}

// ORDERED single-pass compaction reference: walk the instances in source order; KEEP each instance
// whose world sphere is NOT fully outside the frustum, appending its model matrix to `outSurvivors`
// in source-index order. Returns the survivor COUNT (== outSurvivors.size()). This is exactly what
// shaders/cull.comp.hlsl produces via its single-workgroup prefix sum.
inline uint32_t CompactSurvivors(const std::vector<float>& instanceModels,  // 16 floats per instance
                                 uint32_t instanceCount,
                                 const frustum::Frustum& f,
                                 const math::Vec3& localCenter, float localRadius,
                                 std::vector<float>& outSurvivors) {
    outSurvivors.clear();
    outSurvivors.reserve((size_t)instanceCount * 16);
    uint32_t count = 0;
    for (uint32_t i = 0; i < instanceCount; ++i) {
        const float* model = instanceModels.data() + (size_t)i * 16;
        math::Vec3 center;
        float radius;
        InstanceWorldSphere(model, localCenter, localRadius, center, radius);
        if (!frustum::SphereOutside(f, center, radius)) {
            for (int k = 0; k < 16; ++k) outSurvivors.push_back(model[k]);
            ++count;
        }
    }
    return count;
}

// Convenience: just the survivor COUNT (the exact-count proof only needs the integer). Same scan.
inline uint32_t SurvivorCount(const std::vector<float>& instanceModels, uint32_t instanceCount,
                              const frustum::Frustum& f, const math::Vec3& localCenter,
                              float localRadius) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < instanceCount; ++i) {
        const float* model = instanceModels.data() + (size_t)i * 16;
        math::Vec3 center;
        float radius;
        InstanceWorldSphere(model, localCenter, localRadius, center, radius);
        if (!frustum::SphereOutside(f, center, radius)) ++count;
    }
    return count;
}

}  // namespace hf::render::gpu_cull
