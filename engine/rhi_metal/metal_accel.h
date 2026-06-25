#pragma once
#import <Metal/Metal.h>
#include "rhi/rhi.h"
#include <vector>

namespace hf::rhi::mtl {

// Slice METAL-RT S1 — Hardware Ray Tracing: a Metal acceleration structure (BLAS or TLAS) behind the
// RT1 IAccelStructure seam. The Metal twin of VulkanAccelStructure (rhi_vulkan/vulkan_accel.h). Owns the
// built id<MTLAccelerationStructure> + the input buffers/handles it must outlive:
//   * a BLAS retains its margin-inflated bounding-box buffer.
//   * a TLAS retains its instance-descriptor buffer AND the child BLAS handles (so they outlive it; the
//     dispatch must `useResource:` both the TLAS and every child BLAS — see MetalCommandBuffer::
//     BindAccelStructure).
//
// THE DETERMINISM CONTRACT (see shaders/rt_query.metal): the float HW BVH is ONLY a candidate-AABB
// GENERATOR — every BLAS AABB is INFLATED by kRtAabbMargin so the driver's FLOAT overlap test is a strict
// SUPERSET of every true fx hit; the kernel re-checks correctness in fx Q16.16 and owns the closest hit.
// So this build is allowed to be non-deterministic (only candidate COMPLETENESS matters).
//
// kRtAabbMargin: the fixed Q16.16 margin (1/64 world unit) added to each AABB half-extent, expressed as a
// float (the geometry is float AABBs) so it exactly matches the fx kOne/64 the host snaps from. Mirrors
// rhi_vulkan/vulkan_accel.h:23 and the showcase value at visual_test.mm:25624.
inline constexpr float kRtAabbMargin = 1.0f / 64.0f;

class MetalAccelStructure final : public IAccelStructure {
public:
    MetalAccelStructure() = default;
    ~MetalAccelStructure() override = default;  // ARC releases as_ / the retained buffers + child handles.

    // The built acceleration structure (a primitive AS for a BLAS, an instance AS for a true TLAS, or —
    // in the S1 degenerate simplification — the single child BLAS's primitive AS shared through).
    id<MTLAccelerationStructure> Handle() const { return as_; }

    // The child BLAS handles a TLAS references (empty for a BLAS). The dispatch must `useResource:` each
    // of these in addition to the TLAS itself, so the GPU traversal can read them.
    const std::vector<id<MTLAccelerationStructure>>& ChildHandles() const { return children_; }

    // --- builder-only setters (the device's CreateBlas/CreateTlas populate these) ---
    void SetHandle(id<MTLAccelerationStructure> as) { as_ = as; }
    void RetainBuffer(id<MTLBuffer> buf) { buffers_.push_back(buf); }
    void AddChild(id<MTLAccelerationStructure> child) { children_.push_back(child); }

private:
    id<MTLAccelerationStructure> as_ = nil;
    // Input buffers kept alive for the AS's lifetime (the bbox buffer for a BLAS, the instance-descriptor
    // buffer for a TLAS). ARC retains them via these strong refs.
    std::vector<id<MTLBuffer>> buffers_;
    // The child BLAS handles a TLAS references (kept alive so they outlive the TLAS). Empty for a BLAS.
    std::vector<id<MTLAccelerationStructure>> children_;
};

} // namespace hf::rhi::mtl
