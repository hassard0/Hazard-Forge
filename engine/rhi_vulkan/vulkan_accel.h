#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanDevice;

// Slice RT2 — Hardware Ray Tracing: a Vulkan acceleration structure (BLAS or TLAS) behind the RT1
// IAccelStructure seam. Owns the VkAccelerationStructureKHR + its backing accel-struct buffer (VMA
// device-local) and exposes the structure's GPU device address (for a TLAS instance reference) and the
// handle itself (for the inline-ray-query descriptor write). Procedural-AABB geometry only (no triangles
// — Tier B / RT6); identity instance transforms in RT2.
//
// THE DETERMINISM CONTRACT (see rt_query.comp): the HW BVH is ONLY a candidate-AABB generator — every
// BLAS AABB is INFLATED by kRtAabbMargin so the driver's FLOAT overlap test is a strict SUPERSET of every
// true fx hit; the kernel re-checks correctness in fx and owns the closest-hit. So this build is allowed
// to be non-deterministic (only candidate COMPLETENESS matters).
//
// kRtAabbMargin: the fixed Q16.16 margin (1/64 world unit) added to each AABB half-extent. Expressed here
// as a float (the BLAS geometry is float AABBs) so it exactly matches the fx kOne/64 the host snaps from.
inline constexpr float kRtAabbMargin = 1.0f / 64.0f;

class VulkanAccelStructure final : public IAccelStructure {
public:
    // Takes ownership of a built accel structure + its backing buffer. `address` is the device address used
    // by a TLAS to reference this BLAS (0 for a TLAS — a TLAS is referenced by descriptor, not address).
    VulkanAccelStructure(VulkanDevice& device, VkAccelerationStructureKHR accel,
                         VkBuffer buffer, VmaAllocation alloc, VkDeviceAddress address);
    ~VulkanAccelStructure() override;

    VkAccelerationStructureKHR handle() const { return accel_; }
    VkDeviceAddress deviceAddress() const { return address_; }

    // Issue #34 / fix-rhi-binding13: the REGULAR descriptor set holding this TLAS for the GRAPHICS
    // (fragment RayQuery) path, lazily allocated + written on first bind and cached here (an immutable
    // set — never updated while bound in a recording command buffer). Replaces the former push-descriptor
    // accel set, which made the RT-graphics pipeline layout carry TWO push-descriptor set layouts
    // (VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293) and clobbered cluster binding 13. Freed back
    // to the device's accel pool in the dtor.
    VkDescriptorSet graphicsSet(uint32_t slot);

private:
    VulkanDevice& device_;
    VkAccelerationStructureKHR accel_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation alloc_ = VK_NULL_HANDLE;
    VkDeviceAddress address_ = 0;
    VkDescriptorSet graphicsSet_ = VK_NULL_HANDLE;  // cached RT-graphics TLAS set (fix-rhi-binding13)
};

} // namespace hf::rhi::vk
