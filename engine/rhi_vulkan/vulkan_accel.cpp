// Slice RT2 — Hardware Ray Tracing: the Vulkan acceleration-structure backend behind the RT1
// IAccelStructure seam (VulkanDevice::CreateBlas / CreateTlas + VulkanAccelStructure). Builds a
// procedural-AABB BLAS (each AABB inflated by kRtAabbMargin per the determinism contract) and a TLAS of
// those instances synchronously (one-shot command buffer + vkQueueWaitIdle — the staging-upload pattern),
// then stores the VkAccelerationStructureKHR + its device address for the inline-ray-query kernel.
//
// SEAM: reached ONLY by the new --rt2-query-shot showcase; every existing path is byte-for-byte unaffected
// (CreateBlas/CreateTlas were defaulted-no-op in RT1 and no existing pipeline binds an accel structure).
#include "rhi_vulkan/vulkan_accel.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace hf::rhi::vk {

namespace {

// A small RAII-free scratch/storage buffer with a device address, allocated device-local via VMA. The
// accel-struct STORAGE buffer is kept alive by VulkanAccelStructure; scratch / instance buffers are
// transient (freed after the synchronous build). Returns the VkBuffer + allocation; `outAddr` (if non-null)
// receives the buffer's device address.
struct DevBuffer { VkBuffer buf = VK_NULL_HANDLE; VmaAllocation alloc = VK_NULL_HANDLE; void* mapped = nullptr; };

DevBuffer CreateDeviceBuffer(VmaAllocator allocator, VkDevice device, VkDeviceSize size,
                             VkBufferUsageFlags usage, bool hostVisible, VkDeviceAddress* outAddr) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (hostVisible) {
        // The TLAS instance buffer is written by the host (one VkAccelerationStructureInstanceKHR per
        // instance) then read by the build on the device.
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    DevBuffer out;
    VmaAllocationInfo info{};
    Check(vmaCreateBuffer(allocator, &bci, &aci, &out.buf, &out.alloc, &info), "vmaCreateBuffer(accel)");
    out.mapped = info.pMappedData;

    if (outAddr) {
        VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        bai.buffer = out.buf;
        *outAddr = vkGetBufferDeviceAddress(device, &bai);
    }
    return out;
}

}  // namespace

VulkanAccelStructure::VulkanAccelStructure(VulkanDevice& device, VkAccelerationStructureKHR accel,
                                           VkBuffer buffer, VmaAllocation alloc, VkDeviceAddress address)
    : device_(device), accel_(accel), buffer_(buffer), alloc_(alloc), address_(address) {}

VulkanAccelStructure::~VulkanAccelStructure() {
    device_.freeAccelGraphicsSet(graphicsSet_);  // no-op if never bound to a graphics pipeline
    if (accel_ && device_.destroyAccelStructFn())
        device_.destroyAccelStructFn()(device_.device(), accel_, nullptr);
    if (buffer_) vmaDestroyBuffer(device_.allocator(), buffer_, alloc_);
}

VkDescriptorSet VulkanAccelStructure::graphicsSet(uint32_t slot) {
    // Lazily allocate + write the graphics TLAS set once; re-binds reuse the immutable cached set.
    if (graphicsSet_ == VK_NULL_HANDLE)
        graphicsSet_ = device_.allocateAccelGraphicsSet(accel_, slot);
    return graphicsSet_;
}

// --- CreateBlas: a procedural-AABB bottom-level accel structure -------------------------------------
std::unique_ptr<IAccelStructure> VulkanDevice::CreateBlas(const BlasDesc& desc) {
    if (!hwRayQuery_) return nullptr;

    // 1) Build the float VkAabbPositionsKHR array — each AABB INFLATED by kRtAabbMargin so the driver's
    //    float BVH overlap is a strict SUPERSET of every true fx hit (C1; the candidate-completeness
    //    guarantee). Triangle geoms are out of scope (Tier B / RT6) — skipped.
    std::vector<VkAabbPositionsKHR> aabbs;
    aabbs.reserve(desc.geoms.size());
    for (const AccelGeometry& g : desc.geoms) {
        if (g.kind != AccelGeometry::Kind::AabbProcedural) continue;
        VkAabbPositionsKHR a{};
        a.minX = g.lo.x - kRtAabbMargin; a.minY = g.lo.y - kRtAabbMargin; a.minZ = g.lo.z - kRtAabbMargin;
        a.maxX = g.hi.x + kRtAabbMargin; a.maxY = g.hi.y + kRtAabbMargin; a.maxZ = g.hi.z + kRtAabbMargin;
        aabbs.push_back(a);
    }
    if (aabbs.empty()) return nullptr;
    const uint32_t primCount = (uint32_t)aabbs.size();

    // 2) Upload the AABB data to a device-address buffer (the geometry's vertexData/aabbData pointer).
    VkDeviceAddress aabbAddr = 0;
    DevBuffer aabbBuf = CreateDeviceBuffer(
        allocator_, device_, primCount * sizeof(VkAabbPositionsKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, /*hostVisible*/ true, &aabbAddr);
    std::memcpy(aabbBuf.mapped, aabbs.data(), primCount * sizeof(VkAabbPositionsKHR));
    vmaFlushAllocation(allocator_, aabbBuf.alloc, 0, VK_WHOLE_SIZE);

    // 3) Describe the AABB geometry + query the build sizes.
    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;  // opaque AABBs (no any-hit); we drain candidates ourselves
    geom.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geom.geometry.aabbs.data.deviceAddress = aabbAddr;
    geom.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

    VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bgi.geometryCount = 1;
    bgi.pGeometries = &geom;

    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR_(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                             &bgi, &primCount, &sizes);

    // 4) Allocate the accel-struct STORAGE buffer (kept alive by VulkanAccelStructure) + a transient
    //    SCRATCH buffer, both device-address.
    DevBuffer accelBuf = CreateDeviceBuffer(
        allocator_, device_, sizes.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, /*hostVisible*/ false, nullptr);
    VkDeviceAddress scratchAddr = 0;
    DevBuffer scratchBuf = CreateDeviceBuffer(
        allocator_, device_, sizes.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, /*hostVisible*/ false, &scratchAddr);

    // 5) Create the (empty) accel structure over the storage buffer.
    VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    aci.buffer = accelBuf.buf;
    aci.size = sizes.accelerationStructureSize;
    aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
    Check(vkCreateAccelerationStructureKHR_(device_, &aci, nullptr, &accel), "vkCreateAccelerationStructureKHR(blas)");

    bgi.dstAccelerationStructure = accel;
    bgi.scratchData.deviceAddress = scratchAddr;

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = primCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

    // 6) Record + submit the build on the one-shot rt command buffer, then WAIT (synchronous load path).
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(rtCmd_, &cbi), "vkBeginCommandBuffer(blas)");
    vkCmdBuildAccelerationStructuresKHR_(rtCmd_, 1, &bgi, &pRange);
    Check(vkEndCommandBuffer(rtCmd_), "vkEndCommandBuffer(blas)");

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &rtCmd_;
    Check(vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit(blas)");
    Check(vkQueueWaitIdle(graphicsQueue_), "vkQueueWaitIdle(blas)");
    vkResetCommandBuffer(rtCmd_, 0);

    // The accel device address (for a TLAS instance to reference this BLAS).
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addrInfo.accelerationStructure = accel;
    VkDeviceAddress accelAddr = vkGetAccelerationStructureDeviceAddressKHR_(device_, &addrInfo);

    // Free the transient build inputs (the BLAS is built; the accel buffer holds the result).
    vmaDestroyBuffer(allocator_, scratchBuf.buf, scratchBuf.alloc);
    vmaDestroyBuffer(allocator_, aabbBuf.buf, aabbBuf.alloc);

    return std::make_unique<VulkanAccelStructure>(*this, accel, accelBuf.buf, accelBuf.alloc, accelAddr);
}

// --- CreateTlas: a top-level accel structure of BLAS instances --------------------------------------
std::unique_ptr<IAccelStructure> VulkanDevice::CreateTlas(const TlasDesc& desc) {
    if (!hwRayQuery_) return nullptr;
    if (desc.instances.empty()) return nullptr;

    // 1) One VkAccelerationStructureInstanceKHR per instance: instanceCustomIndex = instanceId; the BLAS
    //    referenced by its device address (RT2 uses identity transforms; transform[12] is row-major 3x4).
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(desc.instances.size());
    for (const TlasInstance& ti : desc.instances) {
        if (!ti.blas) continue;
        auto* blas = static_cast<VulkanAccelStructure*>(ti.blas);
        VkAccelerationStructureInstanceKHR inst{};
        std::memcpy(&inst.transform.matrix[0][0], ti.transform, 12 * sizeof(float));  // row-major 3x4
        inst.instanceCustomIndex = ti.instanceId & 0xFFFFFFu;  // 24-bit field
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas->deviceAddress();
        instances.push_back(inst);
    }
    if (instances.empty()) return nullptr;
    const uint32_t instCount = (uint32_t)instances.size();

    // 2) Upload the instance array to a device-address buffer.
    VkDeviceAddress instAddr = 0;
    DevBuffer instBuf = CreateDeviceBuffer(
        allocator_, device_, instCount * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, /*hostVisible*/ true, &instAddr);
    std::memcpy(instBuf.mapped, instances.data(), instCount * sizeof(VkAccelerationStructureInstanceKHR));
    vmaFlushAllocation(allocator_, instBuf.alloc, 0, VK_WHOLE_SIZE);

    // 3) Describe the instances geometry + query the build sizes.
    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geom.geometry.instances.arrayOfPointers = VK_FALSE;
    geom.geometry.instances.data.deviceAddress = instAddr;

    VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bgi.geometryCount = 1;
    bgi.pGeometries = &geom;

    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR_(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                             &bgi, &instCount, &sizes);

    DevBuffer accelBuf = CreateDeviceBuffer(
        allocator_, device_, sizes.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, /*hostVisible*/ false, nullptr);
    VkDeviceAddress scratchAddr = 0;
    DevBuffer scratchBuf = CreateDeviceBuffer(
        allocator_, device_, sizes.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, /*hostVisible*/ false, &scratchAddr);

    VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    aci.buffer = accelBuf.buf;
    aci.size = sizes.accelerationStructureSize;
    aci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
    Check(vkCreateAccelerationStructureKHR_(device_, &aci, nullptr, &accel), "vkCreateAccelerationStructureKHR(tlas)");

    bgi.dstAccelerationStructure = accel;
    bgi.scratchData.deviceAddress = scratchAddr;

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = instCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(rtCmd_, &cbi), "vkBeginCommandBuffer(tlas)");
    vkCmdBuildAccelerationStructuresKHR_(rtCmd_, 1, &bgi, &pRange);
    Check(vkEndCommandBuffer(rtCmd_), "vkEndCommandBuffer(tlas)");

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &rtCmd_;
    Check(vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit(tlas)");
    Check(vkQueueWaitIdle(graphicsQueue_), "vkQueueWaitIdle(tlas)");
    vkResetCommandBuffer(rtCmd_, 0);

    vmaDestroyBuffer(allocator_, scratchBuf.buf, scratchBuf.alloc);
    vmaDestroyBuffer(allocator_, instBuf.buf, instBuf.alloc);

    // A TLAS is referenced by descriptor (not address), so the stored address is 0.
    return std::make_unique<VulkanAccelStructure>(*this, accel, accelBuf.buf, accelBuf.alloc, 0);
}

} // namespace hf::rhi::vk
