#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanBuffer final : public IBuffer {
public:
    VulkanBuffer(VmaAllocator allocator, const BufferDesc& desc);
    ~VulkanBuffer() override;
    VkBuffer handle() const { return buffer_; }
    VmaAllocator allocator() const { return allocator_; }
    VmaAllocation allocation() const { return allocation_; }
    // Persistently-mapped host pointer (HOST_ACCESS_* | MAPPED on create); null if unmapped. Used by
    // VulkanDevice::ReadBuffer to read back the GPU-cull survivor count (Slice AR).
    void* mappedData() const { return mapped_; }

private:
    VmaAllocator allocator_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
};

} // namespace hf::rhi::vk
