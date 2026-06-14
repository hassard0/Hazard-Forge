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

private:
    VmaAllocator allocator_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
};

} // namespace hf::rhi::vk
