#include "rhi_vulkan/vulkan_buffer.h"
#include "rhi_vulkan/vk_common.h"
#include <cstring>

namespace hf::rhi::vk {

VulkanBuffer::VulkanBuffer(VmaAllocator allocator, const BufferDesc& desc)
    : allocator_(allocator) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = desc.size;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    Check(vmaCreateBuffer(allocator_, &bci, &aci, &buffer_, &allocation_, &info),
          "vmaCreateBuffer");

    if (desc.initialData && info.pMappedData) {
        std::memcpy(info.pMappedData, desc.initialData, desc.size);
        vmaFlushAllocation(allocator_, allocation_, 0, desc.size);
    }
}

VulkanBuffer::~VulkanBuffer() {
    if (buffer_) vmaDestroyBuffer(allocator_, buffer_, allocation_);
}

} // namespace hf::rhi::vk
