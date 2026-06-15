#include "rhi_vulkan/vulkan_buffer.h"
#include "rhi_vulkan/vk_common.h"
#include <cstring>

namespace hf::rhi::vk {

VulkanBuffer::VulkanBuffer(VmaAllocator allocator, const BufferDesc& desc)
    : allocator_(allocator) {
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    // Slice AR: a Storage/Indirect buffer is also host-READABLE (RANDOM access) so the host can read
    // back the compute-written GPU-cull survivor count; the others stay write-only (SEQUENTIAL).
    bool hostReadable = false;
    switch (desc.usage) {
        case BufferUsage::Vertex:  usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;  break;
        case BufferUsage::Index:   usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;   break;
        case BufferUsage::Uniform: usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT; break;
        // A storage (SSBO) buffer is also bindable as a vertex stream, so the graphics pass can draw
        // the compute-written particle/survivor data directly (no copy). TRANSFER_DST allows an init
        // upload.
        case BufferUsage::Storage:
            usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            hostReadable = true;
            break;
        // Indirect (Slice AR): the GPU-cull compute shader WRITES the draw-args here (so it is also a
        // storage buffer + index-readable count), and DrawIndexedIndirect READS them. The host reads
        // back the instanceCount for the exact-count proof, so it is host-readable too.
        case BufferUsage::Indirect:
            usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            hostReadable = true;
            break;
    }

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = desc.size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = (hostReadable ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                              : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    Check(vmaCreateBuffer(allocator_, &bci, &aci, &buffer_, &allocation_, &info),
          "vmaCreateBuffer");
    mapped_ = info.pMappedData;

    if (desc.initialData && info.pMappedData) {
        std::memcpy(info.pMappedData, desc.initialData, desc.size);
        vmaFlushAllocation(allocator_, allocation_, 0, desc.size);
    }
}

VulkanBuffer::~VulkanBuffer() {
    if (buffer_) vmaDestroyBuffer(allocator_, buffer_, allocation_);
}

} // namespace hf::rhi::vk
