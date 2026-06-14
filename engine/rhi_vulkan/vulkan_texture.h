#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanDevice;

// Device-local sampled image + view + a pre-baked descriptor set (combined image sampler).
class VulkanTexture final : public ITexture {
public:
    VulkanTexture(VulkanDevice& device, const TextureDesc& desc);
    ~VulkanTexture() override;
    VkDescriptorSet descriptorSet() const { return set_; }

private:
    VulkanDevice& device_;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation alloc_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
};

} // namespace hf::rhi::vk
