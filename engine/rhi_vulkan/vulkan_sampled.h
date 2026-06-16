#pragma once
#include <vulkan/vulkan.h>

namespace hf::rhi::vk {

// Backend-internal interface for anything that can be bound as a sampled material texture.
// Both VulkanTexture and VulkanRenderTarget implement it, so VulkanCommandBuffer::BindTexture
// can resolve the material descriptor set for either via dynamic_cast<ISampledVk*>.
class ISampledVk {
public:
    virtual ~ISampledVk() = default;
    virtual VkDescriptorSet vkDescriptorSet() const = 0;
    // The sampled image VIEW, so the bindless texture array (Slice BZ) can fill its runtime
    // sampled-image descriptor array from any ISampledVk (VulkanTexture or a sampled render target).
    virtual VkImageView vkImageView() const = 0;
};

} // namespace hf::rhi::vk
