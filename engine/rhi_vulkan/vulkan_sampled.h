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
};

} // namespace hf::rhi::vk
