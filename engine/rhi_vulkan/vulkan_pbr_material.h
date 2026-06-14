#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanDevice;

// A bound full-PBR material: one descriptor set (the wider set-1 layout) that points at five
// textures' image views — base-color, normal, metallic-roughness, emissive, occlusion — each with
// the device's default sampler. Built once from five rhi::ITexture (VulkanTexture) and bound per
// draw via ICommandBuffer::BindMaterialPBR. Owns only the descriptor set; the textures own their
// images/views and outlive this object (the caller keeps them alive).
class VulkanPbrMaterial {
public:
    VulkanPbrMaterial(VulkanDevice& device, ITexture& base, ITexture& metalRough,
                      ITexture& normalMap, ITexture& emissive, ITexture& occlusion);
    ~VulkanPbrMaterial();

    VkDescriptorSet descriptorSet() const { return set_; }

private:
    VulkanDevice& device_;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
};

} // namespace hf::rhi::vk
