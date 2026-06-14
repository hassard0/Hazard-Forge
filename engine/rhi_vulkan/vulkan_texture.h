#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"
#include "rhi_vulkan/vulkan_sampled.h"

namespace hf::rhi::vk {

class VulkanDevice;

// Device-local sampled image + view + a pre-baked descriptor set (combined image sampler).
class VulkanTexture final : public ITexture, public ISampledVk {
public:
    VulkanTexture(VulkanDevice& device, const TextureDesc& desc);
    ~VulkanTexture() override;
    VkDescriptorSet descriptorSet() const { return set_; }
    VkDescriptorSet vkDescriptorSet() const override { return set_; }
    VkImageView view() const { return view_; }

    // Point this material set's normal-map slot (binding 3/4) at `normalView` (+ the default
    // sampler). Cached: a no-op when re-bound with the same view, so the per-frame BindMaterial
    // call does not re-issue vkUpdateDescriptorSets every frame. Pass VK_NULL_HANDLE to reset to
    // the device's flat default normal.
    void attachNormalMap(VkImageView normalView);

private:
    VulkanDevice& device_;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation alloc_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkImageView boundNormalView_ = VK_NULL_HANDLE;  // last normal-map view written into the set
};

} // namespace hf::rhi::vk
