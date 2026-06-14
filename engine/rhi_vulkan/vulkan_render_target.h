#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"
#include "rhi_vulkan/vulkan_sampled.h"

namespace hf::rhi::vk {

class VulkanDevice;

// An offscreen color image (COLOR_ATTACHMENT | SAMPLED, swapchain format) plus its own depth
// image (D32). The color image has a material descriptor set so a later pass can sample it
// (ISampledVk). Tracks the color image's current layout for correct transitions.
class VulkanRenderTarget final : public IRenderTarget, public ISampledVk {
public:
    VulkanRenderTarget(VulkanDevice& device, uint32_t width, uint32_t height);
    ~VulkanRenderTarget() override;

    uint32_t width() const override { return width_; }
    uint32_t height() const override { return height_; }

    VkImage colorImage() const { return colorImage_; }
    VkImageView colorView() const { return colorView_; }
    VkImageView depthView() const { return depthView_; }
    VkImage depthImage() const { return depthImage_; }

    VkDescriptorSet descriptorSet() const { return set_; }
    VkDescriptorSet vkDescriptorSet() const override { return set_; }

    // Layout bookkeeping for the color image (driven by Begin/EndRenderTargetFrame).
    VkImageLayout colorLayout() const { return colorLayout_; }
    void setColorLayout(VkImageLayout l) { colorLayout_ = l; }

private:
    VulkanDevice& device_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;

    VkImage         colorImage_ = VK_NULL_HANDLE;
    VmaAllocation   colorAlloc_ = VK_NULL_HANDLE;
    VkImageView     colorView_  = VK_NULL_HANDLE;
    VkImageLayout   colorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage         depthImage_ = VK_NULL_HANDLE;
    VmaAllocation   depthAlloc_ = VK_NULL_HANDLE;
    VkImageView     depthView_  = VK_NULL_HANDLE;

    VkDescriptorSet set_ = VK_NULL_HANDLE;
};

} // namespace hf::rhi::vk
