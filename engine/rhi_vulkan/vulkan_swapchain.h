#pragma once
#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include "rhi/rhi.h"
#include "rhi_vulkan/vk_common.h"
#include <vector>

namespace hf::rhi::vk {

class VulkanSwapchain final : public ISwapchain {
public:
    VulkanSwapchain(VkDevice device, vkb::Device& vkbDevice,
                    uint32_t width, uint32_t height);
    ~VulkanSwapchain() override;

    Format ColorFormat() const override { return FromVk(format_); }
    void Recreate(uint32_t width, uint32_t height) override;

    VkSwapchainKHR handle() const { return swapchain_; }
    VkFormat vkFormat() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    VkImage image(uint32_t i) const { return images_[i]; }
    VkImageView view(uint32_t i) const { return views_[i]; }
    uint32_t imageCount() const { return (uint32_t)images_.size(); }

private:
    void Build(uint32_t width, uint32_t height);
    void Destroy();

    VkDevice device_;
    vkb::Device& vkbDevice_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
};

} // namespace hf::rhi::vk
