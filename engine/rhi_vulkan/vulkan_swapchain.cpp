#include "rhi_vulkan/vulkan_swapchain.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanSwapchain::VulkanSwapchain(VkDevice device, vkb::Device& vkbDevice,
                                 uint32_t width, uint32_t height)
    : device_(device), vkbDevice_(vkbDevice) {
    Build(width, height);
}

VulkanSwapchain::~VulkanSwapchain() { Destroy(); }

void VulkanSwapchain::Build(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder builder{vkbDevice_};
    auto ret = builder
        .set_desired_format(VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM,
                                               VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        .build();
    if (!ret) {
        throw std::runtime_error("Swapchain build failed: " + ret.error().message());
    }
    vkb::Swapchain vkbSwap = ret.value();
    swapchain_ = vkbSwap.swapchain;
    format_ = vkbSwap.image_format;
    extent_ = vkbSwap.extent;
    images_ = vkbSwap.get_images().value();
    views_ = vkbSwap.get_image_views().value();
}

void VulkanSwapchain::Destroy() {
    for (auto v : views_) vkDestroyImageView(device_, v, nullptr);
    views_.clear();
    images_.clear();
    if (swapchain_) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VulkanSwapchain::Recreate(uint32_t width, uint32_t height) {
    vkDeviceWaitIdle(device_);
    Destroy();
    Build(width, height);
}

} // namespace hf::rhi::vk
