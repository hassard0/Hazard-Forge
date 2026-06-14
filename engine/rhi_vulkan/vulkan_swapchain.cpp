#include "rhi_vulkan/vulkan_swapchain.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanSwapchain::VulkanSwapchain(VkDevice device, vkb::Device& vkbDevice, VmaAllocator allocator,
                                 uint32_t width, uint32_t height)
    : device_(device), vkbDevice_(vkbDevice), allocator_(allocator) {
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
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
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

    // --- Depth image (GPU-only, sized to swapchain extent) ---
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = depthFormat_;
    ici.extent = {extent_.width, extent_.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    Check(vmaCreateImage(allocator_, &ici, &aci, &depthImage_, &depthAllocation_, nullptr),
          "vmaCreateImage(depth)");

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = depthImage_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = depthFormat_;
    vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    Check(vkCreateImageView(device_, &vci, nullptr, &depthView_), "vkCreateImageView(depth)");
}

void VulkanSwapchain::Destroy() {
    if (depthView_) {
        vkDestroyImageView(device_, depthView_, nullptr);
        depthView_ = VK_NULL_HANDLE;
    }
    if (depthImage_) {
        vmaDestroyImage(allocator_, depthImage_, depthAllocation_);
        depthImage_ = VK_NULL_HANDLE;
        depthAllocation_ = VK_NULL_HANDLE;
    }
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
