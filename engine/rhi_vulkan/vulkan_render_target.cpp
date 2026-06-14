#include "rhi_vulkan/vulkan_render_target.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanRenderTarget::VulkanRenderTarget(VulkanDevice& device, uint32_t width, uint32_t height)
    : device_(device), width_(width), height_(height) {
    // Color image: same format as the swapchain so the existing lit pipeline renders into it
    // unchanged. Usage: render target + sampled (so the post pass can read it).
    const VkFormat colorFormat = device_.swapchainFormat();
    {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = colorFormat;
        ici.extent = {width_, height_, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        Check(vmaCreateImage(device_.allocator(), &ici, &aci, &colorImage_, &colorAlloc_, nullptr),
              "vmaCreateImage(rt color)");

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = colorImage_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = colorFormat;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        Check(vkCreateImageView(device_.device(), &vci, nullptr, &colorView_),
              "vkCreateImageView(rt color)");
    }

    // Depth image: D32, the RT owns it (matches the depth format the pipelines expect).
    {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_D32_SFLOAT;
        ici.extent = {width_, height_, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        Check(vmaCreateImage(device_.allocator(), &ici, &aci, &depthImage_, &depthAlloc_, nullptr),
              "vmaCreateImage(rt depth)");

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = depthImage_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_D32_SFLOAT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        Check(vkCreateImageView(device_.device(), &vci, nullptr, &depthView_),
              "vkCreateImageView(rt depth)");
    }

    // Descriptor set on the material set layout (set 1) so the post pass can BindTexture(*this).
    // Same two-write pattern as VulkanTexture: binding 0 sampled image, binding 1 sampler.
    VkDescriptorSetLayout layout = device_.materialSetLayout();
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = device_.descriptorPool();
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &layout;
    Check(vkAllocateDescriptorSets(device_.device(), &dai, &set_),
          "vkAllocateDescriptorSets(rt)");

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = colorView_;
    // The set is read while the color image is in SHADER_READ_ONLY_OPTIMAL (after the RT pass).
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = device_.defaultSampler();

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[0].pImageInfo = &imageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[1].pImageInfo = &samplerInfo;

    vkUpdateDescriptorSets(device_.device(), 2, writes, 0, nullptr);
}

VulkanRenderTarget::~VulkanRenderTarget() {
    if (set_) vkFreeDescriptorSets(device_.device(), device_.descriptorPool(), 1, &set_);
    if (colorView_) vkDestroyImageView(device_.device(), colorView_, nullptr);
    if (depthView_) vkDestroyImageView(device_.device(), depthView_, nullptr);
    if (colorImage_) vmaDestroyImage(device_.allocator(), colorImage_, colorAlloc_);
    if (depthImage_) vmaDestroyImage(device_.allocator(), depthImage_, depthAlloc_);
}

} // namespace hf::rhi::vk
