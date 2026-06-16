#include "rhi_vulkan/vulkan_cubemap_target.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanCubemapTarget::VulkanCubemapTarget(VulkanDevice& device, uint32_t size, Format colorFormatReq)
    : device_(device), size_(size) {
    const VkFormat colorFormat = (colorFormatReq == Format::Undefined)
                                     ? device_.swapchainFormat()
                                     : ToVk(colorFormatReq);
    colorFormat_ = colorFormat;

    // --- Cube COLOR image: 6 array layers, CUBE_COMPATIBLE so a cube view can sample it. Usage:
    //     COLOR_ATTACHMENT (rendered per face) | SAMPLED (sampled in the reflection pass) |
    //     TRANSFER_SRC (per-face readback for the capture==direct-render proof). ---
    {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = colorFormat;
        ici.extent = {size_, size_, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 6;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        Check(vmaCreateImage(device_.allocator(), &ici, &aci, &colorImage_, &colorAlloc_, nullptr),
              "vmaCreateImage(cube color)");

        // Per-face single-layer 2D views (the render attachments, one per cube face).
        for (uint32_t f = 0; f < 6; ++f) {
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = colorImage_;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = colorFormat;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, f, 1};
            Check(vkCreateImageView(device_.device(), &vci, nullptr, &faceViews_[f]),
                  "vkCreateImageView(cube face)");
        }

        // The cube sampling view (all 6 layers, VK_IMAGE_VIEW_TYPE_CUBE).
        VkImageViewCreateInfo cvi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        cvi.image = colorImage_;
        cvi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        cvi.format = colorFormat;
        cvi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
        Check(vkCreateImageView(device_.device(), &cvi, nullptr, &cubeView_),
              "vkCreateImageView(cube)");
    }

    // --- Shared single-layer DEPTH image (D32), re-cleared per face. ---
    {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_D32_SFLOAT;
        ici.extent = {size_, size_, 1};
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
              "vmaCreateImage(cube depth)");

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = depthImage_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_D32_SFLOAT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        Check(vkCreateImageView(device_.device(), &vci, nullptr, &depthView_),
              "vkCreateImageView(cube depth)");
    }

    // --- Environment descriptor set (set 3, binding 11 = cube view, 12 = env sampler) so the cube
    //     binds at the env slot via BindCubemapProbe — mirrors VulkanRenderTarget::environmentSet but
    //     for a CUBE view. Allocated up front (the cube is always sampled). ---
    {
        VkDescriptorSetLayout layout = device_.environmentSetLayout();
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = device_.descriptorPool();
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &layout;
        Check(vkAllocateDescriptorSets(device_.device(), &dai, &environmentSet_),
              "vkAllocateDescriptorSets(cube env)");

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = cubeView_;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo samplerInfo{};
        samplerInfo.sampler = device_.environmentSampler();

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = environmentSet_;
        writes[0].dstBinding = 11;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[0].pImageInfo = &imageInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = environmentSet_;
        writes[1].dstBinding = 12;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[1].pImageInfo = &samplerInfo;
        vkUpdateDescriptorSets(device_.device(), 2, writes, 0, nullptr);
    }
}

VulkanCubemapTarget::~VulkanCubemapTarget() {
    if (environmentSet_)
        vkFreeDescriptorSets(device_.device(), device_.descriptorPool(), 1, &environmentSet_);
    if (cubeView_) vkDestroyImageView(device_.device(), cubeView_, nullptr);
    for (VkImageView v : faceViews_)
        if (v) vkDestroyImageView(device_.device(), v, nullptr);
    if (depthView_) vkDestroyImageView(device_.device(), depthView_, nullptr);
    if (colorImage_) vmaDestroyImage(device_.allocator(), colorImage_, colorAlloc_);
    if (depthImage_) vmaDestroyImage(device_.allocator(), depthImage_, depthAlloc_);
}

} // namespace hf::rhi::vk
