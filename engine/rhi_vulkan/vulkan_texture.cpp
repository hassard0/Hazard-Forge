#include "rhi_vulkan/vulkan_texture.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"
#include <vector>

namespace hf::rhi::vk {

VulkanTexture::VulkanTexture(VulkanDevice& device, const TextureDesc& desc)
    : device_(device) {
    VkFormat format = ToVk(desc.format);
    const uint32_t mipLevels = desc.mipLevels > 0 ? desc.mipLevels : 1;

    // Device-local sampled image (optionally N mip levels for the HDR environment map).
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {desc.width, desc.height, 1};
    ici.mipLevels = mipLevels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    Check(vmaCreateImage(device_.allocator(), &ici, &aci, &image_, &alloc_, nullptr),
          "vmaCreateImage");

    // Stage the pixels in: single-mip (existing path) or N mips (HDR env, Slice R).
    if (mipLevels > 1 && desc.mipData) {
        std::vector<uint64_t> mipBytes(mipLevels);
        // Element size from the format (RGBA16F = 8 bytes/texel, RGBA8 = 4).
        const uint64_t bpp = (desc.format == Format::RGBA16_Float) ? 8u : 4u;
        for (uint32_t i = 0; i < mipLevels; ++i) {
            uint32_t mw = desc.width >> i; if (mw == 0) mw = 1;
            uint32_t mh = desc.height >> i; if (mh == 0) mh = 1;
            mipBytes[i] = (uint64_t)mw * mh * bpp;
        }
        device_.UploadToImageMips(image_, desc.width, desc.height, mipLevels,
                                  desc.mipData, mipBytes.data());
    } else {
        device_.UploadToImage(image_, desc.width, desc.height, desc.data, desc.dataSize);
    }

    // View (covers all mip levels).
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};
    Check(vkCreateImageView(device_.device(), &vci, nullptr, &view_), "vkCreateImageView");

    // --- HDR environment texture (Slice R): only the dedicated environment set (set 3), NOT the
    // material set. The equirect env map is never bound as a material, so building the 2-texture
    // material set would just waste descriptors. ---
    if (desc.environment) {
        VkDescriptorSetLayout envLayout = device_.environmentSetLayout();
        VkDescriptorSetAllocateInfo edai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        edai.descriptorPool = device_.descriptorPool();
        edai.descriptorSetCount = 1;
        edai.pSetLayouts = &envLayout;
        Check(vkAllocateDescriptorSets(device_.device(), &edai, &environmentSet_),
              "vkAllocateDescriptorSets(env)");

        VkDescriptorImageInfo envImage{};
        envImage.imageView = view_;
        envImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo envSampler{};
        envSampler.sampler = device_.environmentSampler();
        VkWriteDescriptorSet ew[2]{};
        ew[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ew[0].dstSet = environmentSet_;
        ew[0].dstBinding = 11;  // matches environmentSetLayout_ + the HLSL [[vk::binding(11,3)]]
        ew[0].descriptorCount = 1;
        ew[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        ew[0].pImageInfo = &envImage;
        ew[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ew[1].dstSet = environmentSet_;
        ew[1].dstBinding = 12;  // matches environmentSetLayout_ + the HLSL [[vk::binding(12,3)]]
        ew[1].descriptorCount = 1;
        ew[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        ew[1].pImageInfo = &envSampler;
        vkUpdateDescriptorSets(device_.device(), 2, ew, 0, nullptr);
        return;  // no material set for an env texture
    }

    // Descriptor set: allocate from the device pool with the shared layout, then update.
    VkDescriptorSetLayout layout = device_.texturedSetLayout();
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = device_.descriptorPool();
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &layout;
    Check(vkAllocateDescriptorSets(device_.device(), &dai, &set_), "vkAllocateDescriptorSets");

    // binding 0 = base-color sampled image (the view), binding 1 = base sampler. DXC emits the
    // Texture2D and SamplerState as two separate descriptors, not one combined. binding 3/4 =
    // normal-map image + sampler, defaulted to the device's flat (0,0,1) normal so the set is
    // complete; BindMaterial(attachNormalMap) overrides binding 3 with a real normal map.
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view_;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = device_.defaultSampler();

    VkDescriptorImageInfo normalImageInfo{};
    normalImageInfo.imageView = device_.defaultNormalView();
    normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo normalSamplerInfo{};
    normalSamplerInfo.sampler = device_.defaultSampler();

    VkWriteDescriptorSet writes[4]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set_;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[0].pImageInfo = &imageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set_;
    writes[1].dstBinding = 1;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[1].pImageInfo = &samplerInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = set_;
    writes[2].dstBinding = 3;
    writes[2].dstArrayElement = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[2].pImageInfo = &normalImageInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = set_;
    writes[3].dstBinding = 4;
    writes[3].dstArrayElement = 0;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[3].pImageInfo = &normalSamplerInfo;

    vkUpdateDescriptorSets(device_.device(), 4, writes, 0, nullptr);
    boundNormalView_ = device_.defaultNormalView();
}

void VulkanTexture::attachNormalMap(VkImageView normalView) {
    VkImageView target = normalView ? normalView : device_.defaultNormalView();
    if (target == boundNormalView_) return;  // already pointing here — skip the redundant update

    VkDescriptorImageInfo normalImageInfo{};
    normalImageInfo.imageView = target;
    normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set_;
    write.dstBinding = 3;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &normalImageInfo;
    vkUpdateDescriptorSets(device_.device(), 1, &write, 0, nullptr);
    boundNormalView_ = target;
}

VulkanTexture::~VulkanTexture() {
    if (set_) vkFreeDescriptorSets(device_.device(), device_.descriptorPool(), 1, &set_);
    if (environmentSet_)
        vkFreeDescriptorSets(device_.device(), device_.descriptorPool(), 1, &environmentSet_);
    if (view_) vkDestroyImageView(device_.device(), view_, nullptr);
    if (image_) vmaDestroyImage(device_.allocator(), image_, alloc_);
}

} // namespace hf::rhi::vk
