#include "rhi_vulkan/vulkan_pbr_material.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vulkan_texture.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanPbrMaterial::VulkanPbrMaterial(VulkanDevice& device, ITexture& base, ITexture& metalRough,
                                     ITexture& normalMap, ITexture& emissive, ITexture& occlusion)
    : device_(device) {
    // Allocate one set from the wider full-PBR layout (set 1 for the lit-PBR pipeline).
    VkDescriptorSetLayout layout = device_.pbrMaterialSetLayout();
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = device_.descriptorPool();
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &layout;
    Check(vkAllocateDescriptorSets(device_.device(), &dai, &set_),
          "vkAllocateDescriptorSets(pbr-material)");

    // Five (image, sampler) pairs at the chosen bindings (matching the layout + the generated MSL):
    //   base 0/1, normal 3/4, metalRough 5/6, emissive 7/8, occlusion 9/10.
    ITexture* tex[5] = {&base, &normalMap, &metalRough, &emissive, &occlusion};
    const uint32_t imgBindings[5] = {0, 3, 5, 7, 9};
    const uint32_t smpBindings[5] = {1, 4, 6, 8, 10};

    VkDescriptorImageInfo imageInfos[5]{};
    VkDescriptorImageInfo samplerInfos[5]{};
    VkWriteDescriptorSet writes[10]{};
    for (int k = 0; k < 5; ++k) {
        auto& vt = static_cast<VulkanTexture&>(*tex[k]);
        imageInfos[k].imageView = vt.view();
        imageInfos[k].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        samplerInfos[k].sampler = device_.defaultSampler();

        writes[k * 2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[k * 2].dstSet = set_;
        writes[k * 2].dstBinding = imgBindings[k];
        writes[k * 2].descriptorCount = 1;
        writes[k * 2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[k * 2].pImageInfo = &imageInfos[k];

        writes[k * 2 + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[k * 2 + 1].dstSet = set_;
        writes[k * 2 + 1].dstBinding = smpBindings[k];
        writes[k * 2 + 1].descriptorCount = 1;
        writes[k * 2 + 1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[k * 2 + 1].pImageInfo = &samplerInfos[k];
    }
    vkUpdateDescriptorSets(device_.device(), 10, writes, 0, nullptr);
}

VulkanPbrMaterial::~VulkanPbrMaterial() {
    if (set_) vkFreeDescriptorSets(device_.device(), device_.descriptorPool(), 1, &set_);
}

} // namespace hf::rhi::vk
