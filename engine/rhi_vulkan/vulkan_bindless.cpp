#include "rhi_vulkan/vulkan_bindless.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"
#include "rhi_vulkan/vulkan_sampled.h"

#include <vector>

namespace hf::rhi::vk {

VulkanBindlessTextureSet::VulkanBindlessTextureSet(VulkanDevice& device,
                                                   std::span<ITexture* const> textures)
    : device_(device), count_((uint32_t)textures.size()) {
    VkDevice dev = device_.device();

    // Dedicated UPDATE_AFTER_BIND descriptor pool sized for one bindless set (the array binding +
    // the shared sampler). Separate from the device's main pool because the bindless layout needs the
    // UPDATE_AFTER_BIND pool flag (the main pool/sets are not update-after-bind).
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    sizes[0].descriptorCount = VulkanDevice::kBindlessMaxTextures;
    sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    sizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pci.maxSets = 1;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = sizes;
    Check(vkCreateDescriptorPool(dev, &pci, nullptr, &pool_), "vkCreateDescriptorPool(bindless)");

    // Allocate the set, declaring the VARIABLE descriptor count for the array binding (binding 0): the
    // runtime array is sized to exactly `count_` textures (<= kBindlessMaxTextures).
    VkDescriptorSetLayout layout = device_.bindlessSetLayout();
    uint32_t variableCount = count_;
    VkDescriptorSetVariableDescriptorCountAllocateInfo varInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO};
    varInfo.descriptorSetCount = 1;
    varInfo.pDescriptorCounts = &variableCount;
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.pNext = &varInfo;
    dai.descriptorPool = pool_;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &layout;
    Check(vkAllocateDescriptorSets(dev, &dai, &set_), "vkAllocateDescriptorSets(bindless)");

    // Write the shared sampler (binding 1) + the whole sampled-image array (binding 0, element i = the
    // i-th texture's view). The sampler is the device default (LINEAR / REPEAT) — IDENTICAL to the
    // per-material bound path's sampler, so the bindless fetch is byte-identical.
    std::vector<VkDescriptorImageInfo> imgInfos(count_);
    for (uint32_t i = 0; i < count_; ++i) {
        auto* sampled = dynamic_cast<ISampledVk*>(textures[i]);
        imgInfos[i].imageView = sampled ? sampled->vkImageView() : VK_NULL_HANDLE;
        imgInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imgInfos[i].sampler = VK_NULL_HANDLE;  // separate sampler descriptor (binding 1)
    }

    VkDescriptorImageInfo smpInfo{};
    smpInfo.sampler = device_.defaultSampler();

    // binding 0 = the shared sampler; binding 1 = the sampled-image array (the highest-numbered,
    // variable-count binding — see the layout).
    VkWriteDescriptorSet writes[2]{};
    uint32_t writeCount = 0;
    writes[writeCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[writeCount].dstSet = set_;
    writes[writeCount].dstBinding = 0;
    writes[writeCount].dstArrayElement = 0;
    writes[writeCount].descriptorCount = 1;
    writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[writeCount].pImageInfo = &smpInfo;
    ++writeCount;
    if (count_ > 0) {
        writes[writeCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[writeCount].dstSet = set_;
        writes[writeCount].dstBinding = 1;
        writes[writeCount].dstArrayElement = 0;
        writes[writeCount].descriptorCount = count_;
        writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[writeCount].pImageInfo = imgInfos.data();
        ++writeCount;
    }

    vkUpdateDescriptorSets(dev, writeCount, writes, 0, nullptr);
}

VulkanBindlessTextureSet::~VulkanBindlessTextureSet() {
    // Destroying the pool frees the set.
    if (pool_) vkDestroyDescriptorPool(device_.device(), pool_, nullptr);
}

} // namespace hf::rhi::vk
