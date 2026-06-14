#include "rhi_vulkan/vulkan_compute_pipeline.h"
#include "rhi_vulkan/vulkan_shader.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"
#include <vector>

namespace hf::rhi::vk {

VulkanComputePipeline::VulkanComputePipeline(VulkanDevice& device, const ComputePipelineDesc& desc)
    : device_(device.device()),
      storageBufferCount_(desc.storageBufferCount),
      pushConstantSize_(desc.pushConstantSize) {
    auto* cs = static_cast<VulkanShaderModule*>(desc.compute);

    // Descriptor set layout: `storageBufferCount` STORAGE_BUFFER bindings (0..N-1), compute stage.
    std::vector<VkDescriptorSetLayoutBinding> bindings(storageBufferCount_);
    for (uint32_t i = 0; i < storageBufferCount_; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    // PUSH_DESCRIPTOR: the storage buffers are bound inline via vkCmdPushDescriptorSetKHR (no pool).
    slci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    slci.bindingCount = (uint32_t)bindings.size();
    slci.pBindings = bindings.data();
    Check(vkCreateDescriptorSetLayout(device_, &slci, nullptr, &setLayout_),
          "vkCreateDescriptorSetLayout(compute)");

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = pushConstantSize_;

    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 1;
    lci.pSetLayouts = &setLayout_;
    if (pushConstantSize_ > 0) {
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pushRange;
    }
    Check(vkCreatePipelineLayout(device_, &lci, nullptr, &layout_),
          "vkCreatePipelineLayout(compute)");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs->handle();
    stage.pName = "main";

    VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pci.stage = stage;
    pci.layout = layout_;
    Check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_),
          "vkCreateComputePipelines");
}

VulkanComputePipeline::~VulkanComputePipeline() {
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
    if (setLayout_) vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
}

} // namespace hf::rhi::vk
