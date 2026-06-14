#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanDevice;

class VulkanPipeline final : public IPipeline {
public:
    VulkanPipeline(VulkanDevice& device, const GraphicsPipelineDesc& desc);
    ~VulkanPipeline() override;
    VkPipeline handle() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }

private:
    VkDevice device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

} // namespace hf::rhi::vk
