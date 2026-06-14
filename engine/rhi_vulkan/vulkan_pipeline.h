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
    // True when the pipeline layout has the per-frame UBO set (set 0); BindPipeline binds it.
    bool hasFrameSet() const { return hasFrameSet_; }
    // Set index the material (texture) descriptor occupies: 1 if a frame set precedes it, else 0.
    uint32_t materialSetIndex() const { return hasFrameSet_ ? 1u : 0u; }
    // True when the pipeline declares the joint-palette set (set 2); BindPipeline binds it there.
    bool hasJointSet() const { return hasJointSet_; }

private:
    VkDevice device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    bool hasFrameSet_ = false;
    bool hasJointSet_ = false;
};

} // namespace hf::rhi::vk
