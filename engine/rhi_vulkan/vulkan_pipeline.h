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
    // True when the pipeline declares the dedicated HDR environment set (set 3, Slice R).
    bool hasEnvironmentSet() const { return hasEnvironmentSet_; }
    // The environment set's index in the pipeline layout (always 3 for the IBL pipeline: frame 0,
    // material 1, joint-set placeholder 2, env 3).
    uint32_t environmentSetIndex() const { return 3u; }
    // True when the pipeline declares the dedicated light-cluster set (set 3, Slice AG).
    bool hasClusterSet() const { return hasClusterSet_; }
    // The cluster set's index in the pipeline layout (always 3: frame 0, material 1, joint
    // placeholder 2, clusters 3 — mirrors the env set).
    uint32_t clusterSetIndex() const { return 3u; }
    // Stage flags the push-constant range is visible to (VERTEX, or VERTEX|FRAGMENT for the bloom
    // fullscreen passes). PushConstants pushes to exactly these stages.
    uint32_t pushConstantStages() const { return pushConstantStages_; }

private:
    VkDevice device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    bool hasFrameSet_ = false;
    bool hasJointSet_ = false;
    bool hasEnvironmentSet_ = false;
    bool hasClusterSet_ = false;
    uint32_t pushConstantStages_ = VK_SHADER_STAGE_VERTEX_BIT;
};

} // namespace hf::rhi::vk
