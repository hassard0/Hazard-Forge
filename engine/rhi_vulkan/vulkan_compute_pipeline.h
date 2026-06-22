#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanDevice;

// VkPipeline (compute bind point) + its layout + a descriptor set layout holding `storageBufferCount`
// STORAGE_BUFFER bindings (0..count-1). The descriptor set itself is allocated per-dispatch by the
// command buffer (push-descriptor-free path: a small pool owned here), so the same pipeline can be
// re-bound to different storage buffers each frame.
class VulkanComputePipeline final : public IComputePipeline {
public:
    VulkanComputePipeline(VulkanDevice& device, const ComputePipelineDesc& desc);
    ~VulkanComputePipeline() override;

    VkPipeline handle() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }
    VkDescriptorSetLayout setLayout() const { return setLayout_; }
    uint32_t storageBufferCount() const { return storageBufferCount_; }
    uint32_t pushConstantSize() const { return pushConstantSize_; }
    bool sampledShadowMap() const { return sampledShadowMap_; }
    int accelStructureBinding() const { return accelStructureBinding_; }

private:
    VkDevice device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    uint32_t storageBufferCount_ = 0;
    uint32_t pushConstantSize_ = 0;
    bool sampledShadowMap_ = false;
    int accelStructureBinding_ = -1;  // Slice RT2: TLAS descriptor slot, or -1 if none
};

} // namespace hf::rhi::vk
