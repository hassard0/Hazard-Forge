#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanDevice;

// Thin recorder over a VkCommandBuffer owned by VulkanDevice's per-frame sync.
class VulkanCommandBuffer final : public ICommandBuffer {
public:
    explicit VulkanCommandBuffer(VulkanDevice& device);

    // Called by VulkanDevice::BeginFrame to retarget this recorder.
    void Begin(VkCommandBuffer cmd, VkImageView colorView, VkExtent2D extent);

    void BeginRenderPass(const ClearColor& clear) override;
    void BindPipeline(IPipeline& pipeline) override;
    void BindVertexBuffer(IBuffer& buffer) override;
    void Draw(uint32_t vertexCount, uint32_t firstVertex) override;
    void EndRenderPass() override;

private:
    VulkanDevice& device_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkImageView colorView_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
};

} // namespace hf::rhi::vk
