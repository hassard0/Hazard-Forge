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
    void Begin(VkCommandBuffer cmd, VkImageView colorView, VkImageView depthView,
               VkExtent2D extent);

    void BeginRenderPass(const ClearColor& clear) override;
    void BindPipeline(IPipeline& pipeline) override;
    void BindVertexBuffer(IBuffer& buffer) override;
    void BindIndexBuffer(IBuffer& buffer) override;
    void Draw(uint32_t vertexCount, uint32_t firstVertex) override;
    void DrawIndexed(uint32_t indexCount, uint32_t firstIndex) override;
    void PushConstants(const void* data, uint32_t size) override;
    void EndRenderPass() override;

private:
    VulkanDevice& device_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkImageView colorView_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
    VkPipelineLayout boundLayout_ = VK_NULL_HANDLE;
};

} // namespace hf::rhi::vk
