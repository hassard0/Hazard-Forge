#include "rhi_vulkan/vulkan_command_buffer.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vulkan_pipeline.h"
#include "rhi_vulkan/vulkan_buffer.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanCommandBuffer::VulkanCommandBuffer(VulkanDevice& device) : device_(device) {}

void VulkanCommandBuffer::Begin(VkCommandBuffer cmd, VkImageView colorView, VkExtent2D extent) {
    cmd_ = cmd;
    colorView_ = colorView;
    extent_ = extent;
}

void VulkanCommandBuffer::BeginRenderPass(const ClearColor& clear) {
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = colorView_;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{clear.r, clear.g, clear.b, clear.a}};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent_};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cmd_, &ri);

    VkViewport vp{0, 0, (float)extent_.width, (float)extent_.height, 0.0f, 1.0f};
    vkCmdSetViewport(cmd_, 0, 1, &vp);
    VkRect2D scissor{{0, 0}, extent_};
    vkCmdSetScissor(cmd_, 0, 1, &scissor);
}

void VulkanCommandBuffer::BindPipeline(IPipeline& pipeline) {
    auto& p = static_cast<VulkanPipeline&>(pipeline);
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, p.handle());
}

void VulkanCommandBuffer::BindVertexBuffer(IBuffer& buffer) {
    auto& b = static_cast<VulkanBuffer&>(buffer);
    VkBuffer vb = b.handle();
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd_, 0, 1, &vb, &offset);
}

void VulkanCommandBuffer::Draw(uint32_t vertexCount, uint32_t firstVertex) {
    vkCmdDraw(cmd_, vertexCount, 1, firstVertex, 0);
}

void VulkanCommandBuffer::EndRenderPass() {
    vkCmdEndRendering(cmd_);
}

} // namespace hf::rhi::vk
