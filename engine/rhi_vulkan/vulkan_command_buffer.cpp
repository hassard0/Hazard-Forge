#include "rhi_vulkan/vulkan_command_buffer.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vulkan_pipeline.h"
#include "rhi_vulkan/vulkan_compute_pipeline.h"
#include "rhi_vulkan/vulkan_buffer.h"
#include "rhi_vulkan/vulkan_texture.h"
#include "rhi_vulkan/vulkan_render_target.h"
#include "rhi_vulkan/vulkan_sampled.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanCommandBuffer::VulkanCommandBuffer(VulkanDevice& device) : device_(device) {}

void VulkanCommandBuffer::Begin(VkCommandBuffer cmd, VkImageView colorView,
                                VkImageView depthView, VkExtent2D extent) {
    cmd_ = cmd;
    colorView_ = colorView;
    depthView_ = depthView;
    extent_ = extent;
    boundLayout_ = VK_NULL_HANDLE;
    boundMaterialSet_ = 0;
    boundEnvironmentSet_ = 0;
    boundClusterSet_ = 0;
}

void VulkanCommandBuffer::BeginRenderPass(const ClearColor& clear) {
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = colorView_;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{clear.r, clear.g, clear.b, clear.a}};

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView = depthView_;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // Depth-only (shadow) pass must STORE depth so the lit pass can sample it; the color passes
    // never read depth back, so DONT_CARE there.
    depth.storeOp = (colorView_ == VK_NULL_HANDLE) ? VK_ATTACHMENT_STORE_OP_STORE
                                                   : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent_};
    ri.layerCount = 1;
    // Depth-only pass (shadow map): colorView_ is null -> no color attachment.
    ri.colorAttachmentCount = (colorView_ == VK_NULL_HANDLE) ? 0 : 1;
    ri.pColorAttachments = (colorView_ == VK_NULL_HANDLE) ? nullptr : &color;
    ri.pDepthAttachment = &depth;
    vkCmdBeginRendering(cmd_, &ri);

    VkViewport vp{0, 0, (float)extent_.width, (float)extent_.height, 0.0f, 1.0f};
    vkCmdSetViewport(cmd_, 0, 1, &vp);
    VkRect2D scissor{{0, 0}, extent_};
    vkCmdSetScissor(cmd_, 0, 1, &scissor);
}

void VulkanCommandBuffer::BindPipeline(IPipeline& pipeline) {
    auto& p = static_cast<VulkanPipeline&>(pipeline);
    boundLayout_ = p.layout();
    boundMaterialSet_ = p.materialSetIndex();
    boundEnvironmentSet_ = p.hasEnvironmentSet() ? p.environmentSetIndex() : 0;
    boundClusterSet_ = p.hasClusterSet() ? p.clusterSetIndex() : 0;
    boundPushStages_ = p.pushConstantStages();
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, p.handle());
    // If the pipeline declares a per-frame set, bind the device's current frame set at set 0.
    // (frameIndex_ matches the UBO SetFrameUniforms wrote this frame — see VulkanDevice.)
    if (p.hasFrameSet()) {
        VkDescriptorSet fs = device_.currentFrameSet();
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_,
                                0, 1, &fs, 0, nullptr);
    }
    // If the pipeline declares the joint-palette set, bind the device's current palette UBO at
    // set 2 (mirrors the frame-set auto-bind; the UBO was written by SetJointPalette this frame).
    if (p.hasJointSet()) {
        VkDescriptorSet js = device_.currentJointPaletteSet();
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_,
                                2, 1, &js, 0, nullptr);
    }
}

void VulkanCommandBuffer::BindVertexBuffer(IBuffer& buffer) {
    auto& b = static_cast<VulkanBuffer&>(buffer);
    VkBuffer vb = b.handle();
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd_, 0, 1, &vb, &offset);
}

void VulkanCommandBuffer::BindInstanceBuffer(IBuffer& buffer) {
    auto& b = static_cast<VulkanBuffer&>(buffer);
    VkBuffer vb = b.handle();
    VkDeviceSize offset = 0;
    // Per-instance stream at binding 1 (binding 0 is the per-vertex stream).
    vkCmdBindVertexBuffers(cmd_, 1, 1, &vb, &offset);
}

void VulkanCommandBuffer::BindIndexBuffer(IBuffer& buffer) {
    auto& b = static_cast<VulkanBuffer&>(buffer);
    vkCmdBindIndexBuffer(cmd_, b.handle(), 0, VK_INDEX_TYPE_UINT32);
}

void VulkanCommandBuffer::BindTexture(ITexture& texture) {
    // Resolve the material descriptor set via the backend-internal ISampledVk interface so this
    // works for both VulkanTexture and VulkanRenderTarget (a sampled offscreen color).
    auto* sampled = dynamic_cast<ISampledVk*>(&texture);
    VkDescriptorSet s = sampled ? sampled->vkDescriptorSet() : VK_NULL_HANDLE;
    // Material set index is whatever the bound pipeline put it at (1 behind a frame set, else 0).
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_,
                            boundMaterialSet_, 1, &s, 0, nullptr);
}

void VulkanCommandBuffer::BindTexturePair(ITexture& primary, ITexture& secondary) {
    // The primary RT owns a material set (binding 0 = its color). Repoint its second slot (binding 3)
    // at the secondary image's view, then bind the one set so the fragment shader sees primary at
    // gTex (binding 0) and secondary at gTex2 (binding 3). Used by the bloom composite (Slice U).
    auto* primaryRT = dynamic_cast<VulkanRenderTarget*>(&primary);
    auto* secSampled = dynamic_cast<ISampledVk*>(&secondary);
    VkImageView secView = VK_NULL_HANDLE;
    if (auto* secRT = dynamic_cast<VulkanRenderTarget*>(&secondary)) secView = secRT->colorView();
    if (primaryRT && secView) primaryRT->attachSecondaryColor(secView);
    VkDescriptorSet s = (primaryRT) ? primaryRT->vkDescriptorSet()
                                    : (dynamic_cast<ISampledVk*>(&primary)
                                           ? dynamic_cast<ISampledVk*>(&primary)->vkDescriptorSet()
                                           : VK_NULL_HANDLE);
    (void)secSampled;
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_,
                            boundMaterialSet_, 1, &s, 0, nullptr);
}

void VulkanCommandBuffer::BindMaterial(ITexture& base, ITexture& normalMap) {
    // The base texture owns the material descriptor set (binding 0/1 = base-color). Point its
    // normal-map slot (binding 3/4) at `normalMap`'s view, then bind the single combined set. The
    // attach is cached, so re-binding the same (base, normalMap) pair each frame issues no update.
    auto& baseTex = static_cast<VulkanTexture&>(base);
    auto& normalTex = static_cast<VulkanTexture&>(normalMap);
    baseTex.attachNormalMap(normalTex.view());

    VkDescriptorSet s = baseTex.vkDescriptorSet();
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_,
                            boundMaterialSet_, 1, &s, 0, nullptr);
}

void VulkanCommandBuffer::BindMaterialPBR(ITexture& base, ITexture& metalRough, ITexture& normalMap,
                                          ITexture& emissive, ITexture& occlusion) {
    // The device builds (once) + caches the wider full-PBR descriptor set pointing at all five
    // textures, then we bind it at the bound pipeline's material set index.
    VkDescriptorSet s = device_.pbrMaterialSet(base, metalRough, normalMap, emissive, occlusion);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_,
                            boundMaterialSet_, 1, &s, 0, nullptr);
}

void VulkanCommandBuffer::BindEnvironment(ITexture& env) {
    // Bind the HDR equirect env texture's dedicated set (set 3) for image-based lighting. The
    // VulkanTexture (created with desc.environment) owns the env descriptor set.
    auto& tex = static_cast<VulkanTexture&>(env);
    VkDescriptorSet s = tex.environmentSet();
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_,
                            boundEnvironmentSet_, 1, &s, 0, nullptr);
}

void VulkanCommandBuffer::BindReflectionProbe(ITexture& probeAtlas) {
    // Slice AK — bind the baked probe atlas (an offscreen RGBA16F render target) at the dedicated
    // environment set (set 3) so the lit_probe fragment reads it as gProbe (binding 11/12). The RT
    // lazily allocates an env-layout descriptor set pointing at its own color view + the env sampler.
    auto& rt = static_cast<VulkanRenderTarget&>(probeAtlas);
    VkDescriptorSet s = rt.environmentSet();
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_,
                            boundEnvironmentSet_, 1, &s, 0, nullptr);
}

void VulkanCommandBuffer::BindLightClusters(IBuffer& clusters, IBuffer& lightIndices,
                                            IBuffer& lights) {
    // Push the three clustered-lighting storage buffers into the cluster set (set 3) of the bound
    // GRAPHICS pipeline, inline via vkCmdPushDescriptorSetKHR — same mechanism the compute path uses
    // for its SSBOs, but for VK_PIPELINE_BIND_POINT_GRAPHICS so the FRAGMENT stage can read them.
    auto& cb = static_cast<VulkanBuffer&>(clusters);
    auto& lb = static_cast<VulkanBuffer&>(lightIndices);
    auto& gb = static_cast<VulkanBuffer&>(lights);
    VkDescriptorBufferInfo infos[3] = {
        {cb.handle(), 0, VK_WHOLE_SIZE},
        {lb.handle(), 0, VK_WHOLE_SIZE},
        {gb.handle(), 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstBinding = 13 + i;   // clusters=13, lightIndices=14, lights=15
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    device_.pushDescriptorFn()(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_,
                               boundClusterSet_, 3, writes);
}

void VulkanCommandBuffer::Draw(uint32_t vertexCount, uint32_t firstVertex) {
    vkCmdDraw(cmd_, vertexCount, 1, firstVertex, 0);
}

void VulkanCommandBuffer::DrawIndexed(uint32_t indexCount, uint32_t firstIndex,
                                      int32_t vertexOffset) {
    vkCmdDrawIndexed(cmd_, indexCount, 1, firstIndex, vertexOffset, 0);
}

void VulkanCommandBuffer::DrawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount,
                                               uint32_t firstIndex, int32_t vertexOffset,
                                               uint32_t firstInstance) {
    vkCmdDrawIndexed(cmd_, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

void VulkanCommandBuffer::PushConstants(const void* data, uint32_t size) {
    // Push to whatever stages the bound pipeline's range covers: VERTEX for every geometry pass,
    // VERTEX|FRAGMENT for the bloom fullscreen passes (Slice U) whose params are read in fragment.
    vkCmdPushConstants(cmd_, boundLayout_, boundPushStages_, 0, size, data);
}

void VulkanCommandBuffer::SetScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) {
    // Override the full-extent scissor BeginRenderPass set, for the following ImGui draw(s).
    VkRect2D scissor{{x, y}, {width, height}};
    vkCmdSetScissor(cmd_, 0, 1, &scissor);
}

void VulkanCommandBuffer::SetViewport(int32_t x, int32_t y, uint32_t width, uint32_t height) {
    // Render the following draw(s) into a sub-rect of the attachment: set the viewport so NDC maps
    // into [x, x+width] x [y, y+height], and a matching scissor so nothing spills outside. Used by
    // the CSM shadow-atlas pass to target one cascade tile (Slice AD).
    VkViewport vp{(float)x, (float)y, (float)width, (float)height, 0.0f, 1.0f};
    vkCmdSetViewport(cmd_, 0, 1, &vp);
    VkRect2D scissor{{x, y}, {width, height}};
    vkCmdSetScissor(cmd_, 0, 1, &scissor);
}

void VulkanCommandBuffer::EndRenderPass() {
    vkCmdEndRendering(cmd_);
}

// --- Compute recording (outside any render pass) -----------------------------

void VulkanCommandBuffer::BindComputePipeline(IComputePipeline& pipeline) {
    auto& p = static_cast<VulkanComputePipeline&>(pipeline);
    boundComputeLayout_ = p.layout();
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p.handle());
}

void VulkanCommandBuffer::BindStorageBuffer(IBuffer& buffer, uint32_t index) {
    auto& b = static_cast<VulkanBuffer&>(buffer);
    // Push-descriptor: write the storage buffer into binding `index` of the compute set (set 0).
    VkDescriptorBufferInfo info{b.handle(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstBinding = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info;
    device_.pushDescriptorFn()(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, boundComputeLayout_,
                               /*set=*/0, 1, &write);
}

void VulkanCommandBuffer::ComputePushConstants(const void* data, uint32_t size) {
    vkCmdPushConstants(cmd_, boundComputeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
}

void VulkanCommandBuffer::DispatchCompute(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) {
    vkCmdDispatch(cmd_, groupsX, groupsY, groupsZ);
}

void VulkanCommandBuffer::ComputeToVertexBarrier() {
    // Compute writes (SHADER_WRITE) -> vertex-input reads (VERTEX_ATTRIBUTE_READ): the draw consumes
    // the particle buffer as a vertex stream. sync2 buffer memory barrier on the whole pipeline.
    VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
    b.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd_, &di);
}

} // namespace hf::rhi::vk
