#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanDevice;

// Thin recorder over a VkCommandBuffer owned by VulkanDevice's per-frame sync.
class VulkanCommandBuffer final : public ICommandBuffer {
public:
    explicit VulkanCommandBuffer(VulkanDevice& device);

    // Called by VulkanDevice::BeginFrame to retarget this recorder. colorFormat/depthFormat are the
    // attachment formats of the pass this primary will open; they are forwarded to the device as the
    // secondary-command-buffer inheritance info when BeginRenderPass(clear, expectsSecondaries=true)
    // is used (Slice AU). Defaults keep the historical 4-arg call sites valid.
    void Begin(VkCommandBuffer cmd, VkImageView colorView, VkImageView depthView,
               VkExtent2D extent, VkFormat colorFormat = VK_FORMAT_UNDEFINED,
               VkFormat depthFormat = VK_FORMAT_UNDEFINED);

    // Retarget this recorder onto a SECONDARY command buffer (Slice AU). The secondary was already
    // begun (with render-pass-continue + dynamic-rendering inheritance) by the device; here we just
    // bind the recording target + extent and set the inherited dynamic viewport/scissor (NOT
    // inherited across secondaries in dynamic rendering, so each must set its own). Does NOT open a
    // render pass — the primary already did. The worker then records Bind*/Draw* as usual.
    void BeginSecondary(VkCommandBuffer cmd, VkExtent2D extent);

    void BeginRenderPass(const ClearColor& clear) override;
    void BeginRenderPass(const ClearColor& clear, bool expectsSecondaries) override;
    void ExecuteSecondaries(std::span<ICommandBuffer* const> secondaries) override;
    void BindPipeline(IPipeline& pipeline) override;
    void BindVertexBuffer(IBuffer& buffer) override;
    void BindInstanceBuffer(IBuffer& buffer) override;
    void BindIndexBuffer(IBuffer& buffer) override;
    void BindTexture(ITexture& texture) override;
    void BindTexturePair(ITexture& primary, ITexture& secondary) override;
    void BindMaterial(ITexture& base, ITexture& normalMap) override;
    void BindMaterialPBR(ITexture& base, ITexture& metalRough, ITexture& normalMap,
                         ITexture& emissive, ITexture& occlusion) override;
    void BindEnvironment(ITexture& env) override;
    void BindReflectionProbe(ITexture& probeAtlas) override;
    void BindLightClusters(IBuffer& clusters, IBuffer& lightIndices, IBuffer& lights) override;
    void BindPerDrawData(IBuffer& perDraw) override;
    void BindBindlessTextures(IBindlessTextureSet& set) override;
    void Draw(uint32_t vertexCount, uint32_t firstVertex) override;
    void DrawIndexed(uint32_t indexCount, uint32_t firstIndex, int32_t vertexOffset) override;
    void DrawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
                              int32_t vertexOffset, uint32_t firstInstance) override;
    void DrawIndexedIndirect(IBuffer& argsBuffer, size_t offset) override;
    void DrawIndexedMultiIndirect(IBuffer& argsBuffer, uint32_t drawCount, uint32_t stride) override;
    void PushConstants(const void* data, uint32_t size) override;
    void SetScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) override;
    void SetViewport(int32_t x, int32_t y, uint32_t width, uint32_t height) override;
    void EndRenderPass() override;

    void BindComputePipeline(IComputePipeline& pipeline) override;
    void BindStorageBuffer(IBuffer& buffer, uint32_t index) override;
    void ComputePushConstants(const void* data, uint32_t size) override;
    void DispatchCompute(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) override;
    void ComputeToVertexBarrier() override;
    void ResourceBarrier(IRenderTarget& resource, ResourceState from, ResourceState to) override;

private:
    VulkanDevice& device_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkImageView colorView_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
    VkFormat colorFormat_ = VK_FORMAT_UNDEFINED;  // pass color format (for secondary inheritance)
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;  // pass depth format (for secondary inheritance)
    VkPipelineLayout boundLayout_ = VK_NULL_HANDLE;
    uint32_t boundMaterialSet_ = 0;  // set index for BindTexture; from the bound pipeline
    uint32_t boundEnvironmentSet_ = 0;  // set index for BindEnvironment (set 3); 0 = pipeline has none
    uint32_t boundClusterSet_ = 0;      // set index for BindLightClusters (set 3); 0 = pipeline has none
    uint32_t boundPerDrawSet_ = 0;      // set index for BindPerDrawData (set 2, Slice BM); 0 = none
    uint32_t boundBindlessSet_ = 0;     // set index for BindBindlessTextures (set 4, Slice BZ); 0 = none
    uint32_t boundPushStages_ = VK_SHADER_STAGE_VERTEX_BIT;  // stages PushConstants targets
    VkPipelineLayout boundComputeLayout_ = VK_NULL_HANDLE;  // for compute push-constants/descriptors
};

} // namespace hf::rhi::vk
