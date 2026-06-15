#pragma once
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "rhi/rhi.h"

namespace hf::rhi::mtl {

class MetalDevice;

// Records one frame. BeginFrame on the device retargets this recorder with the frame's
// MTLCommandBuffer + the acquired drawable + depth texture; BeginRenderPass opens the single
// MTLRenderCommandEncoder (color=drawable, depth=depth texture).
class MetalCommandBuffer final : public ICommandBuffer {
public:
    explicit MetalCommandBuffer(MetalDevice& device);

    // Retarget for a new frame (called by MetalDevice::BeginFrame).
    void Begin(id<MTLCommandBuffer> cmd, id<MTLTexture> colorTex, id<MTLTexture> depthTex,
               uint32_t width, uint32_t height);

    // Retarget this recorder onto a sub-encoder vended by a parallel render command encoder (Slice
    // AU). The worker drives this recorder over its draw range. `enc` is the sub-encoder; width/
    // height size its viewport. Does NOT open a render pass — the primary's parallel encoder did.
    void BeginSecondary(id<MTLRenderCommandEncoder> enc, uint32_t width, uint32_t height);

    // Vend the next sub-encoder from this primary's open parallel render command encoder (creation
    // order == worker index == commit order). Called by MetalDevice::CreateSecondaryCommandBuffer.
    id<MTLRenderCommandEncoder> nextParallelSubEncoder();

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

    id<MTLRenderCommandEncoder> encoder() const { return encoder_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

private:
    MetalDevice& device_;
    id<MTLCommandBuffer>        cmd_     = nil;
    id<MTLTexture>             colorTex_ = nil;
    id<MTLTexture>             depthTex_ = nil;
    id<MTLRenderCommandEncoder> encoder_ = nil;
    // Parallel render command encoder open between BeginRenderPass(clear,true) and EndRenderPass
    // (Slice AU); vends sub-encoders for the worker threads, committed in creation order on
    // endEncoding. nil on the normal single-encoder path.
    id<MTLParallelRenderCommandEncoder> parallelEncoder_ = nil;
    id<MTLComputeCommandEncoder> computeEncoder_ = nil;  // open between BindComputePipeline..Dispatch
    id<MTLBuffer>              indexBuffer_ = nil;  // stored by BindIndexBuffer; used by DrawIndexed
    bool boundFrameUniforms_ = false;  // current pipeline declares the per-frame UBO
    bool boundPointList_ = false;      // current graphics pipeline draws points (particles)
    bool boundLineList_ = false;       // current graphics pipeline draws lines (debug-draw, Slice W)
    bool boundFragmentPushConst_ = false;  // current pipeline reads push constants in fragment (bloom)
    uint32_t computeThreadsPerGroup_ = 64;  // [numthreads(64,1,1)] in particles.comp.hlsl
    uint32_t width_ = 0;
    uint32_t height_ = 0;
};

} // namespace hf::rhi::mtl
