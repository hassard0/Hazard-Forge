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

    void BeginRenderPass(const ClearColor& clear) override;
    void BindPipeline(IPipeline& pipeline) override;
    void BindVertexBuffer(IBuffer& buffer) override;
    void BindIndexBuffer(IBuffer& buffer) override;
    void BindTexture(ITexture& texture) override;
    void Draw(uint32_t vertexCount, uint32_t firstVertex) override;
    void DrawIndexed(uint32_t indexCount, uint32_t firstIndex) override;
    void PushConstants(const void* data, uint32_t size) override;
    void EndRenderPass() override;

    id<MTLRenderCommandEncoder> encoder() const { return encoder_; }

private:
    MetalDevice& device_;
    id<MTLCommandBuffer>        cmd_     = nil;
    id<MTLTexture>             colorTex_ = nil;
    id<MTLTexture>             depthTex_ = nil;
    id<MTLRenderCommandEncoder> encoder_ = nil;
    id<MTLBuffer>              indexBuffer_ = nil;  // stored by BindIndexBuffer; used by DrawIndexed
    bool boundFrameUniforms_ = false;  // current pipeline declares the per-frame UBO
    uint32_t width_ = 0;
    uint32_t height_ = 0;
};

} // namespace hf::rhi::mtl
