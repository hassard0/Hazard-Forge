#pragma once
#import <Metal/Metal.h>
#include "rhi/rhi.h"
#include "rhi_metal/metal_sampled.h"

namespace hf::rhi::mtl {

class MetalDevice;

// An offscreen color MTLTexture (BGRA8, usage RenderTarget | ShaderRead) plus its own depth
// MTLTexture (Depth32Float). Mirrors VulkanRenderTarget: the scene renders into it, then a later
// fullscreen post pass samples it. Because it IS an ITexture (via IRenderTarget) and exposes the
// IMetalSampled accessors, MetalCommandBuffer::BindTexture binds it exactly like a MetalTexture.
//
// Unlike Vulkan there is no layout bookkeeping: Metal tracks hazards automatically, and
// EndRenderTargetFrame commits+waits, so the color texture is fully written before it is sampled.
class MetalRenderTarget final : public IRenderTarget, public IMetalSampled {
public:
    MetalRenderTarget(MetalDevice& device, uint32_t width, uint32_t height);
    ~MetalRenderTarget() override;

    uint32_t width() const override { return width_; }
    uint32_t height() const override { return height_; }

    // Attachment textures used by BeginRenderTargetFrame to open the offscreen encoder.
    id<MTLTexture> colorTexture() const { return color_; }
    id<MTLTexture> depthTexture() const { return depth_; }

    // IMetalSampled: the post pass samples the color image.
    id<MTLTexture>      sampledTexture() const override { return color_; }
    id<MTLSamplerState> sampledSampler() const override { return sampler_; }

private:
    uint32_t width_  = 0;
    uint32_t height_ = 0;
    id<MTLTexture>      color_   = nil;
    id<MTLTexture>      depth_   = nil;
    id<MTLSamplerState> sampler_ = nil;
};

} // namespace hf::rhi::mtl
