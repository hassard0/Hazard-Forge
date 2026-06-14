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
//
// In depthOnly mode (shadow map): no color image; the depth MTLTexture gets usage
// RenderTarget | ShaderRead (sampleable), so the lit pass samples it as the shadow map. The
// sampled accessors then return the DEPTH texture so SetShadowMap can bind it. Mirrors
// VulkanRenderTarget's depthOnly ctor path.
class MetalRenderTarget final : public IRenderTarget, public IMetalSampled {
public:
    MetalRenderTarget(MetalDevice& device, uint32_t width, uint32_t height,
                      bool depthOnly = false);
    ~MetalRenderTarget() override;

    uint32_t width() const override { return width_; }
    uint32_t height() const override { return height_; }
    bool depthOnly() const { return depthOnly_; }

    // Attachment textures used by BeginRenderTargetFrame / BeginShadowPass to open the encoder.
    id<MTLTexture> colorTexture() const { return color_; }   // nil in depthOnly mode
    id<MTLTexture> depthTexture() const { return depth_; }

    // IMetalSampled: the post pass samples the color image; the lit pass samples the depth image
    // (depthOnly / shadow map). sampledSampler() returns a clamp-to-edge linear sampler.
    id<MTLTexture>      sampledTexture() const override { return depthOnly_ ? depth_ : color_; }
    id<MTLSamplerState> sampledSampler() const override { return sampler_; }

private:
    uint32_t width_  = 0;
    uint32_t height_ = 0;
    bool     depthOnly_ = false;
    id<MTLTexture>      color_   = nil;
    id<MTLTexture>      depth_   = nil;
    id<MTLSamplerState> sampler_ = nil;
};

} // namespace hf::rhi::mtl
