#pragma once
#import <Metal/Metal.h>

namespace hf::rhi::mtl {

// Common Metal-side accessor for anything a fragment shader can sample: a sampled
// MTLTexture + a sampler state. Implemented by both MetalTexture (uploaded material) and
// MetalRenderTarget (offscreen color image). MetalCommandBuffer::BindTexture downcasts an
// ITexture& to this so it can bind either without knowing the concrete type — mirroring the
// Vulkan ISampledVk seam (vulkan_sampled.h).
class IMetalSampled {
public:
    virtual ~IMetalSampled() = default;
    virtual id<MTLTexture>      sampledTexture() const = 0;
    virtual id<MTLSamplerState> sampledSampler() const = 0;
};

} // namespace hf::rhi::mtl
