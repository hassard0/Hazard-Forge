#pragma once
#import <Metal/Metal.h>
#include "rhi/rhi.h"
#include "rhi_metal/metal_sampled.h"

namespace hf::rhi::mtl {

class MetalDevice;

// Sampled 2D MTLTexture + a sampler state. Bound at fragment texture(0)/sampler(0).
class MetalTexture final : public ITexture, public IMetalSampled {
public:
    MetalTexture(MetalDevice& device, const TextureDesc& desc);
    ~MetalTexture() override;

    id<MTLTexture>      texture() const { return texture_; }
    id<MTLSamplerState> sampler() const { return sampler_; }

    // IMetalSampled: lets BindTexture bind this without knowing the concrete type.
    id<MTLTexture>      sampledTexture() const override { return texture_; }
    id<MTLSamplerState> sampledSampler() const override { return sampler_; }

private:
    id<MTLTexture>      texture_ = nil;
    id<MTLSamplerState> sampler_ = nil;
};

} // namespace hf::rhi::mtl
