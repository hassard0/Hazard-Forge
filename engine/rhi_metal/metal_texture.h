#pragma once
#import <Metal/Metal.h>
#include "rhi/rhi.h"

namespace hf::rhi::mtl {

class MetalDevice;

// Sampled 2D MTLTexture + a sampler state. Bound at fragment texture(0)/sampler(0).
class MetalTexture final : public ITexture {
public:
    MetalTexture(MetalDevice& device, const TextureDesc& desc);
    ~MetalTexture() override;

    id<MTLTexture>      texture() const { return texture_; }
    id<MTLSamplerState> sampler() const { return sampler_; }

private:
    id<MTLTexture>      texture_ = nil;
    id<MTLSamplerState> sampler_ = nil;
};

} // namespace hf::rhi::mtl
