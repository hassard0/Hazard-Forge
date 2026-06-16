#pragma once
#import <Metal/Metal.h>
#include "rhi/rhi.h"
#include "rhi_metal/metal_sampled.h"

namespace hf::rhi::mtl {

class MetalDevice;

// Slice DD — a sampleable CUBEMAP render target: a cube MTLTexture (textureType MTLTextureTypeCube,
// 6 faces, `size`x`size`, usage RenderTarget | ShaderRead) + a shared Depth32Float depth texture +
// a clamp-to-edge sampler. The probe capture renders into each face via a render encoder whose color
// attachment slice == the face index (set in BeginCubemapFace); the reflection pass samples the whole
// cube as a hardware texturecube<float> bound at the env slot (fragment texture(11)/sampler(12)) via
// BindCubemapProbe. Mirrors MetalRenderTarget; the cube-specific bits live ONLY here in the backend
// dir. Being an IMetalSampled whose sampledTexture() is the CUBE makes BindCubemapProbe identical to
// BindReflectionProbe. No layout bookkeeping (Metal tracks hazards automatically; the per-face passes
// commit+wait so the cube is fully written before it is sampled).
class MetalCubemapTarget final : public ICubemapTarget, public IMetalSampled {
public:
    MetalCubemapTarget(MetalDevice& device, uint32_t size, Format colorFormat);
    ~MetalCubemapTarget() override;

    uint32_t size() const override { return size_; }

    id<MTLTexture> cubeTexture() const { return color_; }
    id<MTLTexture> depthTexture() const { return depth_; }

    id<MTLTexture>      sampledTexture() const override { return color_; }  // the CUBE
    id<MTLSamplerState> sampledSampler() const override { return sampler_; }

private:
    uint32_t size_ = 0;
    id<MTLTexture>      color_   = nil;   // cube color (6 faces)
    id<MTLTexture>      depth_   = nil;   // shared 2D depth
    id<MTLSamplerState> sampler_ = nil;
};

} // namespace hf::rhi::mtl
