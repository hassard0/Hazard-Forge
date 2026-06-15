#include "rhi_metal/metal_texture.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

MetalTexture::MetalTexture(MetalDevice& device, const TextureDesc& desc) {
    id<MTLDevice> dev = device.device();

    const uint32_t mipLevels = desc.mipLevels > 0 ? desc.mipLevels : 1;
    // Bytes per texel: RGBA16F = 8, everything else (RGBA8/BGRA8) = 4.
    const NSUInteger bpp = (desc.format == Format::RGBA16_Float) ? 8u : 4u;

    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:ToMetalPixelFormat(desc.format)
                                     width:desc.width
                                    height:desc.height
                                 mipmapped:(mipLevels > 1 ? YES : NO)];
    td.mipmapLevelCount = mipLevels;
    td.usage = MTLTextureUsageShaderRead;
    // Shared storage so we can replaceRegion directly (no blit/staging on unified memory).
    td.storageMode = MTLStorageModeShared;
    texture_ = [dev newTextureWithDescriptor:td];
    if (!texture_) Fail("newTextureWithDescriptor failed");

    if (mipLevels > 1 && desc.mipData) {
        // HDR environment map (Slice R): upload each CPU-prefiltered mip level.
        for (uint32_t i = 0; i < mipLevels; ++i) {
            uint32_t mw = desc.width >> i; if (mw == 0) mw = 1;
            uint32_t mh = desc.height >> i; if (mh == 0) mh = 1;
            MTLRegion region = MTLRegionMake2D(0, 0, mw, mh);
            [texture_ replaceRegion:region
                        mipmapLevel:i
                          withBytes:desc.mipData[i]
                        bytesPerRow:(NSUInteger)mw * bpp];
        }
    } else if (desc.data) {
        // Tightly packed source rows: bytesPerRow = width * bpp.
        const NSUInteger bytesPerRow = (NSUInteger)desc.width * bpp;
        MTLRegion region = MTLRegionMake2D(0, 0, desc.width, desc.height);
        [texture_ replaceRegion:region
                    mipmapLevel:0
                      withBytes:desc.data
                    bytesPerRow:bytesPerRow];
    }

    // Sampler. The HDR environment map (Slice R) uses trilinear + repeat-U/clamp-V (equirect
    // longitude wraps; poles must not). Everything else uses the default linear + repeat addressing
    // (matches the Vulkan default sampler).
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.mipFilter = MTLSamplerMipFilterLinear;
    sd.sAddressMode = MTLSamplerAddressModeRepeat;
    sd.tAddressMode = desc.environment ? MTLSamplerAddressModeClampToEdge
                                       : MTLSamplerAddressModeRepeat;
    sampler_ = [dev newSamplerStateWithDescriptor:sd];
    if (!sampler_) Fail("newSamplerStateWithDescriptor failed");
}

MetalTexture::~MetalTexture() {
    // ARC releases texture_ / sampler_.
}

} // namespace hf::rhi::mtl
