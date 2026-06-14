#include "rhi_metal/metal_texture.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

MetalTexture::MetalTexture(MetalDevice& device, const TextureDesc& desc) {
    id<MTLDevice> dev = device.device();

    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:ToMetalPixelFormat(desc.format)
                                     width:desc.width
                                    height:desc.height
                                 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    // Shared storage so we can replaceRegion directly (no blit/staging on unified memory).
    td.storageMode = MTLStorageModeShared;
    texture_ = [dev newTextureWithDescriptor:td];
    if (!texture_) Fail("newTextureWithDescriptor failed");

    if (desc.data) {
        // Tightly packed source rows: bytesPerRow = width * 4 (RGBA8/BGRA8).
        const NSUInteger bytesPerRow = (NSUInteger)desc.width * 4;
        MTLRegion region = MTLRegionMake2D(0, 0, desc.width, desc.height);
        [texture_ replaceRegion:region
                    mipmapLevel:0
                      withBytes:desc.data
                    bytesPerRow:bytesPerRow];
    }

    // Default sampler: linear filtering, repeat addressing (matches the Vulkan default sampler).
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.mipFilter = MTLSamplerMipFilterLinear;
    sd.sAddressMode = MTLSamplerAddressModeRepeat;
    sd.tAddressMode = MTLSamplerAddressModeRepeat;
    sampler_ = [dev newSamplerStateWithDescriptor:sd];
    if (!sampler_) Fail("newSamplerStateWithDescriptor failed");
}

MetalTexture::~MetalTexture() {
    // ARC releases texture_ / sampler_.
}

} // namespace hf::rhi::mtl
