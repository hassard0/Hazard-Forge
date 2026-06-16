#include "rhi_metal/metal_cubemap_target.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

MetalCubemapTarget::MetalCubemapTarget(MetalDevice& device, uint32_t size, Format colorFormat)
    : size_(size) {
    id<MTLDevice> dev = device.device();

    // Cube color texture: 6 faces, RenderTarget | ShaderRead (rendered per face, sampled as a cube).
    const MTLPixelFormat colorPF = (colorFormat == Format::Undefined)
                                       ? MTLPixelFormatBGRA8Unorm
                                       : ToMetalPixelFormat(colorFormat);
    MTLTextureDescriptor* cd = [MTLTextureDescriptor
        textureCubeDescriptorWithPixelFormat:colorPF
                                        size:size
                                   mipmapped:NO];
    cd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    cd.storageMode = MTLStorageModePrivate;
    color_ = [dev newTextureWithDescriptor:cd];
    if (!color_) Fail("cubemap target: cube color newTextureWithDescriptor failed");

    // Shared 2D depth (re-cleared per face).
    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                     width:size
                                    height:size
                                 mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModePrivate;
    depth_ = [dev newTextureWithDescriptor:dd];
    if (!depth_) Fail("cubemap target: depth newTextureWithDescriptor failed");

    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler_ = [dev newSamplerStateWithDescriptor:sd];
    if (!sampler_) Fail("cubemap target: newSamplerStateWithDescriptor failed");
}

MetalCubemapTarget::~MetalCubemapTarget() {
    // ARC releases color_ / depth_ / sampler_.
}

} // namespace hf::rhi::mtl
