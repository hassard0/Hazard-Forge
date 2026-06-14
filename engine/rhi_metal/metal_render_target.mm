#include "rhi_metal/metal_render_target.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

MetalRenderTarget::MetalRenderTarget(MetalDevice& device, uint32_t width, uint32_t height)
    : width_(width), height_(height) {
    id<MTLDevice> dev = device.device();

    // Color: BGRA8 (matches the headless swapchain format so the lit pipeline renders into it
    // unchanged), usage = render target + shader read (so the post pass can sample it).
    MTLTextureDescriptor* cd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:width
                                    height:height
                                 mipmapped:NO];
    cd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    // Private storage: rendered+sampled GPU-side only, never read back on the CPU (the final post
    // output is what gets captured). Private is the right hint on unified memory for an attachment.
    cd.storageMode = MTLStorageModePrivate;
    color_ = [dev newTextureWithDescriptor:cd];
    if (!color_) Fail("render target: color newTextureWithDescriptor failed");

    // Depth: Depth32Float, render-target usage only.
    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                     width:width
                                    height:height
                                 mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModePrivate;
    depth_ = [dev newTextureWithDescriptor:dd];
    if (!depth_) Fail("render target: depth newTextureWithDescriptor failed");

    // Sampler the post pass uses to read the color image. Clamp-to-edge (the fullscreen triangle
    // samples within [0,1]; edge clamping avoids wrap artifacts at the borders for FXAA/glow taps).
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler_ = [dev newSamplerStateWithDescriptor:sd];
    if (!sampler_) Fail("render target: newSamplerStateWithDescriptor failed");
}

MetalRenderTarget::~MetalRenderTarget() {
    // ARC releases color_ / depth_ / sampler_.
}

} // namespace hf::rhi::mtl
