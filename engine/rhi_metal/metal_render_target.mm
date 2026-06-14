#include "rhi_metal/metal_render_target.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

MetalRenderTarget::MetalRenderTarget(MetalDevice& device, uint32_t width, uint32_t height,
                                     bool depthOnly)
    : width_(width), height_(height), depthOnly_(depthOnly) {
    id<MTLDevice> dev = device.device();

    if (!depthOnly_) {
        // Color: BGRA8 (matches the headless swapchain format so the lit pipeline renders into it
        // unchanged), usage = render target + shader read (so the post pass can sample it).
        MTLTextureDescriptor* cd = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                         width:width
                                        height:height
                                     mipmapped:NO];
        cd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        // Private storage: rendered+sampled GPU-side only, never read back on the CPU (the final
        // post output is what gets captured). Private is the right hint on unified memory.
        cd.storageMode = MTLStorageModePrivate;
        color_ = [dev newTextureWithDescriptor:cd];
        if (!color_) Fail("render target: color newTextureWithDescriptor failed");
    }

    // Depth: Depth32Float. In the normal RT it is a render-target-only attachment; in depthOnly
    // (shadow map) mode it is ALSO shader-readable so the lit pass can sample it as the shadow map.
    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                     width:width
                                    height:height
                                 mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget |
               (depthOnly_ ? MTLTextureUsageShaderRead : 0);
    dd.storageMode = MTLStorageModePrivate;
    depth_ = [dev newTextureWithDescriptor:dd];
    if (!depth_) Fail("render target: depth newTextureWithDescriptor failed");

    // Sampler. For the post pass: reads the color image. For the shadow map: reads the depth image
    // with 3x3 PCF taps in the lit shader. Clamp-to-edge keeps taps just outside the light's frustum
    // from wrapping (a fragment outside the shadow UV range is guarded in the shader anyway).
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
