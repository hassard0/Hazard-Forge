#include "rhi_metal/metal_swapchain.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

MetalSwapchain::MetalSwapchain(id<MTLDevice> device, CAMetalLayer* layer,
                               uint32_t width, uint32_t height)
    : device_(device), layer_(layer), width_(width), height_(height) {
    // Configure the layer for our device + color format. SDL_Metal_CreateView already created
    // the layer; we own its pixel format / drawable size from here.
    layer_.device = device_;
    layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer_.framebufferOnly = NO;  // NO so headless capture can read the drawable texture back.
    layer_.drawableSize = CGSizeMake(width_, height_);
    BuildDepth();
}

MetalSwapchain::MetalSwapchain(id<MTLDevice> device, uint32_t width, uint32_t height)
    : device_(device), layer_(nil), width_(width), height_(height) {
    // Headless: no CAMetalLayer. Render into an owned offscreen BGRA8 color texture.
    BuildOffscreenColor();
    BuildDepth();
}

MetalSwapchain::~MetalSwapchain() {
    // ARC releases depthTexture_ / offscreenColor_. layer_ is owned by the window's SDL_MetalView.
}

void MetalSwapchain::BuildOffscreenColor() {
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:width_
                                    height:height_
                                 mipmapped:NO];
    // RenderTarget so we can draw into it; ShaderRead is harmless and keeps it general. Shared
    // storage on Apple Silicon lets us getBytes the result back directly (no blit/staging).
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    offscreenColor_ = [device_ newTextureWithDescriptor:td];
    if (!offscreenColor_) Fail("offscreen color newTextureWithDescriptor failed");
}

void MetalSwapchain::BuildDepth() {
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                     width:width_
                                    height:height_
                                 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;  // GPU-only depth.
    depthTexture_ = [device_ newTextureWithDescriptor:td];
    if (!depthTexture_) Fail("depth newTextureWithDescriptor failed");
}

void MetalSwapchain::Recreate(uint32_t width, uint32_t height) {
    width_ = width;
    height_ = height;
    if (layer_) {
        layer_.drawableSize = CGSizeMake(width_, height_);
    } else {
        BuildOffscreenColor();
    }
    BuildDepth();
}

id<CAMetalDrawable> MetalSwapchain::AcquireNext() {
    if (!layer_) return nil;  // headless: no drawable
    return [layer_ nextDrawable];
}

} // namespace hf::rhi::mtl
