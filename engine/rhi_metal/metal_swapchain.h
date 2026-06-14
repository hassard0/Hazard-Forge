#pragma once
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "rhi/rhi.h"

namespace hf::rhi::mtl {

// Color target abstraction for the Metal backend. Two modes:
//   - Windowed:  wraps the window's CAMetalLayer; AcquireNext() vends the next drawable.
//   - Headless:  owns an offscreen BGRA8 (shared-storage) color texture; no CAMetalLayer, no
//                drawable, no present. Used for SSH / window-server-less rendering + readback.
// Color format is BGRA8Unorm in both modes; owns a matching D32F depth texture.
class MetalSwapchain final : public ISwapchain {
public:
    // Windowed: drive a CAMetalLayer-backed swapchain.
    MetalSwapchain(id<MTLDevice> device, CAMetalLayer* layer, uint32_t width, uint32_t height);
    // Headless: render to an owned offscreen color texture (no window/CAMetalLayer).
    MetalSwapchain(id<MTLDevice> device, uint32_t width, uint32_t height);
    ~MetalSwapchain() override;

    Format ColorFormat() const override { return Format::BGRA8_UNorm; }
    void Recreate(uint32_t width, uint32_t height) override;

    bool headless() const { return layer_ == nil; }

    // Acquire the next drawable for this frame (windowed only). nil in headless mode or when the
    // layer has no drawable available (e.g. zero-sized / occluded).
    id<CAMetalDrawable> AcquireNext();

    // The offscreen color texture in headless mode (nil when windowed). The device renders into
    // this and reads it back for capture.
    id<MTLTexture> offscreenColor() const { return offscreenColor_; }

    CAMetalLayer*  layer() const { return layer_; }
    id<MTLTexture> depthTexture() const { return depthTexture_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

private:
    void BuildDepth();
    void BuildOffscreenColor();

    id<MTLDevice>  device_       = nil;
    CAMetalLayer*  layer_        = nil;   // windowed: owned by the window/view; nil when headless
    id<MTLTexture> offscreenColor_ = nil; // headless: owned BGRA8 color target; nil when windowed
    id<MTLTexture> depthTexture_ = nil;
    uint32_t width_  = 0;
    uint32_t height_ = 0;
};

} // namespace hf::rhi::mtl
