// Windowed (CAMetalLayer) construction path for MetalDevice. Split out from metal_device.mm so the
// headless (SSH / CI golden-image) target can link the Metal backend WITHOUT pulling in
// hal/window.h or SDL. This file holds the ONLY place the Metal backend references a CAMetalLayer
// source.
//
// Two windowed ctors, both feeding the SAME CAMetalLayer present path (BeginFrame acquires the
// layer's next drawable, EndFrame presents it — see metal_device.mm):
//   - MetalDevice(hf::hal::Window&)        : SDL HAL path (Windows/Vulkan-parity sample). Needs
//                                            hal/window.h (SDL), so it is compiled only when the
//                                            build defines HF_HAVE_SDL_WINDOW (the main engine build).
//   - MetalDevice(void* caMetalLayer, w, h): SDL-free path. The native Cocoa mac_window/ entry
//                                            creates the CAMetalLayer on its NSView and hands it in
//                                            as void* — no hal/window.h, no SDL. Always compiled.
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_windowed.h"
#include "rhi_metal/metal_command_buffer.h"  // complete type for unique_ptr<MetalCommandBuffer>
#include "rhi_metal/metal_common.h"
#import <QuartzCore/CAMetalLayer.h>

#if defined(HF_HAVE_SDL_WINDOW)
#include "hal/window.h"
#endif

namespace hf::rhi::mtl {

#if defined(HF_HAVE_SDL_WINDOW)
MetalDevice::MetalDevice(hf::hal::Window& window) : window_(&window) {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) Fail("MTLCreateSystemDefaultDevice returned nil");

    queue_ = [device_ newCommandQueue];
    if (!queue_) Fail("newCommandQueue failed");

    // Window vends its CAMetalLayer (created via SDL_Metal_CreateView under the hood).
    CAMetalLayer* layer = (__bridge CAMetalLayer*)window_->CreateMetalLayer();
    if (!layer) Fail("window returned a null CAMetalLayer");

    swapchain_ = std::make_unique<MetalSwapchain>(
        device_, layer, (uint32_t)window_->FramebufferWidth(),
        (uint32_t)window_->FramebufferHeight());

    Init();
}

// Free factory used by rhi_factory.cpp (a plain .cpp that must not include Metal headers).
// Declared in rhi_factory.cpp under __APPLE__; defined here so the Obj-C++ stays in this .mm.
std::unique_ptr<IRHIDevice> CreateMetalDevice(hf::hal::Window& window) {
    return std::make_unique<MetalDevice>(window);
}
#endif  // HF_HAVE_SDL_WINDOW

// SDL-free windowed ctor: build directly from a CAMetalLayer* (passed as void*). The layer is owned
// by the caller's NSView; this device only configures + presents to it. Same present path as the
// Window& ctor — only the layer source differs.
MetalDevice::MetalDevice(void* caMetalLayer, uint32_t width, uint32_t height) {
    if (!caMetalLayer) Fail("MetalDevice(void* layer): null CAMetalLayer");
    CAMetalLayer* layer = (__bridge CAMetalLayer*)caMetalLayer;

    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) Fail("MTLCreateSystemDefaultDevice returned nil");

    queue_ = [device_ newCommandQueue];
    if (!queue_) Fail("newCommandQueue failed");

    swapchain_ = std::make_unique<MetalSwapchain>(device_, layer, width, height);

    Init();
}

std::unique_ptr<IRHIDevice> CreateMetalDeviceWindowedLayer(void* caMetalLayer,
                                                           uint32_t width, uint32_t height) {
    return std::make_unique<MetalDevice>(caMetalLayer, width, height);
}

} // namespace hf::rhi::mtl
