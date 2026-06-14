// Windowed (CAMetalLayer / SDL) construction path for MetalDevice. Split out from
// metal_device.mm so the headless (SSH / CI golden-image) target can link the Metal backend
// WITHOUT pulling in hal/window.h or SDL. The headless path (metal_device.mm) never touches a
// Window; this file is the only place the Metal backend references one.
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_command_buffer.h"  // complete type for unique_ptr<MetalCommandBuffer>
#include "rhi_metal/metal_common.h"
#include "hal/window.h"
#import <QuartzCore/CAMetalLayer.h>

namespace hf::rhi::mtl {

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

} // namespace hf::rhi::mtl
