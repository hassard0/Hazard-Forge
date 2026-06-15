#pragma once
// Apple-only, Metal-free factory so plain-C++ / SDL-free callers (the native Cocoa mac_window/
// entry) can construct a WINDOWED Metal device from a CAMetalLayer* (passed as void*) WITHOUT
// including hal/window.h (SDL) or any Metal/Obj-C headers. Implemented in metal_device_windowed.mm.
// The device drives the same CAMetalLayer present path as the SDL Window& ctor: BeginFrame acquires
// the layer's next drawable, EndFrame presents it.
#include "rhi/rhi.h"
#include <cstdint>
#include <memory>

namespace hf::rhi::mtl {

// `caMetalLayer` must be a CAMetalLayer* (the caller created it on its NSView); cast back in the .mm.
std::unique_ptr<IRHIDevice> CreateMetalDeviceWindowedLayer(void* caMetalLayer,
                                                           uint32_t width, uint32_t height);

} // namespace hf::rhi::mtl
