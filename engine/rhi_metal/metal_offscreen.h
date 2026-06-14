#pragma once
// Apple-only, Metal-free factory so plain-C++ callers (samples, tests) can construct a HEADLESS
// (offscreen, window-server-less) Metal device without including any Metal/Obj-C headers.
// Implemented in metal_device.mm. Use CaptureNextFrame()+GetCapturedPixels() to read frames back.
#include "rhi/rhi.h"
#include <cstdint>
#include <memory>

namespace hf::rhi::mtl {

std::unique_ptr<IRHIDevice> CreateMetalDeviceHeadless(uint32_t width, uint32_t height);

} // namespace hf::rhi::mtl
