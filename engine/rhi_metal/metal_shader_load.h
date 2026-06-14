#pragma once
// Apple-only, Metal-free helper so plain-C++ callers (the sample) can build MSL shader modules
// without including any Metal/Obj-C headers. Implemented in metal_shader_load.mm.
#include "rhi/rhi.h"
#include <memory>
#include <string>

namespace hf::rhi::mtl {

// Compile MSL `source`, returning the named entry point as an IShaderModule. `device` must be a
// MetalDevice (the one CreateDevice(Backend::Metal,...) returned); a bad-cast throws.
std::unique_ptr<IShaderModule> MakeShaderModuleFromMSL(IRHIDevice& device,
                                                       const std::string& source,
                                                       const std::string& entryPoint);

} // namespace hf::rhi::mtl
