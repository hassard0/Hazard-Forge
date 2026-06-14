#pragma once
#import <Metal/Metal.h>
#include "rhi/rhi.h"
#include <string>

namespace hf::rhi::mtl {

// Wraps one MTLLibrary + the named entry-point MTLFunction. For the first cut the engine
// compiles MSL source at runtime (newLibraryWithSource:); there is no offline .metallib.
class MetalShaderModule final : public IShaderModule {
public:
    // Compiles `source` (MSL) and looks up `entryPoint`. Throws on compile/lookup failure.
    MetalShaderModule(id<MTLDevice> device, const std::string& source,
                      const std::string& entryPoint);
    ~MetalShaderModule() override;

    id<MTLFunction> function() const { return function_; }

private:
    id<MTLLibrary>  library_  = nil;
    id<MTLFunction> function_ = nil;
};

} // namespace hf::rhi::mtl
