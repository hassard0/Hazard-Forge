#include "rhi_metal/metal_shader.h"
#include "rhi_metal/metal_common.h"
#import <Foundation/Foundation.h>

namespace hf::rhi::mtl {

MetalShaderModule::MetalShaderModule(id<MTLDevice> device, const std::string& source,
                                     const std::string& entryPoint) {
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    // Default language version is fine for the first cut; the Mac may need a pin (e.g. 2.4).
    library_ = [device newLibraryWithSource:src options:opts error:&err];
    if (!library_) {
        std::string msg = "newLibraryWithSource failed";
        if (err) msg += std::string(": ") + [[err localizedDescription] UTF8String];
        Fail(msg);
    }
    NSString* name = [NSString stringWithUTF8String:entryPoint.c_str()];
    function_ = [library_ newFunctionWithName:name];
    if (!function_) {
        Fail("entry point not found in MSL library: " + entryPoint);
    }
}

MetalShaderModule::~MetalShaderModule() {
    // ARC releases library_ / function_ automatically.
}

} // namespace hf::rhi::mtl
