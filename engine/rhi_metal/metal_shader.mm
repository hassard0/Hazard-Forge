#include "rhi_metal/metal_shader.h"
#include "rhi_metal/metal_common.h"
#import <Foundation/Foundation.h>

namespace hf::rhi::mtl {

MetalShaderModule::MetalShaderModule(id<MTLDevice> device, const std::string& source,
                                     const std::string& entryPoint) {
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    // Slice METAL-RT S1: pin MSL 2.4. metal::raytracing::intersection_query (shaders/rt_query.metal,
    // bound via the RT1 IAccelStructure seam) REQUIRES Metal 2.4 — matches the proven showcase's pin
    // (visual_test.mm:25678, opts.languageVersion = MTLLanguageVersion2_4). 2.4 is a superset of the
    // default + the >=2.2 visbuffer/visresolve shaders need, so the existing MSL still compiles and the
    // existing Metal goldens are byte-unaffected (this is a compile TARGET, not a codegen change).
    opts.languageVersion = MTLLanguageVersion2_4;
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
