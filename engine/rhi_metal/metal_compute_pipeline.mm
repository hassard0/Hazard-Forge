#include "rhi_metal/metal_compute_pipeline.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_shader.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

MetalComputePipeline::MetalComputePipeline(MetalDevice& device, const ComputePipelineDesc& desc) {
    auto* cs = static_cast<MetalShaderModule*>(desc.compute);
    NSError* err = nil;
    state_ = [device.device() newComputePipelineStateWithFunction:cs->function() error:&err];
    if (!state_) {
        std::string msg = "newComputePipelineStateWithFunction failed";
        if (err) msg += std::string(": ") + [[err localizedDescription] UTF8String];
        Fail(msg);
    }
}

MetalComputePipeline::~MetalComputePipeline() {
    // ARC releases state_.
}

} // namespace hf::rhi::mtl
