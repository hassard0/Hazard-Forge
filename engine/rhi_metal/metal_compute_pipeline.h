#pragma once
#import <Metal/Metal.h>
#include "rhi/rhi.h"

namespace hf::rhi::mtl {

class MetalDevice;

// MTLComputePipelineState from a compute (kernel) MSL function. The storage buffer(s) and push
// constants are bound at dispatch time by MetalCommandBuffer (Metal binds resources at the call
// site, not in the pipeline). storageBufferCount/pushConstantSize are kept for parity but Metal
// needs neither at creation.
class MetalComputePipeline final : public IComputePipeline {
public:
    MetalComputePipeline(MetalDevice& device, const ComputePipelineDesc& desc);
    ~MetalComputePipeline() override;

    id<MTLComputePipelineState> state() const { return state_; }

private:
    id<MTLComputePipelineState> state_ = nil;
};

} // namespace hf::rhi::mtl
