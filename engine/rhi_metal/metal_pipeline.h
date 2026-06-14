#pragma once
#import <Metal/Metal.h>
#include "rhi/rhi.h"

namespace hf::rhi::mtl {

class MetalDevice;

// MTLRenderPipelineState (from a vertex descriptor + the two MSL functions) plus a
// MTLDepthStencilState. Both are bound together by MetalCommandBuffer::BindPipeline.
class MetalPipeline final : public IPipeline {
public:
    MetalPipeline(MetalDevice& device, const GraphicsPipelineDesc& desc);
    ~MetalPipeline() override;

    id<MTLRenderPipelineState> pipelineState() const { return pipelineState_; }
    id<MTLDepthStencilState>   depthState() const { return depthState_; }
    bool usesFrameUniforms() const { return usesFrameUniforms_; }
    bool usesTexture() const { return usesTexture_; }
    bool fullscreen() const { return fullscreen_; }
    bool depthOnly() const { return depthOnly_; }

private:
    id<MTLRenderPipelineState> pipelineState_ = nil;
    id<MTLDepthStencilState>   depthState_    = nil;
    bool usesFrameUniforms_ = false;
    bool usesTexture_       = false;
    bool fullscreen_        = false;
    bool depthOnly_         = false;
};

} // namespace hf::rhi::mtl
