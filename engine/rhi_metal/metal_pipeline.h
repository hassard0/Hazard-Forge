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
    bool usesJointPalette() const { return usesJointPalette_; }
    bool fullscreen() const { return fullscreen_; }
    bool depthOnly() const { return depthOnly_; }
    bool pointList() const { return pointList_; }
    // No back-face culling (fullscreen passes, or UI quads which are clockwise-wound).
    bool cullNone() const { return fullscreen_ || cullNone_; }
    // True when push constants must also reach the FRAGMENT stage (bloom fullscreen passes).
    bool fragmentPushConstants() const { return fragmentPushConstants_; }

private:
    id<MTLRenderPipelineState> pipelineState_ = nil;
    id<MTLDepthStencilState>   depthState_    = nil;
    bool usesFrameUniforms_ = false;
    bool usesTexture_       = false;
    bool usesJointPalette_  = false;
    bool fullscreen_        = false;
    bool depthOnly_         = false;
    bool pointList_         = false;
    bool cullNone_          = false;
    bool fragmentPushConstants_ = false;
};

} // namespace hf::rhi::mtl
