#include "rhi_metal/metal_command_buffer.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_pipeline.h"
#include "rhi_metal/metal_buffer.h"
#include "rhi_metal/metal_texture.h"
#include "rhi_metal/metal_sampled.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

MetalCommandBuffer::MetalCommandBuffer(MetalDevice& device) : device_(device) {}

void MetalCommandBuffer::Begin(id<MTLCommandBuffer> cmd, id<MTLTexture> colorTex,
                               id<MTLTexture> depthTex, uint32_t width, uint32_t height) {
    cmd_ = cmd;
    colorTex_ = colorTex;
    depthTex_ = depthTex;
    width_ = width;
    height_ = height;
    encoder_ = nil;
    indexBuffer_ = nil;
    boundFrameUniforms_ = false;
}

void MetalCommandBuffer::BeginRenderPass(const ClearColor& clear) {
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = colorTex_;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(clear.r, clear.g, clear.b, clear.a);

    rpd.depthAttachment.texture = depthTex_;
    rpd.depthAttachment.loadAction = MTLLoadActionClear;
    rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
    rpd.depthAttachment.clearDepth = 1.0;

    encoder_ = [cmd_ renderCommandEncoderWithDescriptor:rpd];
    if (!encoder_) Fail("renderCommandEncoderWithDescriptor failed");

    // Match the Vulkan winding (CCW front face, back-face cull).
    [encoder_ setFrontFacingWinding:MTLWindingCounterClockwise];
    [encoder_ setCullMode:MTLCullModeBack];

    MTLViewport vp{0.0, 0.0, (double)width_, (double)height_, 0.0, 1.0};
    [encoder_ setViewport:vp];
}

void MetalCommandBuffer::BindPipeline(IPipeline& pipeline) {
    auto& p = static_cast<MetalPipeline&>(pipeline);
    [encoder_ setRenderPipelineState:p.pipelineState()];
    [encoder_ setDepthStencilState:p.depthState()];
    boundFrameUniforms_ = p.usesFrameUniforms();

    // Fullscreen post pass: the [[vertex_id]]-generated triangle's winding depends on the clip
    // convention; disable culling so it is never back-face culled to black (mirrors Vulkan, which
    // sets cullMode NONE for fullscreen pipelines). BeginRenderPass defaults to cull-back.
    [encoder_ setCullMode:(p.fullscreen() ? MTLCullModeNone : MTLCullModeBack)];

    // Bind the device's current per-frame UBO (set 0) to both stages, matching the Vulkan path
    // where BindPipeline auto-binds the frame set.
    if (boundFrameUniforms_) {
        id<MTLBuffer> ubo = device_.currentFrameUbo();
        [encoder_ setVertexBuffer:ubo offset:0 atIndex:kVbFrameUbo];
        [encoder_ setFragmentBuffer:ubo offset:0 atIndex:kFbFrameUbo];
    }
}

void MetalCommandBuffer::BindVertexBuffer(IBuffer& buffer) {
    auto& b = static_cast<MetalBuffer&>(buffer);
    [encoder_ setVertexBuffer:b.handle() offset:0 atIndex:kVbVertex];
}

void MetalCommandBuffer::BindIndexBuffer(IBuffer& buffer) {
    auto& b = static_cast<MetalBuffer&>(buffer);
    indexBuffer_ = b.handle();
}

void MetalCommandBuffer::BindTexture(ITexture& texture) {
    // Works for both a MetalTexture (material) and a MetalRenderTarget (offscreen color image):
    // both implement IMetalSampled, so the post pass binds the RT exactly like a material.
    auto* s = dynamic_cast<IMetalSampled*>(&texture);
    if (!s) Fail("BindTexture: texture is not an IMetalSampled");
    [encoder_ setFragmentTexture:s->sampledTexture() atIndex:kFragTexture];
    [encoder_ setFragmentSamplerState:s->sampledSampler() atIndex:kFragSampler];
}

void MetalCommandBuffer::Draw(uint32_t vertexCount, uint32_t firstVertex) {
    [encoder_ drawPrimitives:MTLPrimitiveTypeTriangle
                 vertexStart:firstVertex
                 vertexCount:vertexCount];
}

void MetalCommandBuffer::DrawIndexed(uint32_t indexCount, uint32_t firstIndex) {
    // RHI index buffers are uint32 (Vulkan binds VK_INDEX_TYPE_UINT32).
    const NSUInteger indexBytes = (NSUInteger)firstIndex * sizeof(uint32_t);
    [encoder_ drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                         indexCount:indexCount
                          indexType:MTLIndexTypeUInt32
                        indexBuffer:indexBuffer_
                  indexBufferOffset:indexBytes];
}

void MetalCommandBuffer::PushConstants(const void* data, uint32_t size) {
    // Vertex push constants (model matrix) -> inline setVertexBytes at the push-constant slot.
    [encoder_ setVertexBytes:data length:size atIndex:kVbPushConst];
}

void MetalCommandBuffer::EndRenderPass() {
    [encoder_ endEncoding];
    encoder_ = nil;
}

} // namespace hf::rhi::mtl
