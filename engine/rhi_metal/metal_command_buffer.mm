#include "rhi_metal/metal_command_buffer.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_pipeline.h"
#include "rhi_metal/metal_compute_pipeline.h"
#include "rhi_metal/metal_buffer.h"
#include "rhi_metal/metal_texture.h"
#include "rhi_metal/metal_sampled.h"
#include "rhi_metal/metal_render_target.h"
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
    computeEncoder_ = nil;
    indexBuffer_ = nil;
    boundFrameUniforms_ = false;
    boundPointList_ = false;
}

void MetalCommandBuffer::BeginRenderPass(const ClearColor& clear) {
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];

    // Depth-only (shadow) pass: colorTex_ is nil -> no color attachment, and the depth must be
    // STORED so the lit pass can sample it. The scene/post passes have a color attachment and only
    // need depth transiently (store = DontCare).
    const bool depthOnly = (colorTex_ == nil);
    if (!depthOnly) {
        rpd.colorAttachments[0].texture = colorTex_;
        rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(clear.r, clear.g, clear.b, clear.a);
    }

    rpd.depthAttachment.texture = depthTex_;
    rpd.depthAttachment.loadAction = MTLLoadActionClear;
    rpd.depthAttachment.storeAction = depthOnly ? MTLStoreActionStore : MTLStoreActionDontCare;
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
    boundPointList_ = p.pointList();

    // Joint-palette skinning set (set 2): bind the device's current palette UBO at the skinning
    // vertex-buffer slot, mirroring the Vulkan set-2 auto-bind. Only the skinned pipelines set this.
    if (p.usesJointPalette()) {
        [encoder_ setVertexBuffer:device_.currentJointPalette() offset:0 atIndex:kVbJointPalette];
    }

    // Fullscreen post pass: the [[vertex_id]]-generated triangle's winding depends on the clip
    // convention; disable culling so it is never back-face culled to black (mirrors Vulkan, which
    // sets cullMode NONE for fullscreen pipelines). BeginRenderPass defaults to cull-back.
    [encoder_ setCullMode:(p.cullNone() ? MTLCullModeNone : MTLCullModeBack)];

    // Bind the device's current per-frame UBO (set 0) to both stages, matching the Vulkan path
    // where BindPipeline auto-binds the frame set.
    if (boundFrameUniforms_) {
        id<MTLBuffer> ubo = device_.currentFrameUbo();
        [encoder_ setVertexBuffer:ubo offset:0 atIndex:kVbFrameUbo];
        [encoder_ setFragmentBuffer:ubo offset:0 atIndex:kFbFrameUbo];

        // Auto-bind the shadow map (depth texture + clamp-to-edge sampler) to the lit fragment
        // shader's shadow slots, mirroring the Vulkan per-frame set (set 0, bindings 1+2). Only
        // matters for the lit pass; the depth-only shadow pipeline has no fragment stage so the
        // binding is harmless there. Skip when no shadow map has been set (e.g. the plain RT path).
        if (MetalRenderTarget* sm = device_.currentShadowMap()) {
            [encoder_ setFragmentTexture:sm->sampledTexture() atIndex:kFragShadowTex];
            [encoder_ setFragmentSamplerState:sm->sampledSampler() atIndex:kFragShadowSmp];
        }
    }

    // Depth-only (shadow) pipeline: a modest depth bias pushes caster depths away from the light to
    // fight shadow acne (Metal's encoder-level equivalent of Vulkan's rasterization depthBias).
    if (p.depthOnly()) {
        [encoder_ setDepthBias:1.25f slopeScale:1.75f clamp:0.0f];
    }
}

void MetalCommandBuffer::BindVertexBuffer(IBuffer& buffer) {
    auto& b = static_cast<MetalBuffer&>(buffer);
    [encoder_ setVertexBuffer:b.handle() offset:0 atIndex:kVbVertex];
}

void MetalCommandBuffer::BindInstanceBuffer(IBuffer& buffer) {
    auto& b = static_cast<MetalBuffer&>(buffer);
    // Per-instance stream at the flat vertex-buffer slot kVbInstance (the pipeline's vertex
    // descriptor marks this slot per-instance). Mirrors Vulkan binding 1.
    [encoder_ setVertexBuffer:b.handle() offset:0 atIndex:kVbInstance];
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

void MetalCommandBuffer::BindMaterial(ITexture& base, ITexture& normalMap) {
    // Base-color at texture(0)/sampler(1) (same as BindTexture); the tangent-space normal map at
    // texture(3)/sampler(4), matching the generated lit.frag MSL (gNormalMap/gNormalSmp).
    auto* b = dynamic_cast<IMetalSampled*>(&base);
    auto* n = dynamic_cast<IMetalSampled*>(&normalMap);
    if (!b || !n) Fail("BindMaterial: texture is not an IMetalSampled");
    [encoder_ setFragmentTexture:b->sampledTexture() atIndex:kFragTexture];
    [encoder_ setFragmentSamplerState:b->sampledSampler() atIndex:kFragSampler];
    [encoder_ setFragmentTexture:n->sampledTexture() atIndex:kFragNormalTex];
    [encoder_ setFragmentSamplerState:n->sampledSampler() atIndex:kFragNormalSmp];
}

void MetalCommandBuffer::BindMaterialPBR(ITexture& base, ITexture& metalRough, ITexture& normalMap,
                                         ITexture& emissive, ITexture& occlusion) {
    // Full glTF metallic-roughness material: bind all five image+sampler pairs at the flat fragment
    // indices the generated lit_pbr.frag MSL declares (base 0/1, normal 3/4, metalRough 5/6,
    // emissive 7/8, occlusion 9/10).
    auto bind = [&](ITexture& t, uint32_t tex, uint32_t smp) {
        auto* s = dynamic_cast<IMetalSampled*>(&t);
        if (!s) Fail("BindMaterialPBR: texture is not an IMetalSampled");
        [encoder_ setFragmentTexture:s->sampledTexture() atIndex:tex];
        [encoder_ setFragmentSamplerState:s->sampledSampler() atIndex:smp];
    };
    bind(base,       kFragTexture,        kFragSampler);
    bind(normalMap,  kFragNormalTex,      kFragNormalSmp);
    bind(metalRough, kFragMetalRoughTex,  kFragMetalRoughSmp);
    bind(emissive,   kFragEmissiveTex,    kFragEmissiveSmp);
    bind(occlusion,  kFragOcclusionTex,   kFragOcclusionSmp);
}

void MetalCommandBuffer::BindEnvironment(ITexture& env) {
    // HDR equirect environment map (Slice R) for image-based lighting: bind at the dedicated
    // fragment texture(11)/sampler(12), matching the generated sky_hdr/lit_pbr_ibl MSL (gEnv/gEnvSmp).
    auto* s = dynamic_cast<IMetalSampled*>(&env);
    if (!s) Fail("BindEnvironment: texture is not an IMetalSampled");
    [encoder_ setFragmentTexture:s->sampledTexture() atIndex:kFragEnvTex];
    [encoder_ setFragmentSamplerState:s->sampledSampler() atIndex:kFragEnvSmp];
}

void MetalCommandBuffer::Draw(uint32_t vertexCount, uint32_t firstVertex) {
    // Point list (GPU particles) vs the default triangle list (geometry / fullscreen tris).
    MTLPrimitiveType prim = boundPointList_ ? MTLPrimitiveTypePoint : MTLPrimitiveTypeTriangle;
    [encoder_ drawPrimitives:prim vertexStart:firstVertex vertexCount:vertexCount];
}

void MetalCommandBuffer::DrawIndexed(uint32_t indexCount, uint32_t firstIndex,
                                     int32_t vertexOffset) {
    // RHI index buffers are uint32 (Vulkan binds VK_INDEX_TYPE_UINT32).
    const NSUInteger indexBytes = (NSUInteger)firstIndex * sizeof(uint32_t);
    [encoder_ drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                         indexCount:indexCount
                          indexType:MTLIndexTypeUInt32
                        indexBuffer:indexBuffer_
                  indexBufferOffset:indexBytes
                      instanceCount:1
                         baseVertex:vertexOffset
                       baseInstance:0];
}

void MetalCommandBuffer::DrawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount,
                                              uint32_t firstIndex, int32_t vertexOffset,
                                              uint32_t firstInstance) {
    const NSUInteger indexBytes = (NSUInteger)firstIndex * sizeof(uint32_t);
    [encoder_ drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                         indexCount:indexCount
                          indexType:MTLIndexTypeUInt32
                        indexBuffer:indexBuffer_
                  indexBufferOffset:indexBytes
                      instanceCount:instanceCount
                         baseVertex:vertexOffset
                       baseInstance:firstInstance];
}

void MetalCommandBuffer::PushConstants(const void* data, uint32_t size) {
    // Vertex push constants (model matrix) -> inline setVertexBytes at the push-constant slot.
    [encoder_ setVertexBytes:data length:size atIndex:kVbPushConst];
}

void MetalCommandBuffer::SetScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) {
    // Override the render pass's full-extent scissor (the editor isn't used on Metal, but the seam
    // stays complete so the Metal build compiles + the golden is unaffected). Clamp to the attachment.
    int32_t cx = x < 0 ? 0 : x;
    int32_t cy = y < 0 ? 0 : y;
    uint32_t cw = width, ch = height;
    if ((uint32_t)cx + cw > width_)  cw = (cx < (int32_t)width_)  ? (width_  - (uint32_t)cx) : 0;
    if ((uint32_t)cy + ch > height_) ch = (cy < (int32_t)height_) ? (height_ - (uint32_t)cy) : 0;
    MTLScissorRect r{(NSUInteger)cx, (NSUInteger)cy, (NSUInteger)cw, (NSUInteger)ch};
    [encoder_ setScissorRect:r];
}

void MetalCommandBuffer::EndRenderPass() {
    [encoder_ endEncoding];
    encoder_ = nil;
}

// --- Compute recording. Metal forbids an open compute encoder and render encoder simultaneously,
// so the compute encoder is opened on BindComputePipeline and closed on ComputeToVertexBarrier
// (before BeginRenderPass opens the render encoder). Metal auto-tracks the hazard between the
// compute encoder's storage-buffer writes and the render encoder's vertex reads of the same buffer,
// so no explicit barrier object is needed — closing the encoder is the ordering point. ---

void MetalCommandBuffer::BindComputePipeline(IComputePipeline& pipeline) {
    auto& p = static_cast<MetalComputePipeline&>(pipeline);
    if (!computeEncoder_) {
        computeEncoder_ = [cmd_ computeCommandEncoder];
        if (!computeEncoder_) Fail("computeCommandEncoder failed");
    }
    [computeEncoder_ setComputePipelineState:p.state()];
}

void MetalCommandBuffer::BindStorageBuffer(IBuffer& buffer, uint32_t index) {
    auto& b = static_cast<MetalBuffer&>(buffer);
    // Storage buffers bind at kCsStorage + index (the particle SSBO is at buffer(0)).
    [computeEncoder_ setBuffer:b.handle() offset:0 atIndex:kCsStorage + index];
}

void MetalCommandBuffer::ComputePushConstants(const void* data, uint32_t size) {
    // Compute params (dt/time/count) -> inline setBytes at the params buffer slot.
    [computeEncoder_ setBytes:data length:size atIndex:kCsPushConst];
}

void MetalCommandBuffer::DispatchCompute(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) {
    MTLSize groups = MTLSizeMake(groupsX, groupsY, groupsZ);
    MTLSize perGroup = MTLSizeMake(computeThreadsPerGroup_, 1, 1);  // [numthreads(64,1,1)]
    [computeEncoder_ dispatchThreadgroups:groups threadsPerThreadgroup:perGroup];
}

void MetalCommandBuffer::ComputeToVertexBarrier() {
    // Close the compute encoder; the next render encoder (BeginRenderPass) reads the written buffer,
    // and Metal's automatic hazard tracking orders the dependency across encoders.
    if (computeEncoder_) {
        [computeEncoder_ endEncoding];
        computeEncoder_ = nil;
    }
}

} // namespace hf::rhi::mtl
