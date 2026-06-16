#include "rhi_metal/metal_command_buffer.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_pipeline.h"
#include "rhi_metal/metal_compute_pipeline.h"
#include "rhi_metal/metal_buffer.h"
#include "rhi_metal/metal_texture.h"
#include "rhi_metal/metal_sampled.h"
#include "rhi_metal/metal_render_target.h"
#include "rhi_metal/metal_cubemap_target.h"
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
    parallelEncoder_ = nil;
    computeEncoder_ = nil;
    indexBuffer_ = nil;
    boundFrameUniforms_ = false;
    boundPointList_ = false;
    colorSlice_ = 0;  // Slice DD: normal 2D target (no cube-face slice) unless a face is selected
}

// Build the render-pass descriptor shared by the single-encoder and parallel-encoder paths.
static MTLRenderPassDescriptor* MakeRenderPassDescriptor(id<MTLTexture> colorTex,
                                                         id<MTLTexture> depthTex,
                                                         const ClearColor& clear) {
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    // Depth-only (shadow) pass: colorTex is nil -> no color attachment, and the depth must be
    // STORED so the lit pass can sample it. The scene/post passes have a color attachment and only
    // need depth transiently (store = DontCare).
    const bool depthOnly = (colorTex == nil);
    if (!depthOnly) {
        rpd.colorAttachments[0].texture = colorTex;
        rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(clear.r, clear.g, clear.b, clear.a);
    }
    rpd.depthAttachment.texture = depthTex;
    rpd.depthAttachment.loadAction = MTLLoadActionClear;
    rpd.depthAttachment.storeAction = depthOnly ? MTLStoreActionStore : MTLStoreActionDontCare;
    rpd.depthAttachment.clearDepth = 1.0;
    return rpd;
}

void MetalCommandBuffer::BeginRenderPass(const ClearColor& clear) {
    BeginRenderPass(clear, /*expectsSecondaries=*/false);
}

void MetalCommandBuffer::BeginRenderPass(const ClearColor& clear, bool expectsSecondaries) {
    MTLRenderPassDescriptor* rpd = MakeRenderPassDescriptor(colorTex_, depthTex_, clear);
    // Slice DD: render into a specific cube FACE by selecting the color attachment's array slice.
    if (colorTex_ != nil && colorSlice_ != 0)
        rpd.colorAttachments[0].slice = colorSlice_;

    if (expectsSecondaries) {
        // Slice AU: open a PARALLEL render command encoder. It vends N sub-encoders (one per worker)
        // that record concurrently and are committed in CREATION ORDER on endEncoding — so the draw
        // order is the worker-index order (deterministic, == single-threaded). The primary records
        // no draws directly; CreateSecondaryCommandBuffer pulls sub-encoders via nextParallelSubEncoder.
        parallelEncoder_ = [cmd_ parallelRenderCommandEncoderWithDescriptor:rpd];
        if (!parallelEncoder_) Fail("parallelRenderCommandEncoderWithDescriptor failed");
        encoder_ = nil;
        device_.SetActiveParallelRecorder(this);  // so CreateSecondaryCommandBuffer vends from here
        return;
    }

    encoder_ = [cmd_ renderCommandEncoderWithDescriptor:rpd];
    if (!encoder_) Fail("renderCommandEncoderWithDescriptor failed");

    // Match the Vulkan winding (CCW front face, back-face cull).
    [encoder_ setFrontFacingWinding:MTLWindingCounterClockwise];
    [encoder_ setCullMode:MTLCullModeBack];

    MTLViewport vp{0.0, 0.0, (double)width_, (double)height_, 0.0, 1.0};
    [encoder_ setViewport:vp];
}

id<MTLRenderCommandEncoder> MetalCommandBuffer::nextParallelSubEncoder() {
    if (!parallelEncoder_) Fail("nextParallelSubEncoder without an open parallel encoder");
    return [parallelEncoder_ renderCommandEncoder];
}

void MetalCommandBuffer::BeginSecondary(id<MTLRenderCommandEncoder> enc, uint32_t width,
                                        uint32_t height) {
    // Retarget this recorder onto a parallel sub-encoder (Slice AU). Configure the same fixed state
    // BeginRenderPass sets on the single encoder (CCW front face, back-face cull, full viewport) so a
    // worker's draws land identically to single-threaded recording.
    encoder_ = enc;
    width_ = width;
    height_ = height;
    indexBuffer_ = nil;
    boundFrameUniforms_ = false;
    boundPointList_ = false;
    [encoder_ setFrontFacingWinding:MTLWindingCounterClockwise];
    [encoder_ setCullMode:MTLCullModeBack];
    MTLViewport vp{0.0, 0.0, (double)width_, (double)height_, 0.0, 1.0};
    [encoder_ setViewport:vp];
}

void MetalCommandBuffer::ExecuteSecondaries(std::span<ICommandBuffer* const> secondaries) {
    // Slice AU: end each worker's sub-encoder, ON THE PRIMARY THREAD after the join barrier (so no two
    // threads touch the parallel encoder concurrently). The parallel encoder then commits the
    // sub-encoders in CREATION ORDER (== worker index) when EndRenderPass calls endEncoding — giving
    // the deterministic, single-threaded-equivalent draw order.
    for (ICommandBuffer* s : secondaries) {
        auto* ms = static_cast<MetalCommandBuffer*>(s);
        if (ms->encoder_) { [ms->encoder_ endEncoding]; ms->encoder_ = nil; }
    }
}

void MetalCommandBuffer::BindPipeline(IPipeline& pipeline) {
    auto& p = static_cast<MetalPipeline&>(pipeline);
    [encoder_ setRenderPipelineState:p.pipelineState()];
    [encoder_ setDepthStencilState:p.depthState()];
    boundFrameUniforms_ = p.usesFrameUniforms();
    boundPointList_ = p.pointList();
    boundLineList_ = p.lineList();
    boundFragmentPushConst_ = p.fragmentPushConstants();

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

void MetalCommandBuffer::BindTexturePair(ITexture& primary, ITexture& secondary) {
    // Bloom composite (Slice U): the HDR scene at texture(0)/sampler(1) (gTex) and the bloom result
    // at texture(3)/sampler(4) (gTex2 — the same second material slot BindMaterial uses).
    auto* p = dynamic_cast<IMetalSampled*>(&primary);
    auto* s = dynamic_cast<IMetalSampled*>(&secondary);
    if (!p || !s) Fail("BindTexturePair: texture is not an IMetalSampled");
    [encoder_ setFragmentTexture:p->sampledTexture() atIndex:kFragTexture];
    [encoder_ setFragmentSamplerState:p->sampledSampler() atIndex:kFragSampler];
    [encoder_ setFragmentTexture:s->sampledTexture() atIndex:kFragNormalTex];
    [encoder_ setFragmentSamplerState:s->sampledSampler() atIndex:kFragNormalSmp];
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

void MetalCommandBuffer::BindReflectionProbe(ITexture& probeAtlas) {
    // Slice AK — bind the baked probe atlas (an offscreen RGBA16F render target) at the SAME flat
    // fragment env texture(11)/sampler(12) as BindEnvironment, so the generated lit_probe MSL reads it
    // as gProbe/gProbeSmp. A render target is an IMetalSampled, so this is identical to BindEnvironment.
    auto* s = dynamic_cast<IMetalSampled*>(&probeAtlas);
    if (!s) Fail("BindReflectionProbe: texture is not an IMetalSampled");
    [encoder_ setFragmentTexture:s->sampledTexture() atIndex:kFragEnvTex];
    [encoder_ setFragmentSamplerState:s->sampledSampler() atIndex:kFragEnvSmp];
}

void MetalCommandBuffer::BindCubemapProbe(ITexture& cubemap) {
    // Slice DD — bind the CAPTURED cubemap (a MetalCubemapTarget) at the SAME flat fragment env
    // texture(11)/sampler(12) as BindReflectionProbe, so the generated captureprobe MSL reads it as a
    // texturecube<float> gCube/gCubeSmp. A MetalCubemapTarget is an IMetalSampled whose sampledTexture
    // is the cube, so this is identical to BindReflectionProbe.
    auto* s = dynamic_cast<IMetalSampled*>(&cubemap);
    if (!s) Fail("BindCubemapProbe: texture is not an IMetalSampled");
    [encoder_ setFragmentTexture:s->sampledTexture() atIndex:kFragEnvTex];
    [encoder_ setFragmentSamplerState:s->sampledSampler() atIndex:kFragEnvSmp];
}

void MetalCommandBuffer::BindLightClusters(IBuffer& clusters, IBuffer& lightIndices,
                                           IBuffer& lights) {
    // Clustered Forward+ lighting (Slice AG): bind the three storage buffers to the FRAGMENT stage
    // at flat Metal fragment buffer slots 13/14/15, matching the generated lit_clustered MSL
    // (gClusters/gLightIndices/gLights). On Metal a storage buffer is just a setFragmentBuffer: at
    // a buffer index — no descriptor set. The shader reads them as device const arrays.
    auto& cb = static_cast<MetalBuffer&>(clusters);
    auto& lb = static_cast<MetalBuffer&>(lightIndices);
    auto& gb = static_cast<MetalBuffer&>(lights);
    [encoder_ setFragmentBuffer:cb.handle() offset:0 atIndex:kFragClusterBuf];
    [encoder_ setFragmentBuffer:lb.handle() offset:0 atIndex:kFragLightIndexBuf];
    [encoder_ setFragmentBuffer:gb.handle() offset:0 atIndex:kFragLightBuf];
}

void MetalCommandBuffer::Draw(uint32_t vertexCount, uint32_t firstVertex) {
    // Line list (debug-draw, Slice W) > point list (GPU particles) > the default triangle list
    // (geometry / fullscreen tris).
    MTLPrimitiveType prim = boundLineList_  ? MTLPrimitiveTypeLine
                          : boundPointList_ ? MTLPrimitiveTypePoint
                                            : MTLPrimitiveTypeTriangle;
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

void MetalCommandBuffer::DrawIndexedIndirect(IBuffer& argsBuffer, size_t offset) {
    // Slice AR — GPU-driven indexed draw: read the {indexCount, instanceCount, firstIndex,
    // vertexOffset, firstInstance} record (MTLDrawIndexedPrimitivesIndirectArguments) from the buffer
    // at `offset`. Single indirect draw — no indirect-command-buffer machinery. firstIndex is encoded
    // as the indexBufferOffset in BYTES (uint32 indices); the args' firstIndex field stays 0 (we pass
    // 0 here), matching the compute shader which writes firstIndex=0.
    auto& b = static_cast<MetalBuffer&>(argsBuffer);
    [encoder_ drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                          indexType:MTLIndexTypeUInt32
                        indexBuffer:indexBuffer_
                  indexBufferOffset:0
                     indirectBuffer:b.handle()
               indirectBufferOffset:(NSUInteger)offset];
}

void MetalCommandBuffer::DrawIndexedMultiIndirect(IBuffer& /*argsBuffer*/, uint32_t /*drawCount*/,
                                                  uint32_t /*stride*/) {
    // Slice BM — GPU multi-draw-indirect. The TRUE multi-draw + gl_DrawID-indexed per-draw data is the
    // VULKAN GPU-driven demonstration (one vkCmdDrawIndexedIndirect(drawCount=N)). Metal's equivalent
    // is an MTLIndirectCommandBuffer (ICB), which is OPTIONAL for this slice: the Metal golden renders
    // the IDENTICAL N-object scene via the working per-object path (visual_test --mdi), so mdi.png is
    // backend-identical to the Vulkan MDI image either way. This entry point is therefore a documented
    // no-op on Metal (no ICB wired); the showcase never routes the Metal frame through it.
}

void MetalCommandBuffer::PushConstants(const void* data, uint32_t size) {
    // Vertex push constants (model matrix) -> inline setVertexBytes at the push-constant slot.
    [encoder_ setVertexBytes:data length:size atIndex:kVbPushConst];
    // Bloom fullscreen passes (Slice U) read their per-pass params in the FRAGMENT stage; mirror the
    // Vulkan VERTEX|FRAGMENT push-constant range by also binding the bytes to the fragment slot.
    if (boundFragmentPushConst_) {
        [encoder_ setFragmentBytes:data length:size atIndex:kFbPushConst];
    }
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

void MetalCommandBuffer::SetViewport(int32_t x, int32_t y, uint32_t width, uint32_t height) {
    // Render the following draw(s) into a sub-rect of the attachment: set BOTH the viewport (so NDC
    // maps into the sub-rect) and a matching scissor (so nothing spills outside). Used by the CSM
    // shadow-atlas pass to target one cascade tile (Slice AD). Clamp to the attachment bounds.
    int32_t vx = x < 0 ? 0 : x;
    int32_t vy = y < 0 ? 0 : y;
    uint32_t vw = width, vh = height;
    if ((uint32_t)vx + vw > width_)  vw = (vx < (int32_t)width_)  ? (width_  - (uint32_t)vx) : 0;
    if ((uint32_t)vy + vh > height_) vh = (vy < (int32_t)height_) ? (height_ - (uint32_t)vy) : 0;
    MTLViewport vp{(double)vx, (double)vy, (double)vw, (double)vh, 0.0, 1.0};
    [encoder_ setViewport:vp];
    MTLScissorRect r{(NSUInteger)vx, (NSUInteger)vy, (NSUInteger)vw, (NSUInteger)vh};
    [encoder_ setScissorRect:r];
}

void MetalCommandBuffer::EndRenderPass() {
    if (parallelEncoder_) {
        // Slice AU: closing the parallel encoder commits its sub-encoders in creation order (== worker
        // index), giving the deterministic, single-threaded-equivalent draw order.
        [parallelEncoder_ endEncoding];
        parallelEncoder_ = nil;
        encoder_ = nil;
        return;
    }
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
    // Size the threadgroup from the pipeline's [numthreads(X,1,1)] width (Slice AR: GPU cull uses
    // 1024, one workgroup; particles use 64).
    computeThreadsPerGroup_ = p.threadsPerGroupX();
}

void MetalCommandBuffer::BindStorageBuffer(IBuffer& buffer, uint32_t index) {
    auto& b = static_cast<MetalBuffer&>(buffer);
    // Storage buffers bind at kCsStorage + index (the particle SSBO is at buffer(0)).
    [computeEncoder_ setBuffer:b.handle() offset:0 atIndex:kCsStorage + index];
}

void MetalCommandBuffer::BindShadowMapCompute(IRenderTarget& shadowMap) {
    // Slice CX: bind the sun's CSM shadow depth texture + its clamp sampler to the compute encoder at
    // texture(4)/sampler(5) (mirrors the Vulkan binding 4/5; spirv-cross --msl-decoration-binding lands
    // them there). The SAME depth map + sampler the lit pass samples. Metal's tracked-hazard model orders
    // the shadow-pass write -> this compute read across encoders (the shadow render encoder closed before
    // the compute encoder opened), so no explicit barrier object is needed.
    auto& sm = static_cast<MetalRenderTarget&>(shadowMap);
    [computeEncoder_ setTexture:sm.sampledTexture() atIndex:kCsShadowTex];
    [computeEncoder_ setSamplerState:sm.sampledSampler() atIndex:kCsShadowSmp];
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

void MetalCommandBuffer::ComputeToComputeBarrier() {
    // Slice CS froxel inject -> integrate. Close the compute encoder so the NEXT BindComputePipeline
    // opens a fresh compute encoder; Metal's tracked-hazard model orders the second encoder's reads of
    // the storage volume after the first encoder's writes. (Same ordering-by-encoder-boundary discipline
    // as ComputeToVertexBarrier — the boundary is the dependency point, no explicit barrier object.)
    if (computeEncoder_) {
        [computeEncoder_ endEncoding];
        computeEncoder_ = nil;
    }
}

void MetalCommandBuffer::ComputeToFragmentBarrier() {
    // Slice CS froxel integrate -> apply. Close the compute encoder so the following render encoder's
    // fragment reads of the integrated volume are ordered after the compute writes (tracked-hazard).
    if (computeEncoder_) {
        [computeEncoder_ endEncoding];
        computeEncoder_ = nil;
    }
}

void MetalCommandBuffer::ResourceBarrier(IRenderTarget& /*resource*/, ResourceState /*from*/,
                                         ResourceState /*to*/) {
    // Slice AS: HONEST NO-OP on Metal. The render graph's barrier solver computes the same inter-pass
    // transitions it does on Vulkan, but Metal resources default to MTLResourceHazardTrackingModeTracked,
    // so the driver automatically inserts the required execution/memory dependencies between an
    // encoder that WRITES a texture (shadow-map depth render, scene-color render) and a later encoder
    // that READS it (the lit / composite pass sampling it). The producer/consumer ordering point is
    // the encoder boundary (endEncoding), which the existing Begin*/End* scaffolding already creates
    // per pass. There is therefore no cross-encoder hazard left for an explicit barrier to cover, and
    // no Metal layout concept to transition — so this is correctly a no-op (NOT a fabricated barrier
    // that does nothing useful). If a future pass introduced an untracked resource or an intra-encoder
    // RAW/WAR hazard, this is where a [encoder memoryBarrierWithScope:] would go.
}

} // namespace hf::rhi::mtl
