#include "rhi_metal/metal_pipeline.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_shader.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

MetalPipeline::MetalPipeline(MetalDevice& device, const GraphicsPipelineDesc& desc)
    : usesFrameUniforms_(desc.usesFrameUniforms), usesTexture_(desc.usesTexture),
      usesJointPalette_(desc.usesJointPalette),
      fullscreen_(desc.fullscreen), depthOnly_(desc.depthOnly), pointList_(desc.pointList),
      cullNone_(desc.cullNone) {
    id<MTLDevice> dev = device.device();

    auto* vs = static_cast<MetalShaderModule*>(desc.vertex);
    // Depth-only shadow pipeline: no fragment stage (writes only depth). desc.fragment may be null.
    auto* fs = static_cast<MetalShaderModule*>(desc.fragment);

    // --- Vertex descriptor from the RHI vertex layout. ---
    // Attribute shader index == RHI location. Buffer slot for vertex data == kVbVertex (0).
    // Fullscreen post pass: no vertex input — the MSL post vertex shader generates the triangle
    // from [[vertex_id]], so leave the vertex descriptor nil (matches Vulkan's empty vertex input).
    MTLVertexDescriptor* vd = nil;
    if (!desc.fullscreen) {
        vd = [[MTLVertexDescriptor alloc] init];
        for (const auto& a : desc.vertexLayout.attributes) {
            vd.attributes[a.location].format = ToMetalVertexFormat(a.format);
            vd.attributes[a.location].offset = a.offset;
            vd.attributes[a.location].bufferIndex = kVbVertex;
        }
        vd.layouts[kVbVertex].stride = desc.vertexLayout.stride;
        vd.layouts[kVbVertex].stepFunction = MTLVertexStepFunctionPerVertex;
        vd.layouts[kVbVertex].stepRate = 1;

        // Optional per-instance stream (binding 1 -> Metal vertex buffer slot kVbInstance). Its
        // attributes (a mat4 transform at locations 7-10) advance once per instance, mirroring
        // Vulkan's VK_VERTEX_INPUT_RATE_INSTANCE on binding 1.
        if (!desc.instanceLayout.attributes.empty()) {
            for (const auto& a : desc.instanceLayout.attributes) {
                vd.attributes[a.location].format = ToMetalVertexFormat(a.format);
                vd.attributes[a.location].offset = a.offset;
                vd.attributes[a.location].bufferIndex = kVbInstance;
            }
            vd.layouts[kVbInstance].stride = desc.instanceLayout.stride;
            vd.layouts[kVbInstance].stepFunction = MTLVertexStepFunctionPerInstance;
            vd.layouts[kVbInstance].stepRate = 1;
        }
    }

    // --- Render pipeline state. ---
    MTLRenderPipelineDescriptor* rpd = [[MTLRenderPipelineDescriptor alloc] init];
    rpd.vertexFunction = vs->function();
    // Depth-only (shadow) pipeline: no fragment function, no color attachment — just depth.
    rpd.fragmentFunction = desc.depthOnly ? nil : fs->function();
    rpd.vertexDescriptor = vd;
    if (!desc.depthOnly) {
        rpd.colorAttachments[0].pixelFormat = ToMetalPixelFormat(desc.colorFormat);
        if (desc.additiveBlend) {
            // src*1 + dst*1: glowing particles accumulate over the scene (mirrors Vulkan).
            rpd.colorAttachments[0].blendingEnabled = YES;
            rpd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
            rpd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
            rpd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
            rpd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
            rpd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            rpd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
        } else if (desc.alphaBlend) {
            // Standard alpha blend (ImGui/UI): src*srcAlpha + dst*(1-srcAlpha). Mirrors Vulkan.
            rpd.colorAttachments[0].blendingEnabled = YES;
            rpd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
            rpd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
            rpd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            rpd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            rpd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            rpd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        }
    }
    if (desc.depthTest || desc.depthOnly) {
        rpd.depthAttachmentPixelFormat = ToMetalPixelFormat(desc.depthFormat);
    }

    NSError* err = nil;
    pipelineState_ = [dev newRenderPipelineStateWithDescriptor:rpd error:&err];
    if (!pipelineState_) {
        std::string msg = "newRenderPipelineStateWithDescriptor failed";
        if (err) msg += std::string(": ") + [[err localizedDescription] UTF8String];
        Fail(msg);
    }

    // --- Depth-stencil state (LESS + write, matching the Vulkan pipeline). ---
    MTLDepthStencilDescriptor* dsd = [[MTLDepthStencilDescriptor alloc] init];
    if (desc.depthTest || desc.depthOnly) {
        dsd.depthCompareFunction = MTLCompareFunctionLess;
        dsd.depthWriteEnabled = YES;
    } else {
        dsd.depthCompareFunction = MTLCompareFunctionAlways;
        dsd.depthWriteEnabled = NO;
    }
    depthState_ = [dev newDepthStencilStateWithDescriptor:dsd];
    if (!depthState_) Fail("newDepthStencilStateWithDescriptor failed");
}

MetalPipeline::~MetalPipeline() {
    // ARC releases pipelineState_ / depthState_.
}

} // namespace hf::rhi::mtl
