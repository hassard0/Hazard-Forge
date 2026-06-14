#include "rhi_metal/metal_pipeline.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_shader.h"
#include "rhi_metal/metal_common.h"

namespace hf::rhi::mtl {

MetalPipeline::MetalPipeline(MetalDevice& device, const GraphicsPipelineDesc& desc)
    : usesFrameUniforms_(desc.usesFrameUniforms), usesTexture_(desc.usesTexture) {
    id<MTLDevice> dev = device.device();

    auto* vs = static_cast<MetalShaderModule*>(desc.vertex);
    auto* fs = static_cast<MetalShaderModule*>(desc.fragment);

    // --- Vertex descriptor from the RHI vertex layout. ---
    // Attribute shader index == RHI location. Buffer slot for vertex data == kVbVertex (0).
    MTLVertexDescriptor* vd = [[MTLVertexDescriptor alloc] init];
    for (const auto& a : desc.vertexLayout.attributes) {
        vd.attributes[a.location].format = ToMetalVertexFormat(a.format);
        vd.attributes[a.location].offset = a.offset;
        vd.attributes[a.location].bufferIndex = kVbVertex;
    }
    vd.layouts[kVbVertex].stride = desc.vertexLayout.stride;
    vd.layouts[kVbVertex].stepFunction = MTLVertexStepFunctionPerVertex;
    vd.layouts[kVbVertex].stepRate = 1;

    // --- Render pipeline state. ---
    MTLRenderPipelineDescriptor* rpd = [[MTLRenderPipelineDescriptor alloc] init];
    rpd.vertexFunction = vs->function();
    rpd.fragmentFunction = fs->function();
    rpd.vertexDescriptor = vd;
    rpd.colorAttachments[0].pixelFormat = ToMetalPixelFormat(desc.colorFormat);
    if (desc.depthTest) {
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
    if (desc.depthTest) {
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
