#include "rhi_vulkan/vulkan_pipeline.h"
#include "rhi_vulkan/vulkan_shader.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"
#include <vector>

namespace hf::rhi::vk {

VulkanPipeline::VulkanPipeline(VulkanDevice& device, const GraphicsPipelineDesc& desc)
    : device_(device.device()), hasFrameSet_(desc.usesFrameUniforms),
      hasJointSet_(desc.usesJointPalette), hasEnvironmentSet_(desc.usesEnvironment),
      hasClusterSet_(desc.usesLightClusters), hasPerDrawSet_(desc.usesPerDrawData),
      hasBindlessSet_(desc.usesBindlessTextures) {
    auto* vs = static_cast<VulkanShaderModule*>(desc.vertex);
    auto* fs = static_cast<VulkanShaderModule*>(desc.fragment);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs->handle();
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs->handle();
    stages[1].pName = "main";

    // Binding 0 = per-vertex stream. Binding 1 (optional) = per-instance stream, present when the
    // desc carries a non-empty instanceLayout (e.g. the instanced lit pipeline's mat4 transform).
    const bool hasInstance = !desc.instanceLayout.attributes.empty();
    std::vector<VkVertexInputBindingDescription> bindings;
    bindings.push_back({0, desc.vertexLayout.stride, VK_VERTEX_INPUT_RATE_VERTEX});
    if (hasInstance) {
        bindings.push_back({1, desc.instanceLayout.stride, VK_VERTEX_INPUT_RATE_INSTANCE});
    }

    std::vector<VkVertexInputAttributeDescription> attrs;
    for (const auto& a : desc.vertexLayout.attributes) {
        attrs.push_back({a.location, 0, ToVk(a.format), a.offset});
    }
    if (hasInstance) {
        for (const auto& a : desc.instanceLayout.attributes) {
            attrs.push_back({a.location, 1, ToVk(a.format), a.offset});
        }
    }

    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    if (desc.fullscreen) {
        // Fullscreen pass: vertices are generated in the shader from SV_VertexID — no inputs.
        vi.vertexBindingDescriptionCount = 0;
        vi.vertexAttributeDescriptionCount = 0;
    } else {
        vi.vertexBindingDescriptionCount = (uint32_t)bindings.size();
        vi.pVertexBindingDescriptions = bindings.data();
        vi.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
        vi.pVertexAttributeDescriptions = attrs.data();
    }

    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    // Topology: line list (debug-draw, Slice W) > point list (particles) > triangle list (default).
    ia.topology = desc.lineList  ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                : desc.pointList ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                                 : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    // Fullscreen passes generate a single triangle from SV_VertexID whose winding depends on the
    // clip-space convention; disable culling so it is never back-face culled to black.
    rs.cullMode = (desc.fullscreen || desc.cullNone) ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    // Depth-only shadow pass: enable depth bias to push shadow-caster depths away from the light
    // and fight shadow acne. Keep back-face culling (the shader bias does the rest).
    if (desc.depthOnly) {
        rs.depthBiasEnable = VK_TRUE;
        rs.depthBiasConstantFactor = 1.25f;
        rs.depthBiasSlopeFactor = 1.75f;
    }

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (desc.additiveBlend) {
        // src*1 + dst*1: glowing particles accumulate over the scene.
        blendAtt.blendEnable = VK_TRUE;
        blendAtt.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAtt.colorBlendOp = VK_BLEND_OP_ADD;
        blendAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAtt.alphaBlendOp = VK_BLEND_OP_ADD;
    } else if (desc.alphaBlend) {
        // Standard alpha blend (ImGui/UI): src*srcAlpha + dst*(1-srcAlpha).
        blendAtt.blendEnable = VK_TRUE;
        blendAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAtt.colorBlendOp = VK_BLEND_OP_ADD;
        blendAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAtt.alphaBlendOp = VK_BLEND_OP_ADD;
    } else if (desc.oitRevealageBlend) {
        // Weighted-Blended-OIT revealage (Slice CO): src*0 + dst*(1-srcAlpha) -> dst *= (1-srcAlpha),
        // the order-independent revealage PRODUCT Π(1-alpha) over the transparent set.
        blendAtt.blendEnable = VK_TRUE;
        blendAtt.srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAtt.colorBlendOp = VK_BLEND_OP_ADD;
        blendAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAtt.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    // Depth-only pass has no color attachment.
    cb.attachmentCount = desc.depthOnly ? 0 : 1;
    cb.pAttachments = desc.depthOnly ? nullptr : &blendAtt;

    VkPipelineDepthStencilStateCreateInfo ds{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = desc.depthTest ? VK_TRUE : VK_FALSE;
    // Depth WRITE is gated by both depthTest and the new depthWrite flag. depthWrite defaults true,
    // so depth-tested pipelines write depth exactly as before; the translucent pass sets it false to
    // depth-test (read) without writing (Slice T).
    ds.depthWriteEnable = (desc.depthTest && desc.depthWrite) ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    VkPushConstantRange pushRange{};
    // Vertex-only by default (byte-for-byte unchanged for every existing pipeline). The bloom
    // fullscreen passes (Slice U) read their per-pass params in the FRAGMENT stage, so they widen
    // the range to VERTEX|FRAGMENT; PushConstants then targets both stages for that pipeline.
    pushRange.stageFlags = desc.fragmentPushConstants
                               ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                               : VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = desc.pushConstantSize;
    pushConstantStages_ = pushRange.stageFlags;

    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    if (desc.pushConstantSize > 0) {
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pushRange;
    }
    // Set-layout array. Push order = set index: frame (set 0), material (set 1), joint palette
    // (set 2). Descriptor sets must occupy consecutive indices, so when the joint set (set 2) is
    // present without a material set (set 1) — e.g. the skinned depth-only shadow pipeline — set 1
    // is filled with the material layout as a harmless placeholder to keep set 2 at index 2.
    std::vector<VkDescriptorSetLayout> setLayouts;
    if (desc.usesFrameUniforms) setLayouts.push_back(device.frameSetLayout());   // set 0
    // set 1: the lit-PBR pipeline uses the WIDER full-PBR material set layout; everything else uses
    // the existing 2-texture material set (also the joint-set placeholder). When usesEnvironment is
    // set but there is no material set (the HDR sky pipeline), a material-layout placeholder fills
    // set 1 so the dedicated environment set always sits at index 3.
    if (desc.usesJointPalette && !desc.usesTexture)
        setLayouts.push_back(device.materialSetLayout());                        // set 1 placeholder
    else if (desc.usesTexture)
        setLayouts.push_back(desc.pbrMaterial ? device.pbrMaterialSetLayout()
                                              : device.materialSetLayout());     // set 1
    else if (desc.usesEnvironment || desc.usesLightClusters)
        setLayouts.push_back(device.materialSetLayout());                        // set 1 placeholder (env/cluster)
    // set 2: the joint palette, OR the MDI per-draw SSBO set (Slice BM), OR (for the IBL / clustered
    // pipelines, which have no skinning but need set 3 to sit at index 3) a harmless placeholder so
    // descriptor sets stay at consecutive indices.
    if (desc.usesJointPalette) setLayouts.push_back(device.jointPaletteSetLayout()); // set 2
    else if (desc.usesPerDrawData) setLayouts.push_back(device.perDrawSetLayout());  // set 2 (MDI per-draw)
    else if (desc.usesEnvironment || desc.usesLightClusters || desc.usesBindlessTextures)
        setLayouts.push_back(device.jointPaletteSetLayout());                       // set 2 placeholder
    // set 3: the dedicated HDR environment set (Slice R) OR the clustered-lighting set (Slice AG).
    // They are mutually exclusive (both occupy index 3); a pipeline declares at most one. The bindless
    // pipeline (Slice BZ) needs a set-3 placeholder so its bindless set sits at index 4.
    if (desc.usesEnvironment) setLayouts.push_back(device.environmentSetLayout()); // set 3 (env)
    else if (desc.usesLightClusters) setLayouts.push_back(device.clusterSetLayout()); // set 3 (clusters)
    else if (desc.usesBindlessTextures) setLayouts.push_back(device.environmentSetLayout()); // set 3 placeholder
    // set 4: the dedicated bindless texture array set (Slice BZ). Only the bindless lit pipeline.
    if (desc.usesBindlessTextures) setLayouts.push_back(device.bindlessSetLayout());  // set 4 (bindless)
    // Issue #34: the dedicated RT-graphics accel set — appended at the NEXT free index past every set
    // above (NOT a cluster binding; the cluster set 3 stays the spheres/aabbs SSBOs). With the RT
    // showcase (frame set 0 + cluster placeholders 1/2 + cluster set 3) it lands at index 4. A ps_6_5
    // RayQuery fragment shader reads the TLAS pushed here at VK_PIPELINE_BIND_POINT_GRAPHICS.
    if (desc.accelStructureBinding >= 0) {
        accelSetIndex_ = (uint32_t)setLayouts.size();
        hasAccelSet_ = true;
        setLayouts.push_back(device.accelGraphicsSetLayout((uint32_t)desc.accelStructureBinding));
    }
    if (!setLayouts.empty()) {
        lci.setLayoutCount = (uint32_t)setLayouts.size();
        lci.pSetLayouts = setLayouts.data();
    }
    Check(vkCreatePipelineLayout(device_, &lci, nullptr, &layout_), "vkCreatePipelineLayout");

    VkFormat colorFormat = ToVk(desc.colorFormat);
    VkPipelineRenderingCreateInfo rendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    // Depth-only pass: no color attachment format, only the D32 depth target.
    rendering.colorAttachmentCount = desc.depthOnly ? 0 : 1;
    rendering.pColorAttachmentFormats = desc.depthOnly ? nullptr : &colorFormat;
    rendering.depthAttachmentFormat = ToVk(desc.depthFormat);

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.pNext = &rendering;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &dyn;
    pci.layout = layout_;
    Check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_),
          "vkCreateGraphicsPipelines");
}

VulkanPipeline::~VulkanPipeline() {
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
}

} // namespace hf::rhi::vk
