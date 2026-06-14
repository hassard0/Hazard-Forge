// Hand-written MSL equivalent of shaders/shadow.vert.hlsl: the depth-only shadow pass.
//
// Transforms scene geometry into the directional light's clip space; the render pipeline has NO
// fragment function (MetalPipeline builds it with desc.depthOnly), so only depth is written into
// the shadow map. Only POSITION is consumed, but the full scene vertex layout is declared so this
// binds against the same vertex buffers as the lit pass.
//
// Binding convention (must match engine/rhi_metal/metal_common.h + the command-buffer calls):
//   vertex: buffer(0) = vertex stream (stage_in)
//           buffer(1) = FrameData UBO  (lightViewProj lives here)
//           buffer(2) = push constants (model matrix), via setVertexBytes
//
// Y convention: lightViewProj is built from math::Ortho, which bakes the same Vulkan clip-space
// Y-flip as math::Perspective (m[5] negative, +Y down). Metal NDC is +Y up, so — exactly like
// lit.metal's vertex shader — we undo the flip on the output clip (out.clip.y = -out.clip.y). This
// makes the shadow texture store light-space geometry right-side-up (row 0 = NDC +Y). The lit pass
// samples with smUV = proj.xy*0.5+0.5: the render flip and that sample formula are self-consistent
// (the flip cancels), so the same UV math as the Vulkan backend works unchanged. See lit.metal.

#include <metal_stdlib>
using namespace metal;

struct VSInput {
    float3 pos    [[attribute(0)]];
    float3 color  [[attribute(1)]];
    float2 uv     [[attribute(2)]];
    float3 normal [[attribute(3)]];
};

// MUST match shaders/lit.metal's FrameData (and the C++ FrameData) byte-for-byte: the depth pass
// and the lit pass read the SAME per-frame UBO, so lightViewProj must sit at the same offset.
struct FrameData {
    float4x4 viewProj;
    float4   lightDir;
    float4   lightColor;
    float4   viewPos;
    float4   ptCount;
    float4   ptPos[3];
    float4   ptColor[3];
    float4x4 lightViewProj;
    float4   camFwd;
    float4   camRight;
    float4   camUp;
    float4   skyParams;
};

struct PushConstants {
    float4x4 model;
};

struct ShadowVSOut {
    float4 clip [[position]];
};

vertex ShadowVSOut shadow_vertex(VSInput in [[stage_in]],
                                 constant FrameData&     frame [[buffer(1)]],
                                 constant PushConstants& pc    [[buffer(2)]]) {
    ShadowVSOut out;
    float4 world = pc.model * float4(in.pos, 1.0);
    out.clip = frame.lightViewProj * world;
    out.clip.y = -out.clip.y;   // undo the Ortho Vulkan Y-flip for Metal +Y-up NDC (see header).
    return out;
}
