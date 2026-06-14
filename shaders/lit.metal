// Hand-written MSL equivalent of shaders/lit.vert.hlsl + shaders/lit.frag.hlsl.
//
// Binding convention (must match engine/rhi_metal/metal_common.h + the command-buffer calls):
//   vertex:   buffer(0) = vertex stream (stage_in via [[stage_in]])
//             buffer(1) = FrameData UBO
//             buffer(2) = push constants (model matrix), set via setVertexBytes
//   fragment: buffer(0) = FrameData UBO
//             texture(0) / sampler(0) = material
//
// Matrices: the engine is column-major (math::Mat4 stores columns in m[16]); MSL float4x4 is
// also column-major and is constructed from 4 column float4s, so the raw 16 floats map 1:1.
// HLSL mul(M, v) with column-major M == MSL (M * v), preserved below.

#include <metal_stdlib>
using namespace metal;

struct VSInput {
    float3 pos    [[attribute(0)]];
    float3 color  [[attribute(1)]];
    float2 uv     [[attribute(2)]];
    float3 normal [[attribute(3)]];
};

struct VSOutput {
    float4 clip [[position]];
    float3 color;
    float2 uv;
    float3 wnormal;
    float3 wpos;
};

struct FrameData {
    float4x4 viewProj;
    float4   lightDir;
    float4   lightColor;
    float4   viewPos;
};

struct PushConstants {
    float4x4 model;
};

vertex VSOutput vertex_main(VSInput in [[stage_in]],
                            constant FrameData&    frame [[buffer(1)]],
                            constant PushConstants& pc   [[buffer(2)]]) {
    VSOutput out;
    float4 world = pc.model * float4(in.pos, 1.0);
    out.wpos = world.xyz;
    out.clip = frame.viewProj * world;
    // NDC-Y convention fix, owned by the Metal backend. The engine's math::Perspective bakes in
    // the Vulkan clip-space Y flip (m[5] = -1/t): Vulkan NDC has +Y pointing DOWN. Metal's NDC
    // (like OpenGL) has +Y pointing UP, so the shared view-proj would render upside-down on Metal.
    // We undo the flip here, in the Metal-only shader, so the fix lives entirely in the Metal path
    // and the engine-side math + the Vulkan backend stay untouched. (Equivalent to negating
    // viewProj's row 1, but doing it on the output clip vector keeps it a single, obvious line.)
    out.clip.y = -out.clip.y;
    // Rotation + uniform scale only (matches the HLSL note). Use the upper-left 3x3 of model.
    float3x3 nmat = float3x3(pc.model[0].xyz, pc.model[1].xyz, pc.model[2].xyz);
    out.wnormal = normalize(nmat * in.normal);
    out.color = in.color;
    out.uv = in.uv;
    return out;
}

fragment float4 fragment_main(VSOutput in [[stage_in]],
                              constant FrameData& frame [[buffer(0)]],
                              texture2d<float> tex      [[texture(0)]],
                              sampler smp               [[sampler(0)]]) {
    float3 N = normalize(in.wnormal);
    float3 L = normalize(-frame.lightDir.xyz);
    float  diff = max(dot(N, L), 0.0);
    float3 V = normalize(frame.viewPos.xyz - in.wpos);
    float3 H = normalize(L + V);
    float  spec = pow(max(dot(N, H), 0.0), 32.0);
    float  ambient = 0.15;
    float3 texColor = tex.sample(smp, in.uv).rgb * in.color;
    float3 rgb = texColor * ((ambient + diff) * frame.lightColor.rgb)
               + spec * frame.lightColor.rgb * 0.4;
    return float4(rgb, 1.0);
}
