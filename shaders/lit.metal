// Hand-written MSL equivalent of shaders/lit.vert.hlsl + shaders/lit.frag.hlsl.
//
// Binding convention (must match engine/rhi_metal/metal_common.h + the command-buffer calls):
//   vertex:   buffer(0) = vertex stream (stage_in via [[stage_in]])
//             buffer(1) = FrameData UBO
//             buffer(2) = push constants (model matrix), set via setVertexBytes
//   fragment: buffer(0) = FrameData UBO
//             texture(0) / sampler(0) = material
//             texture(1) / sampler(1) = shadow map (depth) + clamp-to-edge sampler
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

// MUST match the C++ FrameData (metal_headless/visual_test.mm) AND shaders/shadow.metal byte-for-
// byte: all three read the same per-frame UBO. Layout mirrors the Vulkan shaders/lit.frag.hlsl
// FrameData (288 bytes), so lightViewProj sits at a consistent offset for the shadow pass.
struct FrameData {
    float4x4 viewProj;
    float4   lightDir;
    float4   lightColor;
    float4   viewPos;
    float4   ptCount;          // x = number of active point lights (unused on Metal so far)
    float4   ptPos[3];
    float4   ptColor[3];
    float4x4 lightViewProj;    // directional light's view*ortho, for shadow mapping
    float4   camFwd;           // sky/camera-basis fields (unused here; kept for layout parity)
    float4   camRight;
    float4   camUp;
    float4   skyParams;
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
                              texture2d<float> tex       [[texture(0)]],
                              sampler smp                [[sampler(0)]],
                              texture2d<float> shadowTex [[texture(1)]],
                              sampler shadowSmp          [[sampler(1)]]) {
    float3 N = normalize(in.wnormal);
    float3 L = normalize(-frame.lightDir.xyz);
    float  diff = max(dot(N, L), 0.0);
    float3 V = normalize(frame.viewPos.xyz - in.wpos);
    float3 H = normalize(L + V);
    float  spec = pow(max(dot(N, H), 0.0), 32.0);
    float  ambient = 0.15;
    float3 texColor = tex.sample(smp, in.uv).rgb * in.color;

    // --- Directional shadow: project world pos into the light's clip space, 3x3 PCF compare. ---
    // lightViewProj uses math::Ortho (the same Vulkan Y-flip as Perspective); the shadow pass undid
    // that flip when rendering (shadow.metal: clip.y = -clip.y). The render flip and the sample
    // formula below are self-consistent, so smUV = proj.xy*0.5+0.5 matches the shadow texel layout
    // directly — identical to the Vulkan backend (shaders/lit.frag.hlsl). Depth is [0,1] (Ortho).
    float shadow = 1.0;
    {
        float4 lp = frame.lightViewProj * float4(in.wpos, 1.0);
        float3 proj = lp.xyz / lp.w;
        float2 smUV = proj.xy * 0.5 + 0.5;
        float  curDepth = proj.z;
        if (smUV.x >= 0.0 && smUV.x <= 1.0 && smUV.y >= 0.0 && smUV.y <= 1.0 &&
            curDepth >= 0.0 && curDepth <= 1.0) {
            float bias = 0.0025;
            float s = 0.0;
            float texel = 1.0 / 2048.0;
            for (int sx = -1; sx <= 1; ++sx)
                for (int sy = -1; sy <= 1; ++sy) {
                    float d = shadowTex.sample(shadowSmp,
                                               smUV + float2(sx, sy) * texel).r;
                    s += (curDepth - bias > d) ? 0.0 : 1.0;
                }
            shadow = s / 9.0;
        }
    }

    // Ambient is unshadowed; the directional diffuse+spec is attenuated by the shadow factor
    // (mirrors shaders/lit.frag.hlsl).
    float3 rgb = texColor * (ambient * frame.lightColor.rgb)
               + shadow * (texColor * (diff * frame.lightColor.rgb)
                           + spec * frame.lightColor.rgb * 0.4);
    return float4(rgb, 1.0);
}
