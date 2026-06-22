// Hazard Forge — shared PBR lighting CORE for the data-driven material graph (Slice AV).
//
// This factors the metallic-roughness lighting core out of lit_pbr.frag.hlsl so the build-time
// codegen'd material shaders (shaders/generated/mat_*.frag.hlsl) reuse the EXACT same FrameData,
// varyings, descriptor bindings, and Cook-Torrance / IBL / shadow path. The generated shaders
// declare the same PSInput and call hfShadePBR() with the baseColor/metallic/roughness/emissive the
// graph computed. lit_pbr.frag.hlsl itself is intentionally NOT modified (its golden-locked SPIR-V/
// MSL must stay byte-identical), but this core mirrors its math so the graph shading is coherent.
//
// IMPORTANT: the bindings/varyings here are IDENTICAL to lit_pbr.frag.hlsl, so a material pipeline
// using a generated fragment reuses the existing PBR descriptor layout (set 0 frame + shadow, set 1
// material textures) with NO new RHI seam.
#ifndef HF_PBR_CORE_HLSLI
#define HF_PBR_CORE_HLSLI

// Shared procedural sky (HFSkyColor): the graph-material IBL below pulls the SAME sky as the sky pass
// and lit.frag, so a sky retune updates every reflection too (issue #4). See procedural_sky.hlsli.
#include "procedural_sky.hlsli"

struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    // skyParams: x=tanHalfFov, y=aspect, z=time(seconds), w=frameIndex (issue #5 time channel).
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
// Material set (set 1) — IDENTICAL bindings to lit_pbr.frag.hlsl. The graph's TextureSample nodes
// read the base-color texture (gBase 0/1); the wider set is declared so the existing PBR material
// pipeline/descriptor layout binds unchanged.
[[vk::binding(0, 1)]] Texture2D    gBase        : register(t0);  // base-color (sRGB)
[[vk::binding(1, 1)]] SamplerState gBaseSmp     : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap   : register(t3);  // tangent-space normal (linear)
[[vk::binding(4, 1)]] SamplerState gNormalSmp   : register(s3);
[[vk::binding(5, 1)]] Texture2D    gMetalRough  : register(t5);
[[vk::binding(6, 1)]] SamplerState gMetalRoughSmp : register(s5);
[[vk::binding(7, 1)]] Texture2D    gEmissive    : register(t7);
[[vk::binding(8, 1)]] SamplerState gEmissiveSmp : register(s7);
[[vk::binding(9, 1)]] Texture2D    gOcclusion   : register(t9);
[[vk::binding(10, 1)]] SamplerState gOcclusionSmp : register(s9);

struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    [[vk::location(4)]] nointerpolation float2 material : TEXCOORD1;
    [[vk::location(5)]] float3 wtangent : TANGENT;
};

static const float HF_PI = 3.14159265358979323846;

float hfDistributionGGX(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float d  = (NoH * NoH) * (a2 - 1.0) + 1.0;
    return a2 / max(HF_PI * d * d, 1e-7);
}
float hfGeometrySchlickGGX(float NoX, float k) { return NoX / (NoX * (1.0 - k) + k); }
float hfGeometrySmith(float NoV, float NoL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return hfGeometrySchlickGGX(NoV, k) * hfGeometrySchlickGGX(NoL, k);
}
float3 hfFresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (float3(1.0, 1.0, 1.0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}
// Thin wrapper over the SHARED HFSkyColor (procedural_sky.hlsli) so the graph IBL and the sky pass
// cannot drift — the sun term keys off the directional light, so forward f.lightDir (issue #4).
float3 hfSkyColor(float3 dir) { return HFSkyColor(dir, f.lightDir.xyz); }
float3 hfCookTorrance(float3 N, float3 V, float3 L, float3 radiance,
                      float3 albedo, float metallic, float roughness, float3 F0) {
    float3 H   = normalize(L + V);
    float  NoV = max(dot(N, V), 1e-4);
    float  NoL = max(dot(N, L), 0.0);
    float  NoH = max(dot(N, H), 0.0);
    float  VoH = max(dot(V, H), 0.0);
    float  alpha = roughness * roughness;
    float  D = hfDistributionGGX(NoH, alpha);
    float  G = hfGeometrySmith(NoV, NoL, roughness);
    float3 F = hfFresnelSchlick(VoH, F0);
    float3 spec = (D * G) * F / max(4.0 * NoV * NoL, 1e-4);
    float3 kd = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);
    float3 diff = kd * albedo / HF_PI;
    return (diff + spec) * radiance * NoL;
}
float3 hfSrgbToLinear(float3 c) { return pow(saturate(c), 2.2); }

// The view-direction cosine the Fresnel graph node uses (N·V on the geometric normal). The material
// graph's Fresnel node samples this; the codegen calls hfNoV(i) and the CPU interpreter's NoV
// argument is the same quantity, so the interpreter parity test pins the same math.
float hfNoV(PSInput i) {
    float3 Ng = normalize(i.wnormal);
    float3 V  = normalize(f.viewPos.xyz - i.wpos);
    return max(dot(Ng, V), 0.0);
}

// The shared shading core: given the graph-computed surface params (baseColor LINEAR, metallic,
// roughness, emissive LINEAR), run the SAME directional+point Cook-Torrance + procedural IBL + PCF
// shadow path lit_pbr.frag.hlsl uses. baseColor/emissive here are ALREADY linear (the graph operates
// in linear space; TextureSample of an sRGB base texture is decoded by the codegen).
// Slice BE: the same shading core but with an EXPLICIT world-space shading normal N (e.g. the graph's
// tangent-space NormalMap rotated into world space via the interpolated TBN — exactly as
// lit.frag.hlsl perturbs its normal). hfShadePBR() below is the unchanged geometric-normal path; this
// overload is ONLY emitted when a material graph connects PBROutput.normal, so the existing generated
// shaders (which call hfShadePBR) stay byte-identical.
float4 hfShadePBRN(PSInput i, float3 baseColor, float metallic, float roughness, float3 emissive,
                   float3 N) {
    float3 V  = normalize(f.viewPos.xyz - i.wpos);

    float3 albedo = baseColor * i.color;
    metallic  = saturate(metallic);
    roughness = clamp(roughness, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // Directional shadow (project world pos into light clip space, PCF compare).
    float shadow = 1.0;
    {
        float4 lp = mul(f.lightViewProj, float4(i.wpos, 1.0));
        float3 proj = lp.xyz / lp.w;
        float2 smUV = proj.xy * 0.5 + 0.5;
#ifdef HF_MSL_GEN
        smUV.y = 1.0 - smUV.y;
#endif
        float  curDepth = proj.z;
        if (smUV.x >= 0.0 && smUV.x <= 1.0 && smUV.y >= 0.0 && smUV.y <= 1.0 &&
            curDepth >= 0.0 && curDepth <= 1.0) {
            float bias = 0.0025;
            float s = 0.0;
            float texel = 1.0 / 2048.0;
            [unroll] for (int sx = -1; sx <= 1; ++sx)
            [unroll] for (int sy = -1; sy <= 1; ++sy) {
                float d = gShadow.Sample(gShadowSmp, smUV + float2(sx, sy) * texel).r;
                s += (curDepth - bias > d) ? 0.0 : 1.0;
            }
            shadow = s / 9.0;
        }
    }

    float3 rgb = albedo * 0.03;

    { // Directional light.
        float3 L = normalize(-f.lightDir.xyz);
        float3 radiance = f.lightColor.rgb * shadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }

    // Colored point lights.
    int n = (int)f.ptCount.x;
    for (int li = 0; li < n; ++li) {
        float3 lp = f.ptPos[li].xyz;
        float  radius = f.ptPos[li].w;
        float3 lc = f.ptColor[li].rgb;
        float  intensity = f.ptColor[li].w;
        float3 Lv = lp - i.wpos;
        float  dist = length(Lv);
        float3 Ld = Lv / max(dist, 1e-4);
        float  att = saturate(1.0 - dist / radius);
        att *= att;
        float3 radiance = lc * intensity * att;
        rgb += hfCookTorrance(N, V, Ld, radiance, albedo, metallic, roughness, F0);
    }

    // Procedural image-based lighting (environment reflection).
    {
        float3 R = reflect(-V, N);
        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);
        float3 envColor = hfSkyColor(R);
        envColor = lerp(envColor, hfSkyColor(float3(0.0, 1.0, 0.0)), roughness * 0.7);
        float3 iblSpecular = envColor * F;
        rgb += iblSpecular;
        rgb += (1.0 - metallic) * albedo * hfSkyColor(N) * 0.15;
    }

    rgb += emissive;
    return float4(rgb, 1.0);
}

float4 hfShadePBR(PSInput i, float3 baseColor, float metallic, float roughness, float3 emissive) {
    float3 Ng = normalize(i.wnormal);
    float3 N  = Ng;  // graph materials use the geometric normal (no tangent normal-map node in MVP).
    float3 V  = normalize(f.viewPos.xyz - i.wpos);

    float3 albedo = baseColor * i.color;
    metallic  = saturate(metallic);
    roughness = clamp(roughness, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // Directional shadow (project world pos into light clip space, PCF compare).
    float shadow = 1.0;
    {
        float4 lp = mul(f.lightViewProj, float4(i.wpos, 1.0));
        float3 proj = lp.xyz / lp.w;
        float2 smUV = proj.xy * 0.5 + 0.5;
#ifdef HF_MSL_GEN
        smUV.y = 1.0 - smUV.y;
#endif
        float  curDepth = proj.z;
        if (smUV.x >= 0.0 && smUV.x <= 1.0 && smUV.y >= 0.0 && smUV.y <= 1.0 &&
            curDepth >= 0.0 && curDepth <= 1.0) {
            float bias = 0.0025;
            float s = 0.0;
            float texel = 1.0 / 2048.0;
            [unroll] for (int sx = -1; sx <= 1; ++sx)
            [unroll] for (int sy = -1; sy <= 1; ++sy) {
                float d = gShadow.Sample(gShadowSmp, smUV + float2(sx, sy) * texel).r;
                s += (curDepth - bias > d) ? 0.0 : 1.0;
            }
            shadow = s / 9.0;
        }
    }

    float3 rgb = albedo * 0.03;

    { // Directional light.
        float3 L = normalize(-f.lightDir.xyz);
        float3 radiance = f.lightColor.rgb * shadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }

    // Colored point lights.
    int n = (int)f.ptCount.x;
    for (int li = 0; li < n; ++li) {
        float3 lp = f.ptPos[li].xyz;
        float  radius = f.ptPos[li].w;
        float3 lc = f.ptColor[li].rgb;
        float  intensity = f.ptColor[li].w;
        float3 Lv = lp - i.wpos;
        float  dist = length(Lv);
        float3 Ld = Lv / max(dist, 1e-4);
        float  att = saturate(1.0 - dist / radius);
        att *= att;
        float3 radiance = lc * intensity * att;
        rgb += hfCookTorrance(N, V, Ld, radiance, albedo, metallic, roughness, F0);
    }

    // Procedural image-based lighting (environment reflection).
    {
        float3 R = reflect(-V, N);
        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);
        float3 envColor = hfSkyColor(R);
        envColor = lerp(envColor, hfSkyColor(float3(0.0, 1.0, 0.0)), roughness * 0.7);
        float3 iblSpecular = envColor * F;
        rgb += iblSpecular;
        rgb += (1.0 - metallic) * albedo * hfSkyColor(N) * 0.15;
    }

    rgb += emissive;
    return float4(rgb, 1.0);
}

#endif  // HF_PBR_CORE_HLSLI
