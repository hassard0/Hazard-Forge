// Slice HRR1 — KAJIYA-KAY HAIR FRAGMENT: the anisotropic strand-shading clone of lit.frag for the
// HRR1 camera-facing hair ribbons (engine/render/hair_render.h). Pairs with the EXISTING lit.vert:
// the SAME PSInput contract — the ribbon builder puts the strand TANGENT in the vertex TANGENT slot
// (location 5, forwarded world-space by lit.vert) and the camera-facing ribbon normal in the NORMAL
// slot, so NO vertex-shader change is needed. FrameData / shadow-map / material bindings are
// IDENTICAL to lit.frag (same sets/bindings -> the same pipeline-layout + BindMaterial contract, and
// the same HLSL->SPIR-V->MSL toolchain constructs: no int64, standard ops only).
//
// THE KAJIYA-KAY TERMS (Kajiya & Kay 1989, the standard hair BSDF approximation):
//   diffuse  = albedo * sin(T, L)            = albedo * sqrt(1 - dot(T,L)^2)
//   specular = ks * sin(T, H)^shininess      = ks * sqrt(1 - dot(T,H)^2)^shininess, H = normalize(L+V)
// (light scattered by a thin cylinder depends on the angle to the strand TANGENT, not a surface
// normal — the anisotropic highlight runs ACROSS the strands at constant tangent-angle). Per-draw
// knobs ride the lit.vert material push constant: material.x = ks (specular strength), material.y *
// 128 = the shininess exponent. A root->tip albedo ramp rides uv.y (v = the along-strand parameter
// from the ribbon builder): roots darker, tips lighter — the classic groom depth cue.
//
// KEPT FROM lit.frag VERBATIM: the FrameData contract (sun dir/color, viewPos), the PCF directional
// shadow (incl. the HF_MSL_GEN shadow-V flip), the procedural-sky ambient tint, the vertex-color *
// texture albedo path. DROPPED (documented): the Cook-Torrance/IBL surface model (replaced by
// Kajiya-Kay — ribbons are not micro-facet surfaces), the point-light loop (the groom showcase has
// none; a Kajiya point loop is a future refinement), the tangent-space normal map (the tangent slot
// now carries the STRAND tangent, so normal-mapping does not apply; the bindings stay declared for
// the shared BindMaterial contract).
// HONEST CAVEATS: no strand<->strand self-shadowing (deep-opacity maps are future); camera-facing
// ribbons shimmer at grazing tangents (a geometry-side artifact, documented in hair_render.h).
#include "procedural_sky.hlsli"
#include "frame_data.hlsli"
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Shadow map lives in the per-frame set (set 0): binding 1 = depth image, binding 2 = sampler.
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
// Declared for the shared BindMaterial contract (see the banner); not sampled by Kajiya-Kay.
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    [[vk::location(4)]] nointerpolation float2 material : TEXCOORD1; // x=ks, y=shininess/128
    [[vk::location(5)]] float3 wtangent : TANGENT;   // the STRAND tangent (hair_render.h)
};

// Procedural sky color for a world-space direction (the shared HFSkyColor, keyed off f.lightDir —
// the same one-source discipline as lit.frag's SkyColor wrapper).
float3 SkyColor(float3 dir) { return HFSkyColor(dir, f.lightDir.xyz); }

float4 main(PSInput i) : SV_Target {
    // The strand frame: T = the strand tangent (Kajiya-Kay's axis), N = the camera-facing ribbon
    // normal (ambient/sky tint only — Kajiya-Kay does not use a surface normal for the light terms).
    float3 T = normalize(i.wtangent);
    float3 N = normalize(i.wnormal);
    float3 V = normalize(f.viewPos.xyz - i.wpos);

    // Albedo: texture * vertex color, with the root->tip ramp on uv.y (v = the along-strand
    // parameter: 0 at the root, 1 at the tip — roots darker, tips lighter).
    float3 tex = gTex.Sample(gSmp, i.uv).rgb * i.color;
    float  ramp = lerp(0.55, 1.1, saturate(i.uv.y));
    float3 albedo = tex * ramp;

    // --- Directional shadow: IDENTICAL to lit.frag (PCF 3x3 + the HF_MSL_GEN V flip). ---
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

    // --- The Kajiya-Kay terms (see the banner). ---
    float3 L = normalize(-f.lightDir.xyz);
    float3 H = normalize(L + V);
    float  TdotL = dot(T, L);
    float  TdotH = dot(T, H);
    float  sinTL = sqrt(saturate(1.0 - TdotL * TdotL));   // the Kajiya-Kay diffuse term
    float  sinTH = sqrt(saturate(1.0 - TdotH * TdotH));   // the Kajiya-Kay specular base
    float  ks        = saturate(i.material.x);
    float  shininess = max(1.0, i.material.y * 128.0);

    // Small ambient floor + sky tint through the camera-facing normal (the house lit look).
    float3 rgb = albedo * 0.04;
    rgb += albedo * SkyColor(N) * 0.15;

    // Directional light: Kajiya-Kay diffuse + specular, shadow on the directional only (lit.frag's
    // convention). The specular is light-colored (hair's primary highlight is unpigmented).
    {
        float3 radiance = f.lightColor.rgb * shadow;
        rgb += albedo * sinTL * radiance;
        rgb += ks * pow(sinTH, shininess) * radiance;
    }

    return float4(rgb, 1.0);
}
