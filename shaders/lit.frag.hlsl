// Shared procedural sky (HFSkyColor): the IBL reflection below MUST match the sky pass, so both pull
// the gradient/haze/sun from ONE source (issue #4). See shaders/procedural_sky.hlsli.
#include "procedural_sky.hlsli"
#include "frame_data.hlsli"
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Shadow map lives in the per-frame set (set 0): binding 1 = depth image, binding 2 = sampler.
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
// Tangent-space normal map (binding 2/3 of the material set). Encoded 0..1; flat regions store
// (0.5,0.5,1) which decodes to (0,0,1) = no perturbation. spirv-cross --msl-decoration-binding maps
// these SPIR-V bindings straight to Metal texture(3)/sampler(4) (see engine/rhi_metal/metal_common.h).
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    [[vk::location(4)]] nointerpolation float2 material : TEXCOORD1; // x=metallic, y=roughness
    [[vk::location(5)]] float3 wtangent : TANGENT;
};
static const float HF_PI = 3.14159265358979323846;

// GGX / Trowbridge-Reitz normal distribution. alpha = roughness^2.
float hfDistributionGGX(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float d  = (NoH * NoH) * (a2 - 1.0) + 1.0;
    return a2 / max(HF_PI * d * d, 1e-7);
}
// Smith geometry with Schlick-GGX (direct-lighting k = (r+1)^2 / 8).
float hfGeometrySchlickGGX(float NoX, float k) {
    return NoX / (NoX * (1.0 - k) + k);
}
float hfGeometrySmith(float NoV, float NoL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return hfGeometrySchlickGGX(NoV, k) * hfGeometrySchlickGGX(NoL, k);
}
// Fresnel-Schlick.
float3 hfFresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (float3(1.0, 1.0, 1.0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}
// Procedural sky color for a world-space direction. Used for image-based lighting: metals reflect
// this in the reflection direction. Thin wrapper over the SHARED HFSkyColor (procedural_sky.hlsli) so
// this IBL and the sky pass cannot drift — change the look in ONE place (issue #4). The sun term keys
// off the directional light (f.lightDir), so the wrapper forwards it.
float3 SkyColor(float3 dir) { return HFSkyColor(dir, f.lightDir.xyz); }

// Cook-Torrance contribution for a single light of given radiance.
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
    // Energy conservation: diffuse only from the non-reflected, non-metallic fraction.
    float3 kd = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);
    float3 diff = kd * albedo / HF_PI;
    return (diff + spec) * radiance * NoL;
}

float4 main(PSInput i) : SV_Target {
    // Geometric (interpolated) world normal.
    float3 Ng = normalize(i.wnormal);

    // --- Tangent-space normal mapping. Build an orthonormal TBN from the interpolated world normal
    // and tangent (Gram-Schmidt so a non-orthogonal interpolated tangent stays well-conditioned),
    // sample the tangent-space normal (0..1 -> -1..1), and rotate it into world space. A flat normal
    // map (0.5,0.5,1)->(0,0,1) leaves N == Ng, so untextured surfaces are unaffected. ---
    float3 T = normalize(i.wtangent - Ng * dot(Ng, i.wtangent));
    float3 B = cross(Ng, T);                       // handedness baked into T at authoring time
    float3x3 TBN = float3x3(T, B, Ng);             // rows = T,B,N
    float3 nTS = gNormalMap.Sample(gNormalSmp, i.uv).xyz * 2.0 - 1.0;
    // mul(nTS, TBN) = nTS.x*T + nTS.y*B + nTS.z*N (row-vector * row-matrix), the standard
    // tangent->world transform for this row-major TBN.
    float3 N = normalize(mul(nTS, TBN));
    float3 V = normalize(f.viewPos.xyz - i.wpos);
    float3 tex = gTex.Sample(gSmp, i.uv).rgb * i.color;

    // --- Per-material PBR: metallic + roughness arrive per-draw via the push constant,
    // passed through as flat (nointerpolation) interpolants. metallic selects between a
    // dielectric F0 (0.04) and a metallic F0 (= albedo); roughness drives the GGX alpha and
    // the diffuse fraction (handled inside hfCookTorrance via the (1-metallic) kd term). ---
    float3 albedo   = tex;
    float  metallic = saturate(i.material.x);
    float  roughness = clamp(i.material.y, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic); // dielectric vs. metal base reflectance

    // --- Directional shadow: project world pos into the light's clip space, compare depth. ---
    // lightViewProj uses Ortho (same Y-flip as Perspective), so smUV = proj.xy*0.5+0.5 matches
    // the shadow map's texel layout (Vulkan NDC y-down -> texture row 0 at NDC y=-1) directly.
    float shadow = 1.0;
    {
        float4 lp = mul(f.lightViewProj, float4(i.wpos, 1.0));
        float3 proj = lp.xyz / lp.w;
        float2 smUV = proj.xy * 0.5 + 0.5;
        // Shadow-map texture-origin flip (Metal only). On the Metal path lightViewProj has its NDC
        // Y-flip baked in CPU-side, so proj.y is +Y-up; the shadow map texture stores row 0 = top
        // (V down), so V must be flipped to sample the matching texel. The shadow-map RENDER and
        // this SAMPLE both derive from the same CPU-flipped lightViewProj, so they stay
        // self-consistent. Vulkan keeps the original mapping (unflipped lightViewProj, V down).
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

    // Small ambient term (unshadowed) so unlit areas aren't pure black.
    float3 rgb = albedo * 0.03;

    // Directional light: Cook-Torrance, with the directional radiance attenuated by the
    // PCF shadow factor (KEEP shadow applied to the directional light only).
    {
        float3 L = normalize(-f.lightDir.xyz);
        float3 radiance = f.lightColor.rgb * shadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }

    // Accumulate colored point lights with smooth radius-based attenuation.
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
        att *= att;                              // smooth falloff to the radius
        float3 radiance = lc * intensity * att;
        rgb += hfCookTorrance(N, V, Ld, radiance, albedo, metallic, roughness, F0);
    }

    // --- Procedural image-based lighting (environment reflection). Metals have no diffuse and no
    // direct environment to reflect, so they render dark; here they reflect the procedural sky in
    // the mirror direction R. Dielectrics pick up a subtle Fresnel-rim sky sheen. This is the
    // *environment* reflection, additive to the direct lights (standard for this lite approach). ---
    {
        float3 R = reflect(-V, N);
        // Roughness-aware Fresnel-Schlick (uses N.V, not VoH): grazing angles reflect more, and a
        // rough surface caps the rim brightness at (1-roughness).
        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);
        // Sharp mirror reflection, blurred toward the up-sky average as roughness rises.
        float3 envColor = SkyColor(R);
        envColor = lerp(envColor, SkyColor(float3(0.0, 1.0, 0.0)), roughness * 0.7);
        float3 iblSpecular = envColor * F;
        rgb += iblSpecular;
        // Subtle sky-tinted ambient diffuse for dielectrics: shadowed/unlit dielectric areas pick up
        // sky color from the surface-normal direction (scales with the non-metallic fraction).
        rgb += (1.0 - metallic) * albedo * SkyColor(N) * 0.15;
    }

    return float4(rgb, 1.0);
}
