// Substrate-lite LAYERED-MATERIAL fragment shader (Issue #11, Slice SB1: clearcoat). A copy of the
// frozen lit_pbr_ibl.frag.hlsl (declarations + main + base direct/IBL/shadow/normal-map/emissive path,
// VERBATIM) PLUS an ADDITIVE clearcoat BRDF lobe gated by the new FrameData.substrateParams scalars.
// The base PBR is byte-identical to lit_pbr_ibl.frag; the clearcoat lobe (a 2nd GGX dielectric film
// over the base, like glTF KHR_materials_clearcoat) is added after the base direct+IBL accumulation,
// before emissive. Kept SEPARATE so the golden-locked lit_pbr_ibl pipeline is undisturbed.
//
// MAKE-OR-BREAK invariant: clearcoat=f.substrateParams.x=0 makes this shader EXACTLY lit_pbr_ibl.frag
// (rgb *= 1.0; both clearcoat += terms are +0.0). The --sb1-clearcoat showcase asserts this byte-for-byte.
#include "frame_data.hlsli"
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Shadow map lives in the per-frame set (set 0): binding 1 = depth image, binding 2 = sampler.
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
// --- Full-PBR material set (set 1). Bindings match lit_pbr.frag exactly. ---
[[vk::binding(0, 1)]] Texture2D    gBase        : register(t0);  // base-color (sRGB)
[[vk::binding(1, 1)]] SamplerState gBaseSmp     : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap   : register(t3);  // tangent-space normal (linear)
[[vk::binding(4, 1)]] SamplerState gNormalSmp   : register(s3);
[[vk::binding(5, 1)]] Texture2D    gMetalRough  : register(t5);  // G=roughness, B=metallic (linear)
[[vk::binding(6, 1)]] SamplerState gMetalRoughSmp : register(s5);
[[vk::binding(7, 1)]] Texture2D    gEmissive    : register(t7);  // emissive (sRGB)
[[vk::binding(8, 1)]] SamplerState gEmissiveSmp : register(s7);
[[vk::binding(9, 1)]] Texture2D    gOcclusion   : register(t9);  // R=ambient occlusion (linear)
[[vk::binding(10, 1)]] SamplerState gOcclusionSmp : register(s9);
// HDR environment map (Slice R): dedicated set 3 (binding 11 image, 12 sampler) -> Metal
// texture(11)/sampler(12) via --msl-decoration-binding.
[[vk::binding(11, 3)]] Texture2D    gEnv    : register(t11);
[[vk::binding(12, 3)]] SamplerState gEnvSmp : register(s11);
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    [[vk::location(4)]] nointerpolation float2 material : TEXCOORD1; // x=metallicFactor, y=roughnessFactor
    [[vk::location(5)]] float3 wtangent : TANGENT;
};
static const float HF_PI = 3.14159265358979323846;

// GGX / Trowbridge-Reitz normal distribution. alpha = roughness^2.
float hfDistributionGGX(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float d  = (NoH * NoH) * (a2 - 1.0) + 1.0;
    return a2 / max(HF_PI * d * d, 1e-7);
}
float hfGeometrySchlickGGX(float NoX, float k) {
    return NoX / (NoX * (1.0 - k) + k);
}
float hfGeometrySmith(float NoV, float NoL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return hfGeometrySchlickGGX(NoV, k) * hfGeometrySchlickGGX(NoL, k);
}
float3 hfFresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (float3(1.0, 1.0, 1.0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// Equirectangular projection: world direction -> equirect UV (matches sky_hdr.frag).
float2 EquirectUV(float3 dir) {
    float3 d = normalize(dir);
    float u = atan2(d.z, d.x) / (2.0 * HF_PI) + 0.5;
    float v = acos(clamp(d.y, -1.0, 1.0)) / HF_PI;
    return float2(u, v);
}

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
    float3 kd = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);
    float3 diff = kd * albedo / HF_PI;
    return (diff + spec) * radiance * NoL;
}

// Approximate sRGB -> linear (gamma 2.2).
float3 SrgbToLinear(float3 c) { return pow(saturate(c), 2.2); }

float4 main(PSInput i) : SV_Target {
    float3 Ng = normalize(i.wnormal);

    // --- Tangent-space normal mapping (same as lit_pbr.frag). ---
    float3 T = normalize(i.wtangent - Ng * dot(Ng, i.wtangent));
    float3 B = cross(Ng, T);
    float3x3 TBN = float3x3(T, B, Ng);
    float3 nTS = gNormalMap.Sample(gNormalSmp, i.uv).xyz * 2.0 - 1.0;
    float3 N = normalize(mul(nTS, TBN));
    float3 V = normalize(f.viewPos.xyz - i.wpos);

    // --- Base color (sRGB -> linear) * vertex color. ---
    float3 baseSrgb = gBase.Sample(gBaseSmp, i.uv).rgb;
    float3 albedo = SrgbToLinear(baseSrgb) * i.color;

    // --- Metallic-roughness (glTF G=roughness, B=metallic), scaled by the per-draw factors. ---
    float3 mr = gMetalRough.Sample(gMetalRoughSmp, i.uv).rgb;
    float  metallic  = saturate(mr.b * i.material.x);
    float  roughness = clamp(mr.g * i.material.y, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // --- Ambient occlusion (R channel, linear). ---
    float ao = gOcclusion.Sample(gOcclusionSmp, i.uv).r;

    // --- Directional shadow (project world pos into light clip space, PCF compare). ---
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

    // Small ambient term (unshadowed) so unlit areas aren't pure black; modulated by AO.
    float3 rgb = albedo * 0.03 * ao;

    // Directional light (Cook-Torrance, shadowed).
    {
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

    // --- HDR ENVIRONMENT image-based lighting (Slice R). Specular reflects the prefiltered env in
    // the mirror direction at a roughness-selected mip; diffuse irradiance is a very-blurred mip of
    // the env in the surface-normal direction. Replaces the procedural SkyColor() IBL. ---
    {
        float maxLod = f.iblParams.x;   // = mipLevels - 1 (issue #33: dedicated slot, was skyParams.z)
        // Per-pipeline IBL exposure multiplier (issue #39): scales the env's specular + diffuse
        // contribution so a sample can dim an over-bright HDR probe without engine-shader edits.
        // 1.0 = stock (every existing showcase sets it, so the IBL goldens are unchanged).
        float iblIntensity = f.iblParams.y;
        float3 R = reflect(-V, N);
        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);
        // Specular: sharp mirror at roughness 0, blurrier (higher mip) as roughness rises.
        float3 prefiltered = gEnv.SampleLevel(gEnvSmp, EquirectUV(R), roughness * maxLod).rgb;
        float3 iblSpecular = prefiltered * F * ao * iblIntensity;
        rgb += iblSpecular;
        // Diffuse irradiance: a very blurred mip (maxLod - 1) of the env in the normal direction.
        float3 irradiance = gEnv.SampleLevel(gEnvSmp, EquirectUV(N), max(maxLod - 1.0, 0.0)).rgb;
        rgb += (1.0 - metallic) * albedo * irradiance * ao * iblIntensity;
    }

    // --- SUBSTRATE-LITE CLEARCOAT LOBE (Issue #11, SB1). An additive 2nd GGX specular over the base
    // (a smooth dielectric film, fixed F0=0.04, its OWN roughness, on the GEOMETRIC normal Ng — not
    // the normal-mapped N). Gated by f.substrateParams: clearcoat=0 is an EXACT identity over the
    // base lit_pbr_ibl path above (rgb *= 1.0; both += terms are +0.0). THIS IS THE MAKE-OR-BREAK. ---
    {
        float clearcoat = f.substrateParams.x;
        float ccRough   = clamp(f.substrateParams.y, 0.05, 1.0);
        float3 Lc  = normalize(-f.lightDir.xyz);
        float3 Hc  = normalize(Lc + V);
        float NoLc = max(dot(Ng, Lc), 0.0);
        float NoHc = max(dot(Ng, Hc), 0.0);
        float NoVc = max(dot(Ng, V), 1e-4);
        float VoHc = max(dot(V, Hc), 0.0);
        float ccAlpha = ccRough * ccRough;
        float Dc = hfDistributionGGX(NoHc, ccAlpha);          // alpha-convention, matches the helper
        float Gc = hfGeometrySmith(NoVc, NoLc, ccRough);
        float Fc = 0.04 + 0.96 * pow(saturate(1.0 - VoHc), 5.0);
        float ccSpec = (Dc * Gc * Fc) / max(4.0 * NoVc * NoLc, 1e-4) * NoLc;
        rgb *= (1.0 - clearcoat * Fc);                        // attenuate what's under the film
        rgb += clearcoat * ccSpec * (f.lightColor.rgb * shadow);   // direct clearcoat lobe (scene shadow)
        float3 Rc = reflect(-V, Ng);
        float3 ccEnv = gEnv.SampleLevel(gEnvSmp, EquirectUV(Rc), ccRough * f.iblParams.x).rgb;  // sharp dielectric env reflection
        rgb += clearcoat * Fc * ccEnv * f.iblParams.y;
    }

    // --- SUBSTRATE-LITE SHEEN LOBE (Issue #11, SB2). A retroreflective fabric/velvet rim
    // (glTF KHR_materials_sheen: Charlie distribution + Ashikhmin visibility), on the normal-mapped N
    // for the DIRECTIONAL light. Gated by f.substrateParams.z: sheen=0 is an EXACT identity over the
    // base+clearcoat path (rgb += 0.0). THIS IS THE MAKE-OR-BREAK alongside SB1. ---
    {
        float sheen = f.substrateParams.z;
        float3 sheenColor = float3(1.0, 1.0, 1.0);
        float3 Ls = normalize(-f.lightDir.xyz);
        float3 Hs = normalize(Ls + V);
        float NoLs = max(dot(N, Ls), 0.0);
        float NoVs = max(dot(N, V), 1e-4);
        float NoHs = max(dot(N, Hs), 0.0);
        float sr = clamp(0.3, 0.07, 1.0);                  // sheen roughness (fixed velvet)
        float invR = 1.0 / sr;
        float cos2h = NoHs * NoHs;
        float sin2h = max(1.0 - cos2h, 0.0078125);
        float Dch = (2.0 + invR) * pow(sin2h, invR * 0.5) * (1.0 / (2.0 * 3.14159265));   // D_charlie
        float Vash = 1.0 / max(4.0 * (NoLs + NoVs - NoLs * NoVs), 1e-4);                   // V_ashikhmin
        float3 sheenLobe = sheenColor * (Dch * Vash * NoLs);
        rgb += sheen * sheenLobe * (f.lightColor.rgb * shadow) * ao;
    }

    // --- Emissive (sRGB -> linear), ADDED after lighting. ---
    float3 emis = SrgbToLinear(gEmissive.Sample(gEmissiveSmp, i.uv).rgb);
    rgb += emis;

    return float4(rgb, 1.0);
}
