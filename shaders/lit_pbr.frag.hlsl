// Full glTF metallic-roughness PBR fragment shader. A superset of lit.frag.hlsl: same FrameData /
// shadow / point-light / procedural-IBL path, plus the full glTF material texture set —
// metallic-roughness (G=roughness, B=metallic), emissive, and ambient occlusion — on top of the
// base-color + tangent-space normal map the lit pass already had. Kept as a SEPARATE shader so the
// golden-locked lit pipeline is undisturbed. The vertex shader is the shared lit.vert.hlsl.
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Shadow map lives in the per-frame set (set 0): binding 1 = depth image, binding 2 = sampler.
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
// --- Full-PBR material set (set 1). Binding numbers match the wider material set layout + are chosen
// so spirv-cross --msl-decoration-binding lands them on the engine's flat fragment indices. ---
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
// Procedural sky color (matches sky.frag.hlsl + lit.frag.hlsl) for image-based lighting.
float3 SkyColor(float3 dir) {
    float3 d = normalize(dir);
    float  h       = saturate(d.y * 0.5 + 0.5);
    float3 zenith  = float3(0.18, 0.30, 0.62);
    float3 horizon = float3(0.65, 0.72, 0.82);
    float3 sky     = lerp(horizon, zenith, pow(h, 0.8));
    float3 ground = float3(0.12, 0.11, 0.10);
    if (d.y < 0.0) {
        float g = saturate(-d.y * 2.0);
        sky = lerp(sky, ground, g);
    }
    float3 sunDir = normalize(-f.lightDir.xyz);
    float  s = pow(max(dot(d, sunDir), 0.0), 256.0);
    sky += float3(1.0, 0.95, 0.8) * s * 2.0;
    return sky;
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

// Approximate sRGB -> linear (gamma 2.2). The base-color / emissive glTF textures are authored in
// sRGB but uploaded as plain UNORM, so decode here before lighting (which is linear).
float3 SrgbToLinear(float3 c) { return pow(saturate(c), 2.2); }

float4 main(PSInput i) : SV_Target {
    // Geometric (interpolated) world normal.
    float3 Ng = normalize(i.wnormal);

    // --- Tangent-space normal mapping (same approach as lit.frag). ---
    float3 T = normalize(i.wtangent - Ng * dot(Ng, i.wtangent));
    float3 B = cross(Ng, T);
    float3x3 TBN = float3x3(T, B, Ng);
    float3 nTS = gNormalMap.Sample(gNormalSmp, i.uv).xyz * 2.0 - 1.0;
    float3 N = normalize(mul(nTS, TBN));
    float3 V = normalize(f.viewPos.xyz - i.wpos);

    // --- Base color (sRGB -> linear) * vertex color. ---
    float3 baseSrgb = gBase.Sample(gBaseSmp, i.uv).rgb;
    float3 albedo = SrgbToLinear(baseSrgb) * i.color;

    // --- Metallic-roughness: glTF packing G=roughness, B=metallic (linear), scaled by the factors
    // (passed per-draw in i.material: x=metallicFactor, y=roughnessFactor). ---
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

    // --- Procedural image-based lighting (environment reflection), modulated by AO. ---
    {
        float3 R = reflect(-V, N);
        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);
        float3 envColor = SkyColor(R);
        envColor = lerp(envColor, SkyColor(float3(0.0, 1.0, 0.0)), roughness * 0.7);
        float3 iblSpecular = envColor * F * ao;
        rgb += iblSpecular;
        rgb += (1.0 - metallic) * albedo * SkyColor(N) * 0.15 * ao;
    }

    // --- Emissive (sRGB -> linear) * emissiveFactor, ADDED after lighting. The emissiveFactor is
    // folded into the texture at authoring for DamagedHelmet (factor [1,1,1]); we still decode the
    // texture from sRGB. (A separate emissiveFactor uniform could scale this if needed.) ---
    float3 emis = SrgbToLinear(gEmissive.Sample(gEmissiveSmp, i.uv).rgb);
    rgb += emis;

    return float4(rgb, 1.0);
}
