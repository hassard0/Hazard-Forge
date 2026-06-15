// Spot-light lit fragment shader (Slice AE). Same PBR/IBL/ambient machinery as lit.frag, but the
// shadowing light is a SPOT: a point light with a cone cutoff + a single PERSPECTIVE shadow map.
//
// The directional + IBL + ambient terms remain (it's a lit scene) but the showcase passes a DIM
// directional color so the spot reads as the dominant light and its shadow is unambiguous.
//
// Spot FrameData layout (own struct; fits kFrameUboSize=1024). The --spot-shot showcase fills the
// per-frame UBO with THIS layout; other showcases keep their own. See docs/.../spot-shadows-design.md.
struct FrameData {
    float4x4 viewProj;     //   0
    float4   lightDir;     //  64  directional (dim in the showcase)
    float4   lightColor;   //  80
    float4   viewPos;      //  96
    float4x4 spotViewProj; // 112  perspective light matrix -> ends 176
    float4   spotPos;      // 176  xyz position
    float4   spotDir;      // 192  xyz cone axis (unit)
    float4   spotColor;    // 208  rgb color
    float4   spotParams;   // 224  x=cosInner, y=cosOuter, z=range, w=intensity
    float4   camFwd;       // 240
    float4   camRight;     // 256
    float4   camUp;        // 272
    float4   skyParams;    // 288  x=tan(0.5*fovY), y=aspect
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Spot shadow map lives in the per-frame set (set 0): binding 1 = depth image, binding 2 = sampler.
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
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

float4 main(PSInput i) : SV_Target {
    float3 Ng = normalize(i.wnormal);
    float3 T = normalize(i.wtangent - Ng * dot(Ng, i.wtangent));
    float3 B = cross(Ng, T);
    float3x3 TBN = float3x3(T, B, Ng);
    float3 nTS = gNormalMap.Sample(gNormalSmp, i.uv).xyz * 2.0 - 1.0;
    float3 N = normalize(mul(nTS, TBN));
    float3 V = normalize(f.viewPos.xyz - i.wpos);
    float3 tex = gTex.Sample(gSmp, i.uv).rgb * i.color;

    float3 albedo   = tex;
    float  metallic = saturate(i.material.x);
    float  roughness = clamp(i.material.y, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    float3 rgb = albedo * 0.03;  // small ambient

    // --- Dim directional light (no shadow): keeps unlit areas from going pure black. ---
    {
        float3 L = normalize(-f.lightDir.xyz);
        rgb += hfCookTorrance(N, V, L, f.lightColor.rgb, albedo, metallic, roughness, F0);
    }

    // --- SPOT LIGHT: cone-cut + distance falloff + a single PERSPECTIVE shadow map. ---
    {
        float3 Lv   = f.spotPos.xyz - i.wpos;        // light -> fragment (negated)
        float  dist = length(Lv);
        float3 L    = Lv / max(dist, 1e-4);          // unit dir fragment -> light
        float3 spotDir = normalize(f.spotDir.xyz);   // cone axis (light -> scene)

        // Cone: soft edge from inner to outer half-angle. dot(spotDir, -L) = cos(angle off axis).
        float cosInner = f.spotParams.x, cosOuter = f.spotParams.y;
        float range    = f.spotParams.z, intensity = f.spotParams.w;
        float coneCos  = dot(spotDir, -L);
        float cone     = smoothstep(cosOuter, cosInner, coneCos);

        // Distance falloff over the range (smooth, like the point lights).
        float att = saturate(1.0 - dist / range);
        att *= att;

        // Perspective spot shadow: project wpos by spotViewProj, compare depth with 3x3 PCF.
        // spotViewProj has the Vulkan Y-flip baked in (Metal flips it CPU-side); on Metal the
        // sample V is flipped to match the texture-origin, exactly like the directional shadow.
        float shadow = 1.0;
        {
            float4 lp = mul(f.spotViewProj, float4(i.wpos, 1.0));
            if (lp.w > 0.0) {
                float3 proj = lp.xyz / lp.w;
                float2 smUV = proj.xy * 0.5 + 0.5;
#ifdef HF_MSL_GEN
                smUV.y = 1.0 - smUV.y;
#endif
                float curDepth = proj.z;
                if (smUV.x >= 0.0 && smUV.x <= 1.0 && smUV.y >= 0.0 && smUV.y <= 1.0 &&
                    curDepth >= 0.0 && curDepth <= 1.0) {
                    float bias = 0.0015;
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
        }

        float3 radiance = f.spotColor.rgb * intensity * att * cone * shadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }

    // --- Procedural image-based lighting (environment reflection), same as lit.frag. ---
    {
        float3 R = reflect(-V, N);
        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);
        float3 envColor = SkyColor(R);
        envColor = lerp(envColor, SkyColor(float3(0.0, 1.0, 0.0)), roughness * 0.7);
        float3 iblSpecular = envColor * F;
        rgb += iblSpecular;
        rgb += (1.0 - metallic) * albedo * SkyColor(N) * 0.15;
    }

    return float4(rgb, 1.0);
}
