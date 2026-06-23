// Slice DE — Planar Reflections: the MIRROR-SURFACE fragment shader. A flat reflective surface (a
// mirror floor / still water plane) shaded EXACTLY like the standard lit pass (lit.frag's matte
// Cook-Torrance result) and then BLENDED with the planar reflection: the scene rendered a second time
// through the camera REFLECTED across the mirror plane into a 2D reflection render target, sampled here
// at this pixel's OWN screen-space position. The blend is lerp(matteColor, reflectionSample,
// reflectivity).
//
// THE reflectivity=0 NO-OP PROOF (the golden-safety identity, mirrored by render::planar's doc): with
// reflectivity == 0 the lerp returns matteColor EXACTLY — the reflection texture is NEVER read (it is
// short-circuited out) — so the planar render is byte-identical to the matte (non-reflective) lit
// render of the same scene. The showcase renders reflectivity=0 and asserts SHA-equality to the matte
// render, then renders the real reflectivity>0 version as the golden. No blend bias, no UV drift.
//
// REFLECTION SAMPLE: the reflection RT holds the scene as seen by the reflected camera, already in the
// MAIN camera's screen space (it was rendered with reflectedViewProj = mainProj * ReflectionMatrix *
// mainView at the same viewport). So a mirror pixel samples the reflection at its OWN screen UV
// (SV_Position.xy / screenSize) — the reflected geometry lands exactly where the mirror shows it. This
// is the standard planar-reflection projection (the reflection is "pinned" to the screen, like an SSR
// composite, because both renders share the screen projection).
//
// reflectivity lives in the per-frame FrameData.skyParams.x (uniform for the whole mirror surface — a
// single flat reflector), so this reuses lit.vert UNCHANGED (no new vertex output / no golden-locked
// shader touched). The matte lighting below is copied VERBATIM from lit.frag so the matte result is
// identical to the engine's standard lit floor.
//
// Bindings: set 0 = Frame UBO + shadow map (as lit.frag); set 1 = base+normal material (as lit.frag);
// set 3 binding 11/12 = the reflection RT (a 2D color target bound via BindReflectionProbe — the SAME
// dedicated environment slot the probe reflection uses; here it is a flat Texture2D, sampled by screen
// UV, NOT a cubemap). The [[vk::binding]] decorations are seam-discipline decorations, not backend
// code symbols.
#include "frame_data.hlsli"
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);
// The planar reflection render target (a standard 2D color target), bound at the dedicated environment
// slot. Sampled by this pixel's screen UV. Only read when reflectivity > 0 (the no-op proof).
[[vk::binding(11, 3)]] Texture2D    gReflRT    : register(t11);
[[vk::binding(12, 3)]] SamplerState gReflSmp   : register(s11);

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

// --- Cook-Torrance lighting (copied VERBATIM from lit.frag so the matte result is byte-identical). ---
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

// The matte lit color (identical to lit.frag::main): this is the surface's color WITHOUT the planar
// reflection. With reflectivity=0 the shader returns exactly this.
float3 MatteLit(PSInput i) {
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
    {
        float3 L = normalize(-f.lightDir.xyz);
        float3 radiance = f.lightColor.rgb * shadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }
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
    return rgb;
}

float4 main(PSInput i) : SV_Target {
    float3 matte = MatteLit(i);

    // reflectivity from the per-frame uniform (skyParams.x). The NO-OP early-out: at reflectivity==0
    // the reflection RT is NEVER sampled, so the result is matte EXACTLY (byte-identical to the matte
    // render). This branch also keeps the matte render independent of the reflection RT's contents.
    float reflectivity = f.skyParams.x;
    if (reflectivity <= 0.0) return float4(matte, 1.0);

    // Sample the planar reflection at this pixel's OWN screen UV. SV_Position.xy is in window-space
    // pixels; skyParams.zw carry 1/width, 1/height (the reflection RT shares the main viewport size), so
    // screenUV = SV_Position.xy * (1/w, 1/h) is the [0,1] UV. The reflection RT was rendered with the
    // reflected-camera view-proj (reflVP = obliqueProj * ReflectionMatrix * mainView), which shares the
    // main camera's screen projection — so the reflection is "pinned" to screen space and the reflected
    // geometry lands at the SAME screen position the mirror shows it: the mirror pixel samples it at its
    // own screen UV. skyParams.y is a V-flip toggle (1 -> flip V) for a backend whose RT texture-origin
    // convention differs (the host sets it per backend); on Vulkan the reflected-camera render already
    // matches the main pass framebuffer orientation, so no flip is needed.
    float2 screenUV = i.clip.xy * float2(f.skyParams.z, f.skyParams.w);
    if (f.skyParams.y > 0.5) screenUV.y = 1.0 - screenUV.y;
    float3 reflColor = gReflRT.Sample(gReflSmp, screenUV).rgb;

    float3 outRgb = lerp(matte, reflColor, saturate(reflectivity));
    return float4(outRgb, 1.0);
}
