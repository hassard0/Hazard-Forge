// Cloud-shadowed lit fragment shader (Slice CK). A VARIANT of lit.frag: IDENTICAL lighting, except the
// DIRECT directional-light (sun) contribution is additionally attenuated by the CLOUD SHADOW — the
// transmittance of the sunlight through the volumetric cloud slab (CH) on its way to this surface
// point. Ambient, IBL/environment reflection, and point lights are UNAFFECTED (clouds only block the
// sun). Used ONLY by --cloud-shadows-shot; lit.frag + its goldens are byte-identical (new path).
//
// The cloud-shadow factor reuses the CH cloud math (engine/render/clouds.h), mirrored VERBATIM below
// (must stay bit-identical to clouds.h + clouds.frag) so the CPU unit test, the cloudscape pass, and
// this surface-shadow term all sample the SAME deterministic field. CloudShadow marches from the
// surface world position TOWARD the sun (-lightDir) through the slab, accumulates optical depth from
// Density, and returns Beer(opticalDepth) in [0,1] (1 = full sun, 0 = fully shadowed). The slab
// altitudes / fixed time / coverage / march steps arrive in a push constant (set by the showcase to the
// SAME clouds:: constants as CH). DETERMINISTIC: fixed time, fixed steps, integer-lattice hash noise.
#include "frame_data.hlsli"
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);

// Push constant (shared vertex+fragment range; lit.vert reads only model+material from the head).
// cloudParams.x = slabBottom, .y = slabTop, .z = time, .w = shadow march steps.
// HF_MSL_GEN: the engine binds the push-constant bytes to the FRAGMENT push-const slot (Metal
// buffer(1) = kFbPushConst) when fragmentPushConstants is set, so for the MSL path this fragment
// cbuffer is declared at [[vk::binding(1,0)]] (NOT (2,0), which is the vertex push-const slot) — the
// SAME slot the bloom/clouds fragment push constants use. The full {model,material,cloudParams} struct
// is declared so cloudParams sits at the correct byte offset (80) within the shared 96-byte range; the
// fragment reads only cloudParams. The Vulkan/DXC path keeps the real [[vk::push_constant]].
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer FragPushC { float4x4 cs_model; float4 cs_material; float4 cloudParams; };
#define HF_CLOUDPARAMS cloudParams
#else
[[vk::push_constant]] struct { float4x4 model; float4 material; float4 cloudParams; } pc;
#define HF_CLOUDPARAMS pc.cloudParams
#endif

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

// --- clouds.h math, mirrored VERBATIM (must stay bit-identical to engine/render/clouds.h). ----------
static const float kHashA = 12.9898;
static const float kHashB = 78.233;
static const float kHashC = 37.719;
static const float kHashM = 43758.5453;
static const float kNoiseScale = 0.06;
static const float kWindX = 0.8;
static const float kWindZ = 0.25;
static const float kFbmLacunarity = 2.0;
static const float kFbmGain = 0.5;
static const float kCoverage = 0.42;   // clouds::kCoverage (the CH default; baked, matches the showcase)

float Hash3(float ix, float iy, float iz) {
    float n = ix * kHashA + iy * kHashB + iz * kHashC;
    return frac(sin(n) * kHashM);
}
float Fade(float t) { return t * t * (3.0 - 2.0 * t); }

float Noise3(float3 p) {
    float fx = floor(p.x), fy = floor(p.y), fz = floor(p.z);
    float tx = Fade(p.x - fx), ty = Fade(p.y - fy), tz = Fade(p.z - fz);
    float c000 = Hash3(fx,       fy,       fz);
    float c100 = Hash3(fx + 1.0, fy,       fz);
    float c010 = Hash3(fx,       fy + 1.0, fz);
    float c110 = Hash3(fx + 1.0, fy + 1.0, fz);
    float c001 = Hash3(fx,       fy,       fz + 1.0);
    float c101 = Hash3(fx + 1.0, fy,       fz + 1.0);
    float c011 = Hash3(fx,       fy + 1.0, fz + 1.0);
    float c111 = Hash3(fx + 1.0, fy + 1.0, fz + 1.0);
    float x00 = c000 + (c100 - c000) * tx;
    float x10 = c010 + (c110 - c010) * tx;
    float x01 = c001 + (c101 - c001) * tx;
    float x11 = c011 + (c111 - c011) * tx;
    float y0 = x00 + (x10 - x00) * ty;
    float y1 = x01 + (x11 - x01) * ty;
    return y0 + (y1 - y0) * tz;
}

float Fbm(float3 p, int octaves) {
    float sum = 0.0;
    float amp = kFbmGain;
    float freq = 1.0;
    [loop] for (int o = 0; o < octaves; ++o) {
        sum += amp * Noise3(p * freq);
        freq *= kFbmLacunarity;
        amp  *= kFbmGain;
    }
    return sum;
}

float HeightGradient(float y, float slabBottom, float slabTop) {
    if (y <= slabBottom || y >= slabTop) return 0.0;
    float h = (y - slabBottom) / (slabTop - slabBottom);
    return 4.0 * h * (1.0 - h);
}

float Density(float3 worldPos, float t, float slabBottom, float slabTop, float coverage) {
    if (worldPos.y <= slabBottom || worldPos.y >= slabTop) return 0.0;
    float3 sp = float3((worldPos.x + t * kWindX) * kNoiseScale,
                       worldPos.y * kNoiseScale,
                       (worldPos.z + t * kWindZ) * kNoiseScale);
    float fbm = Fbm(sp, 5);
    float carved = fbm - coverage;
    if (carved <= 0.0) return 0.0;
    return carved * HeightGradient(worldPos.y, slabBottom, slabTop);
}

float Beer(float opticalDepth) { return exp(-opticalDepth); }

// CloudShadow (clouds::CloudShadow): the sun's transmittance to `worldPos` through the cloud slab.
// `sunDir` is the directional-light travel direction; -sunDir points toward the sun. March toward the
// sun, clipped to the slab, accumulate optical depth, return Beer. 1 = full sun, 0 = fully shadowed.
float CloudShadow(float3 worldPos, float3 sunDir, float t, float slabBottom, float slabTop, int steps) {
    float3 toSun = normalize(-sunDir);
    if (abs(toSun.y) <= 1e-4) return 1.0;
    float t0 = (slabBottom - worldPos.y) / toSun.y;
    float t1 = (slabTop    - worldPos.y) / toSun.y;
    float tEnter = min(t0, t1);
    float tExit  = max(t0, t1);
    tEnter = max(tEnter, 0.0);
    if (tExit <= tEnter) return 1.0;
    int   n = steps > 0 ? steps : 1;
    float stepLen = (tExit - tEnter) / (float)n;
    float opticalDepth = 0.0;
    [loop] for (int s = 0; s < n; ++s) {
        float ts = tEnter + ((float)s + 0.5) * stepLen;
        float3 p = worldPos + toSun * ts;
        opticalDepth += Density(p, t, slabBottom, slabTop, kCoverage) * stepLen;
    }
    return Beer(opticalDepth);
}
// --- end mirrored math ------------------------------------------------------------------------------

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

    // --- Directional shadow map (IDENTICAL to lit.frag). ---
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

    // --- CLOUD SHADOW (Slice CK): the sun's transmittance through the cloud slab to THIS surface
    // point. Multiplies ONLY the directional (sun) radiance, ON TOP of the shadow-map term — so a
    // point in cloud shadow loses its direct sun while still keeping ambient/IBL/point lights. ---
    float cloudShadow = CloudShadow(i.wpos, f.lightDir.xyz, HF_CLOUDPARAMS.z,
                                    HF_CLOUDPARAMS.x, HF_CLOUDPARAMS.y, (int)HF_CLOUDPARAMS.w);

    float3 rgb = albedo * 0.03;
    {
        float3 L = normalize(-f.lightDir.xyz);
        float3 radiance = f.lightColor.rgb * shadow * cloudShadow;
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

    return float4(rgb, 1.0);
}
