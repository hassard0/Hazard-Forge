// Contact-shadowed lit fragment shader (Slice CT). A VARIANT of lit.frag: IDENTICAL lighting, except the
// DIRECT directional-light (sun) contribution is additionally attenuated by the SCREEN-SPACE CONTACT
// SHADOW factor — the fine-scale contact occlusion the cascaded shadow map (CSM) is too coarse to
// capture (the dark line where a sphere rests on the ground, the crease where a small box nearly touches
// a surface). Ambient, IBL/environment reflection, and point lights are UNAFFECTED (contact shadows only
// occlude the sun, exactly like the CSM shadow term + the CK cloud-shadow term). Used ONLY by
// --contactshadow-shot; lit.frag + its goldens stay byte-identical (new path).
//
// Unlike the CK cloud shadow (computed in-shader from the world position), the contact-shadow factor is
// SCREEN-SPACE: it needs the full-scene G-buffer depth, so it is computed in a SEPARATE fullscreen pass
// (contact_shadows.frag, running render/contact_shadows.h RayMarchShadow over the G-buffer) into a
// single-channel factor RENDER TARGET, which THIS shader samples at the pixel's screen position and
// multiplies into the sun radiance. The factor RT is bound at the SECOND material slot (t3/s3 -> Metal
// texture(3)/sampler(4)) via BindTexturePair(base, factorRT) — the SAME render-target-as-sampled-input
// path the SSAO/GTAO composite uses (BindTexturePair(scene, aoRT)), which both backends support for
// render targets. (This pass forgoes tangent-space normal mapping so the normal-map slot is free to
// carry the factor; the showcase surfaces use a flat normal anyway.) NO new RHI seam.
//
// THE maxDist=0 NO-OP PROOF (backend-portable, per the POM/GTAO/CS lesson): the showcase renders THIS
// SAME shader twice — once with the contact_shadows.frag factor computed at maxDist=0 (RayMarchShadow
// returns 1 for every pixel -> the factor texture is all-1), and once with an all-white factor texture —
// and asserts they are BYTE-IDENTICAL. With the factor == 1 everywhere the sun radiance is multiplied by
// 1 -> unchanged -> the contact-shadowed render equals the no-contact render EXACTLY (no constant bias,
// no self-occlusion acne leaking through). The real maxDist>0 factor is the golden.
//
// SEAM DISCIPLINE: above the RHI seam; the only vk/MSL mentions are the HF_MSL_GEN guards +
// [[vk::binding]] decorations (same as lit.frag/lit_cloudshadow.frag), not backend CODE symbols.
#include "frame_data.hlsli"
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
// The precomputed contact-shadow factor RENDER TARGET (R = factor in [0,1], 1 = lit) at the ENVIRONMENT
// slot (set 3, binding 11/12 -> Metal texture(11)/sampler(12)), bound via BindReflectionProbe — the SAME
// render-target-as-sampled-input path the Slice-AK probe atlas uses to feed an offscreen render target
// into a forward lit pass. (BindMaterialPBR can't carry a render target, and BindTexturePair needs a
// render-target PRIMARY; the env/probe path is the established RT-as-material-input for a forward pass.)
// This pass forgoes tangent-space normal mapping (the showcase surfaces use a flat normal anyway).
[[vk::binding(11, 3)]] Texture2D    gContact    : register(t11);
[[vk::binding(12, 3)]] SamplerState gContactSmp : register(s11);

// NOTE: this fragment reads NO push constant (so the lit pipeline keeps lit.vert's vertex-only push
// constant range — matching lit.frag, which also reads its per-draw material from the interpolant, not
// the push constant). The screen texel (1/width, 1/height) needed to map SV_Position pixel coords -> the
// factor-texture UV is carried in FrameData.skyParams.zw (the showcase sets it; skyParams.xy already
// hold tanHalfFovY + aspect for the sky/recon passes). metallic+roughness arrive via the i.material
// interpolant exactly like lit.frag.

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
    // Geometric world normal (this pass forgoes tangent-space normal mapping — the normal-map material
    // slot carries the contact-shadow factor RT instead; the showcase surfaces are flat-normalled).
    float3 N = normalize(i.wnormal);
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

    // --- CONTACT SHADOW (Slice CT): the screen-space contact-shadow factor at THIS pixel, precomputed by
    // contact_shadows.frag into the factor RT (bound at the env slot). Sample it at the screen position
    // (SV_Position pixel coords * texel -> the factor UV; SV_Position is in framebuffer pixels with the
    // same top-left origin on both backends, and the factor RT is the same screen-space framebuffer, so
    // the UVs align). Multiplies ONLY the directional (sun) radiance, ON TOP of the shadow-map term — a
    // pixel in contact shadow loses its direct sun while keeping ambient/IBL/point lights. factor == 1
    // everywhere (the maxDist=0 path) leaves the sun radiance unchanged. ---
    float2 screenUV = i.clip.xy * f.skyParams.zw;   // texel (1/w, 1/h) from FrameData
    float  contact = gContact.Sample(gContactSmp, screenUV).r;

    float3 rgb = albedo * 0.03;
    {
        float3 L = normalize(-f.lightDir.xyz);
        float3 radiance = f.lightColor.rgb * shadow * contact;
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
