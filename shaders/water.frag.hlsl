// Water surface fragment shader (Slice CF). Recomputes the ANALYTIC Gerstner wave normal at the
// fragment's grid (x,z) (water::Normal, mirrored below), then shades the water as a fresnel blend of:
//   * a SKY REFLECTION  — reflect the view ray about the wave normal and evaluate the procedural sky
//     (the SAME SkyColor ray->sky function lit.frag/sky.frag use), and
//   * a REFRACTED scene — sample the HDR scene-color RT (the opaque scene rendered first, bound at
//     t0/s0 like SSR) at a normal-perturbed screen UV, tinted by RefractTint over the depth of the
//     scene BELOW the water (from the G-buffer linear depth at t3/s3),
// plus a sharp SUN SPECULAR glint about the wave normal from the directional light. The fresnel weight
// (water::Fresnel) makes grazing angles mirror-like and head-on angles see into the water. Output:
// rgb = water shading, a = 1 (full coverage where the grid rasterizes; the composite blends it over
// the scene). Bindings: FrameData at set 0 b0 (sky/light/camera), the scene-color RT + G-buffer pair
// at set 1 (t0/s0 + t3/s3, bound via BindTexturePair). Existing shaders + goldens UNTOUCHED.
#include "frame_data.hlsli"
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(0, 1)]] Texture2D    gScene : register(t0);   // HDR opaque scene color (to refract)
[[vk::binding(1, 1)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 1)]] Texture2D    gGbuf  : register(t3);   // view normal.xyz + linear depth.w
[[vk::binding(4, 1)]] SamplerState gGSmp  : register(s3);

struct WaterParams {
    float4x4 model;
    float4 waveA[3];   // dir.x, dir.y, amplitude, wavelength
    float4 waveB[3];   // steepness, speed, _, _
    float4 cfg0;       // time, tanHalfFovY, aspect, waterLevel
    float4 cfg1;       // fresnelF0, absorption, numWaves, _
    float4 shallow;
    float4 deep;
    float4 texel;      // 1/w, 1/h
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer WaterPC { WaterParams wp; };
#define HF_WP wp
#else
[[vk::push_constant]] struct { WaterParams p; } pc;
#define HF_WP pc.p
#endif

struct PSInput {
    float4 clip   : SV_Position;
    [[vk::location(0)]] float3 wpos : POSITION0;
    [[vk::location(1)]] float2 grid : TEXCOORD0;
};

static const float HF_W_PI = 3.14159265358979323846;

// Procedural sky color for a world-space direction — IDENTICAL to lit.frag/sky.frag (zenith/horizon/
// ground gradient + sun glow keyed off the directional light). Reused for the water reflection.
float3 SkyColor(float3 dir) {
    float3 d = normalize(dir);
    float  h       = saturate(d.y * 0.5 + 0.5);
    float3 zenith  = float3(0.18, 0.30, 0.62);
    float3 horizon = float3(0.65, 0.72, 0.82);
    float3 sky     = lerp(horizon, zenith, pow(h, 0.8));
    float3 ground  = float3(0.12, 0.11, 0.10);
    if (d.y < 0.0) {
        float g = saturate(-d.y * 2.0);
        sky = lerp(sky, ground, g);
    }
    float3 sunDir = normalize(-f.lightDir.xyz);
    float  s = pow(max(dot(d, sunDir), 0.0), 256.0);
    sky += float3(1.0, 0.95, 0.8) * s * 2.0;
    return sky;
}

// ANALYTIC summed Gerstner normal at grid (x,z), time t — IDENTICAL math to water::Normal in water.h.
float3 WaterNormal(float x, float z, float t) {
    int n = (int)HF_WP.cfg1.z;
    float3 T = float3(1.0, 0.0, 0.0);   // dP/dx
    float3 B = float3(0.0, 0.0, 1.0);   // dP/dz
    [loop] for (int i = 0; i < n; ++i) {
        float2 dir = HF_WP.waveA[i].xy;
        float  amp = HF_WP.waveA[i].z;
        float  wl  = HF_WP.waveA[i].w;
        float  q   = HF_WP.waveB[i].x;
        float  spd = HF_WP.waveB[i].y;
        float  k   = (wl > 1e-6) ? (2.0 * HF_W_PI / wl) : 0.0;
        float  w   = spd * k;
        float  theta = k * (dir.x * x + dir.y * z) - w * t;
        float  c = cos(theta), sgn = sin(theta);
        float  qa = q * amp;
        T.x += -qa * dir.x * dir.x * k * sgn;
        T.y +=  amp * dir.x * k * c;
        T.z += -qa * dir.x * dir.y * k * sgn;
        B.x += -qa * dir.x * dir.y * k * sgn;
        B.y +=  amp * dir.y * k * c;
        B.z += -qa * dir.y * dir.y * k * sgn;
    }
    return normalize(cross(B, T));   // flat -> +Y
}

// Schlick fresnel — IDENTICAL to water::Fresnel.
float WaterFresnel(float NdotV, float f0) {
    float cc = saturate(NdotV);
    float m = 1.0 - cc;
    return f0 + (1.0 - f0) * (m * m * m * m * m);
}

// Beer-Lambert depth tint — IDENTICAL to water::RefractTint.
float3 RefractTint(float depth, float3 shallow, float3 deep, float absorption) {
    float d = max(depth, 0.0);
    float tf = 1.0 - exp(-absorption * d);
    return lerp(shallow, deep, tf);
}

#ifdef HF_MSL_GEN
static const float HF_YS = 1.0;
#else
static const float HF_YS = -1.0;
#endif

float4 main(PSInput i) : SV_Target {
    float3 N = WaterNormal(i.grid.x, i.grid.y, HF_WP.cfg0.x);   // world-space wave normal (+Y up)

    // View ray (world space) from the camera to this surface point.
    float3 V = normalize(i.wpos - f.viewPos.xyz);
    float NdotV = saturate(dot(N, -V));

    // --- Sky reflection: reflect the view ray about the wave normal, evaluate the procedural sky. ---
    float3 R = reflect(V, N);
    float3 reflColor = SkyColor(R);

    // --- Screen UV of this fragment (for the refraction sample), perturbed by the wave normal. ---
    float2 uv = i.clip.xy * HF_WP.texel.xy;             // SV_Position.xy is in pixels
    float2 perturb = N.xz * 0.04;                       // small normal-driven offset (ripple refraction)
    float2 rUV = saturate(uv + perturb);

    // Scene linear depth below the water at this screen location (G-buffer .w). 0 == background/sky.
    float sceneDepth = gGbuf.Sample(gGSmp, rUV).w;
    // Depth of the submerged scene BELOW the water surface = (scene depth) - (this fragment's depth).
    // Approximate the water fragment's own view depth by its clip-space w isn't available post-divide;
    // use the camera distance projected on the forward axis as the water depth reference.
    float waterViewDepth = dot(i.wpos - f.viewPos.xyz, f.camFwd.xyz);
    float belowDepth = max(sceneDepth - waterViewDepth, 0.0);
    if (sceneDepth <= 0.0001) belowDepth = 8.0;          // no geometry (open water) -> deep tint

    float3 sceneColor = gScene.Sample(gSmp, rUV).rgb;
    float3 tint = RefractTint(belowDepth, HF_WP.shallow.rgb, HF_WP.deep.rgb, HF_WP.cfg1.y);
    float3 refrColor = sceneColor * tint;

    // --- Fresnel blend: head-on sees into the water, grazing mirrors the sky. The fresnel weight is
    // CLAMPED below 1 so even the grazing far water keeps some of the body/refraction color (so the
    // surface reads as tinted WATER rather than a pure sky mirror), and the sky reflection is slightly
    // tinted toward the water hue so the reflected sky sits in the water rather than washing it out. ---
    float fres = min(WaterFresnel(NdotV, HF_WP.cfg1.x), 0.82);
    float3 tintedRefl = reflColor * float3(0.80, 0.88, 0.95);
    float3 col = lerp(refrColor, tintedRefl, fres);

    // --- Sun specular glint: sharp Blinn-Phong-ish highlight about the wave normal. ---
    float3 L = normalize(-f.lightDir.xyz);
    float3 H = normalize(L - V);                         // half vector (view points INTO surface)
    float  spec = pow(max(dot(N, H), 0.0), 400.0);
    col += f.lightColor.rgb * spec * 3.0;

    return float4(col, 1.0);
}
