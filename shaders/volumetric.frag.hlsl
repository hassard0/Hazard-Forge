// Volumetric fog / light shafts — shadow-marched in-scattering (Slice AJ). For each pixel this
// reconstructs the world-space view ray from the camera basis in the per-frame UBO (no matrix
// inverse, identical to sky.frag), clamps the march end to the scene's linear depth (from the SSAO
// g-buffer .w — so fog stops at solid surfaces), and RAY-MARCHES N steps along the ray. At each step
// it samples the directional shadow map (set 0 t1/s1, the SAME one the lit pass uses) to decide if
// the air at that point is LIT; lit steps add in-scattering = lightColor * density * HG(cosTheta,g)
// (Henyey-Greenstein, forward-scatter g) attenuated by Beer-Lambert transmittance exp(-extinction*t).
// A baked 4x4 Bayer dither offsets the march START to break banding (deterministic, no RNG). Output:
// rgb = accumulated in-scattered radiance (additive over the scene in the composite). Existing
// lit/ssao/bloom/ssr shaders + their goldens are UNTOUCHED.
//
// FrameData layout is byte-identical to lit.frag (set 0 b0). The shadow map is bound into set 0 t1/s1
// by SetShadowMap; the g-buffer is bound into set 1 t0/s0 by BindTexture.
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams; // skyParams.x=tanHalfFov, .y=aspect
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gGbuf : register(t0);   // view normal.xyz + linear depth.w
[[vk::binding(1, 1)]] SamplerState gGSmp : register(s0);

struct VolParams {
    float2 texel;      // 1/size (unused in-shader; kept for layout parity / future half-res)
    float  density;    // scattering density per unit length
    float  g;          // Henyey-Greenstein asymmetry (forward-scatter)
    float  extinction; // Beer-Lambert extinction coefficient
    float  marchDist;  // total march length when the ray hits no geometry (open air)
    float  steps;      // march step count
    float  pad;
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer VolPC { VolParams vp; };
#define HF_VP vp
#else
[[vk::push_constant]] struct { VolParams p; } pc;
#define HF_VP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

static const float HF_PI = 3.14159265358979323846;

// Henyey-Greenstein phase function — matches engine/render/volumetric.h (unit-tested). cosTheta is
// between the photon travel direction and the view ray; g>0 forward-scatters so looking toward the
// light glows. g=0 -> isotropic 1/(4*pi).
float HG(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    denom = max(denom, 1e-6);
    return (1.0 - g2) / (4.0 * HF_PI * denom * sqrt(denom));
}

// Baked 4x4 ordered dither (Bayer/16) in [0,1): jitters the march START by a sub-step fraction so the
// fixed step grid does not band. Deterministic per pixel -> byte-stable golden.
static const float kDither[16] = {
    0.0 / 16.0,  8.0 / 16.0,  2.0 / 16.0, 10.0 / 16.0,
   12.0 / 16.0,  4.0 / 16.0, 14.0 / 16.0,  6.0 / 16.0,
    3.0 / 16.0, 11.0 / 16.0,  1.0 / 16.0,  9.0 / 16.0,
   15.0 / 16.0,  7.0 / 16.0, 13.0 / 16.0,  5.0 / 16.0,
};

// 1.0 = the air at this world point is LIT (reached by the directional light), 0.0 = in shadow.
// Same projection + Metal V-flip convention as lit.frag's shadow sample, but a single tap (no PCF —
// the dithered march already smooths the volume).
float ShadowLit(float3 wpos) {
    float4 lp = mul(f.lightViewProj, float4(wpos, 1.0));
    float3 proj = lp.xyz / lp.w;
    float2 smUV = proj.xy * 0.5 + 0.5;
#ifdef HF_MSL_GEN
    smUV.y = 1.0 - smUV.y;
#endif
    float curDepth = proj.z;
    // Outside the shadow frustum -> treat as lit (open sky lights the air).
    if (smUV.x < 0.0 || smUV.x > 1.0 || smUV.y < 0.0 || smUV.y > 1.0 ||
        curDepth < 0.0 || curDepth > 1.0)
        return 1.0;
    float bias = 0.0025;
    float d = gShadow.Sample(gShadowSmp, smUV).r;
    return (curDepth - bias > d) ? 0.0 : 1.0;
}

float4 main(PSInput i) : SV_Target {
    // World-space view ray (un-normalised; unit camFwd projection) — identical to sky.frag. A point at
    // view-linear-depth t along the ray is viewPos + rayU * t.
    float2 ndc = i.uv * 2.0 - 1.0;
    float3 rayU = f.camFwd.xyz
                + f.camRight.xyz * ndc.x * f.skyParams.x * f.skyParams.y
                + f.camUp.xyz    * (-ndc.y) * f.skyParams.x;
    float3 rayDir = normalize(rayU);

    // Scene-depth clamp: the g-buffer .w is the view-space LINEAR depth of the nearest surface. If a
    // surface was hit, the march ends there (fog stops at solids); otherwise (background/sky, w<=0)
    // the ray fogs the open air out to marchDist.
    float dz = gGbuf.Sample(gGSmp, i.uv).w;
    float tEnd = (dz > 0.0001) ? min(dz, HF_VP.marchDist) : HF_VP.marchDist;

    int   steps   = (int)HF_VP.steps;
    float stepLen = tEnd / (float)steps;

    // photonDir = direction the light TRAVELS (f.lightDir). cosTheta with the view ray peaks the HG
    // term when the camera looks toward the light -> bright shafts streaming toward the viewer.
    float3 photonDir = normalize(f.lightDir.xyz);
    float  cosTheta  = dot(rayDir, photonDir);
    float  phase     = HG(cosTheta, HF_VP.g);

    // Dither the march start by a sub-step fraction.
    int2  px = int2(i.uv / HF_VP.texel);
    float jitter = kDither[(px.x & 3) + ((px.y & 3) << 2)];

    float3 scatter = float3(0.0, 0.0, 0.0);
    [loop] for (int s = 0; s < steps; ++s) {
        float t = (float(s) + jitter) * stepLen;
        if (t > tEnd) break;
        float3 P = f.viewPos.xyz + rayU * t;
        float lit = ShadowLit(P);
        // Beer-Lambert transmittance from the camera to this sample.
        float T = exp(-HF_VP.extinction * t);
        // In-scatter contributed by this lit step.
        scatter += f.lightColor.rgb * (lit * HF_VP.density * phase * stepLen * T);
    }

    return float4(scatter, 1.0);
}
