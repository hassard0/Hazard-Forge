// Volumetric clouds — fullscreen raymarched cumulus LAYER (Slice CH). For each pixel this
// reconstructs the world-space view ray from the camera basis in the per-frame UBO (no matrix inverse,
// identical to sky.frag / volumetric.frag), intersects the ray with the cloud SLAB (two altitude
// planes slabBottom..slabTop), and RAY-MARCHES N fixed steps between the entry/exit. At each step it
// samples clouds::Density (the SAME math as engine/render/clouds.h, mirrored verbatim below), then
// does a SHORT secondary march toward the sun to estimate the light's optical depth -> Beer-Lambert
// transmittance of the sunlight to that sample. The in-scattered sunlight (lightColor * sunTransmit *
// HenyeyGreenstein(cosTheta,g)) is accumulated front-to-back, attenuated by the view transmittance
// (Beer of the accumulated view optical depth). Output: rgb = cloud radiance, a = coverage (1 -
// final view transmittance). The composite blends this over the procedural sky where there is no scene
// geometry (g-buffer depth .w <= 0 = background); the lit scene in front is left untouched (clouds are
// a distant background layer). DETERMINISTIC: fixed time, fixed steps, integer-lattice hash noise (no
// RNG) -> two runs byte-identical. EXISTING sky/scene/volumetric shaders + goldens are UNTOUCHED.
//
// FrameData layout is byte-identical to sky.frag/volumetric.frag (set 0 b0). The g-buffer (view
// normal.xyz + linear depth.w) is bound into set 1 t0/s0 by BindTexture (same as the volumetric pass),
// used only to mask the clouds to the sky background. CloudParams are a fragment push constant.
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams; // skyParams.x=tanHalfFov, .y=aspect
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(0, 1)]] Texture2D    gGbuf : register(t0);   // view normal.xyz + linear depth.w
[[vk::binding(1, 1)]] SamplerState gGSmp : register(s0);

struct CloudParams {
    float slabBottom; float slabTop; float time;      float coverage;
    float steps;      float lightSteps; float g;       float densityMul;
    float3 sunColor;  float ambient;
    float3 skyTop;    float pad0;
    float3 skyBottom; float pad1;
    float2 texel;     float exposure;  float dbg;      // dbg<0 => show raw cloud rgb (composite path)
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer CloudPC { CloudParams cp; };
#define HF_CP cp
#else
[[vk::push_constant]] struct { CloudParams p; } pc;
#define HF_CP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

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

// Density with explicit coverage (matches clouds::DensityCoverage). 5 octaves (kFbmOctaves).
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

float HenyeyGreenstein(float cosAngle, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosAngle;
    denom = max(denom, 1e-6);
    return (1.0 - g2) / (4.0 * HF_PI * denom * sqrt(denom));
}
// --- end mirrored math ------------------------------------------------------------------------------

float4 main(PSInput i) : SV_Target {
    // World-space view ray (un-normalised camFwd projection) — identical to sky.frag.
    float2 ndc = i.uv * 2.0 - 1.0;
    float3 rayU = f.camFwd.xyz
                + f.camRight.xyz * ndc.x * f.skyParams.x * f.skyParams.y
                + f.camUp.xyz    * (-ndc.y) * f.skyParams.x;
    float3 rayDir = normalize(rayU);
    float3 ro = f.viewPos.xyz;

    float bottom = HF_CP.slabBottom;
    float top    = HF_CP.slabTop;

    // Intersect the ray with the two horizontal slab planes (y = bottom, y = top). Where the ray is
    // nearly horizontal (|rayDir.y| ~ 0) the slab is effectively not entered.
    float tEnter = 0.0, tExit = 0.0;
    bool hitSlab = false;
    if (abs(rayDir.y) > 1e-4) {
        float t0 = (bottom - ro.y) / rayDir.y;
        float t1 = (top    - ro.y) / rayDir.y;
        tEnter = min(t0, t1);
        tExit  = max(t0, t1);
        tEnter = max(tEnter, 0.0);   // never march behind the camera
        hitSlab = (tExit > tEnter);
    }

    // Mask to the SKY background: where the g-buffer recorded a surface (.w > 0) the lit scene is in
    // front, so emit NO cloud there (clouds are a distant background layer behind the scene).
    float sceneDepth = gGbuf.Sample(gGSmp, i.uv).w;
    if (sceneDepth > 0.0001) hitSlab = false;

    if (!hitSlab) return float4(0.0, 0.0, 0.0, 0.0);

    int   steps      = (int)HF_CP.steps;
    int   lightSteps = (int)HF_CP.lightSteps;
    float marchLen   = tExit - tEnter;
    float stepLen    = marchLen / (float)steps;

    // Sun direction (toward the sun = opposite the light travel direction), and the phase toward it.
    float3 sunDir   = normalize(-f.lightDir.xyz);
    float  cosTheta = dot(rayDir, sunDir);
    float  phase    = HenyeyGreenstein(cosTheta, HF_CP.g);

    // Light-march step length: span the slab thickness in lightSteps so the self-shadow reads.
    float slabThick = top - bottom;
    float lightStepLen = slabThick / (float)max(lightSteps, 1);

    float3 scatter = float3(0.0, 0.0, 0.0);
    float  transmittance = 1.0;   // view transmittance from the camera to the current sample

    [loop] for (int s = 0; s < steps; ++s) {
        float t = tEnter + (float(s) + 0.5) * stepLen;
        float3 P = ro + rayDir * t;
        float d = Density(P, HF_CP.time, bottom, top, HF_CP.coverage) * HF_CP.densityMul;
        if (d > 0.0) {
            // Short secondary march toward the sun -> optical depth of the sunlight to this sample.
            float lightOD = 0.0;
            [loop] for (int ls = 0; ls < lightSteps; ++ls) {
                float3 LP = P + sunDir * ((float(ls) + 0.5) * lightStepLen);
                lightOD += Density(LP, HF_CP.time, bottom, top, HF_CP.coverage)
                         * HF_CP.densityMul * lightStepLen;
            }
            float sunTransmit = Beer(lightOD);
            // In-scattered sunlight + a little ambient sky fill so shadowed cloud cores aren't black.
            float3 inScatter = HF_CP.sunColor * (sunTransmit * phase) + HF_CP.ambient.xxx;
            // Energy this step extincts from the view ray (Beer over the step's optical depth).
            float stepOD = d * stepLen;
            float stepT  = Beer(stepOD);
            // Accumulate front-to-back: this slab's scattered light, attenuated by the transmittance
            // already accumulated, weighted by how much it absorbs (1 - stepT).
            scatter += transmittance * inScatter * (1.0 - stepT);
            transmittance *= stepT;
            if (transmittance < 0.003) break;   // saturated -> stop early (deterministic threshold)
        }
    }

    float alpha = 1.0 - transmittance;   // coverage: 0 = clear sky, 1 = opaque cloud
    return float4(scatter, alpha);
}
