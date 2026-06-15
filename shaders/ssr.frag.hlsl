// Screen-space reflections — ray-march pass (Slice AH). Classic single-bounce SSR. For each pixel
// it reconstructs the VIEW-SPACE position P and normal N from the g-buffer (view normal.xyz +
// linear depth.w, written by gbuffer.frag — the SAME g-buffer SSAO uses), computes the view-space
// reflection ray R = reflect(normalize(P), N), and RAY-MARCHES along R in view space. Each step is
// projected to a screen UV (ProjectToUV) and the ray's view depth is compared to the g-buffer depth
// stored at that UV. On a hit (ray crosses behind the stored surface within `thickness`) a binary
// search refines the intersection UV, then the HDR scene-color RT is sampled there -> reflected
// radiance. The reflection is masked by a FLOOR reflectivity (high dot(N, viewUp)), a Fresnel term,
// a screen-edge fade, and faded to 0 when the ray exits the screen / runs past maxDist with no hit
// (fallback = the scene's existing in-shader IBL, shown by the composite). Output: rgb = reflected
// radiance, a = blended weight (reflectivity * Fresnel * edge/march fade). Deterministic — a baked
// 4x4 dither jitters the march START by a fraction of a step to break banding; no RNG.
//
// Bindings mirror the bloom/SSAO composite slots: scene HDR color at t0/s0, g-buffer at t3/s3 (bound
// together via ICommandBuffer::BindTexturePair). Existing gbuffer/lit/ssao/bloom shaders + their
// goldens are UNTOUCHED.
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // HDR scene color (radiance to reflect)
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gGbuf  : register(t3);   // view normal.xyz + linDepth.w
[[vk::binding(4, 0)]] SamplerState gGSmp  : register(s3);

struct SsrParams {
    float2 texel;        // 1/size
    float  tanHalfFovY;  // view-space reconstruction
    float  aspect;       // width/height
    float  maxDist;      // total view-space march length
    float  thickness;    // depth-compare band (view units)
    float  reflMin;      // dot(N,viewUp) smoothstep lo -> floor reflectivity mask
    float  reflMax;      // dot(N,viewUp) smoothstep hi
    float4 viewUp;       // camera world-up expressed in VIEW space (xyz); w unused
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer SsrPC { SsrParams sp; };
#define HF_SP sp
#else
[[vk::push_constant]] struct { SsrParams p; } pc;
#define HF_SP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// Y-flip sign mapping screen UV.y <-> view-space Y — IDENTICAL convention to ssao.frag. On Vulkan the
// projection bakes a Y-flip and post.vert gives a V-down UV (sign -1); on Metal FlipProjY + post.vert
// V-flip compose the other way (sign +1). ReconstructViewPos/ProjectToUV stay mutual inverses AND
// agree with how the g-buffer was rasterized on each backend, so the screen-space march samples the
// right texel without any extra flip.
#ifdef HF_MSL_GEN
static const float HF_YS = 1.0;
#else
static const float HF_YS = -1.0;
#endif

float3 ReconstructViewPos(float2 uv, float linDepth) {
    float2 ndc = uv * 2.0 - 1.0;
    float vx = ndc.x * (HF_SP.aspect * HF_SP.tanHalfFovY) * linDepth;
    float vy = HF_YS * ndc.y * (HF_SP.tanHalfFovY) * linDepth;
    float vz = -linDepth;                   // RH view space: -Z forward
    return float3(vx, vy, vz);
}
float2 ProjectToUV(float3 vp) {
    float invZ = 1.0 / max(-vp.z, 1e-4);
    float ndcx = vp.x / (HF_SP.aspect * HF_SP.tanHalfFovY) * invZ;
    float ndcy = HF_YS * vp.y / (HF_SP.tanHalfFovY) * invZ;
    return float2(ndcx, ndcy) * 0.5 + 0.5;
}

// Baked 4x4 ordered-dither (Bayer/16) in [0,1): jitters the march start by a sub-step fraction so the
// fixed step grid doesn't band. Deterministic per pixel -> byte-stable golden.
static const float kDither[16] = {
    0.0 / 16.0,  8.0 / 16.0,  2.0 / 16.0, 10.0 / 16.0,
   12.0 / 16.0,  4.0 / 16.0, 14.0 / 16.0,  6.0 / 16.0,
    3.0 / 16.0, 11.0 / 16.0,  1.0 / 16.0,  9.0 / 16.0,
   15.0 / 16.0,  7.0 / 16.0, 13.0 / 16.0,  5.0 / 16.0,
};

static const int   kSteps   = 48;   // linear march steps
static const int   kRefine  = 6;    // binary-search refinement iterations
static const float kEdgeFade = 0.12; // UV-border fade band

float4 main(PSInput i) : SV_Target {
    float4 g = gGbuf.Sample(gGSmp, i.uv);
    float  linDepth = g.w;
    // Background / no surface (g-buffer cleared w == 0): no reflection.
    if (linDepth <= 0.0001) return float4(0.0, 0.0, 0.0, 0.0);

    float3 N = normalize(g.xyz);                 // view-space surface normal
    float3 P = ReconstructViewPos(i.uv, linDepth);
    float3 V = normalize(P);                      // camera(origin) -> fragment (incident direction)

    // Reflectivity mask: the FLOOR is (near-)horizontal -> its view normal aligns with the camera's
    // world-up expressed in view space. Objects (spheres/cubes) have varied normals and fail this, so
    // they get ~0 reflectivity. (Keeps gbuffer.frag + its golden untouched — no per-pixel material.)
    float upAlign = abs(dot(N, normalize(HF_SP.viewUp.xyz)));
    float reflectivity = smoothstep(HF_SP.reflMin, abs(HF_SP.reflMax), upAlign);
    // Debug mode (reflMax < 0): show upAlign in R (floor-mask source) and reflectivity in G so we can
    // confirm the floor is detected before the march runs. B holds the march result (set below).
    bool dbg = (HF_SP.reflMax < 0.0);
    if (dbg && reflectivity <= 0.001) return float4(upAlign, 0.0, 0.0, 1.0);
    if (!dbg && reflectivity <= 0.001) return float4(0.0, 0.0, 0.0, 0.0);

    float3 R = reflect(V, N);                     // view-space reflection ray
    // Ray pointing back toward/behind the camera (into the near plane): can't march on-screen.
    if (R.z > -0.001) {
        if (dbg) return float4(upAlign, reflectivity, 0.0, 1.0);
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    // March in view space. Dither the start by a sub-step fraction to break banding.
    int2 px = int2(i.uv / HF_SP.texel);
    float jitter = kDither[(px.x & 3) + ((px.y & 3) << 2)];
    float stepLen = HF_SP.maxDist / (float)kSteps;

    float3 prevPos = P;                           // last sample known to be IN FRONT of the surface
    float  prevDelta = 0.0;                       // rayDepth - sceneDepth at prevPos (negative = in front)
    bool   hit = false;
    float2 hitUV = float2(0.0, 0.0);

    [loop] for (int s = 1; s <= kSteps; ++s) {
        float t = ((float)s + jitter) * stepLen;
        float3 samplePos = P + R * t;
        float2 sUV = ProjectToUV(samplePos);
        // Ray left the screen -> stop (no further on-screen geometry to reflect).
        if (sUV.x < 0.0 || sUV.x > 1.0 || sUV.y < 0.0 || sUV.y > 1.0) break;

        float storedDepth = gGbuf.Sample(gGSmp, sUV).w;   // surface depth at this screen location
        if (storedDepth <= 0.0001) { prevPos = samplePos; prevDelta = -1.0; continue; } // background gap

        float rayDepth = -samplePos.z;                    // ray's own view-space linear depth
        float delta = rayDepth - storedDepth;             // >0: ray is BEHIND the stored surface
        // HIT: the ray crossed behind the surface AND it's within the thickness band (a real surface,
        // not the far void behind a silhouette).
        if (delta > 0.0 && delta < HF_SP.thickness) {
            // Binary-search refine between prevPos (in front) and samplePos (behind).
            float3 lo = prevPos;
            float3 hi2 = samplePos;
            [unroll] for (int r = 0; r < kRefine; ++r) {
                float3 mid = (lo + hi2) * 0.5;
                float2 mUV = ProjectToUV(mid);
                float  mD  = gGbuf.Sample(gGSmp, mUV).w;
                float  md  = -mid.z - mD;
                if (mD > 0.0001 && md > 0.0) hi2 = mid; else lo = mid;
            }
            hitUV = ProjectToUV(hi2);
            hit = true;
            break;
        }
        prevPos = samplePos;
        prevDelta = delta;
    }

    if (!hit) {
        if (dbg) return float4(upAlign, reflectivity, 0.0, 1.0);
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    // Screen-edge fade: attenuate as the hit UV approaches any border so reflections don't pop where
    // the marched ray exits the screen.
    float2 fade2 = smoothstep(0.0.xx, kEdgeFade.xx, hitUV)
                 * smoothstep(0.0.xx, kEdgeFade.xx, 1.0.xx - hitUV);
    float edgeFade = fade2.x * fade2.y;

    // Fresnel-Schlick on the view angle: grazing reflections are stronger. A polished-floor base
    // reflectance (kBaseRefl) is used instead of the dielectric 0.04 so the reflection reads clearly
    // even at the near-normal angles in the lower-foreground floor; the Schlick term still boosts the
    // grazing far-floor toward a full mirror.
    static const float kBaseRefl = 0.55;
    float NoV = saturate(dot(N, -V));
    float fresnel = kBaseRefl + (1.0 - kBaseRefl) * pow(1.0 - NoV, 5.0);

    float weight = saturate(reflectivity * edgeFade * fresnel);
    float3 reflected = gScene.Sample(gSmp, hitUV).rgb;
    // Debug: reflMax < 0 -> R=upAlign, G=reflectivity, B=hit weight for pipeline inspection.
    if (dbg) return float4(upAlign, reflectivity, weight, 1.0);
    return float4(reflected, weight);
}
