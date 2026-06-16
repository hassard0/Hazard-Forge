// Screen-space global illumination — indirect-diffuse gather pass (Slice BP). One bounce of indirect
// diffuse light via screen space. For each pixel it reconstructs the VIEW-SPACE position P and normal
// N from the g-buffer (view normal.xyz + linear depth.w, written by gbuffer.frag — the SAME g-buffer
// SSAO/SSR/decal read), then traces K rays in the COSINE-WEIGHTED hemisphere about N. Each ray is
// ray-marched in view space (the SAME march + binary-search as ssr.frag); on a hit the ALREADY-LIT
// HDR scene color at that screen UV is the incoming radiance for that ray. The mean of the K hit
// radiances is the indirect diffuse irradiance: output rgb = (sum/K) * intensity, a = 1. The composite
// (ssgi_composite) ADDS this over the scene. A baked 4x4 dither rotates the hemisphere kernel per
// pixel (a FIXED pattern, NO RNG) to break banding — single frame, NO temporal accumulation.
//
// The hemisphere kernel (HemisphereDir) is the EXACT CPU math in engine/render/ssgi.h: Hammersley
// (u1 stratified, u2 = radical-inverse base 2) -> cosine (Malley) map -> rotate by a stable TBN built
// from N. The view<->screen reconstruction (ReconstructViewPos/ProjectToUV) is COPIED VERBATIM from
// ssr.frag (the shared SSR math, mirrored on the CPU by render/ssr.h). Bindings mirror SSR: scene HDR
// at t0/s0, g-buffer at t3/s3 (BindTexturePair). Existing gbuffer/lit/ssao/ssr/bloom shaders + their
// goldens are UNTOUCHED.
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // lit HDR scene color (incoming radiance)
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gGbuf  : register(t3);   // view normal.xyz + linDepth.w
[[vk::binding(4, 0)]] SamplerState gGSmp  : register(s3);

struct SsgiParams {
    float2 texel;        // 1/size
    float  tanHalfFovY;  // view-space reconstruction (matches ssr.frag)
    float  aspect;       // width/height
    float  maxDist;      // total view-space march length per ray
    float  thickness;    // depth-compare band (view units)
    float  intensity;    // indirect-diffuse gain
    float  rayCount;     // K hemisphere rays (as a float; clamped to [1, kMaxRays])
    float  frame;        // Slice BV: temporal accumulation frame index (0 = base kernel, no rotation)
    float3 _pad;         // pad to a 16-byte boundary (frame + 3 floats = one float4 row)
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer SsgiPC { SsgiParams sp; };
#define HF_SP sp
#else
[[vk::push_constant]] struct { SsgiParams p; } pc;
#define HF_SP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// Y-flip sign mapping screen UV.y <-> view-space Y — IDENTICAL convention to ssr.frag/ssao.frag.
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

// Baked 4x4 ordered-dither (Bayer/16) in [0,1): rotates the hemisphere kernel per pixel by a FIXED
// pattern so the fixed ray set doesn't band. Deterministic per pixel -> byte-stable golden (no RNG).
static const float kDither[16] = {
    0.0 / 16.0,  8.0 / 16.0,  2.0 / 16.0, 10.0 / 16.0,
   12.0 / 16.0,  4.0 / 16.0, 14.0 / 16.0,  6.0 / 16.0,
    3.0 / 16.0, 11.0 / 16.0,  1.0 / 16.0,  9.0 / 16.0,
   15.0 / 16.0,  7.0 / 16.0, 13.0 / 16.0,  5.0 / 16.0,
};

// Slice BV: golden-angle per-frame azimuth step (as a fraction of a full turn) — IDENTICAL to
// render/ssgi.h kGoldenAngleTurns. rot(frame) = frame * this is ADDED to the per-pixel dither so the
// spatial dither and the temporal rotation compose into the single azimuth offset. frame 0 -> 0 -> the
// raw --ssgi-shot kernel is byte-identical.
static const float kGoldenAngleTurns = 0.38196601125010515;

static const int   kMaxRays  = 32;   // hard upper bound on hemisphere rays (loop cap)
static const int   kSteps    = 32;   // linear march steps per ray
static const int   kRefine   = 5;    // binary-search refinement iterations

// Van der Corput radical inverse base 2 — IDENTICAL to render/ssgi.h RadicalInverse2.
float RadicalInverse2(uint i) {
    i = (i << 16) | (i >> 16);
    i = ((i & 0x55555555u) << 1) | ((i & 0xAAAAAAAAu) >> 1);
    i = ((i & 0x33333333u) << 2) | ((i & 0xCCCCCCCCu) >> 2);
    i = ((i & 0x0F0F0F0Fu) << 4) | ((i & 0xF0F0F0F0u) >> 4);
    i = ((i & 0x00FF00FFu) << 8) | ((i & 0xFF00FF00u) >> 8);
    return float(i) * 2.3283064365386963e-10;  // / 2^32
}

// Stable TBN basis for a unit normal — IDENTICAL to render/ssgi.h BuildTangentBasis.
void BuildTangentBasis(float3 n, out float3 T, out float3 B) {
    float3 an = abs(n);
    float3 ref = (an.x <= an.y && an.x <= an.z) ? float3(1, 0, 0)
               : (an.y <= an.z ? float3(0, 1, 0) : float3(0, 0, 1));
    T = normalize(cross(ref, n));
    B = cross(n, T);
}

// i-th of K cosine-weighted hemisphere directions about N — IDENTICAL to render/ssgi.h HemisphereDir.
// `rot` rotates the azimuth by the per-pixel dither (a fixed offset, applied to phi) so different
// pixels sample rotated copies of the same fixed set (still deterministic, no RNG).
float3 HemisphereDir(int i, int K, float3 N, float rot) {
    float u1 = (float(i) + 0.5) / float(K);
    float u2 = RadicalInverse2((uint)i);
    float r = sqrt(max(0.0, u1));
    float phi = 6.28318530718 * (u2 + rot);
    float lx = r * cos(phi);
    float ly = r * sin(phi);
    float lz = sqrt(max(0.0, 1.0 - u1));
    float3 T, B; BuildTangentBasis(N, T, B);
    return normalize(T * lx + B * ly + N * lz);
}

float4 main(PSInput i) : SV_Target {
    float4 g = gGbuf.Sample(gGSmp, i.uv);
    float  linDepth = g.w;
    // Background / no surface (g-buffer cleared w == 0, e.g. skybox): no indirect.
    if (linDepth <= 0.0001) return float4(0.0, 0.0, 0.0, 0.0);

    float3 N = normalize(g.xyz);                 // view-space surface normal
    float3 P = ReconstructViewPos(i.uv, linDepth);

    int K = (int)clamp(HF_SP.rayCount, 1.0, (float)kMaxRays);

    // Per-pixel kernel rotation from the baked 4x4 dither, PLUS the Slice BV temporal per-frame
    // golden-angle rotation (frame 0 adds 0 -> byte-identical to the raw single-frame --ssgi-shot).
    int2 px = int2(i.uv / HF_SP.texel);
    float rot = kDither[(px.x & 3) + ((px.y & 3) << 2)] + HF_SP.frame * kGoldenAngleTurns;

    float stepLen = HF_SP.maxDist / (float)kSteps;
    float3 sum = float3(0.0, 0.0, 0.0);

    [loop] for (int rIdx = 0; rIdx < kMaxRays; ++rIdx) {
        if (rIdx >= K) break;
        float3 dir = HemisphereDir(rIdx, K, N, rot);

        // March `dir` from P. Nudge the start slightly off the surface to avoid self-hit.
        float3 prevPos = P + dir * (stepLen * 0.5);
        bool   hit = false;
        float2 hitUV = float2(0.0, 0.0);

        [loop] for (int s = 1; s <= kSteps; ++s) {
            float3 samplePos = P + dir * ((float)s * stepLen);
            float2 sUV = ProjectToUV(samplePos);
            if (sUV.x < 0.0 || sUV.x > 1.0 || sUV.y < 0.0 || sUV.y > 1.0) break;  // off-screen

            float storedDepth = gGbuf.Sample(gGSmp, sUV).w;
            if (storedDepth <= 0.0001) { prevPos = samplePos; continue; }         // background gap

            float rayDepth = -samplePos.z;
            float delta = rayDepth - storedDepth;            // >0: ray is BEHIND the stored surface
            if (delta > 0.0 && delta < HF_SP.thickness) {
                // Binary-search refine between prevPos (in front) and samplePos (behind).
                float3 lo = prevPos;
                float3 hi2 = samplePos;
                [unroll] for (int rr = 0; rr < kRefine; ++rr) {
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
        }

        // On a hit, the lit scene color there is the incoming radiance; misses contribute 0.
        if (hit) sum += gScene.Sample(gSmp, hitUV).rgb;
    }

    // Monte-Carlo estimator: mean over K (cosine weight baked into the cosine-distributed sampling).
    float3 indirect = (sum / (float)K) * HF_SP.intensity;
    return float4(indirect, 1.0);
}
