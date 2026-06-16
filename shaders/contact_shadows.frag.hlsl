// Slice CT — Screen-Space Contact Shadows pass (NEW fullscreen pass). Mirrors gtao.frag / ssao.frag's
// G-buffer setup: a fullscreen pass over the view-space normal (xyz) + view-space linear depth (w)
// g-buffer (bound at t3/s3 via BindTexturePair, like the SSAO/GTAO apply), reconstructing each pixel's
// view-space position from the linear depth + screen UV + projection params (tanHalfFovY, aspect) — NO
// matrix inverse. It then marches a SHORT ray TOWARD the sun through the depth field
// (render/contact_shadows.h RayMarchShadow, copied VERBATIM below) and outputs the contact-shadow factor
// (single channel, R; 1 = lit, < 1 = in contact shadow). A resolve/apply step (the lit_contactshadow
// scene shader) MULTIPLIES the scene's DIRECT SUN contribution by this factor — ambient/IBL/point lights
// unaffected — exactly like the CSM shadow term. The EXISTING lit/CSM shadow shaders + their goldens +
// the default lit path stay BYTE-IDENTICAL (contact shadows are a NEW path behind --contactshadow-shot).
// With maxDist == 0 (or steps == 0) the march takes no steps -> no occluder is found -> the factor is 1
// EXACTLY -> the sun multiply by 1 leaves the scene byte-identical to the no-contact-shadow render (the
// maxDist=0 no-op proof).
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL here are the
// HF_MSL_GEN generation-path guards + [[vk::binding]] decorations (same as gtao.frag/ssao.frag), not
// backend CODE symbols. spirv-cross maps these SPIR-V bindings to the engine's flat Metal texture/sampler
// indices so the SAME HLSL feeds Vulkan (DXC) and Metal (glslang->spirv-cross): bit-identical math.
[[vk::binding(3, 0)]] Texture2D    gGbuf : register(t3);   // view-space normal.xyz + linDepth.w (t3/s3)
[[vk::binding(4, 0)]] SamplerState gSmp  : register(s3);

struct ContactParams {
    float2 texel;       // 1/gbufferSize
    float  maxDist;     // march length in view-space units (0 -> factor == 1 everywhere)
    float  steps;       // march step count (cast to int)
    float  thickness;   // max depth-difference window (no false shadow from a distant background)
    float  bias;        // self-occlusion guard (avoid contact-acne on flat lit surfaces)
    float  tanHalfFovY; // tan(0.5*fovY) for view-space reconstruction
    float  aspect;      // width/height
    float4 sunDirView;  // xyz = view-space direction TO the sun (normalized); w unused
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer ContactPC { ContactParams cp; };
#define HF_CP cp
#else
[[vk::push_constant]] struct { ContactParams p; } pc;
#define HF_CP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// Y-flip sign mapping screen UV.y <-> view-space Y, IDENTICAL to gtao.frag/ssao.frag: -1 on Vulkan
// (projection bakes a Y-flip + post.vert gives a V-down UV), +1 on Metal (FlipProjY + post.vert V-flip).
#ifdef HF_MSL_GEN
static const float HF_YS = 1.0;
#else
static const float HF_YS = -1.0;
#endif

// Reconstruct view-space position for a UV + linear depth (IDENTICAL to gtao.frag::ReconstructViewPos).
float3 ReconstructViewPos(float2 uv, float linDepth) {
    float2 ndc = uv * 2.0 - 1.0;
    float vx = ndc.x * (HF_CP.aspect * HF_CP.tanHalfFovY) * linDepth;
    float vy = HF_YS * ndc.y * (HF_CP.tanHalfFovY) * linDepth;
    float vz = -linDepth;                   // RH view space: -Z forward
    return float3(vx, vy, vz);
}

// ---- render/contact_shadows.h ProjectToUV, copied VERBATIM (forward-project a view-space position to
// UV; IDENTICAL to gtao.frag::ProjectToUV / ssr.h::ViewToScreenUV). ----
float2 ProjectToUV(float3 vp) {
    float invZ = 1.0 / max(-vp.z, 1e-4);
    float ndcx = vp.x / (HF_CP.aspect * HF_CP.tanHalfFovY) * invZ;
    float ndcy = HF_YS * vp.y / (HF_CP.tanHalfFovY) * invZ;
    return float2(ndcx, ndcy) * 0.5 + 0.5;
}

// ---- render/contact_shadows.h RayMarchShadow, copied VERBATIM (the sampler is inlined as a G-buffer
// depth fetch via ProjectToUV). Returns the contact-shadow factor in [0,1] (1 = lit). maxDist == 0 ||
// steps == 0 -> 1 EXACTLY (the no-op identity). ----
float RayMarchShadow(float3 viewPos, float3 lightDirView, float maxDist, int steps,
                     float thickness, float bias, float ditherOffset) {
    if (maxDist <= 0.0 || steps <= 0) return 1.0;     // no march -> no occluder -> fully lit EXACTLY

    float3 L = normalize(lightDirView);               // toward the sun, in view space
    float stepLen = maxDist / (float)steps;

    [loop] for (int k = 1; k <= steps; ++k) {
        float t = stepLen * ((float)(k - 1) + ditherOffset + 1.0);
        if (t > maxDist) t = maxDist;
        float3 marchPos = viewPos + L * t;

        float rayDist = -marchPos.z;                   // view-space distance of the marched ray point
        if (rayDist <= 1e-4) continue;                 // behind the camera -> skip

        float2 uv = ProjectToUV(marchPos);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;  // off-screen -> skip

        float sceneDist = gGbuf.Sample(gSmp, uv).w;    // view-space linear depth of the real surface
        if (sceneDist <= 1e-4) continue;               // background (cleared w == 0) -> skip

        float diff = rayDist - sceneDist;              // > 0 when the surface is closer than the ray
        if (diff > bias && diff <= thickness) {
            float frac = t / maxDist;                  // soft falloff by march progress
            return saturate(frac);
        }
    }
    return 1.0;                                        // clear march -> fully lit
}

float4 main(PSInput i) : SV_Target {
    float4 g = gGbuf.Sample(gSmp, i.uv);
    float  linDepth = g.w;
    // Background / no geometry (cleared w == 0): fully lit (no contact shadow on the sky).
    if (linDepth <= 0.0001) return float4(1.0, 1.0, 1.0, 1.0);

    float3 P = ReconstructViewPos(i.uv, linDepth);
    int steps = (int)(HF_CP.steps + 0.5);

    // BAKED per-pixel dither (NO RNG): a fixed 4x4-tiled fractional step offset from the integer pixel
    // coords, hiding the march step banding. Deterministic -> two runs identical. The maxDist=0 identity
    // holds for ANY dither (every step is still 0), so this never breaks the byte-identical proof.
    int2 px = int2(i.uv / HF_CP.texel);
    int di = (px.x & 3) + ((px.y & 3) << 2);          // 0..15 tile index
    float dither = (1.0 / 16.0) * (float)di;          // baked offset in [0, 1)

    float factor = RayMarchShadow(P, HF_CP.sunDirView.xyz, HF_CP.maxDist, steps,
                                  HF_CP.thickness, HF_CP.bias, dither);
    return float4(factor, factor, factor, 1.0);
}
