// Slice CR — Ground-Truth Ambient Occlusion (GTAO) pass (NEW fullscreen pass). Mirrors ssao.frag's
// G-buffer setup: a fullscreen pass over the view-space normal (xyz) + view-space linear depth (w)
// g-buffer (bound at t0/s0 like ssao.frag), reconstructing each pixel's view-space position from the
// linear depth + screen UV + projection params (tanHalfFovY, aspect) — NO matrix inverse. It then runs
// the render/gtao.h horizon-search visibility integral (IntegrateArc + HorizonAngle + Visibility copied
// VERBATIM below) with a BAKED per-pixel slice rotation (no RNG) and outputs the AO factor (single
// channel, R; 1 = open, < 1 = occluded). A resolve/apply step (the EXISTING ssao_composite.frag,
// reused unchanged) MULTIPLIES the scene's AMBIENT term by this AO — direct sun unaffected — exactly
// like the SSAO apply. The EXISTING SSAO shader + its golden + the default lit path stay BYTE-IDENTICAL
// (GTAO is a NEW path behind --gtao-shot). With radius == 0 (or a FLAT field) the horizon search raises
// no horizon -> Visibility == 1 EXACTLY -> the composite multiply by 1 leaves the scene byte-identical
// to the no-AO render (the radius=0 equivalence proof).
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL here are the
// HF_MSL_GEN generation-path guards + [[vk::binding]] decorations (same as ssao.frag), not backend
// CODE symbols. spirv-cross maps these SPIR-V bindings to the engine's flat Metal texture/sampler
// indices so the SAME HLSL feeds Vulkan (DXC) and Metal (glslang->spirv-cross): bit-identical math.
[[vk::binding(0, 0)]] Texture2D    gGbuf : register(t0);   // view-space normal.xyz + linDepth.w
[[vk::binding(1, 0)]] SamplerState gSmp  : register(s0);

struct GtaoParams {
    float2 texel;       // 1/gbufferSize
    float  radius;      // horizon-search radius in view-space units (0 -> AO == 1 everywhere)
    float  slices;      // number of slice directions (cast to int)
    float  steps;       // steps per slice per side (cast to int)
    float  tanHalfFovY; // tan(0.5*fovY) for view-space reconstruction
    float  aspect;      // width/height
    float  intensity;   // AO strength multiplier on (1-ao); 1 = physical
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer GtaoPC { GtaoParams gp; };
#define HF_GP gp
#else
[[vk::push_constant]] struct { GtaoParams p; } pc;
#define HF_GP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

static const float HF_PI     = 3.14159265358979323846;
static const float HF_HALFPI = 1.57079632679489661923;

// Y-flip sign mapping screen UV.y <-> view-space Y, IDENTICAL to ssao.frag: -1 on Vulkan (projection
// bakes a Y-flip + post.vert gives a V-down UV), +1 on Metal (FlipProjY + post.vert V-flip compose).
#ifdef HF_MSL_GEN
static const float HF_YS = 1.0;
#else
static const float HF_YS = -1.0;
#endif

// Reconstruct view-space position for a UV + linear depth (IDENTICAL to ssao.frag::ReconstructViewPos).
float3 ReconstructViewPos(float2 uv, float linDepth) {
    float2 ndc = uv * 2.0 - 1.0;
    float vx = ndc.x * (HF_GP.aspect * HF_GP.tanHalfFovY) * linDepth;
    float vy = HF_YS * ndc.y * (HF_GP.tanHalfFovY) * linDepth;
    float vz = -linDepth;                   // RH view space: -Z forward
    return float3(vx, vy, vz);
}

// Forward-project a view-space position to UV (IDENTICAL to ssao.frag::ProjectToUV).
float2 ProjectToUV(float3 vp) {
    float invZ = 1.0 / max(-vp.z, 1e-4);
    float ndcx = vp.x / (HF_GP.aspect * HF_GP.tanHalfFovY) * invZ;
    float ndcy = HF_YS * vp.y / (HF_GP.tanHalfFovY) * invZ;
    return float2(ndcx, ndcy) * 0.5 + 0.5;
}

// ---- render/gtao.h IntegrateArc, copied VERBATIM. The published Jimenez et al. closed form for the
// per-slice cosine-weighted visibility between two horizons h1,h2 with the slice-projected normal n.
// IntegrateArc(n - π/2, n + π/2, n) == the fully-open visibility (1 at n == 0) -> the radius=0 proof. ----
float IntegrateArc(float h1, float h2, float n) {
    float side1 = -cos(2.0 * h1 - n) + cos(n) + 2.0 * h1 * sin(n);
    float side2 = -cos(2.0 * h2 - n) + cos(n) + 2.0 * h2 * sin(n);
    return 0.25 * (side1 + side2);
}

// ---- render/gtao.h HorizonAngle, copied VERBATIM. The angle of a marched sample FROM the view dir
// toward the slice tangent: atan2(d·tangent, d·viewDir). A flush tangent-plane sample -> ±π/2 (open);
// an occluder rising toward the camera -> angle pulled in toward 0 (the horizon raised). ----
float HorizonAngle(float3 viewPos, float3 samplePos, float3 sliceTangent, float3 viewDir) {
    float3 d = samplePos - viewPos;
    float t = dot(d, sliceTangent);
    float p = dot(d, viewDir);
    if (abs(t) < 1e-9 && abs(p) < 1e-9) return (t < 0.0) ? -HF_HALFPI : HF_HALFPI;  // coincident -> open
    return atan2(t, p);
}

// ---- render/gtao.h Visibility, copied VERBATIM (the sampler is inlined as a G-buffer depth fetch via
// ProjectToUV). Returns AO ∈ [0,1] (1 = unoccluded). radius == 0 -> 1 EXACTLY. ----
float GtaoVisibility(float3 viewPos, float3 viewNormal, float radius, int slices, int stepsPerSlice,
                     float rotationOffset) {
    if (radius <= 0.0) return 1.0;                 // radius=0 -> no horizon search -> AO == 1 EXACTLY
    if (slices < 1) slices = 1;
    if (stepsPerSlice < 1) stepsPerSlice = 1;

    float3 viewDir = normalize(-viewPos);          // fragment -> eye
    float3 N = normalize(viewNormal);

    // View-space basis spanning the plane perpendicular to the view direction (deterministic, no RNG).
    float3 up0 = (abs(viewDir.y) < 0.99) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 right = normalize(cross(up0, viewDir));
    float3 up    = cross(viewDir, right);

    float visSum = 0.0;
    [loop] for (int s = 0; s < slices; ++s) {
        float phi = rotationOffset + HF_PI * (float)s / (float)slices;
        float cphi = cos(phi), sphi = sin(phi);
        float3 sliceTangent = right * cphi + up * sphi;

        float nTan  = dot(N, sliceTangent);
        float nPerp = dot(N, viewDir);
        float n = atan2(nTan, nPerp);

        float h2 =  HF_HALFPI;   // +tangent horizon (min toward 0 as occluders rise)
        float h1 = -HF_HALFPI;   // -tangent horizon (max toward 0 as occluders rise)
        float dt = radius / (float)stepsPerSlice;
        [loop] for (int k = 1; k <= stepsPerSlice; ++k) {
            float t = dt * (float)k;
            // +sliceTangent sample: the screen offset, depth from the G-buffer at its projected UV.
            float3 sPos = viewPos + sliceTangent * t;
            float2 uvP = ProjectToUV(sPos);
            float dP = gGbuf.Sample(gSmp, uvP).w;          // view-space linear depth at the sample
            if (dP > 0.0001) {                              // skip background (cleared w == 0)
                sPos.z = -dP;
                float aPos = HorizonAngle(viewPos, sPos, sliceTangent, viewDir);
                if (aPos < h2) h2 = aPos;
            }
            // -sliceTangent sample.
            float3 sNeg = viewPos - sliceTangent * t;
            float2 uvN = ProjectToUV(sNeg);
            float dN = gGbuf.Sample(gSmp, uvN).w;
            if (dN > 0.0001) {
                sNeg.z = -dN;
                float aNeg = HorizonAngle(viewPos, sNeg, sliceTangent, viewDir);
                if (aNeg > h1) h1 = aNeg;
            }
        }
        h2 = min(h2, n + HF_HALFPI);
        h1 = max(h1, n - HF_HALFPI);
        visSum += IntegrateArc(h1, h2, n);
    }
    float ao = visSum / (float)slices;
    return saturate(ao);
}

float4 main(PSInput i) : SV_Target {
    float4 g = gGbuf.Sample(gSmp, i.uv);
    float  linDepth = g.w;
    // Background / no geometry (cleared w == 0): fully unoccluded.
    if (linDepth <= 0.0001) return float4(1.0, 1.0, 1.0, 1.0);

    float3 P = ReconstructViewPos(i.uv, linDepth);
    float3 N = normalize(g.xyz);
    // Orient the normal toward the camera (IDENTICAL to ssao.frag) so a back-facing geometric normal
    // (e.g. the wide ground plane) does not invert the hemisphere.
    if (dot(N, -normalize(P)) < 0.0) N = -N;

    int slices = (int)(HF_GP.slices + 0.5);
    int steps  = (int)(HF_GP.steps + 0.5);
    // BAKED per-pixel slice rotation (NO RNG): a fixed 4x4-tiled rotation from the integer pixel coords,
    // decorrelating neighbor pixels' slice fans to reduce banding. Deterministic -> two runs identical.
    // The radius=0 / flat-field identity holds for any rotation, so this never breaks the proof.
    int2 px = int2(i.uv / HF_GP.texel);
    int ri = (px.x & 3) + ((px.y & 3) << 2);          // 0..15 tile index
    float rot = (HF_PI / 16.0) * (float)ri;            // baked rotation offset in [0, π)
    float vis = GtaoVisibility(P, N, HF_GP.radius, slices, steps, rot);

    // intensity scales the OCCLUSION (1 - vis); intensity 1 = physical. radius=0 -> vis=1 -> ao=1.
    float ao = 1.0 - (1.0 - vis) * HF_GP.intensity;
    ao = saturate(ao);
    return float4(ao, ao, ao, 1.0);
}
