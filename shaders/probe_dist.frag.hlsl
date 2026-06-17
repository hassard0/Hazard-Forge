// Slice DO — DDGI Visibility Slice 1: per-probe DISTANCE-moment capture fragment shader. The DI
// radiance-capture analog (DI:radiance :: DO:distance): instead of lit radiance, each fragment outputs the
// LINEAR WORLD-DISTANCE from the probe centre to this surface — d = length(worldPos - probeCentre) —
// packed as the two MOMENTS float2(d, d*d) into the RGBA16F distance target (B = A = 0). This is the
// visibility data layer the next slice (DP, Chebyshev occlusion) consumes to KILL DDGI light-leak through
// walls: meanDist = m[0], variance = m[1] - m[0]², Chebyshev p = variance/(variance+(d-mean)²).
//
// PURE GEOMETRY: NO lighting, NO shadows, NO texture — the output depends ONLY on the world-space distance
// from the probe centre, so it is VIEW-INDEPENDENT per probe and (given the identical mesh) cross-backend-
// identical. Where no geometry is hit (sky / miss) the render-pass CLEAR provides the far value; the host
// clears the distance face to kDistFar = 1e4 (a kRayMiss-style far moment {1e4, 1e8}).
//
// THE BIT-EXACT FRAMING (the make-or-break correctness detail — see engine/render/probe_dist.h banner):
// the distance target is RGBA16F, so the stored R = fp16(d) is what the host reads back as the distance.
// To make the GPU's SECOND moment bit-identical to the CPU reference computed from that SAME read-back
// distance, the shader ROUNDS d to fp16 (f32tof16/f16tof32) BEFORE squaring — so the d that gets squared
// is EXACTLY the d the host reads back (the stored R), not the full-precision pre-rounding distance. The
// second moment is then a BARE MULTIPLY `dh*dh` (one IEEE-754 fp32 round, then the fp16 store-round), NOT
// mad/fma/a+b*c. The CPU reference probedist::MomentsFromDistance(dRead) = { dRead, dRead*dRead } takes the
// read-back distance dRead = f16tof32(R) (which == dh exactly) and does the SAME single bare multiply, so
// GIVEN the SAME distance bytes the GPU's stored G = fp16(dh*dh) == the CPU's fp16(dRead*dRead) to the BIT.
// We deliberately avoid an fma where a plain multiply is already exactly correctly-rounded for one product
// (forcing an fma risks the DH contraction divergence — Metal contracts a+b*c to fma, others don't). The
// DISTANCE itself uses length()==sqrt, which is NOT guaranteed bit-identical CPU↔GPU, so the GPU==CPU
// bit-exact proof is on THIS moment-from-distance step (given the same fp16 d), and the capture-distance
// proof is a render-equivalence proof (the captured face == a direct distance render with that face's
// view/proj, per backend).
//
// Push constant: { float4x4 faceViewProj; float4x4 model; float4 probeCentre; } = 144 bytes (shared with
// probe_dist.vert; the frag reads probeCentre.xyz). NEW shader behind the --probedist-shot / --probedist
// showcase ONLY; lit.frag + every existing pipeline + golden stay BYTE-IDENTICAL.
struct PSInput {
    float4 clip : SV_Position;
    [[vk::location(0)]] float3 wpos : POSITION0;   // world position from probe_dist.vert
};
// HF_MSL_GEN seam-discipline: in Vulkan a SINGLE [[vk::push_constant]] block is shared vertex+fragment;
// for MSL generation the Metal RHI binds the push bytes to the VERTEX push slot (buffer 2) AND, when the
// pipeline sets fragmentPushConstants=true, also to the FRAGMENT push slot (buffer 1). So the fragment's
// MSL cbuffer is declared at binding(1,0) (-> fragment buffer 1 = kFbPushConst, like cas/bloom/dof frags),
// while the vertex declares the SAME block at binding(2,0) (-> vertex buffer 2). The block LAYOUT
// (faceViewProj, model, probeCentre) is identical in both so probeCentre sits at the SAME 128-byte offset.
// (#ifdef'd names are NOT backend code symbols — seam-discipline macros only.)
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer PushC { float4x4 faceViewProj; float4x4 model; float4 probeCentre; };
#define HF_CENTRE probeCentre.xyz
#else
[[vk::push_constant]] struct { float4x4 faceViewProj; float4x4 model; float4 probeCentre; } pc;
#define HF_CENTRE pc.probeCentre.xyz
#endif

float4 main(PSInput i) : SV_Target {
    // Linear world distance from the probe centre to this surface (length == sqrt(dot(.,.))).
    float d = length(i.wpos - HF_CENTRE);
    // The two moments at single-sample granularity. d*d is a BARE MULTIPLY (one rounding) — NOT mad/fma —
    // so it matches the CPU reference probedist::MomentsFromDistance(d).m[1] = d*d (see the banner). NOTE:
    // the RGBA16F target stores R = fp16(d) and G = fp16(d*d); the HOST derives the moment store from the
    // read-back R via MomentsFromDistance, so the GPU==CPU bit-exact proof is on that moment-from-distance
    // step over the SAME distance bytes (the fp16-store rounding of G is not on the proof path).
    float dd = d * d;
    return float4(d, dd, 0.0, 0.0);
}
