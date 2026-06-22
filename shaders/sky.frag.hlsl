// Procedural sky: gradient horizon->zenith + a sun glow. Reconstructs a world-space view ray
// from the camera basis in the per-frame UBO (no matrix inverse). Set 0 b0 = FrameData.
//
// The gradient/haze/sun math lives in the SHARED procedural_sky.hlsli (HFSkyColor) so the sky pass and
// the lit pass's IBL reflection stay in lock-step (issue #4) — see lit.frag.hlsl / pbr_core.hlsli.
#include "procedural_sky.hlsli"
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    // skyParams: x=tanHalfFov, y=aspect, z=time(seconds), w=frameIndex (issue #5; the time channel an
    // animated sky/water/foliage shader reads — see shaders/sky_animated.frag.hlsl).
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

float4 main(PSInput i) : SV_Target {
    float2 ndc = i.uv * 2.0 - 1.0;
    // Un-normalized world-space view ray from the camera basis; HFSkyColor normalizes it ONCE
    // internally (don't pre-normalize here — a double normalize would perturb the golden bits).
    float3 ray = f.camFwd.xyz
                 + f.camRight.xyz * ndc.x * f.skyParams.x * f.skyParams.y
                 + f.camUp.xyz    * (-ndc.y) * f.skyParams.x;

    // Shared procedural sky (gradient + ground haze + sun glow) — single source of truth.
    float3 sky = HFSkyColor(ray, f.lightDir.xyz);

    return float4(sky, 1.0);
}
