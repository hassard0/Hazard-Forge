// Hazard Forge — ANIMATED procedural sky (the issue #5 time-channel DELIVERABLE).
//
// A near-twin of sky.frag.hlsl: it reconstructs the same world-space view ray and starts from the
// SHARED HFSkyColor base (procedural_sky.hlsli, issue #4), then ADDS a time-animated cloud band that
// drifts across the sky. The whole point is to demonstrate how a sample author reads the per-frame
// TIME channel that issue #5 wired into FrameData.skyParams.z — there was previously NOWHERE in the
// per-frame uniforms to read time, so animated sky/water/foliage/VFX shaders were stuck static.
//
//   float time = f.skyParams.z;   // <-- seconds since the sample started (z=time, w=frameIndex)
//
// The CPU fills skyParams.z each frame from runtime::FixedTimestep's accumulated time (deterministic:
// same inputs -> same time -> bit-exact golden). The --sky-animated-shot golden is captured at a
// FIXED time (2.0s) so it is reproducible; an interactive run advances time so the clouds drift.
#include "procedural_sky.hlsli"
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    // skyParams: x=tanHalfFov, y=aspect, z=time(seconds), w=frameIndex (issue #5 time channel).
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// Cheap 2D value-noise (hash-lerp), enough for a drifting cloud band.
float hash21(float2 p) {
    p = frac(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return frac(p.x * p.y);
}
float noise2(float2 p) {
    float2 i = floor(p);
    float2 fp = frac(p);
    float2 u = fp * fp * (3.0 - 2.0 * fp);
    float a = hash21(i + float2(0.0, 0.0));
    float b = hash21(i + float2(1.0, 0.0));
    float c = hash21(i + float2(0.0, 1.0));
    float d = hash21(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float4 main(PSInput i) : SV_Target {
    float2 ndc = i.uv * 2.0 - 1.0;
    float3 ray = f.camFwd.xyz
                 + f.camRight.xyz * ndc.x * f.skyParams.x * f.skyParams.y
                 + f.camUp.xyz    * (-ndc.y) * f.skyParams.x;

    // Shared procedural sky base (gradient + ground haze + sun glow).
    float3 sky = HFSkyColor(ray, f.lightDir.xyz);

    // --- THE TIME CHANNEL (issue #5): drift a cloud band across the upper sky. ---
    float  time = f.skyParams.z;                 // seconds since start (deterministic)
    float3 d    = normalize(ray);
    if (d.y > 0.0) {
        // Project the view ray onto a plane and PAN the noise field by time (this is what makes the
        // band animate: the noise lookup is offset by time, so a different time -> a different sky).
        float2 sp = d.xz / max(d.y, 0.06) * 0.8 + float2(time * 0.20, time * 0.07);
        float n  = noise2(sp) * 0.6 + noise2(sp * 2.3) * 0.3 + noise2(sp * 5.1) * 0.1;
        float cloud = saturate((n - 0.30) * 3.2) * saturate(d.y * 2.5);
        sky = lerp(sky, float3(1.0, 1.0, 1.0), cloud);
    }

    return float4(sky, 1.0);
}
