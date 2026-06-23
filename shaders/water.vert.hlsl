// Water surface vertex shader (Slice CF). Displaces a flat NxN grid by the summed Gerstner waves at a
// FIXED time (water::Displace, engine/render/water.h, mirrored below) so the rasterized surface
// ripples, and passes the WORLD position + the grid (x,z) params + the wave time to the fragment stage
// so the fragment recomputes the analytic wave normal at the displaced point. Clip position uses the
// shared FrameData.viewProj (so the surface rasterizes consistently with the lit pass). The push
// constant carries the model matrix + the water params (shared with the fragment stage via
// fragmentPushConstants). Mirrors gbuffer.vert's HF_MSL_GEN binding convention.
struct VSInput {
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 color   : COLOR;
    [[vk::location(2)]] float2 uv      : TEXCOORD0;
    [[vk::location(3)]] float3 normal  : NORMAL;
    [[vk::location(4)]] float3 tangent : TANGENT;
};
struct VSOutput {
    float4 clip   : SV_Position;
    [[vk::location(0)]] float3 wpos : POSITION0;   // displaced world position
    [[vk::location(1)]] float2 grid : TEXCOORD0;   // base grid (x,z) BEFORE displacement (for normals)
};
#include "frame_data.hlsli"

// Water params (SHARED by the vertex + fragment stage; fragmentPushConstants=true). Layout matches the
// CPU WaterParams struct in main.cpp byte-for-byte. The 3-wave set is passed as 3*float4
// (dir.xy, amplitude, wavelength) + 3*float4 (steepness, speed, _, _) so the GPU recomputes the EXACT
// same field the CPU water.h uses.
struct WaterParams {
    float4x4 model;     // grid model matrix (places + scales the water plane at the water level)
    float4 waveA[3];    // per wave: dir.x, dir.y, amplitude, wavelength
    float4 waveB[3];    // per wave: steepness, speed, _, _
    float4 cfg0;        // x=time, y=tanHalfFovY, z=aspect, w=waterLevel
    float4 cfg1;        // x=fresnelF0, y=absorption, z=numWaves, w=_
    float4 shallow;     // shallow water tint (rgb)
    float4 deep;        // deep water tint (rgb)
    float4 texel;       // x=1/w, y=1/h
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { WaterParams wp; };
#define HF_WP wp
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { WaterParams p; } pc;
#define HF_WP pc.p
#endif

static const float HF_W_PI = 3.14159265358979323846;

// Summed Gerstner displacement (P - basePoint) — IDENTICAL math to water::Displace in water.h.
float3 WaterDisplace(float x, float z, float t) {
    int n = (int)HF_WP.cfg1.z;
    float3 d = float3(0.0, 0.0, 0.0);
    [loop] for (int i = 0; i < n; ++i) {
        float2 dir = HF_WP.waveA[i].xy;
        float  amp = HF_WP.waveA[i].z;
        float  wl  = HF_WP.waveA[i].w;
        float  q   = HF_WP.waveB[i].x;
        float  spd = HF_WP.waveB[i].y;
        float  k   = (wl > 1e-6) ? (2.0 * HF_W_PI / wl) : 0.0;
        float  w   = spd * k;
        float  theta = k * (dir.x * x + dir.y * z) - w * t;
        float  qa = q * amp;
        d += float3(qa * dir.x * cos(theta), amp * sin(theta), qa * dir.y * cos(theta));
    }
    return d;
}

VSOutput main(VSInput i) {
    VSOutput o;
    // The base grid vertex in MODEL space is (i.pos.x, 0, i.pos.z) on the flat plane; transform to
    // world to get the grid (x,z) the wave field is sampled at, then displace IN WORLD SPACE so the
    // wave wavelengths are world-consistent (independent of the grid's model scale).
    float4 baseWorld = mul(HF_WP.model, float4(i.pos.x, 0.0, i.pos.z, 1.0));
    float gx = baseWorld.x;
    float gz = baseWorld.z;
    float t  = HF_WP.cfg0.x;
    float3 disp = WaterDisplace(gx, gz, t);
    float3 world = float3(gx + disp.x, HF_WP.cfg0.w + disp.y, gz + disp.z);
    o.clip = mul(f.viewProj, float4(world, 1.0));
    o.wpos = world;
    o.grid = float2(gx, gz);
    return o;
}
