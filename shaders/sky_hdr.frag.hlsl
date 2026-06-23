// HDR equirectangular skybox (Slice R). Like sky.frag.hlsl, reconstructs a world-space view ray
// from the camera basis in the per-frame UBO, but samples a real HDR environment map (an equirect
// 2D texture on the dedicated environment set, set 3) at LOD 0 for the background instead of the
// procedural gradient. Outputs LINEAR HDR radiance — the post pass does exposure + ACES, so this
// must NOT pre-tonemap. Kept as a SEPARATE shader so the golden-locked sky.frag is undisturbed.
#include "frame_data.hlsli"
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// HDR environment map: dedicated set 3 (binding 11 = image, 12 = sampler). The binding numbers are
// chosen so spirv-cross --msl-decoration-binding lands them on Metal texture(11)/sampler(12).
[[vk::binding(11, 3)]] Texture2D    gEnv    : register(t11);
[[vk::binding(12, 3)]] SamplerState gEnvSmp : register(s11);
struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

static const float HF_PI = 3.14159265358979323846;

// Equirectangular projection: world direction -> equirect UV. u from longitude atan2(z,x), v from
// latitude acos(y). u wraps in [0,1) (sampler repeats), v clamps at the poles.
float2 EquirectUV(float3 dir) {
    float3 d = normalize(dir);
    float u = atan2(d.z, d.x) / (2.0 * HF_PI) + 0.5;
    float v = acos(clamp(d.y, -1.0, 1.0)) / HF_PI;
    return float2(u, v);
}

float4 main(PSInput i) : SV_Target {
    float2 ndc = i.uv * 2.0 - 1.0;
    float3 ray = normalize(f.camFwd.xyz
                 + f.camRight.xyz * ndc.x * f.skyParams.x * f.skyParams.y
                 + f.camUp.xyz    * (-ndc.y) * f.skyParams.x);

    // Sample the HDR sky at the sharpest mip (LOD 0). Linear HDR radiance straight out.
    float3 sky = gEnv.SampleLevel(gEnvSmp, EquirectUV(ray), 0.0).rgb;
    return float4(sky, 1.0);
}
