// Procedural sky: gradient horizon->zenith + a sun glow. Reconstructs a world-space view ray
// from the camera basis in the per-frame UBO (no matrix inverse). Set 0 b0 = FrameData.
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams; // skyParams.x=tanHalfFov, .y=aspect
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

float4 main(PSInput i) : SV_Target {
    float2 ndc = i.uv * 2.0 - 1.0;
    float3 ray = normalize(f.camFwd.xyz
                 + f.camRight.xyz * ndc.x * f.skyParams.x * f.skyParams.y
                 + f.camUp.xyz    * (-ndc.y) * f.skyParams.x);

    // Horizon -> zenith gradient.
    float  h       = saturate(ray.y * 0.5 + 0.5);
    float3 zenith  = float3(0.18, 0.30, 0.62);
    float3 horizon = float3(0.65, 0.72, 0.82);
    float3 sky     = lerp(horizon, zenith, pow(h, 0.8));

    // Dim ground haze for the lower hemisphere.
    float3 ground = float3(0.12, 0.11, 0.10);
    if (ray.y < 0.0) {
        float g = saturate(-ray.y * 2.0);
        sky = lerp(sky, ground, g);
    }

    // Sun glow toward the (incoming) directional light direction.
    float3 sunDir = normalize(-f.lightDir.xyz);
    float  s = pow(max(dot(ray, sunDir), 0.0), 256.0);
    sky += float3(1.0, 0.95, 0.8) * s * 2.0;

    return float4(sky, 1.0);
}
