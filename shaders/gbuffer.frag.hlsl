// SSAO normal+depth prepass — fragment shader (Slice Y). Writes the VIEW-SPACE normal (xyz) and the
// VIEW-SPACE LINEAR DEPTH (w = -vpos.z, positive and increasing with distance in front of the camera;
// RH view space has -Z forward) into an RGBA16F g-buffer. The SSAO pass reconstructs each pixel's
// view-space position from this linear depth + the screen UV + the projection params, and reads the
// normal here to orient the hemisphere kernel. Background (where no geometry draws) is the cleared
// value (w == 0), which the SSAO pass treats as "no surface" and skips.
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 vnormal : NORMAL;
    [[vk::location(1)]] float3 vpos    : POSITION0;
};

float4 main(PSInput i) : SV_Target {
    float3 N = normalize(i.vnormal);
    float  linDepth = -i.vpos.z;   // positive in front of the camera
    return float4(N, linDepth);
}
