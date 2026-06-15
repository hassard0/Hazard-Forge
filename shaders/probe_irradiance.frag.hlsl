// Slice AK — probe IRRADIANCE convolution. A fullscreen pass run once PER irradiance tile (via
// SetViewport) that reconstructs the output world direction N for each output texel, then cosine-
// convolves the REFLECTION block of the probe atlas (a coarse diffuse irradiance) around N. The
// result is written into the irradiance block of the FINAL atlas; reading the reflection atlas and
// writing the (separate) final atlas avoids read-after-write hazards.
//
// Output direction reconstruction is the EXACT inverse of the reflection bake's faceVP projection:
// for output uv, build the face-clip point and transform by invFaceVP (pushed per tile), then
// subtract the probe position -> the world direction this irradiance texel represents. That direction
// is then re-projected (the lit pass does the same with faceVP) so what we WRITE here lines up with
// what lit_probe READS.
//
// FrameData (set 0): supplies faceVP[6] + atlas params so we can PROJECT sample directions into the
// reflection block. Reflection atlas bound at set 1 (gRefl 0/1). Per-tile push constant: invFaceVP +
// probePos.
struct FrameData {
    float4x4 viewProj;
    float4   lightDir;
    float4   lightColor;
    float4   viewPos;
    float4x4 faceVP[6];
    float4   probePos;
    float4   atlasParams;   // x=reflTileU y=reflTileV (z/w irr — unused here)
    float4   atlasParams2;  // x=irrBlockV0 y=texelU z=texelV w=tilesPerRow
    float4   camFwd;
    float4   camRight;
    float4   camUp;
    float4   skyParams;
    float4   pad0;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Reflection atlas (the bake's reflection block) at the material set.
[[vk::binding(0, 1)]] Texture2D    gRefl    : register(t0);
[[vk::binding(1, 1)]] SamplerState gReflSmp : register(s0);

struct IrrParams { float4x4 invFaceVP; float4 probePos; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer IrrPC { IrrParams ip; };
#define HF_IP ip
#else
[[vk::push_constant]] struct { IrrParams p; } pc;
#define HF_IP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

int SelectFace(float3 dir) {
    float ax = abs(dir.x), ay = abs(dir.y), az = abs(dir.z);
    if (ax >= ay && ax >= az) return dir.x >= 0.0 ? 0 : 1;
    if (ay >= ax && ay >= az) return dir.y >= 0.0 ? 2 : 3;
    return dir.z >= 0.0 ? 4 : 5;
}

// Sample the REFLECTION block of the atlas for a world direction (mirror of lit_probe's block-0 path).
float3 SampleRefl(float3 dir) {
    int face = SelectFace(dir);
    float4 cp = mul(f.faceVP[face], float4(f.probePos.xyz + normalize(dir), 1.0));
    if (cp.w <= 0.0) return float3(0, 0, 0);
    float2 faceUV = (cp.xy / cp.w) * 0.5 + 0.5;
#ifdef HF_MSL_GEN
    faceUV.y = 1.0 - faceUV.y;
#endif
    faceUV = saturate(faceUV);
    int tilesPerRow = (int)f.atlasParams2.w;
    int col = face % tilesPerRow, row = face / tilesPerRow;
    float2 tileSize = f.atlasParams.xy;
    float2 tileOrigin = float2((float)col, (float)row) * tileSize;
    float2 texel = f.atlasParams2.yz;
    float2 atlasUV = clamp(tileOrigin + faceUV * tileSize,
                           tileOrigin + texel, tileOrigin + tileSize - texel);
    return gRefl.SampleLevel(gReflSmp, atlasUV, 0.0).rgb;
}

float4 main(PSInput i) : SV_Target {
    // Reconstruct the output world direction N as the inverse of this tile's faceVP projection.
    float2 ndc = i.uv * 2.0 - 1.0;
#ifdef HF_MSL_GEN
    ndc.y = -ndc.y;   // undo the post.vert Metal V-flip so N matches the Vulkan reconstruction
#endif
    float4 wp = mul(HF_IP.invFaceVP, float4(ndc, 0.5, 1.0));
    float3 N = normalize(wp.xyz / wp.w - HF_IP.probePos.xyz);

    // Build a tangent basis around N and cosine-weight a hemisphere of taps from the reflection block.
    float3 up = abs(N.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);

    const float HF_PI = 3.14159265358979323846;
    float3 sum = float3(0, 0, 0);
    float  wsum = 0.0;
    // Coarse cosine-hemisphere convolution: a few rings of azimuth x elevation, weighted by cos.
    const int kPhi = 8;     // azimuth steps
    const int kThe = 4;     // elevation rings
    for (int t = 0; t < kThe; ++t) {
        float theta = (((float)t + 0.5) / (float)kThe) * (0.5 * HF_PI);  // 0..pi/2
        float cosT = cos(theta), sinT = sin(theta);
        for (int p = 0; p < kPhi; ++p) {
            float phi = (((float)p) / (float)kPhi) * 2.0 * HF_PI;
            float3 dirL = float3(sinT * cos(phi), sinT * sin(phi), cosT);
            float3 dirW = normalize(dirL.x * T + dirL.y * B + dirL.z * N);
            float w = cosT * sinT;  // cosine weight x solid-angle (sinT) factor
            sum += SampleRefl(dirW) * w;
            wsum += w;
        }
    }
    float3 irradiance = sum / max(wsum, 1e-4);
    return float4(irradiance, 1.0);
}
