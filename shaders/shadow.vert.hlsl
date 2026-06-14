// Depth-only shadow pass: transform geometry into the light's clip space. Only POSITION is used,
// but the full scene vertex layout is declared so it binds against the same vertex buffers.
struct VSInput {
    [[vk::location(0)]] float3 pos    : POSITION;
    [[vk::location(1)]] float3 color  : COLOR;
    [[vk::location(2)]] float2 uv     : TEXCOORD0;
    [[vk::location(3)]] float3 normal : NORMAL;
};
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; } pc;

float4 main(VSInput i) : SV_Position {
    float4 world = mul(pc.model, float4(i.pos, 1.0));
    return mul(f.lightViewProj, world);
}
