struct VSInput {
    [[vk::location(0)]] float3 pos    : POSITION;
    [[vk::location(1)]] float3 color  : COLOR;
    [[vk::location(2)]] float2 uv     : TEXCOORD0;
    [[vk::location(3)]] float3 normal : NORMAL;
};
struct VSOutput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
};
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3];
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; } pc;

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(pc.model, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(f.viewProj, world);
    // (float3x3)model is correct for rotation + uniform scale only. Non-uniform scale needs the
    // inverse-transpose normal matrix (pass it separately when scaled geometry is introduced).
    o.wnormal = normalize(mul((float3x3)pc.model, i.normal));
    o.color = i.color; o.uv = i.uv;
    return o;
}
