struct VSInput {
    [[vk::location(0)]] float3 pos   : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
    [[vk::location(2)]] float2 uv    : TEXCOORD0;
};
struct VSOutput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
    [[vk::location(1)]] float2 uv    : TEXCOORD0;
};
[[vk::push_constant]] struct { float4x4 mvp; } pc;

VSOutput main(VSInput i) {
    VSOutput o;
    o.pos = mul(pc.mvp, float4(i.pos, 1.0));
    o.color = i.color;
    o.uv = i.uv;
    return o;
}
