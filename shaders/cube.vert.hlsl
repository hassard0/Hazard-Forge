struct VSInput {
    [[vk::location(0)]] float3 pos   : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
};
struct VSOutput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};
[[vk::push_constant]] struct { float4x4 mvp; } pc;

VSOutput main(VSInput input) {
    VSOutput o;
    o.pos = mul(pc.mvp, float4(input.pos, 1.0));
    o.color = input.color;
    return o;
}
