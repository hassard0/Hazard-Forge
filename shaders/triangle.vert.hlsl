struct VSInput {
    [[vk::location(0)]] float2 pos   : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
};

struct VSOutput {
    float4 pos   : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};

VSOutput main(VSInput input) {
    VSOutput o;
    o.pos = float4(input.pos, 0.0, 1.0);
    o.color = input.color;
    return o;
}
