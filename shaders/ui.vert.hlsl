// ImGui UI vertex shader. Maps ImDrawVert (pos: float2, uv: float2, col: RGBA8) to clip space via
// a screen-space ortho transform passed as a vertex push constant {scale.xy, translate.xy}.
// Vertex color arrives as a normalized RGBA8_UNorm attribute (0..1) and is passed straight through.
struct PushConstants { float2 scale; float2 translate; };
[[vk::push_constant]] PushConstants pc;

struct VSInput {
    [[vk::location(0)]] float2 pos : POSITION;
    [[vk::location(1)]] float2 uv  : TEXCOORD0;
    [[vk::location(2)]] float4 col : COLOR0;   // RGBA8_UNorm -> normalized 0..1
};
struct VSOutput {
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 col : COLOR0;
    [[vk::location(1)]] float2 uv  : TEXCOORD0;
};

VSOutput main(VSInput i) {
    VSOutput o;
    o.pos = float4(i.pos * pc.scale + pc.translate, 0.0, 1.0);
    o.col = i.col;
    o.uv  = i.uv;
    return o;
}
