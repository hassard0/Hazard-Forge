// Slice BA — Text / HUD vertex shader. The HUD text quads are laid out CPU-side (engine/ui/text.cpp
// LayoutText) directly in NDC (pixel positions already converted via screenW/H), so there is NO
// transform here: pass the NDC position and the atlas UV straight through. Pairs with an
// alphaBlend + cullNone, no-depth pipeline (the screen-space overlay drawn over the scene after
// post). Matches the ui.vert.hlsl screen-space convention but needs no ortho push constant because
// the layout already emitted NDC.
struct VSInput {
    [[vk::location(0)]] float2 pos : POSITION;   // NDC x,y in [-1,1]
    [[vk::location(1)]] float2 uv  : TEXCOORD0;  // atlas UV in [0,1]
};
struct VSOutput {
    float4 pos : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

VSOutput main(VSInput i) {
    VSOutput o;
    o.pos = float4(i.pos, 0.0, 1.0);
    o.uv  = i.uv;
    return o;
}
