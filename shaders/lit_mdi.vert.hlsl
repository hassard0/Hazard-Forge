// Slice BM — multi-draw-indirect lit vertex shader. VULKAN-ONLY (uses the SPIR-V DrawIndex builtin).
// One vkCmdDrawIndexedIndirect(drawCount=N) issues N indexed draws from a packed indirect-args buffer;
// this shader reads the per-draw model matrix + material from a STORAGE buffer PerDraw[ ] indexed by
// gl_DrawID (the draw's index within the multi-draw, exposed via [[vk::builtin("DrawIndex")]]).
//
// This is the GPU-driven equivalent of lit_instanced.vert (which reads the model from a per-instance
// vertex stream) and lit.vert (which reads it from a push constant). The per-object REFERENCE render
// pushes the SAME PerDraw[i] values via the push-constant lit.vert, so the MDI image is BYTE-IDENTICAL
// to the per-object image (the render-invariance proof in --mdi-shot). The fragment stage is the
// SHARED, UNCHANGED lit.frag.hlsl, so VSOutput here matches lit.vert's VSOutput exactly.
//
// NOT in the Metal MSL-gen list: glslc/spirv-cross cannot lower the DrawIndex builtin, and the Metal
// golden renders the identical scene via its per-object path anyway (the image is backend-identical).
struct VSInput {
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 color   : COLOR;
    [[vk::location(2)]] float2 uv      : TEXCOORD0;
    [[vk::location(3)]] float3 normal  : NORMAL;
    [[vk::location(4)]] float3 tangent : TANGENT;
};
struct VSOutput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    [[vk::location(4)]] nointerpolation float2 material : TEXCOORD1; // x=metallic, y=roughness
    [[vk::location(5)]] float3 wtangent : TANGENT;
};
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};

// Per-draw record (matches render::mdi::PerDraw byte layout exactly: column-major mat4 + float4
// material = 80 bytes, std430). HLSL float4x4 is row-major by default; we read the four COLUMNS into a
// column-major model (same convention as lit_instanced.vert's column reassembly) so mul(model, v)
// matches the push-constant per-object path bit-for-bit.
struct PerDraw {
    float4 modelCol0;  // column 0
    float4 modelCol1;  // column 1
    float4 modelCol2;  // column 2
    float4 modelCol3;  // column 3
    float4 material;   // x=metallic, y=roughness
};

// Frame UBO at set 0 (binding 0), exactly like lit.vert's Vulkan path. The per-draw SSBO at set 2
// (binding 0) — a dedicated VERTEX-stage storage set bound via BindPerDrawData. The shared lit.frag
// samples its base/normal material at set 1 (BindMaterial), untouched by this vertex shader.
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(0, 2)]] StructuredBuffer<PerDraw> gPerDraw;

VSOutput main(VSInput i, [[vk::builtin("DrawIndex")]] uint drawId : DRAWID) {
    VSOutput o;
    PerDraw pd = gPerDraw[drawId];
    // Reassemble the column-major model matrix from the four per-draw columns (HLSL float4x4(rows...)
    // takes ROWS, so row r is the r-th component of each column). Identical to lit_instanced.vert.
    float4x4 model = float4x4(
        float4(pd.modelCol0.x, pd.modelCol1.x, pd.modelCol2.x, pd.modelCol3.x),
        float4(pd.modelCol0.y, pd.modelCol1.y, pd.modelCol2.y, pd.modelCol3.y),
        float4(pd.modelCol0.z, pd.modelCol1.z, pd.modelCol2.z, pd.modelCol3.z),
        float4(pd.modelCol0.w, pd.modelCol1.w, pd.modelCol2.w, pd.modelCol3.w));

    float4 world = mul(model, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(f.viewProj, world);
    o.wnormal = normalize(mul((float3x3)model, i.normal));
    o.wtangent = mul((float3x3)model, i.tangent);
    o.color = i.color; o.uv = i.uv;
    o.material = pd.material.xy;
    return o;
}
