// Slice BZ — bindless-textures lit vertex shader. VULKAN-ONLY. Byte-for-byte identical to lit.vert,
// EXCEPT the per-draw push constant carries an extra `uint texIndex` (the material's index into the
// bindless texture array) which is passed FLAT to the fragment so lit_bindless.frag can sample
// gTextures[NonUniformResourceIndex(texIndex)]. The model-matrix + material handling is unchanged
// from lit.vert, so the transformed/shaded geometry is identical to the bound reference path; only the
// base-color TEXTURE SOURCE differs (an array index vs. a per-material bound set) — and since both
// sample the SAME texel of the SAME texture, the bindless image is byte-identical to the bound image
// (the render-invariance proof in --bindless-shot).
//
// Not in the Metal MSL-gen list: the Metal golden renders the identical scene via its per-material
// bound path (the image is backend-identical), so this Vulkan-only variant needs no MSL lowering.
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
    // The bindless texture index, flat across the primitive (an integer material index — must NOT be
    // interpolated). lit_bindless.frag reads it to select gTextures[NonUniformResourceIndex(texIndex)].
    [[vk::location(6)]] nointerpolation uint texIndex : TEXCOORD2;
};
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Per-draw push constant: the lit.vert {model mat4, float4 material} PLUS a uint texIndex (the bindless
// array index for this draw's base-color texture). 84 bytes; pad to 16-byte multiple is unnecessary for
// a single trailing scalar.
[[vk::push_constant]] struct { float4x4 model; float4 material; uint texIndex; } pc;

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(pc.model, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(f.viewProj, world);
    o.wnormal = normalize(mul((float3x3)pc.model, i.normal));
    o.wtangent = mul((float3x3)pc.model, i.tangent);
    o.color = i.color; o.uv = i.uv;
    o.material = pc.material.xy;
    o.texIndex = pc.texIndex;
    return o;
}
