// Slice CB — fully-GPU-driven lit vertex shader (MDI + bindless capstone). VULKAN-ONLY (uses the SPIR-V
// DrawIndex builtin). COMPOSES the two proven slices:
//   * BM (lit_mdi.vert): ONE vkCmdDrawIndexedIndirect(drawCount=N) issues N indexed draws; this shader
//     reads the per-draw model matrix + material from a STORAGE buffer PerDraw[ ] indexed by gl_DrawID
//     ([[vk::builtin("DrawIndex")]]).
//   * BZ (lit_bindless.vert): the per-draw record ALSO carries a `texIndex` (the object's slot in the
//     bindless texture array); this shader passes it FLAT to lit_gpudriven.frag, which samples
//     gTextures[NonUniformResourceIndex(texIndex)].
//
// The PER-OBJECT BOUND reference render pushes the SAME model+material via the push-constant lit.vert and
// binds the SAME texture (the one this texIndex selects) — so the transformed/shaded geometry AND the
// sampled texel are identical, and the GPU-driven image is BYTE-IDENTICAL to the per-object bound image
// (the render-invariance proof in --gpudriven-shot). DXC must emit BOTH DrawParameters (gl_DrawID) AND
// SPV_EXT_descriptor_indexing (the NonUniform sample is in the fragment) — confirmed in the .spv.
//
// NOT in the Metal MSL-gen list: glslc/spirv-cross cannot lower the DrawIndex builtin; the Metal golden
// renders the identical scene via its per-object bound path (the image is backend-identical).
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
    // interpolated). lit_gpudriven.frag reads it to select gTextures[NonUniformResourceIndex(texIndex)].
    [[vk::location(6)]] nointerpolation uint texIndex : TEXCOORD2;
};
#include "frame_data.hlsli"

// Per-draw record (matches render::gpudriven::GpuDrivenPerDraw byte layout exactly: column-major mat4 +
// float4 material + uint texIndex + 3x uint pad = 96 bytes, std430). HLSL float4x4 is row-major by
// default; we read the four COLUMNS into a column-major model (same convention as lit_mdi.vert) so
// mul(model, v) matches the push-constant per-object path bit-for-bit.
struct PerDraw {
    float4 modelCol0;  // column 0
    float4 modelCol1;  // column 1
    float4 modelCol2;  // column 2
    float4 modelCol3;  // column 3
    float4 material;   // x=metallic, y=roughness
    uint   texIndex;   // bindless array index
    uint3  pad;        // pad to 96 bytes (16-byte multiple)
};

// Frame UBO at set 0 (binding 0), like lit_mdi.vert. The per-draw SSBO at set 2 (binding 0) — the
// dedicated VERTEX-stage storage set bound via BindPerDrawData. The shared lit/bindless frag samples its
// normal map at set 1 + the bindless array at set 4 (untouched by this vertex shader).
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(0, 2)]] StructuredBuffer<PerDraw> gPerDraw;

VSOutput main(VSInput i, [[vk::builtin("DrawIndex")]] uint drawId : DRAWID) {
    VSOutput o;
    PerDraw pd = gPerDraw[drawId];
    // Reassemble the column-major model matrix from the four per-draw columns (HLSL float4x4(rows...)
    // takes ROWS, so row r is the r-th component of each column). Identical to lit_mdi.vert.
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
    o.texIndex = pd.texIndex;   // pass the bindless index flat to the fragment (BZ)
    return o;
}
