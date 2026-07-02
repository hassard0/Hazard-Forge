// Alpha-mask (cutout) vertex shader — Slice SC1b. A VERBATIM copy of lit.vert.hlsl plus ONE extra
// nointerpolation varying (location 6): the per-draw alpha-test cutoff, read from the push-constant
// material.w per the #38 packing convention {x=metallic, y=roughness, z=albedo tint, w=alpha cutoff}.
// Kept as a SEPARATE shader (paired with lit_pbr_cutout.frag) so the golden-locked lit.vert and every
// pipeline built on it are byte-for-byte undisturbed — only draws that explicitly select the cutout
// pipeline (glTF alphaMode MASK, or BLEND approximated as MASK@0.5) go through here.
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
    // Per-draw material, constant across the primitive. nointerpolation keeps it exact
    // (no perspective division) since metallic/roughness come straight from the push constant.
    [[vk::location(4)]] nointerpolation float2 material : TEXCOORD1; // x=metallic, y=roughness
    // World-space tangent (location 5); the fragment shader builds the TBN basis from this + the
    // interpolated world normal (B = cross(N,T)). Interpolated like the normal.
    [[vk::location(5)]] float3 wtangent : TANGENT;
    // SC1b: the alpha-test cutoff (push-constant material.w). 0 == never discards (sample alpha is
    // always >= 0), so an accidental opaque draw through this pipeline still renders correctly.
    [[vk::location(6)]] nointerpolation float cutoff : TEXCOORD2;
};
#include "frame_data.hlsli"
// HF_MSL_GEN: same convention as lit.vert.hlsl — glslang ignores [[vk::push_constant]], so the MSL
// generation path declares the model+material as an explicit cbuffer at the bindings that
// spirv-cross --msl-decoration-binding maps onto the engine's flat Metal buffer indices
// (vertex: buffer0 = vertex stream, buffer1 = FrameData, buffer2 = push-constant block).
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 model; float4 material; };
#define HF_MODEL model
#define HF_MATERIAL material
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; float4 material; } pc;
#define HF_MODEL pc.model
#define HF_MATERIAL pc.material
#endif

// Per-draw albedo tint (issue #38): material.z packs an RGB multiplier so a sample can recolor a mesh
// ("make the car red") WITHOUT swapping its texture material. Packing: z = floor(r*255)*65536 +
// floor(g*255)*256 + floor(b*255), each channel 0-255 — exact in a float (max 2^24-1). z == 0 is the
// UNTINTED sentinel -> white. Same helper as lit.vert.hlsl so tinting works on cutout draws too.
float3 HfUnpackTint(float packed) {
    if (packed <= 0.0) return float3(1.0, 1.0, 1.0);
    float r = floor(packed / 65536.0);
    float g = floor((packed - r * 65536.0) / 256.0);
    float b = packed - r * 65536.0 - g * 256.0;
    return float3(r, g, b) / 255.0;
}

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(f.viewProj, world);
    // (float3x3)model is correct for rotation + uniform scale only. Non-uniform scale needs the
    // inverse-transpose normal matrix (pass it separately when scaled geometry is introduced).
    o.wnormal = normalize(mul((float3x3)HF_MODEL, i.normal));
    // World-space tangent (rotation + uniform scale only, same caveat as the normal). Not yet
    // re-orthonormalized; the fragment shader Gram-Schmidts it against the interpolated normal.
    o.wtangent = mul((float3x3)HF_MODEL, i.tangent);
    o.color = i.color * HfUnpackTint(HF_MATERIAL.z); o.uv = i.uv;  // issue #38: per-draw albedo tint
    o.material = HF_MATERIAL.xy;  // pass metallic+roughness through to the fragment
    o.cutoff = HF_MATERIAL.w;     // SC1b: alpha-test cutoff (0 = no test)
    return o;
}
