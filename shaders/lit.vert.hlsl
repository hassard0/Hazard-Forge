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
};
#include "frame_data.hlsli"
// HF_MSL_GEN: the glslang HLSL frontend (glslc/glslangValidator, used on macOS where DXC is
// unavailable) does NOT honour [[vk::push_constant]] — it lowers it to a plain Uniform buffer at
// (set 0, binding 0), colliding with the Frame cbuffer. So for the MSL-generation path only we
// declare the model matrix as an explicit cbuffer with distinct bindings chosen so that
// `spirv-cross --msl-decoration-binding` emits exactly the engine's flat Metal buffer indices
// (vertex: buffer0 = vertex stream, buffer1 = FrameData, buffer2 = model). The Vulkan/DXC path is
// untouched: it keeps the real push constant and the original [[vk::binding(0,0)]] Frame.
// The push constant now carries the per-draw material (material.x=metallic, material.y=roughness)
// alongside the model matrix: { float4x4 model; float4 material; } = 80 bytes. BOTH the real
// push_constant (DXC/Vulkan) and the HF_MSL_GEN cbuffer (glslc->spirv-cross->Metal) are extended
// identically so the single HLSL source feeds both backends.
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
// UNTINTED sentinel -> white (every existing draw sets material.z=0, so all goldens are unchanged).
// The tint rides on o.color, which every lit fragment shader already multiplies into albedo, so this
// one vertex-shader line gives per-draw tinting across all lit pipelines with no fragment changes.
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
    return o;
}
