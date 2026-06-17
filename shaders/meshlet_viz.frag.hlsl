// Slice DS — virtual-geometry meshlet viz fragment shader. A trivial flat-color pass: it outputs the
// per-cluster color the vertex shader computed (the pushed hashColor(meshletIndex) * a fixed Lambert).
// No lights/shadows/textures/RNG — the cluster color is decided entirely CPU-side, so this fragment is
// bit-exact across Vulkan and Metal by construction (no GPU-side hash, no per-backend math).
struct PSInput {
    float4 clip  : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};
float4 main(PSInput i) : SV_Target { return float4(i.color, 1.0); }
