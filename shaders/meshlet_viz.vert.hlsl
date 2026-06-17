// Slice DS — virtual-geometry meshlet viz vertex shader. Transforms the mesh vertex by the per-draw
// model matrix + the shared FrameData viewProj, and forwards a per-CLUSTER flat color (pushed CPU-side
// from hashColor(meshletIndex) in engine/render/meshlet.h) modulated by a FIXED-direction Lambert term
// for shape readability. Fully deterministic: no lights/shadows/RNG/clock — the only inputs are the
// vertex, the model matrix, viewProj, and the pushed color. The fragment shader is a trivial pass-
// through of the computed color, so there is NO GPU-side hash to keep bit-exact across backends.
// Only pos (location 0) + normal (location 3) are consumed (the color is a per-draw push constant, not
// a vertex attribute). The showcase binds a 2-attribute vertex layout (pos@0, normal@32) over the same
// stride-56 scene::Vertex buffer so EVERY declared attribute is consumed — no "attribute not consumed"
// validation warnings (born validation-clean).
struct VSInput {
    [[vk::location(0)]] float3 pos    : POSITION;
    [[vk::location(3)]] float3 normal : NORMAL;
};
struct VSOutput {
    float4 clip : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;  // per-cluster flat color * fixed Lambert
};
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
// Same HF_MSL_GEN split as lit.vert: glslang (macOS, no DXC) lowers [[vk::push_constant]] to a plain
// Uniform at (set 0, binding 0), colliding with the Frame cbuffer — so for the MSL-gen path the push
// constant is an explicit cbuffer at distinct bindings (Frame=binding1, PushC=binding2) that
// spirv-cross maps to the engine's flat Metal buffer indices (vertex: buffer0=stream, buffer1=Frame,
// buffer2=push). Push constant: { float4x4 model; float4 color; } = 80 bytes (color.xyz = cluster RGB).
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 model; float4 clusterColor; };
#define HF_MODEL model
#define HF_COLOR clusterColor
#else
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::push_constant]] struct { float4x4 model; float4 color; } pc;
#define HF_MODEL pc.model
#define HF_COLOR pc.color
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    o.clip = mul(f.viewProj, world);
    // Fixed-direction Lambert (a constant key direction, NOT the scene light) so the sphere reads as a
    // 3D shape rather than a flat disc. Deterministic + backend-agnostic. Range [0.35,1] so back faces
    // keep their cluster hue rather than going black.
    float3 N = normalize(mul((float3x3)HF_MODEL, i.normal));
    float3 keyDir = normalize(float3(0.4, 0.7, 0.6));
    float lambert = saturate(dot(N, keyDir)) * 0.65 + 0.35;
    o.color = HF_COLOR.xyz * lambert;
    return o;
}
