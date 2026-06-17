// Slice DO — DDGI Visibility Slice 1: per-probe DISTANCE-moment capture vertex shader. A near-clone of
// probe_bake.vert (the DD/DI per-face capture vertex): transforms the room geometry into ONE cube face's
// clip space with the face view-proj carried in the PUSH CONSTANT alongside the model matrix, and passes
// the WORLD position through to the fragment. The ONLY difference from probe_bake.vert is the push
// constant ALSO carries `probeCentre` (float3) so the fragment can compute the linear world-distance
// length(worldPos - probeCentre) — the vertex stage itself ignores probeCentre (it only needs faceVP +
// model), but the block must be declared identically in both stages so the Vulkan push-constant layout +
// the MSL buffer binding agree.
//
// Push constant: { float4x4 faceViewProj; float4x4 model; float4 probeCentre; } = 144 bytes (the frag
// reads probeCentre.xyz; .w is padding). NO lighting/normal needed for the distance render, but we keep
// the same VSInput as probe_bake.vert so the SAME mesh vertex layout binds.
struct VSInput {
    [[vk::location(0)]] float3 pos     : POSITION;
    [[vk::location(1)]] float3 color   : COLOR;
    [[vk::location(2)]] float2 uv      : TEXCOORD0;
    [[vk::location(3)]] float3 normal  : NORMAL;
    [[vk::location(4)]] float3 tangent : TANGENT;
};
struct VSOutput {
    float4 clip : SV_Position;
    [[vk::location(0)]] float3 wpos : POSITION0;   // world position -> the frag computes the distance
};
// Same HF_MSL_GEN seam-discipline as probe_bake.vert: glslang lowers [[vk::push_constant]] to a plain
// uniform, so for MSL generation the push block is declared as an explicit cbuffer at binding (2,0) whose
// spirv-cross flat index matches the engine's buffer2=push layout. The Vulkan/DXC path keeps the real
// push constant. (These #ifdef'd names are NOT backend code symbols — seam-discipline doc/macros only.)
#ifdef HF_MSL_GEN
[[vk::binding(2, 0)]] cbuffer PushC { float4x4 faceViewProj; float4x4 model; float4 probeCentre; };
#define HF_VP    faceViewProj
#define HF_MODEL model
#else
[[vk::push_constant]] struct { float4x4 faceViewProj; float4x4 model; float4 probeCentre; } pc;
#define HF_VP    pc.faceViewProj
#define HF_MODEL pc.model
#endif

VSOutput main(VSInput i) {
    VSOutput o;
    float4 world = mul(HF_MODEL, float4(i.pos, 1.0));
    o.wpos = world.xyz;
    o.clip = mul(HF_VP, world);
    return o;
}
