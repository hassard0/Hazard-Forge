// Slice DW — visibility-buffer fragment shader. Outputs a SINGLE uint to SV_Target0 (an R32_Uint
// render target): the packed identity of the front-most surface fragment,
//     (clusterID << 7) | (SV_PrimitiveID & 0x7F)
// where clusterID is the flat survivor-draw index forwarded by visbuffer.vert and SV_PrimitiveID is
// the per-primitive triangle index WITHIN this survivor's draw (the FIRST SV_PrimitiveID use in the
// repo; Vulkan DXC lowers it to the SPIR-V PrimitiveId builtin). NO lighting / color / texture / RNG —
// the written value is a FLAT integer, so it is bit-exact across Vulkan and Metal by construction.
//
// The pipeline enables depthTest + depthWrite (no blend), so the rasterizer's depth resolve keeps the
// front-most cluster/triangle per pixel — exactly the visibility a deferred material resolve (DX) then
// texel-fetches. This MUST stay in lockstep with hf::render::vg::PackVisId (kTriIdBits = 7).
struct PSInput {
    float4 clip : SV_Position;
    [[vk::location(0)]] nointerpolation uint clusterID : CLUSTERID;
};

uint main(PSInput i, uint primID : SV_PrimitiveID) : SV_Target0 {
    return (i.clusterID << 7) | (primID & 0x7Fu);
}
