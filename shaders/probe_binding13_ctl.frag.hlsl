// REGRESSION PROBE (fix-rhi-binding13) — the CONTROL variant. Identical readback of the three cluster-set
// storage buffers (set 3 bindings 13/14/15 via BindLightClusters) but WITHOUT the RT-graphics accel set —
// i.e. the pipeline layout contains exactly ONE push-descriptor set layout (the cluster set), the same
// shape as the shipped swraster_resolve_blit / clustered-lit pipelines, where binding 13 always worked.
// The control PASSING while the RT variant (probe_binding13.frag) FAILED pins the bug on the SECOND
// push-descriptor set layout (VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293), not on the placeholder
// set-1/set-2 layouts or the cluster push itself. Plain :ps (ps_6_0).

[[vk::binding(13, 3)]] StructuredBuffer<uint> gB13 : register(t13, space3);
[[vk::binding(14, 3)]] StructuredBuffer<uint> gB14 : register(t14, space3);
[[vk::binding(15, 3)]] StructuredBuffer<uint> gB15 : register(t15, space3);

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

uint main(PSInput i) : SV_Target0 {
    uint x = (uint)i.pos.x;
    if (x < 64u)  return gB13[x];
    if (x < 128u) return gB14[x - 64u];
    return gB15[x - 128u];
}
