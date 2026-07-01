// REGRESSION PROBE (fix-rhi-binding13) — the RT-GRAPHICS variant. Reads back the THREE cluster-set
// storage buffers (set 3 bindings 13/14/15, pushed via BindLightClusters) from a GRAPHICS pipeline that
// ALSO declares the dedicated RT-graphics accel set (Issue #34, [[vk::binding(0,4)]]). This is the exact
// pipeline-layout shape that broke cluster binding 13 (documented in
// docs/superpowers/specs/2026-06-23-rt-graphics-reflect-design.md): the layout used to contain TWO
// push-descriptor set layouts (cluster set 3 + accel set 4), violating
// VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293 ("at most ONE push descriptor set layout per pipeline
// layout") — undefined behavior, which on the test driver clobbered the FIRST descriptor written by the
// cluster push (binding 13) while 14/15 survived. The host fills each buffer with a distinctive pattern
// and asserts the rendered readback equals it, per binding.
//
// Pixel x in [0,64)   -> gB13[x]        (expected 0xA1B2C3D4)
// Pixel x in [64,128) -> gB14[x-64]     (expected 0x0EA7BEEFu)
// Pixel x in [128,192)-> gB15[x-128]    (expected 0x5EED5EEDu)
//
// The TLAS is STATICALLY USED (a runtime-never-taken ray query gated on a sentinel the host never
// writes) so DXC keeps the accel binding and the pipeline faithfully reproduces the failing config.
// :psrq stage (ps_6_5 + SPV_KHR_ray_query) — Vulkan-SPIR-V-only, NOT in the Metal MSL list.

[[vk::binding(13, 3)]] StructuredBuffer<uint> gB13 : register(t13, space3);
[[vk::binding(14, 3)]] StructuredBuffer<uint> gB14 : register(t14, space3);
[[vk::binding(15, 3)]] StructuredBuffer<uint> gB15 : register(t15, space3);
[[vk::binding(0, 4)]]  RaytracingAccelerationStructure gTlas;

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

uint main(PSInput i) : SV_Target0 {
    uint x = (uint)i.pos.x;
    // Statically use the TLAS (never taken at runtime: the host never writes the sentinel).
    if (gB15[63] == 0xDEADBEEFu) {
        RayDesc r;
        r.Origin = float3(0.0f, 0.0f, 0.0f);
        r.Direction = float3(0.0f, 1.0f, 0.0f);
        r.TMin = 0.0f;
        r.TMax = 1.0f;
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, r);
        q.Proceed();
        return 0u;
    }
    if (x < 64u)  return gB13[x];
    if (x < 128u) return gB14[x - 64u];
    return gB15[x - 128u];
}
