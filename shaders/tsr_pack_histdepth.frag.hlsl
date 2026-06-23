// US3 history+depth PACK pass (Slice US3, issue #20). The RHI fragment material set carries only TWO
// sampled images (binding 0/1 + binding 3/4 — the same two slots BindTexturePair / bloom-composite use),
// but the reproject resolve needs THREE inputs: the half-res current scene, the full-res history color,
// AND the full-res current linear depth. Rather than change the RHI, this tiny fullscreen pass folds the
// two FULL-res inputs into ONE RGBA16F target: rgb = the previous-frame history color (gHist, t0), a =
// the current-frame G-buffer LINEAR depth (gDepth.w, t3). The reproject resolve then binds (sceneLow,
// packed) via BindTexturePair and reads gHistory.rgb (at the reprojected UV) + gHistory.a (the linear
// depth, at the identity UV) from the single packed texture — staying within the 2-slot material set.
// NEW shader; no RHI change; existing shaders + goldens untouched.
[[vk::binding(0, 0)]] Texture2D    gHist  : register(t0);   // FULL-res history color (HDR)
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gDepth : register(t3);   // FULL-res G-buffer (view normal.xyz + linDepth.w)
[[vk::binding(4, 0)]] SamplerState gDSmp  : register(s3);

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

float4 main(PSInput i) : SV_Target {
    float3 history  = gHist.Sample(gSmp, i.uv).rgb;
    float  linDepth = gDepth.Sample(gDSmp, i.uv).w;   // positive in front of the camera; 0 = no surface
    return float4(history, linDepth);
}
