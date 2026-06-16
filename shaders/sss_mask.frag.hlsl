// Subsurface "SSS G-buffer" prepass -- fragment shader (Slice CZ). Reuses gbuffer.vert (model + view in
// the push constant, view-space normal + position to the fragment). ONLY the flagged SUBSURFACE objects
// are drawn through this pass; the RT is cleared to 0 first, so every opaque/background pixel keeps
// mask == 0 + depth == 0 and every subsurface pixel gets mask == 1 + its view-space linear depth. The
// sss_blur pass reads .r as the subsurface MASK and .w as the VIEW-SPACE LINEAR DEPTH (= -vpos.z, the
// SAME convention gbuffer.frag uses), so SSS diffuses ONLY within the flagged region and is depth-aware
// across silhouettes. This is the "flagged material channel" the design calls for, carried in a
// purpose-built mask+depth RT bound as the BindTexturePair partner -- the SAME fullscreen two-pass path
// SSR/DoF/SSGI-denoise use, with NO new RHI seam.
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 vnormal : NORMAL;    // unused here (SSS needs no normal); from gbuffer.vert
    [[vk::location(1)]] float3 vpos    : POSITION0;
};

float4 main(PSInput i) : SV_Target {
    float linDepth = -i.vpos.z;     // positive in front of the camera (RH view space, -Z forward)
    // mask = 1 (this pass only ever draws subsurface objects); depth in .w. The middle channels are 0.
    return float4(1.0, 0.0, 0.0, linDepth);
}
