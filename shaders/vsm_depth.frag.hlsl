// Slice VB — Virtual Shadow Maps physical-page DEPTH render. Paired with shadow_csm.vert (which
// transforms geometry into a page's light clip space via the cascadeViewProj push constant). Unlike the
// depth-only CSM shadow pass (shadow.frag, no color), VB renders into a COLOR atlas so the per-tile
// depth can be ReadRenderTarget'd back as the golden. The fragment outputs the post-perspective-divide
// NDC depth (SV_Position.z, already in [0,1] for the page's ortho) as GRAYSCALE — near=bright, far=dark.
//
// The depth pass uses an ortho per page, so SV_Position.z is the linear normalized depth across the
// page's near/far -> the tile shows the casters' silhouettes as a depth gradient. Resident tiles get
// these gradients; non-resident tiles keep the render-target CLEAR color (the showcase never draws into
// them). CPU-colorization for the final BGRA viz is applied by the showcase after readback; this shader
// just lays down per-tile grayscale depth so the read-back atlas already encodes each page's depth.
//
// SV_Position in a pixel shader is the screen-space position; .z is the interpolated NDC depth (the same
// value written to the depth buffer). For an ortho projection that is a LINEAR remap of view-space Z, so
// near surfaces -> small z (bright), far -> large z (dark). We invert (1 - z) so closer casters read
// BRIGHTER than the receiver/background, which makes the silhouettes pop in the colorized atlas.
struct PSInput {
    float4 pos : SV_Position;
};

float4 main(PSInput i) : SV_Target {
    float d = saturate(i.pos.z);   // NDC depth in [0,1] (Vulkan/Metal depth range)
    float g = 1.0 - d;             // near = bright, far = dark
    return float4(g, g, g, 1.0);
}
