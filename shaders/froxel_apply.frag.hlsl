// Slice CS — Froxel Volumetric Fog: the APPLY fullscreen pass (NEW path behind --froxelfog-shot). Per
// pixel it reconstructs the view-space LINEAR depth from the G-buffer (gGbuf.w, the SSAO/SSR view-space
// normal+linear-depth gbuffer), maps it to a froxel Z slice via render::froxel::ViewZToSlice, samples the
// INTEGRATED volume (float4 #1: accumulated in-scatter + transmittance) at that pixel's screen tile +
// depth slice (NEAREST — deterministic, documented), and composites out = scene * transmittance +
// inScatter into the HDR RT. The EXISTING post.frag then tonemaps the result (untouched).
//
// NEAREST sampling (documented choice): the apply floors the continuous (slice,tileX,tileY) coordinate to
// the integer froxel index and reads its accumulated value. Nearest is simpler + fully deterministic
// (no trilinear weights to drift cross-backend) and is sufficient for the showcase; trilinear/temporal is
// an explicit future slice.
//
// THE ZERO-DENSITY NO-OP PROOF: with baseDensity==0 every integrated cell has transmittance==1,
// inScatter==0 -> out = scene*1 + 0 = scene EXACTLY (byte-identical to the no-fog scene — the showcase
// compares the SAME apply path at density=0 against the no-fog render, backend-portable). A Z-slice
// off-by-one or a constant bias would break this, so the proof pins the whole inject->integrate->apply
// chain.
//
// SEAM DISCIPLINE: above the RHI seam; vk/MSL mentions are only HF_MSL_GEN guards + [[vk::binding]]
// decorations. The froxel volume reuses the Slice-AG cluster fragment-storage binding (binding 13,
// space 3 on Vulkan; a flat fragment buffer on Metal) bound via BindLightClusters — NO new RHI seam.
//
// BINDING LAYOUT (mirrors lit_clustered_cl.frag so the cluster volume lands at set 3 behind the
// frame(set0)+material(set1) sets): a frame UBO at set 0 (DECLARED but UNUSED — present only so the
// material set sits at index 1 and the cluster set at index 3), the scene+gbuffer textures at the
// MATERIAL set (set 1, bindings 0/1 + 3/4 — the base+normal slots BindTexturePair fills), and the
// froxel volume at the cluster set (set 3, binding 13).
struct FrameDummy { float4x4 a; float4x4 b; float4 c[6]; };  // never read; matches the frame UBO size class
[[vk::binding(0, 0)]] cbuffer Frame { FrameDummy gFrameUnused; };

[[vk::binding(0, 1)]] Texture2D    gScene : register(t0);   // lit HDR scene color (the no-fog render)
[[vk::binding(1, 1)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 1)]] Texture2D    gGbuf  : register(t3);   // view-space normal.xyz + linear depth .w
[[vk::binding(4, 1)]] SamplerState gGbufSmp : register(s3);

// The integrated froxel volume (flat FroxelCell[dimX*dimY*dimZ]) at the cluster-set fragment-storage
// binding 13/space3 (Vulkan) / flat fragment buffer (Metal), bound via BindLightClusters(volume,...).
struct FroxelCell {
    float4 scatterExt;   // xyz = injected scatter, w = extinction (unused here)
    float4 resultT;      // xyz = integrated in-scatter, w = transmittance (this pass reads)
};
[[vk::binding(13, 3)]] StructuredBuffer<FroxelCell> gVolume : register(t13, space3);

struct FroxelApplyParams {
    float4 dims;     // x=dimX, y=dimY, z=dimZ, w=enable (1 = apply fog, 0 = pass-through scene)
    float4 range;    // x=zNear, y=zFar, z/w unused
};
#ifdef HF_MSL_GEN
// MSL-gen path: declare the params as a cbuffer at binding(1,0) so spirv-cross --msl-decoration-binding
// lands it on Metal FRAGMENT buffer(1) == kFbPushConst (where MetalCommandBuffer::PushConstants writes).
// The frame UBO above is at binding(0,0) -> buffer(0) == kFbFrameUbo; the volume SSBO at buffer(13).
[[vk::binding(1, 0)]] cbuffer FroxelApplyPC { FroxelApplyParams ap; };
#define HF_AP ap
#else
[[vk::push_constant]] struct { FroxelApplyParams p; } pc;
#define HF_AP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// render::froxel::ViewZToSlice, copied VERBATIM: positive view-distance -> REAL slice coordinate, the
// exact inverse of SliceZ. Clamped to [0, dimZ].
float ViewZToSlice(float viewZ, float zNear, float zFar, uint dimZ) {
    if (viewZ <= zNear) return 0.0;
    if (viewZ >= zFar)  return (float)dimZ;
    return (float)dimZ * log(viewZ / zNear) / log(zFar / zNear);
}

float4 main(PSInput i) : SV_Target {
    float3 scene = gScene.Sample(gSmp, i.uv).rgb;       // linear HDR lit scene (the no-fog image)

    // enable==0: pass the scene through untouched (used by the no-fog reference path so the SAME apply
    // shader produces the byte-identical baseline — though density=0 already does that via T==1/L==0).
    if (HF_AP.dims.w < 0.5) return float4(scene, 1.0);

    uint dimX = (uint)(HF_AP.dims.x + 0.5);
    uint dimY = (uint)(HF_AP.dims.y + 0.5);
    uint dimZ = (uint)(HF_AP.dims.z + 0.5);
    float zNear = HF_AP.range.x;
    float zFar  = HF_AP.range.y;

    float linDepth = gGbuf.Sample(gGbufSmp, i.uv).w;    // view-space linear depth (positive in front)

    // Background / no geometry (cleared w == 0): use the FAR slice so distant haze still fogs the sky
    // toward zFar. (The fog over the sky reads as atmospheric depth haze.)
    float vz = (linDepth > 0.0001) ? linDepth : zFar;

    // Map the pixel to its froxel: screen tile (NEAREST) + depth slice (NEAREST, floored). The integrated
    // cell holds the fog accumulated between the eye and that froxel.
    uint tx = (uint)clamp(floor(i.uv.x * (float)dimX), 0.0, (float)dimX - 1.0);
    uint ty = (uint)clamp(floor(i.uv.y * (float)dimY), 0.0, (float)dimY - 1.0);
    float sliceF = ViewZToSlice(vz, zNear, zFar, dimZ);
    uint tz = (uint)clamp(floor(sliceF), 0.0, (float)dimZ - 1.0);

    uint c = tx + ty * dimX + tz * (dimX * dimY);
    float4 fog = gVolume[c].resultT;                    // xyz = in-scatter, w = transmittance

    // Single-scattering composite: out = scene * T + inScatter. T==1 & inScatter==0 -> out == scene.
    float3 outc = scene * fog.w + fog.xyz;
    return float4(outc, 1.0);
}
