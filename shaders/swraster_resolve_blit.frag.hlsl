// Slice SW4 — Software-Raster vis-buffer -> Deferred-Material RESOLVE: the "unpack" BLIT. A FULLSCREEN
// pass (paired with the post.vert fullscreen triangle) that converts the SOFTWARE rasterizer's packed
// `depth|id` visibility-buffer SSBO (render/swraster.h::PackSw / shaders/swraster.comp.hlsl) into the
// R32_Uint visId render target the EXISTING deferred resolve (shaders/visresolve.frag.hlsl) consumes,
// WITHOUT touching that golden-locked resolve shader. This makes the SW vis-buffer conform to the
// resolve's input contract so visresolve.frag is reused BYTE-FOR-BYTE.
//
// THE TRANSFORM (plain uint32 — NO int64, NO integer texture.read; this WRITES an integer SV_Target,
// which is plain MSL, so NO --msl-version 20200 is needed for this shader):
//   uint packed = gSwVis[y*width + x];                    // the SW SSBO (set 3 binding 13)
//   uint out    = (packed == 0xFFFFFFFFu) ? 0xFFFFFFFFu   // kSwClear -> kVisBackground (identical
//                                         : (packed & 0xFFFFu);  // strip the HIGH-16 depth, keep the
//                                                                // LOW-16 visId = (clusterID<<7)|triID
// Correct because (1) kSwClear == kVisBackground == 0xFFFFFFFF (swraster.h:65 / visbuffer.h:45) -> the
// background sentinel passes through and the resolve's sky branch fires identically; (2) the SW visId
// low-16 bits ARE exactly (clusterID<<kTriIdBits)|triID (the same PackVisId, kTriIdBits=7) and
// zero-extend into the resolve's 25-bit cluster field with no loss; (3) DEPTH is discardable — the SW
// InterlockedMin/serial-min over the depth|id key already resolved occlusion, so each blit texel holds
// the winning fragment and the resolve runs depthTest=false.
//
// BINDINGS (NO new RHI):
//   set 0 binding 0      : a tiny UBO {width,height} bound via SetFrameUniforms (the same frame-uniform
//                          path the vrPipeline uses); only .x/.y are read here.
//   set 3 binding 13     : the SW vis SSBO (READ-ONLY StructuredBuffer<uint>) bound via the existing
//                          BindLightClusters fragment-stage storage-buffer path (the froxel-apply
//                          precedent: spirv-cross maps space3 13 -> Metal fragment buffer 13).
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::) symbols. The only vk/MSL mention is the [[vk::binding]]
// decorations (same as visresolve.frag / froxel_apply.frag), not backend CODE.

static const uint kSwClear        = 0xFFFFFFFFu;  // swraster.h:65 (the packed depth|id clear sentinel)
static const uint kVisBackground  = 0xFFFFFFFFu;  // visbuffer.h:45 (the resolve's sky sentinel)
static const uint kSwIdMask       = 0xFFFFu;      // swraster.h:54 (the LOW-16 visId field, depth stripped)

// The tiny per-pass UBO: x=width, y=height (the vis-buffer dimensions, == this pass's target size). The
// first 64 bytes stay a mat4-shaped slot for UBO layout regularity (same convention as visresolve.frag).
struct BlitParams {
    float4x4 pad0;   //   0  (unused; keeps the UBO's first 64 bytes a mat4 like the others)
    uint4    dims;   //  64  x = width, y = height, zw pad
};
[[vk::binding(0, 0)]] cbuffer Frame { BlitParams p; };

// The SW vis SSBO at the fragment-stage cluster-set slot (set 3 binding 13), bound via BindLightClusters.
// READ-ONLY StructuredBuffer (the SAME read-only fragment-storage decl visresolve.frag uses for
// gClusterMeta at this exact slot) so DXC emits the NonWritable decoration — a WRITABLE (RW) storage
// buffer in the fragment stage requires the fragmentStoresAndAtomics device feature (which the engine
// does NOT enable), so a read-only StructuredBuffer is both correct and validation-clean. spirv-cross
// lowers it to a `const device uint*` Metal fragment buffer at 13.
[[vk::binding(13, 3)]] StructuredBuffer<uint> gSwVis : register(t13, space3);

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// SV_Target0 is an R32_Uint texel (the unpacked visId). uint return -> OpStore to an integer color
// attachment (plain MSL — NOT the integer texture.read that needed MSL 2.2 in visresolve.frag).
uint main(PSInput i) : SV_Target0 {
    // Integer texel coordinate of THIS pixel (== visresolve.frag's convention). SV_Position.xy is the
    // pixel center (px+0.5); floor to the integer texel. The SSBO is the same w*h as this pass's target.
    int2 px = int2(i.pos.xy);
    uint width  = p.dims.x;
    uint idx    = (uint)px.y * width + (uint)px.x;
    uint packed = gSwVis[idx];
    return (packed == kSwClear) ? kVisBackground : (packed & kSwIdMask);
}
