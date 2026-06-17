// Slice DX — visibility-buffer DEFERRED MATERIAL RESOLVE fragment shader. A FULLSCREEN pass (paired
// with the post.vert fullscreen triangle) that texel-fetches the DW R32_Uint visibility buffer per
// pixel, reconstructs the covering triangle, computes a FLAT geometric normal + Lambert shade, and
// outputs the lit color — the Nanite-style decoupled geometry->material resolve.
//
// This is the VERBATIM GPU copy of hf::render::vg::ResolveFlatShade / ResolvePixel (engine/render/
// visresolve.h). The flat normal's cross product uses `mad` (DXC -> SPIR-V the driver fuses; spirv-cross
// -> Metal fma) to match the CPU std::fma bit-for-bit (the DH/DV FP discipline) so the GPU resolve ==
// the CPU mirror at interior pixels. NO per-pixel barycentric interpolation (flat). The light direction
// arrives PRE-NORMALIZED in the UBO (never renormalized on the GPU).
//
// BINDINGS (NO new RHI):
//   set 0 binding 0 : the per-pass UBO (lightDir/albedo/ambient/sky/drawnClusterCount + per-cluster
//                     model matrices), bound via SetFrameUniforms.
//   set 1 binding 0 : the integer visibility buffer as a SAMPLED uint texture, bound via BindTexture
//                     (the DW vis-buffer RT). Texel-FETCHED via .Load(int3(px,py,0)) -> OpImageFetch /
//                     MSL texture.read — NO sampler (the integer image is never combined with one). This
//                     is the shader that needs MSL 2.2 (--msl-version 20200) for the integer texture.read.
//   set 3 binding 13/14/15 : gClusterMeta / gIndices / gVerts fragment storage buffers, bound via the
//                     existing BindLightClusters path (spirv-cross maps space3 13/14/15 -> Metal
//                     fragment buffers 13/14/15).

// kTriIdBits / kVisBackground MUST match hf::render::vg (visbuffer.h). 7-bit triID, 0xFFFFFFFF sentinel.
static const uint kTriIdBits = 7;
static const uint kTriIdMask = 0x7Fu;
static const uint kVisBackground = 0xFFFFFFFFu;

// Per-pass params (the showcase fills the per-frame UBO with THIS layout via SetFrameUniforms). lit.vert
// is NOT used (fullscreen), so there is no viewProj contract to honor — but offset 0 stays a float4x4
// slot for layout regularity. drawnClusterCount + the per-cluster meta drive the resolve.
struct ResolveParams {
    float4x4 pad0;            //   0  (unused; keeps the UBO's first 64 bytes a mat4 like the others)
    float4   lightDir;        //  64  world-space light TRAVEL dir, PRE-NORMALIZED (surface lit by -dir)
    float4   albedo;          //  80  xyz = diffuse albedo, w = ambient term
    float4   sky;             //  96  xyz = background/sky color (where the vis-buffer is the sentinel)
    uint4    counts;          // 112  x = drawnClusterCount, yzw pad
};
[[vk::binding(0, 0)]] cbuffer Frame { ResolveParams p; };

// The integer visibility buffer as a SAMPLED uint texture at the material slot (set 1 binding 0). Bound
// via BindTexture(visRT). Texel-fetched (no sampler) -> the integer fetch needs MSL 2.2.
[[vk::binding(0, 1)]] Texture2D<uint> gVisBuffer : register(t0);

// Per-cluster meta the resolve needs: triOffset (into gIndices, in TRIANGLES) + the owning instance's
// model matrix (column-major float4x4). std430; the showcase uploads one ClusterMeta per SURVIVOR-DRAW
// (indexed by the unpacked clusterID directly). 80 bytes: mat4 (64) + uint4 (16).
struct ClusterMeta {
    float4 modelCol0;        // the four columns of the column-major model matrix
    float4 modelCol1;
    float4 modelCol2;
    float4 modelCol3;
    uint4  info;             // x = triOffset (triangles), yzw pad
};
[[vk::binding(13, 3)]] StructuredBuffer<ClusterMeta> gClusterMeta : register(t13, space3);
// The shared reordered index buffer (DS/DT) + the shared vertex buffer. Bound at the next two cluster-set
// slots (14/15) via BindLightClusters(meta, indices, verts).
[[vk::binding(14, 3)]] StructuredBuffer<uint>  gIndices : register(t14, space3);
// The shared vertex buffer as a FLAT float array (the engine scene::Vertex is a TIGHTLY-PACKED 56-byte
// struct = 14 floats: pos[3] color[3] uv[2] normal[3] tangent[3]). A StructuredBuffer<float> avoids the
// HLSL float3-in-structured-buffer alignment trap (a struct mirror would pad float3 members to 16 bytes
// and read the wrong bytes). The resolve only needs pos = the first 3 floats of each 14-float stride.
static const uint kVertStrideF = 14u;   // 56 bytes / 4
[[vk::binding(15, 3)]] StructuredBuffer<float> gVertsF : register(t15, space3);
float3 LoadVertPos(uint vidx) {
    uint o = vidx * kVertStrideF;
    return float3(gVertsF[o + 0], gVertsF[o + 1], gVertsF[o + 2]);
}

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// VERBATIM mirror of vg::FlatNormal: cross(p1-p0, p2-p0) with each component as one `mad` (== std::fma),
// then normalize. Outward geometric normal for the engine's CCW-front-facing geometry.
float3 FlatNormal(float3 p0, float3 p1, float3 p2) {
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float3 n;
    n.x = mad(e1.y, e2.z, -(e1.z * e2.y));
    n.y = mad(e1.z, e2.x, -(e1.x * e2.z));
    n.z = mad(e1.x, e2.y, -(e1.y * e2.x));
    return normalize(n);
}

// VERBATIM mirror of vg::LambertShade: ndl = max(0, dot(n,-lightDir)); out = albedo*ambient + albedo*ndl
// (one mad per channel == std::fma).
float3 LambertShade(float3 n, float3 lightDir, float3 albedo, float ambient) {
    float ndl = max(0.0, -dot(n, lightDir));
    return float3(mad(albedo.x, ndl, albedo.x * ambient),
                  mad(albedo.y, ndl, albedo.y * ambient),
                  mad(albedo.z, ndl, albedo.z * ambient));
}

float4 main(PSInput i) : SV_Target {
    // Integer texel coordinate of THIS pixel. SV_Position.xy is the pixel center (px+0.5); floor to the
    // integer texel. The vis-buffer RT is the same size as this fullscreen pass's target (1:1), so no
    // scaling — texel (px,py) of the vis-buffer is exactly this pixel.
    int2 px = int2(i.pos.xy);
    uint v = gVisBuffer.Load(int3(px, 0));   // OpImageFetch / texture.read (no sampler)

    // Background sentinel OR an out-of-range survivor -> sky. (clusterID >= drawnClusterCount can only
    // happen for the sentinel here, but the explicit guard mirrors the CPU ResolvePixel branch.)
    uint cid = v >> kTriIdBits;
    uint tid = v & kTriIdMask;
    if (v == kVisBackground || cid >= p.counts.x) {
        return float4(p.sky.xyz, 1.0);
    }

    ClusterMeta cm = gClusterMeta[cid];
    uint triOffset = cm.info.x;
    // Reassemble the column-major model matrix from the four per-cluster columns (same convention as
    // visbuffer.vert / cluster_viz.vert) so the world positions match the rasterized geometry.
    float4x4 model = float4x4(
        float4(cm.modelCol0.x, cm.modelCol1.x, cm.modelCol2.x, cm.modelCol3.x),
        float4(cm.modelCol0.y, cm.modelCol1.y, cm.modelCol2.y, cm.modelCol3.y),
        float4(cm.modelCol0.z, cm.modelCol1.z, cm.modelCol2.z, cm.modelCol3.z),
        float4(cm.modelCol0.w, cm.modelCol1.w, cm.modelCol2.w, cm.modelCol3.w));

    uint base = 3u * (triOffset + tid);
    uint i0 = gIndices[base + 0];
    uint i1 = gIndices[base + 1];
    uint i2 = gIndices[base + 2];
    float3 o0 = LoadVertPos(i0);
    float3 o1 = LoadVertPos(i1);
    float3 o2 = LoadVertPos(i2);
    float3 w0 = mul(model, float4(o0, 1.0)).xyz;
    float3 w1 = mul(model, float4(o1, 1.0)).xyz;
    float3 w2 = mul(model, float4(o2, 1.0)).xyz;

    float3 n = FlatNormal(w0, w1, w2);
    float3 lit = LambertShade(n, p.lightDir.xyz, p.albedo.xyz, p.albedo.w);
    return float4(lit, 1.0);
}
