// Slice CS — Froxel Volumetric Fog: the INJECT compute pass. ONE thread per froxel (dimX*dimY*dimZ),
// dispatched as enough HF_FROXEL_THREADS workgroups to cover the volume. Each thread reconstructs its
// froxel's VIEW-space center (the screen-tile NDC center unprojected via invProj at the slice's center
// view-distance — the EXACT mirror of render::froxel::SliceCenterViewZ + cluster.h's UnprojectToView),
// transforms it to WORLD space (via invView) to evaluate the height-fog Density, computes the sun in-
// scatter (sunColor * Phase(cos(viewDir,sunDir), g) * density) and the extinction (density), and writes
// (scatterRGB, extinction) into float4 #0 of the FLAT SSBO froxel volume cell [fx + fy*dimX + fz*dimX*dimY].
//
// The math (exponential Z slices, NDC-corner unprojection via invProj, height density, HG phase) is the
// verbatim mirror of engine/render/froxel.h. With baseDensity==0 every cell's density is 0 -> scatter 0,
// extinction 0 -> the integrate pass is a per-step no-op -> the apply is a pure pass-through (the
// zero-density == no-fog-scene proof).
//
// Buffers (storage, bound at compute bindings 0..1; on Metal they land at buffer(kCsStorage+0..1)):
//   b0 gVolume : the flat FroxelCell[dimX*dimY*dimZ] volume, WRITE (float4 #0 = scatter.rgb + extinction).
//   b1 gParams : grid dims + zNear/zFar + sun + fog params + invProj + invView, READ.
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL are the HF_MSL_GEN
// generation-path guards + [[vk::binding]] decorations (same as cluster_assign / gtao.frag), not backend
// CODE symbols. spirv-cross maps the SPIR-V bindings to Metal's flat compute buffer indices so the SAME
// HLSL feeds Vulkan (DXC) and Metal (glslang->spirv-cross): bit-identical math.

#define HF_FROXEL_THREADS 64

static const float HF_PI = 3.14159265358979323846;

// One froxel cell (std430, 32 bytes): mirrors render::froxel::FroxelCell. float4 #0 = injected scatter +
// extinction (this pass writes it); float4 #1 = integrated in-scatter + transmittance (integrate writes).
struct FroxelCell {
    float4 scatterExt;   // xyz = in-scatter radiance, w = extinction
    float4 resultT;      // xyz = integrated in-scatter, w = transmittance
};

// Froxel-grid + camera + fog params (std430). Shared by all three froxel passes.
struct FroxelParams {
    uint4    dims;        // x=dimX, y=dimY, z=dimZ, w=unused
    float4   range;       // x=zNear, y=zFar, z/w unused
    float4   sunDir;      // xyz = directional-light TRAVEL direction (world space), w unused
    float4   sunColor;    // rgb = sun color * intensity, w unused
    float4   fog;         // x=baseDensity, y=heightFalloff, z=heightRef, w=g (HG anisotropy)
    float4x4 invProj;     // proj.Inverse(), column-major (recovers view-space XY from NDC)
    float4x4 invView;     // view.Inverse(), column-major (view -> world for density height)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FroxelCell>   gVolume : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FroxelParams> gParams : register(u1);

// SliceZ(k): positive view-distance of exponential slice boundary at REAL k. Mirrors froxel.h::SliceZ.
float SliceZ(float k, uint dimZ, float zNear, float zFar) {
    return zNear * pow(zFar / zNear, k / (float)dimZ);
}

// Unproject NDC (nx,ny) at positive view-distance vz to view space, via invProj. Mirrors
// cluster.h/froxel.h UnprojectToView: form the NDC ray (z=0), scale to the plane z = -vz.
float3 UnprojectToView(float4x4 invProj, float nx, float ny, float vz) {
    float4 h = mul(invProj, float4(nx, ny, 0.0, 1.0));
    float3 vDir = h.xyz / h.w;
    float  s = (abs(vDir.z) > 1e-9) ? (-vz / vDir.z) : 0.0;
    return float3(vDir.x * s, vDir.y * s, -vz);
}

// Height-fog density at a world point. Mirrors froxel.h::Density EXACTLY (baseDensity==0 -> 0 EXACTLY).
float Density(float3 worldPos, float baseDensity, float heightFalloff, float heightRef) {
    if (baseDensity == 0.0) return 0.0;
    float d = baseDensity * exp(-heightFalloff * (worldPos.y - heightRef));
    return d > 0.0 ? d : 0.0;
}

// Henyey-Greenstein phase. Mirrors froxel.h::Phase (forward-peaked g>0, isotropic g=0, finite).
float Phase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    if (denom < 1e-6) denom = 1e-6;
    return (1.0 - g2) / (4.0 * HF_PI * denom * sqrt(denom));
}

[numthreads(HF_FROXEL_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint dimX = gParams[0].dims.x;
    uint dimY = gParams[0].dims.y;
    uint dimZ = gParams[0].dims.z;
    float zNear = gParams[0].range.x;
    float zFar  = gParams[0].range.y;
    float4x4 invProj = gParams[0].invProj;
    float4x4 invView = gParams[0].invView;
    float3 sunDirWorld = normalize(gParams[0].sunDir.xyz);   // light TRAVEL direction (world)
    float3 sunColor = gParams[0].sunColor.rgb;
    float baseDensity = gParams[0].fog.x;
    float heightFalloff = gParams[0].fog.y;
    float heightRef = gParams[0].fog.z;
    float g = gParams[0].fog.w;

    uint froxelCount = dimX * dimY * dimZ;
    uint c = gid.x;
    if (c >= froxelCount) return;

    // Decode (fx,fy,fz) from the flat index: idx = fx + fy*dimX + fz*(dimX*dimY).
    uint fz  = c / (dimX * dimY);
    uint rem = c % (dimX * dimY);
    uint fy  = rem / dimX;
    uint fx  = rem % dimX;

    // The froxel's view-space CENTER: the screen-tile NDC center unprojected at the slice center view-z.
    float nx = (((float)fx + 0.5) / (float)dimX) * 2.0 - 1.0;
    float ny = (((float)fy + 0.5) / (float)dimY) * 2.0 - 1.0;
    float vzCenter = SliceZ((float)fz + 0.5, dimZ, zNear, zFar);   // SliceCenterViewZ
    float3 viewCenter = UnprojectToView(invProj, nx, ny, vzCenter);

    // View -> world for the height-based density (invView is affine; w stays 1).
    float3 worldCenter = mul(invView, float4(viewCenter, 1.0)).xyz;
    float density = Density(worldCenter, baseDensity, heightFalloff, heightRef);

    // Sun single-scatter in-scatter: the cosine between the VIEW ray (froxel center -> eye, i.e. toward
    // the camera = -normalize(viewCenter)) and the direction TOWARD the sun (-sunDir). Looking toward the
    // sun through the fog -> cosTheta near 1 -> a forward-peaked phase glow.
    float3 toCamera = normalize(-viewCenter);
    // Transform the world sun travel-dir into view space via invView's inverse is the view matrix; but we
    // only need the COSINE, which is rotation-invariant — so compute it in WORLD space: the world view ray
    // is worldCenter -> camera. Reconstruct the world ray as the world-space direction of -viewCenter.
    float3 viewRayWorld = normalize(mul(invView, float4(toCamera, 0.0)).xyz);  // world dir toward camera
    float3 toSunWorld = normalize(-sunDirWorld);                               // world dir toward the sun
    float cosTheta = dot(viewRayWorld, toSunWorld);
    float phase = Phase(cosTheta, g);

    float3 scatter = sunColor * (phase * density);
    float extinction = density;

    gVolume[c].scatterExt = float4(scatter, extinction);
    // Leave resultT untouched here; the integrate pass overwrites it. (Initialize for safety/determinism.)
    gVolume[c].resultT = float4(0.0, 0.0, 0.0, 1.0);
}
