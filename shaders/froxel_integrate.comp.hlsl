// Slice CS — Froxel Volumetric Fog: the INTEGRATE compute pass. ONE thread per (x,y) COLUMN
// (dimX*dimY threads). Each thread marches its column's froxels Z 0->dimZ FRONT-TO-BACK, applying the
// render::froxel::IntegrateStep single-scattering recurrence per froxel and writing the accumulated
// (inScatterRGB, transmittance) into float4 #1 of each cell. The per-froxel step length is the view-space
// thickness of that slice (SliceZ(fz+1) - SliceZ(fz)), so the optical depth is physically the integral of
// extinction along the view ray and the transmittance is the analytic Beer-Lambert product.
//
// The IntegrateStep math is the verbatim mirror of engine/render/froxel.h: inScatter is added at the
// CURRENT (front-accumulated) transmittance BEFORE this froxel's extinction is applied (so a near
// scatterer contributes at full T==1), then T is attenuated by exp(-extinction*stepLen). With every
// cell's extinction/scatter == 0 (baseDensity==0) the recurrence is a per-step NO-OP: every cell ends
// with T==1, inScatter==0 -> the apply is a pure pass-through (the zero-density == no-fog-scene proof).
//
// Buffers (storage, bound at compute bindings 0..1):
//   b0 gVolume : the flat FroxelCell[dimX*dimY*dimZ], READ float4 #0 (scatter+ext) / WRITE float4 #1.
//   b1 gParams : grid dims + zNear/zFar (the step lengths), READ.
//
// SEAM DISCIPLINE: above the RHI seam; vk/MSL mentions are only HF_MSL_GEN guards + [[vk::binding]]
// decorations (not backend CODE symbols). Bit-identical Vulkan (DXC) / Metal (spirv-cross) math.

#define HF_FROXEL_THREADS 64

struct FroxelCell {
    float4 scatterExt;   // xyz = injected in-scatter, w = extinction (inject wrote)
    float4 resultT;      // xyz = integrated in-scatter, w = transmittance (this pass writes)
};

struct FroxelParams {
    uint4    dims;        // x=dimX, y=dimY, z=dimZ, w=unused
    float4   range;       // x=zNear, y=zFar, z/w unused
    float4   sunDir;
    float4   sunColor;
    float4   fog;
    float4x4 invProj;
    float4x4 invView;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FroxelCell>   gVolume : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FroxelParams> gParams : register(u1);

// SliceZ(k): positive view-distance of exponential slice boundary at REAL k. Mirrors froxel.h::SliceZ.
float SliceZ(float k, uint dimZ, float zNear, float zFar) {
    return zNear * pow(zFar / zNear, k / (float)dimZ);
}

[numthreads(HF_FROXEL_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint dimX = gParams[0].dims.x;
    uint dimY = gParams[0].dims.y;
    uint dimZ = gParams[0].dims.z;
    float zNear = gParams[0].range.x;
    float zFar  = gParams[0].range.y;

    uint columnCount = dimX * dimY;
    uint col = gid.x;
    if (col >= columnCount) return;

    uint fy = col / dimX;
    uint fx = col % dimX;

    // Front-to-back single-scatter march of this column. T starts at 1 (clear), L at 0.
    float  T = 1.0;
    float3 L = float3(0.0, 0.0, 0.0);
    for (uint fz = 0; fz < dimZ; ++fz) {
        uint c = fx + fy * dimX + fz * (dimX * dimY);
        float4 se = gVolume[c].scatterExt;          // xyz = scatter (per unit length), w = extinction
        float stepLen = SliceZ((float)(fz + 1), dimZ, zNear, zFar) -
                        SliceZ((float)fz,        dimZ, zNear, zFar);

        // render::froxel::IntegrateStep, copied VERBATIM:
        //   L += T * scatter * stepLen;   then   T *= exp(-extinction * stepLen);
        L = L + se.xyz * (T * stepLen);
        T *= exp(-se.w * stepLen);

        // Store the ACCUMULATED (in-scatter, transmittance) AT this froxel (front-inclusive): the apply
        // samples the integrated value at the pixel's depth slice, which is exactly the fog accumulated
        // between the eye and that froxel.
        gVolume[c].resultT = float4(L, T);
    }
}
