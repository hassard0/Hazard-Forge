// Slice DL — DDGI Slice 4: the PROBE TRILINEAR SH-INTERP compute pass. ONE thread per QUERY POINT,
// dispatched as enough HF_INTERP_THREADS workgroups to cover the query count. Each thread reads its
// query's world position, finds the FLOOR cell whose 8 corner probes bracket it (NearestProbes), and
// trilinearly blends those 8 probes' 3rd-order real-SH coefficients (9 per RGB channel = 27 floats) into
// a single ProbeSH, writing it to the per-query output SSBO. This is the trilinear interpolation
// primitive the GI composite (the next DDGI slice) samples per pixel.
//
// THE BIT-EXACT GPU==CPU PROOF: NearestProbes (the FLOOR-cell 8-corner lookup + polynomial trilinear
// weights) and InterpolateSH (the per-corner fma SH accumulation) are copied VERBATIM from
// engine/render/probe_gi.h + engine/render/probe_sh.h, so this pass and the CPU reference
// (probesh::InterpolateSH over the SAME uploaded ProbeSH[] + query points) produce a BIT-EXACT per-query
// blended SH buffer (memcmp == 0). THE CROSS-BACKEND FP DISCIPLINE (from DH): the trilinear weights are
// POLYNOMIAL (floor + subtract + multiply, no transcendentals), but the per-corner products MUST use
// explicit mad — a plain a+b*c contracts to fma on Metal's fast-math but not on Vulkan/DXC or the CPU, a
// ~1-ULP divergence that would break the memcmp. The per-corner weight product (wx*wy*wz) and the
// per-coeff blend accumulation (w*coeff + acc) are both single correctly-rounded mads matching std::fma
// in the headers. THE probeCount=0 NO-OP PROOF: dimX==0 -> probeCount==0 -> the host dispatches 0 groups
// -> this main() never runs -> the output SSBO is untouched (== the cleared upload, all zero).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal they land at buffer(kCsStorage+0..2)):
//   b0 gProbeSH  : the per-probe ProbeSH[probeCount] input (9 SH coeffs x 3 channels), READ.
//   b1 gParams   : InterpParams — {dimX,dimY,dimZ,queryCount} + {originX,originY,originZ,spacing} + the
//                  flat query-point world positions (float4 per query, xyz used), READ.
//   b2 gOutSH    : the flat ProbeSH[queryCount] blended output (one blended SH per query), WRITE.
// (NO sampled textures — same style as froxel_inject / probe_sh_encode. The ProbeSH[] + query points are
//  flat SSBOs so the GPU and CPU read them identically.)
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL are the [[vk::binding]]
// decorations (same as froxel_inject / probe_sh_encode), not backend CODE symbols. spirv-cross maps the
// SPIR-V bindings to Metal's flat compute buffer indices so the SAME HLSL feeds Vulkan (DXC) and Metal
// (glslang->spirv-cross): bit-identical math.

#define HF_INTERP_THREADS 64
#define HF_MAX_INTERP_QUERIES 4096   // the fixed-cap query table (the showcase's deterministic query grid)

// One per-probe / per-query SH record (std430, 108 bytes): mirrors render::probesh::ProbeSH.
struct ProbeSH {
    float coeff[9][3];
};

// The interp params + the flat query-point table (std430). dims/queryCount drive the loop; origin/spacing
// place the grid; `queries` is the world position (xyz) of each query point (w unused).
struct InterpParams {
    uint4  dims;                          // x=dimX, y=dimY, z=dimZ, w=queryCount
    float4 grid;                          // x=originX, y=originY, z=originZ, w=spacing
    float4 queries[HF_MAX_INTERP_QUERIES];// per-query world position (xyz; w unused)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ProbeSH>      gProbeSH : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<InterpParams> gParams  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<ProbeSH>      gOutSH   : register(u2);

// cx-major flat index (VERBATIM probegi::ProbeGrid::flatIndex): idx = px + py*dimX + pz*(dimX*dimY).
int FlatIndex(int px, int py, int pz, int dimX, int dimY) {
    return px + py * dimX + pz * (dimX * dimY);
}

[numthreads(HF_INTERP_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int dimX = (int)gParams[0].dims.x;
    int dimY = (int)gParams[0].dims.y;
    int dimZ = (int)gParams[0].dims.z;
    uint queryCount = gParams[0].dims.w;
    float ox = gParams[0].grid.x, oy = gParams[0].grid.y, oz = gParams[0].grid.z;
    float spacing = gParams[0].grid.w;

    uint q = gid.x;
    if (q >= queryCount) return;

    float3 worldPos = gParams[0].queries[q].xyz;

    // Zeroed output accumulator.
    float acc[9][3];
    [unroll] for (int i = 0; i < 9; ++i) { acc[i][0] = 0.0; acc[i][1] = 0.0; acc[i][2] = 0.0; }

    // --- NearestProbes (VERBATIM probegi::NearestProbes): the FLOOR-cell 8 corners + trilinear weights. ---
    bool valid = (spacing > 0.0) && (dimX > 0) && (dimY > 0) && (dimZ > 0);
    if (valid) {
        // Per-axis FLOOR base + fractional position; a dim==1 axis collapses to base 0, frac 0.
        int bx, by, bz;
        float fx, fy, fz;
        // X axis.
        if (dimX <= 1) { bx = 0; fx = 0.0; }
        else {
            float g = (worldPos.x - ox) / spacing;
            float fb = floor(g);
            fb = clamp(fb, 0.0, (float)(dimX - 2));
            bx = (int)fb;
            fx = clamp(g - fb, 0.0, 1.0);
        }
        // Y axis.
        if (dimY <= 1) { by = 0; fy = 0.0; }
        else {
            float g = (worldPos.y - oy) / spacing;
            float fb = floor(g);
            fb = clamp(fb, 0.0, (float)(dimY - 2));
            by = (int)fb;
            fy = clamp(g - fb, 0.0, 1.0);
        }
        // Z axis.
        if (dimZ <= 1) { bz = 0; fz = 0.0; }
        else {
            float g = (worldPos.z - oz) / spacing;
            float fb = floor(g);
            fb = clamp(fb, 0.0, (float)(dimZ - 2));
            bz = (int)fb;
            fz = clamp(g - fb, 0.0, 1.0);
        }

        // The 8 corners: blend each corner's ProbeSH with the per-corner weight via mad (matching the
        // header's std::fma). The corner coordinate clamps to [0,dim-1] (a dim==1 axis's +offset corner
        // collapses onto the single valid index — weight 0 there). w = mad(wx*wy, wz, 0) (a single mad).
        [unroll] for (int c = 0; c < 8; ++c) {
            int sx = (c & 1), sy = ((c >> 1) & 1), sz = ((c >> 2) & 1);
            int cx = min(bx + sx, dimX - 1);
            int cy = min(by + sy, dimY - 1);
            int cz = min(bz + sz, dimZ - 1);
            int flat = FlatIndex(cx, cy, cz, dimX, dimY);
            float wx = sx ? fx : (1.0 - fx);
            float wy = sy ? fy : (1.0 - fy);
            float wz = sz ? fz : (1.0 - fz);
            float wc = mad(wx * wy, wz, 0.0);

            // --- InterpolateSH accumulation (VERBATIM probesh::InterpolateSH): per-corner per-coeff mad. ---
            [unroll] for (int j = 0; j < 9; ++j) {
                acc[j][0] = mad(wc, gProbeSH[flat].coeff[j][0], acc[j][0]);
                acc[j][1] = mad(wc, gProbeSH[flat].coeff[j][1], acc[j][1]);
                acc[j][2] = mad(wc, gProbeSH[flat].coeff[j][2], acc[j][2]);
            }
        }
    }
    // !valid -> acc stays zero (the documented disabled fallback == probesh::InterpolateSH's zero SH).

    [unroll] for (int o = 0; o < 9; ++o) {
        gOutSH[q].coeff[o][0] = acc[o][0];
        gOutSH[q].coeff[o][1] = acc[o][1];
        gOutSH[q].coeff[o][2] = acc[o][2];
    }
}
