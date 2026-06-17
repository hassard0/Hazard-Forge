// Slice MC1 — GPU Isosurface Meshing Slice 1: the per-cell MARCHING-CUBES CASE CLASSIFICATION compute
// pass. ONE thread per CELL (cell < cellCount). Each thread decomposes its flat cell id -> (cx,cy,cz),
// gathers the 8 cube-CORNER scalars from gField (b0) at the SAME canonical kCornerOffset render/mc.h
// pins, runs CaseIndex copied VERBATIM from render/mc.h (idx = Σ (corner[i] > isovalue) << i), and
// writes gCases[cell] = caseIdx (b1). The per-cell case is computed independently from that cell's own
// 8 corners, so the write is ORDER-INDEPENDENT — race-free, NO atomics, bit-identical GPU==CPU +
// cross-backend (the VT1 set-write / SW2 integer-replay argument applied to voxel-cell classification).
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it
// consumes the SAME int32 host-quantized scalars gField holds (the host filled the SDF via the integer
// render/mc.h::MakeSphereField) and runs the SAME pure-integer corner-sign compares CaseIndex runs. A
// divergence here vs the header is exactly what the host's GPU==CPU memcmp catches.
//
// classifyEnabled push/param flag: 0 -> every thread writes 0 (the disabled-path no-op; gCases stays
// the cleared all-zero upload, byte-identical).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gField  : the int32 host-quantized scalar field, scalar[(z*ny+y)*nx+x], READ.
//   b1 gCases  : one uint per cell (the uint8 case packed into a uint slot), WRITE.
//   b2 gParams : { nx, ny, nz, isovalue, classifyEnabled, _,_,_ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations
// (same as vsm_mark.comp / vt_feedback.comp), not backend CODE symbols. Plain integer SSBO write ->
// the default MSL gen suffices (NO --msl-version 20200 — the vsm_mark/vt_feedback lesson).

#define HF_MC_THREADS 64

// Params (std430). Mirrors the C++ upload struct.
//   dims  : x=nx, y=ny, z=nz, w=unused (CORNER counts; cells span [0,nx-1)x[0,ny-1)x[0,nz-1))
//   cfg   : x=isovalue, y=classifyEnabled, z=cellCount, w=unused
struct Params {
    int4 dims;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<int>    gField  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gCases  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams : register(u2);

// The canonical MC corner offsets — VERBATIM render/mc.h::kCornerOffset (0=(0,0,0)..7=(0,1,1)).
static const int3 kCornerOffset[8] = {
    int3(0, 0, 0), int3(1, 0, 0), int3(1, 1, 0), int3(0, 1, 0),
    int3(0, 0, 1), int3(1, 0, 1), int3(1, 1, 1), int3(0, 1, 1),
};

// Clamp-to-edge field accessor — VERBATIM render/mc.h::SampleField.
int SampleField(int x, int y, int z, int nx, int ny, int nz) {
    if (x < 0) x = 0; else if (x > nx - 1) x = nx - 1;
    if (y < 0) y = 0; else if (y > ny - 1) y = ny - 1;
    if (z < 0) z = 0; else if (z > nz - 1) z = nz - 1;
    return gField[(uint)((z * ny + y) * nx + x)];
}

// idx = Σ_{i<8} ((corner[i] > isovalue) ? 1 : 0) << i — VERBATIM render/mc.h::CaseIndex. Pure integer.
uint CaseIndex(int corner[8], int isovalue) {
    uint idx = 0u;
    [unroll] for (int i = 0; i < 8; ++i)
        if (corner[i] > isovalue) idx |= (1u << (uint)i);
    return idx;
}

[numthreads(HF_MC_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int nx = gParams[0].dims.x;
    int ny = gParams[0].dims.y;
    int nz = gParams[0].dims.z;
    int isovalue        = gParams[0].cfg.x;
    int classifyEnabled = gParams[0].cfg.y;
    int cellCount       = gParams[0].cfg.z;

    uint cell = gid.x;
    if ((int)cell >= cellCount) return;

    // Disabled -> write 0 (gCases stays the cleared all-zero upload; the byte-identical no-op proof).
    if (classifyEnabled == 0) { gCases[cell] = 0u; return; }

    // Decompose cell -> (cx,cy,cz). cellId = (cz*(ny-1) + cy)*(nx-1) + cx (VERBATIM render/mc.h).
    int cxN = nx - 1, cyN = ny - 1;
    int cx =  (int)cell % cxN;
    int cy = ((int)cell / cxN) % cyN;
    int cz =  (int)cell / (cxN * cyN);

    // Gather the 8 corners at the SAME kCornerOffset, run CaseIndex VERBATIM.
    int corner[8];
    [unroll] for (int i = 0; i < 8; ++i)
        corner[i] = SampleField(cx + kCornerOffset[i].x,
                                cy + kCornerOffset[i].y,
                                cz + kCornerOffset[i].z, nx, ny, nz);

    gCases[cell] = CaseIndex(corner, isovalue);
}
