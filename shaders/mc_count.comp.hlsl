// Slice MC2 — GPU Isosurface Meshing Slice 2: the per-cell MARCHING-CUBES TRIANGLE-COUNT compute pass.
// ONE thread per CELL (cell < cellCount). Each thread decomposes its flat cell id -> (cx,cy,cz), gathers
// the 8 cube-CORNER scalars from gField (b0) at the SAME canonical kCornerOffset render/mc.h pins, runs
// CaseIndex copied VERBATIM from render/mc.h (idx = Σ (corner[i] > isovalue) << i), looks up
// triCount = gTriCount[caseIdx] (b1, the host-uploaded 256-entry per-case count table built from
// render/mc.h::kTriTable), writes gCounts[cell] = triCount (b2), and InterlockedAdd(gTotal[0], triCount)
// (b3, a single-uint grand total).
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the per-cell case + table-lookup are PURE
// INTEGER — the GPU consumes the SAME int32 host-quantized scalars gField holds + the SAME count table
// the host built from kTriTable. The per-cell WRITE (gCounts[cell]) is order-independent (each cell's
// count depends only on its own 8 corners), and the only atomic — InterlockedAdd of integer counts into
// gTotal — is COMMUTATIVE + associative, so the grand total is INDEPENDENT of the thread execution order
// -> bit-deterministic GPU==CPU + cross-backend (the autoexposure_histogram.comp InterlockedAdd(...,1u)
// argument; integer atomic add, NOT a float atomic). A divergence vs the header is exactly what the host's
// GPU==CPU memcmp (counts) + total compare catch.
//
// countEnabled push/param flag: 0 -> every thread writes gCounts[cell]=0 and does NO atomic add (the
// disabled no-op -> counts all-zero, total 0, byte-identical to the cleared upload).
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gField    : the int32 host-quantized scalar field, scalar[(z*ny+y)*nx+x], READ.
//   b1 gTriCount : the 256-entry per-case triangle-count table (uint per case), READ.
//   b2 gCounts   : one uint per cell (the triangle count), WRITE.
//   b3 gTotal    : a SINGLE uint grand total, InterlockedAdd WRITE (cleared to 0 by the host).
//   b4 gParams   : { nx, ny, nz, isovalue } + { countEnabled, cellCount, _, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations (same
// as mc_classify.comp / autoexposure_histogram.comp), not backend CODE symbols. Plain integer SSBO +
// integer InterlockedAdd -> the default MSL gen suffices (NO --msl-version 20200 — the mc_classify lesson;
// the integer atomic lowers on Metal with no MSL-2.2 feature).

#define HF_MC_THREADS 64

// Params (std430). Mirrors the C++ upload struct.
//   dims  : x=nx, y=ny, z=nz, w=isovalue (CORNER counts; cells span [0,nx-1)x[0,ny-1)x[0,nz-1))
//   cfg   : x=countEnabled, y=cellCount, z=unused, w=unused
struct Params {
    int4 dims;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<int>    gField    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gTriCount : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>   gCounts   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>   gTotal    : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<Params> gParams   : register(u4);

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
    int isovalue     = gParams[0].dims.w;
    int countEnabled = gParams[0].cfg.x;
    int cellCount    = gParams[0].cfg.y;

    uint cell = gid.x;
    if ((int)cell >= cellCount) return;

    // Disabled -> write 0, NO atomic add (gCounts stays the cleared all-zero upload, gTotal stays 0;
    // the byte-identical no-op proof).
    if (countEnabled == 0) { gCounts[cell] = 0u; return; }

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

    uint caseIdx  = CaseIndex(corner, isovalue);
    uint triCount = gTriCount[caseIdx];       // host-uploaded count table (built from kTriTable)

    gCounts[cell] = triCount;                  // order-independent per-cell integer write

    // INTEGER atomic add -> order-independent grand total -> deterministic (autoexposure precedent).
    InterlockedAdd(gTotal[0], triCount);
}
