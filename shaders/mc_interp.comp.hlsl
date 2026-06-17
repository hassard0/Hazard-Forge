// Slice MC4 — GPU Isosurface Meshing Slice 4: the per-cell TRIANGLE EMISSION pass with FIXED-POINT
// INTERPOLATED vertex placement — the QUALITY step that makes the extracted surface smooth. This is
// MC3's mc_emit.comp with EdgeMidpoint replaced by EdgeInterp (the fixed-point isosurface-crossing
// interpolation, copied VERBATIM from render/mc.h). ONE thread per CELL (cell < cellCount). Each thread
// RECLASSIFIES its cell from gField via CaseIndex (VERBATIM render/mc.h), reads its write offset
// triOffset = gOffsets[cell] (the MC3 prefix-sum, REUSED UNCHANGED), and walks gTriTable[caseIndex] in
// groups of 3, EMITTING each triangle's 3 INTERPOLATED edge-crossing vertices into gVerts + the identity
// index into gIdx at slot 3*(triOffset+t)+k. Each cell writes into its own DISJOINT range -> race-free,
// NO atomics. A meshEnabled=0 flag -> emit nothing (the disabled no-op).
//
// WHY BIT-IDENTICAL to the CPU MarchCellsInterp (the make-or-break): every value written is PURE INTEGER
// — the case is the SAME int-compare CaseIndex over the SAME host-quantized gField, the interpolation is
// the SAME fixed-point t = ((iso - s0) * kSub) / (s1 - s0) using the SAME TRUNCATING integer divide (the
// C++/HLSL `/` default — truncation toward zero — is consistent across backends), the vertex coord is
// Ca*kSub + (Cb - Ca)*t on the 1/kSub lattice, the index is the identity slot, and each cell's range is
// disjoint so a thread race CANNOT change any byte. INT32 throughout: the numerator (iso - s0)*kSub fits
// int32 for this field (|scalar| ~ field*256, ×256 ~ 2^22 << 2^31), so NO int64 (no glslc/SPIR-V int64
// issue, no --msl-version 20200 — the mc_classify/mc_count/mc_emit lesson). A divergence vs the header is
// exactly what the host's GPU==CPU memcmp (gVerts + gIdx) catches.
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these land at buffer(0..5)) — IDENTICAL to
// mc_emit.comp's binding layout, so the host wiring is the same:
//   b0 gField    : the int32 host-quantized scalar field, scalar[(z*ny+y)*nx+x], READ.
//   b1 gOffsets  : one uint per cell (the MC3 exclusive prefix-sum write offset), READ.
//   b2 gTriTable : the 256x16 canonical triangle table (int per entry; -1 terminator), READ.
//   b3 gVerts    : the output vertex buffer, McVertex (int4) per slot, WRITE (pre-cleared).
//   b4 gIdx      : the output index buffer, one uint per slot, WRITE (pre-cleared).
//   b5 gParams   : { nx, ny, nz, isovalue } + { meshEnabled, cellCount, _, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations. Plain
// integer SSBO writes -> the default MSL gen suffices (NO --msl-version 20200).

#define HF_MC_THREADS 64

// The fixed-point edge lattice (render/mc.h::kSub): a vertex coord is in 1/kSub grid units, midpoint =
// kSub/2, t in [0,kSub].
static const int kSub = 256;

// McVertex mirror (render/mc.h::McVertex): now an INTERPOLATED edge crossing in 1/kSub units, w=0 pad.
struct McVertex {
    int x, y, z, w;
};

// Params (std430). Mirrors the C++ upload struct.
//   dims : x=nx, y=ny, z=nz, w=isovalue (CORNER counts; cells span [0,nx-1)x[0,ny-1)x[0,nz-1))
//   cfg  : x=meshEnabled, y=cellCount, z=unused, w=unused
struct Params {
    int4 dims;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<int>      gField    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>     gOffsets  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<int>      gTriTable : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<McVertex> gVerts    : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>     gIdx      : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<Params>   gParams   : register(u5);

// The canonical MC corner offsets — VERBATIM render/mc.h::kCornerOffset (0=(0,0,0)..7=(0,1,1)).
static const int3 kCornerOffset[8] = {
    int3(0, 0, 0), int3(1, 0, 0), int3(1, 1, 0), int3(0, 1, 0),
    int3(0, 0, 1), int3(1, 0, 1), int3(1, 1, 1), int3(0, 1, 1),
};

// The canonical MC edge->corner pairs — VERBATIM render/mc.h::kEdgeCorner (edge e connects (a,b)).
static const int2 kEdgeCorner[12] = {
    int2(0, 1), int2(1, 2), int2(2, 3), int2(3, 0),   // bottom ring (z=0)
    int2(4, 5), int2(5, 6), int2(6, 7), int2(7, 4),   // top ring    (z=1)
    int2(0, 4), int2(1, 5), int2(2, 6), int2(3, 7),   // verticals
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

// EdgeInterp(cell, edge) -> the INTERPOLATED isosurface crossing in 1/kSub units. The fixed-point t +
// clamp + the Ca*kSub + (Cb-Ca)*t placement are copied VERBATIM from render/mc.h::EdgeInterp. The two
// edge-corner scalars are read from gField (already bound). Pure INT32 (NO float, NO int64): the SAME
// TRUNCATING integer divide as the CPU header -> bit-identical.
McVertex EdgeInterp(int cx, int cy, int cz, int edge, int isovalue, int nx, int ny, int nz) {
    int a = kEdgeCorner[edge].x;
    int b = kEdgeCorner[edge].y;
    int3 Ca = int3(cx, cy, cz) + kCornerOffset[a];
    int3 Cb = int3(cx, cy, cz) + kCornerOffset[b];
    int s0 = SampleField(Ca.x, Ca.y, Ca.z, nx, ny, nz);
    int s1 = SampleField(Cb.x, Cb.y, Cb.z, nx, ny, nz);

    int t;
    if (s1 == s0) {
        t = kSub / 2;                              // degenerate edge -> the deterministic midpoint
    } else {
        t = ((isovalue - s0) * kSub) / (s1 - s0);  // truncating integer divide (C++/HLSL identical)
        if (t < 0) t = 0; else if (t > kSub) t = kSub;
    }

    McVertex v;
    v.x = Ca.x * kSub + (Cb.x - Ca.x) * t;
    v.y = Ca.y * kSub + (Cb.y - Ca.y) * t;
    v.z = Ca.z * kSub + (Cb.z - Ca.z) * t;
    v.w = 0;
    return v;
}

[numthreads(HF_MC_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int nx = gParams[0].dims.x;
    int ny = gParams[0].dims.y;
    int nz = gParams[0].dims.z;
    int isovalue    = gParams[0].dims.w;
    int meshEnabled = gParams[0].cfg.x;
    int cellCount   = gParams[0].cfg.y;

    uint cell = gid.x;
    if ((int)cell >= cellCount) return;

    // Disabled -> emit nothing (gVerts/gIdx stay the pre-cleared upload; the byte-identical no-op).
    if (meshEnabled == 0) return;

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

    uint caseIdx   = CaseIndex(corner, isovalue);
    uint triOffset = gOffsets[cell];   // this cell's disjoint write base (from the prefix-sum)

    // Walk gTriTable[caseIdx] in groups of 3 (until -1), emit each triangle's 3 INTERPOLATED verts +
    // identity indices at slot 3*(triOffset+t)+k. The EmitCellInterp walk copied VERBATIM from render/mc.h.
    uint t = 0u;
    for (int e = 0; e + 2 < 16; e += 3) {
        int e0 = gTriTable[caseIdx * 16u + (uint)e];
        if (e0 < 0) break;             // -1 terminator -> done
        [unroll] for (int k = 0; k < 3; ++k) {
            int edge = gTriTable[caseIdx * 16u + (uint)(e + k)];
            uint slot = 3u * (triOffset + t) + (uint)k;
            gVerts[slot] = EdgeInterp(cx, cy, cz, edge, isovalue, nx, ny, nz);
            gIdx[slot] = slot;         // identity index (triangle soup)
        }
        ++t;
    }
}
