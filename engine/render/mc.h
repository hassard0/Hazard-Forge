#pragma once
// Slice MC1 — GPU Isosurface Meshing Slice 1: per-cell MARCHING-CUBES CASE CLASSIFICATION (the
// BEACHHEAD of FLAGSHIP #5: GPU Isosurface Meshing via Marching Cubes). Pure CPU (header-only, no
// device, no backend symbols). Namespace hf::render::mc.
//
// WHAT THIS IS: the integer core of Marching Cubes — a scalar (voxel/SDF) field is classified, per
// CELL, into its 8-bit MC case index by comparing each of the cell's 8 cube CORNERS against an
// isovalue. The GPU compute pass (shaders/mc_classify.comp.hlsl) copies CaseIndex VERBATIM and proves
// bit-identical to this header's ClassifyCells (memcmp GPU==CPU, NO tolerance) — the VT1 / SW2 pattern
// applied to voxel-cell classification.
//
// THE CROSS-BACKEND CRUX (why GPU==CPU + Vulkan==Metal hold bit-exactly): the classification is PURE
// INTEGER compares of host-quantized int32 scalars (corner[i] > isovalue ? 1 : 0) — NO float, NO
// transcendental — and the per-cell write is order-independent (every cell's case is computed
// independently from its own 8 corners, so a GPU thread-race CANNOT change the result; the VT1
// set-write argument). Integer-in -> integer-out -> identical bits on every vendor.
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::/Backend::) symbols. NO GPU, NO new RHI. Mentions of
// "GPU" here are doc-only. INTEGER on every path (no float) so the case-index golden is cross-backend
// bit-identical (the strict zero-differing-pixel bar, like VT1 / swraster / VSM).

#include <cstdint>
#include <vector>

namespace hf::render::mc {

// ----- The voxel scalar field --------------------------------------------------------------------
// A host-filled INTEGER scalar field (a quantized SDF/density). scalar[(z*ny + y)*nx + x], flat
// row-major (the swraster vis-buffer / VoxelField layout). nx/ny/nz = the CORNER counts per axis;
// cells span [0, nx-1) x [0, ny-1) x [0, nz-1), so cellCount() = (nx-1)*(ny-1)*(nz-1). A cell (cx,cy,cz)
// owns the 8 corners (cx,cy,cz)+kCornerOffset[i]. The flat cell id is
//   cellId = (cz*(ny-1) + cy)*(nx-1) + cx.
struct VoxelField {
    int nx = 0, ny = 0, nz = 0;       // CORNER counts per axis
    std::vector<int32_t> scalar;      // size nx*ny*nz, scalar[(z*ny + y)*nx + x]

    int cornerCount() const { return nx * ny * nz; }
    int cellCount() const {
        if (nx < 2 || ny < 2 || nz < 2) return 0;
        return (nx - 1) * (ny - 1) * (nz - 1);
    }
};

// ----- The LOCKED canonical Marching-Cubes corner numbering 0..7 ---------------------------------
// Corner i sits at the cube-local offset kCornerOffset[i] = (dx,dy,dz), dx/dy/dz in {0,1}. This is the
// STANDARD MC ordering (the Lorensen-Cline / Bourke "Polygonising a Scalar Field" convention) — the
// bottom face (z=0) wound CCW corners 0..3, then the top face (z=1) corners 4..7 directly above:
//   0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)   (bottom, z=0)
//   4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)   (top,    z=1)
// MC2's 256-entry triangle table MUST use THIS exact numbering (so caseIndex bit i <-> corner i).
static constexpr int kCornerOffset[8][3] = {
    {0, 0, 0},  // 0
    {1, 0, 0},  // 1
    {1, 1, 0},  // 2
    {0, 1, 0},  // 3
    {0, 0, 1},  // 4
    {1, 0, 1},  // 5
    {1, 1, 1},  // 6
    {0, 1, 1},  // 7
};

// MC2: the canonical edge numbering 0..11 that the 256-case triangle table indexes (pinned here so the
// table this header gains in MC2 matches the corner numbering above). Edge e connects two corners:
//   bottom ring (z=0): 0:(0-1) 1:(1-2) 2:(2-3) 3:(3-0)
//   top    ring (z=1): 4:(4-5) 5:(5-6) 6:(6-7) 7:(7-4)
//   verticals:         8:(0-4) 9:(1-5) 10:(2-6) 11:(3-7)
// MC2: static constexpr int kEdgeCorner[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},
//                                                 {0,4},{1,5},{2,6},{3,7}};
// MC2: a forward-compatible stub for the 256-case topology table (NOT implemented this slice):
//   using TriTable = int8_t[256][16];  // -1-terminated edge triplets per case
//   int CaseTriangleCount(uint8_t caseIdx);  // popcount of the case's emitted triangles

// ----- The case index (the cross-backend crux) ---------------------------------------------------
// idx = Σ_{i<8} ((corner[i] > isovalue) ? 1 : 0) << i. Pure integer compare. CONVENTION (pinned; MC2's
// tables MUST agree): a corner is "inside" the surface when its scalar is STRICTLY GREATER than the
// isovalue (scalar > isovalue). So all-corners-below -> 0x00, all-above -> 0xFF, only corner 0 above ->
// 0x01. THE cross-backend crux: identical on CPU/Vulkan/Metal by construction (no float, no
// transcendental). The GPU shader copies this body VERBATIM.
inline uint8_t CaseIndex(const int32_t corner[8], int32_t isovalue) {
    uint8_t idx = 0;
    for (int i = 0; i < 8; ++i)
        if (corner[i] > isovalue) idx |= (uint8_t)(1u << i);
    return idx;
}

// ----- The host integer field accessor -----------------------------------------------------------
// Bounds-checked / clamped sample (clamp-to-edge so a corner gather at the field boundary is well
// defined; cells never read past nx-1/ny-1/nz-1 anyway, but clamping keeps the accessor UB-free).
inline int32_t SampleField(const VoxelField& f, int x, int y, int z) {
    if (x < 0) x = 0; else if (x > f.nx - 1) x = f.nx - 1;
    if (y < 0) y = 0; else if (y > f.ny - 1) y = f.ny - 1;
    if (z < 0) z = 0; else if (z > f.nz - 1) z = f.nz - 1;
    return f.scalar[(size_t)(z * f.ny + y) * f.nx + x];
}

// ----- The CPU reference classifier --------------------------------------------------------------
// caseOut sized cellCount(); for each cell (cx,cy,cz) gather its 8 corner scalars (SampleField at the 8
// kCornerOffset), caseOut[cellId] = CaseIndex(corners, isovalue), cellId = (cz*(ny-1)+cy)*(nx-1)+cx.
// Order-independent -> a GPU thread-race CANNOT change it (the VT1 set-write argument). This is the
// reference the GPU mc_classify.comp memcmp's against.
inline void ClassifyCells(const VoxelField& f, int32_t isovalue, std::vector<uint8_t>& caseOut) {
    const int cxN = f.nx - 1, cyN = f.ny - 1, czN = f.nz - 1;
    caseOut.assign((size_t)f.cellCount(), 0u);
    for (int cz = 0; cz < czN; ++cz)
        for (int cy = 0; cy < cyN; ++cy)
            for (int cx = 0; cx < cxN; ++cx) {
                int32_t corner[8];
                for (int i = 0; i < 8; ++i)
                    corner[i] = SampleField(f, cx + kCornerOffset[i][0],
                                            cy + kCornerOffset[i][1],
                                            cz + kCornerOffset[i][2]);
                const int cellId = (cz * cyN + cy) * cxN + cx;
                caseOut[(size_t)cellId] = CaseIndex(corner, isovalue);
            }
}

// A cell is a SURFACE cell iff the isosurface crosses it, i.e. its case is neither all-out (0x00) nor
// all-in (0xFF). Pure helper for the {...} stat line.
inline bool IsSurfaceCase(uint8_t caseIdx) { return caseIdx != 0x00 && caseIdx != 0xFF; }

// ----- The deterministic showcase SDF generator (the heightmap.h discipline) ---------------------
// MakeSphereField(n, radiusCells): an n³-CORNER field (n³ corners -> (n-1)³ cells) holding the QUANTIZED
// SIGNED DISTANCE to a sphere of radius `radiusCells` centred at the field centre, in FIXED-POINT
// (kFixed units per cell). scalar = (radiusCells*kFixed - distFromCentre*kFixed), so scalar > 0 INSIDE
// the sphere and < 0 outside -> isovalue 0 places the surface exactly on the sphere shell. The distance
// is computed in INTEGER fixed-point via an integer isqrt of the squared distance (NO float, NO
// transcendental) so the field — and therefore every classified case — is bit-identical cross-target
// (the heightmap.h::Height pure-function determinism, kept integer for the cross-backend golden).
static constexpr int32_t kFixed = 256;  // fixed-point scale (8 fractional bits) for the SDF

// Deterministic integer square root of a non-negative int64 (floor(sqrt(v))). Pure integer (binary
// digit-by-digit) — identical on every compiler/vendor, no <cmath>.
inline int64_t ISqrt(int64_t v) {
    if (v <= 0) return 0;
    int64_t bit = (int64_t)1 << 62;
    while (bit > v) bit >>= 2;
    int64_t res = 0;
    while (bit != 0) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else { res >>= 1; }
        bit >>= 2;
    }
    return res;
}

inline VoxelField MakeSphereField(int n, int radiusCells) {
    VoxelField f;
    f.nx = f.ny = f.nz = n;
    f.scalar.assign((size_t)n * n * n, 0);
    // Centre at the field's middle corner (in fixed-point); distances in fixed-point cell units.
    const int64_t cFx = (int64_t)(n - 1) * kFixed / 2;          // centre coord * kFixed
    const int64_t rFx = (int64_t)radiusCells * kFixed;          // radius * kFixed
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const int64_t dx = (int64_t)x * kFixed - cFx;
                const int64_t dy = (int64_t)y * kFixed - cFx;
                const int64_t dz = (int64_t)z * kFixed - cFx;
                const int64_t d2 = dx * dx + dy * dy + dz * dz;  // (dist*kFixed)²
                const int64_t distFx = ISqrt(d2);               // dist * kFixed (floor)
                // scalar = (radius - dist) * kFixed -> >0 inside, <0 outside; surface at scalar==0.
                f.scalar[(size_t)(z * n + y) * n + x] = (int32_t)(rFx - distFx);
            }
    return f;
}

}  // namespace hf::render::mc
