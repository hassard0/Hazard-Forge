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

#include <cmath>
#include <cstdint>
#include <span>
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
// table below matches the corner numbering above). Edge e connects two corners:
//   bottom ring (z=0): 0:(0-1) 1:(1-2) 2:(2-3) 3:(3-0)
//   top    ring (z=1): 4:(4-5) 5:(5-6) 6:(6-7) 7:(7-4)
//   verticals:         8:(0-4) 9:(1-5) 10:(2-6) 11:(3-7)
static constexpr int kEdgeCorner[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},   // bottom ring (z=0)
    {4, 5}, {5, 6}, {6, 7}, {7, 4},   // top ring    (z=1)
    {0, 4}, {1, 5}, {2, 6}, {3, 7},   // verticals
};

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

// ----- MC2: the canonical 256-case Marching-Cubes TRIANGLE TABLE -----------------------------------
// kTriTable[caseIndex] lists the triangles the isosurface emits inside a cell of that 8-bit case, as
// EDGE indices (0..11, the kEdgeCorner numbering above) in groups of 3 (one triangle per triplet),
// terminated by -1. Each emitted vertex lies on the named cube edge (MC3 interpolates the exact
// position; MC2 only needs how MANY triangles, derived below).
//
// SOURCE + CONVENTION (auditable): this is the public-domain canonical Marching-Cubes triangle table
// from Paul Bourke, "Polygonising a Scalar Field" (http://paulbourke.net/geometry/polygonise/),
// originally Lorensen & Cline (SIGGRAPH 1987). Bourke's corner numbering is EXACTLY MC1's locked
// kCornerOffset (0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0) 4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)) and
// Bourke's edge numbering is EXACTLY kEdgeCorner above (0:(0-1)..3:(3-0) bottom ring, 4:(4-5)..7:(7-4)
// top ring, 8:(0-4)..11:(3-7) verticals). The "inside" bit convention also matches MC1's CaseIndex
// (bit i set <-> corner i scalar > isovalue). Therefore the table is used VERBATIM with NO remap; the
// MC2 self-consistency + well-formedness unit tests are the guard that this alignment holds (every row
// is groups-of-3-then-(-1), every edge index in [0,11], case 0x00 and 0xFF emit zero triangles).
static constexpr int8_t kTriTable[256][16] = {
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 1, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 8, 3, 9, 8, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3, 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 2,10, 0, 2, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 2, 8, 3, 2,10, 8,10, 9, 8,-1,-1,-1,-1,-1,-1,-1},
    { 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0,11, 2, 8,11, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 9, 0, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1,11, 2, 1, 9,11, 9, 8,11,-1,-1,-1,-1,-1,-1,-1},
    { 3,10, 1,11,10, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0,10, 1, 0, 8,10, 8,11,10,-1,-1,-1,-1,-1,-1,-1},
    { 3, 9, 0, 3,11, 9,11,10, 9,-1,-1,-1,-1,-1,-1,-1},
    { 9, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 3, 0, 7, 3, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 1, 9, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 1, 9, 4, 7, 1, 7, 3, 1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 4, 7, 3, 0, 4, 1, 2,10,-1,-1,-1,-1,-1,-1,-1},
    { 9, 2,10, 9, 0, 2, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1},
    { 2,10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4,-1,-1,-1,-1},
    { 8, 4, 7, 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11, 4, 7,11, 2, 4, 2, 0, 4,-1,-1,-1,-1,-1,-1,-1},
    { 9, 0, 1, 8, 4, 7, 2, 3,11,-1,-1,-1,-1,-1,-1,-1},
    { 4, 7,11, 9, 4,11, 9,11, 2, 9, 2, 1,-1,-1,-1,-1},
    { 3,10, 1, 3,11,10, 7, 8, 4,-1,-1,-1,-1,-1,-1,-1},
    { 1,11,10, 1, 4,11, 1, 0, 4, 7,11, 4,-1,-1,-1,-1},
    { 4, 7, 8, 9, 0,11, 9,11,10,11, 0, 3,-1,-1,-1,-1},
    { 4, 7,11, 4,11, 9, 9,11,10,-1,-1,-1,-1,-1,-1,-1},
    { 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 5, 4, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 5, 4, 1, 5, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 8, 5, 4, 8, 3, 5, 3, 1, 5,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 0, 8, 1, 2,10, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1},
    { 5, 2,10, 5, 4, 2, 4, 0, 2,-1,-1,-1,-1,-1,-1,-1},
    { 2,10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8,-1,-1,-1,-1},
    { 9, 5, 4, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0,11, 2, 0, 8,11, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1},
    { 0, 5, 4, 0, 1, 5, 2, 3,11,-1,-1,-1,-1,-1,-1,-1},
    { 2, 1, 5, 2, 5, 8, 2, 8,11, 4, 8, 5,-1,-1,-1,-1},
    {10, 3,11,10, 1, 3, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1},
    { 4, 9, 5, 0, 8, 1, 8,10, 1, 8,11,10,-1,-1,-1,-1},
    { 5, 4, 0, 5, 0,11, 5,11,10,11, 0, 3,-1,-1,-1,-1},
    { 5, 4, 8, 5, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1},
    { 9, 7, 8, 5, 7, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 3, 0, 9, 5, 3, 5, 7, 3,-1,-1,-1,-1,-1,-1,-1},
    { 0, 7, 8, 0, 1, 7, 1, 5, 7,-1,-1,-1,-1,-1,-1,-1},
    { 1, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 7, 8, 9, 5, 7,10, 1, 2,-1,-1,-1,-1,-1,-1,-1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3,-1,-1,-1,-1},
    { 8, 0, 2, 8, 2, 5, 8, 5, 7,10, 5, 2,-1,-1,-1,-1},
    { 2,10, 5, 2, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1},
    { 7, 9, 5, 7, 8, 9, 3,11, 2,-1,-1,-1,-1,-1,-1,-1},
    { 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,11,-1,-1,-1,-1},
    { 2, 3,11, 0, 1, 8, 1, 7, 8, 1, 5, 7,-1,-1,-1,-1},
    {11, 2, 1,11, 1, 7, 7, 1, 5,-1,-1,-1,-1,-1,-1,-1},
    { 9, 5, 8, 8, 5, 7,10, 1, 3,10, 3,11,-1,-1,-1,-1},
    { 5, 7, 0, 5, 0, 9, 7,11, 0, 1, 0,10,11,10, 0,-1},
    {11,10, 0,11, 0, 3,10, 5, 0, 8, 0, 7, 5, 7, 0,-1},
    {11,10, 5, 7,11, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 0, 1, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 8, 3, 1, 9, 8, 5,10, 6,-1,-1,-1,-1,-1,-1,-1},
    { 1, 6, 5, 2, 6, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 6, 5, 1, 2, 6, 3, 0, 8,-1,-1,-1,-1,-1,-1,-1},
    { 9, 6, 5, 9, 0, 6, 0, 2, 6,-1,-1,-1,-1,-1,-1,-1},
    { 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8,-1,-1,-1,-1},
    { 2, 3,11,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11, 0, 8,11, 2, 0,10, 6, 5,-1,-1,-1,-1,-1,-1,-1},
    { 0, 1, 9, 2, 3,11, 5,10, 6,-1,-1,-1,-1,-1,-1,-1},
    { 5,10, 6, 1, 9, 2, 9,11, 2, 9, 8,11,-1,-1,-1,-1},
    { 6, 3,11, 6, 5, 3, 5, 1, 3,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8,11, 0,11, 5, 0, 5, 1, 5,11, 6,-1,-1,-1,-1},
    { 3,11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9,-1,-1,-1,-1},
    { 6, 5, 9, 6, 9,11,11, 9, 8,-1,-1,-1,-1,-1,-1,-1},
    { 5,10, 6, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 3, 0, 4, 7, 3, 6, 5,10,-1,-1,-1,-1,-1,-1,-1},
    { 1, 9, 0, 5,10, 6, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4,-1,-1,-1,-1},
    { 6, 1, 2, 6, 5, 1, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7,-1,-1,-1,-1},
    { 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,-1,-1,-1,-1},
    { 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,-1},
    { 3,11, 2, 7, 8, 4,10, 6, 5,-1,-1,-1,-1,-1,-1,-1},
    { 5,10, 6, 4, 7, 2, 4, 2, 0, 2, 7,11,-1,-1,-1,-1},
    { 0, 1, 9, 4, 7, 8, 2, 3,11, 5,10, 6,-1,-1,-1,-1},
    { 9, 2, 1, 9,11, 2, 9, 4,11, 7,11, 4, 5,10, 6,-1},
    { 8, 4, 7, 3,11, 5, 3, 5, 1, 5,11, 6,-1,-1,-1,-1},
    { 5, 1,11, 5,11, 6, 1, 0,11, 7,11, 4, 0, 4,11,-1},
    { 0, 5, 9, 0, 6, 5, 0, 3, 6,11, 6, 3, 8, 4, 7,-1},
    { 6, 5, 9, 6, 9,11, 4, 7, 9, 7,11, 9,-1,-1,-1,-1},
    {10, 4, 9, 6, 4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4,10, 6, 4, 9,10, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1},
    {10, 0, 1,10, 6, 0, 6, 4, 0,-1,-1,-1,-1,-1,-1,-1},
    { 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1,10,-1,-1,-1,-1},
    { 1, 4, 9, 1, 2, 4, 2, 6, 4,-1,-1,-1,-1,-1,-1,-1},
    { 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4,-1,-1,-1,-1},
    { 0, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 8, 3, 2, 8, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1},
    {10, 4, 9,10, 6, 4,11, 2, 3,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 2, 2, 8,11, 4, 9,10, 4,10, 6,-1,-1,-1,-1},
    { 3,11, 2, 0, 1, 6, 0, 6, 4, 6, 1,10,-1,-1,-1,-1},
    { 6, 4, 1, 6, 1,10, 4, 8, 1, 2, 1,11, 8,11, 1,-1},
    { 9, 6, 4, 9, 3, 6, 9, 1, 3,11, 6, 3,-1,-1,-1,-1},
    { 8,11, 1, 8, 1, 0,11, 6, 1, 9, 1, 4, 6, 4, 1,-1},
    { 3,11, 6, 3, 6, 0, 0, 6, 4,-1,-1,-1,-1,-1,-1,-1},
    { 6, 4, 8,11, 6, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 7,10, 6, 7, 8,10, 8, 9,10,-1,-1,-1,-1,-1,-1,-1},
    { 0, 7, 3, 0,10, 7, 0, 9,10, 6, 7,10,-1,-1,-1,-1},
    {10, 6, 7, 1,10, 7, 1, 7, 8, 1, 8, 0,-1,-1,-1,-1},
    {10, 6, 7,10, 7, 1, 1, 7, 3,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,-1,-1,-1,-1},
    { 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,-1},
    { 7, 8, 0, 7, 0, 6, 6, 0, 2,-1,-1,-1,-1,-1,-1,-1},
    { 7, 3, 2, 6, 7, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 2, 3,11,10, 6, 8,10, 8, 9, 8, 6, 7,-1,-1,-1,-1},
    { 2, 0, 7, 2, 7,11, 0, 9, 7, 6, 7,10, 9,10, 7,-1},
    { 1, 8, 0, 1, 7, 8, 1,10, 7, 6, 7,10, 2, 3,11,-1},
    {11, 2, 1,11, 1, 7,10, 6, 1, 6, 7, 1,-1,-1,-1,-1},
    { 8, 9, 6, 8, 6, 7, 9, 1, 6,11, 6, 3, 1, 3, 6,-1},
    { 0, 9, 1,11, 6, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 7, 8, 0, 7, 0, 6, 3,11, 0,11, 6, 0,-1,-1,-1,-1},
    { 7,11, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 0, 8,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 1, 9,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 8, 1, 9, 8, 3, 1,11, 7, 6,-1,-1,-1,-1,-1,-1,-1},
    {10, 1, 2, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10, 3, 0, 8, 6,11, 7,-1,-1,-1,-1,-1,-1,-1},
    { 2, 9, 0, 2,10, 9, 6,11, 7,-1,-1,-1,-1,-1,-1,-1},
    { 6,11, 7, 2,10, 3,10, 8, 3,10, 9, 8,-1,-1,-1,-1},
    { 7, 2, 3, 6, 2, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 7, 0, 8, 7, 6, 0, 6, 2, 0,-1,-1,-1,-1,-1,-1,-1},
    { 2, 7, 6, 2, 3, 7, 0, 1, 9,-1,-1,-1,-1,-1,-1,-1},
    { 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6,-1,-1,-1,-1},
    {10, 7, 6,10, 1, 7, 1, 3, 7,-1,-1,-1,-1,-1,-1,-1},
    {10, 7, 6, 1, 7,10, 1, 8, 7, 1, 0, 8,-1,-1,-1,-1},
    { 0, 3, 7, 0, 7,10, 0,10, 9, 6,10, 7,-1,-1,-1,-1},
    { 7, 6,10, 7,10, 8, 8,10, 9,-1,-1,-1,-1,-1,-1,-1},
    { 6, 8, 4,11, 8, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 6,11, 3, 0, 6, 0, 4, 6,-1,-1,-1,-1,-1,-1,-1},
    { 8, 6,11, 8, 4, 6, 9, 0, 1,-1,-1,-1,-1,-1,-1,-1},
    { 9, 4, 6, 9, 6, 3, 9, 3, 1,11, 3, 6,-1,-1,-1,-1},
    { 6, 8, 4, 6,11, 8, 2,10, 1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10, 3, 0,11, 0, 6,11, 0, 4, 6,-1,-1,-1,-1},
    { 4,11, 8, 4, 6,11, 0, 2, 9, 2,10, 9,-1,-1,-1,-1},
    {10, 9, 3,10, 3, 2, 9, 4, 3,11, 3, 6, 4, 6, 3,-1},
    { 8, 2, 3, 8, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1},
    { 0, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8,-1,-1,-1,-1},
    { 1, 9, 4, 1, 4, 2, 2, 4, 6,-1,-1,-1,-1,-1,-1,-1},
    { 8, 1, 3, 8, 6, 1, 8, 4, 6, 6,10, 1,-1,-1,-1,-1},
    {10, 1, 0,10, 0, 6, 6, 0, 4,-1,-1,-1,-1,-1,-1,-1},
    { 4, 6, 3, 4, 3, 8, 6,10, 3, 0, 3, 9,10, 9, 3,-1},
    {10, 9, 4, 6,10, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 9, 5, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3, 4, 9, 5,11, 7, 6,-1,-1,-1,-1,-1,-1,-1},
    { 5, 0, 1, 5, 4, 0, 7, 6,11,-1,-1,-1,-1,-1,-1,-1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5,-1,-1,-1,-1},
    { 9, 5, 4,10, 1, 2, 7, 6,11,-1,-1,-1,-1,-1,-1,-1},
    { 6,11, 7, 1, 2,10, 0, 8, 3, 4, 9, 5,-1,-1,-1,-1},
    { 7, 6,11, 5, 4,10, 4, 2,10, 4, 0, 2,-1,-1,-1,-1},
    { 3, 4, 8, 3, 5, 4, 3, 2, 5,10, 5, 2,11, 7, 6,-1},
    { 7, 2, 3, 7, 6, 2, 5, 4, 9,-1,-1,-1,-1,-1,-1,-1},
    { 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7,-1,-1,-1,-1},
    { 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0,-1,-1,-1,-1},
    { 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,-1},
    { 9, 5, 4,10, 1, 6, 1, 7, 6, 1, 3, 7,-1,-1,-1,-1},
    { 1, 6,10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,-1},
    { 4, 0,10, 4,10, 5, 0, 3,10, 6,10, 7, 3, 7,10,-1},
    { 7, 6,10, 7,10, 8, 5, 4,10, 4, 8,10,-1,-1,-1,-1},
    { 6, 9, 5, 6,11, 9,11, 8, 9,-1,-1,-1,-1,-1,-1,-1},
    { 3, 6,11, 0, 6, 3, 0, 5, 6, 0, 9, 5,-1,-1,-1,-1},
    { 0,11, 8, 0, 5,11, 0, 1, 5, 5, 6,11,-1,-1,-1,-1},
    { 6,11, 3, 6, 3, 5, 5, 3, 1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,10, 9, 5,11, 9,11, 8,11, 5, 6,-1,-1,-1,-1},
    { 0,11, 3, 0, 6,11, 0, 9, 6, 5, 6, 9, 1, 2,10,-1},
    {11, 8, 5,11, 5, 6, 8, 0, 5,10, 5, 2, 0, 2, 5,-1},
    { 6,11, 3, 6, 3, 5, 2,10, 3,10, 5, 3,-1,-1,-1,-1},
    { 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2,-1,-1,-1,-1},
    { 9, 5, 6, 9, 6, 0, 0, 6, 2,-1,-1,-1,-1,-1,-1,-1},
    { 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,-1},
    { 1, 5, 6, 2, 1, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 3, 6, 1, 6,10, 3, 8, 6, 5, 6, 9, 8, 9, 6,-1},
    {10, 1, 0,10, 0, 6, 9, 5, 0, 5, 6, 0,-1,-1,-1,-1},
    { 0, 3, 8, 5, 6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10, 5, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11, 5,10, 7, 5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11, 5,10,11, 7, 5, 8, 3, 0,-1,-1,-1,-1,-1,-1,-1},
    { 5,11, 7, 5,10,11, 1, 9, 0,-1,-1,-1,-1,-1,-1,-1},
    {10, 7, 5,10,11, 7, 9, 8, 1, 8, 3, 1,-1,-1,-1,-1},
    {11, 1, 2,11, 7, 1, 7, 5, 1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2,11,-1,-1,-1,-1},
    { 9, 7, 5, 9, 2, 7, 9, 0, 2, 2,11, 7,-1,-1,-1,-1},
    { 7, 5, 2, 7, 2,11, 5, 9, 2, 3, 2, 8, 9, 8, 2,-1},
    { 2, 5,10, 2, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1},
    { 8, 2, 0, 8, 5, 2, 8, 7, 5,10, 2, 5,-1,-1,-1,-1},
    { 9, 0, 1, 5,10, 3, 5, 3, 7, 3,10, 2,-1,-1,-1,-1},
    { 9, 8, 2, 9, 2, 1, 8, 7, 2,10, 2, 5, 7, 5, 2,-1},
    { 1, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 7, 0, 7, 1, 1, 7, 5,-1,-1,-1,-1,-1,-1,-1},
    { 9, 0, 3, 9, 3, 5, 5, 3, 7,-1,-1,-1,-1,-1,-1,-1},
    { 9, 8, 7, 5, 9, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 5, 8, 4, 5,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1},
    { 5, 0, 4, 5,11, 0, 5,10,11,11, 3, 0,-1,-1,-1,-1},
    { 0, 1, 9, 8, 4,10, 8,10,11,10, 4, 5,-1,-1,-1,-1},
    {10,11, 4,10, 4, 5,11, 3, 4, 9, 4, 1, 3, 1, 4,-1},
    { 2, 5, 1, 2, 8, 5, 2,11, 8, 4, 5, 8,-1,-1,-1,-1},
    { 0, 4,11, 0,11, 3, 4, 5,11, 2,11, 1, 5, 1,11,-1},
    { 0, 2, 5, 0, 5, 9, 2,11, 5, 4, 5, 8,11, 8, 5,-1},
    { 9, 4, 5, 2,11, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 2, 5,10, 3, 5, 2, 3, 4, 5, 3, 8, 4,-1,-1,-1,-1},
    { 5,10, 2, 5, 2, 4, 4, 2, 0,-1,-1,-1,-1,-1,-1,-1},
    { 3,10, 2, 3, 5,10, 3, 8, 5, 4, 5, 8, 0, 1, 9,-1},
    { 5,10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,-1,-1,-1,-1},
    { 8, 4, 5, 8, 5, 3, 3, 5, 1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 4, 5, 1, 0, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,-1,-1,-1,-1},
    { 9, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4,11, 7, 4, 9,11, 9,10,11,-1,-1,-1,-1,-1,-1,-1},
    { 0, 8, 3, 4, 9, 7, 9,11, 7, 9,10,11,-1,-1,-1,-1},
    { 1,10,11, 1,11, 4, 1, 4, 0, 7, 4,11,-1,-1,-1,-1},
    { 3, 1, 4, 3, 4, 8, 1,10, 4, 7, 4,11,10,11, 4,-1},
    { 4,11, 7, 9,11, 4, 9, 2,11, 9, 1, 2,-1,-1,-1,-1},
    { 9, 7, 4, 9,11, 7, 9, 1,11, 2,11, 1, 0, 8, 3,-1},
    {11, 7, 4,11, 4, 2, 2, 4, 0,-1,-1,-1,-1,-1,-1,-1},
    {11, 7, 4,11, 4, 2, 8, 3, 4, 3, 2, 4,-1,-1,-1,-1},
    { 2, 9,10, 2, 7, 9, 2, 3, 7, 7, 4, 9,-1,-1,-1,-1},
    { 9,10, 7, 9, 7, 4,10, 2, 7, 8, 7, 0, 2, 0, 7,-1},
    { 3, 7,10, 3,10, 2, 7, 4,10, 1,10, 0, 4, 0,10,-1},
    { 1,10, 2, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 9, 1, 4, 1, 7, 7, 1, 3,-1,-1,-1,-1,-1,-1,-1},
    { 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1,-1,-1,-1,-1},
    { 4, 0, 3, 7, 4, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 4, 8, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 9,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 0, 9, 3, 9,11,11, 9,10,-1,-1,-1,-1,-1,-1,-1},
    { 0, 1,10, 0,10, 8, 8,10,11,-1,-1,-1,-1,-1,-1,-1},
    { 3, 1,10,11, 3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 2,11, 1,11, 9, 9,11, 8,-1,-1,-1,-1,-1,-1,-1},
    { 3, 0, 9, 3, 9,11, 1, 2, 9, 2,11, 9,-1,-1,-1,-1},
    { 0, 2,11, 8, 0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 3, 2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 2, 3, 8, 2, 8,10,10, 8, 9,-1,-1,-1,-1,-1,-1,-1},
    { 9,10, 2, 0, 9, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 2, 3, 8, 2, 8,10, 0, 1, 8, 1,10, 8,-1,-1,-1,-1},
    { 1,10, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 1, 3, 8, 9, 1, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 9, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    { 0, 3, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
};

// ----- MC2: per-case triangle COUNT (derived from kTriTable) ---------------------------------------
// kTriCount[c] = (# non-negative entries in kTriTable[c]) / 3, in [0,5]. constexpr-derived so it cannot
// drift from the table (the unit test re-derives it independently as the guard). kTriCount[0x00] and
// kTriCount[0xFF] are 0 (fully-out / fully-in cells emit no triangles).
namespace detail {
struct TriCountTable {
    int v[256];
    constexpr TriCountTable() : v{} {
        for (int c = 0; c < 256; ++c) {
            int n = 0;
            for (int e = 0; e < 16; ++e)
                if (kTriTable[c][e] >= 0) ++n;
            v[c] = n / 3;
        }
    }
};
inline constexpr TriCountTable kTriCountTable{};
}  // namespace detail

// The per-case triangle count, constexpr-derived from kTriTable. kTriCount[c] in [0,5].
inline constexpr const int (&kTriCount)[256] = detail::kTriCountTable.v;

// CountTriangles(caseIndex) = number of triangles the cell of that MC case emits (0..5).
inline int CountTriangles(uint8_t caseIndex) { return kTriCount[caseIndex]; }

// CPU reference: countOut sized cellCount(); countOut[cellId] = CountTriangles(CaseIndex(corners,iso)),
// reusing the EXACT MC1 classify math (cellId = (cz*(ny-1)+cy)*(nx-1)+cx). Order-independent per cell
// -> a GPU thread-race CANNOT change it. This is the reference the GPU mc_count.comp memcmp's against.
inline void CountCells(const VoxelField& f, int32_t isovalue, std::vector<uint32_t>& countOut) {
    const int cxN = f.nx - 1, cyN = f.ny - 1, czN = f.nz - 1;
    countOut.assign((size_t)f.cellCount(), 0u);
    for (int cz = 0; cz < czN; ++cz)
        for (int cy = 0; cy < cyN; ++cy)
            for (int cx = 0; cx < cxN; ++cx) {
                int32_t corner[8];
                for (int i = 0; i < 8; ++i)
                    corner[i] = SampleField(f, cx + kCornerOffset[i][0],
                                            cy + kCornerOffset[i][1],
                                            cz + kCornerOffset[i][2]);
                const int cellId = (cz * cyN + cy) * cxN + cx;
                countOut[(size_t)cellId] = (uint32_t)CountTriangles(CaseIndex(corner, isovalue));
            }
}

// TotalTriangles = Σ counts (the CPU mirror of the GPU InterlockedAdd grand total). u32 suffices:
// 32768 cells × max 5 tris = 163840 << 2^32, so no overflow on the showcase field (documented).
inline uint32_t TotalTriangles(std::span<const uint32_t> counts) {
    uint32_t total = 0u;
    for (uint32_t c : counts) total += c;
    return total;
}

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

// ===== MC3: prefix-sum compaction + triangle EMISSION (midpoint vertices) ==========================
// Turn the per-cell triangle COUNTS (MC2) into an actual MESH: an exclusive prefix-sum of the counts
// gives each cell its write OFFSET into the output buffers, then each cell EMITS its triangles as
// cell-edge MIDPOINT vertices + an index buffer. Kept PURE INTEGER (a midpoint = the SUM of the two
// integer corner coords in HALF-grid units = 2×the actual midpoint — no division, no float), so the
// vertex+index buffers are GPU==CPU BIT-EXACT and the golden is cross-backend BIT-IDENTICAL. The GPU
// shaders (shaders/mc_scan.comp.hlsl single-thread exclusive scan + shaders/mc_emit.comp.hlsl one
// thread per cell) copy EdgeMidpoint + the kTriTable walk VERBATIM and memcmp GPU==CPU.

// ----- The emitted vertex: an edge-MIDPOINT in HALF-grid units -----------------------------------
// (x,y,z) = Ca + Cb (each axis), the SUM of the two cube-corner grid coords the edge connects, which
// equals 2×the actual midpoint — kept as an INTEGER on the half-unit lattice so there is no division /
// float (the swraster.h fixed-point-lattice discipline). w=0 padding (future use). Trivially memcmp-able
// (POD, no padding holes — 4×int32).
struct McVertex {
    int32_t x, y, z, w;
};

// EdgeMidpoint(cell, edge): kEdgeCorner[edge] = (a,b) are the two corner indices the edge connects;
// Ca = cell + kCornerOffset[a], Cb = cell + kCornerOffset[b]; the midpoint (half-units) = Ca + Cb. Pure
// integer. The GPU mc_emit.comp copies this body VERBATIM.
inline McVertex EdgeMidpoint(int cx, int cy, int cz, int edge) {
    const int a = kEdgeCorner[edge][0];
    const int b = kEdgeCorner[edge][1];
    const int ax = cx + kCornerOffset[a][0], ay = cy + kCornerOffset[a][1], az = cz + kCornerOffset[a][2];
    const int bx = cx + kCornerOffset[b][0], by = cy + kCornerOffset[b][1], bz = cz + kCornerOffset[b][2];
    return McVertex{ax + bx, ay + by, az + bz, 0};
}

// PrefixSumOffsets: the CPU reference EXCLUSIVE prefix-sum of the per-cell counts. offsetsOut[i] =
// Σ_{j<i} counts[j] (so offsetsOut[0]==0), totalOut = Σ counts (== MC2's TotalTriangles). This is the
// serial scan the GPU mc_scan.comp (single thread) mirrors BIT-EXACT — an inherently sequential running
// sum (the VT2 single-thread-allocator pattern). offsetsOut sized counts.size().
inline void PrefixSumOffsets(std::span<const uint32_t> counts,
                             std::span<uint32_t> offsetsOut, uint32_t& totalOut) {
    uint32_t running = 0u;
    const size_t n = counts.size();
    for (size_t i = 0; i < n; ++i) {
        offsetsOut[i] = running;   // exclusive: the offset is the sum of all PRIOR counts
        running += counts[i];
    }
    totalOut = running;
}

// EmitCell: walk kTriTable[caseIndex] in groups of 3 (until -1); for triangle t (its GLOBAL index is
// triOffset+t), for each of its 3 edges k write vertsOut[3*(triOffset+t)+k] = EdgeMidpoint(cell, edge_k)
// and idxOut[3*(triOffset+t)+k] = 3*(triOffset+t)+k (a triangle SOUP — no cross-cell vertex dedup; the
// index buffer is the identity, deferred to MC: indexed sharing). Winding per the canonical table. The
// per-cell writes land in the cell's DISJOINT [3*triOffset, 3*(triOffset+triCount)) range, so the GPU
// emit is race-free with NO atomics. The GPU mc_emit.comp copies this walk VERBATIM.
inline void EmitCell(int cx, int cy, int cz, uint8_t caseIndex, uint32_t triOffset,
                     std::span<McVertex> vertsOut, std::span<uint32_t> idxOut) {
    const int8_t* row = kTriTable[caseIndex];
    uint32_t t = 0u;
    for (int e = 0; e + 2 < 16; e += 3) {
        if (row[e] < 0) break;   // -1 terminator -> done
        for (int k = 0; k < 3; ++k) {
            const int edge = (int)row[e + k];
            const uint32_t slot = 3u * (triOffset + t) + (uint32_t)k;
            vertsOut[slot] = EdgeMidpoint(cx, cy, cz, edge);
            idxOut[slot] = slot;   // identity index (triangle soup)
        }
        ++t;
    }
}

// MarchCells: the full CPU reference mesher — classify → count → PrefixSumOffsets → EmitCell per cell.
// verts + idx are sized to exactly 3*triCount and filled in cell order (the deterministic source-order
// the GPU's per-cell disjoint-range writes reproduce). The reference the GPU mc_scan+mc_emit memcmp's
// against; verts.size() == idx.size() == 3*triCount.
inline void MarchCells(const VoxelField& f, int32_t iso,
                       std::vector<McVertex>& verts, std::vector<uint32_t>& idx, uint32_t& triCount) {
    const int cxN = f.nx - 1, cyN = f.ny - 1, czN = f.nz - 1;
    const size_t cellCount = (size_t)f.cellCount();

    // 1) classify + count per cell (cellId = (cz*cyN+cy)*cxN+cx).
    std::vector<uint8_t> cases;
    std::vector<uint32_t> counts;
    ClassifyCells(f, iso, cases);
    CountCells(f, iso, counts);

    // 2) exclusive prefix-sum of the counts -> per-cell triangle offset + grand total.
    std::vector<uint32_t> offsets(cellCount, 0u);
    uint32_t total = 0u;
    PrefixSumOffsets(std::span<const uint32_t>(counts), std::span<uint32_t>(offsets), total);
    triCount = total;

    // 3) preallocate the output buffers to exactly 3*total, emit each cell at its offset.
    verts.assign((size_t)total * 3u, McVertex{0, 0, 0, 0});
    idx.assign((size_t)total * 3u, 0u);
    for (int cz = 0; cz < czN; ++cz)
        for (int cy = 0; cy < cyN; ++cy)
            for (int cx = 0; cx < cxN; ++cx) {
                const int cellId = (cz * cyN + cy) * cxN + cx;
                EmitCell(cx, cy, cz, cases[(size_t)cellId], offsets[(size_t)cellId],
                         std::span<McVertex>(verts), std::span<uint32_t>(idx));
            }
}

// ===== MC4: fixed-point INTERPOLATED edge-vertex placement ========================================
// MC3 placed each emitted vertex at the cell-edge MIDPOINT. MC4 places it at the ACTUAL isosurface
// crossing along the edge — linear interpolation by the two corner scalar values — so the extracted
// surface is SMOOTH (watertight-on-the-field) instead of blocky. The interpolation parameter
//   t = (iso - s0) / (s1 - s0)   in [0,1]
// is computed in FIXED-POINT INTEGER on a 1/kSub edge lattice (the swraster.h kSub trick) — NO float
// anywhere on the interpolation path — so the interpolated vertex buffer stays GPU==CPU BIT-EXACT and
// the golden is cross-backend BIT-IDENTICAL. MC3's EdgeMidpoint/EmitCell/MarchCells are left UNCHANGED
// (its mc_emit golden is the t=0.5 reference); MC4 adds the sibling EdgeInterp/EmitCellInterp/
// MarchCellsInterp. The GPU shaders/mc_interp.comp.hlsl copies EdgeInterp's fixed-point t + clamp
// VERBATIM and memcmp's GPU==CPU.

// The fixed-point edge lattice: a vertex coordinate is stored in 1/kSub grid units, so the edge
// MIDPOINT is kSub/2, the cube spans [0, (dim-1)*kSub], and t in [0,kSub]. Finer than swraster's
// kSub=16 (here 256) for a smooth surface. Note MC3's McVertex was HALF-units (Ca+Cb = 2×midpoint);
// MC4's EdgeInterp uses the SAME McVertex struct but in 1/kSub units (Ca*kSub + (Cb-Ca)*t), a finer
// independent lattice — the two emitters are siblings, never mixed in one buffer.
static constexpr int32_t kSub = 256;

// EdgeInterp(cell, edge, field, iso): the INTERPOLATED isosurface crossing on kEdgeCorner[edge]=(a,b).
// Ca = cell + kCornerOffset[a], Cb = cell + kCornerOffset[b] (the two differ by exactly +1 in ONE axis
// — MC edges are axis-aligned). s0 = SampleField(Ca), s1 = SampleField(Cb). Compute t in [0,kSub]:
//   * if s1 == s0 (degenerate, no gradient): t = kSub/2 (the midpoint — matches MC3's choice,
//     deterministic).
//   * else: t = ((iso - s0) * kSub) / (s1 - s0), then clamp t to [0,kSub].
// vertexPos.axis = Ca.axis*kSub + (Cb.axis - Ca.axis)*t per axis (the spanning axis moves by ±t, the
// other two stay at Ca.axis*kSub). Returns {x,y,z,0} in 1/kSub units.
//
// THE CROSS-BACKEND CRUX: the ONE division is INTEGER and TRUNCATES TOWARD ZERO — the DEFAULT for both
// C++ `/` and HLSL `/` — so it is bit-identical CPU↔GPU↔every vendor. We use int32 for t (NOT int64):
// the numerator (iso - s0) * kSub fits int32 — for this field |scalar| ~ radius*kFixed + dist*kFixed,
// and on the showcase field (n=33, r=12, kFixed=256) |scalar| <= ~10200, so |(iso - s0)| <= ~10200 and
// (iso - s0)*kSub (kSub=256) <= ~2.6e6, far inside int32's 2.1e9 range. Staying int32 AVOIDS the
// int64/glslc(SPIR-V) issue (no --msl-version 20200, no Vulkan-only/Metal-CPU split). The t clamp
// guards an out-of-range t from a non-bracketing edge. The GPU mc_interp.comp copies this body VERBATIM.
inline McVertex EdgeInterp(int cx, int cy, int cz, int edge, const VoxelField& f, int32_t iso) {
    const int a = kEdgeCorner[edge][0];
    const int b = kEdgeCorner[edge][1];
    const int ax = cx + kCornerOffset[a][0], ay = cy + kCornerOffset[a][1], az = cz + kCornerOffset[a][2];
    const int bx = cx + kCornerOffset[b][0], by = cy + kCornerOffset[b][1], bz = cz + kCornerOffset[b][2];
    const int32_t s0 = SampleField(f, ax, ay, az);
    const int32_t s1 = SampleField(f, bx, by, bz);

    int32_t t;
    if (s1 == s0) {
        t = kSub / 2;                              // degenerate edge -> the deterministic midpoint
    } else {
        t = ((iso - s0) * kSub) / (s1 - s0);       // truncating integer divide (C++/HLSL identical)
        if (t < 0) t = 0; else if (t > kSub) t = kSub;
    }

    McVertex v;
    v.x = ax * kSub + (bx - ax) * t;
    v.y = ay * kSub + (by - ay) * t;
    v.z = az * kSub + (bz - az) * t;
    v.w = 0;
    return v;
}

// EmitCellInterp: MC3's EmitCell with EdgeInterp substituted for EdgeMidpoint (same triangle-soup
// layout, same identity indices, same kTriTable walk, same disjoint [3*triOffset, ..) range -> race-
// free, NO atomics). The GPU mc_interp.comp copies this walk VERBATIM.
inline void EmitCellInterp(int cx, int cy, int cz, uint8_t caseIndex, uint32_t triOffset,
                           const VoxelField& f, int32_t iso,
                           std::span<McVertex> vertsOut, std::span<uint32_t> idxOut) {
    const int8_t* row = kTriTable[caseIndex];
    uint32_t t = 0u;
    for (int e = 0; e + 2 < 16; e += 3) {
        if (row[e] < 0) break;   // -1 terminator -> done
        for (int k = 0; k < 3; ++k) {
            const int edge = (int)row[e + k];
            const uint32_t slot = 3u * (triOffset + t) + (uint32_t)k;
            vertsOut[slot] = EdgeInterp(cx, cy, cz, edge, f, iso);
            idxOut[slot] = slot;   // identity index (triangle soup)
        }
        ++t;
    }
}

// MarchCellsInterp: MC3's MarchCells with EdgeInterp emission — classify -> count -> PrefixSumOffsets ->
// EmitCellInterp per cell. Same triangle count + index layout as MarchCells (the ONLY difference is the
// vertex POSITIONS — interpolated, not midpoint). The reference the GPU mc_scan+mc_interp memcmp's
// against; verts.size() == idx.size() == 3*triCount.
inline void MarchCellsInterp(const VoxelField& f, int32_t iso,
                             std::vector<McVertex>& verts, std::vector<uint32_t>& idx,
                             uint32_t& triCount) {
    const int cxN = f.nx - 1, cyN = f.ny - 1, czN = f.nz - 1;
    const size_t cellCount = (size_t)f.cellCount();

    std::vector<uint8_t> cases;
    std::vector<uint32_t> counts;
    ClassifyCells(f, iso, cases);
    CountCells(f, iso, counts);

    std::vector<uint32_t> offsets(cellCount, 0u);
    uint32_t total = 0u;
    PrefixSumOffsets(std::span<const uint32_t>(counts), std::span<uint32_t>(offsets), total);
    triCount = total;

    verts.assign((size_t)total * 3u, McVertex{0, 0, 0, 0});
    idx.assign((size_t)total * 3u, 0u);
    for (int cz = 0; cz < czN; ++cz)
        for (int cy = 0; cy < cyN; ++cy)
            for (int cx = 0; cx < cxN; ++cx) {
                const int cellId = (cz * cyN + cy) * cxN + cx;
                EmitCellInterp(cx, cy, cz, cases[(size_t)cellId], offsets[(size_t)cellId],
                               f, iso, std::span<McVertex>(verts), std::span<uint32_t>(idx));
            }
}

// ===== MC5: host RENDER-mesh build (fixed-point MC mesh -> float lit-render vertices) =============
// MC4's MarchCellsInterp produces a fixed-point (1/kSub) triangle SOUP (verts + identity indices). MC5
// converts that PROVEN geometry into the float vertex format the engine's existing lit mesh pipeline
// consumes, then a showcase renders it lit. This is the MC arc's FIRST FLOAT step: the conversion does
// the ONE documented host float divide (position = vert/kSub) + a flat per-face normal. It does NOT
// touch MC1-MC4's integer mesh (the render consumes it verbatim -> provenance is exact) and adds NO
// backend / RHI symbol (pure CPU, header-only) — the showcase copies RenderVertex into scene::Vertex.
//
// FLAT per-face normals: the mesh is a triangle SOUP (no shared-vertex averaging), so every triangle's
// 3 vertices get the SAME face normal normalize(cross(p1-p0, p2-p0)) — a faceted look that reads the
// extracted geometry clearly. std::fma is used in the normal/centering math so a CPU shade mirror (the
// interior GPU==CPU proof) can reproduce the exact host floats (the visresolve FP discipline).

// A render-ready vertex: world position + flat face normal. POD float6 (no padding holes), trivially
// copied into scene::Vertex (pos/color/uv/normal/tangent) by the showcase.
struct RenderVertex {
    float px, py, pz;   // world position = (fixed-point vert / kSub), centered + scaled into frame
    float nx, ny, nz;   // flat per-face normal (unit length), shared by the triangle's 3 verts
};

// BuildRenderMesh: convert MarchCellsInterp's fixed-point (1/kSub) triangle-soup verts + indices into
// RenderVertex world positions + flat per-face normals, centered on the origin and scaled so the mesh's
// largest extent spans `targetExtent` world units (frames the mesh for a fixed camera). Pure +
// deterministic: same fixed-point mesh -> same floats every run. The index buffer is COPYable verbatim
// (identity soup) — the render reuses MarchCellsInterp's idx unchanged; only positions/normals are
// produced here. outVerts.size() == verts.size(); a NON-multiple-of-3 vert count (malformed soup) is
// tolerated by leaving the trailing <3 verts with a zero normal (never happens for a real MC mesh).
inline void BuildRenderMesh(std::span<const McVertex> verts, float targetExtent,
                            std::vector<RenderVertex>& outVerts) {
    const size_t n = verts.size();
    outVerts.assign(n, RenderVertex{0, 0, 0, 0, 0, 0});
    if (n == 0) return;

    // 1) fixed-point -> float world units (the ONE documented host float divide) + the mesh's AABB.
    const float inv = 1.0f / (float)kSub;
    float minx = 1e30f, miny = 1e30f, minz = 1e30f;
    float maxx = -1e30f, maxy = -1e30f, maxz = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        const float x = (float)verts[i].x * inv;
        const float y = (float)verts[i].y * inv;
        const float z = (float)verts[i].z * inv;
        outVerts[i].px = x; outVerts[i].py = y; outVerts[i].pz = z;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
        if (z < minz) minz = z; if (z > maxz) maxz = z;
    }

    // 2) center on the AABB midpoint + scale the largest extent to targetExtent (frame the mesh).
    const float cx = 0.5f * (minx + maxx);
    const float cy = 0.5f * (miny + maxy);
    const float cz = 0.5f * (minz + maxz);
    const float ex = maxx - minx, ey = maxy - miny, ez = maxz - minz;
    float ext = ex; if (ey > ext) ext = ey; if (ez > ext) ext = ez;
    const float scale = (ext > 0.0f) ? (targetExtent / ext) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        outVerts[i].px = std::fma(outVerts[i].px - cx, scale, 0.0f);
        outVerts[i].py = std::fma(outVerts[i].py - cy, scale, 0.0f);
        outVerts[i].pz = std::fma(outVerts[i].pz - cz, scale, 0.0f);
    }

    // 3) flat per-face normal: each triangle (3 consecutive soup verts) gets the flat face normal on
    //    all 3 of its verts. std::fma in the cross/length so a CPU shade mirror reproduces it. The
    //    canonical MC (Bourke / Lorensen-Cline) triangle winding emits cross(p1-p0, p2-p0) pointing
    //    INTO the surface (toward the dense/inside corners), so we NEGATE it to get the geometric
    //    OUTWARD normal of the extracted shell — the surface the camera sees from outside is correctly
    //    lit (outward N·L). Positions/indices are untouched: only the shading normal is produced here.
    const size_t triEnd = (n / 3) * 3;
    for (size_t t = 0; t < triEnd; t += 3) {
        const RenderVertex& a = outVerts[t + 0];
        const RenderVertex& b = outVerts[t + 1];
        const RenderVertex& c = outVerts[t + 2];
        const float e1x = b.px - a.px, e1y = b.py - a.py, e1z = b.pz - a.pz;
        const float e2x = c.px - a.px, e2y = c.py - a.py, e2z = c.pz - a.pz;
        float nx = -std::fma(e1y, e2z, -e1z * e2y);
        float ny = -std::fma(e1z, e2x, -e1x * e2z);
        float nz = -std::fma(e1x, e2y, -e1y * e2x);
        float len2 = std::fma(nx, nx, std::fma(ny, ny, nz * nz));
        float invLen = (len2 > 0.0f) ? (1.0f / std::sqrt(len2)) : 0.0f;
        nx *= invLen; ny *= invLen; nz *= invLen;
        for (int k = 0; k < 3; ++k) {
            outVerts[t + (size_t)k].nx = nx;
            outVerts[t + (size_t)k].ny = ny;
            outVerts[t + (size_t)k].nz = nz;
        }
    }
}

// ===== MC6: SMOOTH field-gradient per-vertex normals (the finishing quality step) =================
// MC5 gave every vertex of a triangle the FLAT FACE normal (faceted). MC6 gives each vertex the SDF
// FIELD-GRADIENT normal at its OWN position — the standard Marching-Cubes smooth-normal technique — so
// adjacent triangles share consistent vertex normals and the extracted sphere shades SMOOTH (Gouraud)
// instead of faceted. Positions + indices stay EXACTLY the MC4 bit-exact mesh (MarchCellsInterp) — only
// the per-vertex NORMAL changes (provenance unchanged). Pure host float (the same float class as MC5's
// flat normal; the render is float regardless). NO new backend / RHI / shader.
//
// THE SIGN CONVENTION (verified against the sphere, documented): for the scalar field the gradient ∇f
// points along INCREASING f. MakeSphereField sets scalar = (radius - dist)*kFixed, so f is LARGEST at
// the centre and DECREASES outward -> ∇f points INWARD (toward the centre). The OUTWARD surface normal
// (the face the camera sees, the one MC5 lit) is therefore -∇f. We NEGATE the central-difference
// gradient to get the outward normal — the SAME outward convention MC5 used (it negated the inward MC
// winding cross). Verified outward on the sphere: dot(N, vertexPos - centre) > 0 (N points away from the
// centre), asserted in the showcase + tests for >90% of non-degenerate surface verts.

// GradientNormal(vert, field): the unit OUTWARD field-gradient normal at the grid neighbourhood of the
// fixed-point vertex `vert` (in 1/kSub units). The vertex's integer grid cell is vert.xyz / kSub (floor
// toward zero; MC verts are non-negative so this is a true floor). Sample f at the 6 axis neighbours via
// SampleField (clamped at borders), central difference grad = (fx+ - fx-, fy+ - fy-, fz+ - fz-), then
// OUTWARD = normalize(-grad) (see the sign note above). A zero gradient (flat field neighbourhood)
// returns {0,0,0} (a degenerate normal — the showcase/test treat it the same as MC5's zero face normal).
// std::fma in the length so a CPU shade mirror reproduces the exact host floats (the visresolve FP
// discipline). Pure + deterministic: same vert + field -> same normal every run.
inline void GradientNormal(const McVertex& vert, const VoxelField& field,
                           float& outNx, float& outNy, float& outNz) {
    // The vertex's integer grid coords from its fixed-point (1/kSub) position. MC verts are >= 0 so
    // integer divide (truncation) == floor — the deterministic grid cell the vertex sits in.
    const int gx = vert.x / kSub;
    const int gy = vert.y / kSub;
    const int gz = vert.z / kSub;

    // Central difference of the integer field at the 6 axis neighbours (SampleField clamps at borders).
    const int32_t fxp = SampleField(field, gx + 1, gy, gz);
    const int32_t fxm = SampleField(field, gx - 1, gy, gz);
    const int32_t fyp = SampleField(field, gx, gy + 1, gz);
    const int32_t fym = SampleField(field, gx, gy - 1, gz);
    const int32_t fzp = SampleField(field, gx, gy, gz + 1);
    const int32_t fzm = SampleField(field, gx, gy, gz - 1);

    // grad = ∇f (points toward INCREASING f = INWARD for this sphere field). OUTWARD normal = -grad.
    float nx = -(float)(fxp - fxm);
    float ny = -(float)(fyp - fym);
    float nz = -(float)(fzp - fzm);

    const float len2 = std::fma(nx, nx, std::fma(ny, ny, nz * nz));
    const float invLen = (len2 > 0.0f) ? (1.0f / std::sqrt(len2)) : 0.0f;
    outNx = nx * invLen; outNy = ny * invLen; outNz = nz * invLen;
}

// BuildSmoothRenderMesh: MC5's BuildRenderMesh with per-vertex SMOOTH field-gradient normals instead of
// the flat per-face normal. POSITIONS + the (implicit identity) INDICES are BYTE-IDENTICAL to
// BuildRenderMesh / MarchCellsInterp (the bit-exact mesh — provenance unchanged); ONLY outVerts[i].n*
// differs. `verts` is the SAME fixed-point (1/kSub) MarchCellsInterp soup; `field` is the SAME field it
// was meshed from (the gradient is sampled from it). Pure + deterministic; std::fma where it helps.
inline void BuildSmoothRenderMesh(std::span<const McVertex> verts, const VoxelField& field,
                                  float targetExtent, std::vector<RenderVertex>& outVerts) {
    const size_t n = verts.size();
    outVerts.assign(n, RenderVertex{0, 0, 0, 0, 0, 0});
    if (n == 0) return;

    // 1) fixed-point -> float world units (the ONE documented host float divide) + the mesh's AABB.
    //    IDENTICAL to BuildRenderMesh so the positions are byte-for-byte the same.
    const float inv = 1.0f / (float)kSub;
    float minx = 1e30f, miny = 1e30f, minz = 1e30f;
    float maxx = -1e30f, maxy = -1e30f, maxz = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        const float x = (float)verts[i].x * inv;
        const float y = (float)verts[i].y * inv;
        const float z = (float)verts[i].z * inv;
        outVerts[i].px = x; outVerts[i].py = y; outVerts[i].pz = z;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
        if (z < minz) minz = z; if (z > maxz) maxz = z;
    }

    // 2) center on the AABB midpoint + scale the largest extent to targetExtent — IDENTICAL math.
    const float cx = 0.5f * (minx + maxx);
    const float cy = 0.5f * (miny + maxy);
    const float cz = 0.5f * (minz + maxz);
    const float ex = maxx - minx, ey = maxy - miny, ez = maxz - minz;
    float ext = ex; if (ey > ext) ext = ey; if (ez > ext) ext = ez;
    const float scale = (ext > 0.0f) ? (targetExtent / ext) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        outVerts[i].px = std::fma(outVerts[i].px - cx, scale, 0.0f);
        outVerts[i].py = std::fma(outVerts[i].py - cy, scale, 0.0f);
        outVerts[i].pz = std::fma(outVerts[i].pz - cz, scale, 0.0f);
    }

    // 3) SMOOTH per-vertex normal: each vertex gets the OUTWARD field-gradient normal at ITS OWN grid
    //    position (NOT the flat face normal). The gradient is computed in field GRID space from the
    //    ORIGINAL fixed-point vert (the world transform above is a uniform scale + translation, which
    //    does NOT change a normal's direction, so the grid-space gradient direction is already the
    //    correct world-space normal direction). Adjacent triangles' shared-edge verts sit at the same
    //    grid position -> the same gradient -> the seams shade smooth.
    for (size_t i = 0; i < n; ++i) {
        float nx, ny, nz;
        GradientNormal(verts[i], field, nx, ny, nz);
        outVerts[i].nx = nx; outVerts[i].ny = ny; outVerts[i].nz = nz;
    }
}

}  // namespace hf::render::mc
