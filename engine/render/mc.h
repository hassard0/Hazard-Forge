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

}  // namespace hf::render::mc
