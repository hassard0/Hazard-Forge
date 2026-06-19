#pragma once
// Slice FR1 — Deterministic Rigid-Body Fracture/Destruction Slice 1: CELL PRE-FRACTURE / VORONOI
// DECOMPOSITION (the BEACHHEAD of FLAGSHIP #14: DETERMINISTIC RIGID-BODY FRACTURE / DESTRUCTION,
// hf::sim::fract). Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::fract.
//
// WHAT THIS IS: the integer core of pre-fracture — a bounded integer LATTICE (the render/mc.h VoxelField
// shape) is partitioned into N fragment CELLS against a host-supplied seed set: each lattice sample is
// assigned to its NEAREST SEED by SQUARED INTEGER distance (dx²+dy²+dz², NO sqrt — the cluster_lod.h DV
// squared-distance trick), with the LOWEST seed index winning ties (the meshlet.h DS total-order
// tie-break — a single deterministic answer). The result cellId[sampleIndex] is a deterministic Voronoi
// partition. The GPU compute pass (shaders/fract_classify.comp.hlsl) copies ClassifyFractCells's
// per-sample loop VERBATIM and proves bit-identical to this header (memcmp GPU==CPU, NO tolerance) — the
// MC1/GF1 pattern applied to nearest-seed cell classification.
//
// THE CROSS-BACKEND CRUX (why GPU==CPU + Vulkan==Metal hold bit-exactly, MSL-NATIVE on BOTH backends):
// the classification is PURE INT32 — squared-distance compares of small lattice integer coords. With each
// axis < ~1024, the max squared distance 3·1023² ≈ 3.1M fits int32 with huge headroom, so NO int64 is
// needed and the shader MSL-generates natively (a TRUE GPU pass on both Vulkan AND Metal — the strongest
// cross-vendor proof, exactly like MC1/GF1). The per-sample write is order-independent (every sample
// computes its own nearest seed from the SAME read-only seed array, so a GPU thread-race CANNOT change
// the result — the MC1/VT1 set-write argument) -> race-free, NO atomics. Integer-in -> integer-out ->
// identical bits on every vendor.
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::/Backend::) symbols. NO GPU, NO new RHI. Mentions of
// "GPU" here are doc-only. INTEGER on every path (no float, NO <cmath>) so the cell golden is
// cross-backend bit-identical. #includes sim/fpx.h read-only ONLY (fx/FxVec3 are carried for FR2's
// world-unit fragment positions; FR1's lattice coords are plain int32).

#include <cstdint>
#include <vector>

#include "sim/fpx.h"  // read-only: fx / FxVec3 (carried for FR2 world-unit fragment positions)

namespace hf::sim::fract {

// Re-export the fixed-point primitives (read-only, carried for later FR slices' world-unit positions).
using fpx::fx;
using fpx::FxVec3;

// ----- The source-volume integer lattice ----------------------------------------------------------
// A bounded integer lattice of nx·ny·nz SAMPLE points (the render/mc.h VoxelField extent shape). A
// sample at integer coord (x,y,z) linearizes to sampleIndex = (z*ny + y)*nx + x (flat row-major — the
// mc.h/swraster VoxelField layout). FR1 classifies each sample to its nearest seed -> a cellId per
// sample.
struct FractField {
    int nx = 0, ny = 0, nz = 0;  // SAMPLE counts per axis

    int sampleCount() const { return nx * ny * nz; }
};

// SampleCount(field): the total number of lattice samples (nx*ny*nz). Pure helper.
inline int SampleCount(const FractField& f) { return f.nx * f.ny * f.nz; }

// SampleIndex(field, x, y, z): the flat row-major index (z*ny + y)*nx + x of the sample at (x,y,z).
// VERBATIM the render/mc.h sample linearization (so the GPU's per-sample write lands at the same slot).
inline int SampleIndex(const FractField& f, int x, int y, int z) {
    return (z * f.ny + y) * f.nx + x;
}

// ----- The fracture seed (a Voronoi site in integer lattice coords) -------------------------------
// A seed lives at an integer lattice coordinate (x,y,z). FR1 assigns each lattice sample to its nearest
// seed by squared integer distance. (An id/payload may be added in a later FR slice; FR1's cellId is the
// seed INDEX into the host seed array.)
struct FractSeed {
    int x = 0, y = 0, z = 0;
};

// ----- The result: a cellId per lattice sample ----------------------------------------------------
// cellId[sampleIndex] = the index (in [0, M)) of the nearest seed to that sample. One uint per lattice
// sample, flat row-major (the SampleIndex layout). This is the deterministic Voronoi partition.
struct FractCells {
    std::vector<uint32_t> cellId;  // size sampleCount(), cellId[(z*ny+y)*nx+x]
};

// ----- The nearest-seed classify (the bit-exact core — order-independent, race-free) --------------
// For a lattice sample at integer coord (x,y,z), over the M seeds in ASCENDING index k:
//   dx = x - s[k].x; dy = y - s[k].y; dz = z - s[k].z          // int32
//   d2 = dx*dx + dy*dy + dz*dz                                  // int32 squared distance (NO sqrt)
//   if d2 < bestD2: { bestD2 = d2; bestId = k }                 // STRICTLY less -> lowest-index tie-break
// returns bestId. `STRICTLY less` (<, not <=) is the tie-break: the first (lowest-index) seed at the
// minimum distance wins, so the answer is a single deterministic value independent of evaluation order.
// Pure int32. The GPU fract_classify.comp copies this body VERBATIM. (If seedCount==0 the result is 0 —
// degenerate; the showcase + tests always supply M>=1.)
inline uint32_t NearestSeed(int x, int y, int z, const FractSeed* seeds, int seedCount) {
    uint32_t bestId = 0u;
    int32_t bestD2 = INT32_MAX;
    for (int k = 0; k < seedCount; ++k) {
        const int32_t dx = (int32_t)(x - seeds[k].x);
        const int32_t dy = (int32_t)(y - seeds[k].y);
        const int32_t dz = (int32_t)(z - seeds[k].z);
        const int32_t d2 = dx * dx + dy * dy + dz * dz;  // pure int32 squared distance
        if (d2 < bestD2) { bestD2 = d2; bestId = (uint32_t)k; }
    }
    return bestId;
}

// ----- The CPU reference classifier ---------------------------------------------------------------
// out.cellId sized sampleCount(); for each lattice sample (x,y,z) write cellId[sampleIndex] =
// NearestSeed(x,y,z,...). Each sample writes ONLY its own cellId[sampleIndex] -> order-independent (a
// GPU thread-race CANNOT change the result; the MC1/VT1 set-write argument). This is the byte-for-byte
// reference the GPU fract_classify.comp memcmp's against.
inline void ClassifyFractCells(const FractField& f, const std::vector<FractSeed>& seeds,
                               FractCells& out) {
    const int seedCount = (int)seeds.size();
    const FractSeed* sp = seeds.data();
    out.cellId.assign((size_t)f.sampleCount(), 0u);
    for (int z = 0; z < f.nz; ++z)
        for (int y = 0; y < f.ny; ++y)
            for (int x = 0; x < f.nx; ++x) {
                const int idx = (z * f.ny + y) * f.nx + x;
                out.cellId[(size_t)idx] = NearestSeed(x, y, z, sp, seedCount);
            }
}

// ----- Per-cell sample-count stat (the {...} stat line + partition checks) -------------------------
// CellSampleCounts(cells, seedCount): out[k] = #{samples whose cellId == k}, one entry per seed (size
// seedCount). The sum equals the total sample count (partition completeness). A seed may own 0 samples
// if it is fully dominated by closer seeds (documented — FR1 does NOT assert every seed owns >=1).
inline std::vector<uint32_t> CellSampleCounts(const FractCells& cells, int seedCount) {
    std::vector<uint32_t> counts((size_t)(seedCount > 0 ? seedCount : 0), 0u);
    for (uint32_t id : cells.cellId)
        if ((int)id < seedCount) ++counts[(size_t)id];
    return counts;
}

// DistinctCellCount(counts): the number of seeds that own at least one sample (the {cells:<>} stat).
inline int DistinctCellCount(const std::vector<uint32_t>& counts) {
    int distinct = 0;
    for (uint32_t c : counts) if (c > 0u) ++distinct;
    return distinct;
}

}  // namespace hf::sim::fract
