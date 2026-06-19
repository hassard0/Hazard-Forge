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

// ====================================================================================================
// Slice FR2 — FRAGMENT EXTRACTION (per-cell mass properties). Additive over FR1 (FR1 code above is
// byte-unchanged). Turns each NON-EMPTY FR1 Voronoi cell into a FRAGMENT RECORD — integer centroid,
// AABB, bounding-sphere radius², volume, invMass — via the GR2/FL2 count->scan->emit CSR compaction
// (grain.h:301 BuildGrainCellTable) over FR1's cellId[] array (cells == seedCount, the sample->cell map
// is cells.cellId[sample] DIRECTLY, no CellOf recompute), then ONE fragment per non-empty cell reducing
// its CSR slice in ASCENDING sample-index order. The output is a COMPACT array of F <= seedCount
// fragments ordered by ASCENDING cell index (the CL2/MC3 stream-compaction) + a cellToFragment[] remap
// (cell -> fragment index, or kNoFragment for an empty/dominated cell).
//
// THE CROSS-BACKEND CRUX — PURE INT32 -> MSL-NATIVE. Centroid/AABB/boundRadiusSq/volume are all small
// integers: lattice coords < ~1024, squared distances 3*1023² ~ 3.1M fit int32 with headroom (the FR1
// bound). NO int64, NO float, NO Q16.16 world scaling at FR2 -> the FR2 shaders MSL-generate natively (a
// TRUE GPU pass on both Vulkan AND Metal, the strict zero-differing-pixel bar). World-unit Q16.16
// fragment positions (for FR4's fpx::FxBody spawn) are DEFERRED to FR4 (worldPos = latticeCoord *
// worldCellSize); FR2 stays integer lattice-space. The fixed ascending reduction order makes the integer
// sums/min/max bit-identical CPU<->GPU<->cross-vendor; each fragment writes ONLY its own record ->
// per-fragment disjoint, race-free, NO atomics in the reduction.

// kNoFragment: the cellToFragment[] sentinel for an empty/dominated cell (no fragment emitted for it).
inline constexpr uint32_t kNoFragment = 0xFFFFFFFFu;

// SampleCoord(field, idx): the (x,y,z) lattice coord of flat sample index `idx` — the INVERSE of
// SampleIndex (x = idx % nx; y = (idx / nx) % ny; z = idx / (nx*ny)). Pure int32 helper. Round-trips
// SampleIndex (SampleIndex(f, SampleCoord(f, i)...) == i for i in [0, sampleCount)).
struct FractCoord { int x, y, z; };
inline FractCoord SampleCoord(const FractField& f, int idx) {
    FractCoord c;
    c.x =  idx % f.nx;
    c.y = (idx / f.nx) % f.ny;
    c.z =  idx / (f.nx * f.ny);
    return c;
}

// ISqrt32(v): deterministic integer floor(sqrt(v)) for a non-negative int32 (pure int32 binary
// digit-by-digit — identical on every compiler/vendor, NO <cmath>, NO int64). The int32 twin of
// render/mc.h::ISqrt; used ONCE per fragment for boundRadius = ISqrt32(boundRadiusSq). boundRadiusSq
// (< ~3.1M) fits int32, so the int32 form is exact and MSL-native.
inline int32_t ISqrt32(int32_t v) {
    if (v <= 0) return 0;
    int32_t bit = (int32_t)1 << 30;           // highest even power-of-4 within int32
    while (bit > v) bit >>= 2;
    int32_t res = 0;
    while (bit != 0) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else { res >>= 1; }
        bit >>= 2;
    }
    return res;
}

// FractInvMass(volume): the per-fragment inverse mass, a PURE-INTEGER Q16.16 reciprocal of the integer
// voxel count — invMass = kOne / volume (an integer truncating divide; kOne=65536, volume>=1, so this is
// pure int32, NO fxdiv/int64). Mass is proportional to the voxel count (uniform density), so invMass is a
// deterministic integer function of volume. volume==0 (never emitted — empty cells produce no fragment)
// -> 0 (a static/immovable guard). Documented derivation: invMass = 1/mass with mass == volume voxels.
inline fx FractInvMass(uint32_t volume) {
    return volume > 0u ? (fx)(fpx::kOne / (int32_t)volume) : (fx)0;
}

// ----- The per-fragment record (integer lattice-space; FR4 maps to world-unit fpx::FxBody) ----------
// Every field is pure-integer-derivable from the cell's member samples (the documented FR2 claim — a
// centroid + AABB + bounding sphere, NOT a triangulated hull; the sphere-bound fragment is the honest
// first cut, the FPX3-sphere-contact reuse). cellId is the source FR1 seed index; volume is the integer
// voxel count; invMass = FractInvMass(volume).
struct FractFragment {
    int32_t cx = 0, cy = 0, cz = 0;                       // integer centroid (Sum lattice-pos / count)
    int32_t minx = 0, miny = 0, minz = 0;                 // AABB min (per-axis member min)
    int32_t maxx = 0, maxy = 0, maxz = 0;                 // AABB max (per-axis member max)
    int32_t boundRadiusSq = 0;                            // max member squared-dist to centroid (int32)
    int32_t boundRadius = 0;                              // ISqrt32(boundRadiusSq), once per fragment
    uint32_t volume = 0;                                  // integer voxel count (== member count)
    uint32_t cellId = 0;                                  // the source FR1 seed index this fragment owns
    fx invMass = 0;                                       // FractInvMass(volume) — Q16.16, pure-integer
};

// ----- The CSR + compact fragment array + remap (the ExtractFragments output) ----------------------
// fragStart[]/fragSamples[] are the GR2 CSR over cellId (one ROW per CELL, cells == seedCount): cell c's
// member sample indices are fragSamples[fragStart[c] .. fragStart[c+1]), ASCENDING sample index.
// fragments[] is the COMPACT array of F <= seedCount fragments, one per NON-EMPTY cell, ordered by
// ASCENDING cell index. cellToFragment[c] is the cell->fragment remap (or kNoFragment if cell c is
// empty). fragmentToCell[f] is the inverse (fragment f owns cell fragmentToCell[f]) — the per-fragment
// reduce reads it to find its CSR slice.
struct FractFragments {
    std::vector<uint32_t> fragStart;        // seedCount+1 exclusive prefix-sum (CSR row pointers)
    std::vector<uint32_t> fragSamples;      // sample indices grouped by cell (size sampleCount)
    std::vector<uint32_t> cellToFragment;   // seedCount entries; cell -> fragment idx or kNoFragment
    std::vector<uint32_t> fragmentToCell;   // F entries; fragment -> its source cell index
    std::vector<FractFragment> fragments;   // F <= seedCount compact fragment records (ascending cell)
};

// ----- ExtractFragments: count->scan->emit CSR over cellId + per-fragment reduction (the CPU reference)
// The byte-for-byte reference the GPU FR2 passes memcmp against (the CSR + the remap + the fragment
// array). Mirrors grain.h::BuildGrainCellTable (count->scan->emit) with cells == seedCount and the
// sample->cell map = cells.cellId[sample] directly:
//   (1) COUNT  per-cell sample count (ascending sample loop).
//   (2) SCAN   exclusive prefix-sum -> fragStart[seedCount+1] (sentinel == sampleCount); SAME pass
//              assigns each NON-EMPTY cell its compact fragment index (ascending cell) -> cellToFragment
//              + fragmentToCell.
//   (3) EMIT   single-thread ASCENDING-sample scatter each sample into its cell's slice (the GR2
//              within-cell ascending order — deterministic).
//   (4) REDUCE one fragment per non-empty cell: walk its CSR slice [fragStart[c], fragStart[c+1]) in
//              ascending sample order -> centroid = Sum pos / count (per-axis truncating divide), AABB =
//              per-axis min/max, boundRadiusSq = max (pos-centroid)·(pos-centroid), volume = count,
//              boundRadius = ISqrt32(boundRadiusSq), invMass = FractInvMass(volume). Each fragment writes
//              ONLY its own record -> disjoint, race-free (the GPU one-thread-per-fragment mirror).
inline void ExtractFragments(const FractField& field, const FractCells& cells, int seedCount,
                             FractFragments& out) {
    const int sampleCount = field.sampleCount();
    const uint32_t cellsN = (uint32_t)(seedCount > 0 ? seedCount : 0);

    // (1) COUNT per-cell sample count.
    std::vector<uint32_t> counts((size_t)cellsN, 0u);
    for (int s = 0; s < sampleCount; ++s) {
        const uint32_t c = cells.cellId[(size_t)s];
        if (c < cellsN) ++counts[(size_t)c];
    }

    // (2) SCAN exclusive prefix-sum -> fragStart (cellsN+1, sentinel == sampleCount); assign compact
    //     fragment indices to non-empty cells in ascending cell order.
    out.fragStart.assign((size_t)cellsN + 1u, 0u);
    out.cellToFragment.assign((size_t)cellsN, kNoFragment);
    out.fragmentToCell.clear();
    uint32_t running = 0u;
    uint32_t fragCount = 0u;
    for (uint32_t c = 0; c < cellsN; ++c) {
        out.fragStart[c] = running;
        if (counts[(size_t)c] > 0u) {
            out.cellToFragment[(size_t)c] = fragCount;
            out.fragmentToCell.push_back(c);
            ++fragCount;
        }
        running += counts[(size_t)c];
    }
    out.fragStart[cellsN] = running;   // sentinel: == sampleCount (every sample owned by exactly one cell)

    // (3) EMIT ascending-sample scatter into each cell's slice (the GR2 single-thread ascending cursor).
    out.fragSamples.assign((size_t)sampleCount, 0u);
    std::vector<uint32_t> cursor((size_t)cellsN, 0u);
    for (int s = 0; s < sampleCount; ++s) {
        const uint32_t c = cells.cellId[(size_t)s];
        if (c >= cellsN) continue;
        out.fragSamples[(size_t)(out.fragStart[c] + cursor[(size_t)c])] = (uint32_t)s;
        ++cursor[(size_t)c];
    }

    // (4) REDUCE one fragment per non-empty cell over its CSR slice (ascending sample order).
    out.fragments.assign((size_t)fragCount, FractFragment{});
    for (uint32_t f = 0; f < fragCount; ++f) {
        const uint32_t c = out.fragmentToCell[(size_t)f];
        const uint32_t begin = out.fragStart[(size_t)c];
        const uint32_t end   = out.fragStart[(size_t)c + 1u];
        const uint32_t count = end - begin;

        // Pass A: centroid (sum / count) + AABB.
        int32_t sx = 0, sy = 0, sz = 0;
        int32_t minx = INT32_MAX, miny = INT32_MAX, minz = INT32_MAX;
        int32_t maxx = INT32_MIN, maxy = INT32_MIN, maxz = INT32_MIN;
        for (uint32_t k = begin; k < end; ++k) {
            const FractCoord p = SampleCoord(field, (int)out.fragSamples[(size_t)k]);
            sx += p.x; sy += p.y; sz += p.z;
            if (p.x < minx) minx = p.x; if (p.x > maxx) maxx = p.x;
            if (p.y < miny) miny = p.y; if (p.y > maxy) maxy = p.y;
            if (p.z < minz) minz = p.z; if (p.z > maxz) maxz = p.z;
        }
        const int32_t cx = sx / (int32_t)count;   // truncating integer divide (deterministic)
        const int32_t cy = sy / (int32_t)count;
        const int32_t cz = sz / (int32_t)count;

        // Pass B: boundRadiusSq = max member squared-dist to centroid.
        int32_t brSq = 0;
        for (uint32_t k = begin; k < end; ++k) {
            const FractCoord p = SampleCoord(field, (int)out.fragSamples[(size_t)k]);
            const int32_t dx = p.x - cx, dy = p.y - cy, dz = p.z - cz;
            const int32_t d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > brSq) brSq = d2;
        }

        FractFragment& fr = out.fragments[(size_t)f];
        fr.cx = cx; fr.cy = cy; fr.cz = cz;
        fr.minx = minx; fr.miny = miny; fr.minz = minz;
        fr.maxx = maxx; fr.maxy = maxy; fr.maxz = maxz;
        fr.boundRadiusSq = brSq;
        fr.boundRadius = ISqrt32(brSq);
        fr.volume = count;
        fr.cellId = c;
        fr.invMass = FractInvMass(count);
    }
}

}  // namespace hf::sim::fract
