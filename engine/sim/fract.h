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

// ====================================================================================================
// Slice FR3 — BONDED-CLUSTER BREAK MODEL (THE NEW PHYSICS — the GR4-friction-equivalent beat). Additive
// over FR1+FR2 (all code above is byte-unchanged). FR1 made the cells, FR2 the fragments; FR3 BONDS
// adjacent fragments into a welded aggregate (shared-face adjacency over the lattice), then BREAKS the
// bonds an impact OVER-STRESSES (a deterministic Jacobi load-diffusion proxy + a strength-scaled Q16.16
// threshold) -> a deterministic SEVERED-BOND SET + a connected-component PIECE COUNT. Emergent crack
// propagation along the pre-fractured cell boundaries.
//
// THE HEADLINE + THE HONEST CAVEAT (state both): the severed-bond SET + the piece count are
// EXACT-DETERMINISTIC and bit-identical CPU<->Vulkan<->Metal (the strong claim). The crack PATTERN (which
// specific bonds break) is EMERGENT / WITHIN-BAND — it depends on the proxy load-diffusion model + the
// tuned kBreakThreshold, NOT an analytic fracture-mechanics solution (the GR4-angle-of-repose caveat
// shape). The claim is determinism + replayability, NOT physical-fracture accuracy. The Jacobi
// single-relaxation load field after K iters is a deterministic approximation (more iters -> more
// diffusion); K is host-fixed for a bounded, deterministic result.
//
// THE INT64 SPLIT (the FPX1/GR3/CL3 precedent): the load-diffusion math is int64 (fxmul/fxdiv of Q16.16
// loads), so shaders/fract_break.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT
// in the Metal hf_gen_msl list; the Metal --fract-break showcase runs THIS CPU ApplyImpactBreak
// (byte-identical by construction). The bond-graph BUILD (BuildFractBonds) is PURE INT32; it is host-built
// in the showcase (the GPU-PROVEN pass is the int64 break — fract_break.comp memcmp's the severed flags +
// per-bond loadAccum). CountFractPieces is PURE INT32 (a stat + the viz cluster colouring).
//
// SEAM DISCIPLINE: unchanged from FR1/FR2 — ZERO backend symbols, header-only. The int64 break math uses
// fpx.h's fxmul/fxdiv (already #included read-only).

// Re-export the int64 fixed-point ops (read-only) for the FR3 break math.
using fpx::fxmul;
using fpx::fxdiv;

// ----- (A) The bond graph (shared-face adjacency over the lattice — pure int32, deterministic) -------
// FractBond: a WELD between two ADJACENT fragments (fragA < fragB, the canonical ascending pair).
//   faceArea  = the count of shared lattice faces between the two fragments' cells (the bond's contact
//               STRENGTH — a large flush interface resists more).
//   midpoint  = the integer mean of the two fragment centroids in lattice coords promoted to Q16.16
//               (carried for the severed-bond viz; render-only, NOT in the bit-exact break math).
//   loadAccum = the Q16.16 diffused load on this bond after the K break iters (the field the break reads).
struct FractBond {
    uint32_t fragA = 0, fragB = 0;     // canonical ascending fragment-index endpoints (fragA < fragB)
    int32_t  faceArea = 0;             // shared lattice-face count (the bond strength) — pure int32
    FxVec3   midpoint{};               // Q16.16 mean of the two centroids (viz-only)
    int64_t  loadAccum = 0;            // Q16.16 diffused load after K iters (the break reads this)
};

// FractBonds: the ascending-(fragA,fragB) bond list + the fragment count it spans.
struct FractBonds {
    std::vector<FractBond> bonds;      // ascending by (fragA, fragB)
    uint32_t fragmentCount = 0;        // == fragments.size() (the cluster-label universe)
};

// BuildFractBonds(field, cells, fragments, out): the +x/+y/+z face-crossing scan -> a dense M×M
// upper-triangle faceArea accumulate (M = fragmentCount, small) -> the ascending-(a,b) bond list.
//   For every lattice sample s, for its +x/+y/+z face-neighbour n (each shared face counted ONCE):
//     a = cellToFragment[ cells.cellId[s] ];  b = cellToFragment[ cells.cellId[n] ];
//     if a==kNoFragment || b==kNoFragment -> skip (a sample of an empty/dominated cell — no fragment);
//     if a==b -> skip (same fragment — not a bond, an interior face);
//     else ++faceArea[min(a,b)][max(a,b)]  (the canonical ordered pair).
//   The bond list = the non-zero upper-triangle entries enumerated in ascending (a,b) order; the midpoint
//   is the Q16.16 mean of the two fragment centroids. PURE INT32 (lattice coords + counts) + a fixed-order
//   scan -> deterministic, no hashing (the navmesh.h canonical-pair/ascending-order discipline, the 3D
//   voxel-face twin). loadAccum starts 0.
inline void BuildFractBonds(const FractField& field, const FractCells& cells,
                            const FractFragments& fragments, FractBonds& out) {
    const uint32_t M = (uint32_t)fragments.fragments.size();
    out.fragmentCount = M;
    out.bonds.clear();
    if (M == 0u) return;

    // Dense M×M upper-triangle faceArea accumulator (M small — e.g. <=24; tiny, fixed-order, no hashing).
    std::vector<int32_t> area((size_t)M * (size_t)M, 0);
    auto At = [&](uint32_t a, uint32_t b) -> int32_t& { return area[(size_t)a * (size_t)M + (size_t)b]; };

    auto fragOf = [&](int sx, int sy, int sz) -> uint32_t {
        const int idx = SampleIndex(field, sx, sy, sz);
        const uint32_t c = cells.cellId[(size_t)idx];
        if (c >= (uint32_t)fragments.cellToFragment.size()) return kNoFragment;
        return fragments.cellToFragment[(size_t)c];
    };

    for (int z = 0; z < field.nz; ++z)
        for (int y = 0; y < field.ny; ++y)
            for (int x = 0; x < field.nx; ++x) {
                const uint32_t a = fragOf(x, y, z);
                if (a == kNoFragment) continue;
                // +x / +y / +z face-neighbours only (each shared face counted exactly once).
                const int nbr[3][3] = {{x + 1, y, z}, {x, y + 1, z}, {x, y, z + 1}};
                const bool inb[3] = {x + 1 < field.nx, y + 1 < field.ny, z + 1 < field.nz};
                for (int d = 0; d < 3; ++d) {
                    if (!inb[d]) continue;
                    const uint32_t b = fragOf(nbr[d][0], nbr[d][1], nbr[d][2]);
                    if (b == kNoFragment || a == b) continue;
                    const uint32_t lo = a < b ? a : b;
                    const uint32_t hi = a < b ? b : a;
                    ++At(lo, hi);
                }
            }

    // Enumerate the non-zero upper-triangle entries in ascending (a,b) order -> the bond list.
    for (uint32_t a = 0; a < M; ++a)
        for (uint32_t b = a + 1u; b < M; ++b) {
            const int32_t fa = At(a, b);
            if (fa <= 0) continue;
            FractBond bond;
            bond.fragA = a; bond.fragB = b;
            bond.faceArea = fa;
            const FractFragment& A = fragments.fragments[(size_t)a];
            const FractFragment& B = fragments.fragments[(size_t)b];
            // Q16.16 mean of the two integer centroids ((ca+cb)/2 * kOne; viz-only).
            bond.midpoint = FxVec3{ (A.cx + B.cx) * (fpx::kOne / 2),
                                    (A.cy + B.cy) * (fpx::kOne / 2),
                                    (A.cz + B.cz) * (fpx::kOne / 2) };
            bond.loadAccum = 0;
            out.bonds.push_back(bond);
        }
}

// ----- (B) The break model (int64 Jacobi load-diffusion + strength-scaled threshold) ----------------
// BreakImpact: a host-supplied impact = a FRAGMENT index + a Q16.16 impulse magnitude (the injected load).
struct BreakImpact {
    uint32_t fragment = 0;             // the fragment the impact lands on
    fx       impulse  = 0;             // Q16.16 load magnitude injected at that fragment
};

// kBreakThreshold: the host-snapped Q16.16 break coefficient. A bond SEVERS iff its diffused loadAccum
// exceeds fxmul(kBreakThreshold, faceArea<<kFrac) — i.e. the per-unit-face break load. Documented choice:
// kBreakThreshold = 0.5 in Q16.16 (kOne/2 = 32768). Tuned so a HARD impact (the showcase's strong impulse)
// over-stresses a coherent subset of bonds near the impact (S>0, P>1 pieces) while a ZERO/sub-threshold
// impact severs nothing (S==0, P==1 intact) — the threshold-gated control. This is a TUNED proxy
// coefficient (the GR4-kFriction/CP2-kBuoy caveat), NOT a material constant.
inline constexpr fx kBreakThreshold = fpx::kOne / 2;   // 0.5 in Q16.16 (32768)

// ApplyImpactBreak(bonds, fragments, impact, K): the deterministic break.
//   (1) Inject impact.impulse as the per-fragment load at impact.fragment (load[other]=0).
//   (2) K JACOBI load-diffusion iters: read iteration-START per-fragment load, write a SEPARATE next-load
//       buffer, then apply. Each INTACT bond transmits a strength-weighted share of the load DIFFERENTIAL
//       between its two fragments toward the lower-loaded fragment (load flows downhill), and the bond
//       ACCUMULATES the transmitted magnitude into loadAccum. Read-start / write-separate / apply == the
//       FL4/GR3 Jacobi discipline -> race-free, multi-thread, NO TDR (fract_break.comp copies this body).
//   (3) After K iters, a bond SEVERS iff loadAccum > fxmul(kBreakThreshold, faceArea<<kFrac).
// Writes bonds[].loadAccum (so the GPU memcmp covers the diffused field) and returns the per-bond severed
// flag (0/1) parallel to bonds.bonds + the severed count. int64 throughout (the load is Q16.16). A
// no-impact / tiny / sub-threshold impulse -> 0 severed (the welded body is intact). Deterministic — the
// fixed bond order + the read-start/write-separate buffers make two runs byte-identical.
inline uint32_t ApplyImpactBreak(FractBonds& bonds, const FractFragments& fragments,
                                 const BreakImpact& impact, int K,
                                 std::vector<uint8_t>& severedOut) {
    (void)fragments;   // the fragment count comes from bonds.fragmentCount; the param keeps the spec'd
                       // signature ApplyImpactBreak(bonds, fragments, impact, K) for FR4 to read centroids.
    const uint32_t M = bonds.fragmentCount;
    const size_t B = bonds.bonds.size();
    severedOut.assign(B, 0u);
    for (auto& bd : bonds.bonds) bd.loadAccum = 0;
    if (M == 0u || B == 0u) return 0u;

    // (1) Inject the impact load at its fragment (Q16.16). All others start at 0.
    std::vector<int64_t> load((size_t)M, 0);
    if (impact.fragment < M) load[(size_t)impact.fragment] = (int64_t)impact.impulse;

    // The per-bond transmission coefficient kFlow (Q16.16): a fraction of the differential moves per iter.
    // 0.25 (kOne/4) — a stable under-relaxation so the deterministic diffusion does not oscillate.
    const int64_t kFlow = (int64_t)(fpx::kOne / 4);   // 0.25 in Q16.16

    // (2) K Jacobi diffusion iterations (read-start / write-separate / apply).
    for (int it = 0; it < K; ++it) {
        std::vector<int64_t> delta((size_t)M, 0);   // SEPARATE next-iter accumulator (the Jacobi buffer)
        for (size_t bi = 0; bi < B; ++bi) {
            FractBond& bd = bonds.bonds[bi];
            const int64_t la = load[(size_t)bd.fragA];
            const int64_t lb = load[(size_t)bd.fragB];
            const int64_t diff = la - lb;            // load differential across the bond (signed)
            const int64_t mag  = diff < 0 ? -diff : diff;  // |differential|
            // Transmitted share: conduct kFlow*mag of the differential per iter (a stable under-relaxed
            // diffusion). A stronger (larger-face) bond is made to RESIST via the strength-scaled
            // THRESHOLD (step 3, which scales WITH faceArea), so the conduction itself stays a simple
            // deterministic int64 proxy. The flow moves toward the lower-loaded fragment.
            const int64_t transmit = (kFlow * mag) >> fpx::kFrac;   // Q16.16 fxmul(kFlow, mag)
            // Accumulate the magnitude carried by this bond (monotone — the total stress it has borne).
            bd.loadAccum += transmit;
            // Diffuse: move `transmit` from the higher- to the lower-loaded fragment in the next buffer.
            if (diff > 0) { delta[(size_t)bd.fragA] -= transmit; delta[(size_t)bd.fragB] += transmit; }
            else if (diff < 0) { delta[(size_t)bd.fragA] += transmit; delta[(size_t)bd.fragB] -= transmit; }
        }
        for (uint32_t f = 0; f < M; ++f) load[(size_t)f] += delta[(size_t)f];
    }

    // (3) Sever iff loadAccum > the strength-scaled threshold fxmul(kBreakThreshold, faceArea<<kFrac).
    uint32_t severed = 0u;
    for (size_t bi = 0; bi < B; ++bi) {
        const FractBond& bd = bonds.bonds[bi];
        const int64_t thresh = (int64_t)fxmul(kBreakThreshold, (fx)((int64_t)bd.faceArea << fpx::kFrac));
        if (bd.loadAccum > thresh) { severedOut[bi] = 1u; ++severed; }
    }
    return severed;
}

// ----- (C) The pieces (connected components over the SURVIVING bonds — pure int32, a stat + the viz) --
// CountFractPieces(fragments, bonds, severed): label-propagate fragments into connected clusters over the
// bonds that DID NOT sever (a deterministic iterate-to-fixpoint min-label propagation — pure int32). The
// cluster count = the number of rigid PIECES the impact produced. 1 = intact; >1 = broken. Fills clusterId
// (one label per fragment) when provided. Deterministic (fixed bond order + min-label propagation).
inline uint32_t CountFractPieces(const FractFragments& fragments, const FractBonds& bonds,
                                 const std::vector<uint8_t>& severed,
                                 std::vector<uint32_t>* clusterIdOut = nullptr) {
    const uint32_t M = (uint32_t)fragments.fragments.size();
    if (M == 0u) { if (clusterIdOut) clusterIdOut->clear(); return 0u; }

    std::vector<uint32_t> label((size_t)M);
    for (uint32_t f = 0; f < M; ++f) label[(size_t)f] = f;   // each fragment its own label initially

    // Iterate to fixpoint: each SURVIVING bond pulls both endpoints to the min of their labels.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t bi = 0; bi < bonds.bonds.size(); ++bi) {
            if (bi < severed.size() && severed[bi]) continue;   // a severed bond no longer connects
            const FractBond& bd = bonds.bonds[bi];
            uint32_t& la = label[(size_t)bd.fragA];
            uint32_t& lb = label[(size_t)bd.fragB];
            const uint32_t m = la < lb ? la : lb;
            if (la != m) { la = m; changed = true; }
            if (lb != m) { lb = m; changed = true; }
        }
    }

    // Count distinct labels = the cluster/piece count.
    std::vector<uint8_t> seen((size_t)M, 0u);
    uint32_t pieces = 0u;
    for (uint32_t f = 0; f < M; ++f) {
        const uint32_t r = label[(size_t)f];
        if (!seen[(size_t)r]) { seen[(size_t)r] = 1u; ++pieces; }
    }
    if (clusterIdOut) *clusterIdOut = label;   // the per-fragment cluster label (the viz colour key)
    return pieces;
}

// ====================================================================================================
// Slice FR4 — THE FRACTURE STEP (released fragments fall). Additive over FR1+FR2+FR3 (ALL code above is
// byte-unchanged). FR1 made the cells, FR2 the fragments, FR3 the break (the severed-bond SET + the piece
// clusters). FR4 turns the break into MOVING RUBBLE: each fragment becomes an independent fpx::FxBody rigid
// body in WORLD UNITS (Q16.16) — the dislodged pieces fall under gravity, collide as bounding spheres, and
// settle into a coherent pile, while the LARGEST (anchor) piece holds STATIC. The object SHATTERS and the
// chunks fall — bit-identical CPU<->Vulkan<->Metal. NO new shader: a host-driven multi-pass driver over the
// EXISTING fpx.h rigid solver (the CP4/CG4/GF4 mold — IntegrateBodyFull + the FPX2 broadphase + the FPX3
// SolveContacts + ResolveGround). This is where the FR1-FR3 integer-lattice world becomes a Q16.16
// world-unit body set.
//
// THE GPU PROOF (NO new shader, the FPX1/GR3 split): StepFracture mirrors fpx_solve.comp's per-step body
// EXACTLY (IntegrateBodyFull then SolveContacts), so the Vulkan --fract-step-shot drives the EXISTING
// shaders/fpx_solve.comp over the fragment bodies (ONE dispatch per tick, host-rebuilding the FPX2 pair
// list each tick from the current positions — the realistic per-tick re-broadphase), and memcmp's the final
// body set vs the CPU StepFractureSteps. fpx_solve.comp is int64 -> Vulkan-only; the Metal --fract-step runs
// the CPU StepFractureSteps (byte-identical by construction). The fragments spawn with angVel==0 + identity
// orient, and IntegrateOrientation with angVel==0 leaves an identity quaternion EXACTLY (omega=0 -> dq=0 ->
// orient unchanged -> FxQuatNormalize(identity)=identity), so IntegrateBodyFull is byte-identical to
// fpx_solve.comp's IntegrateStep on these bodies — the GPU drives fpx_solve.comp with NO orientation pass and
// stays bit-exact to the CPU IntegrateBodyFull reference (the spec's 6-DOF integrate is carried, dormant on
// this scene, ready for spinning rubble in a later slice).
//
// HONEST SIMPLIFICATIONS (the documented first cut, both stated): (1) FPX3 SolveContacts is SPHERE-SPHERE
// (no convex manifold), so each fragment collides as its BOUNDING SPHERE (fragment.boundRadius·cellSize) —
// the rubble is a pile of ROUNDED chunks, NOT interlocking shards (the couple.h body-as-sphere precedent;
// SAT/clipping + an inertia tensor + torque-from-contact does not exist in engine/sim and is the deferred
// refinement). (2) The ANCHOR is the LARGEST piece (most fragments; lowest cluster-id tie-break) — a real
// engine anchors the GROUND-attached piece. Both are deterministic, integer, within-band.
//
// SEAM DISCIPLINE: unchanged — ZERO backend symbols, header-only. FR4 only CALLS fpx.h (read-only).

// ----- (A) The world-unit spawn config (host-fixed Q16.16 constants) --------------------------------
// FractStepConfig carries the lattice->world up-conversion + the gravity/ground + the impact seed. ALL
// host-fixed, pure integer. worldCellSize is the Q16.16 size of ONE lattice cell in world units (the ONLY
// integer up-conversion FR4 introduces — worldPos = latticeCoord·worldCellSize via a single fxmul). impactDir
// is a Q16.16 direction (need NOT be unit — the seeded vel is impactDir·impactSpeed component-wise scaled).
struct FractStepConfig {
    fx     worldCellSize = fpx::kOne;        // Q16.16 world size of one lattice cell (lattice->world scale)
    FxVec3 gravity{};                        // Q16.16 acceleration (e.g. (0, -9.8, 0))
    fx     groundY = 0;                      // Q16.16 ground plane height
    FxVec3 impactDir{};                      // Q16.16 impact direction (vel = impactDir·impactSpeed)
    fx     impactSpeed = 0;                  // Q16.16 impact speed magnitude (seeds the impacted body)
};

// ----- The anchor selection (the LARGEST piece; lowest cluster-id tie-break) ------------------------
// FractAnchorPiece(fragments, clusters): the cluster LABEL of the largest piece = the piece owning the MOST
// fragments (ties broken by the LOWEST cluster label — deterministic total order). That piece's fragments
// become STATIC (the intact base); every other piece's fragments are DYNAMIC (the dislodged rubble). Pure
// int32, fixed scan order. Returns 0xFFFFFFFF if there are no fragments (degenerate — no anchor).
inline uint32_t FractAnchorPiece(const FractFragments& fragments, const std::vector<uint32_t>& clusters) {
    const uint32_t M = (uint32_t)fragments.fragments.size();
    if (M == 0u || (uint32_t)clusters.size() != M) return 0xFFFFFFFFu;
    // Per-cluster fragment count, indexed by cluster LABEL (labels are in [0,M) from CountFractPieces'
    // min-label propagation). A fixed ascending scan -> deterministic.
    std::vector<uint32_t> count((size_t)M, 0u);
    for (uint32_t f = 0; f < M; ++f) {
        const uint32_t c = clusters[(size_t)f];
        if (c < M) ++count[(size_t)c];
    }
    uint32_t bestLabel = 0xFFFFFFFFu;
    uint32_t bestCount = 0u;
    for (uint32_t c = 0; c < M; ++c) {
        if (count[(size_t)c] == 0u) continue;              // not a live cluster root
        // STRICTLY-greater keeps the FIRST (lowest-label) cluster on a tie -> deterministic tie-break.
        if (count[(size_t)c] > bestCount) { bestCount = count[(size_t)c]; bestLabel = c; }
    }
    return bestLabel;
}

// ----- (B) SpawnFractWorld: one fpx::FxBody per fragment, anchor static, impact seeded ---------------
// Builds an fpx::FxWorld from the FR2 fragments + the FR3 break (severed flags + the piece clusters). One
// FxBody per fragment, ordered by ascending fragment index (the FR2 compact order):
//   pos     = (cx,cy,cz)·worldCellSize           (the lattice centroid up-converted to Q16.16 world units)
//   radius  = boundRadius·worldCellSize          (the bounding sphere in world units — the FPX3 collider)
//   invMass = fragment.invMass                   (FR2's Q16.16 reciprocal of the voxel count)
//   orient  = identity, angVel = 0               (no initial spin; the 6-DOF integrate is dormant here)
// The ANCHOR rule: the LARGEST piece's fragments are STATIC (invMass=0, NOT kFlagDynamic — the intact base);
// every OTHER fragment is DYNAMIC (kFlagDynamic). The IMPACTED fragment's body (impact.fragment) is seeded
// with vel = impactDir·impactSpeed (component-wise fxmul) IF it is dynamic (an anchor fragment is never
// kicked loose). `severed` is unused by the spawn itself (the clusters already encode the break) but is kept
// in the signature for the spec'd call shape. Pure integer (fxmul by host constants). Empty fragments ->
// empty world.
inline fpx::FxWorld SpawnFractWorld(const FractFragments& fragments, const FractBonds& bonds,
                                    const std::vector<uint8_t>& severed,
                                    const std::vector<uint32_t>& clusters,
                                    const BreakImpact& impact, const FractStepConfig& cfg) {
    (void)bonds; (void)severed;   // the clusters already encode the connectivity result; kept per the spec.
    fpx::FxWorld world;
    world.gravity = cfg.gravity;
    world.groundY = cfg.groundY;

    const uint32_t M = (uint32_t)fragments.fragments.size();
    if (M == 0u) return world;

    const uint32_t anchor = FractAnchorPiece(fragments, clusters);
    const bool haveClusters = ((uint32_t)clusters.size() == M);

    world.bodies.reserve((size_t)M);
    for (uint32_t f = 0; f < M; ++f) {
        const FractFragment& fr = fragments.fragments[(size_t)f];
        fpx::FxBody b;
        // Lattice centroid -> Q16.16 world position. cx/cy/cz are small integers; promote to Q16.16 then
        // fxmul by the cell size: world = (cx<<kFrac) * worldCellSize >> kFrac == cx · worldCellSize.
        b.pos = FxVec3{ fpx::fxmul((fx)((int64_t)fr.cx << fpx::kFrac), cfg.worldCellSize),
                        fpx::fxmul((fx)((int64_t)fr.cy << fpx::kFrac), cfg.worldCellSize),
                        fpx::fxmul((fx)((int64_t)fr.cz << fpx::kFrac), cfg.worldCellSize) };
        b.vel = FxVec3{0, 0, 0};
        b.radius = fpx::fxmul((fx)((int64_t)fr.boundRadius << fpx::kFrac), cfg.worldCellSize);
        b.invMass = fr.invMass;
        b.orient = fpx::FxQuat{0, 0, 0, fpx::kOne};   // identity
        b.angVel = FxVec3{0, 0, 0};

        // The anchor rule: the largest piece's fragments are STATIC; all others DYNAMIC.
        const bool isAnchor = haveClusters && (clusters[(size_t)f] == anchor) && (anchor != 0xFFFFFFFFu);
        if (isAnchor) {
            b.invMass = 0;          // static: infinite mass, never integrated
            b.flags = 0;            // NOT dynamic
        } else {
            b.flags = fpx::kFlagDynamic;
        }
        world.bodies.push_back(b);
    }

    // Seed the impacted fragment's body with the impact velocity (only if it is a dynamic chunk).
    if (impact.fragment < M) {
        fpx::FxBody& hit = world.bodies[(size_t)impact.fragment];
        if (hit.flags & fpx::kFlagDynamic) {
            hit.vel = FxVec3{ fpx::fxmul(cfg.impactDir.x, cfg.impactSpeed),
                              fpx::fxmul(cfg.impactDir.y, cfg.impactSpeed),
                              fpx::fxmul(cfg.impactDir.z, cfg.impactSpeed) };
        }
    }
    return world;
}

// ----- (C) StepFracture: ONE deterministic tick (the CP4/CG4/GF4 driver over fpx VERBATIM) ----------
// Mirrors fpx_solve.comp's per-step body EXACTLY so the GPU --fract-step-shot can drive that shader (steps=1)
// and stay bit-exact:
//   (a) INTEGRATE each body via IntegrateBodyFull (the FPX4 6-DOF: gravity + linear + orientation tumble),
//       then the FPX1 floor clamp (which IntegrateBodyFull omits — fpx_solve.comp's pass A does it) so the
//       composition == fpx_solve.comp pass A (translate+clamp) + the identity-preserving orientation step.
//   (b) BROADPHASE: CountPairs/BuildPairs rebuilt from the CURRENT positions (the realistic per-tick
//       re-broadphase; the GPU host rebuilds the SAME deterministic list each tick).
//   (c) SOLVE: SolveContacts(world, pairs, solveIters) — the FPX3 sphere-sphere non-penetration Gauss-Seidel
//       (ground-then-pairs each sweep), reused VERBATIM.
// Static bodies (the anchor) never move (IntegrateBodyFull gates translation on kFlagDynamic; ResolveGround
// inside SolveContacts skips invMass==0). Pure integer, fixed op order -> two runs byte-identical AND the GPU
// fpx_solve.comp result is byte-identical to this. Reuses fpx.h with ZERO modification.
inline void StepFracture(fpx::FxWorld& world, fx dt, int solveIters) {
    // (a) integrate (6-DOF) + the FPX1 floor clamp (== fpx_solve.comp pass A on these angVel==0 bodies).
    for (fpx::FxBody& b : world.bodies) {
        fpx::IntegrateBodyFull(b, world.gravity, dt);
        if (b.flags & fpx::kFlagDynamic) {
            if (b.pos.y < world.groundY) {
                b.pos.y = world.groundY;
                if (b.vel.y < 0) b.vel.y = 0;
            }
        }
    }
    // (b) re-broadphase from the current positions.
    std::vector<uint32_t> offsets;
    std::vector<fpx::FxPair> pairs;
    fpx::BuildPairs(world, offsets, pairs);
    // (c) the FPX3 sphere-sphere solve (ground-then-pairs, K sweeps).
    fpx::SolveContacts(world, std::span<const fpx::FxPair>(pairs), solveIters);
}

// StepFractureSteps(world, dt, solveIters, steps): run K StepFracture ticks. The CPU reference the GPU
// fpx_solve.comp per-tick driver memcmp's against. Pure integer -> two runs byte-identical, cross-backend
// identical.
inline void StepFractureSteps(fpx::FxWorld& world, fx dt, int solveIters, int steps) {
    for (int s = 0; s < steps; ++s) StepFracture(world, dt, solveIters);
}

// ----- (D) MeasureFractRubble: the honest emergent-metrics helper (the CG4 MeasureCGrainState twin) -----
// Deterministic Q16.16 stats over the settled world: the rest line of the dynamic chunks (mean pos.y), the
// anchor body's pos.y, and the settled/airborne split. "settled" = a dynamic body whose bottom (pos.y -
// radius) is within a small band of the ground (kRestBand) — resting on the floor/pile; "airborne" = the
// rest. anchorIndex is the body index of the static anchor (or a sentinel >= bodies.size() if none) — its
// pos.y is reported verbatim. Pure integer (a fixed-order sum + an integer mean).
struct FractRubbleState {
    fx       meanDynamicY = 0;   // Q16.16 mean pos.y over the DYNAMIC bodies (the rubble rest line)
    fx       anchorY = 0;        // Q16.16 pos.y of the anchor body (unchanged from spawn — static)
    uint32_t dynamic = 0;        // # dynamic bodies
    uint32_t settled = 0;        // # dynamic bodies resting on/near the ground/pile
    uint32_t airborne = 0;       // # dynamic bodies still above the rest band
    fx       minDynamicY = 0;    // lowest dynamic pos.y (the pile floor)
};
inline FractRubbleState MeasureFractRubble(const fpx::FxWorld& world, uint32_t anchorIndex) {
    FractRubbleState st;
    const fx kRestBand = fpx::kOne;   // 1.0 world unit: a chunk within 1 unit of the floor counts as settled
    int64_t sumY = 0;
    fx minY = (fx)0x7FFFFFFF;
    for (size_t i = 0; i < world.bodies.size(); ++i) {
        const fpx::FxBody& b = world.bodies[i];
        if (!(b.flags & fpx::kFlagDynamic)) continue;
        ++st.dynamic;
        sumY += (int64_t)b.pos.y;
        if (b.pos.y < minY) minY = b.pos.y;
        // settled iff the body's bottom is within kRestBand of the ground (rests on floor or on a chunk
        // that is itself near the floor — a coherent low pile).
        const fx bottom = b.pos.y - b.radius;
        if (bottom <= world.groundY + kRestBand) ++st.settled; else ++st.airborne;
    }
    if (st.dynamic > 0u) { st.meanDynamicY = (fx)(sumY / (int64_t)st.dynamic); st.minDynamicY = minY; }
    if (anchorIndex < (uint32_t)world.bodies.size()) st.anchorY = world.bodies[(size_t)anchorIndex].pos.y;
    return st;
}

// FractAnchorBodyIndex(fragments, clusters): the BODY INDEX (== fragment index) of the FIRST fragment that
// belongs to the anchor piece — the body whose pos.y MeasureFractRubble reports. Deterministic (ascending
// scan). Returns bodies.size()-equivalent sentinel 0xFFFFFFFF if no anchor.
inline uint32_t FractAnchorBodyIndex(const FractFragments& fragments,
                                     const std::vector<uint32_t>& clusters) {
    const uint32_t anchor = FractAnchorPiece(fragments, clusters);
    if (anchor == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    const uint32_t M = (uint32_t)fragments.fragments.size();
    for (uint32_t f = 0; f < M && (uint32_t)clusters.size() == M; ++f)
        if (clusters[(size_t)f] == anchor) return f;
    return 0xFFFFFFFFu;
}

// ====================================================================================================
// Slice FR5 — LOCKSTEP + ROLLBACK over the fracture rubble dynamics (THE NETCODE HEADLINE — PURE CPU).
// Additive over FR1-FR4 (ALL code above is byte-unchanged). FR4 made the rubble FALL + settle; FR5 proves
// that settle is true cross-platform LOCKSTEP + ROLLBACK: a peer fed the INPUT command stream ALONE re-
// derives the authority's exact destroyed/settled state bit-for-bit (every dislodged chunk's tumble + the
// settled pile), and a mispredicted input is corrected by rolling back to a saved snapshot + re-simulating.
// UE5's float Chaos fracture cannot replay a break bit-for-bit; this can.
//
// THE APPROACH — MAXIMAL REUSE (the lowest-risk slice): the fracture world IS an fpx::FxWorld (FR4's
// SpawnFractWorld output). fpx already ships the bit-exact FPX5 lockstep/rollback machinery over FxWorld —
// fpx::FxCommand (an input impulse/spin on a body), fpx::ApplyCommand, fpx::SnapshotWorld (deep-copy the
// world), fpx::RestoreWorld. FR5 REUSES ALL of it VERBATIM and changes ONE thing: the per-tick step is
// fract::StepFracture (FR4) instead of fpx's StepWorld+IntegrateOrientation. So FR5 is a THIN harness:
// SimFractTick is the fpx::SimTick twin with StepFracture substituted; RunFractLockstep/RunFractRollback
// mirror fpx::RunLockstep/RunRollback's control flow EXACTLY but call SimFractTick. Pure integer, fixed op
// order -> bit-identical on every peer/platform (and cross-backend by StepFracture's proven bit-exactness).
//
// WHAT IS REPLAYED (the break is in the init; the SETTLE is replayed): the DESTRUCTION = the break (FR3) +
// the spawn (FR4 SpawnFractWorld). That break is itself deterministic + bit-reproducible (a fixed impact ->
// a fixed severed set -> a fixed dynamic/static body assignment), so the `init` FxWorld two peers start from
// is identical by construction. FR5 replays the RUBBLE DYNAMICS — every dislodged chunk's fall, collision,
// tumble, and settle — from the input shove stream, bit-for-bit. The bond/severed state is fixed after the
// initial break (it lives in the init's body flags), so it does not change during the settle. (Driving a
// FURTHER break mid-replay from a stream command is a documented extension; FR5 replays the post-break
// settle, the faithful version that matches FPX5/GR5/CG5/GF5.) The command is a body shove (kCmdImpulse) /
// spin (kCmdSetAngVel) on a dislodged chunk — "kick a falling chunk", re-simulated identically on two peers.
//
// SEAM DISCIPLINE: unchanged — ZERO backend symbols, header-only, PURE CPU. NO new shader, NO new RHI. FR5
// only ADDS the three harness functions; it reuses fpx.h's FxCommand/ApplyCommand/SnapshotWorld/RestoreWorld
// (already #included read-only) — it does NOT re-implement them.

// Re-export the fpx FPX5 command + snapshot primitives (read-only — REUSED VERBATIM, not re-implemented).
using fpx::FxCommand;
using fpx::kCmdImpulse;
using fpx::kCmdSetAngVel;
using fpx::ApplyCommand;
using fpx::SnapshotWorld;
using fpx::RestoreWorld;

// SimFractTick(world, stream, tick, dt, solveIters): the deterministic per-tick step (the fpx::SimTick twin
// with StepFracture substituted for StepWorld+IntegrateOrientation). (1) apply ALL `stream` commands whose
// .tick == `tick`, in ARRAY ORDER (the deterministic input-order contract — the same order on every peer/
// platform) via fpx::ApplyCommand; (2) StepFracture(world, dt, solveIters) — the FR4 tick (IntegrateBodyFull
// + the FPX1 floor clamp -> per-tick re-broadphase -> the FPX3 SolveContacts), reused VERBATIM. Pure
// integer, fixed order -> bit-identical on every peer/platform. (No new command type — reuses fpx::FxCommand.)
inline void SimFractTick(fpx::FxWorld& w, const std::vector<fpx::FxCommand>& stream, uint32_t tick, fx dt,
                         int solveIters) {
    for (const fpx::FxCommand& c : stream)
        if (c.tick == tick) fpx::ApplyCommand(w, c);
    StepFracture(w, dt, solveIters);
}

// RunFractLockstep(init, stream, ticks, dt, solveIters): THE peer entry point (the fpx::RunLockstep control
// flow over SimFractTick). Run `ticks` SimFractTicks from a COPY of `init`, applying the command stream ->
// the final rubble world. authority = RunFractLockstep(init, stream, N); replica = RunFractLockstep(init,
// stream, N) from the SAME init + stream (INPUTS ONLY — no state shared) -> BIT-IDENTICAL by determinism
// (the lockstep proof memcmps them).
inline fpx::FxWorld RunFractLockstep(const fpx::FxWorld& init, const std::vector<fpx::FxCommand>& stream,
                                     int ticks, fx dt, int solveIters) {
    fpx::FxWorld w = init;
    for (int t = 0; t < ticks; ++t)
        SimFractTick(w, stream, (uint32_t)t, dt, solveIters);
    return w;
}

// RunFractRollback(init, authStream, mispredictStream, ticks, mispredictTick, dt, solveIters): the rollback
// harness (the fpx::RunRollback control flow over SimFractTick). (1) run ticks 0..mispredictTick from `init`
// applying authStream; (2) SAVE a snapshot AT mispredictTick (fpx::SnapshotWorld — the FxWorld deep-copy);
// (2b) speculatively advance <=3 ticks with the MISPREDICTED stream (the wrong input — the client prediction
// that diverges); (3) ROLLBACK — fpx::RestoreWorld to the snapshot + RE-SIMULATE mispredictTick..ticks with
// the CORRECT authStream -> the corrected final world. The proof asserts this == RunFractLockstep(init,
// authStream, ticks) (rollback corrected the misprediction EXACTLY) AND that the speculative pre-rollback
// state DIFFERED from authority (a real divergence was fixed). Reuses fpx::SnapshotWorld/RestoreWorld VERBATIM.
inline fpx::FxWorld RunFractRollback(const fpx::FxWorld& init, const std::vector<fpx::FxCommand>& authStream,
                                     const std::vector<fpx::FxCommand>& mispredictStream, int ticks,
                                     int mispredictTick, fx dt, int solveIters) {
    fpx::FxWorld w = init;
    // (1) advance 0..mispredictTick with the authoritative stream.
    for (int t = 0; t < mispredictTick; ++t)
        SimFractTick(w, authStream, (uint32_t)t, dt, solveIters);
    // (2) SAVE the snapshot at mispredictTick (the rollback restore point).
    const fpx::FxWorld snap = fpx::SnapshotWorld(w);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (bounded to the remaining ticks).
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimFractTick(w, mispredictStream, (uint32_t)(mispredictTick + s), dt, solveIters);
    // (3) ROLLBACK: restore the snapshot + re-simulate mispredictTick..ticks with the CORRECT authStream.
    fpx::RestoreWorld(w, snap);
    for (int t = mispredictTick; t < ticks; ++t)
        SimFractTick(w, authStream, (uint32_t)t, dt, solveIters);
    return w;
}

}  // namespace hf::sim::fract
