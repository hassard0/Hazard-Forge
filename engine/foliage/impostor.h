#pragma once
// Slice FO1 (impostor arc) — DETERMINISTIC FOLIAGE IMPOSTORS + CROSS-FADE LOD. Header-only, namespace
// hf::foliage::impostor. PURE CPU — NO device, NO backend symbols, NO new RHI, NO shader. This is the last
// audit item (FO-B): foliage.h's FO4 LOD hard-swaps a distant instance to a single "billboard stand-in
// (honest v1)" — the LOD switch POPS, and there is NO octahedral impostor atlas addressing. UE5 uses
// octahedral impostors + dithered cross-fade LOD. This header adds the DETERMINISTIC CORE of both:
//
//   (1) OCTAHEDRAL IMPOSTOR ADDRESSING — OctEncode(unitDir) maps a view direction to a [-1,1]^2 octahedral
//       square (the standard Cigolle et al. unit-vector<->octahedron map, folding the lower hemisphere),
//       ImpostorCell(dir,gridN) buckets it to the nearest cell of an N×N impostor atlas. This is the exact
//       addressing an octahedral impostor atlas uses to pick which captured view faces the camera. FO1 pins
//       the CELL SELECTION only; the actual GPU atlas BAKE + textured billboard render is DEFERRED to FO2.
//
//   (2) DITHERED CROSS-FADE LOD — CrossFadeWeight(dist,threshold,band) replaces FO4's hard swap with a
//       TRANSITION BAND [threshold-band, threshold+band] over which both LODs are active with an integer
//       0->kOne weight, and an ordered Bayer-4x4 screen-door dither (+ a per-instance hash threshold) that
//       converts the fractional weight into a STABLE per-pixel / per-instance binary coverage — so the
//       transition is deterministic and shimmer-free (a fixed weight => the identical mask every frame).
//
// STRICT-ZERO INTEGER DISCIPLINE: everything here is Q16.16 / integer and bit-identical CPU<->Vulkan<->Metal
// BY CONSTRUCTION — the octahedral map is integer fixed-point (division truncates toward zero, DEFINED and
// identical on every compiler/vendor), the dither is an integer Bayer matrix + the ParticleHash-shaped
// avalanche, and the LOD distance selection is the fpx.h int64 FxLength path. NO runtime transcendentals, NO
// float, NO clock/RNG. The moat: UE5's octahedral-impostor + cross-fade LOD selection is float/non-
// deterministic; this addressing + cross-fade is a pure function of (dir, dist, ids), so two netcode peers
// address the byte-identical atlas cell and dither the byte-identical transition.
//
// COMPOSES foliage.h READ-ONLY (FxVec3 / fx / kOne / fxmul / FxLength / FoliageInstance). foliage.h is
// BYTE-UNTOUCHED. Reuses NO new placement math — the scene lays a deterministic lattice of FoliageInstance
// bases (the FO4 distance-LOD path) and flies a camera through it.
//
// HONEST SCOPE / KNOWN LIMITS (documented, not hidden):
//   * The octahedral map has the standard DISTORTION + a discontinuity along the fold seams (the |u|+|v|==1
//     diagonals where the hemispheres meet) — cells near the seam cover a warped solid-angle. This is
//     inherent to octahedral parameterization; FO1 pins it, FO2's atlas capture lives with it as UE5 does.
//   * The round-trip OctDecode(OctEncode(dir)) recovers dir only WITHIN a pinned LSB band (two integer
//     divides + a floor-sqrt normalize), NOT bit-exact — the cross-COMPILER identity is the strict-zero
//     guarantee, the round-trip band is a correctness sanity bound.
//   * The dither is ORDERED / screen-door (spatially stable), NOT temporal — no history, no blue-noise.
//   * The GPU octahedral-atlas BAKE + textured billboard render is DEFERRED to FO2 — FO1 ships the
//     deterministic ADDRESSING + CROSS-FADE core ONLY. This header does NOT render impostors.

#include <cstdint>
#include <vector>

#include "foliage/foliage.h"   // read-only: FxVec3 / fx / kOne / fxmul / FxLength / FoliageInstance

namespace hf::foliage {
namespace impostor {

using hf::foliage::fx;
using hf::foliage::kOne;
using hf::foliage::fxmul;
using hf::foliage::FxVec3;
using hf::sim::fpx::FxLength;

// ============================================================================================================
// (1) OCTAHEDRAL IMPOSTOR ADDRESSING  (pure integer Q16.16 unit-vector <-> octahedral square)
// ============================================================================================================
// CONVENTION (documented so the atlas bake in FO2 matches): the octahedral square coordinates are (dir.x,
// dir.z); the fold hemisphere is decided by dir.y (the VERTICAL axis is the pole). So the UP pole (+Y)
// maps to the square CENTER (0,0) and the DOWN pole (-Y) maps to the four CORNERS (±1,±1). This is the
// Cigolle/Meyer octahedral map with the roles of the fold axis assigned to Y (the natural pole for a plant
// viewed from around its vertical). SignNotZero returns +1 for 0 (so the pole folds to a corner, not the
// origin) — the load-bearing detail of the standard map.

struct OctCoord { fx u = 0, v = 0; };   // a point of the octahedral square, each in [-kOne, kOne]
struct OctCell  { int32_t cu = 0, cv = 0; };   // an N×N atlas cell (column, row), each in [0, gridN-1]

// SignNotZero: +1 for v>=0 (INCLUDING 0), -1 for v<0. The standard octahedral-wrap sign — a 0 component
// must fold to +1 so a pole maps to a corner, not collapse to the origin.
inline fx OctSignNotZero(fx v) { return (v >= 0) ? kOne : -kOne; }
inline fx OctAbs(fx v) { return (v < 0) ? -v : v; }
// ApplySign(mag, sign): stamp a ±kOne sign onto a non-negative magnitude.
inline fx OctApplySign(fx mag, fx sign) { return (sign >= 0) ? mag : -mag; }

// OctEncode(dir): map a direction (need NOT be pre-normalized — the L1 norm normalizes it) to the octahedral
// square [-kOne,kOne]^2. Two integer divides (truncate toward zero, DEFINED + identical on MSVC/clang). The
// fold: for the lower hemisphere (dir.y < 0) reflect across the diagonals via OctWrap. Pure integer.
inline OctCoord OctEncode(const FxVec3& dir) {
    const int64_t denom = (int64_t)OctAbs(dir.x) + (int64_t)OctAbs(dir.y) + (int64_t)OctAbs(dir.z);
    if (denom == 0) return OctCoord{0, 0};   // the zero vector -> the center (a defined degenerate)
    // Project onto the plane: u = x / |dir|_1, v = z / |dir|_1, each in [-kOne, kOne].
    fx u = (fx)(((int64_t)dir.x * (int64_t)kOne) / denom);
    fx v = (fx)(((int64_t)dir.z * (int64_t)kOne) / denom);
    if (dir.y < 0) {
        // OctWrap((u,v)) = (1 - |v|, 1 - |u|) * signNotZero(u,v) — fold the lower hemisphere onto the ring.
        const fx fu  = kOne - OctAbs(v);
        const fx fvv = kOne - OctAbs(u);
        u = OctApplySign(fu,  OctSignNotZero(u));
        v = OctApplySign(fvv, OctSignNotZero(v));
    }
    return OctCoord{u, v};
}

// OctDecode(c): the inverse map — the octahedral square point back to a UNIT (kOne-magnitude) direction.
// n = (u, 1-|u|-|v|, v); if the y term is negative fold (u,v) back; then normalize with the fpx int64
// FxLength path. Recovers the encoded dir WITHIN the pinned round-trip band (NOT bit-exact — a sanity bound).
inline FxVec3 OctDecode(const OctCoord& c) {
    fx nx = c.u;
    fx nz = c.v;
    fx ny = kOne - OctAbs(c.u) - OctAbs(c.v);
    if (ny < 0) {
        const fx ox = nx;
        nx = OctApplySign(kOne - OctAbs(nz), OctSignNotZero(ox));
        nz = OctApplySign(kOne - OctAbs(ox), OctSignNotZero(nz));
    }
    // Normalize to a kOne-magnitude vector (the fpx int64 floor-sqrt path -> byte-identical cross-vendor).
    const fx len = FxLength(FxVec3{nx, ny, nz});
    if (len == 0) return FxVec3{0, kOne, 0};
    return FxVec3{ (fx)(((int64_t)nx * (int64_t)kOne) / len),
                   (fx)(((int64_t)ny * (int64_t)kOne) / len),
                   (fx)(((int64_t)nz * (int64_t)kOne) / len) };
}

// OctAxisCell(c, gridN): bucket one octahedral coordinate c in [-kOne,kOne] to a cell index [0, gridN-1].
// (c + kOne) spans [0, 2*kOne]; * gridN / (2*kOne) buckets uniformly; clamp the c==+kOne corner into range.
inline int32_t OctAxisCell(fx c, int gridN) {
    int64_t idx = ((int64_t)c + (int64_t)kOne) * (int64_t)gridN / (2 * (int64_t)kOne);
    if (idx < 0) idx = 0;
    if (idx >= gridN) idx = gridN - 1;
    return (int32_t)idx;
}

// OctToCell(c, gridN): the (column,row) atlas cell of an octahedral coordinate. Uniform in oct-space ==
// nearest cell center in the octahedral parameterization.
inline OctCell OctToCell(const OctCoord& c, int gridN) {
    return OctCell{ OctAxisCell(c.u, gridN), OctAxisCell(c.v, gridN) };
}

// ImpostorCell(dir, gridN): the FLAT atlas cell index (row*gridN + col) an octahedral impostor atlas would
// sample for a plant viewed along `dir` — i.e. which of the N×N captured views best matches the view
// direction. Composes OctEncode + OctToCell. This is the addressing FO1 pins; FO2 renders the cell.
inline int32_t ImpostorCell(const FxVec3& dir, int gridN) {
    const OctCell cell = OctToCell(OctEncode(dir), gridN);
    return cell.cv * gridN + cell.cu;
}

// OctCellCenterDir(cu, cv, gridN): the view direction the CENTER of atlas cell (cu,cv) represents (decode the
// cell-center octahedral coordinate). Used by the showcase's atlas inset + as the "captured view" a cell holds.
inline FxVec3 OctCellCenterDir(int cu, int cv, int gridN) {
    const fx u = (fx)((((int64_t)(2 * cu) + 1) * (int64_t)kOne) / gridN - (int64_t)kOne);
    const fx v = (fx)((((int64_t)(2 * cv) + 1) * (int64_t)kOne) / gridN - (int64_t)kOne);
    return OctDecode(OctCoord{u, v});
}

// ============================================================================================================
// (2) DITHERED CROSS-FADE LOD  (integer cross-fade weight + ordered Bayer + per-instance hash dither)
// ============================================================================================================

// The fade weights over a transition. wNear = the high-detail (near) LOD coverage, wFar = the impostor (far)
// LOD coverage; wNear + wFar == kOne inside the band (a partition of unity). Outside the band one is kOne and
// the other 0 (pure LOD — the FO4 endpoints, now the limits of a smooth ramp instead of a hard pop).
struct FadeWeights { fx wNear = kOne, wFar = 0; };

// CrossFadeWeight(dist, threshold, band): the deterministic cross-fade ramp. The transition band is
// [threshold-band, threshold+band]. dist <= threshold-band -> pure NEAR (wNear=kOne,wFar=0). dist >=
// threshold+band -> pure FAR (wNear=0,wFar=kOne). Inside -> a linear integer ramp t=(dist-(threshold-band))
// /(2*band) in [0,kOne], wFar=t, wNear=kOne-t. band<=0 degenerates to the FO4 hard swap at `threshold`.
inline FadeWeights CrossFadeWeight(fx dist, fx threshold, fx band) {
    if (band <= 0) {   // degenerate: the FO4 hard swap (no transition band)
        return (dist < threshold) ? FadeWeights{kOne, 0} : FadeWeights{0, kOne};
    }
    const fx lo = threshold - band;
    const fx hi = threshold + band;
    if (dist <= lo) return FadeWeights{kOne, 0};
    if (dist >= hi) return FadeWeights{0, kOne};
    // t = (dist - lo) * kOne / (2*band), in (0, kOne).
    fx t = (fx)(((int64_t)(dist - lo) * (int64_t)kOne) / (2 * (int64_t)band));
    if (t < 0) t = 0;
    if (t > kOne) t = kOne;
    return FadeWeights{ kOne - t, t };
}

// ----- The ordered Bayer 4x4 dither matrix (integer thresholds 0..15) -------------------------------------
// The classic recursive Bayer/ordered-dither matrix. Screen-door coverage: a pixel at (x,y) shows the FAR
// (impostor) LOD iff kBayer4[y%4][x%4] < wFar*16. So wFar=0 -> all off (pure near), wFar=kOne -> all on
// (pure far), and wFar=k/4 -> exactly 4k of the 16 cells pass -> a clean, spatially-even, shimmer-free mask.
static const int kBayer4[16] = {
     0,  8,  2, 10,
    12,  4, 14,  6,
     3, 11,  1,  9,
    15,  7, 13,  5,
};

// DitherThreshold(x, y): the Bayer threshold [0,15] at pixel (x,y) (4x4 tiled). A pure lookup.
inline int DitherThreshold(int x, int y) {
    const int xi = ((x % 4) + 4) % 4;
    const int yi = ((y % 4) + 4) % 4;
    return kBayer4[yi * 4 + xi];
}

// DitherScale(wFar): the wFar weight [0,kOne] scaled to the Bayer step count [0,16] (integer, rounds via
// truncation of wFar*16). The per-pixel pass test compares the Bayer threshold against this.
inline int DitherScale(fx wFar) {
    int s = (int)(((int64_t)wFar * 16) / (int64_t)kOne);
    if (s < 0) s = 0;
    if (s > 16) s = 16;
    return s;
}

// DitherPass(x, y, wFar): does pixel (x,y) show the FAR/impostor LOD at cross-fade weight wFar? (true => far,
// false => near). Pure function of (x,y,wFar) -> STABLE frame-to-frame for a fixed weight (no shimmer).
inline bool DitherPass(int x, int y, fx wFar) {
    return DitherThreshold(x, y) < DitherScale(wFar);
}

// DitherCoverage(wFar): how many of the 16 Bayer cells pass at weight wFar (0..16) — the exact screen-door
// coverage count. For wFar = k*kOne/4 this is exactly 4k (0,4,8,12,16) — the pinned cross-fade coverage.
inline int DitherCoverage(fx wFar) {
    const int scale = DitherScale(wFar);
    int n = 0;
    for (int i = 0; i < 16; ++i) if (kBayer4[i] < scale) ++n;
    return n;
}

// ----- The per-instance hash dither (a whole-instance screen-door for the LOD swap) -----------------------
// ImpostorHash: a pure uint32 avalanche (the sim/particles.h ParticleHash shape, redefined locally so this
// header stays self-contained — NO include of particles.h). Deterministic, identical on every vendor/compiler.
inline uint32_t ImpostorHash(uint32_t a, uint32_t b) {
    uint32_t h = a * 2654435761u;
    h ^= (b + 0x9E3779B9u + (h << 6) + (h >> 2));
    h += b * 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}

// InstanceDitherThreshold(instId): a per-instance stable threshold in [0, kOne) from the avalanche — the
// point in the cross-fade at which THIS instance flips near->far. Distinct per instance so the meadow's
// instances flip at spread-out weights (a whole-instance screen-door), not all at once.
inline fx InstanceDitherThreshold(uint32_t instId) {
    return (fx)(ImpostorHash(instId, 0x51A21u) & (uint32_t)(kOne - 1));   // [0, kOne)
}

// InstanceShowsFar(instId, wFar): does THIS whole instance render the FAR/impostor LOD at weight wFar?
// Compares wFar against the instance's stable hash threshold -> a binary per-instance choice that is STABLE
// across frames for a fixed weight (no shimmer) and, over many instances, shows ~wFar fraction as far.
inline bool InstanceShowsFar(uint32_t instId, fx wFar) {
    return InstanceDitherThreshold(instId) < wFar;
}

// ============================================================================================================
// (3) PER-INSTANCE FADE STATE  (which LODs are active + the addressed impostor cell + the dither phase)
// ============================================================================================================

enum FadeMode : uint32_t { kFadeNearOnly = 0, kFadeCross = 1, kFadeFarOnly = 2 };

struct FadeState {
    uint32_t mode  = kFadeNearOnly;   // near-only / cross-fading / far(impostor)-only
    fx       wNear = kOne;            // near (high-detail) LOD coverage
    fx       wFar  = 0;               // far (impostor) LOD coverage
    int32_t  cell  = 0;               // the octahedral atlas cell this instance addresses (view dir -> cell)
    uint32_t phase = 0;               // the per-instance dither phase (stable hash) — flip point salt
    bool     showsFar = false;        // the whole-instance screen-door decision at this weight
};

// ImpostorViewDir(instPos, camPos): the direction FROM the instance TO the camera — the view the impostor
// billboard presents (which captured atlas view faces the camera). Full 3D (Y included) so the octahedral
// map is exercised across both hemispheres.
inline FxVec3 ImpostorViewDir(const FxVec3& instPos, const FxVec3& camPos) {
    return FxVec3{ camPos.x - instPos.x, camPos.y - instPos.y, camPos.z - instPos.z };
}

// ComputeFadeState: the full per-instance fade decision. The LOD distance is the FO4 XZ ground distance (the
// int64 FxLength path). The cross-fade weights come from CrossFadeWeight; the mode from the weights; the
// impostor cell from the (instance->camera) view direction; the dither phase + whole-instance choice from the
// hash. Pure integer -> bit-identical cross-backend.
inline FadeState ComputeFadeState(const FxVec3& instPos, const FxVec3& camPos,
                                  fx threshold, fx band, int gridN, uint32_t instId) {
    FadeState st;
    const fx dist = FxLength(FxVec3{ instPos.x - camPos.x, 0, instPos.z - camPos.z });   // FO4 XZ distance
    const FadeWeights w = CrossFadeWeight(dist, threshold, band);
    st.wNear = w.wNear;
    st.wFar  = w.wFar;
    st.mode  = (w.wFar == 0) ? kFadeNearOnly : (w.wNear == 0) ? kFadeFarOnly : kFadeCross;
    st.cell  = ImpostorCell(ImpostorViewDir(instPos, camPos), gridN);
    st.phase = ImpostorHash(instId, 0x7A1Eu) & 15u;
    st.showsFar = InstanceShowsFar(instId, w.wFar);
    return st;
}

// ============================================================================================================
// (4) THE SCENE  —  a foliage field + a camera fly-through (the addressing + cross-fade digest)
// ============================================================================================================

// ImpFnv64: a 64-bit FNV-1a word mixer for the scene digest (the ai/behavior_tree BtxFnvWord shape).
inline uint64_t ImpFnv64(uint64_t h, uint32_t word) {
    h ^= (uint64_t)word;
    h *= 0x100000001B3ull;
    return h;
}

struct ImpInstance {
    FxVec3   pos;         // the lattice position of the plant (y=0)
    uint32_t id = 0;      // the stable instance id (drives the per-instance dither)
    FadeState st;         // the fade state at the reference (final) tick — for the showcase viz
};

struct ImpTraceEntry { uint32_t tick; fx dist; uint32_t mode; fx wNear; fx wFar; int32_t cell; };

struct ImpostorSceneRun {
    int      gridN     = 8;           // the N×N octahedral atlas resolution
    int      fieldN    = 12;          // the field is fieldN×fieldN plants
    int      instances = 0;           // total plants
    int      ticks     = 0;           // camera fly-through ticks
    int      refTick   = 0;           // the reference tick the field stats + showcase viz sample (mid fly-through)
    int      inBand    = 0;           // plants cross-fading at the reference tick (mode == kFadeCross)
    int      nearOnly  = 0;           // plants near-only at the reference tick
    int      farOnly   = 0;           // plants far/impostor-only at the reference tick
    int      impostorCells = 0;       // DISTINCT atlas cells addressed by far/cross plants at the reference tick
    uint64_t digest    = 0;           // FNV-1a over EVERY (tick, instance) fade state — the strict-zero golden
    int      tracedId  = 0;           // the instance whose per-tick trace is recorded
    std::vector<ImpInstance>   field; // the plants (with their reference-tick fade state) — for the showcase
    std::vector<ImpTraceEntry> trace; // the traced instance's per-tick fade state (far -> cross -> near)
};

// RunImpostorScene: lay a deterministic fieldN×fieldN lattice of plants (spacing worldStep, centered at the
// origin, y=0), fly a camera straight down +Z through the field over `ticks`, and at every tick compute EVERY
// plant's FadeState — folding it all into the scene digest. Records the reference-tick (final) field for the
// showcase + one instance's full per-tick trace. Pure integer -> the whole scene is a bit-identical golden.
inline ImpostorSceneRun RunImpostorScene(int fieldN = 12, int gridN = 8, int ticks = 24) {
    ImpostorSceneRun run;
    run.gridN = gridN;
    run.fieldN = fieldN;
    run.ticks = ticks;

    const fx worldStep = kOne * 3;                       // 3 world units between plants
    const fx half      = (fx)(((int64_t)(fieldN - 1) * worldStep) / 2);
    const fx threshold = kOne * 16;                      // the LOD swap distance (impostor beyond)
    const fx band      = kOne * 6;                       // the cross-fade band half-width (+-6 units)

    // The plants (fixed order): a lattice in XZ, id = row*fieldN + col.
    for (int r = 0; r < fieldN; ++r) {
        for (int c = 0; c < fieldN; ++c) {
            ImpInstance inst;
            inst.pos = FxVec3{ (fx)((int64_t)c * worldStep - half), 0, (fx)((int64_t)r * worldStep - half) };
            inst.id  = (uint32_t)(r * fieldN + c);
            run.field.push_back(inst);
        }
    }
    run.instances = (int)run.field.size();

    // The traced instance: a plant near the +Z edge of the field, so the camera (flying -Z -> +Z) approaches
    // it from far (impostor) through the band to near — the far->cross->near trace.
    run.tracedId = (fieldN - 1) * fieldN + (fieldN / 2);

    // Camera fly-through: y=2 (a low eye), x=0, z from -camSpan to +camSpan across the field.
    const fx camY    = kOne * 2;
    const fx camSpan = kOne * 40;

    uint64_t dg = 0xCBF29CE484222325ull;   // FNV-1a offset basis
    for (int t = 0; t < ticks; ++t) {
        // camZ = -camSpan + t * (2*camSpan)/(ticks-1)  (integer; ticks>=2).
        const fx camZ = (ticks > 1)
            ? (fx)(-(int64_t)camSpan + (int64_t)t * (2 * (int64_t)camSpan) / (ticks - 1))
            : 0;
        const FxVec3 cam{ 0, camY, camZ };

        for (const ImpInstance& inst : run.field) {
            const FadeState st = ComputeFadeState(inst.pos, cam, threshold, band, gridN, inst.id);
            dg = ImpFnv64(dg, (uint32_t)t);
            dg = ImpFnv64(dg, inst.id);
            dg = ImpFnv64(dg, st.mode);
            dg = ImpFnv64(dg, (uint32_t)st.wNear);
            dg = ImpFnv64(dg, (uint32_t)st.wFar);
            dg = ImpFnv64(dg, (uint32_t)st.cell);
            dg = ImpFnv64(dg, st.showsFar ? 1u : 0u);

            if (inst.id == (uint32_t)run.tracedId) {
                const fx dist = FxLength(FxVec3{ inst.pos.x - cam.x, 0, inst.pos.z - cam.z });
                run.trace.push_back(ImpTraceEntry{ (uint32_t)t, dist, st.mode, st.wNear, st.wFar, st.cell });
            }
        }
    }
    run.digest = dg;

    // The reference tick = the MIDDLE of the fly-through (camera at ~field center) — so the showcase frame
    // shows the natural concentric gradient: near plants around the camera, a cross-fade RING, impostor plants
    // beyond. (The final tick has the camera past the field -> all far, a poor showcase.) run.refTick records it.
    run.refTick = ticks / 2;
    const fx camZ = (ticks > 1)
        ? (fx)(-(int64_t)camSpan + (int64_t)run.refTick * (2 * (int64_t)camSpan) / (ticks - 1))
        : 0;
    const FxVec3 cam{ 0, camY, camZ };
    // distinct impostor cells (a fieldN*fieldN worst case never exceeds gridN*gridN — a small bitset).
    std::vector<uint8_t> cellUsed((size_t)gridN * gridN, 0);
    for (ImpInstance& inst : run.field) {
        inst.st = ComputeFadeState(inst.pos, cam, threshold, band, gridN, inst.id);
        if (inst.st.mode == kFadeCross)    ++run.inBand;
        if (inst.st.mode == kFadeNearOnly) ++run.nearOnly;
        if (inst.st.mode == kFadeFarOnly)  ++run.farOnly;
        if (inst.st.mode != kFadeNearOnly) {   // far or cross plants address an impostor cell
            const int32_t idx = inst.st.cell;
            if (idx >= 0 && idx < (int32_t)cellUsed.size()) cellUsed[(size_t)idx] = 1;
        }
    }
    for (uint8_t u : cellUsed) if (u) ++run.impostorCells;
    return run;
}

// ============================================================================================================
// (5) THE SHOWCASE  —  RenderImpostorShot (strict-zero PURE-INTEGER viz both backends call VERBATIM, NO shader)
// ============================================================================================================
// Left: a TOP-DOWN of the foliage field — each plant a cell colored by its LOD/fade mode (green=near, amber=
// cross-fade, blue=impostor/far); cross-fade plants draw a Bayer screen-door checkerboard at their wFar
// coverage (the dithered transition made visible, replacing the pop). Right-top: the N×N OCTAHEDRAL ATLAS
// grid inset — a dot per far/cross plant placed in the cell it addresses (the impostor view it samples).
// Right-bottom: a distance-sorted STRIP of the plants (near->far) colored by mode. Integer RGB only -> both
// backends produce byte-identical BGRA BY CONSTRUCTION.
inline void RenderImpostorShot(const ImpostorSceneRun& run, std::vector<uint8_t>& bgra,
                               uint32_t& outW, uint32_t& outH) {
    const int kMargin = 16;
    const int kCellPx = 26;                               // top-down plant cell size
    const int topW = run.fieldN * kCellPx;                // top-down field width
    const int topH = run.fieldN * kCellPx;
    const int gap  = 20;
    const int octPx = 22;                                 // octahedral atlas inset cell size
    const int octW = run.gridN * octPx;
    const int octH = run.gridN * octPx;
    const int stripH = 40;
    const int rightW = (octW > topW ? octW : topW);
    const int W = kMargin * 3 + topW + rightW;
    const int H = kMargin * 2 + (topH > (octH + gap + stripH) ? topH : (octH + gap + stripH));
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {          // deep slate background
        bgra[p * 4 + 0] = 24; bgra[p * 4 + 1] = 20; bgra[p * 4 + 2] = 30; bgra[p * 4 + 3] = 255;
    }
    auto put = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        uint8_t* d = &bgra[((size_t)y * W + x) * 4];
        d[0] = b; d[1] = g; d[2] = r; d[3] = 255;
    };
    auto fill = [&](int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y < y0 + h; ++y) for (int x = x0; x < x0 + w; ++x) put(x, y, r, g, b);
    };
    struct RGB { uint8_t r, g, b; };
    auto modeColor = [&](uint32_t mode) -> RGB {
        if (mode == kFadeNearOnly) return RGB{ 80, 200, 110 };   // green: near / high-detail
        if (mode == kFadeCross)    return RGB{ 235, 175, 60 };   // amber: cross-fading
        return RGB{ 70, 140, 235 };                              // blue: impostor / far
    };

    // ---- (A) TOP-DOWN field: one cell per plant, colored by fade mode; cross plants show the Bayer dither. ----
    const int topX0 = kMargin, topY0 = kMargin;
    for (const ImpInstance& inst : run.field) {
        // plant grid coord from id (row-major).
        const int col = (int)inst.id % run.fieldN;
        const int row = (int)inst.id / run.fieldN;
        const int x0 = topX0 + col * kCellPx;
        const int y0 = topY0 + row * kCellPx;
        const RGB c = modeColor(inst.st.mode);
        if (inst.st.mode == kFadeCross) {
            // Draw the actual Bayer screen-door at this plant's wFar coverage: far-blue where the dither
            // passes, near-green where it does not — the dithered cross-fade made literally visible.
            const RGB nearC = modeColor(kFadeNearOnly);
            const RGB farC  = modeColor(kFadeFarOnly);
            for (int py = 0; py < kCellPx - 2; ++py) {
                for (int px = 0; px < kCellPx - 2; ++px) {
                    const bool far = DitherPass(px, py, inst.st.wFar);
                    const RGB pc = far ? farC : nearC;
                    put(x0 + px, y0 + py, pc.r, pc.g, pc.b);
                }
            }
        } else {
            fill(x0, y0, kCellPx - 2, kCellPx - 2, c.r, c.g, c.b);
        }
    }

    // ---- (B) OCTAHEDRAL ATLAS inset: the N×N grid, a dot per far/cross plant in the cell it addresses. ----
    const int octX0 = kMargin * 2 + topW;
    const int octY0 = kMargin;
    for (int cv = 0; cv < run.gridN; ++cv) {
        for (int cu = 0; cu < run.gridN; ++cu) {
            const int x0 = octX0 + cu * octPx, y0 = octY0 + cv * octPx;
            // checkerboard the grid so the cells read; center marks the +Y pole cell region.
            const uint8_t base = ((cu + cv) & 1) ? 46 : 38;
            fill(x0, y0, octPx - 1, octPx - 1, base, base, (uint8_t)(base + 8));
        }
    }
    // one dot per impostor-addressing plant, at the center of its addressed cell.
    for (const ImpInstance& inst : run.field) {
        if (inst.st.mode == kFadeNearOnly) continue;   // near plants render the full mesh (no atlas cell)
        const int cu = inst.st.cell % run.gridN;
        const int cv = inst.st.cell / run.gridN;
        const int dx = octX0 + cu * octPx + (octPx - 1) / 2;
        const int dy = octY0 + cv * octPx + (octPx - 1) / 2;
        const RGB c = modeColor(inst.st.mode);
        for (int oy = -2; oy <= 2; ++oy) for (int ox = -2; ox <= 2; ++ox)
            if (ox * ox + oy * oy <= 4) put(dx + ox, dy + oy, c.r, c.g, c.b);
    }

    // ---- (C) distance-sorted STRIP below the inset: plants low->high id colored by mode (a fade profile). ----
    const int stripY0 = octY0 + octH + gap;
    const int nStrip = run.instances;
    for (int i = 0; i < nStrip; ++i) {
        const ImpInstance& inst = run.field[(size_t)i];
        const int sw = (rightW > 0 && nStrip > 0) ? (rightW / (nStrip > 0 ? nStrip : 1)) : 1;
        const int x0 = octX0 + (i * rightW) / (nStrip > 0 ? nStrip : 1);
        const RGB c = modeColor(inst.st.mode);
        fill(x0, stripY0, sw > 0 ? sw : 1, stripH, c.r, c.g, c.b);
    }
    (void)fill;
}

}  // namespace impostor
}  // namespace hf::foliage
