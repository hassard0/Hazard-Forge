#pragma once
// Slice FO1 — Deterministic integer WIND FIELD (the BEACHHEAD of FLAGSHIP #25, DETERMINISTIC FOLIAGE
// AT SCALE). Header-only, namespace hf::foliage. PURE CPU — NO device, NO backend symbols, NO new RHI,
// NO shader. The irreducible primitive: WindBend(wind, pos, frame) returns a Q16.16 bend angle from a sum
// of host-baked sine "gust" waves over (position, frame#), bit-identical CPU<->Vulkan<->Metal BY
// CONSTRUCTION — NO runtime sin/cos/<cmath>: a COMMITTED int16 LUT (kFoliageWind16) indexed by an integer
// phase accumulator, the engine/audio/mixer.cpp kSineTable discipline. This is the moat: UE5/SpeedTree
// wind is float/non-deterministic; this wind is a pure function of position + frame, so two netcode peers
// grow the byte-identical swaying meadow.
//
// REUSE MAP: kFoliageWind16[256] are the SAME integer literals as engine/audio/mixer.cpp's kSineTable
// (kFoliageWind16[i] == round(32767 * sin(2*pi*i/256))) — COPIED VERBATIM (NO runtime sin; the formula is
// only a comment, the data is committed integer literals). fpx.h (fx / kOne / kFrac / fxmul / FxVec3) is
// reused READ-ONLY. Unlike the audio/net flagships, foliage is a render flagship with Mac-BAKED goldens,
// so foliage.h MAY freely #include "sim/fpx.h" (no clang-standalone constraint).

#include <cstdint>
#include <vector>      // Slice FO2: PlaceFoliage returns std::vector<FoliageInstance>

#include "sim/fpx.h"   // Q16.16 toolbox (read-only): fx / kOne / kFrac / fxmul / FxVec3
#include "pcg/pcg.h"   // Slice FO2 (read-only): PcgGraph / PcgStream / PcgInstance / Generate — the placement engine

namespace hf::foliage {

// Pull the Q16.16 primitives from fpx so callers can `using` them.
using hf::sim::fpx::fx;
using hf::sim::fpx::kOne;
using hf::sim::fpx::kFrac;
using hf::sim::fpx::fxmul;
using hf::sim::fpx::FxVec3;

// ----- The committed sine wavetable -----------------------------------------------------------------
// A FIXED, committed full-wave table of 256 int16 samples: kFoliageWind16[i] == round(32767 *
// sin(2*pi*i / 256)), i in [0, 256). COPIED VERBATIM from engine/audio/mixer.cpp's kSineTable (the
// formula above is the GENERATION recipe in a comment ONLY; the data is committed integer literals — NO
// runtime sin). The wind phase is a 32-bit accumulator spanning one full cycle as 2^32; the table index
// is the top 8 bits (phase >> 24). Identical bytes on every compiler/vendor => bit-exact cross-backend.
static const int16_t kFoliageWind16[256] = {
         0,    804,   1608,   2410,   3212,   4011,   4808,   5602,
      6393,   7179,   7962,   8739,   9512,  10278,  11039,  11793,
     12539,  13279,  14010,  14732,  15446,  16151,  16846,  17530,
     18204,  18868,  19519,  20159,  20787,  21403,  22005,  22594,
     23170,  23731,  24279,  24811,  25329,  25832,  26319,  26790,
     27245,  27683,  28105,  28510,  28898,  29268,  29621,  29956,
     30273,  30571,  30852,  31113,  31356,  31580,  31785,  31971,
     32137,  32285,  32412,  32521,  32609,  32678,  32728,  32757,
     32767,  32757,  32728,  32678,  32609,  32521,  32412,  32285,
     32137,  31971,  31785,  31580,  31356,  31113,  30852,  30571,
     30273,  29956,  29621,  29268,  28898,  28510,  28105,  27683,
     27245,  26790,  26319,  25832,  25329,  24811,  24279,  23731,
     23170,  22594,  22005,  21403,  20787,  20159,  19519,  18868,
     18204,  17530,  16846,  16151,  15446,  14732,  14010,  13279,
     12539,  11793,  11039,  10278,   9512,   8739,   7962,   7179,
      6393,   5602,   4808,   4011,   3212,   2410,   1608,    804,
         0,   -804,  -1608,  -2410,  -3212,  -4011,  -4808,  -5602,
     -6393,  -7179,  -7962,  -8739,  -9512, -10278, -11039, -11793,
    -12539, -13279, -14010, -14732, -15446, -16151, -16846, -17530,
    -18204, -18868, -19519, -20159, -20787, -21403, -22005, -22594,
    -23170, -23731, -24279, -24811, -25329, -25832, -26319, -26790,
    -27245, -27683, -28105, -28510, -28898, -29268, -29621, -29956,
    -30273, -30571, -30852, -31113, -31356, -31580, -31785, -31971,
    -32137, -32285, -32412, -32521, -32609, -32678, -32728, -32757,
    -32767, -32757, -32728, -32678, -32609, -32521, -32412, -32285,
    -32137, -31971, -31785, -31580, -31356, -31113, -30852, -30571,
    -30273, -29956, -29621, -29268, -28898, -28510, -28105, -27683,
    -27245, -26790, -26319, -25832, -25329, -24811, -24279, -23731,
    -23170, -22594, -22005, -21403, -20787, -20159, -19519, -18868,
    -18204, -17530, -16846, -16151, -15446, -14732, -14010, -13279,
    -12539, -11793, -11039, -10278,  -9512,  -8739,  -7962,  -7179,
     -6393,  -5602,  -4808,  -4011,  -3212,  -2410,  -1608,   -804,
};

// ----- A wind gust (one component of the field) -----------------------------------------------------
// kx/kz: spatial frequencies — how fast the phase advances across X/Z (multiplied into the integer
// position). speed: the temporal phase advance per frame (how fast this gust animates). amp: the Q16.16
// bend amplitude this gust contributes. The phase is a pure uint32 accumulator; only its top 8 bits index
// the LUT, so the wrapping arithmetic IS the periodicity (no modulo needed).
struct Gust {
    int32_t  kx    = 0;   // spatial frequency along X (per Q16.16 world unit of pos.x)
    int32_t  kz    = 0;   // spatial frequency along Z
    uint32_t speed = 0;   // temporal phase advance per frame
    fx       amp   = 0;   // Q16.16 bend amplitude of this gust
};

// ----- The wind field (a few gusts + a master amplitude) --------------------------------------------
// gustCount <= 4. master scales the summed bend (master=0 => the no-op control: zero bend everywhere).
struct WindField {
    Gust gusts[4];
    int  gustCount = 0;
    fx   master    = kOne;   // 1.0 in Q16.16
};

// ----- The wind bend evaluator (PURE INTEGER) -------------------------------------------------------
// WindBend(w, pos, frame): for each of w.gustCount gusts, build a uint32 PHASE accumulator
//   phase = (uint32_t)g.kx * (uint32_t)pos.x + (uint32_t)g.kz * (uint32_t)pos.z + g.speed * frame
// (pure uint32 WRAPPING arithmetic — only the top bits matter), index the committed LUT
//   int32_t s = kFoliageWind16[(phase >> 24) & 255]   (an int16 in [-32767, 32767], ~Q15),
// and accumulate the gust contribution by scaling the Q15 LUT value by the Q16.16 amp into a Q16.16 angle
//   bend += (fx)(((int64_t)g.amp * s) >> 15).
// Finally scale by the master amplitude: return fxmul(bend, w.master). The result is a small Q16.16 bend
// angle (radians-ish, bounded by the summed amplitudes). Pure integer — the int64 intermediates are
// CPU-side (the wind is host-evaluated; NO shader). NO <cmath>, NO float, NO clock/RNG.
inline fx WindBend(const WindField& w, const FxVec3& pos, uint32_t frame) {
    fx bend = 0;
    const int n = (w.gustCount < 4) ? w.gustCount : 4;
    for (int g = 0; g < n; ++g) {
        const Gust& gu = w.gusts[g];
        const uint32_t phase = (uint32_t)gu.kx * (uint32_t)pos.x +
                               (uint32_t)gu.kz * (uint32_t)pos.z +
                               gu.speed * frame;
        const int32_t s = kFoliageWind16[(phase >> 24) & 255u];
        bend += (fx)(((int64_t)gu.amp * (int64_t)s) >> 15);
    }
    return fxmul(bend, w.master);
}

// ===== Slice FO2 — PCG-driven foliage PLACEMENT (the 2nd slice of FLAGSHIP #25) =======================
// Place the meadow by REUSING the just-completed deterministic PCG (hf::pcg, FLAGSHIP #22) verbatim: a
// FoliageField holds a pcg::PcgGraph (the declarative scatter->mask->transform->prune recipe) and
// PlaceFoliage calls pcg::Generate, wrapping each scattered PcgInstance as a plant in the SAME order. So
// the whole meadow is a PURE FUNCTION of the seed (every PCG stage is already proven bit-exact cross-
// platform), and the foliage layer adds NO new placement math — it COMPOSES PCG. APPEND-ONLY (FO1 above +
// pcg.h/fpx.h read-only). Strict integer, header-only, NO float, NO new RHI, NO shader. The moat: UE5/
// SpeedTree foliage placement is float/non-deterministic; this meadow is the same on every netcode peer.

// ----- FoliageInstance: a placed plant = a PCG instance (kept extensible, base first) ------------------
// A plant IS a scattered pcg::PcgInstance{pos, orient, scale}. base is FIRST so later slices can APPEND a
// bend param (FO3) and a lod field (FO4) without disturbing the placement provenance.
struct FoliageInstance {
    hf::pcg::PcgInstance base;   // the scattered PCG instance (pos / orient / scale) — the plant's transform
    fx                   bend = 0;   // Slice FO3 (APPENDED after base): per-instance Q16.16 wind sway (0 = upright);
                                     // FO2's PlaceFoliage leaves this default-0, so FO2 stays byte-unchanged.
};

// ----- FoliageField: the declarative meadow recipe (a PcgGraph) ----------------------------------------
// The PcgGraph that scatters the plants (area + cells + optional mask + transform + optional prune). Later
// slices may add a plant-type id alongside, fine — the graph stays the source of placement truth.
struct FoliageField {
    hf::pcg::PcgGraph graph;     // the scatter->mask->transform->prune recipe Generate consumes
};

// ----- PlaceFoliage: Generate the meadow from the field + stream (PURE REUSE) --------------------------
// auto pcgInstances = hf::pcg::Generate(field.graph, stream); wrap each into a FoliageInstance{inst} in the
// SAME order; return. No new math, no float — Generate already does scatter/mask/transform/prune
// deterministically, so PlaceFoliage(field, stream).size() == Generate(field.graph, stream).size() and the
// meadow is bit-exact cross-platform. An empty graph (cellsX<=0 || cellsZ<=0) -> Generate returns empty ->
// 0 plants (the no-op control).
inline std::vector<FoliageInstance> PlaceFoliage(const FoliageField& field, const hf::pcg::PcgStream& stream) {
    const std::vector<hf::pcg::PcgInstance> pcgInstances = hf::pcg::Generate(field.graph, stream);
    std::vector<FoliageInstance> plants;
    plants.reserve(pcgInstances.size());
    for (const hf::pcg::PcgInstance& inst : pcgInstances) plants.push_back(FoliageInstance{inst});
    return plants;
}

// ===== Slice FO3 — Per-instance WIND SWAY (the 3rd slice of FLAGSHIP #25, THE DETERMINISM HEADLINE) ====
// The bridge that ties the deterministic wind (FO1) to the placed foliage (FO2): ApplyWind evaluates the
// integer wind field (FO1 WindBend) at each plant's base.pos and stores the per-instance Q16.16 bend — so the
// whole meadow sways as a PURE FUNCTION of (seed, frame), bit-identical cross-platform. The swaying *data* is a
// strict integer golden (NO float anywhere) — the thing UE5/SpeedTree float wind cannot make deterministic.
// APPEND-ONLY (FO1 WindBend + FO2 PlaceFoliage above + pcg.h/fpx.h read-only). Reuses FO1 WindBend VERBATIM.

// ----- ApplyWind: annotate each plant with its local Q16.16 wind bend (PURE INTEGER) -------------------
// For each plant in FIXED order, inst.bend = WindBend(wind, inst.base.pos, frame). Mutates the bend in place;
// the base placement is UNTOUCHED (wind only annotates a sway, never moves the plant). Pure integer (reuses
// FO1 WindBend). master=0 (or all amp=0) -> every inst.bend == 0 (the upright no-op control). NO <cmath>, NO
// float, NO clock/RNG — bit-identical CPU<->Vulkan<->Metal by construction.
inline void ApplyWind(std::vector<FoliageInstance>& instances, const WindField& wind, uint32_t frame) {
    for (FoliageInstance& inst : instances) {
        inst.bend = WindBend(wind, inst.base.pos, frame);
    }
}

}  // namespace hf::foliage
