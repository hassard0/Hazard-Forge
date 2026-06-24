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
#include <cmath>       // Slice FO5 render bridge ONLY: std::sin/std::cos for the float wind lean quat (FLOAT)

#include "sim/fpx.h"   // Q16.16 toolbox (read-only): fx / kOne / kFrac / fxmul / FxVec3
#include "pcg/pcg.h"   // Slice FO2 (read-only): PcgGraph / PcgStream / PcgInstance / Generate — the placement engine
#include "math/math.h" // Slice FO5 render bridge ONLY: math::Mat4 / Quat / Vec3 / FromTRS / Normalize (FLOAT)

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
    uint32_t             lod  = 0;   // Slice FO4 (APPENDED after bend): integer distance-LOD bucket [0,3]
                                     // (0=near/full, 1=mid, 2=far billboard, 3=culled). FO2/FO3 leave this
                                     // default-0, so FO2/FO3 stay byte-unchanged; AssignLods fills it.
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

// ===== Slice FO4 — Integer distance-LOD pick (the 4th slice of FLAGSHIP #25) ===========================
// The SCALE primitive: each plant's LOD bucket is chosen by its INTEGER XZ distance to the camera against
// INTEGER thresholds — so the LOD assignment is bit-identical cross-backend (NO float thresholds, NO trig).
// Near plants render full, mid degrade, far become billboards, beyond-far are culled. The XZ distance is the
// int64 FxLength path (fpx.h, CPU-side -> byte-identical); the thresholds are pure integer compares. The moat:
// UE5/SpeedTree LOD selection is float/non-deterministic; this pick is a pure function of (pos, cam, radii), so
// two netcode peers pick the byte-identical LOD set. APPEND-ONLY (FO1 WindBend + FO2 PlaceFoliage + FO3
// ApplyWind above + fpx.h FxLength read-only). Reuses FxLength VERBATIM. NO <cmath>, NO float, NO new RHI/shader.

// ----- FoliageLod: the integer distance-LOD bucket for one instance (PURE INTEGER) ---------------------
// d = FxLength of (instPos - camPos) with Y zeroed (the XZ ground distance, the int64 path). midR =
// nearR + (farR-nearR)/2 (the integer midpoint). Bucket against the integer thresholds: d<nearR -> 0 (near,
// full plant); d<midR -> 1 (mid); d<farR -> 2 (far billboard); else -> 3 (culled). Pure integer compares;
// returns the bucket in [0,3]. With nearR<=midR<=farR the pick is MONOTONE in d (a farther plant never picks
// a nearer LOD). nearR >= the field extent (+ a huge farR) -> every plant LOD 0 (the no-op control).
inline uint32_t FoliageLod(const FxVec3& instPos, const FxVec3& camPos, fx nearR, fx farR) {
    const fx d    = hf::sim::fpx::FxLength(FxVec3{ instPos.x - camPos.x, 0, instPos.z - camPos.z });
    const fx midR = nearR + (farR - nearR) / 2;
    if (d < nearR) return 0u;
    if (d < midR)  return 1u;
    if (d < farR)  return 2u;
    return 3u;
}

// ----- AssignLods: annotate each plant with its integer LOD bucket (PURE INTEGER) ----------------------
// For each plant in FIXED order, inst.lod = FoliageLod(inst.base.pos, camPos, nearR, farR). Mutates lod in
// place; the base placement + the FO3 bend are UNTOUCHED. Pure integer (reuses FxLength). nearR >= the field
// extent (+ huge farR) -> every inst.lod == 0 (the upright/full no-op control). NO <cmath>, NO float, NO
// clock/RNG — bit-identical CPU<->Vulkan<->Metal by construction.
inline void AssignLods(std::vector<FoliageInstance>& v, const FxVec3& camPos, fx nearR, fx farR) {
    for (FoliageInstance& inst : v) {
        inst.lod = FoliageLod(inst.base.pos, camPos, nearR, farR);
    }
}

// ===== Slice FO5 — LIT 3D render bridge (FLOAT, render-only — the SCALE money-shot of FLAGSHIP #25) =====
// THE ONE FLOAT CROSSING of the whole flagship. FO1-FO4 above stay STRICT INTEGER (bit-exact cross-platform);
// here — and ONLY here — we cross to float to build per-instance render transforms for the rasterizer. This is
// the documented FLOAT visresolve-bar (the PCG6/PT6/GR6 precedent): the foliage DATA (FO2 placement + FO3 wind
// sway + FO4 LOD) is bit-exact, the final raster/shade is float (cross-vendor ~the engine baseline, NOT held to
// the integer zero-diff bar). The provenance proof: every transform derives from the bit-exact FoliageInstance
// {base.pos, base.orient, base.scale, bend, lod} that PlaceFoliage/ApplyWind/AssignLods emit. Render-only — NOT
// used by FO1-FO4, NO data mutation. The pcg::PcgToRenderInstances twin, but per-instance it ALSO bakes the
// per-instance bend (a float LEAN about a fixed horizontal axis) AND HONORS the LOD (cull lod==3, far LODs a
// touch smaller).

// lodScale(lod): the per-LOD presentation scale (1.0 / 1.0 / 0.7 for LOD0/1/2 — far plants a touch smaller as a
// billboard stand-in; honest v1). lod==3 is culled by the caller (never reaches here). FLOAT, render-only.
inline float FoliageLodScale(uint32_t lod) {
    return (lod >= 2u) ? 0.7f : 1.0f;
}

// FoliageToRenderInstances(instances, baseScale): build ONE column-major model matrix per NON-CULLED instance
// (lod != 3). For each plant:
//   t   = {FxToFloat(base.pos.x/y/z)} (the placement position, the ONE host fixed-point->float divide);
//   yaw = normalize(Quat{FxToFloat(base.orient.x..w)}) (the placement yaw, renormalized in float);
//   lean = a float quat rotating about a FIXED horizontal axis (+X) by FxToFloat(bend) — built directly with
//          std::cos/std::sin of bend/2 (this is the FLOAT visresolve-bar code, NOT asserted bit-exact; float is
//          allowed HERE in the render bridge ONLY);
//   scale = FxToFloat(base.scale) * baseScale * FoliageLodScale(lod);
//   out.push_back(FromTRS(t, lean*yaw, {scale,scale,scale})).
// The composed rotation lean*yaw is the Hamilton product (math.h has no quat operator*, so it is spelled out
// here, render-only). Output is the scene::InstanceData / InstanceTransformLayout packing (16 floats column-
// major) the EXISTING instanced lit pipeline consumes, matching pcg::PcgToRenderInstances EXACTLY. Empty input
// -> empty output (the empty no-op). Pure deterministic host float (no RNG, no clock). The ONLY float code in
// foliage.h (clearly commented as the render bridge).
inline std::vector<math::Mat4> FoliageToRenderInstances(const std::vector<FoliageInstance>& instances,
                                                        float baseScale) {
    std::vector<math::Mat4> out;
    out.reserve(instances.size());
    for (const FoliageInstance& inst : instances) {
        if (inst.lod == 3u) continue;   // LOD3 = culled (beyond far radius) — skipped, never rendered
        const math::Vec3 t{hf::sim::fpx::FxToFloat(inst.base.pos.x),
                           hf::sim::fpx::FxToFloat(inst.base.pos.y),
                           hf::sim::fpx::FxToFloat(inst.base.pos.z)};
        const math::Quat yaw = math::Normalize(math::Quat{
            hf::sim::fpx::FxToFloat(inst.base.orient.x), hf::sim::fpx::FxToFloat(inst.base.orient.y),
            hf::sim::fpx::FxToFloat(inst.base.orient.z), hf::sim::fpx::FxToFloat(inst.base.orient.w)});
        // The wind LEAN: a float quaternion about the FIXED horizontal axis +X by the Q16.16 bend angle.
        const float bendRad = hf::sim::fpx::FxToFloat(inst.bend);
        const float hs = std::sin(bendRad * 0.5f), hc = std::cos(bendRad * 0.5f);
        const math::Quat lean{hs, 0.0f, 0.0f, hc};   // (x,y,z,w) = rotate about +X by bendRad
        // lean * yaw (Hamilton product; math.h has no quat operator*, render-only here):
        const math::Quat q{
            lean.w * yaw.x + lean.x * yaw.w + lean.y * yaw.z - lean.z * yaw.y,
            lean.w * yaw.y - lean.x * yaw.z + lean.y * yaw.w + lean.z * yaw.x,
            lean.w * yaw.z + lean.x * yaw.y - lean.y * yaw.x + lean.z * yaw.w,
            lean.w * yaw.w - lean.x * yaw.x - lean.y * yaw.y - lean.z * yaw.z};
        const float s = hf::sim::fpx::FxToFloat(inst.base.scale) * baseScale * FoliageLodScale(inst.lod);
        out.push_back(math::FromTRS(t, q, math::Vec3{s, s, s}));
    }
    return out;
}

// ===== Slice FO6 — HERO render bridge (FLOAT, render-only — the CAPSTONE money-shot of FLAGSHIP #25) =====
// APPEND-ONLY. FO6 is driver tuning (a closer/lower cinematic camera + bigger baseScale + a larger FO6 WindField
// amplitude) on top of the EXISTING FO5 render path. The FO3 wind bend is a SMALL angle; at hero scale the lean
// must be PERCEPTIBLE so the meadow visibly ripples. FoliageToRenderInstancesHero is FoliageToRenderInstances
// VERBATIM except (1) the bend angle is scaled by a render-only `leanGain` BEFORE building the float lean quat,
// and (2) the plant is scaled TALLER than wide by `heightMul` in its LOCAL frame (a non-uniform Y stretch applied
// BEFORE the lean rotation via FromTRS's local scale) — so a sphere mesh reads as an upright BLADE whose tilt is
// VISIBLE (a uniform sphere's lean is invisible; a tall blade's lean reads as a coherent ripple). Both are render-
// only and do NOT touch the bit-exact FO1-FO4 data (the bend stored on each instance is UNCHANGED; only this
// render transform amplifies + stretches it). leanGain==1 + heightMul==1 reproduces FoliageToRenderInstances byte-
// for-byte. This is render-only (NOT used by FO1-FO5, NO data mutation); the provenance proof recomputes against
// THIS same helper so the bit-exact derivation chain holds. The ONLY new code FO6 adds to foliage.h.
inline std::vector<math::Mat4> FoliageToRenderInstancesHero(const std::vector<FoliageInstance>& instances,
                                                           float baseScale, float leanGain, float heightMul) {
    std::vector<math::Mat4> out;
    out.reserve(instances.size());
    for (const FoliageInstance& inst : instances) {
        if (inst.lod == 3u) continue;   // LOD3 = culled (beyond far radius) — skipped, never rendered
        const math::Vec3 t{hf::sim::fpx::FxToFloat(inst.base.pos.x),
                           hf::sim::fpx::FxToFloat(inst.base.pos.y),
                           hf::sim::fpx::FxToFloat(inst.base.pos.z)};
        const math::Quat yaw = math::Normalize(math::Quat{
            hf::sim::fpx::FxToFloat(inst.base.orient.x), hf::sim::fpx::FxToFloat(inst.base.orient.y),
            hf::sim::fpx::FxToFloat(inst.base.orient.z), hf::sim::fpx::FxToFloat(inst.base.orient.w)});
        // The wind LEAN: a float quaternion about the FIXED horizontal axis +X by the Q16.16 bend angle, scaled
        // by the render-only leanGain so the lean READS at hero scale (FO5: leanGain==1.0 reproduces it exactly).
        const float bendRad = hf::sim::fpx::FxToFloat(inst.bend) * leanGain;
        const float hs = std::sin(bendRad * 0.5f), hc = std::cos(bendRad * 0.5f);
        const math::Quat lean{hs, 0.0f, 0.0f, hc};   // (x,y,z,w) = rotate about +X by bendRad
        // lean * yaw (Hamilton product; math.h has no quat operator*, render-only here):
        const math::Quat q{
            lean.w * yaw.x + lean.x * yaw.w + lean.y * yaw.z - lean.z * yaw.y,
            lean.w * yaw.y - lean.x * yaw.z + lean.y * yaw.w + lean.z * yaw.x,
            lean.w * yaw.z + lean.x * yaw.y - lean.y * yaw.x + lean.z * yaw.w,
            lean.w * yaw.w - lean.x * yaw.x - lean.y * yaw.y - lean.z * yaw.z};
        const float s = hf::sim::fpx::FxToFloat(inst.base.scale) * baseScale * FoliageLodScale(inst.lod);
        // Taller-than-wide blade: a non-uniform LOCAL Y stretch (FromTRS applies scale first, in local space) so
        // the lean of the rotated blade reads (a uniform sphere's tilt is invisible). Render-only.
        out.push_back(math::FromTRS(t, q, math::Vec3{s, s * heightMul, s}));
    }
    return out;
}

}  // namespace hf::foliage
