#pragma once
// Slice ECON-S6 — LIT 3D render bridge (FLOAT, render-only — the money-shot capstone of FLAGSHIP #30 ECON).
// THE ONE FLOAT CROSSING of the whole flagship. econ.h's S1-S5 stay STRICT INTEGER + self-contained (the
// cheap clang-hash proof — ledger / crafting / economy-tick / market / quest+lockstep); here — and ONLY
// here — we cross to float to build per-slot render transforms for the rasterizer. This is the documented
// FLOAT visresolve-bar (the WFC6/PCG6/PT6 precedent): the STATE (the integer ledger) is bit-exact, the
// final raster/shade is float (cross-vendor ~the engine baseline, NOT held to the integer zero-diff bar).
// The provenance proof: every transform derives from the bit-exact integer stock the deterministic
// economy script produces (the SAME EconToRenderInstances of the SAME post-script EconState).
//
// SEPARATE HEADER ON PURPOSE: econ.h is self-contained (only <cstddef>/<cstdint>/<vector> + net/session.h)
// so its standalone-clang cross-platform proof stays cheap. The render bridge needs math::Mat4/FromTRS/Vec3
// + fpx::FxToFloat, which would break that — so the bridge lives HERE, NOT in econ.h (mirrors wfc_render.h,
// the known-good twin: wfc.h is self-contained, the render bridge is in a separate header). This header is
// NOT standalone-clang (it's the render layer, only used by the showcase) — that's fine.

#include <cstdint>
#include <vector>

#include "econ/econ.h"          // S1-S5 ECON core (World / EconState / ledger) — UNMODIFIED, self-contained
#include "math/math.h"          // render bridge: math::Mat4 / Quat / Vec3 / FromTRS / Normalize (float)
#include "sim/fpx.h"            // fpx::FxToFloat (the single fixed-point->float convention; mirrors wfc_render.h)

namespace hf::econ {

// --- EconRenderStyle: the FIXED layout/visual constants (golden-stable). -----------------------------
// barSize = world spacing between slot columns (X = entity, Z = item); heightUnit = world height per unit
// of integer stock; maxHeight = a visual-only clamp so one very-rich slot doesn't dwarf the skyline. FIXED
// so the render is a deterministic function of (state, style) alone.
struct EconRenderStyle {
    float barSize    = 1.0f;    // world spacing between slot columns (X = entity, Z = item)
    float heightUnit = 0.05f;   // world height per unit of integer stock (tune for a readable skyline)
    float maxHeight  = 12.0f;   // clamp very tall bars so one rich slot doesn't dwarf the scene (visual only)
};

// --- EconToRenderInstances: the render bridge (the WfcToRenderInstances twin) ------------------------
// For each (entity, item) ledger slot with stock > 0, emit ONE bar instance: a unit CUBE stretched in Y
// into a bar sitting on the ground plane (base at y=0, top at `height`), at the slot's CENTERED grid
// position:
//   x = (entity - (entityCount-1)*0.5) * barSize     (centered so the skyline straddles the origin)
//   z = (item   - (itemCount-1)*0.5)   * barSize
//   y = height * 0.5                                  (the cube center; the base sits on the ground)
// scale = {barSize*0.45, height, barSize*0.45} where height = min(maxHeight, (float)stock * heightUnit).
// The SINGLE float crossing is here (the ledger is integer; heights become float once — the stock is an
// int32 Qty so a plain (float) cast is exact + deterministic, no fxmul). Identity orientation. Empty slots
// (stock <= 0) are skipped (the no-op). An empty ledger -> empty output (the empty no-op). Pure
// deterministic host float (no RNG, no clock). This is the canonical provenance function — the showcase
// memcmp-compares its bytes against a recompute of the same post-script state.
inline std::vector<math::Mat4> EconToRenderInstances(const EconState& s, const EconRenderStyle& style) {
    std::vector<math::Mat4> out;
    const World& w = s.ledger;
    if (w.entityCount == 0 || w.itemCount == 0) return out;
    out.reserve(static_cast<std::size_t>(w.entityCount) * static_cast<std::size_t>(w.itemCount));
    const float halfE = (static_cast<float>(w.entityCount) - 1.0f) * 0.5f;
    const float halfI = (static_cast<float>(w.itemCount) - 1.0f) * 0.5f;
    const math::Quat ident = math::Quat::Identity();
    for (uint32_t e = 0; e < w.entityCount; ++e) {
        for (uint32_t t = 0; t < w.itemCount; ++t) {
            const Qty stock = w.At(e, t);
            if (stock <= 0) continue;  // empty slot — skip (the bar is absent)
            float height = static_cast<float>(stock) * style.heightUnit;  // the ONE float crossing
            if (height > style.maxHeight) height = style.maxHeight;       // visual clamp (no <algorithm>)
            const math::Vec3 trans{(static_cast<float>(e) - halfE) * style.barSize,
                                   height * 0.5f,
                                   (static_cast<float>(t) - halfI) * style.barSize};
            const math::Vec3 scale{style.barSize * 0.45f, height, style.barSize * 0.45f};
            out.push_back(math::FromTRS(trans, ident, scale));
        }
    }
    return out;
}

// --- EconBarItems: the parallel per-instance ITEM-id list (for per-item COLORED draws) ---------------
// SAME iteration order as EconToRenderInstances, so barItems[i] is the item id of mats[i]. The showcase
// uses this to issue ONE draw per item with a per-item albedo (coin gold / ore grey / ingot blue / tool
// red) — a COLORED economy money-shot WITHIN "no new shader" (the existing instanced-lit pipeline binds a
// per-draw albedo texture + material). The provenance compare still uses the flat EconToRenderInstances
// bytes; this is a derived presentation index (mirrors wfc_render.h::WfcTileKinds).
inline std::vector<uint32_t> EconBarItems(const EconState& s) {
    std::vector<uint32_t> items;
    const World& w = s.ledger;
    if (w.entityCount == 0 || w.itemCount == 0) return items;
    items.reserve(static_cast<std::size_t>(w.entityCount) * static_cast<std::size_t>(w.itemCount));
    for (uint32_t e = 0; e < w.entityCount; ++e)
        for (uint32_t t = 0; t < w.itemCount; ++t)
            if (w.At(e, t) > 0) items.push_back(t);  // SAME order/skip rule as EconToRenderInstances
    return items;
}

}  // namespace hf::econ
