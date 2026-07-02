#pragma once
// Hazard Forge — editor MULTI-SELECT + TRANSFORM-SNAPPING primitives (Slice ED4, pure CPU,
// ImGui-free, backend-free).
//
// The third sibling of the FROZEN edit_ops.h (transform/material mutation) and edit_ops2.h (ED6
// entity creation): THIS module adds the two ED4 primitives every editor surface shares —
//   1. the sorted MULTI-SELECT set (the hierarchy panel's Ctrl+click and the --fly viewport's
//      Ctrl+pick both toggle through the same ToggleIndexInSelection), and
//   2. the SNAP quantizer (gizmo-drag results and inspector-committed values quantize through the
//      same SnapValue / SnapTransform* / SnapTransformEdit at the recorded-wrapper boundary, so
//      the ED5 history records the SNAPPED value).
// Pure: math + scene::Transform + the edit_ops TransformEdit payload. ZERO vk*/Metal/rhi rendering
// symbols and ZERO ImGui, so it lives in hf_core, is unit-tested headlessly, and the --ed4-dry-run
// hand-called twins are the SAME functions the panels / fly loop drive.
//
// SNAP QUANTIZATION (documented, LOCKED):
//   * SnapValue(v, step) = std::round(v / step) * step — ROUND-TO-NEAREST, halves away from zero
//     (std::round semantics). A step <= 0 passes v through unchanged.
//   * EXACTNESS: for a BINARY-FRACTION step (0.25 = 2^-2, 0.125 = 2^-3, ...) both v/step and
//     k*step are exact float scalings by a power of two, so the snapped result is the EXACT real
//     multiple k*step (bit-reproducible and decimal-exact) for all in-range v. For a non-binary
//     step (e.g. the 15-degree angle step converted to radians) the result is the DETERMINISTIC
//     float product k*step — bit-identical run-to-run and cross-platform (pure IEEE-754 single
//     ops), just not a decimal-exact multiple. The default posStep (0.25) and scaleStep (0.125)
//     are deliberately binary fractions so position/scale snapping is exact.
//   * Angles snap in RADIANS on the step angleStepDeg * kDegToRad (the float nearest pi/180).
//   * SnapTransformEdit snaps ONLY the absolute `set*` payloads (the inspector's commit route);
//     `add*` deltas pass through untouched — the gizmo path snaps its RESULT transform instead
//     (SnapTransformPosition/Euler/Scale on the dragged field only, never the untouched fields,
//     so a translate drag can never re-quantize a pre-existing off-grid scale).
//
// MULTI-SELECT SET (documented, LOCKED):
//   * The set is a SORTED (ascending) small vector of view-order entity indices with the shared
//     invariant: EMPTY = single-select mode (the pre-ED4 state; the effective selection is
//     {primary}); NON-EMPTY => size >= 2 AND it contains the primary. `primary` is the
//     last-clicked index (what the inspector shows).
//   * ToggleIndexInSelection implements Ctrl+click: toggling an unselected index JOINS it (the
//     empty set is first seeded with the current primary) and makes it the new primary; toggling
//     a selected index LEAVES it (the primary re-anchors to the largest remaining member); a set
//     shrunk below 2 collapses back to single-select (cleared); Ctrl+click on the SOLE current
//     selection is a safe no-op (the selection never empties through toggling).

#include <algorithm>
#include <cmath>
#include <vector>

#include "editor/edit_ops.h"   // TransformEdit — the absolute-set payload SnapTransformEdit snaps
#include "math/math.h"
#include "scene/transform.h"

namespace hf::editor {

// --- Snapping ---------------------------------------------------------------------------------

// The editor's snap configuration. Defaults OFF so every pre-ED4 caller behaves byte-identically.
// posStep/scaleStep are binary fractions (exact in float — see the header note); angleStepDeg is
// authored in degrees and applied in radians.
struct SnapConfig {
    bool  enabled = false;
    float posStep = 0.25f;        // world units per grid cell (2^-2: exact)
    float angleStepDeg = 15.0f;   // degrees per rotation notch (applied as radians)
    float scaleStep = 0.125f;     // scale units per notch (2^-3: exact; chosen over 0.1, which is
                                  // not a binary fraction, so snapped scales are decimal-exact)
};

// The float nearest pi/180 (degree -> radian conversion factor for the angle step).
inline constexpr float kDegToRad = 0.017453292519943295f;

// Round-to-nearest quantization: round(v / step) * step (halves away from zero, std::round).
// step <= 0 passes v through. See the header note for the exactness contract.
inline float SnapValue(float v, float step) {
    if (step <= 0.0f) return v;
    return std::round(v / step) * step;
}

inline math::Vec3 SnapVec3(const math::Vec3& v, float step) {
    return {SnapValue(v.x, step), SnapValue(v.y, step), SnapValue(v.z, step)};
}

// Snap an euler angle (radians) to the nearest multiple of `stepDeg` degrees.
inline float SnapAngleRadians(float rad, float stepDeg) {
    return SnapValue(rad, stepDeg * kDegToRad);
}

inline math::Vec3 SnapEulerVec3(const math::Vec3& eulerRad, float stepDeg) {
    return {SnapAngleRadians(eulerRad.x, stepDeg), SnapAngleRadians(eulerRad.y, stepDeg),
            SnapAngleRadians(eulerRad.z, stepDeg)};
}

// Per-field result snappers for the gizmo drag path: quantize ONLY the field the active gizmo
// mode edits (position / euler / scale), leaving the other fields bit-untouched. All three are
// no-ops (return t unchanged) while cfg.enabled is false.
inline scene::Transform SnapTransformPosition(const scene::Transform& t, const SnapConfig& cfg) {
    if (!cfg.enabled) return t;
    scene::Transform out = t;
    out.position = SnapVec3(t.position, cfg.posStep);
    return out;
}
inline scene::Transform SnapTransformEuler(const scene::Transform& t, const SnapConfig& cfg) {
    if (!cfg.enabled) return t;
    scene::Transform out = t;
    out.eulerRadians = SnapEulerVec3(t.eulerRadians, cfg.angleStepDeg);
    return out;
}
inline scene::Transform SnapTransformScale(const scene::Transform& t, const SnapConfig& cfg) {
    if (!cfg.enabled) return t;
    scene::Transform out = t;
    out.scale = SnapVec3(t.scale, cfg.scaleStep);
    return out;
}

// Snap a TransformEdit's ABSOLUTE set payloads (the inspector commit route) per the config. Only
// the fields whose set* flag is raised are quantized; add* deltas and unset fields pass through
// bit-untouched. A disabled config returns the edit unchanged. Applying this at the recorded-
// wrapper boundary means the ED5 history captures the SNAPPED after-value.
inline TransformEdit SnapTransformEdit(const TransformEdit& e, const SnapConfig& cfg) {
    if (!cfg.enabled) return e;
    TransformEdit out = e;
    if (out.setPosition) out.position = SnapVec3(out.position, cfg.posStep);
    if (out.setEuler) out.euler = SnapEulerVec3(out.euler, cfg.angleStepDeg);
    if (out.setScale) out.scale = SnapVec3(out.scale, cfg.scaleStep);
    return out;
}

// --- Multi-select set -------------------------------------------------------------------------

// Is `index` a member of the sorted multi-select set?
inline bool SelectionContains(const std::vector<int>& sel, int index) {
    return std::binary_search(sel.begin(), sel.end(), index);
}

// The shared Ctrl+click toggle over the sorted multi-select set (see the header invariant).
// `primary` is the last-clicked/primary index (updated in place).
inline void ToggleIndexInSelection(std::vector<int>& sel, int& primary, int index) {
    if (index < 0) return;
    if (sel.empty()) {
        if (index == primary) return;              // Ctrl+click on the sole selection: no-op
        if (primary >= 0) sel.push_back(primary);  // seed the set with the current single selection
    }
    auto it = std::lower_bound(sel.begin(), sel.end(), index);
    if (it != sel.end() && *it == index) {
        // Toggle OUT: leave the set; the primary re-anchors to the largest remaining member.
        sel.erase(it);
        if (primary == index && !sel.empty()) primary = sel.back();
        if (sel.size() < 2) {   // collapse back to single-select
            if (sel.size() == 1) primary = sel[0];
            sel.clear();
        }
    } else {
        // Toggle IN: join sorted; the joined index becomes the primary (last-clicked).
        sel.insert(it, index);
        primary = index;
        if (sel.size() < 2) sel.clear();   // a lone member (no prior selection) = single-select
    }
}

}  // namespace hf::editor
