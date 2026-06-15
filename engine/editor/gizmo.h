#pragma once
// Slice AB — transform gizmo: selection state + immediate-mode gizmo drawing (through the Slice-W
// debug-line layer) + axis hit-testing + drag math. PURE C++ (math + scene::Transform + debug::
// DebugDraw); NO rhi/SDL/backend symbols, so it lives in hf_core (ASan-scoped, unit-tested).
//
// The gizmo is WORLD-AXIS-ALIGNED (drawn at the object's position, ignoring its rotation/scale) —
// the standard editor "global" manipulation mode. Translate = 3 axis arrows; Rotate = 3 axis-aligned
// wire circles; Scale = 3 axis lines with end boxes. The active/hovered axis is brightened.
//
// All hit-testing + drag math is deterministic and window-free (see tests/editor_test.cpp): the live
// mouse-drag in --fly only feeds these functions prev/current rays; the math itself is fully tested.

#include "debug/debug_draw.h"
#include "math/math.h"
#include "scene/transform.h"

namespace hf::editor {

using math::Ray;
using math::Vec3;

enum class GizmoMode { Translate, Rotate, Scale };

// Editor selection: which object (view-order index, -1 = none) and the active gizmo mode.
struct Selection {
    int index = -1;
    GizmoMode mode = GizmoMode::Translate;
    bool Has() const { return index >= 0; }
    void Clear() { index = -1; }
};

// Axis ids returned by PickGizmoAxis / accepted by ApplyDrag.
inline constexpr int kAxisNone = -1;
inline constexpr int kAxisX = 0;
inline constexpr int kAxisY = 1;
inline constexpr int kAxisZ = 2;

// Draw the gizmo for `xform` (uses only its position) into `dd` through the debug-line layer.
// `handleLen` sizes the handles (world units). `activeAxis` (0/1/2) is brightened; -1 = none.
void EmitGizmo(debug::DebugDraw& dd, const scene::Transform& xform, GizmoMode mode, float handleLen,
               int activeAxis = kAxisNone);

// Which axis handle (kAxisX/Y/Z) the ray hits within tolerance, or kAxisNone. Translate/Scale test
// the ray vs each axis segment; Rotate tests the ray vs each axis-aligned circle of radius handleLen.
int PickGizmoAxis(const Ray& ray, const scene::Transform& xform, GizmoMode mode, float handleLen);

// The new transform produced by dragging `axis` from ray `prev` to ray `cur`:
//   Translate -> position shifts along the world axis by the change in the ray's projection onto it.
//   Rotate    -> eulerRadians[axis] += signed angle swept about the axis in its plane.
//   Scale     -> scale[axis] += change in the closest-approach distance along the axis (clamped >0).
// Pure math; `axis` of kAxisNone returns the transform unchanged.
scene::Transform ApplyDrag(const scene::Transform& xform, GizmoMode mode, int axis, const Ray& prev,
                           const Ray& cur);

}  // namespace hf::editor
