#pragma once
// Hazard Forge — editor LIVE-EDIT operations (Slice BX, pure CPU, ImGui-free, backend-free).
//
// The docked editor's Inspector (Slice BT, editor_panel_data.{h,cpp}) was READ-ONLY: it reflects the
// selected entity's TransformC/MaterialC. THIS module adds the WRITE path — deterministic, programmatic
// edit operations that MUTATE the selected entity's components on the live ecs::Registry, so the editor
// can change the scene and (via scene_io::DumpScene/LoadScene) PERSIST those changes. The edit is
// reflected straight back through BuildPanelData (the read path sees the write) and survives a
// scene_io round-trip (DumpScene -> JSON -> LoadScene yields the edited values).
//
// Pure CPU: touches ONLY the ECS Registry + scene components + math. ZERO vk*/Metal/rhi rendering
// symbols (MaterialEdit carries an opaque rhi::ITexture* base-color pointer as an opaque VALUE — never
// dereferenced — exactly the same opaque-pointer contract scene_io / editor_panel_data rely on), so it
// lives in hf_core and is unit-tested headlessly (assert the registry/dump, not pixels).
//
// Set-vs-delta semantics (documented, LOCKED):
//   - Each TransformEdit field carries an independent mode flag. `setPosition`/`setEuler`/`setScale`
//     REPLACE that transform field with the supplied value (absolute set). `addPosition`/`addEuler`/
//     `addScale` ADD the supplied value to the current field (relative delta). A field with neither
//     flag set is left untouched. (Set and add for the SAME field both being true is not meaningful;
//     set wins — it is applied after the delta, so the absolute value dominates.)
//   - MaterialEdit fields are absolute SETS only (metallic/roughness/baseColor), each gated by its own
//     `set*` flag. baseColor is a texture POINTER the caller resolved from a name (SceneResources);
//     edit_ops never dereferences it.
//   - Both ops are SAFE NO-OPS on an out-of-range / stale / component-less entity (they look the entity
//     up in view order and verify the component is present before mutating).

#include "ecs/ecs.h"
#include "math/math.h"
#include "rhi/rhi.h"

namespace hf::editor {

// A transform mutation for one entity. Each field is independent: a `set*` flag REPLACES the field
// with `*` (absolute), an `add*` flag ADDS `*Delta` to the current field (relative). Unset fields are
// left unchanged. When both set and add are requested for the same field, the delta is applied first
// and the absolute set second (set wins).
struct TransformEdit {
    // Position.
    math::Vec3 position{0, 0, 0};
    math::Vec3 positionDelta{0, 0, 0};
    bool setPosition = false;
    bool addPosition = false;
    // Euler (radians).
    math::Vec3 euler{0, 0, 0};
    math::Vec3 eulerDelta{0, 0, 0};
    bool setEuler = false;
    bool addEuler = false;
    // Scale.
    math::Vec3 scale{1, 1, 1};
    math::Vec3 scaleDelta{0, 0, 0};
    bool setScale = false;
    bool addScale = false;
};

// A material mutation for one entity. Each factor is an absolute SET gated by its own flag. baseColor
// is an opaque texture pointer (resolved by the caller from a SceneResources name); it is stored as a
// value and never dereferenced here.
struct MaterialEdit {
    float metallic = 0.0f;
    bool setMetallic = false;
    float roughness = 0.5f;
    bool setRoughness = false;
    rhi::ITexture* baseColor = nullptr;
    bool setBaseColor = false;
};

// The same view-order addressing the editor uses: `entity` is the index into the drawable-entity view
// (TransformC + MeshC + MaterialC), matching HierarchyRow / InspectorData.index and the scene file
// order. Resolve it to the live ECS handle (or kNullEntity if out of range).
ecs::Entity EntityAtViewIndex(ecs::Registry& registry, int entity);

// Apply a transform edit to the entity at view-order index `entity`. Safe no-op if the index is out of
// range or the entity lacks a TransformC. Mutates the TransformC in place per the set/delta semantics.
void ApplyTransformEdit(ecs::Registry& registry, int entity, const TransformEdit& edit);

// Apply a material edit to the entity at view-order index `entity`. Safe no-op if the index is out of
// range or the entity lacks a MaterialC. Mutates the MaterialC's metallic/roughness/base per the set
// flags.
void ApplyMaterialEdit(ecs::Registry& registry, int entity, const MaterialEdit& edit);

}  // namespace hf::editor
