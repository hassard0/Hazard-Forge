// Hazard Forge — editor LIVE-EDIT operations (Slice BX). See edit_ops.h.
#include "editor/edit_ops.h"

#include "scene/components.h"

namespace hf::editor {

using scene::MaterialC;
using scene::MeshC;
using scene::TransformC;

ecs::Entity EntityAtViewIndex(ecs::Registry& registry, int entity) {
    if (entity < 0) return ecs::kNullEntity;
    // Walk the SAME drawable view (TransformC + MeshC + MaterialC) the hierarchy / inspector /
    // scene_io address by, so the index here means exactly the inspector's selected index + the
    // scene-file order. Linear by design: the editor selection is a single index, and the scene
    // has a handful of entities.
    int i = 0;
    for (auto [e, tc, mc, mat] : registry.view<TransformC, MeshC, MaterialC>()) {
        (void)tc; (void)mc; (void)mat;
        if (i == entity) return e;
        ++i;
    }
    return ecs::kNullEntity;
}

void ApplyTransformEdit(ecs::Registry& registry, int entity, const TransformEdit& edit) {
    ecs::Entity e = EntityAtViewIndex(registry, entity);
    if (e == ecs::kNullEntity || !registry.has<TransformC>(e)) return;  // safe no-op.

    scene::Transform& t = registry.get<TransformC>(e).t;

    // Per field: apply the relative DELTA first (if requested), then the absolute SET (if requested),
    // so a set always dominates a same-field delta — the documented "set wins" rule.
    if (edit.addPosition) {
        t.position.x += edit.positionDelta.x;
        t.position.y += edit.positionDelta.y;
        t.position.z += edit.positionDelta.z;
    }
    if (edit.setPosition) t.position = edit.position;

    if (edit.addEuler) {
        t.eulerRadians.x += edit.eulerDelta.x;
        t.eulerRadians.y += edit.eulerDelta.y;
        t.eulerRadians.z += edit.eulerDelta.z;
    }
    if (edit.setEuler) t.eulerRadians = edit.euler;

    if (edit.addScale) {
        t.scale.x += edit.scaleDelta.x;
        t.scale.y += edit.scaleDelta.y;
        t.scale.z += edit.scaleDelta.z;
    }
    if (edit.setScale) t.scale = edit.scale;
}

void ApplyMaterialEdit(ecs::Registry& registry, int entity, const MaterialEdit& edit) {
    ecs::Entity e = EntityAtViewIndex(registry, entity);
    if (e == ecs::kNullEntity || !registry.has<MaterialC>(e)) return;  // safe no-op.

    MaterialC& mat = registry.get<MaterialC>(e);
    if (edit.setMetallic)  mat.metallic = edit.metallic;
    if (edit.setRoughness) mat.roughness = edit.roughness;
    if (edit.setBaseColor) mat.base = edit.baseColor;
}

}  // namespace hf::editor
