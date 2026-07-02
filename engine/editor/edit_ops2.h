#pragma once
// Hazard Forge — editor ENTITY-CREATION edit op (Slice ED6, pure CPU, ImGui-free, backend-free).
//
// The sibling of the FROZEN edit_ops.h (which ships the transform/material mutation ops): THIS
// module adds the entity-LIFETIME write path the asset browser needs — spawn a new drawable entity
// {TransformC + MeshC + MaterialC} for a mesh chosen BY NAME from SceneResources, at a
// deterministic default spawn transform. The op is pure (ECS registry + scene components + opaque
// named resource pointers, ZERO vk*/Metal/rhi rendering symbols — the edit_ops.h contract), so it
// lives in hf_core, is unit-tested headlessly, and the --ed6-dry-run's hand-called twin is the
// SAME function the panel's click affordance drives.
//
// Placement semantics (documented, LOCKED):
//   - The new entity is created via the CANONICAL scene_io creation path (reg.create() + add
//     TransformC/MeshC/MaterialC, the exact LoadScene recipe), so it APPENDS at the END of the
//     drawable view order: every existing entity's view-order index is UNCHANGED (the edit_ops /
//     edit_history addressing stays valid), and the new entity's view index == the pre-spawn
//     drawable count.
//   - The default spawn transform is a pure function of the PRE-SPAWN drawable count
//     (DefaultSpawnTransform): origin + a small per-existing-entity XZ offset so consecutive
//     spawns land at distinct, deterministic spots. Default material = the LoadScene defaults
//     (no textures, metallic 0, roughness 0.5) — DumpScene emits it as baseColor/normalMap null.
//   - An unknown mesh name is a SAFE NO-OP returning -1 (registry untouched) — the edit_ops
//     bad-target discipline.
//   - The undo twin (ApplyDestroyLastEntity) destroys the entity at a view index ONLY when it is
//     the LAST drawable: removing the last dense slot of each component pool is a pure pop_back
//     (no swap), so the surviving view order is restored BIT-IDENTICALLY. Guarded (returns false
//     otherwise) so the LIFO undo contract can never silently corrupt view-index addressing.

#include <string>

#include "ecs/ecs.h"
#include "editor/edit_ops.h"     // EntityAtViewIndex — the shared view-order addressing
#include "scene/components.h"
#include "scene/scene_io.h"

namespace hf::editor {

// Count the drawable entities (TransformC + MeshC + MaterialC) in view order — the denominator of
// the view-index addressing edit_ops uses.
inline int DrawableEntityCount(ecs::Registry& reg) {
    int n = 0;
    for (auto [e, tc, mc, mat] :
         reg.view<scene::TransformC, scene::MeshC, scene::MaterialC>()) {
        (void)e; (void)tc; (void)mc; (void)mat;
        ++n;
    }
    return n;
}

// The deterministic default spawn transform for the (existingCount+1)-th drawable: origin + a
// small XZ offset per existing entity (so consecutive spawns don't overlap), half a unit up (so a
// unit mesh sits on the ground plane). All values exact in float for small counts.
inline scene::Transform DefaultSpawnTransform(int existingCount) {
    scene::Transform t;  // defaults: euler 0, scale 1
    const float o = 0.5f * static_cast<float>(existingCount);
    t.position = {o, 0.5f, o};
    return t;
}

// The default spawn material — the LoadScene defaults (no textures, metallic 0, roughness 0.5).
inline scene::MaterialC DefaultSpawnMaterial() {
    return scene::MaterialC{nullptr, nullptr, 0.0f, 0.5f};
}

// The full-control core: append one drawable entity with EXACTLY the given components (the
// canonical LoadScene creation recipe). Returns the new entity's view-order index (== the
// pre-spawn drawable count), or -1 on a null mesh (safe no-op). The undo/redo replay path uses
// this so a re-created entity is bit-identical to the recorded one.
inline int ApplyCreateEntityRaw(ecs::Registry& reg, scene::Mesh* mesh, const scene::Transform& t,
                                const scene::MaterialC& m) {
    if (!mesh) return -1;
    const int index = DrawableEntityCount(reg);
    ecs::Entity e = reg.create();
    reg.add<scene::TransformC>(e, {t});
    reg.add<scene::MeshC>(e, {mesh});
    reg.add<scene::MaterialC>(e, m);
    return index;
}

// The editor-facing op the asset browser's place affordance drives: spawn a new drawable entity
// with the mesh resolved BY NAME from `resources`, at the deterministic default spawn transform +
// material. Returns the new entity's view-order index, or -1 when the name is unknown (safe
// no-op, registry untouched).
inline int ApplyCreateEntity(ecs::Registry& reg, const scene::SceneResources& resources,
                             const std::string& meshName) {
    scene::Mesh* mesh = resources.FindMesh(meshName);
    if (!mesh) return -1;
    return ApplyCreateEntityRaw(reg, mesh, DefaultSpawnTransform(DrawableEntityCount(reg)),
                                DefaultSpawnMaterial());
}

// The undo twin of ApplyCreateEntityRaw: destroy the entity at view index `viewIndex`, but ONLY
// when it is the LAST drawable (the LIFO-undo guarantee — see the header note). Destroying the
// last drawable pops the last dense slot of each pool without a swap, so the surviving entities'
// view order (and therefore every earlier command's view-index target) is restored bit-identically.
// Returns false (registry untouched) on any other index.
inline bool ApplyDestroyLastEntity(ecs::Registry& reg, int viewIndex) {
    if (viewIndex < 0 || viewIndex != DrawableEntityCount(reg) - 1) return false;
    ecs::Entity e = EntityAtViewIndex(reg, viewIndex);
    if (e == ecs::kNullEntity) return false;
    reg.destroy(e);
    return true;
}

}  // namespace hf::editor
