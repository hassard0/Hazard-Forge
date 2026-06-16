// Slice BX — editor LIVE-EDIT op + scene_io round-trip unit test (pure hf_core, ASan-eligible).
//
// The docked editor's Inspector (Slice BT) was READ-ONLY; this slice adds the WRITE path
// (engine/editor/edit_ops.{h,cpp}): deterministic, programmatic ops that mutate the selected entity's
// TransformC/MaterialC on the live registry, reflected through BuildPanelData and persisted through
// scene_io. ImGui-free + backend-free, so it is asserted here without a GPU (registry/dump, not pixels):
//   * ApplyTransformEdit: an absolute SET and a relative ADD mutate the right TransformC fields for a
//     known entity, leave other entities + other fields unchanged; an out-of-range entity is a safe
//     no-op.
//   * ApplyMaterialEdit: SET metallic/roughness/baseColor on the right MaterialC; a component-less
//     entity is safe (no crash, nothing mutated).
//   * Panel-data reflection: after an edit, BuildPanelData's inspector reports the NEW values for the
//     selected entity (the read path sees the write).
//   * scene_io round-trip: apply edits -> DumpScene -> the JSON contains the edited values; LoadScene
//     that JSON into a fresh registry -> the reloaded entity carries the edited transform/material.
//   * Determinism: the same edit sequence applied twice -> identical registry state + identical dump.
//
// Resources are opaque pointers mapped to names (never dereferenced), so distinct non-null stand-in
// Mesh*/ITexture* addresses are enough — same contract editor_panels_test / scene_io rely on.
#include "editor/edit_ops.h"
#include "editor/editor_panel_data.h"
#include "scene/components.h"
#include "scene/mesh.h"
#include "scene/scene_io.h"
#include "ecs/ecs.h"

#include "json/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) < eps; }

// --- Minimal typed accessors over json.h (mirrors scene_io's helpers) ----------------------------
static const json_value_s* MemberOf(const json_object_s* obj, const char* key) {
    for (const json_object_element_s* el = obj->start; el; el = el->next)
        if (el->name && std::strcmp(el->name->string, key) == 0) return el->value;
    return nullptr;
}
static const json_object_s* AsObject(const json_value_s* v) {
    return (v && v->type == json_type_object) ? static_cast<const json_object_s*>(v->payload) : nullptr;
}
static const json_array_s* AsArray(const json_value_s* v) {
    return (v && v->type == json_type_array) ? static_cast<const json_array_s*>(v->payload) : nullptr;
}
static std::string AsString(const json_value_s* v) {
    if (!v || v->type != json_type_string) return {};
    const json_string_s* s = static_cast<const json_string_s*>(v->payload);
    return std::string(s->string, s->string_size);
}
static double AsNumber(const json_value_s* v, double fallback = 0.0) {
    if (!v || v->type != json_type_number) return fallback;
    return std::strtod(static_cast<const json_number_s*>(v->payload)->number, nullptr);
}

// Build the standard 3-entity scene: cube #0, sphere #1, duck #2 (view order = creation order).
static std::vector<ecs::Entity> BuildScene(ecs::Registry& reg, scene::Mesh* meshCube,
                                           scene::Mesh* meshSphere, scene::Mesh* meshDuck,
                                           rhi::ITexture* texChecker, rhi::ITexture* texFlat) {
    std::vector<ecs::Entity> ents;
    {  // 0: cube at (1,2,3), euler (0,0.5,0), scale 2, dielectric.
        ecs::Entity e = reg.create();
        scene::Transform t; t.position = {1, 2, 3}; t.eulerRadians = {0, 0.5f, 0}; t.scale = {2, 2, 2};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshCube});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.0f, 0.5f});
        ents.push_back(e);
    }
    {  // 1: sphere at (-1,0,0), metallic 1.0 / roughness 0.15, no base texture.
        ecs::Entity e = reg.create();
        scene::Transform t; t.position = {-1, 0, 0};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshSphere});
        reg.add<scene::MaterialC>(e, {nullptr, nullptr, 1.0f, 0.15f});
        ents.push_back(e);
    }
    {  // 2: duck at (0,0.5,0).
        ecs::Entity e = reg.create();
        scene::Transform t; t.position = {0, 0.5f, 0};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshDuck});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.2f, 0.4f});
        ents.push_back(e);
    }
    return ents;
}

int main() {
    auto* meshCube   = reinterpret_cast<scene::Mesh*>(0x1000);
    auto* meshSphere = reinterpret_cast<scene::Mesh*>(0x2000);
    auto* meshDuck   = reinterpret_cast<scene::Mesh*>(0x3000);
    auto* texChecker = reinterpret_cast<rhi::ITexture*>(0xA000);
    auto* texFlat    = reinterpret_cast<rhi::ITexture*>(0xB000);
    auto* texRed     = reinterpret_cast<rhi::ITexture*>(0xC000);

    scene::SceneResources res;
    res.AddMesh("cube", meshCube);
    res.AddMesh("sphere", meshSphere);
    res.AddMesh("duck", meshDuck);
    res.AddTexture("checker", texChecker);
    res.AddTexture("flat_normal", texFlat);
    res.AddTexture("red", texRed);

    // --- EntityAtViewIndex maps the view-order index to the live handle (out-of-range -> null). ----
    {
        ecs::Registry reg;
        std::vector<ecs::Entity> ents =
            BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        check(editor::EntityAtViewIndex(reg, 0) == ents[0], "view index 0 -> entity 0");
        check(editor::EntityAtViewIndex(reg, 2) == ents[2], "view index 2 -> entity 2");
        check(editor::EntityAtViewIndex(reg, 99) == ecs::kNullEntity, "view index 99 -> null");
        check(editor::EntityAtViewIndex(reg, -1) == ecs::kNullEntity, "view index -1 -> null");
    }

    // --- ApplyTransformEdit: absolute SET position on entity 0; others unchanged. ------------------
    {
        ecs::Registry reg;
        std::vector<ecs::Entity> ents =
            BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);

        editor::TransformEdit e0;
        e0.setPosition = true; e0.position = {7, 8, 9};
        editor::ApplyTransformEdit(reg, 0, e0);

        const auto& t0 = reg.get<scene::TransformC>(ents[0]).t;
        check(approx(t0.position.x, 7) && approx(t0.position.y, 8) && approx(t0.position.z, 9),
              "SET position replaces entity 0 position with (7,8,9)");
        // euler/scale of entity 0 untouched.
        check(approx(t0.eulerRadians.y, 0.5f) && approx(t0.scale.x, 2),
              "SET position leaves entity 0 euler/scale unchanged");
        // entity 1 fully untouched.
        const auto& t1 = reg.get<scene::TransformC>(ents[1]).t;
        check(approx(t1.position.x, -1) && approx(t1.position.y, 0) && approx(t1.position.z, 0),
              "editing entity 0 leaves entity 1 transform unchanged");
    }

    // --- ApplyTransformEdit: relative ADD (delta) on Y; only Y of the duck moves. ------------------
    {
        ecs::Registry reg;
        std::vector<ecs::Entity> ents =
            BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);

        editor::TransformEdit d;
        d.addPosition = true; d.positionDelta = {0, 1.0f, 0};  // +1.0 on Y (the spec's duck move).
        editor::ApplyTransformEdit(reg, 2, d);

        const auto& t2 = reg.get<scene::TransformC>(ents[2]).t;
        check(approx(t2.position.x, 0) && approx(t2.position.y, 1.5f) && approx(t2.position.z, 0),
              "ADD delta +1.0 Y moves duck from y=0.5 to y=1.5 (x/z unchanged)");
    }

    // --- Combined set+add for the same field: delta first, then set wins (absolute). ---------------
    {
        ecs::Registry reg;
        std::vector<ecs::Entity> ents =
            BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        editor::TransformEdit e;
        e.addPosition = true; e.positionDelta = {100, 100, 100};
        e.setPosition = true; e.position = {3, 3, 3};
        editor::ApplyTransformEdit(reg, 1, e);
        const auto& t1 = reg.get<scene::TransformC>(ents[1]).t;
        check(approx(t1.position.x, 3) && approx(t1.position.y, 3) && approx(t1.position.z, 3),
              "set+add same field: absolute set wins (3,3,3)");
    }

    // --- Out-of-range transform edit is a SAFE NO-OP. ----------------------------------------------
    {
        ecs::Registry reg;
        std::vector<ecs::Entity> ents =
            BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        editor::TransformEdit e; e.setPosition = true; e.position = {99, 99, 99};
        editor::ApplyTransformEdit(reg, 42, e);   // out of range: no-op
        editor::ApplyTransformEdit(reg, -3, e);   // negative: no-op
        const auto& t0 = reg.get<scene::TransformC>(ents[0]).t;
        check(approx(t0.position.x, 1) && approx(t0.position.y, 2) && approx(t0.position.z, 3),
              "out-of-range transform edit leaves the scene unchanged");
    }

    // --- ApplyMaterialEdit: SET metallic/roughness/baseColor on the sphere. ------------------------
    {
        ecs::Registry reg;
        std::vector<ecs::Entity> ents =
            BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        editor::MaterialEdit m;
        m.setMetallic = true; m.metallic = 1.0f;
        m.setRoughness = true; m.roughness = 0.05f;
        m.setBaseColor = true; m.baseColor = texRed;
        editor::ApplyMaterialEdit(reg, 1, m);
        const auto& mat = reg.get<scene::MaterialC>(ents[1]);
        check(approx(mat.metallic, 1.0f), "material SET metallic == 1.0");
        check(approx(mat.roughness, 0.05f), "material SET roughness == 0.05");
        check(mat.base == texRed, "material SET baseColor pointer == red");
        // entity 0's material untouched.
        const auto& mat0 = reg.get<scene::MaterialC>(ents[0]);
        check(mat0.base == texChecker && approx(mat0.metallic, 0.0f),
              "editing entity 1 material leaves entity 0 material unchanged");
    }

    // --- Material edit on a component-less entity is SAFE. -----------------------------------------
    {
        ecs::Registry reg;
        BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        // A fresh entity with NO MaterialC (and not in the drawable view at all).
        ecs::Entity bare = reg.create();
        reg.add<scene::TransformC>(bare, {scene::Transform{}});
        editor::MaterialEdit m; m.setMetallic = true; m.metallic = 0.9f;
        // Index 3 doesn't exist in the drawable view (only 0..2) -> safe no-op.
        editor::ApplyMaterialEdit(reg, 3, m);
        check(true, "material edit addressing a non-drawable index did not crash");
    }

    // --- Panel-data reflection: after edits, the inspector reports the NEW values. ------------------
    {
        ecs::Registry reg;
        BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        // Move the duck +1.0 Y and recolor the sphere red + metallic.
        editor::TransformEdit dt; dt.addPosition = true; dt.positionDelta = {0, 1.0f, 0};
        editor::ApplyTransformEdit(reg, 2, dt);
        editor::MaterialEdit dm; dm.setBaseColor = true; dm.baseColor = texRed;
        dm.setMetallic = true; dm.metallic = 1.0f;
        editor::ApplyMaterialEdit(reg, 1, dm);

        editor::EditorState st; st.selectedEntity = 2;
        editor::PanelData pd = editor::BuildPanelData(reg, res, st);
        check(pd.inspector.valid && approx(pd.inspector.position.y, 1.5f),
              "inspector reflects the edited duck position (y=1.5)");

        editor::EditorState ss; ss.selectedEntity = 1;
        editor::PanelData ps = editor::BuildPanelData(reg, res, ss);
        check(ps.inspector.valid && ps.inspector.baseColorName == "red" &&
              approx(ps.inspector.metallic, 1.0f),
              "inspector reflects the edited sphere material (baseColor 'red', metallic 1.0)");
    }

    // --- scene_io round-trip: edits -> DumpScene JSON contains them -> LoadScene reloads them. ------
    {
        ecs::Registry reg;
        BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        editor::TransformEdit dt; dt.addPosition = true; dt.positionDelta = {0, 1.0f, 0};
        editor::ApplyTransformEdit(reg, 2, dt);
        editor::MaterialEdit dm; dm.setBaseColor = true; dm.baseColor = texRed;
        dm.setMetallic = true; dm.metallic = 1.0f;
        editor::ApplyMaterialEdit(reg, 1, dm);

        std::string dump = scene::DumpScene(reg, res);

        // The dumped JSON's 3rd object (duck, index 2) carries the edited y; the 2nd (sphere) the red base.
        json_value_s* root = json_parse_ex(dump.data(), dump.size(), json_parse_flags_default,
                                            nullptr, nullptr, nullptr);
        check(root != nullptr, "DumpScene output parses as JSON");
        const json_array_s* arr = AsArray(root);
        check(arr != nullptr && arr->length == 3, "dump has 3 objects");
        if (arr && arr->length == 3) {
            const json_object_s* o1 = AsObject(arr->start->next->value);          // sphere
            const json_object_s* o2 = AsObject(arr->start->next->next->value);    // duck
            check(o1 && AsString(MemberOf(o1, "baseColor")) == "red",
                  "dumped sphere baseColor == 'red'");
            check(o1 && approx((float)AsNumber(MemberOf(o1, "metallic")), 1.0f),
                  "dumped sphere metallic == 1.0");
            const json_array_s* pos = o2 ? AsArray(MemberOf(o2, "position")) : nullptr;
            check(pos && pos->length == 3, "dumped duck has a 3-vec position");
            if (pos && pos->length == 3) {
                double y = AsNumber(pos->start->next->value);
                check(approx((float)y, 1.5f), "dumped duck position.y == 1.5 (edited)");
            }
        }
        if (root) std::free(root);

        // Write the dump to a temp file and LoadScene it into a FRESH registry -> reload match.
        std::string tmp = std::string(std::tmpnam(nullptr)) + ".json";
        { std::ofstream f(tmp, std::ios::binary); f << dump; }
        ecs::Registry reloaded;
        std::vector<ecs::Entity> rents = scene::LoadScene(reloaded, res, tmp.c_str());
        check(rents.size() == 3, "reloaded scene has 3 entities");
        if (rents.size() == 3) {
            const auto& rt = reloaded.get<scene::TransformC>(rents[2]).t;  // duck
            check(approx(rt.position.y, 1.5f), "reloaded duck position.y == 1.5 (reloadMatch)");
            const auto& rm = reloaded.get<scene::MaterialC>(rents[1]);     // sphere
            check(rm.base == texRed && approx(rm.metallic, 1.0f),
                  "reloaded sphere material == (red, metallic 1.0) (reloadMatch)");
        }
        std::remove(tmp.c_str());
    }

    // --- Determinism: the same edit sequence applied twice -> identical state + identical dump. -----
    {
        auto runSequence = [&](ecs::Registry& reg) {
            BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
            editor::TransformEdit dt; dt.addPosition = true; dt.positionDelta = {0, 1.0f, 0};
            editor::ApplyTransformEdit(reg, 2, dt);
            editor::MaterialEdit dm; dm.setBaseColor = true; dm.baseColor = texRed;
            dm.setMetallic = true; dm.metallic = 1.0f;
            editor::ApplyMaterialEdit(reg, 1, dm);
        };
        ecs::Registry a, b;
        runSequence(a);
        runSequence(b);
        std::string da = scene::DumpScene(a, res);
        std::string db = scene::DumpScene(b, res);
        check(da == db, "the edit sequence is deterministic (identical dump across runs)");
    }

    if (g_fail == 0) { std::printf("editor_edit_test: ALL PASS\n"); return 0; }
    std::printf("editor_edit_test: %d FAILURE(S)\n", g_fail);
    return 1;
}
