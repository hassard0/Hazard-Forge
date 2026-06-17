// Slice BT — editor PANEL DATA model unit test (pure hf_core, ASan-eligible).
//
// The docked editor (Hierarchy/Inspector/Stats/Viewport) is golden-verified by --editor-shot, but the
// DATA each panel displays is factored into BuildPanelData (engine/editor/editor_panel_data.{h,cpp}) —
// ImGui-free, backend-free — so it can be asserted here without a GPU or an ImGui context. This test
// builds a known registry + named resources and checks:
//   * Panel data model: the hierarchy lists the expected entities (ids/labels), the inspector for the
//     selected entity reports the expected Transform/Material/Mesh, and stats reports the expected
//     counts (entity/mesh/alive).
//   * Selection: changing EditorState.selectedEntity changes which entity the inspector reports; an
//     out-of-range selection is CLAMPED (>=count -> last; <0 with entities -> 0) and written back;
//     an empty scene yields selection -1 + inspector.valid == false ("none").
//   * Determinism: the panel data for a fixed registry is byte-stable across calls.
//
// Resources are opaque pointers mapped to names (never dereferenced), so distinct non-null stand-in
// Mesh*/ITexture* addresses are enough — same contract introspect_test/scene_io rely on.
#include "editor/editor_panel_data.h"
#include "scene/components.h"
#include "scene/mesh.h"
#include "scene/scene_io.h"
#include "ecs/ecs.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) < eps; }

int main() {
    HF_TEST_MAIN_INIT();
    // --- Build a known scene: cube #0, sphere #1, duck #2 (view order = creation order). ----------
    auto* meshCube   = reinterpret_cast<scene::Mesh*>(0x1000);
    auto* meshSphere = reinterpret_cast<scene::Mesh*>(0x2000);
    auto* meshDuck   = reinterpret_cast<scene::Mesh*>(0x3000);
    auto* texChecker = reinterpret_cast<rhi::ITexture*>(0xA000);
    auto* texFlat    = reinterpret_cast<rhi::ITexture*>(0xB000);

    scene::SceneResources res;
    res.AddMesh("cube", meshCube);
    res.AddMesh("sphere", meshSphere);
    res.AddMesh("duck", meshDuck);
    res.AddTexture("checker", texChecker);
    res.AddTexture("flat_normal", texFlat);

    ecs::Registry reg;
    std::vector<ecs::Entity> ents;
    {  // 0: cube at (1,2,3), euler (0,0.5,0), scale 2, dielectric (metallic 0, roughness 0.5).
        ecs::Entity e = reg.create();
        scene::Transform t; t.position = {1, 2, 3}; t.eulerRadians = {0, 0.5f, 0}; t.scale = {2, 2, 2};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshCube});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.0f, 0.5f});
        ents.push_back(e);
    }
    {  // 1: sphere, metallic 1.0 / roughness 0.15, no base texture.
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

    // --- Panel data model: hierarchy rows (ids/labels). -------------------------------------------
    {
        editor::EditorState state;  // default selection 0.
        editor::PanelData pd = editor::BuildPanelData(reg, res, state);

        check(pd.hierarchy.size() == 3, "hierarchy lists 3 drawable entities");
        if (pd.hierarchy.size() == 3) {
            check(pd.hierarchy[0].label == "cube #0", "row 0 label == 'cube #0'");
            check(pd.hierarchy[1].label == "sphere #1", "row 1 label == 'sphere #1'");
            check(pd.hierarchy[2].label == "duck #2", "row 2 label == 'duck #2'");
            check(pd.hierarchy[0].entity == ents[0], "row 0 entity == created entity 0");
            check(pd.hierarchy[2].entity == ents[2], "row 2 entity == created entity 2");
        }

        // Stats: 3 drawable entities, 3 meshes, 3 alive.
        check(pd.stats.entityCount == 3, "stats.entityCount == 3");
        check(pd.stats.meshCount == 3, "stats.meshCount == 3");
        check(pd.stats.aliveCount == 3, "stats.aliveCount == 3");

        // Inspector for the default selection (index 0 = cube).
        check(pd.inspector.valid, "inspector valid for default selection");
        check(pd.inspector.index == 0, "inspector reflects index 0");
        check(pd.inspector.label == "cube #0", "inspector label == 'cube #0'");
        check(pd.inspector.meshName == "cube", "inspector mesh name == 'cube'");
        check(pd.inspector.baseColorName == "checker", "inspector baseColor == 'checker'");
        check(approx(pd.inspector.position.x, 1) && approx(pd.inspector.position.y, 2) &&
              approx(pd.inspector.position.z, 3), "inspector position == (1,2,3)");
        check(approx(pd.inspector.eulerRadians.y, 0.5f), "inspector euler.y == 0.5");
        check(approx(pd.inspector.scale.x, 2), "inspector scale.x == 2");
        check(approx(pd.inspector.metallic, 0.0f) && approx(pd.inspector.roughness, 0.5f),
              "inspector material == (metallic 0, roughness 0.5)");
    }

    // --- Selection: choosing index 1 reports the sphere; metallic with no base texture. -----------
    {
        editor::EditorState state; state.selectedEntity = 1;
        editor::PanelData pd = editor::BuildPanelData(reg, res, state);
        check(pd.inspector.valid && pd.inspector.index == 1, "inspector follows selection to index 1");
        check(pd.inspector.meshName == "sphere", "selected mesh name == 'sphere'");
        check(pd.inspector.baseColorName.empty(), "metallic sphere has no base-color texture");
        check(approx(pd.inspector.metallic, 1.0f) && approx(pd.inspector.roughness, 0.15f),
              "selected material == (metallic 1, roughness 0.15)");
    }

    // --- Out-of-range selection clamps to the last entity + writes back. --------------------------
    {
        editor::EditorState state; state.selectedEntity = 99;
        editor::PanelData pd = editor::BuildPanelData(reg, res, state);
        check(state.selectedEntity == 2, "selection 99 clamped to last index (2) and written back");
        check(pd.inspector.valid && pd.inspector.index == 2, "inspector reflects clamped index 2");
        check(pd.inspector.meshName == "duck", "clamped selection is the duck");
    }

    // --- Negative selection with entities present snaps to 0. -------------------------------------
    {
        editor::EditorState state; state.selectedEntity = -5;
        editor::PanelData pd = editor::BuildPanelData(reg, res, state);
        check(state.selectedEntity == 0, "negative selection snaps to 0 (entities present)");
        check(pd.inspector.valid && pd.inspector.index == 0, "inspector reflects snapped index 0");
    }

    // --- Empty scene: selection -> -1, inspector.valid == false ("none"). -------------------------
    {
        ecs::Registry empty;
        editor::EditorState state; state.selectedEntity = 0;
        editor::PanelData pd = editor::BuildPanelData(empty, res, state);
        check(pd.hierarchy.empty(), "empty scene has no hierarchy rows");
        check(state.selectedEntity == -1, "empty scene clamps selection to -1 (none)");
        check(!pd.inspector.valid, "empty scene inspector is invalid (none selected)");
        check(pd.stats.entityCount == 0 && pd.stats.meshCount == 0, "empty scene stats are zero");
    }

    // --- Determinism: identical state -> identical panel data across calls. ------------------------
    {
        editor::EditorState s1; s1.selectedEntity = 2;
        editor::EditorState s2; s2.selectedEntity = 2;
        editor::PanelData a = editor::BuildPanelData(reg, res, s1);
        editor::PanelData b = editor::BuildPanelData(reg, res, s2);
        bool same = a.hierarchy.size() == b.hierarchy.size() &&
                    a.inspector.index == b.inspector.index &&
                    a.inspector.label == b.inspector.label &&
                    a.inspector.meshName == b.inspector.meshName &&
                    a.inspector.baseColorName == b.inspector.baseColorName &&
                    approx(a.inspector.position.x, b.inspector.position.x) &&
                    approx(a.inspector.metallic, b.inspector.metallic) &&
                    a.stats.entityCount == b.stats.entityCount &&
                    a.stats.meshCount == b.stats.meshCount;
        for (size_t i = 0; same && i < a.hierarchy.size(); ++i)
            same = same && a.hierarchy[i].label == b.hierarchy[i].label &&
                   a.hierarchy[i].entity == b.hierarchy[i].entity;
        check(same, "BuildPanelData is deterministic (byte-stable across calls)");
    }

    // --- The docked layout is a fixed, deterministic VALUE (code-driven, no imgui.ini). -----------
    {
        editor::DockLayout l1 = editor::DefaultDockLayout();
        editor::DockLayout l2 = editor::DefaultDockLayout();
        check(approx(l1.leftRatio, l2.leftRatio) && approx(l1.rightRatio, l2.rightRatio) &&
              approx(l1.leftBottomRatio, l2.leftBottomRatio), "DefaultDockLayout ratios are fixed");
        check(l1.leftRatio > 0.0f && l1.leftRatio < 1.0f &&
              l1.rightRatio > 0.0f && l1.rightRatio < 1.0f, "layout split ratios are in (0,1)");
        check(std::string(l1.viewportTitle) == "Viewport" &&
              std::string(l1.hierarchyTitle) == "Scene Hierarchy" &&
              std::string(l1.inspectorTitle) == "Inspector" &&
              std::string(l1.statsTitle) == "Stats", "layout panel titles match BuildEditorUI");
    }

    if (g_fail == 0) { std::printf("editor_panels_test: ALL PASS\n"); return 0; }
    std::printf("editor_panels_test: %d FAILURE(S)\n", g_fail);
    return 1;
}
