// Slice ED6 — ENTITY-CREATION edit op + undo-stack enrollment unit test (pure hf_core,
// ASan-eligible).
//
// edit_ops2.h adds the entity-LIFETIME write path behind the asset browser's click-to-place:
// ApplyCreateEntity spawns a drawable entity {TransformC + MeshC + MaterialC} for a mesh chosen BY
// NAME, at the deterministic default spawn transform, APPENDED at the END of the drawable view so
// every existing view-order index (the edit_ops/edit_history addressing) stays valid. Asserted
// here without a GPU (registry state + DumpScene bytes, not pixels — the editor_edit_test
// discipline):
//   (a) PLACEMENT: count +1, the spawn is LAST in view order, mesh resolves by name, TransformC
//       is BITWISE DefaultSpawnTransform(pre-spawn count), material = the LoadScene defaults, the
//       dump round-trips through LoadScene, existing entities' view indices untouched.
//   (b) SAFE NO-OPS: an unknown mesh name returns -1 with the registry untouched (dump
//       byte-identical); ApplyDestroyLastEntity refuses any non-last view index.
//   (c) ED5 ENROLLMENT: RecordedApplyCreateEntity records exactly one EntityCreate command; Undo
//       destroys the spawn (dump == the pre-spawn bytes) and Redo re-creates it bit-exactly
//       (dump == the post-spawn bytes); a transform edit ON the spawn undoes/redoes across the
//       create boundary (LIFO) back to the baseline bytes.
//   (d) ARTIFACT: a history holding an EntityCreate serializes -> deserializes digest-identically
//       under BOTH encodings (raw + NAMED, where the mesh pointer round-trips through its
//       registered name), and ReplayHistory onto a fresh baseline scene reproduces the post-spawn
//       dump byte-for-byte. An unknown mesh name in a named artifact is rejected.
//   (e) DETERMINISM: two identical place sequences on identical scenes -> byte-identical dumps.
//
// Resources are opaque pointers mapped to names (never dereferenced) — the editor_edit_test /
// scene_io contract.
#include "editor/edit_history.h"
#include "editor/edit_ops2.h"
#include "ecs/ecs.h"
#include "scene/components.h"
#include "scene/scene_io.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// The standard 3-entity fixture (the edit_history_test scene): cube #0, sphere #1, duck #2.
static void BuildScene(ecs::Registry& reg, scene::Mesh* meshCube, scene::Mesh* meshSphere,
                       scene::Mesh* meshDuck, rhi::ITexture* texChecker, rhi::ITexture* texFlat) {
    {
        ecs::Entity e = reg.create();
        scene::Transform t; t.position = {1, 2, 3}; t.eulerRadians = {0, 0.5f, 0}; t.scale = {2, 2, 2};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshCube});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.0f, 0.5f});
    }
    {
        ecs::Entity e = reg.create();
        scene::Transform t; t.position = {-1, 0, 0};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshSphere});
        reg.add<scene::MaterialC>(e, {nullptr, nullptr, 1.0f, 0.15f});
    }
    {
        ecs::Entity e = reg.create();
        scene::Transform t; t.position = {0, 0.5f, 0};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshCube});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.2f, 0.4f});
        (void)meshDuck;
    }
}

int main() {
    HF_TEST_MAIN_INIT();

    // Opaque fake resources (never dereferenced — the scene_io contract).
    scene::Mesh* meshCube   = reinterpret_cast<scene::Mesh*>(0x1001);
    scene::Mesh* meshSphere = reinterpret_cast<scene::Mesh*>(0x1002);
    scene::Mesh* meshDuck   = reinterpret_cast<scene::Mesh*>(0x1003);
    rhi::ITexture* texChecker = reinterpret_cast<rhi::ITexture*>(0x2001);
    rhi::ITexture* texFlat    = reinterpret_cast<rhi::ITexture*>(0x2002);
    scene::SceneResources res;
    res.AddMesh("cube", meshCube);
    res.AddMesh("sphere", meshSphere);
    res.AddMesh("duck", meshDuck);
    res.AddTexture("checker", texChecker);
    res.AddTexture("flat_normal", texFlat);

    // ---------------- (a) Placement ----------------------------------------------------------------
    {
        ecs::Registry reg;
        BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        const std::string baseline = scene::DumpScene(reg, res);
        const int before = editor::DrawableEntityCount(reg);
        check(before == 3, "(a) fixture has 3 drawables");

        const int idx = editor::ApplyCreateEntity(reg, res, "duck");
        check(idx == before, "(a) spawn lands at the END of the view (index == pre-spawn count)");
        check(editor::DrawableEntityCount(reg) == before + 1, "(a) drawable count +1");

        ecs::Entity e = editor::EntityAtViewIndex(reg, idx);
        check(e != ecs::kNullEntity, "(a) spawn resolvable at its view index");
        check(reg.get<scene::MeshC>(e).mesh == meshDuck, "(a) MeshC is the chosen mesh");
        check(res.NameOfMesh(reg.get<scene::MeshC>(e).mesh) == "duck",
              "(a) MeshC resolves by name");
        const editor::XformState got = editor::CaptureXform(reg.get<scene::TransformC>(e).t);
        const editor::XformState want =
            editor::CaptureXform(editor::DefaultSpawnTransform(before));
        check(editor::BitEqual(got, want), "(a) TransformC bitwise == DefaultSpawnTransform");
        const scene::MaterialC& m = reg.get<scene::MaterialC>(e);
        check(m.base == nullptr && m.normal == nullptr && m.metallic == 0.0f &&
              m.roughness == 0.5f, "(a) material = the LoadScene defaults");

        // Existing entities' view indices untouched (the addressing contract).
        ecs::Entity e1 = editor::EntityAtViewIndex(reg, 1);
        check(reg.get<scene::MeshC>(e1).mesh == meshSphere, "(a) existing view index 1 unchanged");

        // The dump gained the spawn and round-trips through LoadScene (the scene_io contract).
        const std::string post = scene::DumpScene(reg, res);
        check(post != baseline && post.find("\"mesh\": \"duck\"") != std::string::npos,
              "(a) DumpScene carries the spawn");
        {
            ecs::Registry rt;
            const std::string canon = scene::CanonicalizeSceneText(post.c_str(), rt, res);
            check(canon == post, "(a) post-spawn dump is canonical (LoadScene round-trip)");
        }
    }

    // ---------------- (b) Safe no-ops ---------------------------------------------------------------
    {
        ecs::Registry reg;
        BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        const std::string baseline = scene::DumpScene(reg, res);
        check(editor::ApplyCreateEntity(reg, res, "no-such-mesh") == -1,
              "(b) unknown mesh -> -1");
        check(scene::DumpScene(reg, res) == baseline, "(b) unknown mesh leaves registry untouched");
        check(!editor::ApplyDestroyLastEntity(reg, 0), "(b) destroy refuses a non-last index");
        check(!editor::ApplyDestroyLastEntity(reg, 3), "(b) destroy refuses an out-of-range index");
        check(scene::DumpScene(reg, res) == baseline, "(b) refused destroys leave registry untouched");
        check(editor::ApplyDestroyLastEntity(reg, 2), "(b) destroy accepts the last index");
        check(editor::DrawableEntityCount(reg) == 2, "(b) accepted destroy removed the last");
    }

    // ---------------- (c) ED5 enrollment: record -> undo -> redo ------------------------------------
    std::string postSpawnDump;  // shared with (d)
    editor::EditHistory liveHist;
    {
        ecs::Registry reg;
        BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        const std::string baseline = scene::DumpScene(reg, res);
        const editor::EditTargets targets{&reg, nullptr};

        const int idx = editor::RecordedApplyCreateEntity(liveHist, reg, res, "duck");
        check(idx == 3 && liveHist.commands.size() == 1, "(c) create recorded exactly 1 command");
        check(liveHist.commands[0].kind ==
                  static_cast<uint32_t>(editor::EditCmdKind::EntityCreate),
              "(c) the command kind is EntityCreate");
        postSpawnDump = scene::DumpScene(reg, res);

        check(editor::Undo(liveHist, targets), "(c) Undo succeeds");
        check(scene::DumpScene(reg, res) == baseline, "(c) undo dump == baseline bytes");
        check(editor::DrawableEntityCount(reg) == 3, "(c) undo removed the spawn");
        check(editor::Redo(liveHist, targets), "(c) Redo succeeds");
        check(scene::DumpScene(reg, res) == postSpawnDump, "(c) redo dump == post-spawn bytes");

        // A transform edit ON the spawn, then LIFO undo across the create boundary -> baseline.
        editor::TransformEdit te;
        te.setPosition = true;
        te.position = {9, 9, 9};
        editor::RecordedApplyTransformEdit(liveHist, reg, idx, te);
        check(liveHist.commands.size() == 2, "(c) transform edit on the spawn recorded");
        check(editor::Undo(liveHist, targets) && editor::Undo(liveHist, targets),
              "(c) undo x2 across the create boundary succeeds");
        check(scene::DumpScene(reg, res) == baseline, "(c) LIFO undo x2 -> baseline bytes");
        check(editor::Redo(liveHist, targets) && editor::Redo(liveHist, targets),
              "(c) redo x2 succeeds");
        // Trim back to just the create for (d)'s replay fixture.
        editor::Undo(liveHist, targets);
        liveHist.commands.resize(liveHist.cursor);
    }

    // ---------------- (d) Serialization artifact (raw + named) --------------------------------------
    {
        // Raw (in-process) encoding.
        const std::vector<uint8_t> raw = editor::SerializeHistory(liveHist);
        editor::EditHistory rtRaw;
        check(editor::DeserializeHistory(raw.data(), raw.size(), rtRaw),
              "(d) raw artifact deserializes");
        check(editor::DigestHistory(rtRaw) == editor::DigestHistory(liveHist),
              "(d) raw round-trip digest identical");

        // Named (process-portable) encoding: the mesh pointer rides as its registered NAME.
        const std::vector<uint8_t> named = editor::SerializeHistory(liveHist, &res);
        editor::EditHistory rtNamed;
        check(editor::DeserializeHistory(named.data(), named.size(), rtNamed, &res),
              "(d) named artifact deserializes against the resources table");
        check(editor::DigestHistory(rtNamed, &res) == editor::DigestHistory(liveHist, &res),
              "(d) named round-trip digest identical");
        {
            editor::EditHistory scratch;  // separate out-param: Deserialize clears it on entry
            check(!editor::DeserializeHistory(named.data(), named.size(), scratch),
                  "(d) named artifact REQUIRES a resources table");
        }

        // Replay onto a fresh baseline scene reproduces the post-spawn dump byte-for-byte.
        ecs::Registry fresh;
        BuildScene(fresh, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        check(editor::ReplayHistory(rtNamed, editor::EditTargets{&fresh, nullptr}),
              "(d) replay applies");
        check(scene::DumpScene(fresh, res) == postSpawnDump,
              "(d) replayed fresh scene dump == post-spawn bytes");

        // A named artifact referencing a mesh this process did not register is rejected.
        scene::SceneResources poor;
        poor.AddMesh("cube", meshCube);
        editor::EditHistory rej;
        check(!editor::DeserializeHistory(named.data(), named.size(), rej, &poor),
              "(d) unknown mesh name in a named artifact rejected");
    }

    // ---------------- (e) Determinism ---------------------------------------------------------------
    {
        auto place = [&]() {
            ecs::Registry reg;
            BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
            editor::ApplyCreateEntity(reg, res, "sphere");
            editor::ApplyCreateEntity(reg, res, "cube");
            return scene::DumpScene(reg, res);
        };
        check(place() == place(), "(e) two identical place sequences -> byte-identical dumps");
    }

    if (g_fail == 0) std::printf("edit_ops2_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
