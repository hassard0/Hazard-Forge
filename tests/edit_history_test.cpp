// Slice ED5 — DETERMINISTIC UNDO/REDO COMMAND STACK unit test (pure hf_core, ASan-eligible).
//
// edit_history.h wraps the pure editor edit ops (edit_ops.h scene ops; flow_edit_ops.h graph ops) in
// a recorded, reversible, REPLAYABLE command stack. Asserted here without a GPU (registry/graph
// state + DumpScene bytes, not pixels — the editor_edit_test discipline):
//   (a) UNDO/REDO BYTE-IDENTITY: apply K recorded edits -> Undo K -> the scene's DumpScene equals
//       the pre-edit dump BYTE-IDENTICALLY; Redo K -> equals the post-edit dump byte-identically.
//   (b) INTERLEAVE (pinned): edit, edit, undo, NEW edit -> the redo tail truncates (history = 2
//       commands, the undone command is gone, Redo after the new edit fails).
//   (c) ROUND-TRIP + REPLAY: SerializeHistory -> DeserializeHistory -> DigestHistory identical;
//       ReplayHistory of the deserialized session onto a FRESH scene + FRESH graph reproduces the
//       live session's final DumpScene bytes + flow graph exactly (the edit session is a portable,
//       replayable artifact).
//   (d) PINNED CROSS-COMPILER DIGEST: a FIXED hand-authored history serializes to the pinned
//       FNV-1a-64 digest — the same constant MSVC and clang must both produce (the serialization is
//       hand-LE, layout-free; scripts/ reruns this under both compilers).
//   (e) FLOW-FAMILY ENROLLMENT: RecordedAddFlowNode / RecordedConnectFlow / RecordedDeleteFlowNode
//       round-trip through Undo/Redo with the graph digest chain base -> add -> base -> add (and a
//       delete of a node WITH inbound references undoes to the exact pre-delete graph).
//   Plus: safe no-op wrappers record nothing; malformed artifacts deserialize to false.
//
// Resources are opaque pointers mapped to names (never dereferenced) — the editor_edit_test /
// scene_io contract.
#include "editor/edit_history.h"
#include "scene/components.h"
#include "scene/mesh.h"
#include "scene/scene_io.h"
#include "ecs/ecs.h"

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

// Build the standard 3-entity scene (the editor_edit_test fixture): cube #0, sphere #1, duck #2.
static std::vector<ecs::Entity> BuildScene(ecs::Registry& reg, scene::Mesh* meshCube,
                                           scene::Mesh* meshSphere, scene::Mesh* meshDuck,
                                           rhi::ITexture* texChecker, rhi::ITexture* texFlat) {
    std::vector<ecs::Entity> ents;
    {
        ecs::Entity e = reg.create();
        scene::Transform t; t.position = {1, 2, 3}; t.eulerRadians = {0, 0.5f, 0}; t.scale = {2, 2, 2};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshCube});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.0f, 0.5f});
        ents.push_back(e);
    }
    {
        ecs::Entity e = reg.create();
        scene::Transform t; t.position = {-1, 0, 0};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshSphere});
        reg.add<scene::MaterialC>(e, {nullptr, nullptr, 1.0f, 0.15f});
        ents.push_back(e);
    }
    {
        ecs::Entity e = reg.create();
        scene::Transform t; t.position = {0, 0.5f, 0};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshDuck});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.2f, 0.4f});
        ents.push_back(e);
    }
    return ents;
}

// A small hand-built flow graph with real edges: n0=Const(5), n1=Const(7), n2=Add(n0,n1),
// n3=Mul(n2,n1). n2 has an inbound reference (n3.a) so deleting it exercises the cut-list undo.
static flow::Graph BuildFlowFixture() {
    flow::Graph g;
    auto push = [&](uint32_t kind, flow::NodeId a, flow::NodeId b, flow::NodeId c, flow::Reg ca) {
        flow::Node n; n.kind = kind; n.a = a; n.b = b; n.c = c; n.constArg = ca;
        g.nodes.push_back(n);
    };
    push(flow::kConst, 0, 0, 0, 5);   // n0 (self-sentinel inputs)
    push(flow::kConst, 1, 1, 1, 7);   // n1
    push(flow::kAdd,   0, 1, 2, 0);   // n2 = n0 + n1 (c unused -> sentinel)
    push(flow::kMul,   2, 1, 3, 0);   // n3 = n2 * n1
    return g;
}

// Memberwise graph equality (never memcmp a padded struct — the ed2-dry-run discipline).
static bool FlowEq(const flow::Graph& x, const flow::Graph& y) {
    if (x.nodes.size() != y.nodes.size()) return false;
    for (std::size_t i = 0; i < x.nodes.size(); ++i) {
        const flow::Node& a = x.nodes[i];
        const flow::Node& b = y.nodes[i];
        if (a.kind != b.kind || a.a != b.a || a.b != b.b || a.c != b.c || a.constArg != b.constArg)
            return false;
    }
    return true;
}

// Deterministic graph digest: hand-LE node fields -> DigestBytes (the pinned-golden currency).
static uint64_t FlowDigest(const flow::Graph& g) {
    std::vector<uint8_t> b;
    auto put = [&](uint32_t v) {
        b.push_back((uint8_t)v); b.push_back((uint8_t)(v >> 8));
        b.push_back((uint8_t)(v >> 16)); b.push_back((uint8_t)(v >> 24));
    };
    put((uint32_t)g.nodes.size());
    for (const flow::Node& n : g.nodes) {
        put(n.kind); put(n.a); put(n.b); put(n.c); put((uint32_t)n.constArg);
    }
    return net::DigestBytes(b.data(), b.size());
}

// The FIXED hand-authored history behind the PINNED cross-compiler digest (test d). Mirrored
// byte-for-byte by the standalone clang probe (scripts run it under clang++ and compare stdout);
// touch it ONLY together with the pinned constant.
static editor::EditHistory BuildPinnedHistory() {
    editor::EditHistory h;
    {   // A Transform command: entity 0 moved (1,2,3)->(7,8,9), euler/scale held.
        editor::EditCommand c;
        c.kind = (uint32_t)editor::EditCmdKind::Transform;
        c.target = 0;
        c.xBefore = editor::XformState{1, 2, 3, 0, 0.5f, 0, 2, 2, 2};
        c.xAfter  = editor::XformState{7, 8, 9, 0, 0.5f, 0, 2, 2, 2};
        Record(h, c);
    }
    {   // A Material command: entity 1 (metallic 1.0/rough 0.15/none) -> (0.875/0.05/0xC000).
        editor::EditCommand c;
        c.kind = (uint32_t)editor::EditCmdKind::Material;
        c.target = 1;
        c.mBefore = editor::MatState{1.0f, 0.15f, 0};
        c.mAfter  = editor::MatState{0.875f, 0.05f, 0xC000ull};
        Record(h, c);
    }
    // The flow family through the REAL recorded wrappers on the fixture graph.
    flow::Graph g = BuildFlowFixture();
    const flow::NodeId nid = editor::RecordedAddFlowNode(h, g, flow::kMax, 3);
    editor::RecordedConnectFlow(h, g, /*from=*/0, /*to=*/nid, /*slot=*/0);
    editor::RecordedDeleteFlowNode(h, g, /*victim=*/2);
    // One undo so the serialized cursor (4) differs from the command count (5) — pins that the
    // cursor is part of the artifact.
    editor::EditTargets t; t.flowGraph = &g;
    editor::Undo(h, t);
    return h;
}

// THE PINNED CROSS-COMPILER DIGEST of BuildPinnedHistory() (FNV-1a-64 over the hand-LE bytes).
// MSVC and clang must both produce this exact constant (verified at slice time with the standalone
// clang probe; the serialization is layout-free so any drift is a REAL regression).
static const uint64_t kPinnedHistoryDigest = 0x3cb790d71e9f35d2ull;

int main() {
    HF_TEST_MAIN_INIT();
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

    // The K recorded edits every scene test below applies (2 transform + 2 material = 4 commands).
    auto applyEdits = [&](ecs::Registry& reg, editor::EditHistory& h) {
        editor::TransformEdit t1;                       // absolute set: move the cube
        t1.setPosition = true; t1.position = {7, 8, 9};
        editor::RecordedApplyTransformEdit(h, reg, 0, t1);
        editor::TransformEdit t2;                       // relative add: lift the duck
        t2.addPosition = true; t2.positionDelta = {0, 1.0f, 0};
        editor::RecordedApplyTransformEdit(h, reg, 2, t2);
        editor::MaterialEdit m1;                        // recolor + metalize the sphere
        m1.setMetallic = true; m1.metallic = 1.0f;
        m1.setBaseColor = true; m1.baseColor = texRed;
        editor::RecordedApplyMaterialEdit(h, reg, 1, m1);
        editor::MaterialEdit m2;                        // roughen the duck
        m2.setRoughness = true; m2.roughness = 0.9f;
        editor::RecordedApplyMaterialEdit(h, reg, 2, m2);
    };

    // --- (a) Undo K -> pre-edit dump BYTE-IDENTICAL; Redo K -> post-edit dump byte-identical. -----
    {
        ecs::Registry reg;
        BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        const std::string preDump = scene::DumpScene(reg, res);

        editor::EditHistory h;
        applyEdits(reg, h);
        check(h.commands.size() == 4 && h.cursor == 4, "(a) 4 edits -> 4 commands, cursor 4");
        const std::string postDump = scene::DumpScene(reg, res);
        check(postDump != preDump, "(a) the edits changed the scene dump");

        const editor::EditTargets t{&reg, nullptr};
        for (int i = 0; i < 4; ++i) check(Undo(h, t), "(a) Undo succeeds");
        check(h.cursor == 0, "(a) cursor 0 after Undo 4");
        check(scene::DumpScene(reg, res) == preDump,
              "(a) Undo 4 -> DumpScene BYTE-IDENTICAL to the pre-edit dump");
        check(!Undo(h, t), "(a) Undo past the bottom fails (history untouched)");

        for (int i = 0; i < 4; ++i) check(Redo(h, t), "(a) Redo succeeds");
        check(h.cursor == 4, "(a) cursor 4 after Redo 4");
        check(scene::DumpScene(reg, res) == postDump,
              "(a) Redo 4 -> DumpScene BYTE-IDENTICAL to the post-edit dump");
        check(!Redo(h, t), "(a) Redo past the top fails (history untouched)");
    }

    // --- (b) Interleave (pinned): edit, edit, undo, NEW edit -> the redo tail truncates. -----------
    {
        ecs::Registry reg;
        BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        editor::EditHistory h;
        const editor::EditTargets t{&reg, nullptr};

        editor::TransformEdit a; a.setPosition = true; a.position = {4, 4, 4};   // command A
        editor::RecordedApplyTransformEdit(h, reg, 0, a);
        editor::TransformEdit b; b.setPosition = true; b.position = {5, 5, 5};   // command B
        editor::RecordedApplyTransformEdit(h, reg, 0, b);
        check(h.commands.size() == 2 && h.cursor == 2, "(b) two commands recorded");

        check(Undo(h, t), "(b) undo B");
        check(h.commands.size() == 2 && h.cursor == 1, "(b) redo tail present (cursor 1 of 2)");

        editor::TransformEdit c; c.setPosition = true; c.position = {6, 6, 6};   // NEW command C
        editor::RecordedApplyTransformEdit(h, reg, 0, c);
        check(h.commands.size() == 2 && h.cursor == 2,
              "(b) the NEW edit TRUNCATED the redo tail (history = [A, C])");
        check(!Redo(h, t), "(b) Redo after the truncating edit fails");
        // The surviving commands are A then C (B is gone) — pinned by their after-payloads.
        check(h.commands[0].xAfter.px == 4.0f && h.commands[1].xAfter.px == 6.0f,
              "(b) history holds A then C (B was truncated away)");
        // Undo x2 walks C then A back to the pristine scene.
        check(Undo(h, t) && Undo(h, t), "(b) undo C then A");
        check(reg.get<scene::TransformC>(editor::EntityAtViewIndex(reg, 0)).t.position.x == 1.0f,
              "(b) both undos restore the original cube position");
    }

    // --- (c) Serialize -> Deserialize -> digest equal; ReplayHistory reproduces the session. -------
    {
        // The LIVE session: scene edits + flow edits in one history.
        ecs::Registry reg;
        BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        flow::Graph graph = BuildFlowFixture();
        editor::EditHistory h;
        applyEdits(reg, h);
        const flow::NodeId nid = editor::RecordedAddFlowNode(h, graph, flow::kMin, 9);
        editor::RecordedConnectFlow(h, graph, /*from=*/1, /*to=*/nid, /*slot=*/1);
        const std::string liveDump = scene::DumpScene(reg, res);
        const uint64_t liveFlowDig = FlowDigest(graph);

        const std::vector<uint8_t> bytes = editor::SerializeHistory(h);
        editor::EditHistory rt;
        check(editor::DeserializeHistory(bytes.data(), bytes.size(), rt),
              "(c) DeserializeHistory succeeds");
        check(rt.commands.size() == h.commands.size() && rt.cursor == h.cursor,
              "(c) round-trip preserves count + cursor");
        check(editor::DigestHistory(rt) == editor::DigestHistory(h),
              "(c) round-trip DigestHistory identical");
        // Serialize(Deserialize(x)) is byte-identical to x.
        const std::vector<uint8_t> bytes2 = editor::SerializeHistory(rt);
        check(bytes2 == bytes, "(c) re-serialized artifact byte-identical");

        // REPLAY the deserialized session on a FRESH scene + FRESH graph.
        ecs::Registry fresh;
        BuildScene(fresh, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        flow::Graph freshGraph = BuildFlowFixture();
        const editor::EditTargets ft{&fresh, &freshGraph};
        check(editor::ReplayHistory(rt, ft), "(c) ReplayHistory applies every command");
        check(scene::DumpScene(fresh, res) == liveDump,
              "(c) replayed scene DumpScene BYTE-IDENTICAL to the live session's");
        check(FlowDigest(freshGraph) == liveFlowDig && FlowEq(freshGraph, graph),
              "(c) replayed flow graph identical to the live session's");

        // Malformed artifacts are rejected (never half-applied).
        editor::EditHistory junk;
        std::vector<uint8_t> bad = bytes;
        bad[0] ^= 0xFF;   // magic
        check(!editor::DeserializeHistory(bad.data(), bad.size(), junk),
              "(c) bad magic rejected");
        check(!editor::DeserializeHistory(bytes.data(), bytes.size() - 1, junk),
              "(c) truncated artifact rejected");
        std::vector<uint8_t> extra = bytes;
        extra.push_back(0);
        check(!editor::DeserializeHistory(extra.data(), extra.size(), junk),
              "(c) trailing bytes rejected");
    }

    // --- (c2) NAMED encoding: the artifact is POINTER-FREE / process-portable. ---------------------
    {
        // The same LOGICAL session recorded in two "processes" whose textures live at DIFFERENT
        // addresses (world B offsets every pointer) must serialize to BYTE-IDENTICAL named
        // artifacts — while the raw (pointer) encoding differs. This is the "replays anywhere"
        // property: names travel, pointers don't.
        auto record = [](ecs::Registry& rg, editor::EditHistory& hh, rhi::ITexture* red) {
            editor::TransformEdit t1; t1.setPosition = true; t1.position = {7, 8, 9};
            editor::RecordedApplyTransformEdit(hh, rg, 0, t1);
            editor::MaterialEdit m1; m1.setBaseColor = true; m1.baseColor = red;
            m1.setRoughness = true; m1.roughness = 0.25f;
            editor::RecordedApplyMaterialEdit(hh, rg, 1, m1);
        };
        // World A: the standard fixture pointers/resources.
        ecs::Registry regA;
        BuildScene(regA, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        editor::EditHistory hA;
        record(regA, hA, texRed);
        // World B: every pointer offset by 0x5000, registered under the SAME names.
        auto off = [](void* pv) { return (uintptr_t)pv + 0x5000u; };
        auto* meshCubeB   = reinterpret_cast<scene::Mesh*>(off(meshCube));
        auto* meshSphereB = reinterpret_cast<scene::Mesh*>(off(meshSphere));
        auto* meshDuckB   = reinterpret_cast<scene::Mesh*>(off(meshDuck));
        auto* texCheckerB = reinterpret_cast<rhi::ITexture*>(off(texChecker));
        auto* texFlatB    = reinterpret_cast<rhi::ITexture*>(off(texFlat));
        auto* texRedB     = reinterpret_cast<rhi::ITexture*>(off(texRed));
        scene::SceneResources resB;
        resB.AddMesh("cube", meshCubeB);
        resB.AddMesh("sphere", meshSphereB);
        resB.AddMesh("duck", meshDuckB);
        resB.AddTexture("checker", texCheckerB);
        resB.AddTexture("flat_normal", texFlatB);
        resB.AddTexture("red", texRedB);
        ecs::Registry regB;
        BuildScene(regB, meshCubeB, meshSphereB, meshDuckB, texCheckerB, texFlatB);
        editor::EditHistory hB;
        record(regB, hB, texRedB);

        const std::vector<uint8_t> namedA = editor::SerializeHistory(hA, &res);
        const std::vector<uint8_t> namedB = editor::SerializeHistory(hB, &resB);
        check(namedA == namedB,
              "(c2) NAMED artifacts BYTE-IDENTICAL across different pointer worlds");
        check(editor::DigestHistory(hA, &res) == editor::DigestHistory(hB, &resB),
              "(c2) named DigestHistory process-stable");
        check(editor::SerializeHistory(hA) != editor::SerializeHistory(hB),
              "(c2) raw (pointer) encoding differs across worlds (what named fixes)");

        // Deserialize world A's artifact INTO world B (re-binds "red" to B's pointer) and replay.
        editor::EditHistory rtB;
        check(editor::DeserializeHistory(namedA.data(), namedA.size(), rtB, &resB),
              "(c2) named artifact deserializes against another process's resources");
        ecs::Registry freshB;
        BuildScene(freshB, meshCubeB, meshSphereB, meshDuckB, texCheckerB, texFlatB);
        check(editor::ReplayHistory(rtB, editor::EditTargets{&freshB, nullptr}),
              "(c2) cross-world replay applies");
        check(freshB.get<scene::MaterialC>(
                  editor::EntityAtViewIndex(freshB, 1)).base == texRedB,
              "(c2) cross-world replay re-bound 'red' to world B's pointer");

        // A named artifact REQUIRES a resources table; an unknown name is rejected.
        editor::EditHistory junk;
        check(!editor::DeserializeHistory(namedA.data(), namedA.size(), junk),
              "(c2) named artifact without resources rejected");
        scene::SceneResources empty;
        check(!editor::DeserializeHistory(namedA.data(), namedA.size(), junk, &empty),
              "(c2) named artifact with an unknown texture name rejected");
    }

    // --- (d) The PINNED cross-compiler digest of the fixed hand-authored history. ------------------
    {
        const editor::EditHistory h = BuildPinnedHistory();
        const uint64_t dig = editor::DigestHistory(h);
        std::printf("edit_history_test: pinned-history digest 0x%016llx (expect 0x%016llx)\n",
                    (unsigned long long)dig, (unsigned long long)kPinnedHistoryDigest);
        check(dig == kPinnedHistoryDigest,
              "(d) DigestHistory(fixed history) == the pinned cross-compiler constant");
    }

    // --- (e) Flow enrollment: add/connect/delete recorded, undone, redone (digest chains). ---------
    {
        flow::Graph g = BuildFlowFixture();
        const flow::Graph base = g;
        editor::EditHistory h;
        editor::EditTargets t; t.flowGraph = &g;
        const uint64_t digBase = FlowDigest(g);

        // ADD: base -> add -> base -> add (the pinned digest chain).
        const flow::NodeId nid = editor::RecordedAddFlowNode(h, g, flow::kSub, 2);
        check(nid == 4u && g.nodes.size() == 5u, "(e) RecordedAddFlowNode appended node 4");
        const uint64_t digAdd = FlowDigest(g);
        check(digAdd != digBase, "(e) add moved the graph digest");
        check(Undo(h, t), "(e) undo the add");
        check(FlowDigest(g) == digBase && FlowEq(g, base),
              "(e) undo(add) -> the exact base graph (digest chain: add -> base)");
        check(Redo(h, t), "(e) redo the add");
        check(FlowDigest(g) == digAdd, "(e) redo(add) -> the add digest again (base -> add)");

        // CONNECT: rewire n4.a from the sentinel to n0; undo restores the sentinel exactly.
        check(editor::RecordedConnectFlow(h, g, /*from=*/0, /*to=*/nid, /*slot=*/0),
              "(e) RecordedConnectFlow succeeds");
        check(g.nodes[4].a == 0u, "(e) connect wired n0 -> n4.a");
        const uint64_t digConn = FlowDigest(g);
        check(Undo(h, t) && g.nodes[4].a == nid,
              "(e) undo(connect) restores the sentinel ref exactly");
        check(Redo(h, t) && FlowDigest(g) == digConn, "(e) redo(connect) re-wires exactly");

        // DELETE a node WITH inbound references (n2 <- n3.a): the cut list restores them on undo.
        const flow::Graph preDelete = g;
        check(editor::RecordedDeleteFlowNode(h, g, /*victim=*/2), "(e) RecordedDeleteFlowNode");
        check(g.nodes.size() == 4u, "(e) delete removed the node");
        const uint64_t digDel = FlowDigest(g);
        check(Undo(h, t), "(e) undo the delete");
        check(FlowEq(g, preDelete) && FlowDigest(g) == FlowDigest(preDelete),
              "(e) undo(delete) rebuilt the EXACT pre-delete graph (inbound refs restored)");
        check(Redo(h, t) && FlowDigest(g) == digDel, "(e) redo(delete) re-deletes exactly");

        // A no-op connect (same ref) records nothing.
        const std::size_t before = h.commands.size();
        check(editor::RecordedConnectFlow(h, g, g.nodes[0].a, 0, 0),
              "(e) no-op connect still succeeds");
        check(h.commands.size() == before, "(e) no-op connect recorded NO command");
    }

    // --- Safe no-op wrappers: out-of-range / component-less targets record nothing. ----------------
    {
        ecs::Registry reg;
        BuildScene(reg, meshCube, meshSphere, meshDuck, texChecker, texFlat);
        editor::EditHistory h;
        editor::TransformEdit e; e.setPosition = true; e.position = {9, 9, 9};
        editor::RecordedApplyTransformEdit(h, reg, 42, e);   // out of range
        editor::RecordedApplyTransformEdit(h, reg, -1, e);   // negative
        editor::MaterialEdit m; m.setMetallic = true; m.metallic = 0.9f;
        editor::RecordedApplyMaterialEdit(h, reg, 42, m);
        check(h.commands.empty() && h.cursor == 0, "safe no-op wrappers recorded nothing");
        // And a genuinely value-preserving edit records nothing either (bitwise no-op skip).
        editor::TransformEdit same; same.setPosition = true; same.position = {1, 2, 3};
        editor::RecordedApplyTransformEdit(h, reg, 0, same);   // cube is already at (1,2,3)
        check(h.commands.empty(), "bitwise no-op edit recorded nothing");
    }

    if (g_fail == 0) { std::printf("edit_history_test: ALL PASS\n"); return 0; }
    std::printf("edit_history_test: %d FAILURE(S)\n", g_fail);
    return 1;
}
