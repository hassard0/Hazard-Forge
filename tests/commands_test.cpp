// Unit test for the agentic command interface (engine/scene/commands.{h,cpp}): apply a JSON command
// script to a hand-built registry and assert the live ECS reflects every mutation. No rendering: the
// "capture" op is driven through a stub callback that just records the requested path, so this runs
// headless under ctest on any backend.
//
// Entities are addressed by view-order index (creation order), the same indexing DumpScene emits.
// Resources are opaque pointers to the command layer (it maps pointer <-> name), so the test uses
// distinct non-null stand-in Mesh*/ITexture* addresses — never dereferenced.
#include "scene/commands.h"
#include "scene/components.h"
#include "ecs/ecs.h"

#include <cmath>
#include <cstdio>
#include <string>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b) { return std::fabs(a - b) < 1e-5f; }

// Resolve the entity at a view-order index (mirrors how commands address entities).
static ecs::Entity At(ecs::Registry& reg, int index) {
    int i = 0;
    for (auto [e, tc, mc, mat] : reg.view<scene::TransformC, scene::MeshC, scene::MaterialC>()) {
        (void)tc; (void)mc; (void)mat;
        if (i == index) return e;
        ++i;
    }
    return ecs::kNullEntity;
}
static int Count(ecs::Registry& reg) {
    int i = 0;
    for (auto [e, tc, mc, mat] : reg.view<scene::TransformC, scene::MeshC, scene::MaterialC>()) {
        (void)e; (void)tc; (void)mc; (void)mat; ++i;
    }
    return i;
}

int main() {
    HF_TEST_MAIN_INIT();
    auto* meshCube   = reinterpret_cast<scene::Mesh*>(0x1000);
    auto* meshSphere = reinterpret_cast<scene::Mesh*>(0x2000);
    auto* texChecker = reinterpret_cast<rhi::ITexture*>(0x3000);
    auto* texFlat    = reinterpret_cast<rhi::ITexture*>(0x4000);

    scene::SceneResources res;
    res.AddMesh("cube", meshCube);
    res.AddMesh("sphere", meshSphere);
    res.AddTexture("checker", texChecker);
    res.AddTexture("flat_normal", texFlat);

    // Seed three entities (indices 0,1,2): cube, cube, sphere.
    ecs::Registry reg;
    for (int k = 0; k < 2; ++k) {
        ecs::Entity e = reg.create();
        reg.add<scene::TransformC>(e, {scene::Transform{}});
        reg.add<scene::MeshC>(e, {meshCube});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.0f, 0.5f});
    }
    {
        ecs::Entity e = reg.create();
        reg.add<scene::TransformC>(e, {scene::Transform{}});
        reg.add<scene::MeshC>(e, {meshSphere});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 1.0f, 0.15f});
    }
    check(Count(reg) == 3, "seeded 3 entities");

    // A capture stub: record the path so we can assert the op routed to the callback.
    std::string capturedPath;
    scene::CaptureFn capture = [&](const char* path) { capturedPath = path; return true; };

    // Command script: move entity 1, make entity 0 metallic + clear its normal map, add a sphere,
    // remove entity 2, capture. dump/list exercise the read paths too.
    const char* script = R"([
        { "op": "dump" },
        { "op": "list" },
        { "op": "set_transform", "entity": 1, "position": [5, 6, 7], "scale": [2, 2, 2] },
        { "op": "set_material", "entity": 0, "metallic": 1.0, "roughness": 0.1, "normalMap": null },
        { "op": "add", "mesh": "sphere", "baseColor": "checker", "metallic": 1.0, "position": [1, 1, 1] },
        { "op": "remove", "entity": 2 },
        { "op": "capture", "path": "unit_capture.bmp" }
    ])";

    bool ok = scene::RunCommandsFromText(reg, res, script, capture);
    check(ok, "RunCommandsFromText reports success");

    // After: started 3, added 1, removed 1 -> 3 entities remain.
    check(Count(reg) == 3, "entity count is 3 after add + remove");

    // entity 1: moved + scaled.
    {
        ecs::Entity e = At(reg, 1);
        check(e != ecs::kNullEntity, "entity 1 exists");
        auto& tc = reg.get<scene::TransformC>(e);
        check(approx(tc.t.position.x, 5) && approx(tc.t.position.y, 6) && approx(tc.t.position.z, 7),
              "set_transform moved entity 1");
        check(approx(tc.t.scale.x, 2) && approx(tc.t.scale.y, 2) && approx(tc.t.scale.z, 2),
              "set_transform scaled entity 1");
    }

    // entity 0: metallic set, normal map cleared to null.
    {
        ecs::Entity e = At(reg, 0);
        auto& mat = reg.get<scene::MaterialC>(e);
        check(approx(mat.metallic, 1.0f) && approx(mat.roughness, 0.1f), "set_material factors on entity 0");
        check(mat.normal == nullptr, "set_material cleared normalMap to null");
        check(mat.base == texChecker, "set_material left baseColor untouched");
    }

    // The "remove" destroyed the original sphere (index 2). The "add" appended a new sphere; since
    // the removed slot's index now shifts, the last entity (index 2) is the added sphere.
    {
        ecs::Entity e = At(reg, 2);
        check(e != ecs::kNullEntity, "entity 2 (the added sphere) exists");
        auto& mc = reg.get<scene::MeshC>(e);
        auto& mat = reg.get<scene::MaterialC>(e);
        check(mc.mesh == meshSphere, "added entity is a sphere");
        check(approx(mat.metallic, 1.0f), "added entity is metallic");
        check(mat.base == texChecker, "added entity baseColor resolved by name");
    }

    // The capture op routed to the callback with the requested path.
    check(capturedPath == "unit_capture.bmp", "capture op invoked the callback with the path");

    // An out-of-range entity index makes a command fail (and the run report false) without throwing.
    {
        ecs::Registry r2;
        ecs::Entity e = r2.create();
        r2.add<scene::TransformC>(e, {scene::Transform{}});
        r2.add<scene::MeshC>(e, {meshCube});
        r2.add<scene::MaterialC>(e, {texChecker, texFlat, 0.0f, 0.5f});
        bool bad = scene::RunCommandsFromText(
            r2, res, R"([{ "op": "set_transform", "entity": 99, "position": [0,0,0] }])", nullptr);
        check(!bad, "out-of-range entity index reported as failure");
    }

    if (g_fail == 0) { std::printf("commands_test OK\n"); return 0; }
    std::printf("commands_test: %d failures\n", g_fail);
    return 1;
}
