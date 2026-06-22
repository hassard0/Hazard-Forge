// Unit test for the DX2 scene-query read protocol (engine/scene/commands.{h,cpp}: scene::RunQueries).
// Builds a hand-made registry + resources (opaque stand-in pointers, never dereferenced) and asserts
// the deterministic, index-addressed, field-selectable JSON read. Pure CPU, ASan-clean, headless on
// any backend (no rendering — RunQueries is a string read of the live ECS).
#include "scene/commands.h"
#include "scene/components.h"
#include "ecs/ecs.h"

#include <cstdio>
#include <string>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
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

    // Seed 3 entities (indices 0,1,2): cube, cube, sphere. Entity 1 gets a distinctive transform +
    // material so the field-narrowing assertions are unambiguous.
    ecs::Registry reg;
    for (int k = 0; k < 2; ++k) {
        ecs::Entity e = reg.create();
        reg.add<scene::TransformC>(e, {scene::Transform{}});
        reg.add<scene::MeshC>(e, {meshCube});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.0f, 0.5f});
    }
    {
        ecs::Entity e = reg.create();
        scene::Transform t;
        t.position = {1.0f, 2.0f, 3.0f};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshSphere});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 1.0f, 0.25f});
    }

    // --- Determinism: two RunQueries runs over the same script are byte-identical. ---
    {
        const char* script = R"([
            { "op": "query", "select": "entity", "entity": 2, "fields": ["material", "transform"] },
            { "op": "query", "select": "stats" },
            { "op": "query", "select": "entities" }
        ])";
        std::string a = scene::RunQueries(reg, res, script);
        std::string b = scene::RunQueries(reg, res, script);
        check(a == b, "RunQueries two runs byte-identical");
    }

    // --- select:entity with fields == the narrowed object (only the requested components). ---
    {
        std::string r = scene::RunQueries(
            reg, res, R"([{ "op": "query", "select": "entity", "entity": 2, "fields": ["material"] }])");
        check(contains(r, "\"material\""), "fields:[material] emits material");
        check(!contains(r, "\"transform\""), "fields:[material] omits transform");
        check(!contains(r, "\"mesh\":"), "fields:[material] omits mesh");
        check(contains(r, "\"roughness\": 0.25"), "material roughness present");
    }

    // --- fields omitted == ALL components (transform, material, mesh). ---
    {
        std::string r = scene::RunQueries(
            reg, res, R"([{ "op": "query", "select": "entity", "entity": 2 }])");
        check(contains(r, "\"transform\""), "no-fields emits transform");
        check(contains(r, "\"material\""), "no-fields emits material");
        check(contains(r, "\"mesh\": \"sphere\""), "no-fields emits mesh (sphere)");
        check(contains(r, "\"position\": [1, 2, 3]"), "transform position emitted");
    }

    // --- Field-order canonicalization: reversed fields request -> byte-identical to canonical order. ---
    {
        std::string canon = scene::RunQueries(
            reg, res,
            R"([{ "op": "query", "select": "entity", "entity": 2, "fields": ["transform", "material", "mesh"] }])");
        std::string reversed = scene::RunQueries(
            reg, res,
            R"([{ "op": "query", "select": "entity", "entity": 2, "fields": ["mesh", "material", "transform"] }])");
        // Responses differ only by the verbatim "request" echo; the "response" sub-object must be
        // emitted in the SAME canonical order. Assert transform precedes material precedes mesh.
        size_t rt = reversed.find("\"transform\":", reversed.find("\"response\":"));
        size_t rm = reversed.find("\"material\":", reversed.find("\"response\":"));
        size_t rh = reversed.find("\"mesh\":", reversed.find("\"response\":"));
        check(rt != std::string::npos && rm != std::string::npos && rh != std::string::npos,
              "reversed-fields response has all three components");
        check(rt < rm && rm < rh, "reversed-fields response is in canonical transform<material<mesh order");
        (void)canon;
    }

    // --- select:stats counts == the scene's actual counts. ---
    {
        std::string r = scene::RunQueries(reg, res, R"([{ "op": "query", "select": "stats" }])");
        check(contains(r, "\"entities\": 3"), "stats entities == 3");
        check(contains(r, "\"meshes\": 2"), "stats meshes == 2");
        check(contains(r, "\"textures\": 2"), "stats textures == 2");
    }

    // --- select:entities == terse ordered [{index, mesh}] list. ---
    {
        std::string r = scene::RunQueries(reg, res, R"([{ "op": "query", "select": "entities" }])");
        check(contains(r, "\"index\": 0, \"mesh\": \"cube\""), "entities[0] cube");
        check(contains(r, "\"index\": 2, \"mesh\": \"sphere\""), "entities[2] sphere");
    }

    // --- out-of-range -> deterministic error object, NO throw. ---
    {
        bool threw = false;
        std::string r;
        try {
            r = scene::RunQueries(
                reg, res, R"([{ "op": "query", "select": "entity", "entity": 99 }])");
        } catch (...) { threw = true; }
        check(!threw, "out-of-range does not throw");
        check(contains(r, "\"error\": \"out-of-range\""), "out-of-range error code present");
        check(contains(r, "\"entity\": 99"), "out-of-range echoes the bad index");
        check(contains(r, "\"count\": 3"), "out-of-range reports the live entity count");
    }

    // --- negative index -> the same deterministic out-of-range error (no crash). ---
    {
        std::string r = scene::RunQueries(
            reg, res, R"([{ "op": "query", "select": "entity", "entity": -5 }])");
        check(contains(r, "\"error\": \"out-of-range\""), "negative index -> out-of-range error");
    }

    // --- unknown field name -> listed once under "unknownFields", recognized fields still emit. ---
    {
        std::string r = scene::RunQueries(
            reg, res,
            R"([{ "op": "query", "select": "entity", "entity": 2, "fields": ["mesh", "bogus"] }])");
        check(contains(r, "\"mesh\": \"sphere\""), "unknown-field req still emits the known field");
        check(contains(r, "\"unknownFields\": [\"bogus\"]"), "unknown field listed under unknownFields");
    }

    if (g_fail == 0) { std::printf("agent_query_test OK\n"); return 0; }
    std::printf("agent_query_test: %d failures\n", g_fail);
    return 1;
}
