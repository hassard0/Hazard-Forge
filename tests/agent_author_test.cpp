// DX3 — the declarative scene-spec authoring loop (engine/scene/scene_io.{h,cpp}):
// CanonicalizeScene = LoadScene(spec) then DumpScene. This pure-CPU test pins the four DX3
// properties without any file/GPU:
//   1. round-trip: CanonicalizeSceneText of a small non-canonical spec == the expected canonical text.
//   2. idempotence / fixed point: Canonicalize(Canonicalize(spec)) == Canonicalize(spec).
//   3. determinism: two canonicalizations of the same spec are byte-identical.
//   4. reject: an unknown mesh name throws std::runtime_error (the LoadScene contract).
//
// Resources are opaque pointers to scene_io, so the test registers dummy non-null sentinels under
// names like "cube"/"duck" — scene_io never dereferences them; it maps pointer <-> name only.
#include "scene/scene_io.h"
#include "ecs/ecs.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Build a SceneResources with a few fake-but-named resource pointers (the opaque-pointer contract).
static scene::SceneResources MakeResources() {
    scene::SceneResources res;
    res.AddMesh("cube",   reinterpret_cast<scene::Mesh*>(0x1001));
    res.AddMesh("sphere", reinterpret_cast<scene::Mesh*>(0x1002));
    res.AddMesh("duck",   reinterpret_cast<scene::Mesh*>(0x1003));
    res.AddTexture("checker",     reinterpret_cast<rhi::ITexture*>(0x2001));
    res.AddTexture("flat_normal", reinterpret_cast<rhi::ITexture*>(0x2002));
    return res;
}

int main() {
    HF_TEST_MAIN_INIT();
    scene::SceneResources res = MakeResources();

    // A small NON-canonical spec: keys out of order, optional fields omitted (no euler on either, no
    // normalMap/metallic on the cube; the sphere has a flat_normal + metallic). Two objects.
    const char* spec =
        "[\n"
        "  { \"position\": [1, 2, 3], \"mesh\": \"cube\", \"baseColor\": \"checker\", "
        "\"scale\": [0.5, 0.5, 0.5] },\n"
        "  { \"mesh\": \"sphere\", \"metallic\": 1, \"roughness\": 0.2, "
        "\"normalMap\": \"flat_normal\" }\n"
        "]\n";

    // The expected CANONICAL text — the exact DumpScene byte layout (fixed key order, %g floats,
    // 2-space indent). Omitted fields take their defaults (metallic 0, roughness 0.5, pos/euler 0,
    // scale 1; absent baseColor/normalMap -> null).
    const std::string expected =
        "[\n"
        "  {\n"
        "    \"mesh\": \"cube\",\n"
        "    \"baseColor\": \"checker\",\n"
        "    \"normalMap\": null,\n"
        "    \"metallic\": 0,\n"
        "    \"roughness\": 0.5,\n"
        "    \"position\": [1, 2, 3],\n"
        "    \"euler\": [0, 0, 0],\n"
        "    \"scale\": [0.5, 0.5, 0.5]\n"
        "  },\n"
        "  {\n"
        "    \"mesh\": \"sphere\",\n"
        "    \"baseColor\": null,\n"
        "    \"normalMap\": \"flat_normal\",\n"
        "    \"metallic\": 1,\n"
        "    \"roughness\": 0.2,\n"
        "    \"position\": [0, 0, 0],\n"
        "    \"euler\": [0, 0, 0],\n"
        "    \"scale\": [1, 1, 1]\n"
        "  }\n"
        "]\n";

    // 1. round-trip: canonicalize the non-canonical spec -> the expected canonical text.
    ecs::Registry r1;
    std::string canon = scene::CanonicalizeSceneText(spec, r1, res);
    check(canon == expected, "CanonicalizeSceneText(spec) == expected canonical text");
    if (canon != expected) {
        std::printf("--- got ---\n%s--- want ---\n%s", canon.c_str(), expected.c_str());
    }

    // 2. idempotence / fixed point: feeding the canonical text back yields byte-identical bytes.
    ecs::Registry r2;
    std::string twice = scene::CanonicalizeSceneText(canon.c_str(), r2, res);
    check(twice == canon, "Canonicalize is a fixed point (twice == once)");

    // 3. determinism: two canonicalizations of the SAME spec are byte-identical.
    ecs::Registry r3;
    std::string again = scene::CanonicalizeSceneText(spec, r3, res);
    check(again == canon, "two canonicalizations of the same spec are byte-identical");

    // 4. reject: an unknown mesh name throws std::runtime_error (the LoadScene contract).
    {
        bool threw = false;
        try {
            ecs::Registry rb;
            (void)scene::CanonicalizeSceneText("[{\"mesh\": \"nope\"}]", rb, res);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "unknown mesh name throws std::runtime_error");
    }

    if (g_fail == 0) std::printf("agent_author_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
