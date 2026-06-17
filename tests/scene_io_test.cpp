// Round-trip test for data-driven scenes (engine/scene/scene_io.{h,cpp}): build a registry by
// hand, DumpScene -> a temp JSON file -> LoadScene into a fresh registry, and confirm the entities
// + components survive byte-for-byte (mesh/texture names by reverse-map, transform, material).
//
// Resources are opaque pointers to the scene_io layer, so the test uses dummy non-null addresses
// as stand-in Mesh*/ITexture* — scene_io never dereferences them; it only maps them to/from names.
#include "scene/scene_io.h"
#include "ecs/ecs.h"

#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b) { return std::fabs(a - b) < 1e-5f; }
static bool approx3(const math::Vec3& a, const math::Vec3& b) {
    return approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z);
}

int main() {
    HF_TEST_MAIN_INIT();
    // Dummy resources: distinct non-null pointers used purely as identities. scene_io never
    // dereferences these — it maps pointer <-> name. We register a few named meshes/textures.
    auto* meshCube   = reinterpret_cast<scene::Mesh*>(0x1000);
    auto* meshSphere = reinterpret_cast<scene::Mesh*>(0x2000);
    auto* texChecker = reinterpret_cast<rhi::ITexture*>(0x3000);
    auto* texFlat    = reinterpret_cast<rhi::ITexture*>(0x4000);

    scene::SceneResources res;
    res.AddMesh("cube", meshCube);
    res.AddMesh("sphere", meshSphere);
    res.AddTexture("checker", texChecker);
    res.AddTexture("flat_normal", texFlat);

    // Build a source scene: two objects with distinct transforms + materials.
    ecs::Registry src;
    {
        ecs::Entity e0 = src.create();
        scene::Transform t0;
        t0.position = {1.0f, 2.0f, 3.0f};
        t0.eulerRadians = {0.1f, -0.5f, 0.0f};
        t0.scale = {0.5f, 0.5f, 0.5f};
        src.add<scene::TransformC>(e0, {t0});
        src.add<scene::MeshC>(e0, {meshCube});
        src.add<scene::MaterialC>(e0, {texChecker, texFlat, 0.0f, 0.5f});

        ecs::Entity e1 = src.create();
        scene::Transform t1;
        t1.position = {-1.8f, 0.55f, 1.8f};
        t1.scale = {0.55f, 0.55f, 0.55f};
        src.add<scene::TransformC>(e1, {t1});
        src.add<scene::MeshC>(e1, {meshSphere});
        // Null baseColor / normalMap -> serialized as JSON null, resolved back to nullptr.
        src.add<scene::MaterialC>(e1, {nullptr, nullptr, 1.0f, 0.15f});
    }

    // Dump -> temp file.
    std::string json = scene::DumpScene(src, res);
    check(!json.empty(), "DumpScene produced non-empty JSON");

    std::string path = std::string(std::tmpnam(nullptr)) + ".json";
    { std::ofstream f(path, std::ios::binary); f << json; }

    // Load into a fresh registry.
    ecs::Registry dst;
    std::vector<ecs::Entity> loaded = scene::LoadScene(dst, res, path.c_str());
    std::remove(path.c_str());

    check(loaded.size() == 2, "LoadScene created exactly 2 entities");

    if (loaded.size() == 2) {
        // Entity 0 (cube).
        auto& tc0 = dst.get<scene::TransformC>(loaded[0]);
        auto& mc0 = dst.get<scene::MeshC>(loaded[0]);
        auto& ma0 = dst.get<scene::MaterialC>(loaded[0]);
        check(mc0.mesh == meshCube, "obj0 mesh round-trips to cube");
        check(ma0.base == texChecker, "obj0 baseColor round-trips to checker");
        check(ma0.normal == texFlat, "obj0 normalMap round-trips to flat_normal");
        check(approx(ma0.metallic, 0.0f) && approx(ma0.roughness, 0.5f), "obj0 material factors");
        check(approx3(tc0.t.position, {1.0f, 2.0f, 3.0f}), "obj0 position");
        check(approx3(tc0.t.eulerRadians, {0.1f, -0.5f, 0.0f}), "obj0 euler");
        check(approx3(tc0.t.scale, {0.5f, 0.5f, 0.5f}), "obj0 scale");

        // Entity 1 (sphere, null textures).
        auto& tc1 = dst.get<scene::TransformC>(loaded[1]);
        auto& mc1 = dst.get<scene::MeshC>(loaded[1]);
        auto& ma1 = dst.get<scene::MaterialC>(loaded[1]);
        check(mc1.mesh == meshSphere, "obj1 mesh round-trips to sphere");
        check(ma1.base == nullptr, "obj1 null baseColor stays null");
        check(ma1.normal == nullptr, "obj1 null normalMap stays null");
        check(approx(ma1.metallic, 1.0f) && approx(ma1.roughness, 0.15f), "obj1 material factors");
        check(approx3(tc1.t.position, {-1.8f, 0.55f, 1.8f}), "obj1 position");
        check(approx3(tc1.t.scale, {0.55f, 0.55f, 0.55f}), "obj1 scale");

        // A second dump of the loaded scene must equal the first dump (stable round-trip).
        std::string json2 = scene::DumpScene(dst, res);
        check(json2 == json, "DumpScene(LoadScene(DumpScene)) is stable");
    }

    if (g_fail == 0) { std::printf("scene_io_test OK\n"); return 0; }
    std::printf("scene_io_test: %d failures\n", g_fail);
    return 1;
}
