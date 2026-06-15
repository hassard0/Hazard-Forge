// Unit test for the machine-readable engine-state introspection (engine/editor/introspect.{h,cpp},
// Slice AL). Build a known Registry (2 entities with transforms + materials) + a known EngineState
// (camera + directional + 2 point lights), call DescribeEngine, and assert:
//   * the JSON parses (round-trips through the vendored json.h),
//   * the scene entity count + a known entity's transform values are present,
//   * the camera fields + light values are present,
//   * the engine.backends array is exactly ["vulkan","metal"],
//   * the commands manifest includes "set_transform" (and the new "introspect"),
//   * two calls on identical inputs are BYTE-IDENTICAL (determinism).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests. Resources are opaque pointers (mapped
// to names), so the test uses distinct non-null stand-in Mesh*/ITexture* addresses — never dereferenced.
#include "editor/introspect.h"
#include "scene/components.h"
#include "scene/scene_io.h"
#include "ecs/ecs.h"

#include "json/json.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// --- Minimal typed accessors over json.h (mirrors scene_io's helpers) ---------------------------
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
static bool approx(double a, double b) { return std::fabs(a - b) < 1e-4; }

int main() {
    // --- Build a known scene ---------------------------------------------------------------------
    auto* meshCube   = reinterpret_cast<scene::Mesh*>(0x1000);
    auto* meshSphere = reinterpret_cast<scene::Mesh*>(0x2000);
    auto* texChecker = reinterpret_cast<rhi::ITexture*>(0x3000);
    auto* texFlat    = reinterpret_cast<rhi::ITexture*>(0x4000);

    scene::SceneResources res;
    res.AddMesh("cube", meshCube);
    res.AddMesh("sphere", meshSphere);
    res.AddTexture("checker", texChecker);
    res.AddTexture("flat_normal", texFlat);

    ecs::Registry reg;
    {  // entity 0: cube at (1,2,3), scale 2, dielectric.
        ecs::Entity e = reg.create();
        scene::Transform t;
        t.position = {1, 2, 3};
        t.eulerRadians = {0, 0.5f, 0};
        t.scale = {2, 2, 2};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshCube});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.0f, 0.5f});
    }
    {  // entity 1: sphere, metallic, no normal map.
        ecs::Entity e = reg.create();
        scene::Transform t;
        t.position = {-1, 0, 0};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshSphere});
        reg.add<scene::MaterialC>(e, {texChecker, nullptr, 1.0f, 0.15f});
    }

    // --- Known non-ECS state ---------------------------------------------------------------------
    editor::EngineState state;
    state.backend = "vulkan";
    state.hasCamera = true;
    state.camera.position = {4, 5, 6};
    state.camera.yaw = 0.25f;
    state.camera.pitch = -0.3f;
    state.camera.fovDeg = 60.0f;
    state.hasDirectional = true;
    state.directional.dir = {-0.5f, -1.0f, -0.3f};
    state.directional.color = {0.95f, 0.93f, 0.85f};
    state.points.push_back({{3, 1, 0}, {1, 0.25f, 0.2f}, 3.2f, 1.0f});
    state.points.push_back({{-3, 1, 0}, {0.2f, 1.0f, 0.35f}, 3.2f, 1.0f});

    std::string json = editor::DescribeEngine(reg, res, state);

    // --- Determinism: a second identical call must be byte-identical. ----------------------------
    std::string json2 = editor::DescribeEngine(reg, res, state);
    check(json == json2, "DescribeEngine is deterministic (byte-identical across runs)");

    // --- Parse the JSON (must be well-formed). ---------------------------------------------------
    json_parse_result_s err{};
    json_value_s* root = json_parse_ex(json.data(), json.size(), json_parse_flags_default,
                                       nullptr, nullptr, &err);
    check(root != nullptr, "DescribeEngine output parses as JSON");
    const json_object_s* top = AsObject(root);
    check(top != nullptr, "top-level value is an object");

    if (top) {
        // engine.name + backends == ["vulkan","metal"].
        const json_object_s* engine = AsObject(MemberOf(top, "engine"));
        check(engine != nullptr, "has engine object");
        if (engine) {
            check(AsString(MemberOf(engine, "name")) == "Hazard Forge", "engine.name is Hazard Forge");
            check(AsString(MemberOf(engine, "activeBackend")) == "vulkan", "engine.activeBackend is vulkan");
            const json_array_s* backends = AsArray(MemberOf(engine, "backends"));
            check(backends != nullptr && backends->length == 2, "engine.backends has 2 entries");
            if (backends && backends->length == 2) {
                const json_array_element_s* b0 = backends->start;
                const json_array_element_s* b1 = b0->next;
                check(AsString(b0->value) == "vulkan", "backends[0] == vulkan");
                check(AsString(b1->value) == "metal", "backends[1] == metal");
            }
            const json_array_s* features = AsArray(MemberOf(engine, "features"));
            check(features != nullptr && features->length > 10, "engine.features is a non-trivial list");
            // Slice AP: the temporal-anti-aliasing capability is advertised in the feature manifest.
            bool sawTaaFeature = false;
            if (features)
                for (const json_array_element_s* el = features->start; el; el = el->next)
                    if (AsString(el->value) == "temporal-anti-aliasing") sawTaaFeature = true;
            check(sawTaaFeature, "engine.features includes temporal-anti-aliasing");
        }

        // commands manifest includes set_transform + introspect.
        const json_array_s* commands = AsArray(MemberOf(top, "commands"));
        check(commands != nullptr, "has commands array");
        bool sawSetTransform = false, sawIntrospect = false;
        if (commands) {
            for (const json_array_element_s* el = commands->start; el; el = el->next) {
                const json_object_s* c = AsObject(el->value);
                if (!c) continue;
                std::string op = AsString(MemberOf(c, "op"));
                if (op == "set_transform") sawSetTransform = true;
                if (op == "introspect")    sawIntrospect = true;
            }
        }
        check(sawSetTransform, "commands manifest includes set_transform");
        check(sawIntrospect, "commands manifest includes introspect");

        // showcases present + non-trivial.
        const json_array_s* showcases = AsArray(MemberOf(top, "showcases"));
        check(showcases != nullptr && showcases->length > 10, "showcases is a non-trivial list");
        // Slice AP: the --taa-shot showcase flag is listed in the showcase manifest.
        bool sawTaaShot = false;
        if (showcases)
            for (const json_array_element_s* el = showcases->start; el; el = el->next) {
                const json_object_s* s = AsObject(el->value);
                if (s && AsString(MemberOf(s, "flag")) == "--taa-shot") sawTaaShot = true;
            }
        check(sawTaaShot, "showcases manifest includes --taa-shot");

        // scene.entities: count == 2 + entity 0's transform values.
        const json_object_s* scene = AsObject(MemberOf(top, "scene"));
        const json_array_s* ents = scene ? AsArray(MemberOf(scene, "entities")) : nullptr;
        check(ents != nullptr && ents->length == 2, "scene has exactly 2 entities");
        if (ents && ents->length == 2) {
            const json_object_s* e0 = AsObject(ents->start->value);
            const json_object_s* comps = e0 ? AsObject(MemberOf(e0, "components")) : nullptr;
            const json_object_s* xform = comps ? AsObject(MemberOf(comps, "transform")) : nullptr;
            const json_array_s* pos = xform ? AsArray(MemberOf(xform, "position")) : nullptr;
            check(pos != nullptr && pos->length == 3, "entity 0 has a 3-vec position");
            if (pos && pos->length == 3) {
                const json_array_element_s* px = pos->start;
                const json_array_element_s* py = px->next;
                const json_array_element_s* pz = py->next;
                check(approx(AsNumber(px->value), 1) && approx(AsNumber(py->value), 2) &&
                      approx(AsNumber(pz->value), 3), "entity 0 position == [1,2,3]");
            }
            check(AsString(MemberOf(comps, "mesh")) == "cube", "entity 0 mesh name == cube");
            const json_object_s* mat = comps ? AsObject(MemberOf(comps, "material")) : nullptr;
            check(mat && approx(AsNumber(MemberOf(mat, "metallic")), 0.0), "entity 0 metallic == 0");
            check(mat && AsString(MemberOf(mat, "baseColor")) == "checker", "entity 0 baseColor == checker");
        }

        // camera fields.
        const json_object_s* cam = AsObject(MemberOf(top, "camera"));
        check(cam != nullptr, "has camera object");
        if (cam) {
            check(approx(AsNumber(MemberOf(cam, "fovDeg")), 60.0), "camera.fovDeg == 60");
            check(approx(AsNumber(MemberOf(cam, "yaw")), 0.25), "camera.yaw == 0.25");
            const json_array_s* cpos = AsArray(MemberOf(cam, "position"));
            check(cpos != nullptr && cpos->length == 3, "camera.position is a 3-vec");
        }

        // lights: directional present + 2 points.
        const json_object_s* lights = AsObject(MemberOf(top, "lights"));
        check(lights != nullptr, "has lights object");
        if (lights) {
            const json_object_s* dir = AsObject(MemberOf(lights, "directional"));
            check(dir != nullptr, "lights.directional present");
            const json_array_s* pts = AsArray(MemberOf(lights, "points"));
            check(pts != nullptr && pts->length == 2, "lights.points has 2 entries");
            if (pts && pts->length == 2) {
                const json_object_s* p0 = AsObject(pts->start->value);
                check(p0 && approx(AsNumber(MemberOf(p0, "radius")), 3.2), "point 0 radius == 3.2");
            }
        }

        // stats.
        const json_object_s* stats = AsObject(MemberOf(top, "stats"));
        check(stats != nullptr, "has stats object");
        if (stats) {
            check((int)AsNumber(MemberOf(stats, "entityCount")) == 2, "stats.entityCount == 2");
            check((int)AsNumber(MemberOf(stats, "pointLightCount")) == 2, "stats.pointLightCount == 2");
        }
    }

    if (root) std::free(root);

    if (g_fail == 0) { std::printf("introspect_test OK\n"); return 0; }
    std::printf("introspect_test: %d failures\n", g_fail);
    return 1;
}
