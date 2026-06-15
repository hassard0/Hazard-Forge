// Hazard Forge — agentic command interface (see commands.h).
//
// A command processor over the live ECS scene. Reuses the vendored json.h parser (same as
// scene_io) and addresses entities by view-order index. No vk*/Metal/rhi rendering symbols here:
// the one rendering verb ("capture") is delegated to a caller-supplied std::function.
#include "scene/commands.h"

#include "scene/components.h"
#include "editor/introspect.h"
#include "json/json.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hf::scene {
namespace {

std::string ReadFile(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("RunCommands: cannot open command file: ") + path);
    std::streamsize size = f.tellg();
    f.seekg(0);
    std::string s(static_cast<size_t>(size), '\0');
    f.read(s.data(), size);
    return s;
}

// --- Typed accessors over the json.h DOM (mirrors scene_io's helpers) ----------------------------

const json_value_s* MemberOf(const json_object_s* obj, const char* key) {
    for (const json_object_element_s* el = obj->start; el; el = el->next)
        if (el->name && std::strcmp(el->name->string, key) == 0) return el->value;
    return nullptr;
}

bool HasMember(const json_object_s* obj, const char* key) { return MemberOf(obj, key) != nullptr; }

std::string StringMember(const json_object_s* obj, const char* key) {
    const json_value_s* v = MemberOf(obj, key);
    if (!v || v->type != json_type_string) return {};
    const json_string_s* s = static_cast<const json_string_s*>(v->payload);
    return std::string(s->string, s->string_size);
}

float FloatMember(const json_object_s* obj, const char* key, float fallback) {
    const json_value_s* v = MemberOf(obj, key);
    if (!v || v->type != json_type_number) return fallback;
    const json_number_s* n = static_cast<const json_number_s*>(v->payload);
    return std::strtof(n->number, nullptr);
}

int IntMember(const json_object_s* obj, const char* key, int fallback) {
    const json_value_s* v = MemberOf(obj, key);
    if (!v || v->type != json_type_number) return fallback;
    const json_number_s* n = static_cast<const json_number_s*>(v->payload);
    return static_cast<int>(std::strtol(n->number, nullptr, 10));
}

// Read [x,y,z] into `out` (leaving it unchanged if absent / malformed).
void Vec3Member(const json_object_s* obj, const char* key, math::Vec3& out) {
    const json_value_s* v = MemberOf(obj, key);
    if (!v || v->type != json_type_array) return;
    const json_array_s* arr = static_cast<const json_array_s*>(v->payload);
    float comp[3];
    int i = 0;
    for (const json_array_element_s* el = arr->start; el && i < 3; el = el->next, ++i) {
        if (el->value->type != json_type_number) return;
        comp[i] = std::strtof(static_cast<const json_number_s*>(el->value->payload)->number, nullptr);
    }
    if (i == 3) out = {comp[0], comp[1], comp[2]};
}

// Resolve a texture member with subset/clear semantics. The member may be:
//   absent          -> leave `slot` unchanged, return true.
//   JSON null / ""  -> clear `slot` to nullptr, return true.
//   "name"          -> set `slot` to the named texture, or return false if the name is unknown.
bool ApplyTextureMember(const json_object_s* obj, const char* key, const SceneResources& res,
                        rhi::ITexture*& slot) {
    const json_value_s* v = MemberOf(obj, key);
    if (!v) return true;  // absent: untouched
    if (v->type == json_type_null) { slot = nullptr; return true; }
    if (v->type != json_type_string) {
        std::fprintf(stderr, "command: '%s' must be a texture name or null\n", key);
        return false;
    }
    const json_string_s* s = static_cast<const json_string_s*>(v->payload);
    std::string name(s->string, s->string_size);
    if (name.empty()) { slot = nullptr; return true; }
    rhi::ITexture* tex = res.FindTexture(name);
    if (!tex) {
        std::fprintf(stderr, "command: unknown texture '%s'\n", name.c_str());
        return false;
    }
    slot = tex;
    return true;
}

// Map a view-order index to its live Entity handle. Iterates view<Transform,Mesh,Material>() (the
// same order DumpScene / the editor use). Returns false (and a stderr note) if out of range.
bool EntityAtIndex(ecs::Registry& reg, int index, ecs::Entity& out) {
    if (index < 0) { std::fprintf(stderr, "command: negative entity index %d\n", index); return false; }
    int i = 0;
    for (auto [e, tc, mc, mat] : reg.view<TransformC, MeshC, MaterialC>()) {
        (void)tc; (void)mc; (void)mat;
        if (i == index) { out = e; return true; }
        ++i;
    }
    std::fprintf(stderr, "command: entity index %d out of range (%d entities)\n", index, i);
    return false;
}

// --- Individual ops ------------------------------------------------------------------------------

bool OpDump(ecs::Registry& reg, SceneResources& res) {
    std::string json = DumpScene(reg, res);
    std::fputs(json.c_str(), stdout);
    return true;
}

bool OpList(ecs::Registry& reg, SceneResources& res) {
    int i = 0;
    for (auto [e, tc, mc, mat] : reg.view<TransformC, MeshC, MaterialC>()) {
        (void)e;
        std::string mesh = mc.mesh ? res.NameOfMesh(mc.mesh) : std::string("(none)");
        std::printf("%3d: %-8s pos=[%g, %g, %g] metallic=%g roughness=%g\n", i, mesh.c_str(),
                    tc.t.position.x, tc.t.position.y, tc.t.position.z, mat.metallic, mat.roughness);
        ++i;
    }
    std::printf("(%d entities)\n", i);
    return true;
}

bool OpSetTransform(ecs::Registry& reg, const json_object_s* obj) {
    ecs::Entity e;
    if (!EntityAtIndex(reg, IntMember(obj, "entity", -1), e)) return false;
    auto& tc = reg.get<TransformC>(e);
    Vec3Member(obj, "position", tc.t.position);
    Vec3Member(obj, "euler", tc.t.eulerRadians);
    Vec3Member(obj, "scale", tc.t.scale);
    std::printf("set_transform: entity %d -> pos=[%g, %g, %g] euler=[%g, %g, %g] scale=[%g, %g, %g]\n",
                IntMember(obj, "entity", -1), tc.t.position.x, tc.t.position.y, tc.t.position.z,
                tc.t.eulerRadians.x, tc.t.eulerRadians.y, tc.t.eulerRadians.z,
                tc.t.scale.x, tc.t.scale.y, tc.t.scale.z);
    return true;
}

bool OpSetMaterial(ecs::Registry& reg, SceneResources& res, const json_object_s* obj) {
    ecs::Entity e;
    if (!EntityAtIndex(reg, IntMember(obj, "entity", -1), e)) return false;
    auto& mat = reg.get<MaterialC>(e);
    if (HasMember(obj, "metallic")) mat.metallic = FloatMember(obj, "metallic", mat.metallic);
    if (HasMember(obj, "roughness")) mat.roughness = FloatMember(obj, "roughness", mat.roughness);
    bool ok = ApplyTextureMember(obj, "baseColor", res, mat.base);
    ok = ApplyTextureMember(obj, "normalMap", res, mat.normal) && ok;
    std::printf("set_material: entity %d -> metallic=%g roughness=%g baseColor=%s normalMap=%s\n",
                IntMember(obj, "entity", -1), mat.metallic, mat.roughness,
                mat.base ? res.NameOfTexture(mat.base).c_str() : "null",
                mat.normal ? res.NameOfTexture(mat.normal).c_str() : "null");
    return ok;
}

bool OpAdd(ecs::Registry& reg, SceneResources& res, const json_object_s* obj) {
    std::string meshName = StringMember(obj, "mesh");
    Mesh* mesh = res.FindMesh(meshName);
    if (!mesh) {
        std::fprintf(stderr, "command add: unknown mesh '%s'\n", meshName.c_str());
        return false;
    }
    rhi::ITexture* base = nullptr;
    rhi::ITexture* normal = nullptr;
    if (!ApplyTextureMember(obj, "baseColor", res, base)) return false;
    if (!ApplyTextureMember(obj, "normalMap", res, normal)) return false;

    Transform t;  // defaults: pos 0, euler 0, scale 1
    Vec3Member(obj, "position", t.position);
    Vec3Member(obj, "euler", t.eulerRadians);
    Vec3Member(obj, "scale", t.scale);
    float metallic = FloatMember(obj, "metallic", 0.0f);
    float roughness = FloatMember(obj, "roughness", 0.5f);

    ecs::Entity e = reg.create();
    reg.add<TransformC>(e, {t});
    reg.add<MeshC>(e, {mesh});
    reg.add<MaterialC>(e, {base, normal, metallic, roughness});

    // Report the new entity's view-order index (it appends in creation order -> last).
    int index = 0, found = -1;
    for (auto [ie, tc, mc, mat] : reg.view<TransformC, MeshC, MaterialC>()) {
        (void)tc; (void)mc; (void)mat;
        if (ie == e) found = index;
        ++index;
    }
    std::printf("add: created entity index %d (mesh '%s')\n", found, meshName.c_str());
    return true;
}

bool OpRemove(ecs::Registry& reg, const json_object_s* obj) {
    int idx = IntMember(obj, "entity", -1);
    ecs::Entity e;
    if (!EntityAtIndex(reg, idx, e)) return false;
    reg.destroy(e);
    std::printf("remove: destroyed entity index %d\n", idx);
    return true;
}

bool OpCapture(const json_object_s* obj, const CaptureFn& capture) {
    std::string path = StringMember(obj, "path");
    if (path.empty()) { std::fprintf(stderr, "command capture: missing 'path'\n"); return false; }
    if (!capture) {
        std::fprintf(stderr, "capture: no capture callback provided (skipped '%s')\n", path.c_str());
        return false;
    }
    if (!capture(path.c_str())) {
        std::fprintf(stderr, "capture: callback failed for '%s'\n", path.c_str());
        return false;
    }
    std::printf("capture: wrote %s\n", path.c_str());
    return true;
}

bool OpSaveScene(ecs::Registry& reg, SceneResources& res, const json_object_s* obj) {
    std::string path = StringMember(obj, "path");
    if (path.empty()) { std::fprintf(stderr, "command save_scene: missing 'path'\n"); return false; }
    std::string json = DumpScene(reg, res);
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "save_scene: cannot write '%s'\n", path.c_str()); return false; }
    f << json;
    std::printf("save_scene: wrote %s\n", path.c_str());
    return true;
}

// {"op":"introspect","path":"out.json"} -> write the machine-readable engine-state JSON document
// (editor::DescribeEngine) to `path`, or to stdout if `path` is absent/empty. This is the OBSERVE
// half of the agentic loop, requestable mid-script. The command layer holds no camera/light state,
// so it dumps the scene + the shipped-engine manifest with an empty EngineState (no camera/lights,
// backend ""); the sample's --introspect entry supplies the full camera/light view. The op routes
// purely through hf_core (DescribeEngine touches no backend symbols), so no rendering callback.
bool OpIntrospect(ecs::Registry& reg, SceneResources& res, const json_object_s* obj) {
    editor::EngineState state;  // scene-only view: no camera/lights, backend unset.
    std::string json = editor::DescribeEngine(reg, res, state);
    std::string path = StringMember(obj, "path");
    if (path.empty()) {
        std::fputs(json.c_str(), stdout);
        return true;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "introspect: cannot write '%s'\n", path.c_str()); return false; }
    f << json;
    std::printf("introspect: wrote %s\n", path.c_str());
    return true;
}

// --- Driver --------------------------------------------------------------------------------------

bool RunParsed(json_value_s* root, ecs::Registry& reg, SceneResources& res,
               const CaptureFn& capture) {
    if (root->type != json_type_array)
        throw std::runtime_error("RunCommands: top-level JSON must be an array of command objects");

    const json_array_s* arr = static_cast<const json_array_s*>(root->payload);
    bool allOk = true;
    int cmdIndex = 0;
    for (const json_array_element_s* el = arr->start; el; el = el->next, ++cmdIndex) {
        if (el->value->type != json_type_object) {
            std::fprintf(stderr, "command %d: not an object\n", cmdIndex);
            allOk = false;
            continue;
        }
        const json_object_s* obj = static_cast<const json_object_s*>(el->value->payload);
        std::string op = StringMember(obj, "op");
        bool ok;
        if (op == "dump")              ok = OpDump(reg, res);
        else if (op == "list")         ok = OpList(reg, res);
        else if (op == "set_transform")ok = OpSetTransform(reg, obj);
        else if (op == "set_material") ok = OpSetMaterial(reg, res, obj);
        else if (op == "add")          ok = OpAdd(reg, res, obj);
        else if (op == "remove")       ok = OpRemove(reg, obj);
        else if (op == "capture")      ok = OpCapture(obj, capture);
        else if (op == "save_scene")   ok = OpSaveScene(reg, res, obj);
        else if (op == "introspect")   ok = OpIntrospect(reg, res, obj);
        else {
            std::fprintf(stderr, "command %d: unknown op '%s'\n", cmdIndex, op.c_str());
            ok = false;
        }
        allOk = ok && allOk;
    }
    return allOk;
}

bool RunFromString(const char* text, size_t size, ecs::Registry& reg, SceneResources& res,
                   const CaptureFn& capture) {
    json_parse_result_s err{};
    json_value_s* root = json_parse_ex(text, size, json_parse_flags_default, nullptr, nullptr, &err);
    if (!root) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                      "RunCommands: JSON parse error %u at line %zu, row %zu",
                      static_cast<unsigned>(err.error), err.error_line_no, err.error_row_no);
        throw std::runtime_error(msg);
    }
    struct FreeGuard { void* p; ~FreeGuard() { std::free(p); } } guard{root};
    return RunParsed(root, reg, res, capture);
}

}  // namespace

bool RunCommands(ecs::Registry& reg, SceneResources& resources, const char* commandsJsonPath,
                 const CaptureFn& capture) {
    std::string text = ReadFile(commandsJsonPath);
    return RunFromString(text.data(), text.size(), reg, resources, capture);
}

bool RunCommandsFromText(ecs::Registry& reg, SceneResources& resources, const char* commandsJson,
                         const CaptureFn& capture) {
    return RunFromString(commandsJson, std::strlen(commandsJson), reg, resources, capture);
}

}  // namespace hf::scene
