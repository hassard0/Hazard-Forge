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

// --- DX2 query response builders -----------------------------------------------------------------
//
// These build the JSON read of the scene that the "query" op prints + RunQueries returns. The float
// formatting mirrors scene_io's AppendFloat (%g, double-promoted) so a query:entity response is the
// SAME bytes DumpScene emits for that entity's components — the house JSON style, backend-identical.

void AppendFloatQ(std::string& os, float f) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(f));
    os += buf;
}

void AppendVec3Q(std::string& os, const math::Vec3& v) {
    os += "[";
    AppendFloatQ(os, v.x); os += ", ";
    AppendFloatQ(os, v.y); os += ", ";
    AppendFloatQ(os, v.z);
    os += "]";
}

// The "transform"/"material"/"mesh" component fragments for an entity, indented at `ind` spaces.
// Each appends a single `"key": <value>` (NO leading indent, NO trailing comma/newline) so the caller
// joins them with ",\n" + indent in the canonical order.
void AppendTransformField(std::string& os, const TransformC& tc, const std::string& ind) {
    os += "\"transform\": {\n";
    os += ind + "  \"position\": "; AppendVec3Q(os, tc.t.position); os += ",\n";
    os += ind + "  \"euler\": ";    AppendVec3Q(os, tc.t.eulerRadians); os += ",\n";
    os += ind + "  \"scale\": ";    AppendVec3Q(os, tc.t.scale); os += "\n";
    os += ind + "}";
}

void AppendMaterialField(std::string& os, const MaterialC& mat, const SceneResources& res,
                         const std::string& ind) {
    std::string baseName   = mat.base   ? res.NameOfTexture(mat.base)   : std::string();
    std::string normalName = mat.normal ? res.NameOfTexture(mat.normal) : std::string();
    os += "\"material\": {\n";
    os += ind + "  \"metallic\": ";  AppendFloatQ(os, mat.metallic);  os += ",\n";
    os += ind + "  \"roughness\": "; AppendFloatQ(os, mat.roughness); os += ",\n";
    os += ind + "  \"baseColor\": ";
    if (baseName.empty()) os += "null"; else { os += "\""; os += baseName; os += "\""; }
    os += ",\n";
    os += ind + "  \"normalMap\": ";
    if (normalName.empty()) os += "null"; else { os += "\""; os += normalName; os += "\""; }
    os += "\n";
    os += ind + "}";
}

void AppendMeshField(std::string& os, const MeshC& mc, const SceneResources& res) {
    std::string meshName = mc.mesh ? res.NameOfMesh(mc.mesh) : std::string();
    os += "\"mesh\": \""; os += meshName; os += "\"";
}

// Collect the requested field-set from a query object's optional "fields" array. Returns the
// recognized fields as three booleans (in canonical order) + any UNKNOWN names (in request order,
// for a deterministic "unknownFields" echo). "fields" absent -> all three recognized, no unknowns.
struct FieldSelect {
    bool transform = true, material = true, mesh = true;
    bool explicitFields = false;
    std::vector<std::string> unknown;
};
FieldSelect ParseFields(const json_object_s* obj) {
    FieldSelect sel;
    const json_value_s* v = MemberOf(obj, "fields");
    if (!v || v->type != json_type_array) return sel;  // absent/malformed -> all fields
    sel.explicitFields = true;
    sel.transform = sel.material = sel.mesh = false;
    const json_array_s* arr = static_cast<const json_array_s*>(v->payload);
    for (const json_array_element_s* el = arr->start; el; el = el->next) {
        if (el->value->type != json_type_string) continue;
        const json_string_s* s = static_cast<const json_string_s*>(el->value->payload);
        std::string name(s->string, s->string_size);
        if (name == "transform")      sel.transform = true;
        else if (name == "material")  sel.material = true;
        else if (name == "mesh")      sel.mesh = true;
        else                          sel.unknown.push_back(name);
    }
    return sel;
}

// Build the response object for a single query, indented to start at column `base` (the spaces the
// caller has already placed before the opening '{'). Deterministic + side-effect-free.
std::string QueryResponse(ecs::Registry& reg, SceneResources& res, const json_object_s* obj,
                          const std::string& base) {
    std::string select = StringMember(obj, "select");
    std::string out;
    const std::string ind = base + "  ";  // member indent inside the response object

    if (select == "stats") {
        // Live entity count (view order) + the named mesh/texture resource counts.
        int entities = 0;
        for (auto [e, tc, mc, mat] : reg.view<TransformC, MeshC, MaterialC>()) {
            (void)e; (void)tc; (void)mc; (void)mat; ++entities;
        }
        out += "{\n";
        out += ind + "\"entities\": " + std::to_string(entities) + ",\n";
        out += ind + "\"meshes\": " + std::to_string(res.meshes.size()) + ",\n";
        out += ind + "\"textures\": " + std::to_string(res.textures.size()) + "\n";
        out += base + "}";
        return out;
    }

    if (select == "entities") {
        out += "[";
        int i = 0;
        bool first = true;
        for (auto [e, tc, mc, mat] : reg.view<TransformC, MeshC, MaterialC>()) {
            (void)e; (void)tc; (void)mat;
            if (!first) out += ",";
            first = false;
            std::string meshName = mc.mesh ? res.NameOfMesh(mc.mesh) : std::string();
            out += "\n" + ind + "{ \"index\": " + std::to_string(i) + ", \"mesh\": \"" + meshName + "\" }";
            ++i;
        }
        if (!first) out += "\n" + base;
        out += "]";
        return out;
    }

    // select:"entity" (the default + only remaining recognized select).
    int index = IntMember(obj, "entity", -1);
    ecs::Entity e;
    bool found = false;
    int count = 0;
    {
        int i = 0;
        for (auto [ent, tc, mc, mat] : reg.view<TransformC, MeshC, MaterialC>()) {
            (void)tc; (void)mc; (void)mat;
            if (i == index && index >= 0) { e = ent; found = true; }
            ++i;
        }
        count = i;
    }
    if (!found) {
        out += "{ \"error\": \"out-of-range\", \"entity\": " + std::to_string(index) +
               ", \"count\": " + std::to_string(count) + " }";
        return out;
    }

    FieldSelect sel = ParseFields(obj);
    const auto& tc  = reg.get<TransformC>(e);
    const auto& mc  = reg.get<MeshC>(e);
    const auto& mat = reg.get<MaterialC>(e);

    out += "{\n";
    out += ind + "\"index\": " + std::to_string(index);
    // Canonical field order: transform, material, mesh (REGARDLESS of request order).
    if (sel.transform) { out += ",\n" + ind; AppendTransformField(out, tc, ind); }
    if (sel.material)  { out += ",\n" + ind; AppendMaterialField(out, mat, res, ind); }
    if (sel.mesh)      { out += ",\n" + ind; AppendMeshField(out, mc, res); }
    if (!sel.unknown.empty()) {
        out += ",\n" + ind + "\"unknownFields\": [";
        for (size_t u = 0; u < sel.unknown.size(); ++u) {
            if (u) out += ", ";
            out += "\""; out += sel.unknown[u]; out += "\"";
        }
        out += "]";
    }
    out += "\n" + base + "}";
    return out;
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

// Pretty-print an arbitrary parsed json.h value (used to echo a query's "request" verbatim into the
// RunQueries response array). Deterministic: object member ORDER is preserved as parsed, numbers are
// emitted from their original literal, strings re-quoted, 2-space indent — the house JSON style.
void AppendJsonValue(std::string& os, const json_value_s* v, const std::string& base) {
    const std::string ind = base + "  ";
    switch (v->type) {
        case json_type_object: {
            const json_object_s* o = static_cast<const json_object_s*>(v->payload);
            if (!o->start) { os += "{}"; return; }
            os += "{\n";
            for (const json_object_element_s* el = o->start; el; el = el->next) {
                os += ind + "\"" + std::string(el->name->string, el->name->string_size) + "\": ";
                AppendJsonValue(os, el->value, ind);
                os += el->next ? ",\n" : "\n";
            }
            os += base + "}";
            return;
        }
        case json_type_array: {
            const json_array_s* a = static_cast<const json_array_s*>(v->payload);
            if (!a->start) { os += "[]"; return; }
            os += "[";
            for (const json_array_element_s* el = a->start; el; el = el->next) {
                os += "\n" + ind;
                AppendJsonValue(os, el->value, ind);
                if (el->next) os += ",";
            }
            os += "\n" + base + "]";
            return;
        }
        case json_type_string: {
            const json_string_s* s = static_cast<const json_string_s*>(v->payload);
            os += "\""; os += std::string(s->string, s->string_size); os += "\"";
            return;
        }
        case json_type_number: {
            const json_number_s* n = static_cast<const json_number_s*>(v->payload);
            os += std::string(n->number, n->number_size);
            return;
        }
        case json_type_true:  os += "true";  return;
        case json_type_false: os += "false"; return;
        case json_type_null:  os += "null";  return;
        default:              os += "null";  return;
    }
}

// {"op":"query","select":...} -> print the SELECTIVE JSON read of the scene to stdout (the live
// --commands form of the DX2 read). The response bytes are the SAME QueryResponse RunQueries returns
// (column-0 indented for the standalone print). Always "succeeds" (an out-of-range entity is a valid
// deterministic {"error":"out-of-range"} response, not a command failure).
bool OpQuery(ecs::Registry& reg, SceneResources& res, const json_object_s* obj) {
    std::string response = QueryResponse(reg, res, obj, "");
    std::fputs(response.c_str(), stdout);
    std::fputc('\n', stdout);
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
        else if (op == "query")        ok = OpQuery(reg, res, obj);
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

std::string RunQueries(ecs::Registry& reg, SceneResources& resources, const char* queriesJson) {
    json_parse_result_s err{};
    json_value_s* root = json_parse_ex(queriesJson, std::strlen(queriesJson),
                                       json_parse_flags_default, nullptr, nullptr, &err);
    if (!root) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                      "RunQueries: JSON parse error %u at line %zu, row %zu",
                      static_cast<unsigned>(err.error), err.error_line_no, err.error_row_no);
        throw std::runtime_error(msg);
    }
    struct FreeGuard { void* p; ~FreeGuard() { std::free(p); } } guard{root};

    if (root->type != json_type_array)
        throw std::runtime_error("RunQueries: top-level JSON must be an array of query objects");

    const json_array_s* arr = static_cast<const json_array_s*>(root->payload);
    std::string out = "[";
    bool first = true;
    for (const json_array_element_s* el = arr->start; el; el = el->next) {
        if (el->value->type != json_type_object) continue;  // skip non-object query entries
        const json_object_s* obj = static_cast<const json_object_s*>(el->value->payload);
        if (!first) out += ",";
        first = false;
        out += "\n  {\n";
        out += "    \"request\": ";
        AppendJsonValue(out, el->value, "    ");
        out += ",\n";
        out += "    \"response\": ";
        out += QueryResponse(reg, resources, obj, "    ");
        out += "\n  }";
    }
    if (!first) out += "\n";
    out += "]\n";
    return out;
}

}  // namespace hf::scene
