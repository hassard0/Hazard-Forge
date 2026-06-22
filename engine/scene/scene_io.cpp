// Hazard Forge — data-driven scene load/dump (see scene_io.h).
//
// LoadScene parses the JSON (vendored single-header parser, third_party/json/json.h — public
// domain) and creates one ECS entity per object in file order. DumpScene walks the live ECS view
// and emits the same schema. No vk*/Metal symbols here: resources are handled purely as named,
// opaque pointers via SceneResources.
#include "scene/scene_io.h"

#include "json/json.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace hf::scene {
namespace {

std::string ReadFile(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("LoadScene: cannot open scene file: ") + path);
    std::streamsize size = f.tellg();
    f.seekg(0);
    std::string s(static_cast<size_t>(size), '\0');
    f.read(s.data(), size);
    return s;
}

// --- Small typed accessors over the json.h DOM ---------------------------------------------------

// Find the named element of an object, or nullptr.
const json_value_s* MemberOf(const json_object_s* obj, const char* key) {
    for (const json_object_element_s* el = obj->start; el; el = el->next)
        if (el->name && std::strcmp(el->name->string, key) == 0) return el->value;
    return nullptr;
}

// A string member, or "" if absent / JSON null / not a string.
std::string StringMember(const json_object_s* obj, const char* key) {
    const json_value_s* v = MemberOf(obj, key);
    if (!v || v->type != json_type_string) return {};
    const json_string_s* s = static_cast<const json_string_s*>(v->payload);
    return std::string(s->string, s->string_size);
}

// A float member, or `fallback` if absent / not a number.
float FloatMember(const json_object_s* obj, const char* key, float fallback) {
    const json_value_s* v = MemberOf(obj, key);
    if (!v || v->type != json_type_number) return fallback;
    const json_number_s* n = static_cast<const json_number_s*>(v->payload);
    return std::strtof(n->number, nullptr);
}

// A [x,y,z] member into `out`, leaving `out` unchanged (its default) if absent / malformed.
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

// Append a float to `os` without a trailing-zero blowup, but losslessly enough to round-trip the
// authored values. %g with high precision keeps DumpScene -> LoadScene stable.
void AppendFloat(std::ostringstream& os, float f) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(f));
    os << buf;
}

void AppendVec3(std::ostringstream& os, const math::Vec3& v) {
    os << "[";
    AppendFloat(os, v.x); os << ", ";
    AppendFloat(os, v.y); os << ", ";
    AppendFloat(os, v.z);
    os << "]";
}

}  // namespace

// Parse a scene JSON STRING (already in memory) and create entities, IN TEXT ORDER. The shared
// core of LoadScene (file) and the in-memory CanonicalizeSceneText overload. `srcName` only labels
// parse errors. Throws std::runtime_error on parse error / non-array / unknown mesh name.
static std::vector<ecs::Entity> LoadSceneText(ecs::Registry& reg, const SceneResources& resources,
                                              const std::string& text, const char* srcName) {
    json_parse_result_s err{};
    json_value_s* root = json_parse_ex(text.data(), text.size(), json_parse_flags_default,
                                       nullptr, nullptr, &err);
    if (!root) {
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                      "LoadScene: JSON parse error %u at line %zu, row %zu in %s",
                      static_cast<unsigned>(err.error), err.error_line_no, err.error_row_no, srcName);
        throw std::runtime_error(msg);
    }
    // Own the single allocation json_parse made for the whole DOM.
    struct FreeGuard { void* p; ~FreeGuard() { std::free(p); } } guard{root};

    if (root->type != json_type_array)
        throw std::runtime_error(std::string("LoadScene: top-level JSON must be an array: ") + srcName);

    const json_array_s* arr = static_cast<const json_array_s*>(root->payload);
    std::vector<ecs::Entity> created;
    created.reserve(arr->length);

    int objIndex = 0;
    for (const json_array_element_s* el = arr->start; el; el = el->next, ++objIndex) {
        if (el->value->type != json_type_object)
            throw std::runtime_error("LoadScene: scene array element is not an object");
        const json_object_s* obj = static_cast<const json_object_s*>(el->value->payload);

        std::string meshName = StringMember(obj, "mesh");
        Mesh* mesh = resources.FindMesh(meshName);
        if (!mesh) {
            char msg[160];
            std::snprintf(msg, sizeof(msg), "LoadScene: object %d references unknown mesh '%s'",
                          objIndex, meshName.c_str());
            throw std::runtime_error(msg);
        }

        // baseColor / normalMap may be absent or JSON null -> nullptr texture.
        rhi::ITexture* base = resources.FindTexture(StringMember(obj, "baseColor"));
        rhi::ITexture* normal = resources.FindTexture(StringMember(obj, "normalMap"));

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
        created.push_back(e);
    }

    return created;
}

std::vector<ecs::Entity> LoadScene(ecs::Registry& reg, const SceneResources& resources,
                                   const char* path) {
    return LoadSceneText(reg, resources, ReadFile(path), path);
}

// CanonicalizeScene = DumpScene(LoadScene(spec)) — load the spec FILE into a fresh-state load over
// `reg`, then re-emit the engine's single canonical form. `reg` is used as scratch (the caller
// passes a fresh Registry); the returned string is the canonical scene JSON. Pure; throws
// std::runtime_error on a malformed/unknown-name spec (the LoadScene contract).
std::string CanonicalizeScene(const char* specPath, ecs::Registry& reg,
                              const SceneResources& res) {
    LoadScene(reg, res, specPath);
    return DumpScene(reg, res);
}

// The in-memory overload (file-free): canonicalize a spec held as a JSON STRING. Same contract.
std::string CanonicalizeSceneText(const char* specJson, ecs::Registry& reg,
                                  const SceneResources& res) {
    LoadSceneText(reg, res, std::string(specJson), "<text>");
    return DumpScene(reg, res);
}

std::string DumpScene(ecs::Registry& reg, const SceneResources& resources) {
    std::ostringstream os;
    os << "[\n";
    bool first = true;
    for (auto [e, tc, mc, mat] : reg.view<TransformC, MeshC, MaterialC>()) {
        (void)e;
        if (!first) os << ",\n";
        first = false;

        std::string meshName = resources.NameOfMesh(mc.mesh);
        std::string baseName = mat.base ? resources.NameOfTexture(mat.base) : std::string();
        std::string normalName = mat.normal ? resources.NameOfTexture(mat.normal) : std::string();

        os << "  {\n";
        os << "    \"mesh\": \"" << meshName << "\",\n";
        os << "    \"baseColor\": ";
        if (baseName.empty()) os << "null"; else os << "\"" << baseName << "\"";
        os << ",\n";
        os << "    \"normalMap\": ";
        if (normalName.empty()) os << "null"; else os << "\"" << normalName << "\"";
        os << ",\n";
        os << "    \"metallic\": "; AppendFloat(os, mat.metallic); os << ",\n";
        os << "    \"roughness\": "; AppendFloat(os, mat.roughness); os << ",\n";
        os << "    \"position\": "; AppendVec3(os, tc.t.position); os << ",\n";
        os << "    \"euler\": "; AppendVec3(os, tc.t.eulerRadians); os << ",\n";
        os << "    \"scale\": "; AppendVec3(os, tc.t.scale); os << "\n";
        os << "  }";
    }
    os << "\n]\n";
    return os.str();
}

}  // namespace hf::scene
