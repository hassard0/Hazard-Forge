// Data-driven post-process stack — JSON loader (Slice BN). See engine/render/post_stack.h for the
// model + the shared per-effect math. This TU only adds the data-authoring front door (LoadPostStack)
// using the vendored single-header parser (third_party/json/json.h), as material_loader.cpp does. Pure
// CPU; NO backend symbols. Compiled into both hf_engine and the ASan-pure hf_core.
//
// JSON shape (an ordered chain):
//   { "effects": [
//       { "kind": "tonemap",  "exposure": 1.7 },
//       { "kind": "colorgrade", "lift": [..], "gamma": [..], "gain": [..] },
//       { "kind": "chromatic", "strength": 2.0 },
//       { "kind": "vignette",  "outer": 0.8, "inner": 0.35 },
//       { "kind": "grain",     "intensity": 0.05 }
//   ] }
// An absent/empty "effects" array yields an empty (pass-through) stack. Order is preserved. Unknown
// effect kinds make the load fail (ok=false) so a typo is caught rather than silently dropped.
#include "render/post_stack.h"

#include "json/json.h"

#include <cstdlib>
#include <cstring>
#include <optional>

namespace hf::render::post {
namespace {

const json_value_s* MemberOf(const json_object_s* obj, const char* key) {
    if (!obj) return nullptr;
    for (const json_object_element_s* el = obj->start; el; el = el->next)
        if (el->name && std::strcmp(el->name->string, key) == 0) return el->value;
    return nullptr;
}
const json_object_s* AsObject(const json_value_s* v) {
    return (v && v->type == json_type_object) ? static_cast<const json_object_s*>(v->payload) : nullptr;
}
const json_array_s* AsArray(const json_value_s* v) {
    return (v && v->type == json_type_array) ? static_cast<const json_array_s*>(v->payload) : nullptr;
}
std::string AsString(const json_value_s* v) {
    if (!v || v->type != json_type_string) return {};
    const json_string_s* s = static_cast<const json_string_s*>(v->payload);
    return std::string(s->string, s->string_size);
}
bool IsNumber(const json_value_s* v) { return v && v->type == json_type_number; }
double AsNumber(const json_value_s* v, double fb = 0.0) {
    if (!IsNumber(v)) return fb;
    return std::strtod(static_cast<const json_number_s*>(v->payload)->number, nullptr);
}
float NumberOr(const json_object_s* o, const char* key, float fb) {
    const json_value_s* v = MemberOf(o, key);
    return IsNumber(v) ? (float)AsNumber(v) : fb;
}
// Read a 3-element number array into a Vec3, leaving any missing component at its fallback.
math::Vec3 Vec3Or(const json_object_s* o, const char* key, const math::Vec3& fb) {
    const json_array_s* a = AsArray(MemberOf(o, key));
    if (!a) return fb;
    math::Vec3 out = fb;
    float* comp[3] = {&out.x, &out.y, &out.z};
    int k = 0;
    for (const json_array_element_s* el = a->start; el && k < 3; el = el->next, ++k)
        if (IsNumber(el->value)) *comp[k] = (float)AsNumber(el->value);
    return out;
}

std::optional<Kind> ParseKind(const std::string& s) {
    if (s == "tonemap")   return Kind::Tonemap;
    if (s == "colorgrade") return Kind::ColorGrade;
    if (s == "chromatic") return Kind::ChromaticAberration;
    if (s == "vignette")  return Kind::Vignette;
    if (s == "grain")     return Kind::FilmGrain;
    return std::nullopt;
}

}  // namespace

LoadResult LoadPostStack(const std::string& json) {
    json_parse_result_s perr{};
    json_value_s* root = json_parse_ex(json.data(), json.size(), json_parse_flags_default,
                                       nullptr, nullptr, &perr);
    LoadResult result;
    if (!root) {
        result.ok = false;
        result.error = "JSON parse error at offset " + std::to_string(perr.error_offset);
        return result;
    }
    const json_object_s* top = AsObject(root);
    if (!top) {
        std::free(root);
        result.ok = false;
        result.error = "top-level JSON value is not an object";
        return result;
    }

    // Absent "effects" -> empty (pass-through) stack; that is valid.
    const json_array_s* effects = AsArray(MemberOf(top, "effects"));
    PostStack stack;
    if (effects) {
        for (const json_array_element_s* el = effects->start; el; el = el->next) {
            const json_object_s* eo = AsObject(el->value);
            if (!eo) {
                std::free(root);
                result.ok = false;
                result.error = "effect entry is not an object";
                return result;
            }
            std::string kindStr = AsString(MemberOf(eo, "kind"));
            std::optional<Kind> kind = ParseKind(kindStr);
            if (!kind) {
                std::free(root);
                result.ok = false;
                result.error = "unknown effect kind '" + kindStr + "'";
                return result;
            }
            PostEffect e;
            e.kind = *kind;
            switch (*kind) {
                case Kind::Tonemap:
                    e.exposure = NumberOr(eo, "exposure", 1.0f);
                    break;
                case Kind::ColorGrade:
                    e.lift  = Vec3Or(eo, "lift",  math::Vec3{0.0f, 0.0f, 0.0f});
                    e.gamma = Vec3Or(eo, "gamma", math::Vec3{1.0f, 1.0f, 1.0f});
                    e.gain  = Vec3Or(eo, "gain",  math::Vec3{1.0f, 1.0f, 1.0f});
                    break;
                case Kind::ChromaticAberration:
                    e.strength = NumberOr(eo, "strength", 0.0f);
                    break;
                case Kind::Vignette:
                    e.vignetteOuter = NumberOr(eo, "outer", 0.8f);
                    e.vignetteInner = NumberOr(eo, "inner", 0.35f);
                    break;
                case Kind::FilmGrain:
                    e.intensity = NumberOr(eo, "intensity", 0.0f);
                    break;
            }
            stack.effects.push_back(e);
        }
    }

    std::free(root);
    result.ok = true;
    result.stack = std::move(stack);
    return result;
}

}  // namespace hf::render::post
