// Hazard Forge — JSON loader for material graphs (Slice AV). See material_loader.h.
// Uses the vendored single-header parser (third_party/json/json.h), as scene_io + introspect do.
// Pure CPU; no backend symbols.
#include "material/material_loader.h"

#include "json/json.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace hf::material {
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
bool HasInt(const json_object_s* o, const char* key, int& out) {
    const json_value_s* v = MemberOf(o, key);
    if (!IsNumber(v)) return false;
    out = (int)AsNumber(v);
    return true;
}

LoadResult Err(std::string msg) {
    LoadResult r; r.ok = false; r.error = std::move(msg); return r;
}

}  // namespace

LoadResult LoadGraphFromJson(const std::string& json) {
    json_parse_result_s perr{};
    json_value_s* root = json_parse_ex(json.data(), json.size(), json_parse_flags_default,
                                       nullptr, nullptr, &perr);
    if (!root) return Err("JSON parse error at offset " + std::to_string(perr.error_offset));

    LoadResult result;
    const json_object_s* top = AsObject(root);
    if (!top) { std::free(root); return Err("top-level JSON value is not an object"); }

    result.name = AsString(MemberOf(top, "name"));

    const json_array_s* nodes = AsArray(MemberOf(top, "nodes"));
    if (!nodes) { std::free(root); return Err("missing 'nodes' array"); }
    const json_array_s* edges = AsArray(MemberOf(top, "edges"));  // may be empty/absent.

    Graph g;
    for (const json_array_element_s* el = nodes->start; el; el = el->next) {
        const json_object_s* no = AsObject(el->value);
        if (!no) { std::free(root); return Err("node entry is not an object"); }
        Node n;
        if (!HasInt(no, "id", n.id)) { std::free(root); return Err("node missing integer 'id'"); }
        std::string typeStr = AsString(MemberOf(no, "type"));
        auto kind = ParseNodeKind(typeStr);
        if (!kind) { std::free(root); return Err("unknown node type '" + typeStr + "'"); }
        n.kind = *kind;

        // Params.
        if (const json_array_s* val = AsArray(MemberOf(no, "value"))) {
            int k = 0;
            for (const json_array_element_s* ve = val->start; ve && k < 4; ve = ve->next, ++k)
                n.value[(size_t)k] = (float)AsNumber(ve->value);
        }
        std::string ot = AsString(MemberOf(no, "outType"));
        if (ot == "float")  n.outType = Type::Float;
        else if (ot == "float2") n.outType = Type::Float2;
        else if (ot == "float3") n.outType = Type::Float3;
        else if (ot == "float4") n.outType = Type::Float4;
        // (default float4 if omitted)
        n.texture = AsString(MemberOf(no, "texture"));
        if (IsNumber(MemberOf(no, "power"))) n.power = (float)AsNumber(MemberOf(no, "power"));

        g.nodes.push_back(n);
    }

    if (edges) {
        for (const json_array_element_s* el = edges->start; el; el = el->next) {
            const json_object_s* eo = AsObject(el->value);
            if (!eo) { std::free(root); return Err("edge entry is not an object"); }
            Edge e;
            if (!HasInt(eo, "from", e.fromNode)) { std::free(root); return Err("edge missing 'from'"); }
            if (!HasInt(eo, "to", e.toNode))     { std::free(root); return Err("edge missing 'to'"); }
            e.toPort = AsString(MemberOf(eo, "port"));
            if (e.toPort.empty()) { std::free(root); return Err("edge missing 'port'"); }
            g.edges.push_back(e);
        }
    }

    std::free(root);

    // Validate before handing back a graph.
    ValidationResult vr = Validate(g);
    if (!vr) return Err("graph validation failed: " + vr.error);

    result.ok = true;
    result.graph = std::move(g);
    return result;
}

LoadResult LoadGraphFromFile(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return Err("could not open '" + path + "'");
    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (sz < 0) { std::fclose(fp); return Err("could not size '" + path + "'"); }
    std::string buf((size_t)sz, '\0');
    size_t rd = std::fread(buf.data(), 1, (size_t)sz, fp);
    std::fclose(fp);
    buf.resize(rd);
    return LoadGraphFromJson(buf);
}

}  // namespace hf::material
