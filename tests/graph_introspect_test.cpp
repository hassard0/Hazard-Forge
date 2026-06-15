// Unit test for material-graph introspection (engine/material/graph_introspect.{h,cpp}, Slice BI).
// Pure C++ (hf_core), ASan-eligible. Covers:
//   * STRUCTURE: a hand-built graph -> correct nodeCount/edgeCount, PBROutput id reported as "output",
//     each node's type/params, and each edge's from/to with the right port names (parsed via json.h).
//   * UNCONNECTED DEFAULTS: an unconnected PBROutput input is reported with source:null and its
//     documented default value (baseColor=[1,1,1], metallic=0, roughness=1, emissive=[0,0,0]).
//   * DETERMINISM: DescribeGraphJson on the same graph twice -> byte-identical; node/port order stable.
//   * DOT WELL-FORMEDNESS: ToDot starts with "digraph", one node line per node + one edge line per
//     edge, and is deterministic.
//   * GOLDEN PARITY: DescribeGraphJson(load(showcase3.mat.json)) byte-equals the committed
//     tests/golden/material/showcase3_graph.json (the test READS the golden file and compares).
#include "material/graph_introspect.h"
#include "material/material_loader.h"
#include "material/shader_graph.h"

#include "json/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace hf;
using namespace hf::material;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// --- Minimal typed accessors over json.h --------------------------------------------------------
static const json_value_s* MemberOf(const json_object_s* obj, const char* key) {
    if (!obj) return nullptr;
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
static bool IsNull(const json_value_s* v) { return v && v->type == json_type_null; }
static double AsNumber(const json_value_s* v, double fb = 0.0) {
    if (!v || v->type != json_type_number) return fb;
    return std::strtod(static_cast<const json_number_s*>(v->payload)->number, nullptr);
}
static bool approx(double a, double b) { return std::fabs(a - b) < 1e-5; }

// Find the node object with a given "id" in the parsed "nodes" array.
static const json_object_s* NodeById(const json_array_s* nodes, int id) {
    if (!nodes) return nullptr;
    for (const json_array_element_s* el = nodes->start; el; el = el->next) {
        const json_object_s* o = AsObject(el->value);
        if (o && (int)AsNumber(MemberOf(o, "id")) == id) return o;
    }
    return nullptr;
}

// Find an input port object by name within a node's "inputs" array.
static const json_object_s* InputByPort(const json_object_s* node, const char* port) {
    const json_array_s* ins = AsArray(MemberOf(node, "inputs"));
    if (!ins) return nullptr;
    for (const json_array_element_s* el = ins->start; el; el = el->next) {
        const json_object_s* o = AsObject(el->value);
        if (o && AsString(MemberOf(o, "port")) == port) return o;
    }
    return nullptr;
}

static std::string ReadFile(const char* path) {
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) return {};
    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (sz < 0) { std::fclose(fp); return {}; }
    std::string buf((size_t)sz, '\0');
    size_t rd = std::fread(buf.data(), 1, (size_t)sz, fp);
    std::fclose(fp);
    buf.resize(rd);
    return buf;
}

// Build a small hand graph:  UV(1) -> TextureSample(2,"albedo") -> baseColor of PBROutput(99).
// PBROutput's metallic/roughness/emissive/normal are left UNCONNECTED (default-tested).
static Graph BuildSmall() {
    Graph g;
    { Node n; n.id = 1; n.kind = NodeKind::UV; g.nodes.push_back(n); }
    { Node n; n.id = 2; n.kind = NodeKind::TextureSample; n.texture = "albedo"; g.nodes.push_back(n); }
    { Node n; n.id = 99; n.kind = NodeKind::PBROutput; g.nodes.push_back(n); }
    g.edges.push_back({1, 2, "uv"});
    g.edges.push_back({2, 99, "baseColor"});
    return g;
}

int main() {
    // =============================================================================================
    // STRUCTURE + DEFAULTS
    // =============================================================================================
    Graph small = BuildSmall();
    check(Validate(small).ok, "hand-built small graph validates");

    std::string json = DescribeGraphJson(small, "small");

    // Determinism: a second identical call must be byte-identical.
    std::string json2 = DescribeGraphJson(small, "small");
    check(json == json2, "DescribeGraphJson is deterministic (byte-identical across runs)");

    json_parse_result_s perr{};
    json_value_s* root = json_parse_ex(json.data(), json.size(), json_parse_flags_default,
                                       nullptr, nullptr, &perr);
    check(root != nullptr, "DescribeGraphJson output parses as JSON");
    const json_object_s* top = AsObject(root);
    check(top != nullptr, "top-level value is an object");

    if (top) {
        check(AsString(MemberOf(top, "material")) == "small", "material == small");
        check((int)AsNumber(MemberOf(top, "nodeCount")) == 3, "nodeCount == 3");
        check((int)AsNumber(MemberOf(top, "edgeCount")) == 2, "edgeCount == 2");
        check((int)AsNumber(MemberOf(top, "output")) == 99, "output == 99 (PBROutput id)");

        const json_array_s* nodes = AsArray(MemberOf(top, "nodes"));
        check(nodes != nullptr && nodes->length == 3, "nodes array has 3 entries");

        // Nodes emitted in ASCENDING id order: 1, 2, 99.
        if (nodes && nodes->length == 3) {
            const json_object_s* a = AsObject(nodes->start->value);
            const json_object_s* b = AsObject(nodes->start->next->value);
            const json_object_s* c = AsObject(nodes->start->next->next->value);
            check((int)AsNumber(MemberOf(a, "id")) == 1 &&
                  (int)AsNumber(MemberOf(b, "id")) == 2 &&
                  (int)AsNumber(MemberOf(c, "id")) == 99, "nodes are sorted by ascending id");
        }

        // Node types.
        const json_object_s* uv = NodeById(nodes, 1);
        const json_object_s* tex = NodeById(nodes, 2);
        const json_object_s* pbr = NodeById(nodes, 99);
        check(uv && AsString(MemberOf(uv, "type")) == "UV", "node 1 type == UV");
        check(tex && AsString(MemberOf(tex, "type")) == "TextureSample", "node 2 type == TextureSample");
        check(pbr && AsString(MemberOf(pbr, "type")) == "PBROutput", "node 99 type == PBROutput");

        // TextureSample params.texture == "albedo"; output type float4.
        const json_object_s* texParams = tex ? AsObject(MemberOf(tex, "params")) : nullptr;
        check(texParams && AsString(MemberOf(texParams, "texture")) == "albedo",
              "node 2 params.texture == albedo");
        const json_object_s* texOut = tex ? AsObject(MemberOf(tex, "output")) : nullptr;
        check(texOut && AsString(MemberOf(texOut, "type")) == "float4", "node 2 output type == float4");

        // UV has zero inputs + float2 output.
        const json_array_s* uvIns = uv ? AsArray(MemberOf(uv, "inputs")) : nullptr;
        check(uvIns && uvIns->length == 0, "node 1 (UV) has no inputs");
        const json_object_s* uvOut = uv ? AsObject(MemberOf(uv, "output")) : nullptr;
        check(uvOut && AsString(MemberOf(uvOut, "type")) == "float2", "node 1 output type == float2");

        // TextureSample.uv input is connected to "1.out", default null.
        const json_object_s* texUv = InputByPort(tex, "uv");
        check(texUv && AsString(MemberOf(texUv, "source")) == "1.out",
              "node 2 uv input source == 1.out");
        check(texUv && IsNull(MemberOf(texUv, "default")), "node 2 uv input default == null (connected)");

        // PBROutput.baseColor connected -> "2.out".
        const json_object_s* bc = InputByPort(pbr, "baseColor");
        check(bc && AsString(MemberOf(bc, "source")) == "2.out",
              "PBROutput baseColor source == 2.out");

        // UNCONNECTED PBROutput defaults: metallic=0, roughness=1, emissive=[0,0,0], normal=[0,0,1].
        const json_object_s* met = InputByPort(pbr, "metallic");
        check(met && IsNull(MemberOf(met, "source")), "PBROutput metallic source == null (unconnected)");
        check(met && approx(AsNumber(MemberOf(met, "default")), 0.0), "PBROutput metallic default == 0");
        const json_object_s* rough = InputByPort(pbr, "roughness");
        check(rough && IsNull(MemberOf(rough, "source")), "PBROutput roughness source == null");
        check(rough && approx(AsNumber(MemberOf(rough, "default")), 1.0), "PBROutput roughness default == 1");
        const json_object_s* em = InputByPort(pbr, "emissive");
        const json_array_s* emDef = em ? AsArray(MemberOf(em, "default")) : nullptr;
        check(emDef && emDef->length == 3 && approx(AsNumber(emDef->start->value), 0.0),
              "PBROutput emissive default == [0,0,0]");
        const json_object_s* nrm = InputByPort(pbr, "normal");
        const json_array_s* nrmDef = nrm ? AsArray(MemberOf(nrm, "default")) : nullptr;
        check(nrmDef && nrmDef->length == 3 &&
              approx(AsNumber(nrmDef->start->value), 0.0) &&
              approx(AsNumber(nrmDef->start->next->value), 0.0) &&
              approx(AsNumber(nrmDef->start->next->next->value), 1.0),
              "PBROutput normal default == [0,0,1]");

        // Edges: from "1.out" to "2.uv", from "2.out" to "99.baseColor".
        const json_array_s* edges = AsArray(MemberOf(top, "edges"));
        check(edges != nullptr && edges->length == 2, "edges array has 2 entries");
        if (edges && edges->length == 2) {
            const json_object_s* e0 = AsObject(edges->start->value);
            const json_object_s* e1 = AsObject(edges->start->next->value);
            check(AsString(MemberOf(e0, "from")) == "1.out" && AsString(MemberOf(e0, "to")) == "2.uv",
                  "edge 0 == 1.out -> 2.uv");
            check(AsString(MemberOf(e1, "from")) == "2.out" &&
                  AsString(MemberOf(e1, "to")) == "99.baseColor",
                  "edge 1 == 2.out -> 99.baseColor");
        }
    }
    if (root) std::free(root);

    // =============================================================================================
    // DOT WELL-FORMEDNESS
    // =============================================================================================
    std::string dot = ToDot(small);
    std::string dot2 = ToDot(small);
    check(dot == dot2, "ToDot is deterministic (byte-identical across runs)");
    check(dot.rfind("digraph", 0) == 0, "ToDot starts with 'digraph'");
    // Count node lines vs edge lines per-line: edge lines contain " -> " (and also a label); node
    // lines contain "[label=" but NOT " -> ". One node line per node, one edge line per edge.
    {
        size_t nodeLines = 0, edgeLines = 0, start = 0;
        while (start <= dot.size()) {
            size_t nl = dot.find('\n', start);
            std::string line = dot.substr(start, (nl == std::string::npos ? dot.size() : nl) - start);
            bool isEdge = line.find(" -> ") != std::string::npos;
            bool hasLabel = line.find("[label=") != std::string::npos;
            if (isEdge) ++edgeLines;
            else if (hasLabel) ++nodeLines;
            if (nl == std::string::npos) break;
            start = nl + 1;
        }
        check(nodeLines == small.nodes.size(), "ToDot has one node-label line per node");
        check(edgeLines == small.edges.size(), "ToDot has one edge line per edge");
    }

    // =============================================================================================
    // GOLDEN PARITY: DescribeGraphJson(load(showcase3.mat.json)) == committed golden
    // =============================================================================================
#if defined(HF_MAT3_JSON) && defined(HF_MAT3_GOLDEN)
    LoadResult lr = LoadGraphFromFile(HF_MAT3_JSON);
    check(lr.ok, "showcase3.mat.json loads");
    if (lr.ok) {
        std::string live = DescribeGraphJson(lr.graph, lr.name);
        std::string golden = ReadFile(HF_MAT3_GOLDEN);
        check(!golden.empty(), "showcase3_graph.json golden is readable + non-empty");
        check(live == golden,
              "DescribeGraphJson(showcase3) byte-equals tests/golden/material/showcase3_graph.json");
        if (live != golden) {
            // Help debugging: report the first differing offset + a small window.
            size_t i = 0;
            while (i < live.size() && i < golden.size() && live[i] == golden[i]) ++i;
            std::printf("  golden parity diverges at byte %zu (live len %zu, golden len %zu)\n",
                        i, live.size(), golden.size());
        }
    }
#else
    check(false, "HF_MAT3_JSON / HF_MAT3_GOLDEN must be defined for the golden-parity test");
#endif

    if (g_fail == 0) { std::printf("graph_introspect_test OK\n"); return 0; }
    std::printf("graph_introspect_test: %d failures\n", g_fail);
    return 1;
}
