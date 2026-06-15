// Unit test for the data-driven material / shader graph (engine/material/*, Slice AV). TDD: this is
// written BEFORE the implementation. Three pillars (per the spec):
//
//   1. Graph validation: cycle detection rejects a cyclic graph; a missing PBROutput is rejected;
//      an edge type-mismatch is rejected; a graph with unconnected PBROutput inputs is VALID (the
//      inputs fall back to the documented defaults baseColor=1, metallic=0, roughness=1, emissive=0).
//   2. CPU-interpreter parity: a CPU evaluator of the SAME node semantics (Constant/UV/TextureSample/
//      Multiply/Add/Lerp/Fresnel) matches hand-computed values at sample points. This pins node
//      SEMANTICS independent of the GPU — the generated HLSL must implement the same math (the
//      interpreter + codegen share the per-node formula definitions).
//   3. Codegen determinism/structure: GenerateHlsl is deterministic (byte-identical across runs) and
//      contains the expected temporaries / topological order (structural asserts, not a brittle
//      full-string match).
//
// Plus a loader round-trip: the showcase .mat.json parses, validates, and codegens cleanly.
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "material/shader_graph.h"
#include "material/codegen.h"
#include "material/material_loader.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b) { return std::fabs(a - b) < 1e-4f; }

// A stub texture sampler for the interpreter: returns a deterministic value derived from the slot +
// uv so TextureSample is exercised on the CPU (no GPU). "checker" returns a UV-dependent gradient.
static std::array<float, 4> StubTex(const std::string& slot, float u, float v) {
    if (slot == "checker") return {u, v, 0.5f, 1.0f};
    return {0.25f, 0.25f, 0.25f, 1.0f};
}

// Build the "showcase" graph in code (mirrors assets/materials/showcase.mat.json):
//   baseColor = Lerp( TextureSample(checker).rgb, Constant(tint).rgb, Fresnel(power=3) )
//   metallic  = Constant(0.0)
//   roughness = Constant(0.35)
// Node ids: 1 UV, 2 TextureSample(checker), 3 Constant tint (float3), 4 Fresnel(3), 5 Lerp,
//           6 Constant metallic, 7 Constant roughness, 99 PBROutput.
static material::Graph MakeShowcaseGraph() {
    using namespace material;
    Graph g;
    Node uv;     uv.id = 1; uv.kind = NodeKind::UV;
    Node tex;    tex.id = 2; tex.kind = NodeKind::TextureSample; tex.texture = "checker";
    Node tint;   tint.id = 3; tint.kind = NodeKind::Constant; tint.value = {0.9f, 0.2f, 0.1f, 1.0f}; tint.outType = Type::Float4;
    Node fres;   fres.id = 4; fres.kind = NodeKind::Fresnel; fres.power = 3.0f;
    Node mix;    mix.id = 5; mix.kind = NodeKind::Lerp;
    Node metal;  metal.id = 6; metal.kind = NodeKind::Constant; metal.value = {0.0f, 0, 0, 0}; metal.outType = Type::Float;
    Node rough;  rough.id = 7; rough.kind = NodeKind::Constant; rough.value = {0.35f, 0, 0, 0}; rough.outType = Type::Float;
    Node out;    out.id = 99; out.kind = NodeKind::PBROutput;
    g.nodes = {uv, tex, tint, fres, mix, metal, rough, out};
    g.edges = {
        {1, 2, "uv"},          // UV -> TextureSample.uv
        {2, 5, "a"},           // texture.rgb -> Lerp.a
        {3, 5, "b"},           // tint -> Lerp.b
        {4, 5, "t"},           // fresnel -> Lerp.t
        {5, 99, "baseColor"},  // Lerp -> PBROutput.baseColor
        {6, 99, "metallic"},
        {7, 99, "roughness"},
    };
    return g;
}

int main() {
    using namespace material;

    // ============================ 1. VALIDATION ===============================================
    {
        Graph g = MakeShowcaseGraph();
        ValidationResult vr = Validate(g);
        check(vr.ok, "showcase graph validates");
    }

    // Missing PBROutput -> rejected.
    {
        Graph g = MakeShowcaseGraph();
        // Drop the PBROutput node (id 99) + edges into it.
        std::vector<Node> kept;
        for (auto& n : g.nodes) if (n.kind != NodeKind::PBROutput) kept.push_back(n);
        g.nodes = kept;
        std::vector<Edge> ke;
        for (auto& e : g.edges) if (e.toNode != 99) ke.push_back(e);
        g.edges = ke;
        check(!Validate(g).ok, "graph with no PBROutput is rejected");
    }

    // Two PBROutputs -> rejected (exactly one sink).
    {
        Graph g = MakeShowcaseGraph();
        Node out2; out2.id = 100; out2.kind = NodeKind::PBROutput;
        g.nodes.push_back(out2);
        check(!Validate(g).ok, "graph with two PBROutputs is rejected");
    }

    // Cycle -> rejected. Make Multiply 5 feed itself via an extra node loop: 5->8->5.
    {
        Graph g;
        Node a; a.id = 1; a.kind = NodeKind::Multiply;
        Node b; b.id = 2; b.kind = NodeKind::Multiply;
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {a, b, out};
        g.edges = {
            {1, 2, "a"}, {2, 1, "a"},      // 1 -> 2 -> 1 cycle
            {2, 99, "baseColor"},
        };
        check(!Validate(g).ok, "cyclic graph is rejected (cycle detection)");
    }

    // Edge type mismatch -> rejected. Feed a float2 (UV) into PBROutput.metallic (float).
    {
        Graph g;
        Node uv; uv.id = 1; uv.kind = NodeKind::UV;             // outputs float2
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {uv, out};
        g.edges = { {1, 99, "metallic"} };                      // float2 -> float : mismatch
        check(!Validate(g).ok, "edge type mismatch (float2 -> float) is rejected");
    }

    // Unconnected PBROutput inputs -> VALID, defaults applied.
    {
        Graph g;
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {out};
        check(Validate(g).ok, "bare PBROutput (all inputs unconnected) is VALID");
        PbrResult r = Evaluate(g, 0.3f, 0.7f, 0.5f, StubTex);
        check(approx(r.baseColor[0], 1) && approx(r.baseColor[1], 1) && approx(r.baseColor[2], 1),
              "unconnected baseColor defaults to 1");
        check(approx(r.metallic, 0.0f), "unconnected metallic defaults to 0");
        check(approx(r.roughness, 1.0f), "unconnected roughness defaults to 1");
        check(approx(r.emissive[0], 0) && approx(r.emissive[1], 0) && approx(r.emissive[2], 0),
              "unconnected emissive defaults to 0");
    }

    // ============================ 2. CPU INTERPRETER PARITY ===================================
    // Per-node primitives match hand-computed values.
    {
        Value a; a.count = 3; a.v = {0.2f, 0.4f, 0.6f, 0};
        Value b; b.count = 3; b.v = {0.5f, 0.5f, 2.0f, 0};
        Value m = EvalMultiply(a, b);
        check(approx(m.v[0], 0.1f) && approx(m.v[1], 0.2f) && approx(m.v[2], 1.2f), "EvalMultiply component-wise");
        Value s = EvalAdd(a, b);
        check(approx(s.v[0], 0.7f) && approx(s.v[1], 0.9f) && approx(s.v[2], 2.6f), "EvalAdd component-wise");
        Value l = EvalLerp(a, b, 0.25f);
        // lerp(0.2,0.5,0.25)=0.275 ; lerp(0.4,0.5,0.25)=0.425 ; lerp(0.6,2.0,0.25)=0.95
        check(approx(l.v[0], 0.275f) && approx(l.v[1], 0.425f) && approx(l.v[2], 0.95f), "EvalLerp mix");
        // Fresnel: pow(1 - saturate(NoV), power). NoV=0.25, power=3 -> 0.75^3 = 0.421875.
        check(approx(EvalFresnel(0.25f, 3.0f), 0.421875f), "EvalFresnel pow(1-NoV,power)");
        check(approx(EvalFresnel(1.0f, 3.0f), 0.0f), "EvalFresnel head-on -> 0");
    }

    // Whole-graph interpreter parity on the showcase graph at a sample point. Hand-compute:
    //   uv=(0.3,0.7); tex(checker)=(0.3,0.7,0.5); tint=(0.9,0.2,0.1); NoV=0.25 power=3 -> f=0.421875
    //   baseColor = lerp(tex.rgb, tint, f):
    //     r=lerp(0.3,0.9,0.421875)=0.3+0.6*0.421875=0.553125
    //     g=lerp(0.7,0.2,0.421875)=0.7-0.5*0.421875=0.4890625
    //     b=lerp(0.5,0.1,0.421875)=0.5-0.4*0.421875=0.33125
    //   metallic=0.0 roughness=0.35 emissive default 0.
    {
        Graph g = MakeShowcaseGraph();
        check(Validate(g).ok, "showcase graph valid (parity)");
        PbrResult r = Evaluate(g, 0.3f, 0.7f, 0.25f, StubTex);
        check(approx(r.baseColor[0], 0.553125f), "showcase baseColor.r matches hand-compute");
        check(approx(r.baseColor[1], 0.4890625f), "showcase baseColor.g matches hand-compute");
        check(approx(r.baseColor[2], 0.33125f), "showcase baseColor.b matches hand-compute");
        check(approx(r.metallic, 0.0f), "showcase metallic == 0");
        check(approx(r.roughness, 0.35f), "showcase roughness == 0.35");
    }

    // ============================ 3. CODEGEN DETERMINISM / STRUCTURE ===========================
    {
        Graph g = MakeShowcaseGraph();
        std::string h1 = GenerateHlsl(g);
        std::string h2 = GenerateHlsl(g);
        check(h1 == h2, "GenerateHlsl is deterministic (byte-identical across runs)");
        check(h1.find("// ERROR") == std::string::npos, "GenerateHlsl on a valid graph has no error marker");

        // Structural: it must include the shared PBR core, call hfShadePBR, and declare a temporary
        // for each evaluated node in TOPOLOGICAL order (the temp for an input appears before the temp
        // that consumes it).
        check(h1.find("pbr_core.hlsli") != std::string::npos, "generated HLSL includes pbr_core.hlsli");
        check(h1.find("hfShadePBR") != std::string::npos, "generated HLSL calls hfShadePBR");
        // One temporary per non-sink node, named n<id>.
        check(h1.find("n2") != std::string::npos, "temp for TextureSample node (n2) present");
        check(h1.find("n5") != std::string::npos, "temp for Lerp node (n5) present");
        check(h1.find("lerp(") != std::string::npos, "Lerp emits lerp()");
        check(h1.find("Sample(") != std::string::npos, "TextureSample emits Sample()");
        // Topo order: the texture temp (n2) and tint temp (n3) and fresnel temp (n4) all appear
        // before the lerp temp (n5) that consumes them.
        size_t pn2 = h1.find("float4 n2"); // texture sample temp (float4)
        size_t pn5 = h1.find("n5 ="); // lerp assignment
        check(pn2 != std::string::npos && pn5 != std::string::npos && pn2 < pn5,
              "topological order: TextureSample temp precedes the Lerp temp that consumes it");
        // The same varyings/bindings as lit_pbr: the PSInput (with the TEXCOORD0 uv varying) + the
        // base-color binding come from the shared pbr_core.hlsli include; the generated body consumes
        // the uv varying (i.uv) and the PSInput entry parameter.
        check(h1.find("PSInput i") != std::string::npos, "generated HLSL takes the shared PSInput");
        check(h1.find("i.uv") != std::string::npos, "generated HLSL reads the TEXCOORD0 uv varying (i.uv)");
    }

    // Codegen on an invalid graph yields an error marker (loud failure).
    {
        Graph g;  // empty: no PBROutput.
        std::string h = GenerateHlsl(g);
        check(h.find("// ERROR") == 0, "GenerateHlsl on an invalid graph begins with // ERROR");
    }

    // ============================ 4. LOADER ROUND-TRIP =========================================
    {
        // A compact in-memory document exercising the schema.
        const char* json = R"JSON({
          "name": "test_mat",
          "nodes": [
            { "id": 1, "type": "UV" },
            { "id": 2, "type": "TextureSample", "texture": "checker" },
            { "id": 3, "type": "Constant", "value": [0.9, 0.2, 0.1, 1.0], "outType": "float4" },
            { "id": 4, "type": "Fresnel", "power": 3.0 },
            { "id": 5, "type": "Lerp" },
            { "id": 6, "type": "Constant", "value": [0.0, 0, 0, 0], "outType": "float" },
            { "id": 99, "type": "PBROutput" }
          ],
          "edges": [
            { "from": 1, "to": 2, "port": "uv" },
            { "from": 2, "to": 5, "port": "a" },
            { "from": 3, "to": 5, "port": "b" },
            { "from": 4, "to": 5, "port": "t" },
            { "from": 5, "to": 99, "port": "baseColor" },
            { "from": 6, "to": 99, "port": "metallic" }
          ]
        })JSON";
        LoadResult lr = LoadGraphFromJson(json);
        check(lr.ok, "showcase-shaped JSON loads");
        check(lr.name == "test_mat", "loader reads the material name");
        if (lr.ok) {
            check(lr.graph.nodes.size() == 7, "loaded 7 nodes");
            check(lr.graph.edges.size() == 6, "loaded 6 edges");
            check(Validate(lr.graph).ok, "loaded graph validates");
            std::string h = GenerateHlsl(lr.graph);
            check(h.find("// ERROR") == std::string::npos, "loaded graph codegens cleanly");
            // Loaded Fresnel power round-trips into the codegen (pow(..., 3) appears).
            const Node* f = nullptr;
            for (auto& n : lr.graph.nodes) if (n.kind == NodeKind::Fresnel) f = &n;
            check(f && approx(f->power, 3.0f), "loaded Fresnel power == 3");
        }
        // Malformed JSON -> error, no crash.
        check(!LoadGraphFromJson("{ not json").ok, "malformed JSON is rejected gracefully");
        // Unknown node type -> error.
        check(!LoadGraphFromJson(R"({"nodes":[{"id":1,"type":"Bogus"}],"edges":[]})").ok,
              "unknown node type is rejected");
    }

    if (g_fail == 0) { std::printf("shader_graph_test OK\n"); return 0; }
    std::printf("shader_graph_test: %d failures\n", g_fail);
    return 1;
}
