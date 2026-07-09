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
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

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
    HF_TEST_MAIN_INIT();
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

    // ============================ 5. SLICE AZ NEW NODES =======================================
    // New node types: Swizzle, MakeFloat3, MakeFloat4, Dot, Normalize, Power, OneMinus, Saturate.
    // The CPU interpreter and the codegen share these per-node formula definitions, so the parity
    // checks below PIN the shader semantics. Helper to build a single-source graph driving one
    // PBROutput port through one node-under-test fed by a Constant (or two).

    // --- Per-node primitive parity (hand-computed) ---------------------------------------------
    {
        // Swizzle: mask "x" over a float4 -> component 0 as a scalar (count 1).
        Value v4; v4.count = 4; v4.v = {0.11f, 0.22f, 0.33f, 0.44f};
        Value sx = EvalSwizzle(v4, "x");
        check(sx.count == 1 && approx(sx.v[0], 0.11f), "Swizzle \"x\" on float4 -> component 0 scalar");
        // "xyz" -> float3 prefix.
        Value sxyz = EvalSwizzle(v4, "xyz");
        check(sxyz.count == 3 && approx(sxyz.v[0], 0.11f) && approx(sxyz.v[1], 0.22f) &&
              approx(sxyz.v[2], 0.33f), "Swizzle \"xyz\" -> float3 prefix");
        // "ww" -> float2 of the w component.
        Value sww = EvalSwizzle(v4, "ww");
        check(sww.count == 2 && approx(sww.v[0], 0.44f) && approx(sww.v[1], 0.44f),
              "Swizzle \"ww\" -> float2 of w");
        // rgba aliases map to xyzw.
        Value srg = EvalSwizzle(v4, "rg");
        check(srg.count == 2 && approx(srg.v[0], 0.11f) && approx(srg.v[1], 0.22f),
              "Swizzle \"rg\" alias -> xy");

        // MakeFloat3 / MakeFloat4.
        Value mf3 = EvalMakeFloat({1.0f, 2.0f, 3.0f, 0.0f}, 3);
        check(mf3.count == 3 && approx(mf3.v[0], 1) && approx(mf3.v[1], 2) && approx(mf3.v[2], 3),
              "MakeFloat3 constructs (1,2,3)");
        Value mf4 = EvalMakeFloat({4.0f, 5.0f, 6.0f, 7.0f}, 4);
        check(mf4.count == 4 && approx(mf4.v[3], 7), "MakeFloat4 constructs (...,7)");

        // Dot([1,2,3],[4,5,6]) = 4+10+18 = 32.
        Value a; a.count = 3; a.v = {1, 2, 3, 0};
        Value b; b.count = 3; b.v = {4, 5, 6, 0};
        Value d = EvalDot(a, b);
        check(d.count == 1 && approx(d.v[0], 32.0f), "Dot([1,2,3],[4,5,6]) == 32");

        // Normalize([3,0,4]) -> [0.6,0,0.8].
        Value n; n.count = 3; n.v = {3, 0, 4, 0};
        Value nn = EvalNormalize(n);
        check(nn.count == 3 && approx(nn.v[0], 0.6f) && approx(nn.v[1], 0.0f) && approx(nn.v[2], 0.8f),
              "Normalize([3,0,4]) == [0.6,0,0.8]");

        // Power(2,3) == 8 (component-wise; here scalar base/exponent).
        Value pb; pb.count = 1; pb.v = {2, 0, 0, 0};
        Value pe; pe.count = 1; pe.v = {3, 0, 0, 0};
        Value pw = EvalPower(pb, pe);
        check(pw.count == 1 && approx(pw.v[0], 8.0f), "Power(2,3) == 8");

        // OneMinus(0.25) == 0.75.
        Value om = EvalOneMinus(Value{{0.25f, 0, 0, 0}, 1});
        check(approx(om.v[0], 0.75f), "OneMinus(0.25) == 0.75");

        // Saturate clamps to [0,1] component-wise.
        Value sat = EvalSaturate(Value{{1.5f, -0.5f, 0.5f, 2.0f}, 4});
        check(approx(sat.v[0], 1.0f) && approx(sat.v[1], 0.0f) && approx(sat.v[2], 0.5f) &&
              approx(sat.v[3], 1.0f), "Saturate clamps to [0,1] (1.5->1)");
    }

    // --- Validation: new type rules ------------------------------------------------------------
    {
        // Swizzle out-of-range mask char rejected. Feed a float2 (UV) into a Swizzle masking "z"
        // (index 2 is out of a float2's 0..1 range).
        Graph g;
        Node uv; uv.id = 1; uv.kind = NodeKind::UV;                  // float2
        Node sw; sw.id = 2; sw.kind = NodeKind::Swizzle; sw.swizzle = "z";
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {uv, sw, out};
        g.edges = { {1, 2, "in"}, {2, 99, "metallic"} };
        check(!Validate(g).ok, "Swizzle mask char out of input range is rejected");
    }
    {
        // Swizzle "x" on a float2 -> scalar, valid into metallic.
        Graph g;
        Node uv; uv.id = 1; uv.kind = NodeKind::UV;                  // float2
        Node sw; sw.id = 2; sw.kind = NodeKind::Swizzle; sw.swizzle = "x";
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {uv, sw, out};
        g.edges = { {1, 2, "in"}, {2, 99, "metallic"} };
        check(Validate(g).ok, "Swizzle \"x\" (float2->scalar) into metallic validates");
        check(OutputType(g, g.nodes[1]) == Type::Float, "Swizzle \"x\" output type is float");
    }
    {
        // Empty / overlong mask rejected.
        Graph g;
        Node uv; uv.id = 1; uv.kind = NodeKind::UV;
        Node sw; sw.id = 2; sw.kind = NodeKind::Swizzle; sw.swizzle = "xyzwx";  // len 5 > 4
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {uv, sw, out};
        g.edges = { {1, 2, "in"}, {2, 99, "baseColor"} };
        check(!Validate(g).ok, "Swizzle mask length > 4 is rejected");
    }
    {
        // MakeFloat3 arity: missing one input is rejected (needs all 3 of x/y/z connected).
        Graph g;
        Node cx; cx.id = 1; cx.kind = NodeKind::Constant; cx.value = {1,0,0,0}; cx.outType = Type::Float;
        Node cy; cy.id = 2; cy.kind = NodeKind::Constant; cy.value = {2,0,0,0}; cy.outType = Type::Float;
        Node mk; mk.id = 3; mk.kind = NodeKind::MakeFloat3;
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {cx, cy, mk, out};
        g.edges = { {1, 3, "x"}, {2, 3, "y"}, {3, 99, "baseColor"} };   // missing z
        check(!Validate(g).ok, "MakeFloat3 with a missing input (arity mismatch) is rejected");
        // All three connected -> valid float3.
        Node cz; cz.id = 4; cz.kind = NodeKind::Constant; cz.value = {3,0,0,0}; cz.outType = Type::Float;
        g.nodes.push_back(cz);
        g.edges.push_back({4, 3, "z"});
        check(Validate(g).ok, "MakeFloat3 with all three scalar inputs validates");
        check(OutputType(g, g.nodes[2]) == Type::Float3, "MakeFloat3 output type is float3");
    }
    {
        // MakeFloat3 rejects a NON-scalar input (a float2 into x).
        Graph g;
        Node uv; uv.id = 1; uv.kind = NodeKind::UV;                  // float2
        Node cy; cy.id = 2; cy.kind = NodeKind::Constant; cy.value = {2,0,0,0}; cy.outType = Type::Float;
        Node cz; cz.id = 3; cz.kind = NodeKind::Constant; cz.value = {3,0,0,0}; cz.outType = Type::Float;
        Node mk; mk.id = 4; mk.kind = NodeKind::MakeFloat3;
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {uv, cy, cz, mk, out};
        g.edges = { {1, 4, "x"}, {2, 4, "y"}, {3, 4, "z"}, {4, 99, "baseColor"} };
        check(!Validate(g).ok, "MakeFloat3 with a non-scalar (float2) input is rejected");
    }
    {
        // Dot operand size match: float3 . float2 rejected.
        Graph g;
        Node c3; c3.id = 1; c3.kind = NodeKind::Constant; c3.value = {1,2,3,0}; c3.outType = Type::Float3;
        Node uv; uv.id = 2; uv.kind = NodeKind::UV;                  // float2
        Node dt; dt.id = 3; dt.kind = NodeKind::Dot;
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {c3, uv, dt, out};
        g.edges = { {1, 3, "a"}, {2, 3, "b"}, {3, 99, "metallic"} };
        check(!Validate(g).ok, "Dot with mismatched operand sizes is rejected");
        check(OutputType(g, g.nodes[2]) == Type::Float, "Dot output type is float");
    }

    // --- Codegen structure on the new nodes (showcase3) ----------------------------------------
    {
        LoadResult lr = LoadGraphFromFile(std::string(HF_MAT3_JSON));
        check(lr.ok, "showcase3.mat.json loads + validates");
        if (lr.ok) {
            std::string h1 = GenerateHlsl(lr.graph);
            std::string h2 = GenerateHlsl(lr.graph);
            check(h1 == h2, "showcase3 codegen is deterministic (byte-identical)");
            check(h1.find("// ERROR") == std::string::npos, "showcase3 codegens cleanly");
            // Structural asserts: the new nodes emit the obvious HLSL intrinsics.
            check(h1.find(".x") != std::string::npos, "showcase3 emits a .x swizzle");
            check(h1.find("saturate(") != std::string::npos, "showcase3 emits saturate()");
            check(h1.find("pow(") != std::string::npos, "showcase3 emits pow()");
            check(h1.find("float3(") != std::string::npos, "showcase3 emits a float3() constructor");
            check(h1.find("1.0 - ") != std::string::npos, "showcase3 emits 1.0 - x (OneMinus)");
            check(h1.find("hfShadePBR") != std::string::npos, "showcase3 calls hfShadePBR");
        }
    }

    // --- Loader round-trip of the new node params ----------------------------------------------
    {
        const char* json = R"JSON({
          "name": "nodes_az",
          "nodes": [
            { "id": 1, "type": "TextureSample", "texture": "checker" },
            { "id": 2, "type": "Swizzle", "swizzle": "x" },
            { "id": 3, "type": "Swizzle", "swizzle": "y" },
            { "id": 4, "type": "OneMinus" },
            { "id": 5, "type": "Constant", "value": [0.2, 0.3, 0.4, 1.0], "outType": "float3" },
            { "id": 99, "type": "PBROutput" }
          ],
          "edges": [
            { "from": 1, "to": 2, "port": "in" },
            { "from": 1, "to": 3, "port": "in" },
            { "from": 3, "to": 4, "port": "in" },
            { "from": 2, "to": 99, "port": "metallic" },
            { "from": 4, "to": 99, "port": "roughness" },
            { "from": 5, "to": 99, "port": "baseColor" }
          ]
        })JSON";
        LoadResult lr = LoadGraphFromJson(json);
        check(lr.ok, "AZ-node JSON loads + validates");
        if (lr.ok) {
            const Node* sw = nullptr;
            for (auto& n : lr.graph.nodes) if (n.kind == NodeKind::Swizzle && n.id == 2) sw = &n;
            check(sw && sw->swizzle == "x", "loader reads the swizzle mask param");
            std::string h = GenerateHlsl(lr.graph);
            check(h.find("// ERROR") == std::string::npos, "AZ-node graph codegens cleanly");
        }
    }

    // ============================ 6. SLICE BE — NORMALMAP NODE ================================
    // New tangent-space NormalMap node (sample a normal texture -> decode(c)=c*2-1 -> normalize) and
    // a 5th PBROutput input `normal` (tangent-space float3, DEFAULT (0,0,1) = no perturbation). The
    // decode math is shared by the CPU interpreter + the codegen; the default-(0,0,1) invariant
    // protects the existing material goldens (an UNCONNECTED normal emits the SAME geometric-normal
    // codegen as pre-BE: NO TBN block).

    // --- NormalMap decode parity (shared formula decode(c) = c*2 - 1) ---------------------------
    {
        // decode(0.5)==0, decode(1.0)==1, decode(0.0)==-1 (component-wise).
        check(approx(EvalNormalDecode(0.5f), 0.0f), "NormalMap decode(0.5) == 0");
        check(approx(EvalNormalDecode(1.0f), 1.0f), "NormalMap decode(1.0) == 1");
        check(approx(EvalNormalDecode(0.0f), -1.0f), "NormalMap decode(0.0) == -1");

        // A FLAT normal-map texel (0.5,0.5,1.0) decodes to (0,0,1) (no perturbation) and the node
        // output (normalize(decode(texel))) is exactly (0,0,1).
        Value flat = EvalNormalMap({0.5f, 0.5f, 1.0f, 1.0f});
        check(flat.count == 3 && approx(flat.v[0], 0.0f) && approx(flat.v[1], 0.0f) &&
              approx(flat.v[2], 1.0f), "NormalMap flat texel (0.5,0.5,1) -> (0,0,1)");

        // Output is normalized (unit length) for a non-flat texel. Texel (1.0,1.0,0.5):
        // decode -> (1,1,0); normalize -> (0.7071,0.7071,0).
        Value perturbed = EvalNormalMap({1.0f, 1.0f, 0.5f, 1.0f});
        float len = std::sqrt(perturbed.v[0] * perturbed.v[0] + perturbed.v[1] * perturbed.v[1] +
                              perturbed.v[2] * perturbed.v[2]);
        check(perturbed.count == 3 && approx(len, 1.0f), "NormalMap output is normalized (unit length)");
        check(approx(perturbed.v[0], 0.70710678f) && approx(perturbed.v[1], 0.70710678f) &&
              approx(perturbed.v[2], 0.0f), "NormalMap (1,1,0.5) -> normalized (0.707,0.707,0)");
    }

    // --- A standalone NormalMap node graph: validates; the node outputs float3. -----------------
    {
        Graph g;
        Node nm; nm.id = 1; nm.kind = NodeKind::NormalMap; nm.texture = "normalmap";
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {nm, out};
        g.edges = { {1, 99, "normal"} };
        check(Validate(g).ok, "NormalMap -> PBROutput.normal validates");
        check(OutputType(g, g.nodes[0]) == Type::Float3, "NormalMap output type is float3");
        // A float2 (UV) into PBROutput.normal (float3) is a type error.
        Graph bad;
        Node uv; uv.id = 1; uv.kind = NodeKind::UV;
        Node out2; out2.id = 99; out2.kind = NodeKind::PBROutput;
        bad.nodes = {uv, out2};
        bad.edges = { {1, 99, "normal"} };
        check(!Validate(bad).ok, "float2 -> PBROutput.normal is rejected (type mismatch)");
    }

    // --- PBROutput.normal DEFAULT: unconnected -> (0,0,1) AND byte-identical pre-BE codegen ------
    // CRITICAL invariance: a graph WITHOUT a normal connection must generate the SAME geometric-
    // normal shader as before BE (no TBN block, calls hfShadePBR not hfShadePBRN), so the existing
    // mat_showcase / mat_showcase2 / mat_showcase3 goldens stay byte-identical.
    {
        Graph g = MakeShowcaseGraph();  // no `normal` edge.
        std::string h = GenerateHlsl(g);
        check(h.find("// ERROR") == std::string::npos, "showcase (no normal) codegens cleanly");
        // NO TBN block emitted when normal is unconnected (protects existing goldens).
        check(h.find("wtangent") == std::string::npos,
              "unconnected normal: NO tangent/TBN block emitted (geometric-normal path)");
        check(h.find("hfShadePBRN") == std::string::npos,
              "unconnected normal: calls hfShadePBR (NOT the world-normal overload)");
        check(h.find("hfShadePBR(i,") != std::string::npos,
              "unconnected normal: emits the original hfShadePBR(i, ...) call");
    }

    // --- CONNECTED case: the normalmap material codegens a TBN transform + perturbed normal. -----
    {
        LoadResult lr = LoadGraphFromFile(std::string(HF_MATNORMAL_JSON));
        check(lr.ok, "normalmap.mat.json loads + validates");
        if (lr.ok) {
            std::string h1 = GenerateHlsl(lr.graph);
            std::string h2 = GenerateHlsl(lr.graph);
            check(h1 == h2, "normalmap codegen is deterministic (byte-identical)");
            check(h1.find("// ERROR") == std::string::npos, "normalmap codegens cleanly");
            // The NormalMap node samples the normal-map texture + decodes.
            check(h1.find("gNormalMap.Sample") != std::string::npos,
                  "NormalMap node samples gNormalMap");
            // TBN transform present (Gram-Schmidt against the interpolated tangent, like lit.frag).
            check(h1.find("i.wtangent") != std::string::npos, "connected normal: uses the wtangent varying");
            check(h1.find("float3x3") != std::string::npos, "connected normal: builds a TBN matrix");
            // The perturbed world normal feeds lighting via the world-normal overload.
            check(h1.find("hfShadePBRN") != std::string::npos,
                  "connected normal: feeds the perturbed normal into hfShadePBRN");
        }
    }

    // ============================ 7. SLICE MG1 — NODE-LIBRARY DEPTH ===========================
    // The "surpass UE5 material depth" batch: procedural noise (Value/Perlin/Voronoi/FBM), ~18
    // math/utility nodes, UV animation, a material-LAYER blend, and reusable material FUNCTIONS —
    // all deterministic + cross-platform. The pins here prove: (a) the existing 17-node codegen is
    // BYTE-FROZEN (no perturbation, no noise include leaks in), (b) the noise is deterministic + the
    // CPU value == the documented formula the shader emits, (c) functions inline correctly + reuse,
    // (d) each new node emits pinned codegen + a rich whole-graph digest (MSVC==clang comparable).

    // Stable FNV-1a-64 over the codegen text (compiler-independent, unlike std::hash) so the printed
    // digest is directly comparable between the MSVC and clang builds of this test.
    auto fnv1a = [](const std::string& s) -> uint64_t {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
        return h;
    };

    // --- (a) FROZEN: the existing 17-node showcase codegens UNCHANGED (no MG1 perturbation). --------
    {
        Graph g = MakeShowcaseGraph();
        std::string h = GenerateHlsl(g);
        check(h.find("material_noise.hlsli") == std::string::npos,
              "FROZEN: noise-free graph does NOT emit the MG1 noise include");
        check(h.find("hfFbm") == std::string::npos && h.find("hfValueNoise") == std::string::npos,
              "FROZEN: noise-free graph emits no MG1 noise intrinsics");
        // The GenerateHlsl(g, {}) library overload must equal the frozen GenerateHlsl(g) byte-for-byte.
        check(GenerateHlsl(g, FunctionLibrary{}) == h,
              "FROZEN: GenerateHlsl(g, {}) == GenerateHlsl(g) (empty-library transparency)");
    }

    // --- (b) NOISE: pinned CPU sample values + determinism + grid-corner exactness. -----------------
    {
        // Two runs identical (deterministic integer hash, no global state).
        check(EvalValueNoise(2.5f, 1.5f) == EvalValueNoise(2.5f, 1.5f), "ValueNoise is deterministic");
        check(EvalFbm(2.5f, 1.5f, 5) == EvalFbm(2.5f, 1.5f, 5), "FBM is deterministic");
        // Pinned integer-hash values (the CPU==shader anchor: HfHash2f is bit-exact CPU/GPU).
        check(approx(HfHash2f(0, 0), 0.12865025f), "HfHash2f(0,0) pinned");
        check(approx(HfHash2f(1, 2), 0.75222260f), "HfHash2f(1,2) pinned");
        check(approx(HfHash2f(-1, 5), 0.88656968f), "HfHash2f(-1,5) pinned (negative-cell bias works)");
        // Value noise AT A GRID CORNER equals the cell hash exactly (the exact CPU==shader point).
        check(approx(EvalValueNoise(0.0f, 0.0f), HfHash2f(0, 0)),
              "ValueNoise at a grid corner == HfHash2f (bit-exact CPU==shader anchor)");
        // Pinned interior samples (match the shader to float tolerance = the visresolve bar).
        check(approx(EvalValueNoise(2.5f, 1.5f), 0.48297733f), "ValueNoise(2.5,1.5) pinned");
        check(approx(EvalPerlin(2.5f, 1.5f), 0.48256439f), "Perlin(2.5,1.5) pinned");
        check(approx(EvalVoronoi(2.5f, 1.5f), 0.26895037f), "Voronoi(2.5,1.5) pinned");
        check(approx(EvalFbm(2.5f, 1.5f, 5), 0.52771342f), "FBM(2.5,1.5,octaves=5) pinned");
        // A single-octave FBM reduces to plain value noise (sanity of the fractal sum).
        check(approx(EvalFbm(2.5f, 1.5f, 1), EvalValueNoise(2.5f, 1.5f)),
              "FBM octaves=1 == ValueNoise (fractal-sum sanity)");
        // Noise ranges are sane ([0,1] for value/perlin/fbm).
        for (int k = 0; k < 32; ++k) {
            float x = 0.137f * (float)k, y = 0.291f * (float)k;
            float v = EvalValueNoise(x, y), p = EvalPerlin(x, y), f = EvalFbm(x, y, 5);
            check(v >= 0.0f && v <= 1.0f && p >= 0.0f && p <= 1.0f && f >= 0.0f && f <= 1.0f,
                  "noise stays within [0,1]");
        }
    }

    // --- (b2) NOISE codegen emits the shared include + the exact intrinsic call the CPU replicates. --
    {
        Graph g;
        Node uv;  uv.id = 1;  uv.kind = NodeKind::UV;
        Node fbm; fbm.id = 2; fbm.kind = NodeKind::FBM; fbm.octaves = 4;
        Node out; out.id = 99; out.kind = NodeKind::PBROutput;
        g.nodes = {uv, fbm, out};
        g.edges = { {1, 2, "p"}, {2, 99, "roughness"} };
        check(Validate(g).ok, "UV->FBM->roughness validates");
        std::string h = GenerateHlsl(g);
        check(h.find("material_noise.hlsli") != std::string::npos,
              "a noise graph emits the shared material_noise.hlsli include");
        check(h.find("hfFbm(n1, 4)") != std::string::npos,
              "FBM emits hfFbm(<p>, octaves) — the exact call the CPU EvalFbm replicates");
    }

    // --- (c) MATERIAL FUNCTIONS: a reusable subgraph inlines into a parent; reuse works. ------------
    {
        // Function "warmtint": out = in.c * float3(1.2, 1.0, 0.8).
        Graph fn;
        Node fi; fi.id = 1; fi.kind = NodeKind::FunctionInput; fi.texture = "c"; fi.outType = Type::Float3;
        Node fc; fc.id = 2; fc.kind = NodeKind::Constant; fc.value = {1.2f, 1.0f, 0.8f, 1.0f}; fc.outType = Type::Float3;
        Node fm; fm.id = 3; fm.kind = NodeKind::Multiply;
        Node fo; fo.id = 4; fo.kind = NodeKind::FunctionOutput;
        fn.nodes = {fi, fc, fm, fo};
        fn.edges = { {1, 3, "a"}, {2, 3, "b"}, {3, 4, "in"} };
        FunctionLibrary lib;
        lib.Add("warmtint", fn);

        // Parent calls warmtint TWICE (baseColor + emissive) -> proves REUSE (two inlined copies).
        Graph parent;
        Node c1; c1.id = 1; c1.kind = NodeKind::Constant; c1.value = {0.5f, 0.4f, 0.3f, 1.0f}; c1.outType = Type::Float3;
        Node k1; k1.id = 2; k1.kind = NodeKind::FunctionCall; k1.texture = "warmtint";
        Node c2; c2.id = 3; c2.kind = NodeKind::Constant; c2.value = {0.2f, 0.2f, 0.2f, 1.0f}; c2.outType = Type::Float3;
        Node k2; k2.id = 4; k2.kind = NodeKind::FunctionCall; k2.texture = "warmtint";
        Node po; po.id = 99; po.kind = NodeKind::PBROutput;
        parent.nodes = {c1, k1, c2, k2, po};
        parent.edges = { {1, 2, "c"}, {3, 4, "c"}, {2, 99, "baseColor"}, {4, 99, "emissive"} };

        Graph flat = FlattenFunctions(parent, lib);
        // 3 parent-kept (2 consts + sink) + 2 calls × 2 interior func nodes (Const + Multiply) = 7.
        check(flat.nodes.size() == 7,
              "FlattenFunctions inlines both calls (2 parent consts + sink + 2×2 func interior = 7 nodes)");
        bool anyCall = false;
        for (auto& n : flat.nodes) if (n.kind == NodeKind::FunctionCall) anyCall = true;
        check(!anyCall, "no FunctionCall survives flattening");
        ValidationResult vr = Validate(flat);
        check(vr.ok, "flattened graph validates");
        std::string h = GenerateHlsl(parent, lib);
        check(h.find("// ERROR") == std::string::npos, "function-call graph codegens cleanly");
        check(h.find("unexpanded function") == std::string::npos, "no unexpanded-function error marker");
        // Two inlined multiplies by the warm tint constant (reuse -> two occurrences).
        size_t f1 = h.find("float3(1.2");
        size_t f2 = (f1 == std::string::npos) ? std::string::npos : h.find("float3(1.2", f1 + 1);
        check(f1 != std::string::npos && f2 != std::string::npos,
              "the reused function body appears TWICE (both call sites inlined)");
        // Determinism.
        check(GenerateHlsl(parent, lib) == h, "function codegen is deterministic");
        // A call-free graph flattens to ITSELF (identity) -> frozen codegen.
        Graph noCall = MakeShowcaseGraph();
        Graph flatNoCall = FlattenFunctions(noCall, lib);
        check(GenerateHlsl(flatNoCall) == GenerateHlsl(noCall),
              "FlattenFunctions is identity on a call-free graph (frozen invariant)");
    }

    // --- (d) NEW NODE CODEGEN: each new math/utility node emits its pinned intrinsic. ----------------
    {
        // Build a kitchen-sink graph touching noise + ~20 math/anim/layer nodes; assert the intrinsics
        // + pin the whole-graph codegen digest (deterministic; MSVC==clang comparable).
        Graph g;
        auto add = [&](int id, NodeKind k) { Node n; n.id = id; n.kind = k; g.nodes.push_back(n); return &g.nodes.back(); };
        Node uv; uv.id = 1; uv.kind = NodeKind::UV; g.nodes.push_back(uv);
        Node sc; sc.id = 2; sc.kind = NodeKind::Constant; sc.value = {3, 3, 0, 0}; sc.outType = Type::Float2; g.nodes.push_back(sc);
        add(3, NodeKind::Multiply);                                  // p = uv*scale (float2)
        Node pan; pan.id = 4; pan.kind = NodeKind::Panner; pan.speed = 0.2f; g.nodes.push_back(pan);
        Node rot; rot.id = 5; rot.kind = NodeKind::Rotator; rot.speed = 0.5f; g.nodes.push_back(rot);
        add(6, NodeKind::ValueNoise);
        add(7, NodeKind::PerlinNoise);
        add(8, NodeKind::VoronoiNoise);
        Node fb; fb.id = 9; fb.kind = NodeKind::FBM; fb.octaves = 3; g.nodes.push_back(fb);
        add(10, NodeKind::Sin); add(11, NodeKind::Cos); add(12, NodeKind::Abs);
        add(13, NodeKind::Floor); add(14, NodeKind::Ceil); add(15, NodeKind::Frac);
        add(16, NodeKind::Sqrt); add(17, NodeKind::Sign);
        add(18, NodeKind::Min); add(19, NodeKind::Max); add(20, NodeKind::Step);
        Node c5; c5.id = 21; c5.kind = NodeKind::Constant; c5.value = {0.5f, 0, 0, 0}; c5.outType = Type::Float; g.nodes.push_back(c5);
        add(22, NodeKind::Modulo); add(23, NodeKind::Distance); add(24, NodeKind::Reflect);
        Node cl; cl.id = 25; cl.kind = NodeKind::Clamp; cl.lo = 0.0f; cl.hi = 1.0f; g.nodes.push_back(cl);
        Node ss; ss.id = 26; ss.kind = NodeKind::Smoothstep; ss.lo = 0.1f; ss.hi = 0.9f; g.nodes.push_back(ss);
        Node rm; rm.id = 27; rm.kind = NodeKind::Remap; rm.lo = 0; rm.hi = 1; rm.outLo = 0; rm.outHi = 2; g.nodes.push_back(rm);
        add(28, NodeKind::Time);
        Node ba; ba.id = 29; ba.kind = NodeKind::Constant; ba.value = {0.1f, 0.2f, 0.3f, 1}; ba.outType = Type::Float3; g.nodes.push_back(ba);
        Node bt; bt.id = 30; bt.kind = NodeKind::Constant; bt.value = {0.8f, 0.7f, 0.6f, 1}; bt.outType = Type::Float3; g.nodes.push_back(bt);
        add(31, NodeKind::BlendLayer);
        Node out; out.id = 99; out.kind = NodeKind::PBROutput; g.nodes.push_back(out);
        g.edges = {
            {1, 3, "a"}, {2, 3, "b"},
            {3, 4, "uv"}, {3, 5, "uv"},
            {3, 6, "p"}, {4, 7, "p"}, {5, 8, "p"}, {3, 9, "p"},
            {6, 10, "in"}, {7, 11, "in"}, {10, 12, "in"},
            {9, 13, "in"}, {8, 14, "in"}, {9, 15, "in"}, {6, 16, "in"}, {11, 17, "in"},
            {6, 18, "a"}, {7, 18, "b"}, {8, 19, "a"}, {9, 19, "b"}, {6, 20, "a"}, {7, 20, "b"},
            {9, 22, "a"}, {21, 22, "b"}, {3, 23, "a"}, {4, 23, "b"}, {3, 24, "a"}, {5, 24, "b"},
            {6, 25, "in"}, {7, 26, "in"}, {8, 27, "in"},
            {29, 31, "base"}, {30, 31, "top"}, {12, 31, "mask"},
            {31, 99, "baseColor"}, {25, 99, "metallic"}, {26, 99, "roughness"},
        };
        ValidationResult vr = Validate(g);
        check(vr.ok, (std::string("kitchen-sink graph validates: ") + vr.error).c_str());
        std::string h1 = GenerateHlsl(g);
        std::string h2 = GenerateHlsl(g);
        check(h1 == h2, "kitchen-sink codegen is deterministic (byte-identical)");
        check(h1.find("// ERROR") == std::string::npos, "kitchen-sink codegens cleanly");
        // Spot-check the emitted intrinsics for the new node families.
        const char* wants[] = {"hfValueNoise(", "hfPerlin(", "hfVoronoi(", "hfFbm(",
                               "sin(", "cos(", "abs(", "floor(", "ceil(", "frac(", "sqrt(", "sign(",
                               "min(", "max(", "step(", "fmod(", "distance(", "reflect(",
                               "clamp(", "smoothstep(", "f.skyParams.z", "lerp("};
        for (const char* w : wants)
            check(h1.find(w) != std::string::npos, (std::string("kitchen-sink emits ") + w).c_str());
        // The whole-graph codegen digest — pinned for the MSVC==clang cross-compiler proof (the printed
        // value must MATCH between the two builds; the controller diffs the two runs' stdout).
        uint64_t dig = fnv1a(h1);
        std::printf("mg1-codegen-digest kitchen-sink = 0x%016llx (nodes=%zu)\n",
                    (unsigned long long)dig, g.nodes.size());
    }

    // --- (d2) The mg1_marble showcase material (loaded from JSON) codegens cleanly + digest. --------
    {
        LoadResult lr = LoadGraphFromFile(std::string(HF_MG1_JSON));
        check(lr.ok, "mg1_marble.mat.json loads + validates");
        if (lr.ok) {
            std::string h1 = GenerateHlsl(lr.graph);
            std::string h2 = GenerateHlsl(lr.graph);
            check(h1 == h2, "mg1_marble codegen is deterministic");
            check(h1.find("// ERROR") == std::string::npos, "mg1_marble codegens cleanly");
            check(h1.find("material_noise.hlsli") != std::string::npos, "mg1_marble includes the noise header");
            check(h1.find("hfFbm(") != std::string::npos && h1.find("hfValueNoise(") != std::string::npos,
                  "mg1_marble uses FBM + ValueNoise (procedural detail the 17-node set can't express)");
            std::printf("mg1-codegen-digest marble = 0x%016llx (nodes=%zu)\n",
                        (unsigned long long)fnv1a(h1), lr.graph.nodes.size());
        }
    }

    if (g_fail == 0) { std::printf("shader_graph_test OK\n"); return 0; }
    std::printf("shader_graph_test: %d failures\n", g_fail);
    return 1;
}
