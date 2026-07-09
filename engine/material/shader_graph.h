// Hazard Forge — data-driven material / shader graph (Slice AV, Phase 4 opener).
//
// A small node graph, authored as JSON, that compiles to an HLSL fragment feeding the existing PBR
// lighting core. This header defines the PURE-CPU graph model (no vk*/MTL*/Backend symbols — it lives
// strictly above the RHI seam), the validation rules, and a CPU interpreter that evaluates the SAME
// per-node math the codegen emits. The interpreter and the codegen MUST agree on node semantics; the
// per-node formula definitions live in ONE place (the `eval` free functions + the EmitExpr codegen)
// so that the shader_graph_test can pin the semantics independent of the GPU.
//
// MVP node set (YAGNI — exactly these): Constant, UV, TextureSample, Multiply, Add, Lerp, Fresnel,
// PBROutput (the single sink). Ports are typed (float/float2/float3/float4); the graph is a DAG with
// exactly one PBROutput.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace hf::material {

// --- Port value types ---------------------------------------------------------------------------
enum class Type { Float, Float2, Float3, Float4 };

const char* TypeName(Type t);             // "float" / "float2" / "float3" / "float4"
int         ComponentCount(Type t);       // 1 / 2 / 3 / 4

// --- Node kinds ---------------------------------------------------------------------------------
enum class NodeKind {
    Constant,       // param value (float4); output the declared `outType` swizzle of it
    UV,             // output float2 — the interpolated TEXCOORD0
    TextureSample,  // param texture slot name; input UV (float2); output float4
    Multiply,       // a*b component-wise; output matches input type
    Add,            // a+b component-wise; output matches input type
    Lerp,           // lerp(a, b, t); a,b match; t is float (broadcast)
    Fresnel,        // pow(1 - saturate(N·V), power); output float (scalar)
    // --- Slice AZ node expansion -------------------------------------------------------------
    Swizzle,        // mask (e.g. "x"/"xyz"/"ww") over xyzw/rgba of one vector input; out = mask len.
    MakeFloat3,     // 3 scalar inputs (x,y,z) -> float3.
    MakeFloat4,     // 4 scalar inputs (x,y,z,w) -> float4.
    Dot,            // dot(a, b); two same-size vectors -> scalar (float).
    Normalize,      // normalize(x); vector -> same-size unit vector.
    Power,          // pow(base, exponent); component-wise; out matches base.
    OneMinus,       // 1 - x; component-wise; out matches x.
    Saturate,       // clamp(x, 0, 1); component-wise; out matches x.
    // --- Slice BE node -----------------------------------------------------------------------
    NormalMap,      // param normal-texture slot (default "normalmap"); input UV (default interpolated);
                    // output TANGENT-SPACE float3 = normalize(decode(sample(tex, uv))), decode(c)=c*2-1.
    PBROutput,      // SINK: baseColor(float3) metallic(float) roughness(float) emissive(float3) normal(float3)
    // --- Slice MG1: node-library DEPTH (the "surpass UE5 material depth" batch). All ADDITIVE — the 17
    // kinds above + their codegen are BYTE-FROZEN; these append after PBROutput so every existing
    // enumerator keeps its value + existing material goldens codegen bit-identically. -----------------
    // Procedural noise family (input "p" float2 -> scalar float in [0,1]). The noise is a LIVE integer
    // hash (shaders/material_noise.hlsli) the CPU interpreter replicates bit-for-bit at grid corners
    // (uint xor/shift/mul wraps mod 2^32 identically in C++ and HLSL/MSL); interior samples match to
    // float tolerance (the visresolve bar). NO host-baked LUT, NO new binding — MSL-gen-safe.
    ValueNoise,     // hfValueNoise(p): smoothstep-interpolated integer-hash value noise -> [0,1].
    PerlinNoise,    // hfPerlin(p):     gradient/Perlin noise (angle-hash gradients) remapped -> [0,1].
    VoronoiNoise,   // hfVoronoi(p):    cellular F1 nearest-feature distance -> [0,~1.4].
    FBM,            // hfFbm(p,octaves): fractal sum of ValueNoise octaves (param `octaves`) -> [0,1].
    // Math / utility unary (input "in"; output matches in, component-wise).
    Sin, Cos, Abs, Floor, Ceil, Frac, Sqrt, Sign,
    // Math binary (inputs "a","b" wildcard, a==b type; output matches a). Distance -> scalar.
    Min, Max, Step, Modulo, Distance, Reflect,
    // Ranged (input "in"; params). Clamp(lo,hi) / Smoothstep(lo,hi) / Remap(lo,hi -> outLo,outHi).
    Clamp, Smoothstep, Remap,
    // Animation: Time (uniform f.skyParams.z -> scalar); Panner/Rotator animate a UV (input "uv" float2
    // -> float2) by `speed` * time.
    Time, Panner, Rotator,
    // Material LAYER blend: lerp(base, top, mask) — inputs base(float3) top(float3) mask(float) -> float3.
    BlendLayer,
    // Reusable material FUNCTIONS (INLINE-only subgraph reuse; no separate compilation). A function is a
    // Graph with FunctionInput sources (named via `texture`, typed via `outType`) + one FunctionOutput
    // sink; a parent FunctionCall (function named via `texture`) is expanded by FlattenFunctions BEFORE
    // codegen into primitive nodes, so a graph WITHOUT calls flattens to itself (frozen codegen).
    FunctionInput, FunctionOutput, FunctionCall,
};

const char* NodeKindName(NodeKind k);
// Parse a node "type" string from JSON; returns nullopt for an unknown kind.
std::optional<NodeKind> ParseNodeKind(const std::string& s);

// The PBROutput sink input slots, in a FIXED order (also the topo/emit order). Slice BE appends a
// 5th input `normal` (tangent-space float3, default (0,0,1) = no perturbation) — appended LAST so the
// first four keep their order/semantics and unconnected-normal graphs codegen byte-identically.
enum PbrInput { kBaseColor = 0, kMetallic = 1, kRoughness = 2, kEmissive = 3, kNormal = 4,
                kPbrInputCount = 5 };
const char* PbrInputName(int slot);       // "baseColor"/"metallic"/"roughness"/"emissive"/"normal"
Type        PbrInputType(int slot);       // float3/float/float/float3/float3

// --- Node --------------------------------------------------------------------------------------
struct Node {
    int      id = -1;
    NodeKind kind = NodeKind::Constant;

    // Params (kind-specific; unused fields ignored):
    std::array<float, 4> value{0, 0, 0, 0};  // Constant: the float4 value.
    Type        outType = Type::Float4;       // Constant: which swizzle of `value` to output.
    std::string texture;                      // TextureSample: slot name (e.g. "baseColorTex").
    float       power = 1.0f;                 // Fresnel: exponent.
    std::string swizzle;                      // Swizzle: component mask over xyzw/rgba (len 1..4).
    // --- Slice MG1 params (additive; unused by the frozen 17 kinds so their codegen is unchanged) ---
    int         octaves = 5;                  // FBM: number of fractal octaves (1..8).
    float       lo = 0.0f;                    // Clamp/Smoothstep: low edge. Remap: input-range low.
    float       hi = 1.0f;                    // Clamp/Smoothstep: high edge. Remap: input-range high.
    float       outLo = 0.0f;                 // Remap: output-range low.
    float       outHi = 1.0f;                 // Remap: output-range high.
    float       speed = 1.0f;                 // Panner/Rotator: UV animation speed (× time).
    // `texture` doubles as the FUNCTION NAME for FunctionInput (the input param name) + FunctionCall
    // (the referenced function's name). `outType` doubles as a FunctionInput's declared type.
};

// Map a swizzle mask char (xyzw or rgba alias) to a component index 0..3, or -1 if not a valid char.
int SwizzleIndex(char c);

// --- Edge --------------------------------------------------------------------------------------
// An edge connects (fromNode's single output) -> (toNode's named input port). Each node kind has a
// fixed, named set of input ports (see InputPortCount / InputPortName / InputPortType).
struct Edge {
    int         fromNode = -1;
    int         toNode = -1;
    std::string toPort;   // input-port name on toNode
};

// --- Graph -------------------------------------------------------------------------------------
struct Graph {
    std::vector<Node> nodes;
    std::vector<Edge> edges;

    const Node* FindNode(int id) const;
};

// Per-kind input-port introspection (shared by validation, the interpreter, and codegen).
int         InputPortCount(NodeKind k);
const char* InputPortName(NodeKind k, int idx);
Type        InputPortType(NodeKind k, int idx);
// Output type of a node given its connected inputs' types (resolves Multiply/Add/Lerp which match
// their data inputs). Returns the static type for fixed-output kinds.
Type        OutputType(const Graph& g, const Node& n);

// --- Validation --------------------------------------------------------------------------------
struct ValidationResult {
    bool ok = false;
    std::string error;   // human-readable first error (empty when ok).
    explicit operator bool() const { return ok; }
};

// Validate: exactly one PBROutput; DAG (no cycles); every edge's source-output type matches the
// destination port type; edges reference real nodes/ports. Unconnected PBROutput inputs are allowed
// (they fall back to the documented defaults: baseColor=1, metallic=0, roughness=1, emissive=0) — so
// missing PBROutput input connections are NOT an error, but a MISSING PBROutput node IS.
ValidationResult Validate(const Graph& g);

// Topological order of the nodes that feed the (single) PBROutput, PBROutput last. Deterministic:
// ties are broken by ascending node id so the codegen + interpreter are reproducible. Requires a
// valid graph (call Validate first). Returns node ids.
std::vector<int> TopoOrder(const Graph& g);

// --- CPU interpreter (shared node SEMANTICS) ---------------------------------------------------
// A value carrying up to 4 components (the active count is `count`). The per-node math below is the
// SINGLE source of truth for node semantics; the HLSL codegen emits the textual equivalent.
struct Value {
    std::array<float, 4> v{0, 0, 0, 0};
    int count = 1;
};

// Per-node math primitives (the formula definitions shared, in spirit, with the codegen — see
// codegen.cpp's EmitExpr which emits the HLSL form of each of these).
Value EvalMultiply(const Value& a, const Value& b);   // component-wise a*b
Value EvalAdd(const Value& a, const Value& b);        // component-wise a+b
Value EvalLerp(const Value& a, const Value& b, float t);
float EvalFresnel(float NoV, float power);            // pow(1 - saturate(NoV), power)
// --- Slice AZ node primitives (shared with the codegen) ---------------------------------------
Value EvalSwizzle(const Value& in, const std::string& mask);  // gather components per mask -> len(mask)
Value EvalMakeFloat(const std::array<float, 4>& comps, int count);  // build a floatN from scalars
Value EvalDot(const Value& a, const Value& b);        // dot product -> scalar
Value EvalNormalize(const Value& x);                  // x / length(x); same size
Value EvalPower(const Value& base, const Value& exp);  // component-wise pow
Value EvalOneMinus(const Value& x);                   // component-wise 1 - x
Value EvalSaturate(const Value& x);                   // component-wise clamp(x,0,1)
// --- Slice BE: NormalMap decode (shared with the codegen) -------------------------------------
float EvalNormalDecode(float c);                      // decode(c) = c*2 - 1 (single source of truth)
Value EvalNormalMap(const std::array<float, 4>& texel);  // normalize(decode(texel.rgb)) -> float3

// --- Slice MG1: procedural-noise primitives (the SINGLE source of truth, shared bit-for-bit with the
// HLSL/MSL in shaders/material_noise.hlsli). The integer hash uses only uint xor/shift/mul (wraps mod
// 2^32 identically in C++ and shader) + a power-of-two float divide, so CPU==shader is exact at grid
// corners and float-tolerance-close in the interpolated interior (the visresolve bar). -------------
uint32_t HfHashU(uint32_t x);                         // low-bias integer avalanche (Wang-style).
uint32_t HfHash2(int ix, int iy);                     // 2D cell hash.
float    HfHash2f(int ix, int iy);                    // cell hash -> [0,1).
float    EvalValueNoise(float px, float py);          // smoothstep value noise -> [0,1].
float    EvalPerlin(float px, float py);              // gradient/Perlin noise -> [0,1].
float    EvalVoronoi(float px, float py);             // cellular F1 distance -> [0,~1.4].
float    EvalFbm(float px, float py, int octaves);    // fractal sum of ValueNoise -> [0,1].

// --- Slice MG1: reusable material FUNCTIONS -------------------------------------------------------
// A named library of function subgraphs. Lookup is by name (no map iteration in codegen -> stays
// deterministic). Each function Graph has FunctionInput sources + exactly one FunctionOutput sink.
struct FunctionLibrary {
    std::vector<std::pair<std::string, Graph>> funcs;
    const Graph* Find(const std::string& name) const;
    void Add(const std::string& name, Graph g) { funcs.emplace_back(name, std::move(g)); }
};

// Expand every FunctionCall node in `g` by INLINING the referenced function's subgraph (fresh node
// ids, FunctionInput wired to the caller's incoming edge, the function result rewired to the call's
// consumers). Deterministic. If `g` has NO FunctionCall node the input Graph is returned UNCHANGED
// (so a call-free graph codegens byte-identically — the frozen-goldens invariant). Requires that each
// referenced function exists in `lib`; a missing function leaves the call unexpanded (codegen then
// fails loudly on the unknown FunctionCall kind).
Graph FlattenFunctions(const Graph& g, const FunctionLibrary& lib);

// Evaluate the whole graph at a sample point. `uv` feeds UV nodes; `NoV` feeds Fresnel; `sampleTex`
// supplies a stub texture lookup (slot, uv) -> float4 so TextureSample is testable on the CPU. The
// result is the four PBROutput values (baseColor float3, metallic float, roughness float, emissive
// float3), packed with the documented defaults for any unconnected input.
struct PbrResult {
    std::array<float, 3> baseColor{1, 1, 1};
    float metallic = 0.0f;
    float roughness = 1.0f;
    std::array<float, 3> emissive{0, 0, 0};
};

// sampleTex: (textureSlot, u, v) -> rgba. Pass a stub in tests.
PbrResult Evaluate(const Graph& g, float u, float v, float NoV,
                   const std::function<std::array<float, 4>(const std::string&, float, float)>& sampleTex);

}  // namespace hf::material
