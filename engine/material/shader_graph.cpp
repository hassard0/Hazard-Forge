// Hazard Forge — material / shader graph model, validation, and CPU interpreter (Slice AV).
// See shader_graph.h. Pure CPU; no backend symbols.
#include "material/shader_graph.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <unordered_map>

namespace hf::material {

// --- Type helpers -------------------------------------------------------------------------------
const char* TypeName(Type t) {
    switch (t) {
        case Type::Float:  return "float";
        case Type::Float2: return "float2";
        case Type::Float3: return "float3";
        case Type::Float4: return "float4";
    }
    return "float";
}
int ComponentCount(Type t) {
    switch (t) {
        case Type::Float:  return 1;
        case Type::Float2: return 2;
        case Type::Float3: return 3;
        case Type::Float4: return 4;
    }
    return 1;
}

const char* NodeKindName(NodeKind k) {
    switch (k) {
        case NodeKind::Constant:      return "Constant";
        case NodeKind::UV:            return "UV";
        case NodeKind::TextureSample: return "TextureSample";
        case NodeKind::Multiply:      return "Multiply";
        case NodeKind::Add:           return "Add";
        case NodeKind::Lerp:          return "Lerp";
        case NodeKind::Fresnel:       return "Fresnel";
        case NodeKind::Swizzle:       return "Swizzle";
        case NodeKind::MakeFloat3:    return "MakeFloat3";
        case NodeKind::MakeFloat4:    return "MakeFloat4";
        case NodeKind::Dot:           return "Dot";
        case NodeKind::Normalize:     return "Normalize";
        case NodeKind::Power:         return "Power";
        case NodeKind::OneMinus:      return "OneMinus";
        case NodeKind::Saturate:      return "Saturate";
        case NodeKind::PBROutput:     return "PBROutput";
    }
    return "Constant";
}

int SwizzleIndex(char c) {
    switch (c) {
        case 'x': case 'r': return 0;
        case 'y': case 'g': return 1;
        case 'z': case 'b': return 2;
        case 'w': case 'a': return 3;
        default: return -1;
    }
}

std::optional<NodeKind> ParseNodeKind(const std::string& s) {
    if (s == "Constant")      return NodeKind::Constant;
    if (s == "UV")            return NodeKind::UV;
    if (s == "TextureSample") return NodeKind::TextureSample;
    if (s == "Multiply")      return NodeKind::Multiply;
    if (s == "Add")           return NodeKind::Add;
    if (s == "Lerp")          return NodeKind::Lerp;
    if (s == "Fresnel")       return NodeKind::Fresnel;
    if (s == "Swizzle")       return NodeKind::Swizzle;
    if (s == "MakeFloat3")    return NodeKind::MakeFloat3;
    if (s == "MakeFloat4")    return NodeKind::MakeFloat4;
    if (s == "Dot")           return NodeKind::Dot;
    if (s == "Normalize")     return NodeKind::Normalize;
    if (s == "Power")         return NodeKind::Power;
    if (s == "OneMinus")      return NodeKind::OneMinus;
    if (s == "Saturate")      return NodeKind::Saturate;
    if (s == "PBROutput")     return NodeKind::PBROutput;
    return std::nullopt;
}

const char* PbrInputName(int slot) {
    switch (slot) {
        case kBaseColor: return "baseColor";
        case kMetallic:  return "metallic";
        case kRoughness: return "roughness";
        case kEmissive:  return "emissive";
    }
    return "";
}
Type PbrInputType(int slot) {
    switch (slot) {
        case kBaseColor: return Type::Float3;
        case kMetallic:  return Type::Float;
        case kRoughness: return Type::Float;
        case kEmissive:  return Type::Float3;
    }
    return Type::Float;
}

// --- Input ports per kind -----------------------------------------------------------------------
int InputPortCount(NodeKind k) {
    switch (k) {
        case NodeKind::Constant:      return 0;
        case NodeKind::UV:            return 0;
        case NodeKind::TextureSample: return 1;  // uv
        case NodeKind::Multiply:      return 2;  // a, b
        case NodeKind::Add:           return 2;  // a, b
        case NodeKind::Lerp:          return 3;  // a, b, t
        case NodeKind::Fresnel:       return 0;
        case NodeKind::Swizzle:       return 1;  // in
        case NodeKind::MakeFloat3:    return 3;  // x, y, z
        case NodeKind::MakeFloat4:    return 4;  // x, y, z, w
        case NodeKind::Dot:           return 2;  // a, b
        case NodeKind::Normalize:     return 1;  // in
        case NodeKind::Power:         return 2;  // a (base), b (exponent)
        case NodeKind::OneMinus:      return 1;  // in
        case NodeKind::Saturate:      return 1;  // in
        case NodeKind::PBROutput:     return kPbrInputCount;
    }
    return 0;
}

const char* InputPortName(NodeKind k, int idx) {
    switch (k) {
        case NodeKind::TextureSample: return idx == 0 ? "uv" : "";
        case NodeKind::Multiply:
        case NodeKind::Add:           return idx == 0 ? "a" : (idx == 1 ? "b" : "");
        case NodeKind::Lerp:          return idx == 0 ? "a" : (idx == 1 ? "b" : (idx == 2 ? "t" : ""));
        case NodeKind::Swizzle:
        case NodeKind::Normalize:
        case NodeKind::OneMinus:
        case NodeKind::Saturate:      return idx == 0 ? "in" : "";
        case NodeKind::MakeFloat3:    return idx == 0 ? "x" : (idx == 1 ? "y" : (idx == 2 ? "z" : ""));
        case NodeKind::MakeFloat4:    return idx == 0 ? "x" : (idx == 1 ? "y" : (idx == 2 ? "z" : (idx == 3 ? "w" : "")));
        case NodeKind::Dot:
        case NodeKind::Power:         return idx == 0 ? "a" : (idx == 1 ? "b" : "");
        case NodeKind::PBROutput:     return PbrInputName(idx);
        default:                      return "";
    }
}

// The DECLARED port type. For Multiply/Add/Lerp 'a'/'b' the type is resolved from the connection
// (component-wise math accepts float/float2/float3/float4 as long as a and b agree); we encode that
// "match" semantics in Validate rather than a fixed type, so here we return the float4-superset
// sentinel meaning "any vector type" for those data ports. 'Lerp.t' is always float.
//   To keep the type checker simple + strict, InputPortType returns a CONCRETE type only for ports
//   with a fixed type; for the polymorphic data ports it returns Type::Float4 as "any" and Validate
//   treats Float4 here as a wildcard.
Type InputPortType(NodeKind k, int idx) {
    switch (k) {
        case NodeKind::TextureSample: return Type::Float2;        // uv is float2
        case NodeKind::Multiply:
        case NodeKind::Add:           return Type::Float4;        // wildcard (any vector, a==b)
        case NodeKind::Lerp:
            if (idx == 2) return Type::Float;                     // t is scalar
            return Type::Float4;                                  // a/b wildcard
        // Slice AZ: 'in' (Swizzle/Normalize/OneMinus/Saturate) + Dot/Power a,b are vector wildcards
        // (Float4 sentinel). MakeFloatN x/y/z/w are strictly scalar (Float).
        case NodeKind::Swizzle:
        case NodeKind::Normalize:
        case NodeKind::OneMinus:
        case NodeKind::Saturate:
        case NodeKind::Dot:
        case NodeKind::Power:         return Type::Float4;        // wildcard (any vector)
        case NodeKind::MakeFloat3:
        case NodeKind::MakeFloat4:    return Type::Float;         // each component input is scalar
        case NodeKind::PBROutput:     return PbrInputType(idx);
        default:                      return Type::Float;
    }
}

const Node* Graph::FindNode(int id) const {
    for (const Node& n : nodes) if (n.id == id) return &n;
    return nullptr;
}

// Output type of a node. For Multiply/Add/Lerp the output matches the (resolved) type of input 'a'.
// The `depth` guard makes this safe to call even on a (not-yet-rejected) cyclic graph: it stops
// recursing past the node count so a cycle through 'a' ports can't blow the stack.
static Type OutputTypeImpl(const Graph& g, const Node& n, int depth) {
    switch (n.kind) {
        case NodeKind::Constant:      return n.outType;
        case NodeKind::UV:            return Type::Float2;
        case NodeKind::TextureSample: return Type::Float4;
        case NodeKind::Fresnel:       return Type::Float;
        case NodeKind::Dot:           return Type::Float;        // dot -> scalar.
        case NodeKind::MakeFloat3:    return Type::Float3;
        case NodeKind::MakeFloat4:    return Type::Float4;
        case NodeKind::Swizzle: {
            // Output type = mask length (1..4), clamped to a valid range.
            int len = (int)n.swizzle.size();
            if (len < 1) len = 1;
            if (len > 4) len = 4;
            switch (len) {
                case 1: return Type::Float;
                case 2: return Type::Float2;
                case 3: return Type::Float3;
                default: return Type::Float4;
            }
        }
        case NodeKind::PBROutput:     return Type::Float;  // sink; no output.
        case NodeKind::Multiply:
        case NodeKind::Add:
        case NodeKind::Lerp:
        case NodeKind::Power: {
            if (depth > (int)g.nodes.size()) return Type::Float4;  // cycle guard.
            // Resolve from input 'a' (the first data port).
            for (const Edge& e : g.edges) {
                if (e.toNode == n.id && e.toPort == "a") {
                    if (const Node* src = g.FindNode(e.fromNode))
                        return OutputTypeImpl(g, *src, depth + 1);
                }
            }
            return Type::Float4;  // unresolved (validation will have flagged a missing input).
        }
        case NodeKind::Normalize:
        case NodeKind::OneMinus:
        case NodeKind::Saturate: {
            if (depth > (int)g.nodes.size()) return Type::Float4;  // cycle guard.
            // Resolve from the single 'in' port.
            for (const Edge& e : g.edges) {
                if (e.toNode == n.id && e.toPort == "in") {
                    if (const Node* src = g.FindNode(e.fromNode))
                        return OutputTypeImpl(g, *src, depth + 1);
                }
            }
            return Type::Float4;  // unresolved.
        }
    }
    return Type::Float4;
}
Type OutputType(const Graph& g, const Node& n) { return OutputTypeImpl(g, n, 0); }

// --- Validation ---------------------------------------------------------------------------------
static ValidationResult Fail(std::string msg) {
    ValidationResult r; r.ok = false; r.error = std::move(msg); return r;
}

ValidationResult Validate(const Graph& g) {
    // Unique node ids.
    std::unordered_map<int, const Node*> byId;
    for (const Node& n : g.nodes) {
        if (n.id < 0) return Fail("node has a negative id");
        if (!byId.emplace(n.id, &n).second) return Fail("duplicate node id " + std::to_string(n.id));
    }

    // Exactly one PBROutput.
    int pbrCount = 0, pbrId = -1;
    for (const Node& n : g.nodes) if (n.kind == NodeKind::PBROutput) { ++pbrCount; pbrId = n.id; }
    if (pbrCount == 0) return Fail("graph has no PBROutput node");
    if (pbrCount > 1)  return Fail("graph has more than one PBROutput node");
    (void)pbrId;

    // Cycle detection (run FIRST so a cyclic graph is reported as a cycle rather than tripping a
    // spurious type error during type resolution). DFS over the directed (from -> to) edges.
    {
        std::unordered_map<int, std::vector<int>> adj;
        for (const Edge& e : g.edges) adj[e.fromNode].push_back(e.toNode);
        enum Color { White, Gray, Black };
        std::unordered_map<int, Color> color;
        for (const Node& n : g.nodes) color[n.id] = White;
        bool cyclic = false;
        std::function<void(int)> dfs = [&](int u) {
            color[u] = Gray;
            for (int v : adj[u]) {
                if (color[v] == Gray) { cyclic = true; return; }
                if (color[v] == White) { dfs(v); if (cyclic) return; }
            }
            color[u] = Black;
        };
        for (const Node& n : g.nodes) {
            if (color[n.id] == White) dfs(n.id);
            if (cyclic) return Fail("graph contains a cycle");
        }
    }

    // Edges reference real nodes/ports; each input port has at most one incoming edge.
    std::unordered_map<long long, bool> portUsed;  // key = toNode*64 + portIndex
    for (const Edge& e : g.edges) {
        const Node* from = g.FindNode(e.fromNode);
        const Node* to   = g.FindNode(e.toNode);
        if (!from) return Fail("edge from unknown node " + std::to_string(e.fromNode));
        if (!to)   return Fail("edge to unknown node " + std::to_string(e.toNode));
        if (from->kind == NodeKind::PBROutput)
            return Fail("PBROutput is a sink and cannot be an edge source");

        // Resolve the destination port index by name.
        int portIdx = -1;
        int n = InputPortCount(to->kind);
        for (int i = 0; i < n; ++i)
            if (e.toPort == InputPortName(to->kind, i)) { portIdx = i; break; }
        if (portIdx < 0)
            return Fail("node " + std::to_string(to->id) + " (" + NodeKindName(to->kind) +
                        ") has no input port '" + e.toPort + "'");

        long long key = (long long)to->id * 64 + portIdx;
        if (portUsed[key]) return Fail("input port '" + e.toPort + "' on node " +
                                       std::to_string(to->id) + " is driven by more than one edge");
        portUsed[key] = true;

        // Type check: source output type vs. destination port type. A Float4 destination port is a
        // WILDCARD (the polymorphic data ports of Multiply/Add/Lerp). Otherwise types must be equal.
        Type srcT = OutputType(g, *from);
        Type dstT = InputPortType(to->kind, portIdx);
        bool wildcard = (dstT == Type::Float4 &&
                         (to->kind == NodeKind::Multiply || to->kind == NodeKind::Add ||
                          to->kind == NodeKind::Lerp ||
                          // Slice AZ: the vector-input ports (Swizzle/Normalize/OneMinus/Saturate 'in',
                          // Dot/Power 'a'/'b') accept any vector type; per-node rules below refine this.
                          to->kind == NodeKind::Swizzle || to->kind == NodeKind::Normalize ||
                          to->kind == NodeKind::OneMinus || to->kind == NodeKind::Saturate ||
                          to->kind == NodeKind::Dot || to->kind == NodeKind::Power));
        // PBROutput's float3 vector ports (baseColor/emissive) accept a float4 source by taking its
        // .xyz — a documented narrowing the codegen + interpreter both implement. Scalar ports
        // (metallic/roughness) still require a float source (so e.g. a float2 -> metallic is a real
        // type error). Other ports require type equality.
        bool pbrNarrow = (to->kind == NodeKind::PBROutput &&
                          dstT == Type::Float3 && srcT == Type::Float4);
        if (!wildcard && !pbrNarrow && srcT != dstT)
            return Fail(std::string("type mismatch on edge into node ") + std::to_string(to->id) +
                        " port '" + e.toPort + "': source is " + TypeName(srcT) +
                        ", port expects " + TypeName(dstT));
    }

    // Multiply/Add/Lerp + (Slice AZ) Dot/Power: 'a' and 'b' must agree in type (component-wise).
    for (const Node& n : g.nodes) {
        if (n.kind != NodeKind::Multiply && n.kind != NodeKind::Add && n.kind != NodeKind::Lerp &&
            n.kind != NodeKind::Dot && n.kind != NodeKind::Power)
            continue;
        const Node* a = nullptr; const Node* b = nullptr;
        for (const Edge& e : g.edges) {
            if (e.toNode != n.id) continue;
            if (e.toPort == "a") a = g.FindNode(e.fromNode);
            if (e.toPort == "b") b = g.FindNode(e.fromNode);
        }
        if (a && b && OutputType(g, *a) != OutputType(g, *b))
            return Fail(std::string(NodeKindName(n.kind)) + " node " + std::to_string(n.id) +
                        ": inputs a and b have mismatched types (" + TypeName(OutputType(g, *a)) +
                        " vs " + TypeName(OutputType(g, *b)) + ")");
    }

    // --- Slice AZ: new-node type rules ----------------------------------------------------------
    for (const Node& n : g.nodes) {
        // Swizzle: mask length 1..4; each char a valid xyzw/rgba alias INDEXING WITHIN the input's
        // component count. Resolve the input ('in') type to bound-check the mask.
        if (n.kind == NodeKind::Swizzle) {
            const std::string& mask = n.swizzle;
            if (mask.empty()) return Fail("Swizzle node " + std::to_string(n.id) + " has an empty mask");
            if (mask.size() > 4)
                return Fail("Swizzle node " + std::to_string(n.id) + " mask '" + mask +
                            "' is longer than 4 components");
            const Node* in = nullptr;
            for (const Edge& e : g.edges)
                if (e.toNode == n.id && e.toPort == "in") in = g.FindNode(e.fromNode);
            int inComp = in ? ComponentCount(OutputType(g, *in)) : 0;
            if (!in) return Fail("Swizzle node " + std::to_string(n.id) + " has no 'in' input");
            for (char c : mask) {
                int idx = SwizzleIndex(c);
                if (idx < 0)
                    return Fail(std::string("Swizzle node ") + std::to_string(n.id) +
                                ": invalid mask char '" + c + "' (expected x/y/z/w or r/g/b/a)");
                if (idx >= inComp)
                    return Fail(std::string("Swizzle node ") + std::to_string(n.id) + ": mask char '" +
                                c + "' indexes component " + std::to_string(idx) +
                                " but the input is " + TypeName(OutputType(g, *in)));
            }
        }
        // MakeFloat3 / MakeFloat4: ALL N component inputs must be connected (arity) + scalar (the
        // scalar requirement is already enforced by InputPortType=Float in the edge type check above,
        // so here we only check arity).
        if (n.kind == NodeKind::MakeFloat3 || n.kind == NodeKind::MakeFloat4) {
            int need = InputPortCount(n.kind);
            for (int i = 0; i < need; ++i) {
                const char* port = InputPortName(n.kind, i);
                bool found = false;
                for (const Edge& e : g.edges)
                    if (e.toNode == n.id && e.toPort == port) { found = true; break; }
                if (!found)
                    return Fail(std::string(NodeKindName(n.kind)) + " node " + std::to_string(n.id) +
                                " is missing input '" + port + "' (needs all " +
                                std::to_string(need) + " scalar inputs)");
            }
        }
    }

    ValidationResult ok; ok.ok = true; return ok;
}

// --- Topological order (feeders of PBROutput, sink last) ----------------------------------------
std::vector<int> TopoOrder(const Graph& g) {
    // Build adjacency from -> to and indegree (over all nodes that appear).
    std::unordered_map<int, std::vector<int>> consumers;  // from -> [to...]
    std::unordered_map<int, int> indeg;
    for (const Node& n : g.nodes) indeg[n.id] = 0;
    for (const Edge& e : g.edges) {
        consumers[e.fromNode].push_back(e.toNode);
        indeg[e.toNode]++;
    }
    // Kahn's algorithm with a deterministic tie-break: always pick the smallest available id.
    std::vector<int> ready;
    for (const Node& n : g.nodes) if (indeg[n.id] == 0) ready.push_back(n.id);
    std::sort(ready.begin(), ready.end());

    std::vector<int> order;
    order.reserve(g.nodes.size());
    while (!ready.empty()) {
        int u = ready.front();
        ready.erase(ready.begin());
        order.push_back(u);
        std::vector<int> newly;
        for (int v : consumers[u]) {
            if (--indeg[v] == 0) newly.push_back(v);
        }
        std::sort(newly.begin(), newly.end());
        for (int v : newly) {
            // insert keeping `ready` sorted
            auto it = std::lower_bound(ready.begin(), ready.end(), v);
            ready.insert(it, v);
        }
    }
    return order;
}

// --- Per-node math primitives (the SHARED node semantics) ---------------------------------------
Value EvalMultiply(const Value& a, const Value& b) {
    Value r; r.count = std::max(a.count, b.count);
    for (int i = 0; i < 4; ++i) r.v[i] = a.v[i] * b.v[i];
    return r;
}
Value EvalAdd(const Value& a, const Value& b) {
    Value r; r.count = std::max(a.count, b.count);
    for (int i = 0; i < 4; ++i) r.v[i] = a.v[i] + b.v[i];
    return r;
}
Value EvalLerp(const Value& a, const Value& b, float t) {
    Value r; r.count = std::max(a.count, b.count);
    for (int i = 0; i < 4; ++i) r.v[i] = a.v[i] + (b.v[i] - a.v[i]) * t;
    return r;
}
float EvalFresnel(float NoV, float power) {
    float c = NoV; if (c < 0.0f) c = 0.0f; if (c > 1.0f) c = 1.0f;  // saturate
    return std::pow(1.0f - c, power);
}

// --- Slice AZ node primitives -------------------------------------------------------------------
Value EvalSwizzle(const Value& in, const std::string& mask) {
    Value r;
    int len = (int)mask.size();
    if (len < 1) len = 1;
    if (len > 4) len = 4;
    r.count = len;
    for (int i = 0; i < 4; ++i) r.v[i] = 0.0f;
    for (int i = 0; i < len; ++i) {
        int idx = SwizzleIndex(mask[(size_t)i]);
        if (idx < 0) idx = 0;
        r.v[i] = in.v[(size_t)idx];
    }
    return r;
}
Value EvalMakeFloat(const std::array<float, 4>& comps, int count) {
    Value r; r.count = count;
    for (int i = 0; i < 4; ++i) r.v[i] = (i < count) ? comps[(size_t)i] : 0.0f;
    return r;
}
Value EvalDot(const Value& a, const Value& b) {
    int n = std::max(a.count, b.count);
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a.v[i] * b.v[i];
    Value r; r.count = 1; r.v = {s, 0, 0, 0};
    return r;
}
Value EvalNormalize(const Value& x) {
    Value r; r.count = x.count;
    float len2 = 0.0f;
    for (int i = 0; i < x.count; ++i) len2 += x.v[i] * x.v[i];
    float inv = (len2 > 0.0f) ? 1.0f / std::sqrt(len2) : 0.0f;
    for (int i = 0; i < 4; ++i) r.v[i] = (i < x.count) ? x.v[i] * inv : 0.0f;
    return r;
}
Value EvalPower(const Value& base, const Value& exp) {
    Value r; r.count = std::max(base.count, exp.count);
    for (int i = 0; i < 4; ++i) r.v[i] = std::pow(base.v[i], exp.v[i]);
    return r;
}
Value EvalOneMinus(const Value& x) {
    Value r; r.count = x.count;
    for (int i = 0; i < 4; ++i) r.v[i] = 1.0f - x.v[i];
    return r;
}
Value EvalSaturate(const Value& x) {
    Value r; r.count = x.count;
    for (int i = 0; i < 4; ++i) {
        float c = x.v[i]; if (c < 0.0f) c = 0.0f; if (c > 1.0f) c = 1.0f;
        r.v[i] = c;
    }
    return r;
}

// --- Whole-graph CPU interpreter ----------------------------------------------------------------
namespace {

// Convert a node's full Value into the documented N-component slice for its output type.
Value Slice(const Value& in, int count) {
    Value r = in; r.count = count;
    for (int i = count; i < 4; ++i) r.v[i] = 0.0f;
    return r;
}

}  // namespace

PbrResult Evaluate(const Graph& g, float u, float v, float NoV,
                   const std::function<std::array<float, 4>(const std::string&, float, float)>& sampleTex) {
    std::vector<int> order = TopoOrder(g);
    std::unordered_map<int, Value> out;  // node id -> output value

    auto inputValue = [&](const Node& n, const char* port) -> std::optional<Value> {
        for (const Edge& e : g.edges)
            if (e.toNode == n.id && e.toPort == port) {
                auto it = out.find(e.fromNode);
                if (it != out.end()) return it->second;
            }
        return std::nullopt;
    };

    for (int id : order) {
        const Node* np = g.FindNode(id);
        if (!np) continue;
        const Node& n = *np;
        switch (n.kind) {
            case NodeKind::Constant: {
                Value val; val.count = ComponentCount(n.outType);
                for (int i = 0; i < 4; ++i) val.v[i] = n.value[(size_t)i];
                out[id] = Slice(val, val.count);
                break;
            }
            case NodeKind::UV: {
                Value val; val.count = 2; val.v = {u, v, 0, 0};
                out[id] = val;
                break;
            }
            case NodeKind::TextureSample: {
                float su = u, sv = v;
                if (auto uvIn = inputValue(n, "uv")) { su = uvIn->v[0]; sv = uvIn->v[1]; }
                std::array<float, 4> t = sampleTex ? sampleTex(n.texture, su, sv)
                                                   : std::array<float, 4>{0, 0, 0, 1};
                Value val; val.count = 4; val.v = t;
                out[id] = val;
                break;
            }
            case NodeKind::Multiply: {
                Value a = inputValue(n, "a").value_or(Value{});
                Value b = inputValue(n, "b").value_or(Value{});
                out[id] = EvalMultiply(a, b);
                break;
            }
            case NodeKind::Add: {
                Value a = inputValue(n, "a").value_or(Value{});
                Value b = inputValue(n, "b").value_or(Value{});
                out[id] = EvalAdd(a, b);
                break;
            }
            case NodeKind::Lerp: {
                Value a = inputValue(n, "a").value_or(Value{});
                Value b = inputValue(n, "b").value_or(Value{});
                Value t = inputValue(n, "t").value_or(Value{});
                out[id] = EvalLerp(a, b, t.v[0]);
                break;
            }
            case NodeKind::Fresnel: {
                Value val; val.count = 1; val.v = {EvalFresnel(NoV, n.power), 0, 0, 0};
                out[id] = val;
                break;
            }
            case NodeKind::Swizzle: {
                Value in = inputValue(n, "in").value_or(Value{});
                out[id] = EvalSwizzle(in, n.swizzle);
                break;
            }
            case NodeKind::MakeFloat3:
            case NodeKind::MakeFloat4: {
                int cnt = (n.kind == NodeKind::MakeFloat3) ? 3 : 4;
                std::array<float, 4> comps{0, 0, 0, 0};
                const char* ports[4] = {"x", "y", "z", "w"};
                for (int i = 0; i < cnt; ++i)
                    comps[(size_t)i] = inputValue(n, ports[i]).value_or(Value{}).v[0];
                out[id] = EvalMakeFloat(comps, cnt);
                break;
            }
            case NodeKind::Dot: {
                Value a = inputValue(n, "a").value_or(Value{});
                Value b = inputValue(n, "b").value_or(Value{});
                out[id] = EvalDot(a, b);
                break;
            }
            case NodeKind::Normalize: {
                Value in = inputValue(n, "in").value_or(Value{});
                out[id] = EvalNormalize(in);
                break;
            }
            case NodeKind::Power: {
                Value a = inputValue(n, "a").value_or(Value{});
                Value b = inputValue(n, "b").value_or(Value{});
                out[id] = EvalPower(a, b);
                break;
            }
            case NodeKind::OneMinus: {
                Value in = inputValue(n, "in").value_or(Value{});
                out[id] = EvalOneMinus(in);
                break;
            }
            case NodeKind::Saturate: {
                Value in = inputValue(n, "in").value_or(Value{});
                out[id] = EvalSaturate(in);
                break;
            }
            case NodeKind::PBROutput:
                break;  // handled below.
        }
    }

    // Read the PBROutput inputs, applying defaults for unconnected ports.
    PbrResult r;  // defaults: baseColor=1, metallic=0, roughness=1, emissive=0.
    const Node* sink = nullptr;
    for (const Node& n : g.nodes) if (n.kind == NodeKind::PBROutput) sink = &n;
    if (sink) {
        auto pin = [&](const char* port) -> std::optional<Value> {
            for (const Edge& e : g.edges)
                if (e.toNode == sink->id && e.toPort == port) {
                    auto it = out.find(e.fromNode);
                    if (it != out.end()) return it->second;
                }
            return std::nullopt;
        };
        if (auto bc = pin("baseColor")) { r.baseColor = {bc->v[0], bc->v[1], bc->v[2]}; }
        if (auto m = pin("metallic"))   { r.metallic = m->v[0]; }
        if (auto ro = pin("roughness")) { r.roughness = ro->v[0]; }
        if (auto em = pin("emissive"))  { r.emissive = {em->v[0], em->v[1], em->v[2]}; }
    }
    return r;
}

}  // namespace hf::material
