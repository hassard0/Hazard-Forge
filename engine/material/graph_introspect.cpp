// Hazard Forge — material-graph introspection (Slice BI). See graph_introspect.h.
// Pure CPU; no backend symbols. Deterministic text over the in-memory Graph.
#include "material/graph_introspect.h"

#include <algorithm>
#include <cstdio>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace hf::material {
namespace {

// --- Deterministic helpers ----------------------------------------------------------------------

// Append a float the SAME way scene_io / editor::DescribeEngine do: %g on the double-promoted value.
// Stable + lossless enough to round-trip authored values; no locale dependence (%g uses '.' in C).
void AppendFloat(std::ostream& os, float f) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(f));
    os << buf;
}

// JSON string with minimal escaping (node types/params/material names are simple ASCII, but escape
// the JSON-significant characters just in case).
void AppendString(std::ostream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n";  break;
            case '\t': os << "\\t";  break;
            case '\r': os << "\\r";  break;
            default:   os << c;       break;
        }
    }
    os << '"';
}

std::string Indent(int depth) { return std::string(static_cast<size_t>(depth) * 2, ' '); }

// The single node-output port name (each node kind has exactly one output).
const char* kOutPort = "out";

// Port type label. The polymorphic data ports (Multiply/Add/Lerp a/b, the vector 'in' of
// Swizzle/Normalize/OneMinus/Saturate, Dot/Power a/b) use the Float4 sentinel as a WILDCARD in the
// model; report those as "any" so the dump isn't misleading. Every other port has a concrete type.
bool IsWildcardPort(NodeKind k, int idx) {
    switch (k) {
        case NodeKind::Multiply:
        case NodeKind::Add:
            return true;  // a, b
        case NodeKind::Lerp:
            return idx != 2;  // a, b are wildcard; t is concrete float
        case NodeKind::Swizzle:
        case NodeKind::Normalize:
        case NodeKind::OneMinus:
        case NodeKind::Saturate:
        case NodeKind::Dot:
        case NodeKind::Power:
            return true;  // the vector input(s)
        default:
            return false;
    }
}

std::string PortTypeLabel(NodeKind k, int idx) {
    if (IsWildcardPort(k, idx)) return "any";
    return TypeName(InputPortType(k, idx));
}

// The documented default for an unconnected PBROutput input (the SAME fallbacks the interpreter +
// codegen use): baseColor=1, metallic=0, roughness=1, emissive=0, normal=(0,0,1). Returns false for
// any non-PBROutput port (those have no documented default -> emitted as null).
bool PbrDefault(int slot, float out[4], int& count) {
    switch (slot) {
        case kBaseColor: out[0] = out[1] = out[2] = 1.0f; count = 3; return true;
        case kMetallic:  out[0] = 0.0f;                   count = 1; return true;
        case kRoughness: out[0] = 1.0f;                   count = 1; return true;
        case kEmissive:  out[0] = out[1] = out[2] = 0.0f; count = 3; return true;
        case kNormal:    out[0] = 0.0f; out[1] = 0.0f; out[2] = 1.0f; count = 3; return true;
        default: return false;
    }
}

// Emit a default value (scalar -> number; vector -> JSON array) at the given component count.
void AppendDefault(std::ostream& os, const float v[4], int count) {
    if (count <= 1) { AppendFloat(os, v[0]); return; }
    os << "[";
    for (int i = 0; i < count; ++i) {
        if (i) os << ", ";
        AppendFloat(os, v[i]);
    }
    os << "]";
}

// Node ids sorted ascending (the stable emit order).
std::vector<int> SortedNodeIds(const Graph& g) {
    std::vector<int> ids;
    ids.reserve(g.nodes.size());
    for (const Node& n : g.nodes) ids.push_back(n.id);
    std::sort(ids.begin(), ids.end());
    return ids;
}

// The PBROutput sink id, or -1 if absent.
int OutputNodeId(const Graph& g) {
    for (const Node& n : g.nodes)
        if (n.kind == NodeKind::PBROutput) return n.id;
    return -1;
}

// The edge feeding (toNode, toPort), or nullptr if that port is unconnected.
const Edge* IncomingEdge(const Graph& g, int toNode, const char* toPort) {
    for (const Edge& e : g.edges)
        if (e.toNode == toNode && e.toPort == toPort) return &e;
    return nullptr;
}

// Emit a node's params object body (the key/values between the braces), kind-specific. Returns the
// number of params emitted (0 -> the caller writes an empty "{}").
void AppendParams(std::ostream& os, const Node& n, int depth) {
    std::vector<std::pair<std::string, std::string>> kv;  // pre-rendered (key, value) pairs, in order.
    auto num = [](float f) { std::ostringstream s; AppendFloat(s, f); return s.str(); };

    switch (n.kind) {
        case NodeKind::Constant: {
            std::ostringstream val;
            val << "[";
            for (int i = 0; i < 4; ++i) { if (i) val << ", "; AppendFloat(val, n.value[(size_t)i]); }
            val << "]";
            kv.emplace_back("value", val.str());
            std::ostringstream ot; ot << '"' << TypeName(n.outType) << '"';
            kv.emplace_back("outType", ot.str());
            break;
        }
        case NodeKind::TextureSample:
        case NodeKind::NormalMap: {
            std::ostringstream tex; AppendString(tex, n.texture);
            kv.emplace_back("texture", tex.str());
            break;
        }
        case NodeKind::Fresnel: {
            kv.emplace_back("power", num(n.power));
            break;
        }
        case NodeKind::Swizzle: {
            std::ostringstream sw; AppendString(sw, n.swizzle);
            kv.emplace_back("swizzle", sw.str());
            break;
        }
        default:
            break;
    }

    if (kv.empty()) { os << "{}"; return; }
    os << "{\n";
    for (size_t i = 0; i < kv.size(); ++i) {
        os << Indent(depth + 1) << '"' << kv[i].first << "\": " << kv[i].second
           << (i + 1 < kv.size() ? ",\n" : "\n");
    }
    os << Indent(depth) << "}";
}

}  // namespace

std::string DescribeGraphJson(const Graph& g, const std::string& name) {
    std::ostringstream os;
    std::vector<int> ids = SortedNodeIds(g);
    int outId = OutputNodeId(g);

    os << "{\n";
    os << Indent(1) << "\"material\": "; AppendString(os, name); os << ",\n";
    os << Indent(1) << "\"nodeCount\": " << g.nodes.size() << ",\n";
    os << Indent(1) << "\"edgeCount\": " << g.edges.size() << ",\n";
    os << Indent(1) << "\"output\": ";
    if (outId < 0) os << "null"; else os << outId;
    os << ",\n";

    // --- nodes (sorted by id) -------------------------------------------------------------------
    os << Indent(1) << "\"nodes\": [";
    if (ids.empty()) {
        os << "],\n";
    } else {
        os << "\n";
        for (size_t ni = 0; ni < ids.size(); ++ni) {
            const Node* np = g.FindNode(ids[ni]);
            const Node& n = *np;
            os << Indent(2) << "{\n";
            os << Indent(3) << "\"id\": " << n.id << ",\n";
            os << Indent(3) << "\"type\": \"" << NodeKindName(n.kind) << "\",\n";
            os << Indent(3) << "\"params\": "; AppendParams(os, n, 3); os << ",\n";

            // inputs (declaration order).
            int portCount = InputPortCount(n.kind);
            os << Indent(3) << "\"inputs\": [";
            if (portCount == 0) {
                os << "],\n";
            } else {
                os << "\n";
                for (int pi = 0; pi < portCount; ++pi) {
                    const char* port = InputPortName(n.kind, pi);
                    os << Indent(4) << "{ \"port\": ";
                    AppendString(os, port);
                    os << ", \"type\": \"" << PortTypeLabel(n.kind, pi) << "\", \"source\": ";
                    const Edge* e = IncomingEdge(g, n.id, port);
                    if (e) {
                        std::ostringstream src;
                        src << e->fromNode << "." << kOutPort;
                        AppendString(os, src.str());
                        os << ", \"default\": null";
                    } else {
                        os << "null, \"default\": ";
                        float dv[4] = {0, 0, 0, 0};
                        int dc = 0;
                        if (n.kind == NodeKind::PBROutput && PbrDefault(pi, dv, dc))
                            AppendDefault(os, dv, dc);
                        else
                            os << "null";
                    }
                    os << " }" << (pi + 1 < portCount ? ",\n" : "\n");
                }
                os << Indent(3) << "],\n";
            }

            // output type (resolved).
            os << Indent(3) << "\"output\": { \"type\": \"";
            if (n.kind == NodeKind::PBROutput) os << "none";  // sink: no output.
            else os << TypeName(OutputType(g, n));
            os << "\" }\n";

            os << Indent(2) << "}" << (ni + 1 < ids.size() ? ",\n" : "\n");
        }
        os << Indent(1) << "],\n";
    }

    // --- edges (stored order) -------------------------------------------------------------------
    os << Indent(1) << "\"edges\": [";
    if (g.edges.empty()) {
        os << "]\n";
    } else {
        os << "\n";
        for (size_t ei = 0; ei < g.edges.size(); ++ei) {
            const Edge& e = g.edges[ei];
            std::ostringstream from; from << e.fromNode << "." << kOutPort;
            std::ostringstream to;   to << e.toNode << "." << e.toPort;
            os << Indent(2) << "{ \"from\": "; AppendString(os, from.str());
            os << ", \"to\": "; AppendString(os, to.str());
            os << " }" << (ei + 1 < g.edges.size() ? ",\n" : "\n");
        }
        os << Indent(1) << "]\n";
    }

    os << "}\n";
    return os.str();
}

std::string ToDot(const Graph& g) {
    std::ostringstream os;
    std::vector<int> ids = SortedNodeIds(g);

    os << "digraph material {\n";
    os << "  rankdir=LR;\n";
    os << "  node [shape=box, fontname=\"monospace\"];\n";

    // Nodes (sorted by id). PBROutput sink highlighted.
    for (int id : ids) {
        const Node* np = g.FindNode(id);
        const Node& n = *np;
        os << "  n" << n.id << " [label=\"" << NodeKindName(n.kind) << "\\n#" << n.id << "\"";
        if (n.kind == NodeKind::PBROutput)
            os << ", style=filled, fillcolor=\"#ffd479\"";
        os << "];\n";
    }

    // Edges (stored order), labelled "<outPort>-><inPort>".
    for (const Edge& e : g.edges) {
        os << "  n" << e.fromNode << " -> n" << e.toNode
           << " [label=\"" << kOutPort << "->" << e.toPort << "\"];\n";
    }

    os << "}\n";
    return os.str();
}

}  // namespace hf::material
