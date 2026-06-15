// Hazard Forge — material-graph introspection (Slice BI, Phase 4 #11).
//
// Pure-CPU text generation over the in-memory material::Graph (the same Graph/Node/Edge model the
// loader + codegen use). Produces a DETERMINISTIC, machine-readable description of a material graph —
// its nodes (id, type, params, typed input ports, output type), the edges between them, and the
// resolved PBROutput sink — as either pretty JSON (for an agent / a future visual node editor to
// consume) or a Graphviz DOT digraph (for visualization). Mirrors the engine-state introspection
// (Slice AL / editor::DescribeEngine) but for material graphs.
//
// HARD RULE: no vk*/MTL*/Backend::Metal/mtl:: symbols. This lives strictly above the RHI seam — it is
// a string builder over the CPU graph model, with no I/O of its own (the caller writes the strings).
// Output is byte-identical run-to-run and cross-platform: nodes are emitted in a STABLE order (by
// ascending id), ports in declaration order, and floats with the same "%g on the double-promoted
// value" convention the engine introspect JSON uses. LF newlines.
#pragma once

#include "material/shader_graph.h"

#include <string>

namespace hf::material {

// Describe a material graph as deterministic, pretty-printed JSON. `name` is the material's name (the
// loader's LoadResult::name). Shape:
//   {
//     "material": "<name>",
//     "nodeCount": N,
//     "edgeCount": E,
//     "output": <PBROutput node id, or null if none>,
//     "nodes": [
//       { "id": <int>, "type": "<NodeKind>", "params": { ... },
//         "inputs": [ { "port": "<name>", "type": "<float|float2|float3|float4|any>",
//                       "source": "<fromNodeId.out>" | null, "default": <value> | null } ],
//         "output": { "type": "<resolved output type>" } }
//     ],
//     "edges": [ { "from": "<fromNodeId.out>", "to": "<toNodeId.inPort>" } ]
//   }
// Nodes are emitted sorted by ascending id; each node's input ports in declaration order; edges in the
// graph's stored order. The single node output port is named "out".
std::string DescribeGraphJson(const Graph& g, const std::string& name);

// Describe a material graph as a Graphviz DOT digraph. One `node [label="<type>\n#<id>"]` per node
// (the PBROutput sink highlighted), one edge per connection labelled "<outPort>-><inPort>". Stable
// node + edge order (nodes by ascending id, edges in stored order), so two runs are byte-identical.
std::string ToDot(const Graph& g);

}  // namespace hf::material
