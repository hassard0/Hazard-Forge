// Hazard Forge — runtime material -> SPIR-V compilation via a dxc SUBPROCESS (Slice AW).
//
// PURE HOST/TOOLING logic above the RHI seam (subprocess + temp-file IO only; NO vk*/MTL* symbols).
//
// CompileGraphToSpirv(graph, &err):
//   1. GenerateHlsl(graph)            — reuse the AV codegen (deterministic, build-time-identical).
//   2. write the HLSL to a temp .hlsl under shaders/generated/ (so its `#include "../pbr_core.hlsli"`
//      resolves the same way the build-time generated shader's does).
//   3. invoke the SAME dxc.exe the build uses, with the SAME fragment profile + flags
//      (-spirv -T ps_6_0 -E main -fspv-target-env=vulkan1.3), -> a temp .spv.
//   4. read back the .spv words.
//
// Because the compiler + flags are IDENTICAL to cmake/CompileShaders.cmake, the runtime SPIR-V is
// BYTE-IDENTICAL to the build-time SPIR-V for the same graph — so a runtime-authored material
// provably matches the build-time golden. On ANY failure returns nullopt + an error string (the
// live loop keeps the previous material; it never crashes).
#pragma once

#include "material/shader_graph.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hf::material {

// Compile `graph` -> SPIR-V words via the dxc subprocess. nullopt + *errorOut on failure (invalid
// graph, dxc not found, dxc error, IO error). Deterministic for a fixed graph + dxc.
std::optional<std::vector<uint32_t>> CompileGraphToSpirv(const Graph& graph, std::string* errorOut);

// Resolve the dxc executable path the same way the build does: prefer the configured HF_DXC_PATH
// (the exact dxc CMake resolved for CompileShaders.cmake), else fall back to a PATH/SDK lookup with
// the WinGet/Vulkan-SDK hints. Empty string if none found. Exposed for diagnostics/tests.
std::string ResolveDxcPath();

}  // namespace hf::material
