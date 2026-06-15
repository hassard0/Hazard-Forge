// Hazard Forge — HLSL codegen for the material graph (Slice AV).
//
// GenerateHlsl(graph) emits a COMPLETE HLSL fragment shader: it reuses the IDENTICAL varyings /
// bindings / FrameData / PBR lighting core as shaders/lit_pbr.frag.hlsl (factored into the shared
// include shaders/pbr_core.hlsli), topologically evaluates the graph's nodes into temporaries, and
// feeds baseColor/metallic/roughness/emissive into hfShadePBR(). Build-time only: the generated HLSL
// goes through the SAME DXC->SPIR-V (and glslc->spirv-cross->MSL) toolchain as every other shader, so
// the two backends stay byte-identical.
//
// Pure CPU — no backend symbols. Deterministic: the same graph always yields byte-identical HLSL.
#pragma once

#include "material/shader_graph.h"

#include <string>

namespace hf::material {

// Emit the full HLSL fragment shader text for `g`. Requires a valid graph (Validate first); on an
// invalid graph the returned string begins with "// ERROR:" and the error message (so a broken
// .mat.json fails the shader compile loudly rather than silently emitting garbage).
std::string GenerateHlsl(const Graph& g);

}  // namespace hf::material
