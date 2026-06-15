# Slice AV — Data-Driven Material / Shader Graph (MVP) — Phase 4 opener — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Opens Phase 4
> (content/authoring) — the flagship UE5 "Material Editor" capability, made agentic (materials are data).

**Goal:** Author a surface material as a small **node graph in JSON**, compile it to an HLSL fragment that
feeds the existing PBR lighting, and render it — golden-verified on both backends. This turns shading
from hand-written `.hlsl` into DATA the agent (or a human) generates and mutates. MVP: a fixed node set,
build-time codegen into the existing HLSL→SPIR-V→MSL toolchain, one example graph on a sphere.

## Why build-time codegen (the MVP choice)

Shaders compile at BUILD time via `cmake/CompileShaders.cmake` (DXC → SPIR-V; spirv-cross → MSL). The
Vulkan path has no runtime HLSL compiler (only Metal runtime-compiles MSL). So the deterministic,
cross-backend-identical MVP is: graph JSON → **codegen an HLSL fragment at build time** → compile it like
any other shader. Same toolchain ⇒ same SPIR-V/MSL ⇒ goldens match. (Live runtime graph authoring needs a
runtime HLSL→SPIR-V compiler — explicitly OUT OF SCOPE, a future slice. The agentic loop here is: write
`*.mat.json` → build → `--material-shot` capture.)

## Design decisions (locked)

1. **Graph model (engine/material/shader_graph.{h,cpp}, pure CPU, no backend symbols).** Namespace
   `hf::material`. A `Graph { std::vector<Node> nodes; std::vector<Edge> edges; }` where each `Node` has a
   type, an id, constant params, and typed input/output ports (scalar `float`, `float2`, `float3`,
   `float4`). MVP node set (YAGNI — exactly these):
   - `Constant` (param: a float4 value; output the needed swizzle)
   - `UV` (output float2 — the interpolated TEXCOORD0)
   - `TextureSample` (param: texture slot name e.g. "baseColorTex"; input UV; output float4)
   - `Multiply`, `Add` (two inputs, component-wise, output matches)
   - `Lerp` (a, b, t → mix)
   - `Fresnel` (param: power; uses N·V from the lit varyings; output scalar)
   - `PBROutput` (SINK; inputs: baseColor float3, metallic float, roughness float, emissive float3) — the
     single graph terminal that drives the existing PBR lighting.
   The graph is a DAG with exactly one `PBROutput`. Validate: no cycles, all PBROutput inputs connected
   (defaults if unconnected: baseColor=1, metallic=0, roughness=1, emissive=0), types match per edge.

2. **JSON authoring + loader.** `assets/materials/*.mat.json` describes nodes+edges. A loader parses to
   `Graph`. Provide ONE example graph `assets/materials/showcase.mat.json` that exercises several nodes
   (e.g. baseColor = Lerp(textureSample(checker), Constant(tint), Fresnel(power=3)); metallic=Constant;
   roughness=Constant) so the golden visibly shows graph-driven shading (a fresnel rim over a textured
   sphere). Deterministic.

3. **HLSL codegen (engine/material/codegen.{h,cpp}, pure CPU).** `std::string GenerateHlsl(const Graph&)`
   → emits the BODY of an HLSL fragment that computes the PBROutput values, wrapped in a fixed template
   that (a) declares the standard varyings/bindings IDENTICAL to `lit_pbr.frag.hlsl` (so the same
   descriptor layout + FrameData + the existing PBR lighting helpers apply), (b) topologically evaluates
   nodes into temporaries, (c) feeds baseColor/metallic/roughness/emissive into the SAME PBR lighting code
   path (factor the PBR lighting in lit_pbr into an include/helper both the hand-written and generated
   shaders call, OR have codegen emit the full shader from a template that contains the PBR core). The
   generated file lands at e.g. `shaders/generated/mat_showcase.frag.hlsl` at BUILD time (CMake custom
   command runs a tiny codegen host tool `material_codegen` over the .mat.json), then is compiled by the
   normal shader pipeline. Keep `HF_MSL_GEN` conventions so MSL gen matches.

4. **Codegen host tool.** A small `tools/material_codegen` C++ target: reads a `.mat.json`, writes the
   `.frag.hlsl`. Wired into CMake so building regenerates the shader from the JSON (the JSON is the source
   of truth; the generated HLSL may be git-ignored or committed — commit it for transparency/diff and so
   the Metal standalone build has it without running the tool; document the choice).

5. **Render path.** A material pipeline variant using the generated fragment (same vertex shader as
   lit_pbr, same descriptor sets — base-color at set1 binding0, etc.). Showcase
   `hello_triangle.exe --material-shot <out>` renders a sphere (+ground + skybox + the standard light)
   shaded by `showcase.mat.json`. Metal `visual_test --material` does the same (the generated HLSL is
   compiled to MSL through the standalone Metal build's existing HLSL→MSL step OR the committed generated
   file is fed through the same gen as other shaders — match how metal_headless consumes shaders).

6. **Verification:** new golden `tests/golden/metal/mat_graph.png` (Metal two runs DIFF 0.0000). Existing
   26 image goldens UNTOUCHED. Introspect rebaked exactly `+material-graph` (features) + `--material-shot`
   (showcases).

## Tests
- `tests/shader_graph_test.cpp`:
  - **Graph validation:** cycle detection rejects a cyclic graph; missing PBROutput rejected; type
    mismatch on an edge rejected; unconnected PBROutput inputs get the documented defaults.
  - **CPU interpreter parity:** a CPU evaluator of the SAME node semantics (Multiply/Add/Lerp/Fresnel/
    Constant/TextureSample-with-a-stub-texture) evaluated at sample points matches hand-computed expected
    values — this pins node SEMANTICS independent of the GPU (the generated HLSL must implement the same
    math; document that the CPU interpreter and the codegen share the per-node formula definitions).
  - **Codegen determinism/structure:** `GenerateHlsl` on the example graph is deterministic (same string
    every run) and contains the expected temporaries/topological order (structural asserts, not a brittle
    full-string match). Clean under `windows-msvc-asan`.

## RHI seam additions (summary)
- **None.** Material graph + codegen + loader are pure CPU above the RHI. The render path reuses the
  existing PBR material pipeline/descriptor layout. New files (`engine/material/*`, `tools/material_codegen`,
  `tests/shader_graph_test.cpp`, `assets/materials/*.mat.json`, generated HLSL) add ZERO backend symbols.
  Seam grep stays at the benign baseline (no new code symbols above the seam).

## Out of scope (YAGNI)
Runtime/live graph authoring + runtime HLSL→SPIR-V compile, a visual node editor UI, custom-HLSL-expression
nodes, vertex-stage graphs / displacement, multiple material instances per scene, parameter animation,
subgraphs/functions, more node types than the MVP set, texture-graph baking. One JSON graph, fixed node
set, build-time codegen, one golden, both backends.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 27) + new `shader_graph_test` (validation + CPU
   interpreter parity + codegen determinism). Clean under `windows-msvc-asan`.
2. Build regenerates `shaders/generated/mat_showcase.frag.hlsl` from `assets/materials/showcase.mat.json`
   via the `material_codegen` tool; the shader compiles through the normal HLSL→SPIR-V→MSL pipeline.
3. `--material-shot` on Windows/Vulkan: controller visual review — the sphere shows graph-driven shading
   (textured base with a fresnel rim), coherent and lit. Run under the AT Vulkan-validation gate → ZERO
   errors.
4. Metal: `visual_test --material` → new golden `tests/golden/metal/mat_graph.png`; two runs DIFF 0.0000.
5. **Render-invariance of existing scenes:** `git diff master --stat -- tests/golden/metal` shows ONLY
   `mat_graph.png` added; the other 26 byte-identical.
6. Introspect JSON rebaked exactly `+material-graph` + `--material-shot`; introspect test updated; no other
   drift.
7. Seam grep clean (no new code symbols above the seam). `scripts/verify.ps1` updated to include the new
   `mat_graph` image golden in the Mac round-trip loop.
