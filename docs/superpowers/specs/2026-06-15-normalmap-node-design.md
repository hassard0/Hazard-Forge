# Slice BE — NormalMap Material-Graph Node (Phase 4 #8) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Completes the
> material graph's (AV/AW/AZ) visual range: tangent-space normal mapping as a graph node.

**Goal:** Let a graph material perturb its shading normal via a tangent-space normal map — the last big
visual capability the material graph lacks. Add a `NormalMap` node (sample a normal texture → decode →
tangent-space normal) and a `normal` input on `PBROutput`; the codegen transforms the graph's
tangent-space normal into world space via the existing TBN and uses it for lighting (exactly as
`lit.frag.hlsl` does with its built-in normal map). A `--material-normal-shot` showcase shows visible
surface detail; golden-verified on both backends.

## Design decisions (locked)

1. **New node + PBROutput extension (engine/material/shader_graph, pure CPU, no backend symbols).**
   - `NormalMap` node (param: a normal-texture slot name, default the engine's existing `"normalmap"`
     texture; input: UV (defaults to the interpolated UV); output: a tangent-space float3 normal =
     `normalize(decode(sample(tex, uv)))` where `decode(c) = c*2 - 1`). The CPU interpreter implements
     the same decode (for the parity test, with a stub texture sampler).
   - Extend `PBROutput` with a 5th input `normal` (tangent-space float3, DEFAULT `(0,0,1)` = no
     perturbation, so existing graphs without it are unchanged). When connected, the surface uses the
     graph normal; when not, behaviour is byte-identical to the AV/AW/AZ generated shaders (proven by the
     existing material goldens staying byte-identical).

2. **Codegen — TBN transform before lighting.** When `PBROutput.normal` is connected, the generated
   fragment builds the world-space shading normal from the tangent-space graph normal using the existing
   `wnormal` + `wtangent` varyings (Gram-Schmidt TBN, the SAME construction `lit.frag.hlsl` uses), and
   feeds THAT normal into `pbr_core.hlsli`'s lighting instead of the geometric normal. When NOT connected,
   emit exactly the current code path (geometric normal) — so unconnected graphs produce byte-identical
   SPIR-V/MSL → existing goldens unchanged. The vertex shader already provides `wtangent` (the material
   pipeline's vertex layout includes tangents — confirm; the lit/PBR path already uses them for its
   built-in normal map).

3. **Example material + showcase.** New `assets/materials/normalmap.mat.json` — a material whose
   `PBROutput.normal = NormalMap(slot="normalmap")`, with a plain baseColor/metallic/roughness (so the
   golden's visible difference is the NORMAL detail). `--material-normal-shot <out>` (Vulkan) /
   `--material-normal` (Metal): render a sphere (or a plane — pick the surface that shows the normal
   detail best; a plane facing the camera with the normal map reads clearly) + ground + sky + standard
   light, shaded by `normalmap.mat.json`. Build-time codegen → committed
   `shaders/generated/mat_normalmap.frag.hlsl`. New golden `tests/golden/metal/mat_normal.png` (Metal two
   runs DIFF 0.0000). Existing 33 image goldens UNTOUCHED.

4. **Tests — extend `tests/shader_graph_test.cpp`:**
   - NormalMap decode parity: `decode(0.5)==0`, `decode(1.0)==1`, `decode(0.0)==-1`; a flat normal-map
     texel `(0.5,0.5,1.0)` decodes to `(0,0,1)` (no perturbation); the node output is normalized.
   - PBROutput.normal default: an unconnected `normal` input yields the documented `(0,0,1)` and the
     codegen for such a graph is byte-identical to the pre-BE codegen (structural assert that no TBN
     block is emitted when normal is unconnected — protects existing-golden invariance).
   - Connected case: codegen for `normalmap.mat.json` emits the TBN transform + uses the perturbed normal
     (structural asserts: TBN/tangent usage present, perturbed normal feeds lighting).
   - Clean under `windows-msvc-asan`.

5. **Runtime-path parity (reuse AW).** `--material-live-shot normalmap.mat.json` runtime-compiles +
   renders (the new node survives the dxc-subprocess runtime path); document the result.

6. **Introspect.** Add exactly `--material-normal-shot` (showcases). NO new feature string (material-graph
   covers it). One-line introspect-golden delta; introspect test updated.

## RHI seam additions (summary)
- **None.** Reuses the existing PBR material pipeline (which already binds a normal-map texture +
  provides tangents). New node/codegen + the example material add ZERO backend symbols. Seam grep stays at
  baseline (2).

## Out of scope (YAGNI)
Procedural/height-to-normal generation, detail/secondary normal blending, parallax/POM, derivative-based
normals, per-pixel tangent reconstruction (use the interpolated vertex tangent like lit.frag),
object-space normal maps, a NormalStrength scalar (could add later — keep the MVP to sample+decode+use).
One tangent-space NormalMap node + PBROutput.normal, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 33) + the extended `shader_graph_test` (NormalMap
   decode parity + PBROutput.normal default/connected codegen). Clean under `windows-msvc-asan`.
2. Build regenerates `shaders/generated/mat_normalmap.frag.hlsl` from `normalmap.mat.json`; compiles
   through the normal HLSL→SPIR-V→MSL pipeline.
3. `--material-normal-shot` on Windows/Vulkan: controller visual review — visible normal-mapped surface
   detail (bumps/relief shading) vs a flat surface, lit + shadowed, coherent. Run under the AT
   Vulkan-validation gate → ZERO errors. Plus the `--material-live-shot normalmap.mat.json` runtime check.
4. Metal: `visual_test --material-normal` → new golden `tests/golden/metal/mat_normal.png`; two runs DIFF
   0.0000.
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `mat_normal.png`
   added; the other 33 byte-identical (CRITICAL: mat_graph/mat_graph2/mat_multi must be byte-identical —
   proves the PBROutput.normal default path didn't change existing codegen).
6. Introspect JSON rebaked exactly `+--material-normal-shot` (showcase only, no new feature); introspect
   test updated; no other drift.
7. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `mat_normal`
   image golden in the Mac round-trip loop.
