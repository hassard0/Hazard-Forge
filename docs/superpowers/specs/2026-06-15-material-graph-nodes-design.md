# Slice AZ — Material-Graph Node Expansion + Multi-Material Scene (Phase 4 #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Closes the AW-noted
> swizzle gap and broadens the material graph (AV/AW) toward real authoring range.

**Goal:** Add the node types the AV/AW MVP lacked — most importantly a **Swizzle/component-extract** node
(so a `float4` TextureSample can drive a scalar port, the exact limitation AW worked around) plus
vector-construct and common math nodes — and demonstrate the graph's range with a **multi-material
scene**: three spheres, each shaded by a DISTINCT graph material, in one frame. New golden + extended
node-semantics tests. Existing 29 goldens stay byte-identical.

## Design decisions (locked)

1. **New node types in `engine/material/shader_graph` (pure CPU, no backend symbols).** Add EXACTLY:
   - `Swizzle` (param: a component mask string over `xyzw`/`rgba`, length 1..4, e.g. `"x"`, `"xyz"`,
     `"wzy"`; input any vector; output type = mask length). This CLOSES the AW gap (extract a scalar/
     subvector from a float4). Type rule: each mask char must index within the input's component count.
   - `MakeFloat3` (3 scalar inputs → float3), `MakeFloat4` (4 scalar inputs → float4) — the construct
     inverse of Swizzle (build a vector from scalars).
   - `Dot` (two same-size vectors → scalar), `Normalize` (vector → same-size unit vector),
     `Power` (base, exponent → component-wise pow), `OneMinus` (x → 1-x, component-wise),
     `Saturate` (x → clamp(x,0,1), component-wise).
   The CPU interpreter AND the codegen must implement IDENTICAL per-node math (share the formula
   definitions, as AV established) so the interpreter test pins shader semantics. Update graph validation
   for the new type rules (swizzle mask bounds, Dot/operand size match, MakeFloatN arity).

2. **Codegen.** `GenerateHlsl` emits the new nodes as the obvious HLSL (`.xyz` swizzles, `dot()`,
   `normalize()`, `pow()`, `1.0 - x`, `saturate()`, `floatN(...)` constructors), topologically ordered as
   before. No template/binding changes — the generated shader still feeds baseColor/metallic/roughness/
   emissive into the shared `pbr_core.hlsli`.

3. **Multi-material scene showcase `--material-multi-shot <out>` (Vulkan) / `--material-multi` (Metal).**
   Three spheres in a row (+ground + sky + the standard light), each rendered with a DISTINCT graph
   material via the existing material pipeline (one draw per material — bind that material's pipeline,
   draw its sphere). Materials:
   - `assets/materials/showcase.mat.json` (AV — fresnel-rim textured, reused),
   - `assets/materials/showcase2.mat.json` (AW — reused),
   - `assets/materials/showcase3.mat.json` (NEW) — exercises the new nodes: e.g.
     `metallic = Swizzle(TextureSample(checker), "x")` (the gap-closing case — scalar from a float4),
     `baseColor = MakeFloat3(Saturate(Power(Fresnel, 2)), Dot(N,V)-ish via available inputs, Constant)`,
     `roughness = OneMinus(Swizzle(TextureSample, "y"))`. Pick a concrete validating graph; document it.
   Deterministic fixed camera/lights. New golden `tests/golden/metal/mat_multi.png` (Metal two runs DIFF
   0.0000). The per-material build-time codegen produces `shaders/generated/mat_showcase3.frag.hlsl`
   (committed, like AV/AW).

4. **Tests — extend `tests/shader_graph_test.cpp`:**
   - Swizzle: mask `"x"` on a float4 yields component 0 as a scalar; `"xyz"` yields the float3 prefix;
     `"ww"` yields a float2 of the w component; out-of-range mask char rejected by validation.
   - MakeFloat3/4: constructs the expected vector; arity mismatch rejected.
   - Dot/Normalize/Power/OneMinus/Saturate: CPU-interpreter parity vs hand-computed values at sample
     points (e.g. `Dot([1,2,3],[4,5,6])==32`, `Saturate(1.5)==1`, `OneMinus(0.25)==0.75`,
     `Power(2,3)==8`, `Normalize([3,0,4])==[0.6,0,0.8]`).
   - Codegen determinism on showcase3 + the generated HLSL contains the expected swizzle/intrinsics
     (structural asserts).
   - Clean under `windows-msvc-asan`.

5. **Runtime path parity.** Since AW's runtime compiler reuses the same codegen, `--material-live-shot
   showcase3.mat.json` on Vulkan must render IDENTICALLY to its build-time `mat_multi` contribution for
   that material — assert showcase3 via the live path matches the build-time render (SHA on an isolated
   single-sphere capture, OR just confirm the runtime compile of showcase3 succeeds + renders; document
   the chosen check). (Reuse, no new runtime code.)

6. **Introspect.** Add exactly `--material-multi-shot` (showcases). NO new feature string (the
   `material-graph` feature already covers it). One-line introspect-golden delta; introspect test updated.

## RHI seam additions (summary)
- **None.** All new nodes + codegen + the multi-material draw reuse the existing material pipeline path.
  New/changed files (`engine/material/*`, `assets/materials/showcase3.mat.json`, generated HLSL,
  `tests/shader_graph_test.cpp`) add ZERO backend symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
A NormalMap / tangent-space-perturbation node (visually rich but needs tangent plumbing — a future
slice), Time/animated nodes, noise/procedural-texture nodes, a visual node editor, per-instance material
parameters, conditional/branch nodes, custom-HLSL-expression nodes. Just the swizzle/construct/math node
set + a three-material scene, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 30) + the extended `shader_graph_test` (new-node
   parity + swizzle/MakeFloatN/Dot validation). Clean under `windows-msvc-asan`.
2. Build regenerates `shaders/generated/mat_showcase3.frag.hlsl` from `showcase3.mat.json`; it compiles
   through the normal HLSL→SPIR-V→MSL pipeline.
3. `--material-multi-shot` on Windows/Vulkan: controller visual review — three spheres with three
   visibly-distinct graph materials, lit + shadowed, coherent. Run under the AT Vulkan-validation gate →
   ZERO errors.
4. Metal: `visual_test --material-multi` → new golden `tests/golden/metal/mat_multi.png`; two runs DIFF
   0.0000.
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `mat_multi.png`
   added; the other 29 byte-identical (incl. mat_graph.png + mat_graph2.png from AV/AW).
6. Introspect JSON rebaked exactly `+--material-multi-shot` (showcase only, no new feature); introspect
   test updated; no other drift.
7. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `mat_multi`
   image golden in the Mac round-trip loop.
