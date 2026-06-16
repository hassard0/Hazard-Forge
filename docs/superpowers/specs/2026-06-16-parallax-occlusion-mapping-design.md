# Slice CP — Parallax Occlusion Mapping (Phase 4 #40) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A genuinely-absent
> material technique adding real per-pixel surface depth + self-occlusion, with a clean exact equivalence
> proof (heightScale=0 → the march collapses to the base UV → byte-identical to plain normal mapping).

**Goal:** Parallax Occlusion Mapping — a surface displays apparent depth (bricks recessed, stones raised)
by ray-marching a height field in tangent space and shading the displaced texel, including self-occlusion
at grazing angles. A `--pom-shot` showcase shows a height-mapped surface with clear parallax depth viewed at
an angle; golden-verified on both backends. The pass carries its proof: with `heightScale == 0` the marched
UV equals the base UV exactly, so the POM surface renders BYTE-IDENTICAL to the plain (non-POM, normal-mapped)
surface — asserted internally, fail loudly on any diff.

## Why this is render-safe (the zero-height equivalence proof)

POM offsets the sampled UV by walking the view ray through a height field of amplitude `heightScale`. When
`heightScale == 0` the height field is the flat base plane, the ray intersects it at the entry point, and the
offset UV == the base UV for every pixel → the shaded result is identical to sampling the surface with no
parallax (plain normal mapping). So the showcase INTERNALLY renders the same surface with `heightScale = 0`
and asserts it is BYTE-IDENTICAL (SHA) to the engine's plain lit/normal-mapped render of that surface —
proving POM is a pure pass-through at zero amplitude (no UV drift, no off-by-one in the march/refine) — then
renders the real `heightScale > 0` version as the golden. (Same internal-assert discipline as CN motion-blur
zero-velocity and CO order-independence.)

## Design decisions (locked)

1. **POM math (engine/render/pom.h, header-only pure CPU, no backend symbols).** Namespace `hf::render::pom`.
   Mirrors `dof.h`/`motion_blur.h`/`oit.h` (shared with the shader + the unit test).
   - `float HeightAt(math::Vec2 uv, ...)` is provided by the CALLER (the test uses a deterministic procedural
     height; the shader samples a height texture). The march takes a sampled-height callback / inline form;
     document the height convention (height ∈ [0,1], 1 = surface top, the ray descends into [0,1] depth).
   - `math::Vec2 ParallaxUV(math::Vec2 baseUV, math::Vec3 viewDirTangent, float heightScale, int numSteps,
     <height-sampler>)` — steep-parallax linear march: step the ray from depth 0 downward in `numSteps`
     layers along the tangent-space view direction (scaled by `heightScale`), find the first layer where the
     ray depth exceeds the sampled height (the intersection bracket), then a fixed binary-search refine
     between the last two layers. Returns the UV at the refined intersection. At `heightScale == 0`, OR a
     straight-on view (viewDirTangent.xy == 0), returns `baseUV` EXACTLY (document both degenerate exits).
   - `float SelfShadow(math::Vec2 uv, math::Vec3 lightDirTangent, float heightScale, int numSteps,
     <height-sampler>)` — OPTIONAL soft self-occlusion toward the light (march toward the light, attenuate if
     the height field blocks it). Document; if included, unit-test it; if omitted, note YAGNI. (Recommended
     included — it's the "occlusion" in POM and a strong visual.)
   - Pure functions, deterministic, no RNG, no time. Document the march + refine + the exact degenerate exits.

2. **POM shader `shaders/pom.frag.hlsl` (NEW lit surface variant).** A lit surface that: builds the
   tangent-space basis (TBN) from the surface normal + tangent (compute from UV gradients or a provided
   tangent — document), transforms the view dir to tangent space, calls the `pom::ParallaxUV` march (sampling
   a height texture at t-slot), samples albedo + normal at the parallax-offset UV, applies the optional
   `pom::SelfShadow`, and lights it with the existing directional sun + ambient/IBL (matching the default lit
   path so heightScale=0 is byte-identical). The DEFAULT lit path + its goldens stay BYTE-IDENTICAL (POM is a
   NEW path behind the showcase flag). Bindings reuse the existing albedo/normal/height texture-pair path.
   No new RHI seam. HLSL→SPIR-V→MSL via the existing toolchain.

3. **Showcase `--pom-shot <out>` (Vulkan) / `--pom` (Metal).** A height-mapped quad/surface (a procedural
   brick or tiled-stone height field — deterministic, generated in-engine or a committed small height
   texture) filling the view, lit by the sun, viewed at a GRAZING angle so the parallax depth + self-occlusion
   are obvious (recessed mortar, raised bricks, shadowing in the crevices). Fixed camera, fixed heightScale,
   fixed steps. Print `pom: {heightScale:H, steps:N}` (deterministic). INTERNALLY render the SAME surface with
   `heightScale = 0` and assert it is BYTE-IDENTICAL (SHA) to the plain normal-mapped lit render — fail loudly
   on any diff. New golden `tests/golden/metal/pom.png` (Metal two runs DIFF 0.0000). Existing 60 image
   goldens UNTOUCHED.

4. **Determinism.** Fixed camera/heightScale/steps, deterministic procedural height (no RNG), fixed
   march+refine counts. Two runs byte-identical.

5. **Tests `tests/pom_test.cpp` (pure CPU, no GPU):**
   - **Zero height = identity:** `ParallaxUV(uv, viewDir, heightScale=0, ...) == uv` exactly for any uv/view;
     and a straight-on view (viewDirTangent.xy == 0) → `uv` exactly.
   - **Offset direction + magnitude:** with `heightScale > 0` and a grazing view, the returned UV is offset
     from `baseUV` in the tangent-space view direction; a deeper height (lower sampled value) → larger offset;
     a steeper grazing angle → larger offset; monotone.
   - **March intersection correctness:** the refined intersection point lies ON the height field within the
     march tolerance (the ray depth at the returned UV ≈ the sampled height there) — no overshoot past the
     surface.
   - **SelfShadow (if included):** a point in a crevice with the light occluded by a ridge → shadow factor < 1;
     an exposed point → 1; monotone in heightScale.
   - **Determinism:** same inputs → same UV/shadow.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `parallax-occlusion-mapping` (features) + `--pom-shot` (showcases).

## RHI seam additions (summary)
- **None.** POM is a fragment-shader technique reading a height texture via the existing texture-pair binding
  (like the normal/albedo maps) + the existing lit pass. New files (`engine/render/pom.h`,
  `shaders/pom.frag.hlsl`, `tests/pom_test.cpp`, + an optional committed small height texture or a procedural
  height in-shader) add ZERO backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Tessellation/true displacement (this is the per-pixel parallax approximation), silhouette/contour POM (no
edge displacement — the quad outline stays flat), relief mapping's curved-surface correction, parallax on
arbitrary curved meshes (a flat tangent-plane surface is sufficient for the showcase + proof), multi-layer
material height blending, POM in the shadow pass. One steep-parallax + binary-refine POM surface variant with
optional self-shadow and a zero-height byte-identical proof, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 60) + new `pom_test` (zero-height identity, offset
   direction/magnitude, march intersection correctness, self-shadow if included, determinism). Clean under
   `windows-msvc-asan`.
2. **Zero-height equivalence proof + visual:** `--pom-shot` on Vulkan: the surface shows clear parallax depth
   + self-occlusion at the grazing angle (recessed/raised detail, crevice shadowing); the INTERNAL
   heightScale=0 render is BYTE-IDENTICAL (SHA) to the plain normal-mapped lit render; the `pom: {...}` line
   is deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --pom` → new golden `tests/golden/metal/pom.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `pom.png` added; the
   other 60 byte-identical.
5. Introspect JSON rebaked exactly `+parallax-occlusion-mapping` + `--pom-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `pom` image golden
   in the Mac round-trip loop.
