# Slice BN — Data-Driven Post-Process Stack (Phase 4 #15) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Systematizes post
> effects into a configurable, agentic (data-driven) ordered chain + adds new composable effects.

**Goal:** A configurable, ordered post-process stack — a data-driven chain of effects applied in sequence
to the resolved scene color. Composes the engine's existing tonemap/vignette with NEW deterministic
effects (color grade, chromatic aberration, film grain). The stack config is data (JSON-authorable, so an
agent can compose looks). A `--poststack-shot` showcase applies a fixed configured stack to the standard
scene; golden-verified on both backends.

## Design decisions (locked)

1. **Stack model (engine/render/post_stack.{h,cpp}, pure CPU config + orchestration, no backend symbols).**
   Namespace `hf::render`. A `PostStack` = an ordered `std::vector<PostEffect>`, each
   `PostEffect { enum class Kind { Tonemap, ColorGrade, ChromaticAberration, FilmGrain, Vignette };
   /* per-effect params */ };`. Provide a JSON loader (`LoadPostStack(json)`) + a default stack. The
   renderer applies the enabled effects IN ORDER in the post pass. A pure-CPU evaluator of each effect's
   per-pixel math is shared with the shader (so the math is unit-testable, mirroring `ssr.h`/`decal.h`).

2. **New deterministic effects (the substance — NOT just reordering).** All clock/RNG-free:
   - `ColorGrade` (lift/gamma/gain per channel): `out = gain * pow(max(in + lift, 0), 1/gamma)`. Params:
     `float3 lift, gamma, gain`. (A neutral identity = lift 0, gamma 1, gain 1.)
   - `ChromaticAberration`: sample R/G/B at slightly offset UVs along the radial direction from screen
     center, scaled by `strength` (in pixels). Deterministic per-pixel.
   - `FilmGrain`: add `(hash(pixelCoord) - 0.5) * intensity` luminance noise, where `hash` is a FIXED
     integer hash of the integer pixel coordinate (NO time/frame input → deterministic + golden-stable).
     Params: `float intensity`.
   - Reuse the existing `Tonemap` + `Vignette` (already in `post.frag.hlsl`) as stack entries.

3. **Shader.** Extend the existing post pass / `post.frag.hlsl` (or a new `post_stack.frag.hlsl`) to apply
   the configured chain. The configured stack (which effects, their params, order) arrives via a uniform/
   push constant (a fixed-size effect list + an enabled-count, applied in order). Keep the EXISTING
   `--shot`/post path BYTE-IDENTICAL (the stack showcase is a NEW path; the default scene's post is
   unchanged — verify existing goldens stay byte-identical). HLSL→SPIR-V→MSL via the existing toolchain.

4. **Showcase `--poststack-shot <out>` (Vulkan) / `--poststack` (Metal).** Render the standard
   lit+shadowed scene, then apply a FIXED configured stack (e.g. Tonemap → ColorGrade(a warm teal-orange
   grade) → ChromaticAberration(subtle) → Vignette → FilmGrain(subtle)) → capture. Print
   `poststack: {effects:[tonemap,colorgrade,chromatic,vignette,grain], count:5}`. New golden
   `tests/golden/metal/poststack.png` (Metal two runs DIFF 0.0000). Existing 39 image goldens UNTOUCHED.

5. **Tests `tests/post_stack_test.cpp` (pure CPU, no GPU):**
   - **Config:** `LoadPostStack` parses an ordered effect list + params; an empty/disabled stack is a
     pass-through; order is preserved.
   - **ColorGrade math:** identity params → input unchanged; a known lift/gamma/gain → expected output at
     sample colors.
   - **FilmGrain determinism:** `grain(pixel)` is a pure function of the pixel coord (same coord → same
     value across calls); zero intensity → no change; the hash distribution is bounded `[-0.5,0.5]`.
   - **ChromaticAberration:** zero strength → no offset; center pixel → ~no aberration (radial offset 0).
   - **Stack apply (CPU mirror):** applying the stack to a sample pixel equals composing the per-effect
     evaluators in order.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `post-process-stack` (features) + `--poststack-shot` (showcases).

## RHI seam additions (summary)
- **None.** The stack is config + orchestration over the existing post pass; the new effects are shader
  math. New files (`engine/render/post_stack.{h,cpp}`, `tests/post_stack_test.cpp`, optionally
  `shaders/post_stack.frag.hlsl`) add ZERO backend symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
A node-graph post editor, LUT-texture color grading, depth-of-field, motion blur, lens flares, auto
exposure / eye adaptation, bloom/SSAO/SSR re-plumbed INTO the stack (they remain their own passes — the
stack is the final composite chain; integrating them is a future slice), per-effect live tweak UI,
temporal effects. One fixed-size ordered composite chain + 3 new effects, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 39) + new `post_stack_test` (config, ColorGrade,
   FilmGrain determinism, ChromaticAberration, stack-apply). Clean under `windows-msvc-asan`.
2. `--poststack-shot` on Windows/Vulkan: controller visual review — the scene with the graded/aberrated/
   vignetted/grained look, clearly distinct from the plain `--shot` but coherent; the `poststack: {...}`
   line is deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation gate →
   ZERO errors.
3. Metal: `visual_test --poststack` → new golden `tests/golden/metal/poststack.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `poststack.png` added;
   the other 39 byte-identical (CRITICAL: the existing `--shot`/`bloom`/etc post paths unchanged → their
   goldens byte-identical — the stack is a separate path).
5. Introspect JSON rebaked exactly `+post-process-stack` + `--poststack-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `poststack`
   image golden in the Mac round-trip loop.
