# Slice CO — Order-Independent Transparency (Weighted Blended OIT) (Phase 4 #39) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A genuinely-new
> capability: correct transparency WITHOUT depth-sorting, proven by a draw-ORDER-INDEPENDENCE byte-identical
> equivalence (the airtight verification, like CL clustered==brute-force / CJ occlusion==frustum).

**Goal:** Render overlapping transparent surfaces correctly regardless of draw order, using Weighted Blended
OIT (McGuire & Bavoil 2013): accumulate `(premultColor * w, alpha * w)` additively into an RGBA16F accum
target and `alpha` multiplicatively into an R revealage target, then a fullscreen resolve composites
`accum.rgb / max(accum.a, eps)` over the background weighted by `(1 - revealage)`. A `--oit-shot` showcase
shows several intersecting/overlapping transparent objects composited correctly; golden-verified on both
backends. The pass carries its defining proof: rendering the transparent set in TWO DIFFERENT draw orders
yields a BYTE-IDENTICAL resolved image (order independence) — asserted internally, fail loudly on any diff.

## Why this is render-safe (the order-independence equivalence proof)

WBOIT's accumulation is commutative: the accum target is a SUM (`Σ premultColor*w`, `Σ alpha*w`) and the
revealage is a PRODUCT (`Π (1-alpha)`) — both order-independent by construction (floating-point summation in
the SAME set of values; the showcase uses a fixed, finite, well-separated weight set so the FP sum is
associative-stable, OR the resolve is exact for the chosen values — document). So rendering the same N
transparent objects in order `[0,1,2,3,...]` vs a PERMUTED order `[3,1,0,2,...]` produces the IDENTICAL
accum/revealage and therefore a BYTE-IDENTICAL resolved image. The showcase INTERNALLY renders both orders
and asserts SHA-equality — proving true order independence (the entire point of OIT). (Same internal-assert
discipline as CL/CJ/CN.) NOTE on FP determinism: to keep the two-order sum bit-exact, accumulate with a
deterministic per-object weight and either (a) choose object alphas/weights whose premultiplied contributions
sum without rounding divergence across orders, or (b) document that the additive blend hardware path is
order-stable for these inputs and verify it empirically (the internal assert is the gate — if it ever
differs, the showcase fails loudly rather than baking a bad golden).

## Design decisions (locked)

1. **OIT math (engine/render/oit.h, header-only pure CPU, no backend symbols).** Namespace `hf::render::oit`.
   Mirrors `dof.h`/`cluster.h`/`motion_blur.h` (shared with the shader + the unit test).
   - `float Weight(float viewDepth, float alpha)` — the McGuire depth-based weight (a smooth function that
     down-weights distant fragments to reduce color bleeding; use the published `w = alpha * clamp(0.03 /
     (1e-5 + (z/200)^4), 1e-2, 3e3)` form OR a simpler documented monotone-in-depth variant; document the
     exact formula + the depth units). Positive, finite, decreasing in depth.
   - `math::Vec4 ResolveOver(const math::Vec4& accum, float revealage, const math::Vec3& background)` — the
     resolve composite: `transparentRGB = accum.rgb / max(accum.a, 1e-4)`; `out = transparentRGB * (1 -
     revealage) + background * revealage`. Returns the final RGB (+ document the alpha).
   - `void Accumulate(const math::Vec4& premultColorAlpha, float weight, math::Vec4& accum, float&
     revealage)` — the per-fragment accumulation: `accum += vec4(premultColor*weight, alpha*weight)`;
     `revealage *= (1 - alpha)`. Order-independent (sum + product). Shared with the shader + the test's
     order-permutation check.

2. **OIT shaders.** `shaders/oit_accum.frag.hlsl` — the transparent-geometry pass writing TWO MRT outputs
   (RT0 = RGBA16F accum, RT1 = R revealage) with the blend states: RT0 additive (`ONE, ONE`), RT1
   multiplicative-reveal (`ZERO, ONE_MINUS_SRC` or the documented WBOIT revealage blend). Computes `Weight`
   from view depth (matching `oit::Weight`). `shaders/oit_resolve.frag.hlsl` — a fullscreen pass reading the
   accum + revealage targets, computing `oit::ResolveOver` over the opaque scene color. EXISTING opaque/scene
   shaders + their goldens UNTOUCHED (OIT is a NEW path behind the showcase flag; the default forward
   transparency path, if any, is unchanged). HLSL→SPIR-V→MSL via the existing toolchain. Reuses the existing
   MRT + float-RT support (G-buffer / HDR pipeline already use MRT + RGBA16F).

3. **Showcase `--oit-shot <out>` (Vulkan) / `--oit` (Metal).** A scene (opaque ground + objects) with
   SEVERAL overlapping/intersecting TRANSPARENT objects at different depths (e.g. 5 colored glass
   quads/spheres mutually overlapping). Pipeline: render opaque → render the transparent set into the WBOIT
   accum+revealage MRT → resolve over the opaque scene. Fixed camera. Print `oit: {layers:N, orderIndependent:true}`.
   INTERNALLY render the transparent set in a PERMUTED draw order and assert the resolved image is
   BYTE-IDENTICAL (SHA) to the canonical-order render — fail loudly on any diff. New golden
   `tests/golden/metal/oit.png` (Metal two runs DIFF 0.0000). Existing 59 image goldens UNTOUCHED.

4. **Determinism.** Fixed scene/camera, fixed transparent set + alphas, fixed weights. Two runs byte-identical
   AND the two draw orders byte-identical (the order-independence proof).

5. **Tests `tests/oit_test.cpp` (pure CPU, no GPU):**
   - **Weight properties:** `Weight` positive + finite for all inputs; decreasing in depth (a nearer fragment
     weighted ≥ a farther one at equal alpha); handles alpha 0..1.
   - **Order independence (the core):** accumulate a set of fragments in order A then resolve; accumulate the
     SAME set in a PERMUTED order then resolve; assert the resolved color is BIT-IDENTICAL (the sum+product
     are order-independent for the test's chosen values). Multiple permutations.
   - **Resolve correctness:** a single opaque-ish fragment (alpha≈1) resolves ≈ its own color; a single
     near-transparent fragment (alpha≈0) resolves ≈ the background; revealage 1 (no fragments) → background
     exactly.
   - **Determinism:** same fragments → same resolve.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `order-independent-transparency` (features) + `--oit-shot` (showcases).

## RHI seam additions (summary)
- **None expected.** Reuses the existing MRT + RGBA16F render-target + fullscreen-resolve path (the
  G-buffer/HDR pipeline already binds multiple float RTs) + per-attachment blend state (already used by the
  bloom/transparency paths). IF a per-attachment blend-state knob is genuinely missing from the RHI, add it
  as an ADDITIVE pure-interface field in `rhi.h` with the impl INSIDE the backend dirs (document it) — but
  PREFER reusing the existing blend-state plumbing (no new seam). New non-backend files (`engine/render/oit.h`,
  the two shaders, `tests/oit_test.cpp`) add ZERO above-seam backend code symbols. Report the seam result.

## Out of scope (YAGNI)
Depth-sorted/per-pixel-linked-list OIT (this is the weighted-blended approximation — the modern lightweight
standard), dual-depth-peeling, transparent shadows, refraction through the transparent layers, more than a
handful of layers, MSAA-resolved OIT, transparent-on-transparent SSR. One weighted-blended OIT accum+resolve
pass with a draw-order-independence byte-identical proof, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 59) + new `oit_test` (weight properties, ORDER
   INDEPENDENCE bit-identical resolve over permutations, resolve correctness, determinism). Clean under
   `windows-msvc-asan`.
2. **Order-independence proof + visual:** `--oit-shot` on Vulkan: the overlapping transparent objects
   composite correctly (you can see through each layer, colors blend plausibly), coherent; the INTERNAL
   permuted-draw-order render is BYTE-IDENTICAL (SHA) to the canonical order; the `oit: {...}` line is
   deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --oit` → new golden `tests/golden/metal/oit.png`; two runs DIFF 0.0000.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `oit.png` added; the
   other 59 byte-identical.
5. Introspect JSON rebaked exactly `+order-independent-transparency` + `--oit-shot`; introspect test updated;
   no other drift.
6. Seam grep clean (no new above-seam code symbols; if an additive blend-state field was required, it lives
   in rhi.h as a pure interface with backend-dir impls — report it). `scripts/verify.ps1` updated to include
   the new `oit` image golden in the Mac round-trip loop.
