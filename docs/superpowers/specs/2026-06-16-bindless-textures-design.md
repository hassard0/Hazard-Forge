# Slice BZ — Bindless Textures (Phase 4 #25) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. GPU-driven rendering;
> pairs with MDI (BM). Render-invariant by construction (bindless image == bound image).

**Goal:** Sample scene textures through a single large descriptor ARRAY indexed per-draw, instead of
rebinding a descriptor set per material. On Vulkan: a runtime-sized descriptor array of all scene textures
(`VK_EXT_descriptor_indexing` / `descriptorBindingPartiallyBound` + `runtimeDescriptorArray` +
`shaderSampledImageArrayNonUniformIndexing`, core-ish in Vulkan 1.2/1.3), bound ONCE; each draw's material
carries a texture INDEX; the shader samples `textures[NonUniformResourceIndex(idx)]`. The result must be
BYTE-IDENTICAL to the same scene drawn with per-material bound textures — that equality (plus "one texture
binding for the whole scene") is the proof. This completes the GPU-driven story (MDI for draws + bindless
for textures).

## Cross-backend framing (de-risked, AW/BM-style asymmetry)

True bindless via descriptor-indexing is the Vulkan demonstration. Metal's native equivalent is an
argument buffer of texture handles — heavier to wire headless. To keep the slice tractable + the golden
robust:
- **Vulkan** does TRUE bindless and PROVES it: `--bindless-shot` (bindless path) is byte-identical to the
  same scene via the existing per-material bound path (`--bindless-shot-ref`, an internal reference) — SHA
  match. Report the texture-binding count (1 array bind vs N per-material binds).
- **Metal** renders the IDENTICAL scene for the golden via its working per-material BOUND path (the image
  is the same geometry/textures → `bindless.png` matches across backends). Optionally implement the Metal
  argument-buffer path if straightforward, but it is NOT required — document honestly (bindless is the
  Vulkan GPU-driven demonstration; the image is backend-identical).

## Design decisions (locked)

1. **Texture index table (engine/render/bindless.h, pure CPU, no backend symbols).** Namespace
   `hf::render::bindless`. A deterministic builder that assigns each UNIQUE scene texture a stable index
   and maps each material → its texture index(es):
   - `struct BindlessTable { std::vector<ITexture*> textures; /* index -> texture */
     std::unordered_map<ITexture*, uint32_t> indexOf; };` (or a vector + a lookup).
   - `uint32_t Intern(BindlessTable&, ITexture*)` — returns the stable index for a texture (adds it if
     new); deterministic insertion order. The material's `baseColor`/`normalMap` map to indices.
   - Pure CPU (pointers are opaque handles, never dereferenced); unit-testable (interning is stable +
     deduplicates; the table covers all scene materials' textures).

2. **RHI: a bindless descriptor array (Vulkan) — additive, minimal.** Add a way to create + bind a large
   sampled-image array and per-draw texture indices. Concretely (adapt to the existing RHI style):
   - `IRHIDevice::CreateBindlessTextureSet(span<ITexture*>)` → an opaque handle for a descriptor set with a
     runtime array of all the textures (Vulkan: a set with `VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT` +
     `VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT`, the array filled with the textures + a shared
     sampler). Enable the `descriptorIndexing` features on the device (confirm available; VK 1.2/1.3).
   - `ICommandBuffer::BindBindlessTextures(handle)` (bind the array set once) + the per-draw texture index
     arrives via the existing per-draw/material push constant (a `uint texIndex`).
   - Metal impl: no-op/fallback (the Metal golden uses the bound path) — document.
   - All `vk*`/`MTL*` inside backend dirs; the seam exposes only abstract handles/indices.

3. **Shader.** A lit variant (`lit_bindless.frag.hlsl` or extend the PBR frag) that declares
   `[[vk::binding(0, N)]] Texture2D gTextures[] : register(t0, spaceN);` (unbounded array) + a shared
   sampler, and samples `gTextures[NonUniformResourceIndex(texIndex)]` where `texIndex` is the push-constant
   material index. The bound reference path uses the EXISTING lit/PBR frag with the same texture → the two
   paths sample the SAME texels → byte-identical image. HLSL→SPIR-V via DXC must emit the
   `NonUniformResourceIndex` / `SPV_EXT_descriptor_indexing` capability — confirm the toolchain does.

4. **Showcase `--bindless-shot <out>` (Vulkan) / `--bindless` (Metal).** A scene with SEVERAL distinct
   textures (e.g. the default multi-material scene: checker, normalmap, duck_basecolor, etc.) rendered via
   the bindless array (Vulkan) / bound path (Metal), lit + shadowed, fixed camera. Print `bindless:
   {textures:T, draws:D, textureBinds:1, refTextureBinds:T}` (Vulkan) — and INTERNALLY render the
   per-material BOUND reference + assert `bindlessImage == refImage` byte-identical (fail loudly on
   mismatch). New golden `tests/golden/metal/bindless.png` (Metal two runs DIFF 0.0000). Existing 48 image
   goldens UNTOUCHED.

5. **Tests `tests/bindless_test.cpp` (pure CPU, no GPU):**
   - **Interning:** `Intern` returns stable indices, deduplicates identical textures, assigns distinct
     indices to distinct textures, insertion order deterministic.
   - **Material coverage:** building the table over a known material set covers every texture; each
     material resolves to the right index.
   - **Determinism:** two builds → identical index assignment.
   - Clean under `windows-msvc-asan`.

## RHI seam additions (summary)
- `CreateBindlessTextureSet(span<ITexture*>)` + `BindBindlessTextures(handle)` + a per-draw `texIndex`
  push-constant field — additive pure-interface; Vulkan impl (descriptor-indexing set) inside
  `rhi_vulkan/`; Metal impl no-op/fallback inside `rhi_metal/`. New non-backend files
  (`engine/render/bindless.h`, `tests/bindless_test.cpp`, `shaders/lit_bindless.frag.hlsl`) add ZERO
  backend code symbols. Seam grep stays at baseline (the `vk*`/builtins live inside the backend dir;
  comments tolerated).

## Out of scope (YAGNI)
Bindless BUFFERS / bindless everything, mutating the array mid-frame, descriptor streaming/eviction,
Metal argument-buffer bindless (optional, not required), combining bindless with MDI into one
fully-GPU-driven pass (a natural future combo — note it), texture residency/virtual texturing. One
bindless sampled-image array for a fixed scene, byte-identical to the bound path, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 48) + new `bindless_test` (interning, material
   coverage, determinism). Clean under `windows-msvc-asan`.
2. **bindless==bound proof:** `--bindless-shot` on Vulkan renders via ONE bindless array bind and the
   captured image is BYTE-IDENTICAL (SHA) to the internal per-material BOUND reference render of the same
   scene; the run asserts this + prints `bindless: {textures:T, draws:D, textureBinds:1, refTextureBinds:T}`.
   Two runs identical.
3. `--bindless-shot` visual review (controller): the multi-texture scene renders correctly, lit + shadowed,
   coherent. Run under the AT Vulkan-validation gate → ZERO errors (descriptor-indexing + NonUniform must
   be validation-clean).
4. Metal: `visual_test --bindless` → new golden `tests/golden/metal/bindless.png`; two runs DIFF 0.0000.
   (Document whether Metal used the argument buffer or the bound path; the image is backend-identical.)
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `bindless.png` added;
   the other 48 byte-identical.
6. Introspect JSON rebaked exactly `+bindless-textures` + `--bindless-shot`; introspect test updated; no
   other drift.
7. Seam grep clean (no new above-seam code symbols). `scripts/verify.ps1` updated to include the new
   `bindless` image golden in the Mac round-trip loop.
