# Slice DX — Visibility-Buffer Slice 2: Deferred Material Resolve (Phase 6 #6) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 2nd vis-buffer slice
> (after DW vis-buffer render): a DEFERRED pass texel-fetches the DW vis-buffer per pixel, reconstructs the
> covering cluster+triangle, and SHADES it — Nanite's decoupled geometry→material architecture, where shading
> cost is per-pixel-visible, not per-triangle. **HONEST PROOF SCOPE (the scout established + the controller
> confirms): a byte-identical "resolve == forward render" proof is NOT cross-vendor-feasible (hardware-rasterizer
> barycentric/fill-rule divergence the FP discipline can't close); DX proves ID-provenance bit-exact + GPU==CPU
> resolve-MATH bit-exact + cross-backend via the image golden — NOT byte-identical-to-forward.** Branch:
> slice-dx-visresolve. See [[hazard-forge-visbuffer-roadmap]].

**Goal:** Add a fullscreen deferred-resolve pass that reads the DW R32_Uint vis-buffer, looks up the covering
triangle's vertices, computes a FLAT (per-triangle geometric) normal + Lambert shade, and outputs the lit image —
the spheres rendered via the vis-buffer instead of a forward lit pass. FLAT shading (not barycentric-interpolated)
is the deliberate choice: it keeps the resolve math a deterministic per-triangle value (GPU==CPU bit-exact via the
DH discipline) and sidesteps the cross-vendor perspective-correct-interpolation fragility. Make-safe: a NEW shader
+ NEW showcase + NEW golden; DW's vis-buffer + all 86 goldens untouched.

## Reuse map (file:line)
- **DW `engine/render/visbuffer.h` + the R32_Uint RT** — `PackVisId`/`UnpackVisId`; the vis-buffer is rendered
  exactly as DW (`--visbuffer-shot`), then DX adds the resolve pass over it.
- **The integer-RT bind** — DW's vis-buffer RT already has SAMPLED usage (vulkan_render_target COLOR|SAMPLED|
  TRANSFER_SRC; metal RenderTarget|ShaderRead) and DW gated OFF the float-sampler descriptor for integer formats.
  DX binds it as a sampled texture and reads it via **texel fetch** (`Texture2D<uint>.Load(int3(px,py,0))` →
  `OpImageFetch`, NO sampler; MSL `texture.read(uint2)`). **PREFER NO new RHI** — reuse the existing `BindTexture`
  / material-texture bind path (the integer image + texelFetch needs no sampler). VERIFY this works through the
  existing bind on BOTH backends; if Vulkan validation requires a separate-sampled-image descriptor or Metal
  needs a distinct bind for the uint texture, add the SMALLEST justified additive interface (documented,
  defaulted-no-op, like the R32_Uint addition) — but try the existing path first and report what was needed.
  (Apply the DW Metal lessons: the resolve frag uses SV_PrimitiveID-free integer texelFetch but STILL needs
  `--msl-version 20200` for the integer `texture.read` → add the isolated per-shader MSL-2.2 flag for the resolve
  frag, like DW did for visbuffer.frag.)
- **The cluster/geometry SSBOs** — the resolve needs, per cluster-instance: the model matrix + the cluster's
  `triOffset` (into the reordered index buffer) + the shared vertex + index buffers, all as SSBOs (the DT/DV
  cluster draw already uploads the vertex/index/per-cluster data — reuse those buffers, bind them to the resolve).
- **The DH FP discipline** (probe_gi.h) — `std::fma` + a host-precomputed light direction so the flat-normal +
  Lambert is GPU==CPU bit-exact.

## Design decisions (locked)

1. **`engine/render/visresolve.h` (NEW, pure CPU, `hf::render::vg`; shared by the test + showcases).** The CPU
   mirror of the resolve math: given `(clusterID, triID)` + the cluster's `triOffset` + the index/vertex buffers +
   the model matrix, `ResolveFlatShade(...)` computes the triangle's 3 world positions, the flat geometric normal
   `n = normalize(cross(p1-p0, p2-p0))` (with `std::fma`; document orientation), and the Lambert term `max(0,
   dot(n, -lightDir))` × albedo + ambient (a fixed deterministic light + albedo). Returns the shaded RGB (as the
   exact float bits the shader produces). The shader copies this VERBATIM.

2. **`shaders/visresolve.frag.hlsl` (NEW) + a fullscreen-triangle vert (reuse an existing fullscreen vert if
   present, else a thin one).** Per pixel: `uint v = gVisBuffer.Load(int3(px,py,0))`; `UnpackVisId(v, cid, tid)`;
   if `cid >= drawnClusterCount` (background sentinel) → output the sky/clear color; else fetch the cluster's
   `triOffset`+model from `gClusterMeta[cid]`, the 3 indices `gIndices[3*(triOffset+tid) + {0,1,2}]`, the 3
   vertices `gVerts[...]`, compute the flat normal + Lambert (the verbatim `ResolveFlatShade` math, `mad`), output
   the shaded color. NO per-pixel barycentric interpolation (flat). Texel-fetch the integer vis-buffer (no
   sampler). Bindings: the vis-buffer (sampled uint texture) + `gClusterMeta`/`gIndices`/`gVerts` SSBOs + a UBO
   with `viewProj`/`lightDir`/`drawnClusterCount`.

3. **Showcase `--visresolve-shot <out>` (Vulkan, main.cpp) AND `--visresolve` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm in the diff).** Render the DW vis-buffer (the clustered spheres) → the resolve pass →
   the lit image. The spheres appear FLAT-SHADED (faceted per triangle) lit by the fixed light — the deferred
   resolve payoff (geometry decoupled from shading). PROOFS (fail loudly):
   - **(1) ID-provenance bit-exact (inherited from DW):** at deterministically-chosen INTERIOR pixels per visible
     cluster, the vis-buffer `(clusterID,triID)` the resolve reads == the DW/CPU-predicted covering cluster (the
     integer equality from DW). Print `visresolve ID provenance: <K>/<K> interior pixels EXACT`.
   - **(2) GPU==CPU resolve-math bit-exact:** for those interior pixels, the GPU resolve's shaded RGB ==
     `ResolveFlatShade` CPU mirror over the SAME `(clusterID,triID)` + geometry, bit-for-bit (same-backend; the DV
     "GPU==CPU selected-LOD BIT-EXACT" precedent — read the GPU shaded value back at those pixels via
     `ReadRenderTarget`/`GetCapturedPixels` and memcmp the CPU value). Print `visresolve GPU==CPU shade @interior:
     <K>/<K> EXACT`.
   - **(3) determinism:** two resolve runs byte-identical.
   - **(4) resolve-vs-forward SMOKE (NOT a bit-exact proof — documented):** the resolve image differs from a
     forward flat-shaded render by < a small epsilon (a coarse correctness sanity check; explicitly NOT
     DIFF-0.0000, because cross-vendor rasterizer divergence makes byte-identity infeasible — the scout's finding).
     Print `visresolve vs forward: <maxDiff> < eps (smoke OK)`.
   - **Golden** = the resolved lit image → `tests/golden/metal/visresolve.png` (flat-shaded spheres). Cross-backend
     via the IMAGE golden: same resolve shader both backends + the DH discipline + bit-exact integer IDs → two runs
     DIFF 0.0000, gate on compare.sh EXIT CODE. (The cross-backend guarantee is the same-shader image-golden
     mechanism all goldens use — NOT a compare-to-forward.) Print `visresolve: {survivors:N, shaded:<texels>}`.
     Existing 86 image goldens UNTOUCHED. **GOLDEN DISCIPLINE: ONLY `tests/golden/metal/visresolve.png`.**

4. **Determinism / cross-backend.** The resolve reads bit-exact integer IDs (DW) + computes a deterministic
   per-triangle flat shade (`std::fma` + host-precomputed light dir → GPU==CPU bit-exact, no per-pixel
   interpolation/transcendental). Two runs byte-identical. The image golden is the same-shader cross-backend
   mechanism. **NO byte-identical-to-forward claim** (document why: HW-rasterizer barycentric/fill-rule divergence
   is cross-vendor; even flat shading depends on which triangle covers each edge pixel, which differs cross-vendor
   at edges — hence the interior-only GPU==CPU proof + the image golden + the forward SMOKE bound).

5. **Tests `tests/visresolve_test.cpp` (pure CPU; `hf_add_pure_test`):**
   - `ResolveFlatShade`: a known triangle + light → the expected flat normal + Lambert (hand-computed); the normal
     is unit; back-facing handled (clamp Lambert ≥ 0).
   - The background sentinel path → the sky color (no geometry fetch).
   - Determinism. Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `virtual-geometry-visresolve` (features) + `--visresolve-shot` (showcases).

## RHI seam additions (summary)
- **PREFER none** — reuse `BindTexture` (the integer vis-buffer as a sampled uint texture, texel-fetched, no
  sampler) + `BindStorageBuffer` (cluster meta / index / vertex SSBOs) + the existing fullscreen-pass path. IF the
  integer-texture texel-fetch genuinely requires a new bind on either backend, add the SMALLEST justified additive
  defaulted-no-op interface (document it like the R32_Uint addition; dispatch baseline stays 2; 0 above-seam
  backend symbols). Report exactly what was needed (ideally nothing). The resolve frag needs the isolated
  per-shader `--msl-version 20200` (DW lesson) for the integer `texture.read`.

## Out of scope (YAGNI)
Perspective-correct barycentric ATTRIBUTE interpolation (flat shading first — barycentric is the cross-vendor-
fragile part; a later refinement with the GPU==CPU-interior proof if wanted), full PBR materials / multiple
material IDs in the resolve (one flat Lambert material), software raster (DY), streaming (DZ), a byte-identical
resolve==forward proof (scout-confirmed cross-vendor-infeasible). ONE deferred flat-shaded resolve with
ID-provenance + GPU==CPU resolve-math bit-exact proofs + the resolved image golden + a forward smoke bound.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 86) + new `visresolve_test`. Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--visresolve-shot` on Vulkan: the spheres render FLAT-SHADED lit via the vis-buffer
   (faceted, coherent); `visresolve ID provenance: <K>/<K> EXACT` + `GPU==CPU shade @interior: <K>/<K> EXACT` +
   determinism + the forward SMOKE bound; the `visresolve: {...}` line deterministic. Run under the AT
   Vulkan-validation gate → ZERO errors (the integer texel-fetch bind validation-clean — the DW seam subtlety).
3. Metal: `visual_test --visresolve` → new golden `tests/golden/metal/visresolve.png`; two runs DIFF 0.0000 (gate
   on compare.sh EXIT CODE). The ID-provenance + GPU==CPU proofs also pass. **Confirm visual_test.mm in the diff;
   confirm the resolve frag MSL generates (the --msl-version 20200 isolation) + the integer texel-fetch works on
   Metal.**
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `visresolve.png` added; the
   other 86 byte-identical. `git diff master --stat -- tests/golden` = ONLY `visresolve.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/visresolve.png`, NO other golden changed.
5. Introspect JSON rebaked exactly `+virtual-geometry-visresolve` + `--visresolve-shot`; introspect test updated.
6. **Seam grep:** rhi.h UNCHANGED (or ONLY the smallest justified additive integer-texture bind, documented);
   dispatch baseline 2; 0 above-seam backend symbols. `scripts/verify.ps1` updated to include the new `visresolve`
   image golden in the Mac round-trip loop AND `--visresolve-shot` in the `$vkShots` validation gate.
