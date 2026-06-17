# Slice DO — DDGI Visibility Slice 1: Per-Probe Distance-Moment Capture (Phase 5 #6) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. **First slice of the DDGI
> VISIBILITY sub-arc** — the data layer that makes the next slice (DP, Chebyshev occlusion weighting) able to
> KILL DDGI light-leak through walls (the #1 quality gap of plain irradiance-probe GI). DO captures, per probe,
> the DISTANCE to surrounding geometry in every direction as two moments (meanDist, meanDist²) — exactly the
> structure the Chebyshev/variance visibility test consumes. NO new RHI; mirrors DI (radiance capture) with a
> distance-output instead of radiance. Branch: slice-do-probedist.

**Goal:** Add a per-probe **distance-moment cube** capture: each probe renders the scene from its centre into
the 6 faces of the DD cube RT writing **linear world-distance** (probe-centre → fragment) instead of radiance,
read back into a per-probe distance-moment store of `[meanDist, meanDist²]` per face-texel. This is the visibility
data layer; it produces NO visible lighting change yet (DP consumes it). Make-safe: a dedicated `--probedist`
showcase + golden; all 78 existing goldens byte-untouched (new shader + new showcase only).

## Reuse map (mirror DI exactly — DI:radiance :: DO:distance)
- **DI `engine/render/probe_capture.h`** — the per-probe cube-capture LOOP (single DD `ICubemapTarget`, looped
  one probe at a time: `BeginCubemapFace`×6 render from `probePos` with `cubemap::FaceView(f,centre)`+`FaceProj`,
  `EndCubemapFace`, `ReadCubemapFace` into a per-probe store). DO reuses this loop verbatim; only the fragment
  output (distance, not lit radiance) and the read-back target (a distance-moment store, not a radiance store)
  differ.
- **DD cubemap RHI** (`ICubemapTarget`/`CreateCubemapTarget`/`BeginCubemapFace`/`EndCubemapFace`/`ReadCubemapFace`)
  — UNCHANGED, reused. **NO new RHI.**
- **DD `engine/render/cubemap.h`** — `FaceView`/`FaceProj`/direction↔face-uv math (DP will query the moment cube
  in the probe→point direction using this SAME math; DO just needs the capture views).
- **DH cross-backend FP discipline** — any host-side moment accumulation that must be GPU==CPU bit-exact uses
  host-precomputed inputs + `std::fma`/`mad` (see the proof section).

## Design decisions (locked)

1. **New shader `shaders/probe_dist.frag.hlsl` (+ its vert, or reuse the DD `probe_bake.vert`).** Renders the
   scene geometry from the probe centre and outputs, as the colour attachment, the **linear distance**
   `length(worldPos - gProbeCentre)` packed as `float2(d, d*d)` into an RG (or RGBA16F, B=A=0) target — i.e. the
   per-fragment distance and its square (the two moments at single-sample granularity; a texel's stored moment is
   that texel's centre-ray distance + its square). The probe centre rides a small per-probe uniform (reuse the
   capture pass's existing per-face push/UBO that already carries `FaceView`/`FaceProj`; add `float3 probeCentre`).
   No lighting, no shadows — pure geometric distance. `kRayMiss`-style far value (`1e4f`) where no geometry (sky).
   Document: distance is WORLD-space, view-independent per probe → cross-backend identical given identical mesh.

2. **`engine/render/probe_dist.h` (NEW, pure CPU, namespace `hf::render::probedist`; model on probe_capture.h).**
   - `struct ProbeDistMoments { float m[2]; }` (meanDist, meanDist²); `sizeof==8`, std430 `static_assert`. Per
     probe a small cube of `kDistFace = 8` → `8*8*6 = 384` texels → `ProbeDistMoments[probeCount * 384]`. Flat
     SSBO; probe `p` face `f` texel `(u,v)` at `p*384 + f*64 + v*8 + u` (document the layout; mirror DI's
     `ProbeFaceIndex`). `DistTexelCount(grid) = probeCount*384` (0 at probeCount==0).
   - `ProbeDistFaceIndex(p,f)`, helpers mirroring DI.
   - `MomentsFromDistance(float d) -> ProbeDistMoments{ d, d*d }` (the per-texel moment; the GPU writes the same
     via the shader, the CPU reference via this fn — used by the bit-exact proof). Use `std::fma`-free here (a
     plain `d*d` is bit-identical CPU↔GPU for a single multiply with no contraction; DO NOT introduce an `fma`
     where a bare multiply is exact — document why: `d*d` is one rounding, matches `mad(d,d,0)`? NO — prefer the
     shader also emit `d*d` as a bare multiply so both sides do ONE multiply-rounding. Document the exact-match
     reasoning explicitly).
   - **Cross-backend note:** `length(worldPos-centre)` = `sqrt(dot(...))`; `sqrt` is NOT guaranteed bit-identical
     CPU-libm vs GPU (the DH lesson). So the **GPU==CPU bit-exact proof is on the MOMENT-FROM-DISTANCE step**
     (given the SAME distance bytes, GPU `float2(d,d*d)` == CPU `MomentsFromDistance(d)`), NOT on the `sqrt`
     itself. The capture-distance proof is a RENDER-EQUIVALENCE proof (DI pattern): the captured distance face ==
     a direct distance render with that face's view/proj, byte-identical (same shader, same mesh, per backend).
     Be explicit about which proof covers which step (this is the make-or-break correctness framing).

3. **Showcase `--probedist-shot <out>` (Vulkan) / `--probedist` (Metal).** Reuse the DI 2×2×2=8-probe Cornell box.
   Pipeline: for each probe, render the distance cube (probe_dist.frag) → `ReadCubemapFace` → fill the
   `ProbeDistMoments[]` store. PROOFS (fail loudly):
   - **(1) capture-correctness (DI pattern):** a captured distance face (e.g. probe 0, face 0) is BYTE-IDENTICAL
     (SHA) to the scene rendered DIRECTLY to a 2D RT with that face's `FaceView`/`FaceProj` through the SAME
     `probe_dist.frag`. Print `probe-dist face-0 == direct distance render: BYTE-IDENTICAL`.
   - **(2) moment GPU==CPU bit-exact:** read back the distance face, CPU-compute `MomentsFromDistance(d)` per
     texel, `memcmp` vs the GPU-written moment store → BIT-EXACT. Print `probe-dist moments GPU==CPU: BIT-EXACT`.
   - **(3) probeCount=0 no-op:** `grid.dimX=0` → `DistTexelCount==0` → capture loop skipped → moment store
     byte-identical to its cleared upload.
   - **(4) determinism:** two runs byte-identical.
   - Print `probe-dist: {probes:8, faces:48, distFace:8}`. **Golden** `tests/golden/metal/probe_dist.png`: a
     probe-anchored world viz — the Cornell box + per-probe swatch spheres coloured by the probe's MEAN captured
     distance (near=warm, far=cool, a normalized grayscale/heat ramp) over the room; cross-backend-coherent
     (world-anchored like DI). Metal two runs DIFF 0.0000, **gate on the compare.sh EXIT CODE** (the printed
     "DIFF 0.0000" lies — it exits non-zero on any non-exact diff). Existing 78 image goldens UNTOUCHED.

4. **Determinism / cross-backend.** Fixed scene/grid; the distance render is geometry-only (no lighting/RNG/clock)
   → two runs byte-identical. The moment store is world/probe data → cross-backend-identical given the identical
   mesh (the engine's cross-backend golden contract). The swatch viz is world-anchored → cross-backend-coherent
   golden (verify + report honestly; if the sqrt-derived mean distance shows a sub-LSB cross-backend swatch-colour
   difference, that is a DISPLAY-scaling artifact of the viz, NOT a data-correctness issue — the bit-exact proof
   is on the moment-from-distance step; document any such observation, do NOT mask it).

5. **Tests `tests/probe_dist_test.cpp` (pure CPU, no GPU):**
   - `MomentsFromDistance(d)` == `{d, d*d}` exactly for a fixed set; `MomentsFromDistance(0)=={0,0}`.
   - `DistTexelCount` == `probeCount*384`; 0 at any zero dim / probeCount 0.
   - `ProbeDistFaceIndex` round-trip + full coverage (no overlap, no gap) over the 8-probe grid.
   - The layout index `p*384+f*64+v*8+u` is a bijection over `[0, probeCount*384)`.
   - Determinism. Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `ddgi-probe-distance` (features) + `--probedist-shot` (showcases). Rebake the
   introspect golden, update introspect_test. No other introspect drift.

## RHI seam additions (summary)
- **None.** Reuses DD's `ICubemapTarget`/`Begin-EndCubemapFace`/`ReadCubemapFace` (the DI surface) + the existing
  2D RT for the direct-render proof. New non-backend code (`probe_dist.h`, `shaders/probe_dist.frag.hlsl`,
  `tests/probe_dist_test.cpp`, the showcase block) adds ZERO above-seam backend symbols. The only `vk`/`MTL`
  mentions in the new header are seam-discipline doc comments, NOT code symbols. rhi.h + rhi_factory (dispatch
  baseline 2) + the backend dirs UNCHANGED. Report the seam.

## Out of scope (YAGNI — DP and beyond)
The Chebyshev/variance visibility WEIGHTING in the GI composite (that is slice DP — sample this moment cube in the
probe→shading-point direction, `mean=m[0]`, `variance=m[1]-m[0]²`, Chebyshev `p = variance/(variance+(d-mean)²)`,
weight probes by visibility → kill leak; disabled-path `occlusionStrength=0` → byte-identical to DN; golden = a
leak-test scene with the leak GONE). Also out: octahedral moment maps (we use the DD cube for consistency +
existing direction-query math), depth pre-filtering/blur (single-sample moments first), multi-bounce. ONE per-probe
distance-moment capture with a capture-correctness + moment-GPU==CPU + probeCount=0 + determinism proof set and a
probe-anchored distance-viz golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 78) + new `probe_dist_test`. Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--probedist-shot` on Vulkan: the distance-viz is coherent (near probes warm / far cool);
   `probe-dist face-0 == direct distance render: BYTE-IDENTICAL` + `probe-dist moments GPU==CPU: BIT-EXACT` +
   `probeCount=0` store-untouched + two-run byte-identical; the `probe-dist: {...}` line deterministic. Run under
   the AT Vulkan-validation gate → ZERO errors (the per-probe capture→read barriers SYNC-HAZARD-free, as DI).
3. Metal: `visual_test --probedist` → new golden `tests/golden/metal/probe_dist.png`; two runs DIFF 0.0000 (gate
   on the compare.sh EXIT CODE). The capture==direct + moment GPU==CPU + no-op proofs also pass on Metal.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `probe_dist.png` added; the
   other 78 byte-identical. `git diff master --stat -- tests/golden` = ONLY `probe_dist.png` (metal) + the 2-line
   introspect json — NO loose `tests/golden/probe_dist.png` (the DK stray trap), NO other golden changed.
5. Introspect JSON rebaked exactly `+ddgi-probe-distance` + `--probedist-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI; existing `lit.frag.hlsl` + all pipelines UNCHANGED).
   `scripts/verify.ps1` updated to include the new `probe_dist` image golden in the Mac round-trip loop (gate on
   compare.sh EXIT CODE) + bump its doc count 78→79.
