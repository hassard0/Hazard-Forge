# Slice DL-prep — DDGI Slice 4: Probe-Grid Update / Trilinear SH Interpolation (Phase 5 #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The fourth DDGI slice
> (after DH ray-trace, DI capture, DJ SH-encode): trilinearly interpolate the 8 surrounding probes' SH at an
> arbitrary world position — the primitive the GI composite (next slice) samples per pixel. Standalone so the
> new interpolation math layer gets its OWN GPU==CPU bit-exact proof (the arc's one-proof-per-layer
> discipline). Pure compute, NO new RHI. (Branch: slice-dl-probeinterp.)

**Goal:** Add `NearestProbes` (the 8 trilinear cell corners + weights) + `InterpolateSH` (the fma-blended
SH at a world position) to the probe headers, and verify them with a GPU==CPU bit-exact data proof over a
compute pass + a probeCount=0 no-op + an interpolated-color-field golden. This isolates the trilinear math
before the GI composite uses it.

## Reuse map
- **DH `engine/render/probe_gi.h`** (ProbeGrid + `GetProbeGridIndex` floor/clamp idiom + `ProbeDispatchGroups`)
  and the cross-backend FP discipline (trilinear weights are polynomial — floor+subtract+multiply — but the
  per-corner blend products must use explicit `fma`/`mad` to avoid Metal's a+b*c contraction).
- **DJ `engine/render/probe_sh.h`** (`ProbeSH{float coeff[9][3]}` + `SHEvaluate`) — the SH this blends.
- **DI/DJ showcase + proof template** (`--probesh-shot` GPU==CPU `memcmp` + zero + probeCount=0 + golden).

## Design decisions (locked)

1. **Cell-corner math (extend engine/render/probe_gi.h, pure CPU, no backend symbols, NO SH dependency).**
   - `struct ProbeTrilinear { int idx[8]; float w[8]; bool valid; };` — the 8 cx-major corner flat indices +
     the 8 trilinear weights (sum == 1); `valid == false` on the degenerate/disabled path (idx all 0, w[0]=1).
   - `ProbeTrilinear NearestProbes(const math::Vec3& worldPos, const ProbeGrid& grid)` — mirrors
     `GetProbeGridIndex` but FLOOR (not round) + the 8-corner fractional weights: degenerate guard
     (`spacing<=0 || any dim<=0` → `{idx all 0, w[0]=1, valid=false}`); per axis `g=(v-origin)/spacing`,
     `base=floor(g)` clamped to `[0, dim-2]` (so `base` and `base+1` are both valid; `dim==1` → `base=0`, frac
     forced 0 on that axis), `frac=clamp(g-base,0,1)`; corner `c∈0..7` = `(cx+(c&1), cy+((c>>1)&1),
     cz+((c>>2)&1))` → `grid.flatIndex`; `wx=(c&1)?fx:(1-fx)` etc., `w[c]=wx*wy*wz` computed with `std::fma`.
     Out-of-grid positions clamp to the boundary cell (frac saturates). Document.
   - `int InterpDispatchGroups(const ProbeGrid& grid)` — mirror `ProbeDispatchGroups` / `EncodeDispatchGroups`:
     `nQueries<=0 || probeCount<=0 ? 0 : ceil(nQueries/64)` (the showcase passes the query count; probeCount=0
     → 0 → dispatch 0). Document the disabled-path 0.

2. **SH blend (extend engine/render/probe_sh.h, pure CPU, no backend symbols).**
   - `ProbeSH InterpolateSH(const math::Vec3& worldPos, const probegi::ProbeGrid& grid, const ProbeSH* probes,
     int probeCount)` — `tri = NearestProbes(...)`; if `!tri.valid || probeCount<=0` → a zeroed `ProbeSH`
     (the documented disabled fallback); else for each of the 9×3 coeffs `out.coeff[i][c] = Σ_corner
     std::fma(tri.w[corner], probes[tri.idx[corner]].coeff[i][c], acc)`. (SH projection is linear → blend-then-
     evaluate == evaluate-then-blend; we blend the SH.) 
   - `math::Vec3 InterpolateIrradiance(const math::Vec3& worldPos, const probegi::ProbeGrid& grid, const
     ProbeSH* probes, int probeCount, const math::Vec3& normal)` = `SHEvaluate(InterpolateSH(...),
     normalize(normal))`. Disabled → `{0,0,0}` (the DL composite's identity). Document.

3. **Interp shader `shaders/probe_interp.comp.hlsl`.** One thread per query point. Bindings (SSBOs, no
   textures): the `ProbeSH[]` (read), the query-points array (read), the blended output (write — a `ProbeSH`
   per query, OR the evaluated `float3` irradiance per query; pick + document; PREFER the blended `ProbeSH` so
   the proof covers the full 27-float blend). Copy `NearestProbes` + `InterpolateSH` math VERBATIM from the
   headers (with `mad`). `ComputePipelineDesc{ storageBufferCount = 3, threadsPerGroupX = 64 }`. NO new RHI.
   Only `[[vk::binding]]` + `HF_MSL_GEN` above-seam.

4. **Showcase `--probeinterp-shot <out>` (Vulkan) / `--probeinterp` (Metal).** Mirror `--probesh-shot`. Reuse
   the DJ 8-probe Cornell capture → SH-encode → a real `ProbeSH[8]`. Upload `ProbeSH[]` + a fixed deterministic
   set of query points (a grid spanning + exceeding the lattice — interior cells, boundary clamp, a degenerate
   axis). Run the interp compute → `ReadBuffer` the blended output. PROOFS (fail loudly):
   - **(1) GPU==CPU bit-exact:** CPU-run the SAME `InterpolateSH` over the SAME uploaded `ProbeSH[]` + query
     points; `memcmp(gpuBlend, cpuBlend) == 0`. Print `probe-interp GPU==CPU: BIT-EXACT`.
   - **(2) Lattice-point identity:** a query exactly at `probePos(px,py,pz)` → the blended SH == that probe's
     SH byte-identical (the round-trip). Print `probe-interp lattice-point == probe SH: BYTE-IDENTICAL`.
   - **(3) probeCount=0 no-op:** `grid.dimX=0` → `InterpDispatchGroups==0` → output SSBO byte-identical to the
     cleared/zero upload.
   - **(4) Determinism:** two runs byte-identical.
   - Print `probe-interp: {probes:8, queries:Q}`. **Golden** `probeinterp.png`: render the Cornell room + a
     DENSE grid of small swatch spheres each colored by `SHEvaluate(InterpolateSH(swatchPos,...), up)` — a
     SMOOTH interpolated color field between the probes (vs DJ's 8 discrete swatches), visibly proving
     continuity. Metal two runs DIFF 0.0000, gate on compare.sh EXIT CODE. Existing 76 image goldens UNTOUCHED.

5. **Determinism / cross-backend.** Polynomial weights + fma blend → GPU==CPU bit-exact + two-run identical.
   The blended ProbeSH is world/probe data (not screen-space) → cross-backend-identical; the swatch viz is
   world-anchored (like DI) → the golden should be cross-backend-coherent (verify + report honestly).

6. **Tests `tests/probe_interp_test.cpp` (pure CPU, no GPU):**
   - **Partition of unity:** `Σ NearestProbes(p).w == 1` for random interior points.
   - **Lattice-point identity:** `NearestProbes(probePos(px,py,pz))` → that corner has weight 1, the rest 0;
     `InterpolateSH(probePos(...))` == that probe's SH.
   - **Boundary clamp:** a far-outside point → valid corner indices (in range), weights still sum 1.
   - **dim==1 axis degeneracy:** a 1-thick axis → frac 0 on it, no out-of-range index.
   - **Disabled guard:** `spacing<=0` / `dim<=0` → `valid=false` → `InterpolateSH` returns zero SH;
     `InterpDispatchGroups` 0 at any zero dim / probeCount 0.
   - **Blend linearity:** blending 8 identical SH == that SH; a 2-probe-bright case shifts the blend toward it.
   - **Determinism.** Clean under `windows-msvc-asan`.

7. **Introspect.** Add exactly `ddgi-probe-interp` (features) + `--probeinterp-shot` (showcases).

## RHI seam additions (summary)
- **None.** Pure compute over SSBOs — reuses `ComputePipelineDesc` + `BindStorageBuffer` + `DispatchCompute` +
  `ReadBuffer` (the DH/DJ surface). New non-backend code (probe_gi.h/probe_sh.h additions,
  `shaders/probe_interp.comp.hlsl`, `tests/probe_interp_test.cpp`) adds ZERO above-seam backend symbols. Seam
  grep stays at baseline (2). rhi.h UNCHANGED. Report the seam.

## Out of scope (YAGNI)
The GI composite into the lit pass (the next slice DL — slice-dm-ddgi), neighbor visibility/occlusion weights,
per-frame re-interp. One trilinear SH interpolation primitive with a GPU==CPU bit-exact proof + lattice-point
identity + probeCount=0 no-op + an interpolated-color-field golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 76) + new `probe_interp_test` (partition of unity,
   lattice-point identity, boundary clamp, dim==1, disabled guard, blend linearity, determinism). Clean under
   `windows-msvc-asan`.
2. **GPU==CPU + identity + no-op proofs + visual:** `--probeinterp-shot` on Vulkan: the smooth interpolated
   color field is coherent; `probe-interp GPU==CPU: BIT-EXACT` + lattice-point identity + `probeCount=0`
   SSBO-untouched + two-run byte-identical; the `probe-interp: {...}` line deterministic. Run under the AT
   Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --probeinterp` → new golden `tests/golden/metal/probeinterp.png`; two runs DIFF 0.0000
   (gate on the compare.sh EXIT CODE). The GPU==CPU + no-op proofs also pass on Metal.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `probeinterp.png` added;
   the other 76 byte-identical.
5. Introspect JSON rebaked exactly `+ddgi-probe-interp` + `--probeinterp-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new
   `probeinterp` image golden in the Mac round-trip loop (gate on compare.sh EXIT CODE).
