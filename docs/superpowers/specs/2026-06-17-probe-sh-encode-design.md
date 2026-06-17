# Slice DJ — DDGI Slice 3: Probe SH-Encode (Phase 5 #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The third slice of the
> DDGI GI flagship arc (after DH ray-trace, DI capture): encode each captured probe cubemap into 3rd-order
> spherical harmonics (9 coeffs per RGB channel) — the compact irradiance the GI composite samples. Reuses
> DI's captured cube + the DH cross-backend FP discipline. Deterministic, with a GPU==CPU bit-exact SH proof
> + analytic unit tests + a probeCount=0 no-op.

**Goal:** For each captured probe cubemap (DI), sample the cube in a fixed deterministic order and accumulate
the radiance into 3rd-order real spherical-harmonic coefficients (9 per channel = 27 floats/probe), producing
a per-probe SH irradiance SSBO. A `--probesh-shot` showcase visualizes the SH-reconstructed irradiance and
proves the GPU SH SSBO is BIT-EXACT to a CPU SH-encode over the same cube data (the DH GPU==CPU pattern),
proves the zero-radiance→zero-SH + probeCount=0 no-ops, and writes a deterministic golden. No probe-relighting
or visible GI composite yet (slices DK/DL) — this slice is the verified SH-irradiance layer.

## Reuse map
- **DI (`engine/render/probe_capture.h` + the captured-radiance store SSBO)** — the per-probe cube radiance
  this slice encodes. (DI captured 8 probes' cubes; DJ encodes them.)
- **DH (`engine/render/probe_gi.h` ProbeGrid + the cross-backend FP discipline)** — host-precompute any
  transcendental (the sample directions + their SH basis weights) + upload exact float32 bits + explicit
  `fma`/`mad` in the accumulation, so the GPU encode == the CPU reference bit-exact.

## Design decisions (locked)

1. **SH math (extend engine/render/probe_gi.h — or a new engine/render/probe_sh.h, header-only pure CPU, no
   backend symbols).** Namespace `hf::render::probesh` (or extend `probegi`). Mirrors the DH header style.
   - `struct ProbeSH { float coeff[9][3]; }` (or `math::Vec3 coeff[9]`) — 3rd-order (bands 0,1,2 = 9 basis
     functions) real SH, per RGB channel. `sizeof == 108` (27 floats), std430-clean. `static_assert`.
   - `void SHBasis9(const math::Vec3& dir, float out[9])` — the 9 REAL SH basis functions evaluated at a
     normalized direction (the standard Cartesian-polynomial form: Y00 const; Y1m ∝ y,z,x; Y2m ∝ xy,yz,
     3z²-1,xz,x²-y² with the standard constants). POLYNOMIAL in dir (no transcendentals) → bit-exact-friendly;
     document the exact constants.
   - `math::Vec3 SHEvaluate(const ProbeSH& sh, const math::Vec3& dir)` — reconstruct the irradiance in
     direction `dir` = Σ coeff[i] * SHBasis9(dir)[i] (optionally with the cosine-lobe convolution weights for
     irradiance — document; the standard `A0=π, A1=2π/3, A2=π/4` band scales). Used by the composite slice +
     the viz.
   - `void SHEncodeAccumulate(ProbeSH& sh, const math::Vec3& radiance, const math::Vec3& dir, float
     solidAngleWeight)` — accumulate one sample: `for i in 0..8: sh.coeff[i] += radiance * SHBasis9(dir)[i] *
     solidAngleWeight` (use `fma`). `void SHNormalize(ProbeSH& sh, float totalWeight)` — divide by the total
     solid angle (`4π` for a full-sphere cube). Document the exact normalization.
   - **Determinism / cross-backend:** the sample DIRECTIONS (cube-texel directions, which need a `normalize` =
     sqrt) and their `SHBasis9` weights + solid-angle weights are HOST-PRECOMPUTED once (CPU) and uploaded as
     exact float32 bits both the GPU encode + the CPU reference read — per the DH lesson (sqrt/normalize not
     bit-identical CPU↔GPU). The accumulation uses `fma`/`mad`. This makes the GPU SH == CPU SH bit-exact.
   - Pure, deterministic, no RNG/time. Unit-tested.

2. **Encode shader `shaders/probe_sh_encode.comp.hlsl`.** One thread per probe. Bindings (SSBOs, no
   textures, like froxel_inject): the captured-radiance store (DI, read), the host-precomputed
   sample-dir+weight table (read), the per-probe `ProbeSH` output (write). For each probe, loop over the fixed
   sample set: read the cube radiance at the sample's face/texel, `SHEncodeAccumulate` (copy `SHBasis9` weights
   from the uploaded table, OR recompute from the uploaded dir — but PREFER the uploaded precomputed basis
   weights for exactness), `SHNormalize`. Write `ProbeSH[probe]`. `ComputePipelineDesc{ storageBufferCount =
   3, threadsPerGroupX = 64 }`. NO new RHI. Only `[[vk::binding]]` + `HF_MSL_GEN` above-seam. Copy
   `SHEncodeAccumulate`/`SHNormalize` math verbatim from the header.

3. **Showcase `--probesh-shot <out>` (Vulkan) / `--probesh` (Metal).** Run DI's capture (the 8-probe Cornell
   box) → run the SH-encode compute → `ReadBuffer` the per-probe `ProbeSH` SSBO. Visualize the SH irradiance:
   per-probe swatch spheres colored by `SHEvaluate(sh, probeUpDir)` (or a few directions), OR a small SH-lit
   sphere per probe. Fixed scene/grid. Print `probe-sh: {probes:8, bands:3, coeffs:9}` (deterministic).
   INTERNAL PROOFS (fail loudly):
   - **GPU==CPU bit-exact (the DH pattern):** `ReadBuffer` the captured-radiance store + the GPU `ProbeSH`;
     on the CPU run the SAME `SHEncode` (uploaded dirs+weights, fma) over the read-back radiance; assert
     `memcmp(gpuSH, cpuSH) == 0` BIT-EXACT. Print `probe-sh GPU==CPU: BIT-EXACT`.
   - **Zero-radiance → zero-SH:** a probe whose captured cube is all-zero radiance → its `ProbeSH` is all
     zeros (within exact zero). Print `probe-sh zero-radiance == zero-SH: BYTE-IDENTICAL` (or assert).
   - **probeCount=0 no-op:** `grid.dimX=0` → encode dispatch 0 → the SH SSBO is untouched == cleared.
   - **Determinism:** two runs byte-identical.
   New golden `tests/golden/metal/probe_sh.png` (Metal two runs DIFF 0.0000, gate on compare.sh EXIT CODE).
   Existing 75 image goldens UNTOUCHED.

4. **Determinism.** Host-precomputed sample table + fma accumulation → GPU==CPU bit-exact + two-run
   byte-identical. Document cross-backend (the SH SSBO is world/probe data, not screen-space → should be
   cross-backend-identical; verify + report honestly).

5. **Tests `tests/probe_sh_test.cpp` (pure CPU, no GPU):**
   - **SHBasis9:** Y00 is the constant `0.282095`; the basis is orthonormal on hand-checked directions (+X/+Y/
     +Z axes give the documented basis values); polynomial (no NaN).
   - **Uniform cube → band-0 only:** encoding a uniform-radiance sphere/cube → `coeff[0]` (band 0) == the
     uniform radiance × the DC normalization; `coeff[1..8]` ≈ 0 (within tolerance). (The classic SH sanity
     check.)
   - **Directional lobe:** encoding a single bright sample in direction `d` → `SHEvaluate(sh, d)` is larger
     than `SHEvaluate(sh, -d)` (the SH captures the directional distribution).
   - **Zero radiance → zero SH:** all-zero samples → all-zero coeffs exactly.
   - **Determinism:** same samples → same SH.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `ddgi-probe-sh-encode` (features) + `--probesh-shot` (showcases).

## RHI seam additions (summary)
- **None.** SH encode is a compute pass over SSBOs (DI radiance store + sample table + SH output) — reuses
  `ComputePipelineDesc` + `BindStorageBuffer` + `DispatchCompute` + `ReadBuffer` (the DH/froxel surface). New
  non-backend files (the SH header additions, `shaders/probe_sh_encode.comp.hlsl`, `tests/probe_sh_test.cpp`)
  add ZERO above-seam backend code symbols. Seam grep stays at baseline (2). rhi.h UNCHANGED. Report the seam.

## Out of scope (YAGNI)
Probe relighting / neighbor gather (slice DK), the GI composite into the lit pass (slice DL), higher-order SH
(3rd-order only), per-frame re-encode, SH compression, directional-occlusion/visibility SH. One per-probe
3rd-order SH irradiance encode with a GPU==CPU bit-exact proof + uniform→band-0 + zero→zero unit tests + a
probeCount=0 no-op + an SH-irradiance-viz golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 75) + new `probe_sh_test` (SHBasis9 values/orthonormal,
   uniform→band-0, directional lobe, zero→zero, determinism). Clean under `windows-msvc-asan`.
2. **GPU==CPU + no-op proofs + visual:** `--probesh-shot` on Vulkan: the SH-irradiance viz is coherent (the
   probe swatches show the Cornell box's colored bounce); `probe-sh GPU==CPU: BIT-EXACT` + zero-radiance→zero-SH
   + `probeCount=0` SSBO-untouched + two-run byte-identical; the `probe-sh: {...}` line deterministic. Run
   under the AT Vulkan-validation gate → ZERO errors (compute + SSBO barriers SYNC-HAZARD-free).
3. Metal: `visual_test --probesh` → new golden `tests/golden/metal/probe_sh.png`; two runs DIFF 0.0000 (gate
   on the compare.sh EXIT CODE). The GPU==CPU + no-op proofs also pass on Metal.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `probe_sh.png` added; the
   other 75 byte-identical.
5. Introspect JSON rebaked exactly `+ddgi-probe-sh-encode` + `--probesh-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `probe_sh`
   image golden in the Mac round-trip loop (gate on compare.sh EXIT CODE).
