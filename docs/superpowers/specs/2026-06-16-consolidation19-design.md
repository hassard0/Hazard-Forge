# Slice DC — Consolidation #19 (full A–DB re-verify + docs refresh) — Design

> Consolidation pass (DUE: CZ, DA, DB = 3 slices since CY #18). Re-verify the ENTIRE engine cross-platform
> + refresh docs. Render-invariant: docs/CI text only. (Controller dispatches this with a 3600s fallback —
> the re-verify legitimately runs ~52 min with no local activity + a 0-byte transcript until completion; do
> NOT presume stuck before ~60 min — distinguish active work from a stall via build-artifact mtimes.)

**Goal:** Prove the whole A–DB engine still passes the full cross-platform gate after CZ (subsurface
scattering), DA (box-projected reflection probe), and DB (color grading), then refresh
README/ARCHITECTURE/CI. The commit touches ONLY docs/CI text. If re-verification surfaces a real defect, fix
it minimally + report it.

## Re-verification matrix (all must pass; report each result explicitly)

1. **Windows ctest** `windows-msvc-debug` → 70/70.
2. **ASan** `windows-msvc-asan` ctest → 70/70 (inside the VS dev shell — if a build LINKS and fails LNK1104,
   build via vcvars64), zero memory errors.
3. **All 70 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). **GATE ON THE
   compare.sh EXIT CODE, not the printed rounded "DIFF 0.0000"** (the print rounds away sub-LSB drift; a
   golden passes ONLY if compare.sh EXITS 0). Enumerate from `tests/golden/metal/*.png` (the prior 67 + `sss`
   + `refl_probe` + `color_grade`). If ANY golden EXITS non-zero, STOP-AND-REPORT (do NOT rebake) — quantify
   the drift (decode the PNGs) + identify the cause (a shared-shader edit?) for the controller to adjudicate.
4. **Material-graph JSON golden** byte-exact vs fresh `--material-introspect assets/materials/showcase3.mat.json`.
5. **Audio WAV golden** byte-exact vs a fresh `--audio-render` on Windows.
6. **Engine-introspect JSON golden** byte-exact vs live `--introspect` (must now include `subsurface-scattering`,
   `reflection-probe`, `color-grading` features + `--sss-shot`, `--reflprobe-shot`, `--colorgrade-shot`
   showcases).
7. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation` (sync +
   core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign `[WARNING: Performance]`
   notices allowed; document them). The validation layer is at the conan2 path
   `C:\Users\ihass\.conan2\p\b\vulkab00467b3e25b1\b\build\Debug\layers`; confirm the layer actually ENGAGED.
8. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch lines);
   other matches are comments or `[[vk::binding]]` HLSL decorations. Report the list. Confirm CZ/DA/DB new
   files (sss.h, reflection_probe.h, color_grade.h, the new shaders + tests) contribute ZERO above-seam
   backend symbols. The additive pure-interface RHI fields remain: CO's `oitRevealageBlend`, CS's
   `ComputeToComputeBarrier()`/`ComputeToFragmentBarrier()`, CX's `BindShadowMapCompute(IRenderTarget&)` +
   `ComputePipelineDesc::sampledShadowMap` — confirm each is still a pure interface and the rhi_factory
   dispatch count is still 2.
9. **Runtime==build-time + determinism spot-checks** (two-run byte-identical where noted): the NEW ones —
   `--sss-shot` → `{width:26,strength:14,taps:17}` + sssStrength=0==non-SSS BYTE-IDENTICAL (1b4bb87e79d914c6);
   `--reflprobe-shot` → `{boxSize:12,parallax:1}` + parallaxStrength=0==standard BYTE-IDENTICAL (7bd456b9ca8189f7);
   `--colorgrade-shot` → `{sat:1.18,gamma:1.05}` + identity==ungraded BYTE-IDENTICAL (2e974369d510a410) —
   plus the pre-existing set: `--gtao-shot` radius=0 (e0a46f7a287906f8), `--froxelfog-shot` density=0
   (48a4d3cb2104d3ba) SHA 123A0362C00A, `--contactshadow-shot` maxDist=0 (a41c0ede5a8a3261), `--froxellights-shot`
   lights-off (94398fa72c19c956)+density=0(ef6ffb56167ea125) SHA E3F61CF22AD7, `--autoexposure-shot`
   adaptation-off (b7290c092f82cfc0), `--volshadows-shot` shadows-off==CV (128d407347627b6d) SHA 73695FC2,
   `--oit-shot` permuted==canonical (9007024d7792ac3e), `--pom-shot` heightScale=0 (3d50b38bd46040f0),
   `--motionblur-shot` zero-velocity==unblurred, `--clustered-lights-shot` clustered==brute-force + 5606==5606,
   `--hiz-cull-shot` {31,31,24,7,24}, `--mdi-shot` drawCalls:1/refDrawCalls:144, `--bindless-shot`
   textureBinds:1, `--gpudriven-shot` drawCalls:1/textureBinds:1, `--gpucull-draw-shot` 144/107/107,
   `--mt-shot` workers1==4, `--cloud-shadows-shot` {steps:24,time:2}, `--water-shot` {waves:3,time:1.3,gridN:128},
   `--dof-shot` focalDist:12.0, `--clouds-shot` {steps:64,coverage:0.42,time:2}, `--game-shot`
   score:3/won:true/steps:380, `--audio-render` two-run identical, `--terrain-shot` peak:2.0972,
   `--terrain-stream-shot` frame:45/resident:22, `--anim-fsm-shot` blend:0.53, `--net-shot` replicaMatch:true,
   `--netsim-shot` converged:true, `--netpredict-shot` maxMisprediction:0.0655/converged:true,
   `--editor-edit-shot` saved:true/reloadMatch:true, `--vfx-shot` alive:960/spawned:1800, `--poststack-shot`
   count:5, `--ssgi-shot`/`--ssgi-denoise-shot`/`--ssgi-temporal-shot` two-run identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–DB: add subsurface scattering (CZ — skin/wax screen-space SSS),
  box-projected reflection probe (DA — local cubemap reflections), and color grading (DB — lift/gamma/gain +
  ASC-CDL + saturation). Update the showcase/flag table (+--sss-shot, --reflprobe-shot, --colorgrade-shot) and
  the golden tally (70 image + 1 material-JSON + 1 audio WAV + 1 engine-JSON).
- **docs/ARCHITECTURE.md** — add `engine/render/sss.h` (separable diffusion SSS), `engine/render/reflection_probe.h`
  (box-projected cubemap parallax), and `engine/render/color_grade.h` (analytic grade). Note the
  deterministic + equivalence-proof discipline (each carries an internal byte-identical disabled-path proof).
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (70) + the new goldens. Keep CI
  YAML under `ci/` (the token lacks workflow scope — do NOT create `.github/workflows/`).

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only. (Exception:
if matrix item 3 surfaces a genuine stale-Metal-golden drift from a shared-shader edit, STOP-AND-REPORT for
controller adjudication — do NOT rebake unilaterally in this docs-only pass.)

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly (and the Metal goldens were
   gated on the compare.sh EXIT CODE).
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
