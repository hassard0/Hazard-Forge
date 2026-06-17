# Slice DG — Consolidation #20 (full A–DF re-verify + docs refresh) — Design

> Consolidation pass (DUE: DD, DE, DF = 3 slices since DC #19). Re-verify the ENTIRE engine cross-platform
> + refresh docs. Render-invariant: docs/CI text only. (Controller dispatches this with a 3600s fallback —
> the re-verify legitimately runs ~52 min with no local activity + a 0-byte transcript until completion; do
> NOT presume stuck before ~60 min — distinguish active work from a stall via build-artifact mtimes.)

**Goal:** Prove the whole A–DF engine still passes the full cross-platform gate after DD (runtime cubemap-
capture reflection probe), DE (planar reflections), and DF (CAS sharpening), then refresh
README/ARCHITECTURE/CI. The commit touches ONLY docs/CI text. If re-verification surfaces a real defect, fix
it minimally + report it.

## Re-verification matrix (all must pass; report each result explicitly)

1. **Windows ctest** `windows-msvc-debug` → 73/73. (If a build LINKS/compiles and fails LNK1104/C1083, build
   via vcvars64: `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build --preset windows-msvc-debug'`.)
2. **ASan** `windows-msvc-asan` ctest → 73/73 (ensure the MSVC ASan runtime DLL dir is on PATH / run in the
   vcvars64 shell), zero memory errors.
3. **All 73 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). **GATE ON THE
   compare.sh EXIT CODE, not the printed rounded "DIFF 0.0000"** (a golden passes ONLY if compare.sh EXITS 0).
   Run the round-trip tar/scp from PowerShell (git-bash tar mis-parses the C: temp path). Enumerate from
   `tests/golden/metal/*.png` (the prior 70 + `capture_probe` + `planar_reflection` + `cas`). If ANY golden
   EXITS non-zero, STOP-AND-REPORT (do NOT rebake): quantify the drift + identify the cause (a shared-shader
   edit?). NOTE: DD/DE/DF all used NEW non-shared shaders → no shared-shader drift expected.
4. **Material-graph JSON golden** byte-exact vs fresh `--material-introspect assets/materials/showcase3.mat.json`.
5. **Audio WAV golden** byte-exact vs a fresh `--audio-render` on Windows.
6. **Engine-introspect JSON golden** byte-exact vs live `--introspect` (must now include `capture-reflection-probe`,
   `planar-reflections`, `contrast-adaptive-sharpening` features + `--captureprobe-shot`, `--planar-shot`,
   `--cas-shot` showcases).
7. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation` (sync +
   core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign `[WARNING: Performance]`
   allowed; document). Layer at the conan2 path `C:\Users\ihass\.conan2\p\b\vulkab00467b3e25b1\b\build\Debug\layers`;
   confirm it ENGAGED. Pay attention to `--captureprobe-shot` (6 cube-face passes + capture→sample barrier) and
   `--planar-shot` (reflection-write→sample barrier).
8. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch lines);
   other matches are comments or `[[vk::binding]]` decorations. Report the list. Confirm DD/DE/DF new files
   (cubemap.h, planar_reflection.h, cas.h, the new shaders + tests) contribute ZERO above-seam backend symbols.
   The FOUR additive pure-interface RHI capabilities remain (confirm each is a pure interface with backend-dir
   impls + rhi_factory dispatch still 2): CO's `oitRevealageBlend`; CS's `ComputeToComputeBarrier()`/
   `ComputeToFragmentBarrier()`; CX's `BindShadowMapCompute(IRenderTarget&)` + `ComputePipelineDesc::sampledShadowMap`;
   DD's cubemap-RT set (`ICubemapTarget`, `CreateCubemapTarget`, `BeginCubemapFace`/`EndCubemapFace`/
   `ReadCubemapFace`/`ReadRenderTarget`, `BindCubemapProbe`). DE added NO RHI (reused `cullNone`); DF added none.
9. **Runtime==build-time + determinism spot-checks** (two-run byte-identical where noted): the NEW ones —
   `--captureprobe-shot` → `{faces:6,cubeSize:512}` + captured face-0==direct-render BYTE-IDENTICAL
   (88b1504807350383); `--planar-shot` → `{reflectivity:0.75}` + reflectivity=0==matte BYTE-IDENTICAL
   (0f44066f3005fc01); `--cas-shot` → `{0.8}` + sharpness=0==unsharpened BYTE-IDENTICAL (2e974369d510a410) —
   plus the full pre-existing set: `--reflprobe-shot` parallax=0 (7bd456b9ca8189f7), `--colorgrade-shot`
   identity (2e974369d510a410), `--sss-shot` sssStrength=0 (1b4bb87e79d914c6), `--gtao-shot` radius=0
   (e0a46f7a287906f8), `--froxelfog-shot` density=0 (48a4d3cb2104d3ba), `--contactshadow-shot` maxDist=0
   (a41c0ede5a8a3261), `--froxellights-shot` lights-off (94398fa72c19c956), `--autoexposure-shot` adaptation-off
   (b7290c092f82cfc0), `--volshadows-shot` shadows-off==CV (128d407347627b6d), `--oit-shot` permuted==canonical
   (9007024d7792ac3e), `--pom-shot` heightScale=0 (3d50b38bd46040f0), `--motionblur-shot` zero-velocity,
   `--clustered-lights-shot` clustered==brute-force 5606==5606, `--hiz-cull-shot` {31,31,24,7,24}, `--mdi-shot`
   drawCalls:1/refDrawCalls:144, `--bindless-shot` textureBinds:1, `--gpudriven-shot` drawCalls:1/textureBinds:1,
   `--gpucull-draw-shot` 144/107/107, `--mt-shot` 1==4, `--cloud-shadows-shot` {steps:24,time:2}, `--water-shot`
   {waves:3,time:1.3,gridN:128}, `--dof-shot` focalDist:12.0, `--clouds-shot` {steps:64,coverage:0.42,time:2},
   `--game-shot` score:3/won:true/steps:380, `--audio-render` two-run identical, `--terrain-shot` peak:2.0972,
   `--terrain-stream-shot` frame:45/resident:22, `--anim-fsm-shot` blend:0.53, `--net-shot` replicaMatch:true,
   `--netsim-shot` converged:true, `--netpredict-shot` maxMisprediction:0.0655/converged:true, `--editor-edit-shot`
   saved:true/reloadMatch:true, `--vfx-shot` alive:960/spawned:1800, `--poststack-shot` count:5,
   `--ssgi-shot`/`--ssgi-denoise-shot`/`--ssgi-temporal-shot` two-run identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–DF: add dynamic cubemap-capture reflection probe (DD), planar
  reflections (DE), and CAS sharpening (DF) — note the reflections suite is now SSR + box-projected probe +
  dynamic cubemap-capture probe + planar. Update the showcase/flag table (+--captureprobe-shot, --planar-shot,
  --cas-shot) and the golden tally (73 image + 1 material-JSON + 1 audio WAV + 1 engine-JSON). Note the 4
  reusable RHI capabilities (incl. cubemap render targets).
- **docs/ARCHITECTURE.md** — add `engine/render/cubemap.h` (6-face capture math + the cubemap-RT RHI),
  `engine/render/planar_reflection.h` (householder reflection + oblique near-clip), and `engine/render/cas.h`
  (FidelityFX CAS). Note the determinism + equivalence-proof discipline + the cubemap-RT interface.
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (73) + the new goldens. Keep CI
  YAML under `ci/` (the token lacks workflow scope — do NOT create `.github/workflows/`).

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only. (Exception:
if matrix item 3 surfaces a genuine stale-Metal-golden drift, STOP-AND-REPORT for controller adjudication — do
NOT rebake unilaterally in this docs-only pass.)

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly (Metal goldens gated on the
   compare.sh EXIT CODE).
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
