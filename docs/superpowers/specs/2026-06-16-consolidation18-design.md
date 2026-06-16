# Slice CY — Consolidation #18 (full A–CX re-verify + docs refresh) — Design

> Consolidation pass (DUE: CV, CW, CX = 3 slices since CU #17). Re-verify the ENTIRE engine cross-platform
> + refresh docs. Render-invariant: docs/CI text only. (Controller dispatches this with a 3600s fallback —
> the re-verify legitimately runs ~52 min with no local activity + a 0-byte transcript until completion; do
> NOT presume stuck before ~60 min — distinguish active work from a stall via build-artifact mtimes.)

**Goal:** Prove the whole A–CX engine still passes the full cross-platform gate after CV (per-froxel
clustered-light injection), CW (auto-exposure), and CX (volumetric shadows), then refresh
README/ARCHITECTURE/CI. The commit touches ONLY docs/CI text. If re-verification surfaces a real defect, fix
it minimally + report it.

## Re-verification matrix (all must pass; report each result explicitly)

1. **Windows ctest** `windows-msvc-debug` → 67/67.
2. **ASan** `windows-msvc-asan` ctest → 67/67 (inside the VS dev shell), zero memory errors.
3. **All 67 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). Enumerate from
   `tests/golden/metal/*.png` (the prior 64 + `froxel_lights` + `auto_exposure` + `vol_shadows`).
4. **Material-graph JSON golden** byte-exact vs fresh `--material-introspect assets/materials/showcase3.mat.json`.
5. **Audio WAV golden** byte-exact vs a fresh `--audio-render` on Windows.
6. **Engine-introspect JSON golden** byte-exact vs live `--introspect` (must now include `froxel-light-injection`,
   `auto-exposure`, `volumetric-shadows` features + `--froxellights-shot`, `--autoexposure-shot`,
   `--volshadows-shot` showcases).
7. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation` (sync +
   core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign `[WARNING: Performance]`
   notices allowed; document them). The validation layer is at the conan2 path
   `C:\Users\ihass\.conan2\p\b\vulkab00467b3e25b1\b\build\Debug\layers` (set VK_LAYER_PATH +
   VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation); confirm the layer actually ENGAGED (perf warnings appear).
   Pay attention to the new `--froxellights-shot`, `--autoexposure-shot` (histogram compute), and
   `--volshadows-shot` (the shadow-pass→inject-compute read barrier must be SYNC-HAZARD-free).
8. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch lines);
   other matches are comments or `[[vk::binding]]` HLSL decorations. Report the list. Confirm the CV/CW/CX new
   files (froxel.h additions, auto_exposure.h, the new shaders + tests) contribute ZERO above-seam backend
   symbols. NOTE the additive pure-interface RHI fields: CO's `oitRevealageBlend`, CS's
   `ComputeToComputeBarrier()`/`ComputeToFragmentBarrier()`, and CX's `BindShadowMapCompute(IRenderTarget&)` +
   `ComputePipelineDesc::sampledShadowMap` — all with impls only in the backend dirs; confirm each is a pure
   interface and the rhi_factory dispatch count is still 2.
9. **Runtime==build-time + determinism spot-checks** (two-run byte-identical where noted):
   `--material-live-shot`==`--material-shot`; `--mt-shot --workers 1`==`--workers 4`; `--mdi-shot` →
   `drawCalls:1, refDrawCalls:144`; `--bindless-shot` → `textureBinds:1`; `--gpudriven-shot` →
   `drawCalls:1, textureBinds:1`; `--gpucull-draw-shot` → `total:144, drawn:107, cpuRef:107`; `--hiz-cull-shot`
   → `{total:31,frustumKept:31,occluded:24,drawn:7,cpuOccluded:24}` + BYTE-IDENTICAL to frustum-only;
   `--clustered-lights-shot` → clustered==brute-force BYTE-IDENTICAL + 5606==5606; `--motionblur-shot` →
   zero-velocity==unblurred BYTE-IDENTICAL; `--oit-shot` → permuted==canonical BYTE-IDENTICAL (9007024d7792ac3e);
   `--pom-shot` → heightScale=0 BYTE-IDENTICAL (3d50b38bd46040f0); `--gtao-shot` → radius=0==no-AO BYTE-IDENTICAL
   (e0a46f7a287906f8); `--froxelfog-shot` → density=0==no-fog BYTE-IDENTICAL (48a4d3cb2104d3ba), SHA
   123A0362C00A; `--contactshadow-shot` → maxDist=0==no-contact BYTE-IDENTICAL (a41c0ede5a8a3261);
   `--froxellights-shot` → `{lights:96,froxels:16x9x64,density:0.06,g:0.76}` + lights-off==CS-fog BYTE-IDENTICAL
   (94398fa72c19c956) + density=0==no-fog (ef6ffb56167ea125), SHA E3F61CF22AD7; `--autoexposure-shot` →
   `{bins:256,EV:-3.211,keyValue:0.18}` + adaptationEnabled=false==standard BYTE-IDENTICAL (b7290c092f82cfc0);
   `--volshadows-shot` → `{froxels:16x9x64,cascades:1,density:0.06}` + volumetricShadows=false==CV BYTE-IDENTICAL
   (128d407347627b6d) + density=0==no-fog (ef6ffb56167ea125), SHA 73695FC2; `--cloud-shadows-shot` →
   `{steps:24,time:2}`; `--water-shot` → `waves:3,time:1.3,gridN:128`; `--dof-shot` → `focalDist:12.0`;
   `--clouds-shot` → `steps:64,coverage:0.42,time:2`; `--game-shot` → `score:3,won:true,steps:380`;
   `--audio-render` twice byte-identical; `--terrain-shot` → `peak:2.0972`; `--terrain-stream-shot` →
   `frame:45,resident:22`; `--anim-fsm-shot` → `blend:0.53`; `--net-shot` → `replicaMatch:true`; `--netsim-shot`
   → `converged:true`; `--netpredict-shot` → `maxMisprediction:0.0655,converged:true`; `--editor-edit-shot` →
   `saved:true,reloadMatch:true`; `--vfx-shot` → `alive:960,spawned:1800`; `--poststack-shot` → `count:5`;
   `--ssgi-shot`/`--ssgi-denoise-shot`/`--ssgi-temporal-shot` two-run identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–CX: add per-froxel clustered-light injection (CV — the 96 lights
  cast colored volumetric shafts through the fog), auto-exposure (CW — histogram eye adaptation), and
  volumetric shadows (CX — sun light shafts through the fog, completing the volumetric trilogy). Update the
  showcase/flag table (+--froxellights-shot, --autoexposure-shot, --volshadows-shot) and the golden tally
  (67 image + 1 material-JSON + 1 audio WAV + 1 engine-JSON).
- **docs/ARCHITECTURE.md** — note the `engine/render/froxel.h` `InjectClusteredLights` + `SunVisibility`
  additions (the volumetric trilogy: sun scatter → clustered-light scatter → sun-shadow gate), `engine/render/
  auto_exposure.h` (integer-histogram key-value auto-exposure), and the new `BindShadowMapCompute` RHI
  interface (compute-stage shadow-map sampling). Note the determinism + equivalence-proof discipline.
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (67) + the new goldens. Keep CI
  YAML under `ci/` (the token lacks workflow scope — do NOT create `.github/workflows/`).

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
