# Slice DK — Consolidation #21 (full A–DJ re-verify + docs refresh) — Design

> Consolidation pass (DUE: DH, DI, DJ = 3 GI-arc slices since DG #20). Re-verify the ENTIRE engine
> cross-platform + refresh docs. Render-invariant: docs/CI text only. (Controller dispatches with a 3600s
> fallback — the re-verify legitimately runs ~52 min with no local activity until completion; do NOT presume
> stuck before ~60 min — distinguish active work from a stall via build-artifact mtimes.)

**Goal:** Prove the whole A–DJ engine still passes the full cross-platform gate after the first three DDGI
GI-arc slices — DH (probe ray-trace), DI (probe capture), DJ (probe SH-encode) — then refresh
README/ARCHITECTURE/CI. The commit touches ONLY docs/CI text. If re-verification surfaces a real defect, fix
it minimally + report it.

## Re-verification matrix (all must pass; report each result explicitly)

1. **Windows ctest** `windows-msvc-debug` → 76/76. (vcvars64 if a build LINKS/compiles + fails LNK1104/C1083.)
2. **ASan** `windows-msvc-asan` ctest → 76/76 (ASan runtime DLL on PATH / vcvars64 shell), zero memory errors.
3. **All 76 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). **GATE ON THE
   compare.sh EXIT CODE** (a golden passes ONLY if compare.sh EXITS 0). Run tar/scp from PowerShell. Enumerate
   from `tests/golden/metal/*.png` (the prior 73 + `probegi` + `probe_capture` + `probe_sh`). If ANY golden
   EXITS non-zero, STOP-AND-REPORT (do NOT rebake): quantify the drift + identify the cause. (DH/DI/DJ used NEW
   non-shared shaders → no shared-shader drift expected; DI reused DD's probe_bake shaders unchanged.)
4. **Material-graph JSON golden** byte-exact vs fresh `--material-introspect assets/materials/showcase3.mat.json`.
5. **Audio WAV golden** byte-exact vs a fresh `--audio-render` on Windows.
6. **Engine-introspect JSON golden** byte-exact vs live `--introspect` (must now include `ddgi-probe-raytrace`,
   `ddgi-probe-capture`, `ddgi-probe-sh-encode` features + `--probegi-shot`, `--probecapture-shot`,
   `--probesh-shot` showcases).
7. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation` (sync +
   core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign `[WARNING: Performance]`
   allowed; document). Layer at `C:\Users\ihass\.conan2\p\b\vulkab00467b3e25b1\b\build\Debug\layers`; confirm
   ENGAGED. Watch the 3 new compute showcases (`--probegi-shot`, `--probecapture-shot`, `--probesh-shot`) —
   the compute + readback barriers must be SYNC-HAZARD-free.
8. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch lines);
   other matches are comments or `[[vk::binding]]`. Report the list. Confirm the DH/DI/DJ new files (probe_gi.h,
   probe_capture.h, probe_sh.h, the new compute shaders + tests) contribute ZERO above-seam backend symbols.
   **DH/DI/DJ added NO RHI** (all reused existing compute/SSBO/cubemap-RT surface) — confirm rhi.h is unchanged
   from the DG #20 state + the 4 additive pure-interface RHI capabilities remain pure interfaces (CO
   oitRevealageBlend; CS ComputeToCompute/ComputeToFragmentBarrier; CX BindShadowMapCompute + sampledShadowMap;
   DD cubemap-RT set). rhi_factory dispatch still 2.
9. **Runtime==build-time + determinism spot-checks** (two-run byte-identical where noted): the NEW GI ones —
   `--probegi-shot` → `{probes:256,rays:16,hits:1474}` + GPU==CPU ray-hits BIT-EXACT + probeCount=0 no-op;
   `--probecapture-shot` → `{probes:8,faces:48,cubeSize:64}` + captured face-0==direct render BYTE-IDENTICAL
   (d17a0d18a10ca823) + probeCount=0 store-untouched; `--probesh-shot` → `{probes:8,bands:3,coeffs:9}` + SH
   GPU==CPU BIT-EXACT + zero-radiance==zero-SH + probeCount=0 SSBO-untouched — plus the full pre-existing set:
   `--captureprobe-shot` 88b1504807350383, `--planar-shot` 0f44066f3005fc01, `--cas-shot` 2e974369d510a410,
   `--reflprobe-shot` 7bd456b9ca8189f7, `--colorgrade-shot` 2e974369d510a410, `--sss-shot` 1b4bb87e79d914c6,
   `--gtao-shot` e0a46f7a287906f8, `--froxelfog-shot` 48a4d3cb2104d3ba, `--contactshadow-shot`
   a41c0ede5a8a3261, `--froxellights-shot` 94398fa72c19c956, `--autoexposure-shot` b7290c092f82cfc0,
   `--volshadows-shot` 128d407347627b6d, `--oit-shot` 9007024d7792ac3e, `--pom-shot` 3d50b38bd46040f0,
   `--motionblur-shot` zero-velocity, `--clustered-lights-shot` 5606==5606, `--hiz-cull-shot` {31,31,24,7,24},
   `--mdi-shot` drawCalls:1/refDrawCalls:144, `--bindless-shot` textureBinds:1, `--gpudriven-shot`
   drawCalls:1/textureBinds:1, `--gpucull-draw-shot` 144/107/107, `--mt-shot` 1==4, `--cloud-shadows-shot`
   {steps:24,time:2}, `--water-shot` {waves:3,time:1.3,gridN:128}, `--dof-shot` focalDist:12.0, `--clouds-shot`
   {steps:64,coverage:0.42,time:2}, `--game-shot` score:3/won:true/steps:380, `--audio-render` two-run
   identical, `--terrain-shot` peak:2.0972, `--terrain-stream-shot` frame:45/resident:22, `--anim-fsm-shot`
   blend:0.53, `--net-shot` replicaMatch:true, `--netsim-shot` converged:true, `--netpredict-shot`
   maxMisprediction:0.0655/converged:true, `--editor-edit-shot` saved:true/reloadMatch:true, `--vfx-shot`
   alive:960/spawned:1800, `--poststack-shot` count:5, `--ssgi-shot`/`--ssgi-denoise-shot`/`--ssgi-temporal-shot`
   two-run identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–DJ: note the engine has begun a **DDGI dynamic global-illumination
  pillar** (probe ray-trace + radiance capture + SH-encode landed; probe-update + GI-composite next). Update
  the showcase/flag table (+--probegi-shot, --probecapture-shot, --probesh-shot) and the golden tally (76
  image + 1 material-JSON + 1 audio WAV + 1 engine-JSON). Note the GPU==CPU bit-exact compute-proof discipline.
- **docs/ARCHITECTURE.md** — add `engine/render/probe_gi.h` (probe grid + Fibonacci-sphere ray-trace, the flat-
  SSBO depth approach), `engine/render/probe_capture.h` (per-probe cubemap radiance capture via the cubemap-RT),
  and `engine/render/probe_sh.h` (3rd-order SH encode). Note the cross-backend FP discipline (host-precompute
  transcendentals + fma → GPU==CPU bit-exact) and that the GI arc adds NO new RHI (reuses compute/SSBO/cubemap).
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (76) + the new goldens. Keep CI YAML
  under `ci/` (do NOT create `.github/workflows/`).

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only. (Exception: a
genuine stale-Metal-golden drift → STOP-AND-REPORT for controller adjudication; do NOT rebake unilaterally.)

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly (Metal goldens gated on the
   compare.sh EXIT CODE).
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
