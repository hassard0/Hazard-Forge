# Slice CQ — Consolidation #16 (full A–CP re-verify + docs refresh) — Design

> Consolidation pass (DUE: CN, CO, CP = 3 slices since CM #15). Re-verify the ENTIRE engine cross-platform
> + refresh docs. Render-invariant: docs/CI text only. (Controller dispatches this with a 3600s fallback —
> the re-verify legitimately runs ~52 min with no local activity + a 0-byte transcript until completion; do
> NOT presume stuck before ~60 min.)

**Goal:** Prove the whole A–CP engine still passes the full cross-platform gate after CN (motion blur), CO
(order-independent transparency / WBOIT), and CP (parallax occlusion mapping), then refresh
README/ARCHITECTURE/CI. The commit touches ONLY docs/CI text. If re-verification surfaces a real defect, fix
it minimally + report it.

## Re-verification matrix (all must pass; report each result explicitly)

1. **Windows ctest** `windows-msvc-debug` → 61/61.
2. **ASan** `windows-msvc-asan` ctest → 61/61 (inside the VS dev shell), zero memory errors.
3. **All 61 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). Enumerate from
   `tests/golden/metal/*.png` (the prior 58 + `motion_blur` + `oit` + `pom`).
4. **Material-graph JSON golden** byte-exact vs fresh `--material-introspect assets/materials/showcase3.mat.json`.
5. **Audio WAV golden** byte-exact vs a fresh `--audio-render` on Windows.
6. **Engine-introspect JSON golden** byte-exact vs live `--introspect` (must now include `motion-blur`,
   `order-independent-transparency`, `parallax-occlusion-mapping` features + `--motionblur-shot`, `--oit-shot`,
   `--pom-shot` showcases).
7. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation` (sync +
   core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign `[WARNING: Performance]`
   notices allowed; document them). The validation layer is at the conan2 path
   `C:\Users\ihass\.conan2\p\b\vulkab00467b3e25b1\b\build\Debug\layers` (set VK_LAYER_PATH +
   VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation); confirm the layer actually ENGAGED (perf warnings appear).
   Pay attention to the new `--oit-shot` (MRT/blend-state) showcase.
8. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch lines);
   other matches are comments. Report the list. Confirm the CN/CO/CP new files (motion_blur.h, oit.h, pom.h,
   the new shaders + tests) contribute ZERO above-seam backend symbols. NOTE: CO added ONE additive
   pure-interface field `GraphicsPipelineDesc::oitRevealageBlend` in `engine/rhi/rhi.h` (a bool flag, default
   false) with the blend factors set INSIDE the backend dirs — confirm it is a pure interface (no backend
   code symbol above the seam) and that the rhi_factory dispatch count is still 2.
9. **Runtime==build-time + determinism spot-checks** (two-run byte-identical where noted):
   `--material-live-shot`==`--material-shot` SHA; `--mt-shot --workers 1`==`--workers 4` SHA; `--mdi-shot` →
   `BYTE-IDENTICAL, drawCalls:1, refDrawCalls:144`; `--bindless-shot` → `BYTE-IDENTICAL, textureBinds:1`;
   `--gpudriven-shot` → `BYTE-IDENTICAL, drawCalls:1, textureBinds:1`; `--gpucull-draw-shot` →
   `BYTE-IDENTICAL, total:144, drawn:107, cpuRef:107`; `--hiz-cull-shot` →
   `hiz-cull:{total:31,frustumKept:31,occluded:24,drawn:7,cpuOccluded:24}` + BYTE-IDENTICAL to frustum-only;
   `--cloud-shadows-shot` → `cloud-shadows:{steps:24,time:2}` two-run identical; `--clustered-lights-shot` →
   `clustered-lights:{lights:96,clusters:3456,maxPerCluster:34,avgPerCluster:1.62,brimByteIdentical:true}` +
   clustered==brute-force BYTE-IDENTICAL + GPU assigned 5606==CPU ref 5606, two-run identical;
   `--motionblur-shot` → `motion-blur:{maxBlurPx:28,taps:24,velScale:1.0}` + zero-velocity render
   BYTE-IDENTICAL to the un-blurred scene, two-run identical; `--oit-shot` →
   `oit:{layers:5,orderIndependent:true}` + permuted-draw-order render BYTE-IDENTICAL to canonical
   (canonical-hash 9007024d7792ac3e), two-run identical; `--pom-shot` → `pom:{heightScale:0.08,steps:32}` +
   heightScale=0 render BYTE-IDENTICAL (real-field==flat-field, hash 3d50b38bd46040f0), two-run identical;
   `--game-shot` → `score:3,won:true,steps:380`; `--audio-render` twice byte-identical; `--terrain-shot` →
   `peak:2.0972`; `--terrain-stream-shot` → `frame:45, resident:22`; `--anim-fsm-shot` → `blend:0.53`;
   `--net-shot` → `replicaMatch:true`; `--netsim-shot` → `converged:true`; `--netpredict-shot` →
   `maxMisprediction:0.0655, converged:true`; `--editor-edit-shot` → `saved:true, reloadMatch:true`;
   `--vfx-shot` → `alive:960, spawned:1800`; `--water-shot` → `waves:3, time:1.3, gridN:128` two-run identical;
   `--dof-shot` → `focalDist:12.0` two-run identical; `--clouds-shot` → `steps:64, coverage:0.42, time:2`
   two-run identical; `--poststack-shot` → `count:5`;
   `--ssgi-shot`/`--ssgi-denoise-shot`/`--ssgi-temporal-shot` two-run identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–CP: add motion blur (CN — velocity-gather post pass), order-
  independent transparency (CO — Weighted Blended OIT, correct transparency without sorting), and parallax
  occlusion mapping (CP — per-pixel surface depth + self-occlusion). Update the showcase/flag table
  (+--motionblur-shot, --oit-shot, --pom-shot) and the golden tally (61 image + 1 material-JSON + 1 audio WAV
  + 1 engine-JSON).
- **docs/ARCHITECTURE.md** — add `engine/render/motion_blur.h` (velocity-gather, zero-velocity pass-through
  proof), `engine/render/oit.h` (Weighted Blended OIT accum/revealage/resolve, draw-order-independent SUM +
  PRODUCT, + the additive `oitRevealageBlend` RHI interface flag), and `engine/render/pom.h` (steep-parallax
  + binary-refine + self-shadow, zero-height pass-through proof). Note all are deterministic, pure-CPU-shared-
  math, and that CO + CP carry byte-identical equivalence proofs (order-independence / zero-height).
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (61) + the new goldens. Keep CI
  YAML under `ci/` (the token lacks workflow scope — do NOT create `.github/workflows/`).

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
