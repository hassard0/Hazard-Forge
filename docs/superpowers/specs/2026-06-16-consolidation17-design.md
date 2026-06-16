# Slice CU — Consolidation #17 (full A–CT re-verify + docs refresh) — Design

> Consolidation pass (DUE: CR, CS, CT = 3 slices since CQ #16). Re-verify the ENTIRE engine cross-platform
> + refresh docs. Render-invariant: docs/CI text only. (Controller dispatches this with a 3600s fallback —
> the re-verify legitimately runs ~52 min with no local activity + a 0-byte transcript until completion; do
> NOT presume stuck before ~60 min — distinguish active work from a stall via build-artifact mtimes.)

**Goal:** Prove the whole A–CT engine still passes the full cross-platform gate after CR (GTAO), CS (froxel
volumetric fog), and CT (screen-space contact shadows), then refresh README/ARCHITECTURE/CI. The commit
touches ONLY docs/CI text. If re-verification surfaces a real defect, fix it minimally + report it.

## Re-verification matrix (all must pass; report each result explicitly)

1. **Windows ctest** `windows-msvc-debug` → 64/64.
2. **ASan** `windows-msvc-asan` ctest → 64/64 (inside the VS dev shell), zero memory errors.
3. **All 64 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). Enumerate from
   `tests/golden/metal/*.png` (the prior 61 + `gtao` + `froxel_fog` + `contact_shadows`).
4. **Material-graph JSON golden** byte-exact vs fresh `--material-introspect assets/materials/showcase3.mat.json`.
5. **Audio WAV golden** byte-exact vs a fresh `--audio-render` on Windows.
6. **Engine-introspect JSON golden** byte-exact vs live `--introspect` (must now include
   `ground-truth-ambient-occlusion`, `froxel-volumetric-fog`, `contact-shadows` features + `--gtao-shot`,
   `--froxelfog-shot`, `--contactshadow-shot` showcases).
7. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation` (sync +
   core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign `[WARNING: Performance]`
   notices allowed; document them). The validation layer is at the conan2 path
   `C:\Users\ihass\.conan2\p\b\vulkab00467b3e25b1\b\build\Debug\layers` (set VK_LAYER_PATH +
   VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation); confirm the layer actually ENGAGED (perf warnings appear).
   Pay attention to the new `--froxelfog-shot` (the COMPUTE→COMPUTE→FRAGMENT barriers must be SYNC-HAZARD-free)
   and `--contactshadow-shot`.
8. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch lines);
   other matches are comments or `[[vk::binding]]` HLSL decorations. Report the list. Confirm the CR/CS/CT new
   files (gtao.h, froxel.h, contact_shadows.h, the new shaders + tests) contribute ZERO above-seam backend
   symbols. NOTE the additive pure-interface RHI fields are: CO's `GraphicsPipelineDesc::oitRevealageBlend`
   (bool, default false) and CS's `ICommandBuffer::ComputeToComputeBarrier()`/`ComputeToFragmentBarrier()`
   (empty default bodies) — all with impls only in the backend dirs; confirm each is a pure interface and the
   rhi_factory dispatch count is still 2.
9. **Runtime==build-time + determinism spot-checks** (two-run byte-identical where noted):
   `--material-live-shot`==`--material-shot` SHA; `--mt-shot --workers 1`==`--workers 4` SHA; `--mdi-shot` →
   `BYTE-IDENTICAL, drawCalls:1, refDrawCalls:144`; `--bindless-shot` → `BYTE-IDENTICAL, textureBinds:1`;
   `--gpudriven-shot` → `BYTE-IDENTICAL, drawCalls:1, textureBinds:1`; `--gpucull-draw-shot` →
   `BYTE-IDENTICAL, total:144, drawn:107, cpuRef:107`; `--hiz-cull-shot` →
   `hiz-cull:{total:31,frustumKept:31,occluded:24,drawn:7,cpuOccluded:24}` + BYTE-IDENTICAL to frustum-only;
   `--cloud-shadows-shot` → `cloud-shadows:{steps:24,time:2}` two-run identical; `--clustered-lights-shot` →
   `clustered-lights:{lights:96,clusters:3456,maxPerCluster:34,avgPerCluster:1.62,brimByteIdentical:true}` +
   clustered==brute-force BYTE-IDENTICAL + GPU 5606==CPU 5606, two-run identical; `--motionblur-shot` →
   `motion-blur:{maxBlurPx:28,taps:24,velScale:1.0}` + zero-velocity==unblurred BYTE-IDENTICAL, two-run
   identical; `--oit-shot` → `oit:{layers:5,orderIndependent:true}` + permuted==canonical BYTE-IDENTICAL
   (hash 9007024d7792ac3e), two-run identical; `--pom-shot` → `pom:{heightScale:0.08,steps:32}` +
   heightScale=0 real==flat BYTE-IDENTICAL (hash 3d50b38bd46040f0), two-run identical; `--gtao-shot` →
   `gtao:{slices:8,steps:8,radius:0.6}` + radius=0==no-AO BYTE-IDENTICAL (hash e0a46f7a287906f8), two-run
   identical; `--froxelfog-shot` → `froxel-fog:{froxels:16x9x64,density:0.06,g:0.76}` + density=0==no-fog
   BYTE-IDENTICAL (hash 48a4d3cb2104d3ba), two-run identical; `--contactshadow-shot` →
   `contact-shadows:{steps:24,maxDist:0.9,thickness:0.6}` + maxDist=0==no-contact BYTE-IDENTICAL (hash
   a41c0ede5a8a3261), two-run identical; `--game-shot` → `score:3,won:true,steps:380`; `--audio-render` twice
   byte-identical; `--terrain-shot` → `peak:2.0972`; `--terrain-stream-shot` → `frame:45, resident:22`;
   `--anim-fsm-shot` → `blend:0.53`; `--net-shot` → `replicaMatch:true`; `--netsim-shot` → `converged:true`;
   `--netpredict-shot` → `maxMisprediction:0.0655, converged:true`; `--editor-edit-shot` →
   `saved:true, reloadMatch:true`; `--vfx-shot` → `alive:960, spawned:1800`; `--water-shot` →
   `waves:3, time:1.3, gridN:128` two-run identical; `--dof-shot` → `focalDist:12.0` two-run identical;
   `--clouds-shot` → `steps:64, coverage:0.42, time:2` two-run identical; `--poststack-shot` → `count:5`;
   `--ssgi-shot`/`--ssgi-denoise-shot`/`--ssgi-temporal-shot` two-run identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–CT: add GTAO (CR — ground-truth horizon AO, coexists with SSAO),
  froxel volumetric fog (CS — true 3D view-space single-scattering volume, distinct from the 2D light-shaft
  fog), and screen-space contact shadows (CT — fine contact occlusion augmenting the CSM). Update the
  showcase/flag table (+--gtao-shot, --froxelfog-shot, --contactshadow-shot) and the golden tally (64 image +
  1 material-JSON + 1 audio WAV + 1 engine-JSON).
- **docs/ARCHITECTURE.md** — add `engine/render/gtao.h` (horizon-search visibility integral, radius=0
  pass-through proof), `engine/render/froxel.h` (froxel grid + Beer-Lambert front-to-back single-scattering,
  density=0 no-op proof, + the `ComputeToCompute`/`ComputeToFragment` barrier interface), and
  `engine/render/contact_shadows.h` (screen-space depth-march toward the sun, maxDist=0 no-op proof). Note all
  are deterministic + pure-CPU-shared-math.
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (64) + the new goldens. Keep CI
  YAML under `ci/` (the token lacks workflow scope — do NOT create `.github/workflows/`).

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
