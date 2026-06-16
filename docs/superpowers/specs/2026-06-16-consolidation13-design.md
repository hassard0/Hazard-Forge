# Slice CE — Consolidation #13 (full A–CD re-verify + docs refresh) — Design

> Consolidation pass (DUE: CB, CC, CD = 3 slices since CA #12). Re-verify the ENTIRE engine cross-platform
> + refresh docs. Render-invariant: docs/CI text only. (Controller dispatches this with a 3600s fallback —
> the re-verify legitimately runs ~52 min.)

**Goal:** Prove the whole A–CD engine still passes the full cross-platform gate after CB (GPU-driven
combined pass), CC (VFX emitter), CD (fully-GPU-driven-culled pass), and refresh README/ARCHITECTURE/CI.
Commit touches ONLY docs/CI text. If re-verification surfaces a real defect, fix it minimally + report it.

## Re-verification matrix (all must pass; report each result)

1. **Windows ctest** debug → 52/52.
2. **ASan** `windows-msvc-asan` ctest → 52/52 (run inside the VS dev shell so the ASan runtime DLL
   resolves), zero memory errors.
3. **All 52 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). Enumerate
   from `tests/golden/metal/*.png` (the prior 49 + `gpudriven`, `vfx`, `gpucull_draw`).
4. **Material-graph JSON golden** byte-exact vs fresh `--material-introspect assets/materials/showcase3.mat.json`.
5. **Audio WAV golden** byte-exact vs a fresh `--audio-render` on Windows.
6. **Engine-introspect JSON golden** byte-exact vs live `--introspect`.
7. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation`
   (sync + core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign
   `[WARNING: Performance]` notices allowed; document them).
8. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch
   lines); other matches are comments. Report the list.
9. **Runtime==build-time + determinism spot-checks**: `--material-live-shot` SHA == `--material-shot` SHA;
   `--mt-shot --workers 1`==`--workers 4` SHA; `--mdi-shot` → `mdi==per-draw: BYTE-IDENTICAL`,
   `drawCalls:1, refDrawCalls:144`; `--bindless-shot` → `bindless==bound: BYTE-IDENTICAL`, `textureBinds:1,
   refTextureBinds:5`; `--gpudriven-shot` → `gpudriven==bound: BYTE-IDENTICAL`, `drawCalls:1,
   textureBinds:1`; `--gpucull-draw-shot` → `gpucull-draw==bound: BYTE-IDENTICAL`, `total:144, drawn:107,
   cpuRef:107`; `--game-shot` → `score:3,won:true,steps:380`; `--audio-render` twice → byte-identical;
   `--terrain-shot` → `n:128,...,peak:2.0972`; `--terrain-stream-shot` → `frame:45, resident:22`;
   `--anim-fsm-shot` → `state:walk->run, blend:0.53`; `--net-shot` → `replicaMatch:true, savings:43.5%`;
   `--netsim-shot` → `delivered:226, dropped:42, converged:true`; `--netpredict-shot` →
   `maxMisprediction:0.0655, converged:true`; `--editor-shot` → `selected:9, entities:10`;
   `--editor-edit-shot` → `edits:2, saved:true, reloadMatch:true`; `--vfx-shot` → `alive:960,
   spawned:1800`; `--poststack-shot` → `count:5`; `--ssgi-shot`/`--ssgi-denoise-shot`/`--ssgi-temporal-shot`
   two-run byte-identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–CD: add the fully-GPU-driven combined pass (CB), the CPU
  particle/VFX emitter (CC), and the fully-GPU-driven-CULLED pass (CD — completing the GPU-driven pipeline
  cull→compact→MDI+bindless in one pass). Update the showcase/flag table (+--gpudriven-shot, --vfx-shot,
  --gpucull-draw-shot) and the golden tally (52 image + 1 material-JSON + 1 audio WAV + 1 engine-JSON).
- **docs/ARCHITECTURE.md** — add the GPU-driven pipeline section (`engine/render/gpu_driven.h` +
  `gpu_culled.h` + `shaders/lit_gpudriven.*`/`gpudriven_cull.comp.hlsl`: compute frustum-cull + ordered
  compaction writing the MDI drawCount → 1 indirect draw + 1 bindless bind; reuses AR cull + BM MDI + BZ
  bindless) and `engine/vfx/particles.*` (CPU emitter + lifetime curves + additive billboards).
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (52) + the new goldens. Keep CI
  YAML under `ci/`.

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
