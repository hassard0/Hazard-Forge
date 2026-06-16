# Slice CI — Consolidation #14 (full A–CH re-verify + docs refresh) — Design

> Consolidation pass (DUE: CF, CG, CH = 3 slices since CE #13). Re-verify the ENTIRE engine cross-platform
> + refresh docs. Render-invariant: docs/CI text only. (Controller dispatches this with a 3600s fallback —
> the re-verify legitimately runs ~52 min.)

**Goal:** Prove the whole A–CH engine still passes the full cross-platform gate after CF (water), CG (depth
of field), CH (volumetric clouds), and refresh README/ARCHITECTURE/CI. Commit touches ONLY docs/CI text.
If re-verification surfaces a real defect, fix it minimally + report it.

## Re-verification matrix (all must pass; report each result)

1. **Windows ctest** debug → 55/55.
2. **ASan** `windows-msvc-asan` ctest → 55/55 (run inside the VS dev shell), zero memory errors.
3. **All 55 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). Enumerate
   from `tests/golden/metal/*.png` (the prior 52 + `water`, `dof`, `clouds`).
4. **Material-graph JSON golden** byte-exact vs fresh `--material-introspect assets/materials/showcase3.mat.json`.
5. **Audio WAV golden** byte-exact vs a fresh `--audio-render` on Windows.
6. **Engine-introspect JSON golden** byte-exact vs live `--introspect`.
7. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation`
   (sync + core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign
   `[WARNING: Performance]` notices allowed; document them).
8. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch
   lines); other matches are comments. Report the list.
9. **Runtime==build-time + determinism spot-checks**: `--material-live-shot`==`--material-shot` SHA;
   `--mt-shot --workers 1`==`--workers 4` SHA; `--mdi-shot` → `BYTE-IDENTICAL, drawCalls:1, refDrawCalls:144`;
   `--bindless-shot` → `BYTE-IDENTICAL, textureBinds:1`; `--gpudriven-shot` → `BYTE-IDENTICAL, drawCalls:1,
   textureBinds:1`; `--gpucull-draw-shot` → `BYTE-IDENTICAL, total:144, drawn:107, cpuRef:107`; `--game-shot`
   → `score:3,won:true,steps:380`; `--audio-render` twice byte-identical; `--terrain-shot` → `peak:2.0972`;
   `--terrain-stream-shot` → `frame:45, resident:22`; `--anim-fsm-shot` → `blend:0.53`; `--net-shot` →
   `replicaMatch:true`; `--netsim-shot` → `converged:true`; `--netpredict-shot` → `maxMisprediction:0.0655,
   converged:true`; `--editor-edit-shot` → `saved:true, reloadMatch:true`; `--vfx-shot` → `alive:960,
   spawned:1800`; `--water-shot` → `waves:3, time:1.3, gridN:128` two-run byte-identical; `--dof-shot` →
   `focalDist:12.0` two-run byte-identical; `--clouds-shot` → `steps:64, coverage:0.42, time:2` two-run
   byte-identical; `--poststack-shot` → `count:5`; `--ssgi-shot`/`--ssgi-denoise-shot`/`--ssgi-temporal-shot`
   two-run byte-identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–CH: add water rendering (CF), depth of field (CG), and
  volumetric clouds (CH — completing the sky/atmosphere suite: sky + water + clouds). Update the
  showcase/flag table (+--water-shot, --dof-shot, --clouds-shot) and the golden tally (55 image + 1
  material-JSON + 1 audio WAV + 1 engine-JSON).
- **docs/ARCHITECTURE.md** — add `engine/render/water.h` (Gerstner waves + fresnel reflect/refract),
  `engine/render/dof.h` (thin-lens CoC depth gather), and `engine/render/clouds.h` (raymarched cumulus +
  Beer-Lambert + HG phase). Note all three are deterministic-at-fixed-time, pure-CPU-shared-math + a
  fullscreen/water-mesh pass.
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (55) + the new goldens. Keep CI
  YAML under `ci/`.

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
