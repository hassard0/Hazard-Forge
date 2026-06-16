# Slice BS — Consolidation #10 (full A–BR re-verify + docs refresh) — Design

> Consolidation pass (DUE: BP, BQ, BR = 3 slices since the BO #9 consolidation). Re-verify the ENTIRE
> engine cross-platform + refresh docs. Render-invariant: docs/CI text only.

**Goal:** Prove the whole A–BR engine still passes the full cross-platform gate after BP (SSGI), BQ
(replication), BR (SSGI denoise), and refresh README/ARCHITECTURE/CI. Commit touches ONLY docs/CI text.
If re-verification surfaces a real defect, fix it minimally + report it.

## Re-verification matrix (all must pass; report each result)

1. **Windows ctest** debug → 43/43.
2. **ASan** `windows-msvc-asan` ctest → 43/43, zero memory errors.
3. **All 43 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). The 43 are
   the prior 40 + `ssgi`, `net`, `ssgi_denoise`. (Enumerate from `tests/golden/metal/*.png`.)
4. **Material-graph JSON golden** `tests/golden/material/showcase3_graph.json` byte-exact vs fresh
   `--material-introspect assets/materials/showcase3.mat.json`.
5. **Audio WAV golden** `tests/golden/audio/scene.wav` byte-exact vs a fresh `--audio-render` on Windows.
6. **Engine-introspect JSON golden** byte-exact vs live `--introspect`.
7. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation`
   (sync + core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign
   `[WARNING: Performance]` notices allowed; document them).
8. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch
   lines); other matches are comments. Report the list.
9. **Runtime==build-time + determinism spot-checks**: `--material-live-shot` SHA == `--material-shot` SHA;
   `--mt-shot --workers 1`==`--workers 4` SHA; `--mdi-shot` → `mdi==per-draw: BYTE-IDENTICAL` +
   `drawCalls:1, refDrawCalls:144`; `--game-shot` → `score:3,won:true,steps:380`; `--audio-render` twice →
   byte-identical; `--terrain-shot` → `n:128,verts:16384,tris:32258,peak:2.0972`; `--terrain-stream-shot`
   → `frame:45, resident:22, lod0:4, lod1:8, lod2:10`; `--anim-fsm-shot` → `state:walk->run, blend:0.53,
   speed:0.925, step:37`; `--net-shot` → `replicaMatch:true, savings:43.5%` (fullBytes:37233,
   deltaBytes:21045); `--poststack-shot` → `count:5`; `--ssgi-shot`/`--ssgi-denoise-shot` two-run
   byte-identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–BR: add screen-space global illumination + its denoise, and
  the state-replication snapshot layer. Update the showcase/flag table (+--ssgi-shot, --ssgi-denoise-shot,
  --net-shot) and the golden tally (43 image + 1 material-JSON + 1 audio WAV + 1 engine-JSON). Note the
  "depth pivot" (GI + networking) beyond the breadth phase.
- **docs/ARCHITECTURE.md** — add `engine/render/ssgi.*` (screen-space GI reusing the SSR march + G-buffer;
  + the bilateral denoise), and `engine/net/snapshot.*` (deterministic snapshot/delta replication core,
  in-process channel, no transport).
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (43) + the new goldens. Keep
  CI YAML under `ci/`.

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
