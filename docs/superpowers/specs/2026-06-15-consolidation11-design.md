# Slice BW — Consolidation #11 (full A–BV re-verify + docs refresh) — Design

> Consolidation pass (DUE: BT, BU, BV = 3 slices since the BS #10 consolidation). Re-verify the ENTIRE
> engine cross-platform + refresh docs. Render-invariant: docs/CI text only.

**Goal:** Prove the whole A–BV engine still passes the full cross-platform gate after BT (docked editor),
BU (net transport), BV (temporal SSGI), and refresh README/ARCHITECTURE/CI. Commit touches ONLY docs/CI
text. If re-verification surfaces a real defect, fix it minimally + report it.

## Re-verification matrix (all must pass; report each result)

1. **Windows ctest** debug → 46/46.
2. **ASan** `windows-msvc-asan` ctest → 46/46, zero memory errors.
3. **All 46 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). Enumerate
   from `tests/golden/metal/*.png` (the prior 43 + `editor`, `netsim`, `ssgi_temporal`).
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
   `--mt-shot --workers 1`==`--workers 4` SHA; `--mdi-shot` → `mdi==per-draw: BYTE-IDENTICAL`,
   `drawCalls:1, refDrawCalls:144`; `--game-shot` → `score:3,won:true,steps:380`; `--audio-render` twice →
   byte-identical; `--terrain-shot` → `n:128,verts:16384,tris:32258,peak:2.0972`; `--terrain-stream-shot`
   → `frame:45, resident:22, lod0:4, lod1:8, lod2:10`; `--anim-fsm-shot` → `state:walk->run, blend:0.53,
   speed:0.925, step:37`; `--net-shot` → `replicaMatch:true, savings:43.5%`; `--netsim-shot` →
   `delivered:226, dropped:42, reordered:29, converged:true`; `--editor-shot` → `selected:9, entities:10`
   two-run byte-identical; `--poststack-shot` → `count:5`; `--ssgi-shot`/`--ssgi-denoise-shot`/
   `--ssgi-temporal-shot` two-run byte-identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–BV: add the docked editor, the simulated network transport +
  client interpolation, and temporal SSGI (the GI trilogy: raw → spatial denoise → temporal). Update the
  showcase/flag table (+--editor-shot, --netsim-shot, --ssgi-temporal-shot) and the golden tally (46 image
  + 1 material-JSON + 1 audio WAV + 1 engine-JSON).
- **docs/ARCHITECTURE.md** — add `engine/editor/editor_panel_data.*` + the docked editor (fixed-tile
  layout, non-docking ImGui), `engine/net/transport.*` (seeded SimChannel + client jitter-buffer
  interpolation), and the temporal SSGI accumulation (golden-angle-jittered fixed-N mean).
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (46) + the new goldens. Keep
  CI YAML under `ci/`.

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
