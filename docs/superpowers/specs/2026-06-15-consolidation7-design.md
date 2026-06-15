# Slice BG — Consolidation #7 (full A–BF re-verify + docs refresh) — Design

> Consolidation pass (DUE: BD, BE, BF = 3 slices since the BC #6 consolidation). Re-verify the ENTIRE
> engine cross-platform + refresh docs. Render-invariant: docs/CI text only.

**Goal:** Prove the whole A–BF engine still passes the full cross-platform gate after BD (streaming), BE
(normal-map node), BF (terrain), and refresh README/ARCHITECTURE/CI. Commit touches ONLY docs/CI text
(like AO/AY/BC). If re-verification surfaces a real defect, fix it minimally + report it.

## Re-verification matrix (all must pass; report each result)

1. **Windows ctest** debug → 34/34.
2. **ASan** `windows-msvc-asan` ctest → 34/34, zero memory errors.
3. **All 35 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). The 35:
   scene_shadow, skinning, pbr_helmet, instanced, ibl_helmet, physics, transparency, bloom, scene_import,
   debug_viz, anim_blend, ssao, capstone, camera_pose, gizmo, csm, spot, point_shadow, clustered, ssr,
   volumetric, probe, taa, cull, gpu_cull, mt, mat_graph, mat_graph2, mat_multi, mat_normal, game, hud,
   game_hud, stream, terrain.
4. **Audio WAV golden** `tests/golden/audio/scene.wav` byte-exact vs a fresh `--audio-render` on Windows.
5. **Introspect JSON golden** byte-exact vs live `--introspect`.
6. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation`
   (sync + core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign
   `[WARNING: Performance]` notices allowed; document them).
7. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch
   lines); other matches are comments. Report the list.
8. **Runtime==build-time**: `--material-live-shot` (default showcase.mat.json) SHA == `--material-shot` SHA.
9. **Determinism spot-checks**: `--mt-shot --workers 1`==`--workers 4` SHA; `--game-shot` prints
   `score:3,won:true,steps:380`; `--audio-render` twice → byte-identical WAV; `--terrain-shot` prints
   `n:128,verts:16384,tris:32258,peak:2.0972`; `--stream-shot` prints the fixed frame:40/resident:24 line.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–BF: add the material-graph normal-map node, scene/asset
  streaming, and procedural terrain. Update the showcase/flag table (+--material-normal-shot,
  --stream-shot, --terrain-shot) and the golden tally (35 image + 1 audio WAV + 1 JSON).
- **docs/ARCHITECTURE.md** — add the NormalMap node + PBROutput.normal (and the hfShadePBRN lighting
  variant), `engine/scene/streaming.*` (distance residency + hysteresis + per-frame budget), and
  `engine/terrain/*` (procedural heightmap + finite-difference-normal mesh gen, shared into both builds).
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (34) + the new goldens. Keep
  CI YAML under `ci/` (token lacks workflow scope — do NOT move to `.github/workflows/`).

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
