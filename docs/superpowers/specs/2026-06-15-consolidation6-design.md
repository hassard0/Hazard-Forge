# Slice BC — Consolidation #6 (full A–BB re-verify + docs refresh) — Design

> Autonomous-session spec. Consolidation pass (DUE: AZ, BA, BB = 3 slices since the AY #5 consolidation).
> Re-verify the ENTIRE engine cross-platform and refresh the project docs. Render-invariant: docs/CI text
> only — no engine/shader/golden behavior changes.

**Goal:** Prove the whole A–BB engine still passes the full cross-platform gate after AZ (material-graph
node expansion), BA (text/HUD), BB (audio), and refresh README/ARCHITECTURE/CI. Commit touches ONLY
docs/CI text (like AO/AY). If re-verification surfaces a real defect, fix it minimally + report it.

## Re-verification matrix (all must pass; report each result)

1. **Windows ctest** debug → 32/32.
2. **ASan** `windows-msvc-asan` ctest → 32/32, zero memory errors.
3. **All 32 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). The 32:
   scene_shadow, skinning, pbr_helmet, instanced, ibl_helmet, physics, transparency, bloom, scene_import,
   debug_viz, anim_blend, ssao, capstone, camera_pose, gizmo, csm, spot, point_shadow, clustered, ssr,
   volumetric, probe, taa, cull, gpu_cull, mt, mat_graph, mat_graph2, mat_multi, game, hud, game_hud.
4. **Audio WAV golden** `tests/golden/audio/scene.wav` byte-exact vs a fresh `--audio-render` on Windows.
5. **Introspect JSON golden** byte-exact vs live `--introspect`.
6. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation`
   (sync + core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign
   `[WARNING: Performance]` notices allowed; document them).
7. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch
   lines); other matches are comments. Report the list.
8. **Runtime==build-time**: `--material-live-shot showcase` SHA == `--material-shot` SHA.
9. **Determinism spot-checks**: `--mt-shot --workers 1`==`--workers 4` SHA; `--game-shot` prints
   `score:3,won:true,steps:380`; `--audio-render` twice → byte-identical WAV.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–BB: add Phase 4's material-graph node expansion + multi-
  material scene, the text/HUD renderer, and the audio mixer. Update the showcase/flag table (+--material-
  multi-shot, --hud-shot, --game-hud-shot, --audio-render) and the golden tally (32 image + 1 audio WAV +
  1 JSON).
- **docs/ARCHITECTURE.md** — add the expanded material-graph node set, `engine/ui/` (baked-font text
  renderer + screen-space overlay pass), and `engine/audio/` (integer/fixed-point mixer + WAV writer,
  pure-CPU in hf_core). Note the audio WAV golden category.
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (32) + the audio WAV golden
  check. Keep CI YAML under `ci/` (token lacks workflow scope — do NOT move to `.github/workflows/`).

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only. If a
defect is found, fix it minimally + render-invariantly and call it out; do not expand scope.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
