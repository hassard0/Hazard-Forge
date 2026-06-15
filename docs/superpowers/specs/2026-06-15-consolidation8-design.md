# Slice BK — Consolidation #8 (full A–BJ re-verify + docs refresh) — Design

> Consolidation pass (DUE: BH, BI, BJ = 3 slices since the BG #7 consolidation). Re-verify the ENTIRE
> engine cross-platform + refresh docs. Render-invariant: docs/CI text only.

**Goal:** Prove the whole A–BJ engine still passes the full cross-platform gate after BH (decals), BI
(material-graph introspection), BJ (terrain-streaming LOD), and refresh README/ARCHITECTURE/CI. Commit
touches ONLY docs/CI text. If re-verification surfaces a real defect, fix it minimally + report it.

## Re-verification matrix (all must pass; report each result)

1. **Windows ctest** debug → 37/37.
2. **ASan** `windows-msvc-asan` ctest → 37/37, zero memory errors.
3. **All 37 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). The 37:
   scene_shadow, skinning, pbr_helmet, instanced, ibl_helmet, physics, transparency, bloom, scene_import,
   debug_viz, anim_blend, ssao, capstone, camera_pose, gizmo, csm, spot, point_shadow, clustered, ssr,
   volumetric, probe, taa, cull, gpu_cull, mt, mat_graph, mat_graph2, mat_multi, mat_normal, game, hud,
   game_hud, stream, terrain, decal, terrain_stream.
4. **Material-graph JSON golden** `tests/golden/material/showcase3_graph.json` byte-exact vs a fresh
   `--material-introspect assets/materials/showcase3.mat.json`.
5. **Audio WAV golden** `tests/golden/audio/scene.wav` byte-exact vs a fresh `--audio-render` on Windows.
6. **Engine-introspect JSON golden** byte-exact vs live `--introspect`.
7. **Full Vulkan validation-clean**: every `--*-shot` showcase under `VK_LAYER_KHRONOS_validation`
   (sync + core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines (benign
   `[WARNING: Performance]` notices allowed; document them).
8. **Seam grep audit**: above-seam real backend CODE symbols == baseline (the 2 rhi_factory dispatch
   lines); other matches are comments. Report the list.
9. **Runtime==build-time + determinism spot-checks**: `--material-live-shot` SHA == `--material-shot` SHA;
   `--mt-shot --workers 1`==`--workers 4` SHA; `--game-shot` → `score:3,won:true,steps:380`;
   `--audio-render` twice → byte-identical; `--terrain-shot` → `n:128,verts:16384,tris:32258,peak:2.0972`;
   `--terrain-stream-shot` → `frame:45, resident:22, lod0:4, lod1:8, lod2:10`.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–BJ: add decals, material-graph introspection (JSON/DOT), and
  terrain-streaming LOD. Update the showcase/flag table (+--decal-shot, --material-introspect,
  --terrain-stream-shot) and the golden tally (37 image + 1 material-JSON + 1 audio WAV + 1 engine-JSON).
- **docs/ARCHITECTURE.md** — add `engine/render/decal.h` (screen-space projected decals reusing the
  G-buffer), `engine/material/graph_introspect.*` (JSON/DOT dump completing author→render→inspect), and
  `engine/terrain/terrain_stream.*` (tile streaming + distance-banded LOD over the shared global height).
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (37) + the new goldens. Keep
  CI YAML under `ci/`.

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
