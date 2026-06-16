# Slice BO — Consolidation #9 (full A–BN re-verify + docs refresh) — Design

> Consolidation pass (DUE: BL, BM, BN = 3 slices since the BK #8 consolidation). Re-verify the ENTIRE
> engine cross-platform + refresh docs. Render-invariant: docs/CI text only.

**Goal:** Prove the whole A–BN engine still passes the full cross-platform gate after BL (animation state
machine), BM (GPU multi-draw-indirect), BN (post-process stack), and refresh README/ARCHITECTURE/CI.
Commit touches ONLY docs/CI text. If re-verification surfaces a real defect, fix it minimally + report it.

## Re-verification matrix (all must pass; report each result)

1. **Windows ctest** debug → 40/40.
2. **ASan** `windows-msvc-asan` ctest → 40/40, zero memory errors.
3. **All 40 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). The 40:
   scene_shadow, skinning, pbr_helmet, instanced, ibl_helmet, physics, transparency, bloom, scene_import,
   debug_viz, anim_blend, ssao, capstone, camera_pose, gizmo, csm, spot, point_shadow, clustered, ssr,
   volumetric, probe, taa, cull, gpu_cull, mt, mat_graph, mat_graph2, mat_multi, mat_normal, game, hud,
   game_hud, stream, terrain, decal, terrain_stream, anim_fsm, mdi, poststack.
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
   speed:0.925, step:37`; `--poststack-shot` → `count:5` + two-run byte-identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–BN: add the animation state machine, GPU multi-draw-indirect
  batching, and the data-driven post-process stack. Update the showcase/flag table (+--anim-fsm-shot,
  --mdi-shot, --poststack-shot) and the golden tally (40 image + 1 material-JSON + 1 audio WAV + 1
  engine-JSON).
- **docs/ARCHITECTURE.md** — add `engine/anim/state_machine.*` (parameter-driven FSM + cross-fade over
  the existing skeletal anim), `engine/render/mdi.*` + the `DrawIndexedMultiIndirect`/`BindPerDrawData`
  seam + `gl_DrawID`-indexed per-draw SSBO (Vulkan true MDI; Metal per-object golden), and
  `engine/render/post_stack.*` (data-driven ordered post chain + the 128-byte flat push-constant stream
  constraint + the new ColorGrade/ChromaticAberration/FilmGrain effects).
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (40) + the new goldens. Keep
  CI YAML under `ci/`.

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
