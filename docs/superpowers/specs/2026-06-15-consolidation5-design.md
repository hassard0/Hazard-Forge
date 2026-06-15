# Slice AY — Consolidation #5 (full A–AX re-verify + docs refresh) — Design

> Autonomous-session spec (standing directive). Consolidation pass (DUE: 3 Phase-4 slices AV–AX since the
> AT #4 hardening). Re-verify the ENTIRE engine cross-platform and refresh the project docs. Render-
> invariant: no engine/shader/golden behavior changes — only re-verification + docs/CI text.

**Goal:** Prove the whole A–AX engine still passes the full cross-platform gate after the Phase-4
additions (material graph, live authoring, game sample), and bring the human-facing docs (README,
ARCHITECTURE, CI) up to date. This is a checkpoint, not a feature — the commit should touch ONLY docs/CI
text (like consolidation #3 / Slice AO). If re-verification surfaces a real defect, fix it minimally and
report it; otherwise the engine code, shaders, and goldens are untouched.

## Re-verification matrix (all must pass; report each result)

1. **Windows ctest** `--preset windows-msvc-debug` → 30/30.
2. **ASan** `windows-msvc-asan` ctest → 30/30, zero memory errors.
3. **All 29 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render via `metal_headless`,
   compare; NO rebake). The 29: scene_shadow, skinning, pbr_helmet, instanced, ibl_helmet, physics,
   transparency, bloom, scene_import, debug_viz, anim_blend, ssao, capstone, camera_pose, gizmo, csm,
   spot, point_shadow, clustered, ssr, volumetric, probe, taa, cull, gpu_cull, mt, mat_graph, mat_graph2,
   game.
4. **Introspect JSON golden** byte-exact vs live `--introspect` (now lists features through gameplay-
   sample + all showcases through --game-shot).
5. **Full Vulkan validation-clean**: run EVERY `--*-shot` showcase under `VK_LAYER_KHRONOS_validation`
   (sync + core) → ZERO `VUID-*`/`SYNC-HAZARD-*`/`UNASSIGNED-*`/`[ERROR]` lines. (Benign `[WARNING:
   Performance]` notices — e.g. the depth-only-shadow "vertex attribute not consumed" — are allowed;
   document them.)
6. **Seam grep audit**: above-seam real backend CODE symbols == the pre-existing baseline (the 2
   rhi_factory dispatch lines); any other matches are comments/prose. Report the full list.
7. **Runtime==build-time** still holds: `--material-live-shot showcase` SHA == `--material-shot` SHA.
8. **Determinism spot-checks**: `--mt-shot --workers 1`==`--workers 4` SHA; `--game-shot` prints the
   fixed `score:3,won:true,steps:380` state.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — update the feature list / capability summary to reflect Phase 3 (frustum cull, GPU-
  driven culling + indirect draw, render-graph auto-barriers, validation-clean, multithreaded recording)
  and Phase 4 (data-driven material/shader graph, live runtime material authoring, playable game sample).
  Update the showcase/flag table and the golden count (29 image + 1 JSON). Keep the build instructions
  accurate.
- **docs/ARCHITECTURE.md** — add the new subsystems: `engine/material/` (shader_graph, codegen,
  material_loader, runtime_compile, live_material) + the build-time codegen tool flow + the dxc-subprocess
  runtime path; `engine/game/` (roll_game) gameplay layer; the render-graph resource-state/barrier solver;
  the parallel-record worker pool. Note the validation-clean invariant + the conan validation-layer
  dependency.
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the current test count (30) and any new
  build steps (material_codegen tool, the validation gate). Keep CI YAML under `ci/` (the token lacks
  workflow scope — do NOT move it to `.github/workflows/`).

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change. Any refactor. This is re-verify + docs only.
If a defect is found, fix it minimally + render-invariantly and call it out explicitly; do not expand
scope.

## Verification gate
1. The full re-verification matrix above passes; the VERIFY report states each result explicitly
   (ctest debug 30/30, asan 30/30, 29 Metal goldens DIFF 0.0000, introspect byte-exact, all showcases
   validation-clean, seam baseline, runtime==build-time, determinism spot-checks).
2. `git show --stat` for the commit touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md,
   ci/github-actions-ci.yml, ci/README.md) + this spec — ZERO engine/shader/test/golden changes
   (`git diff master --stat -- tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — do not paper over (no golden rebake, no layer disable).
