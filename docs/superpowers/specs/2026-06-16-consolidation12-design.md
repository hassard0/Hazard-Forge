# Slice CA — Consolidation #12 (full A–BZ re-verify + docs refresh) — Design

> Consolidation pass (DUE: BX, BY, BZ = 3 slices since BW #11). Re-verify the ENTIRE engine cross-platform
> + refresh docs. Render-invariant: docs/CI text only.

**Goal:** Prove the whole A–BZ engine still passes the full cross-platform gate after BX (editor live-edit),
BY (client prediction), BZ (bindless textures), and refresh README/ARCHITECTURE/CI. Commit touches ONLY
docs/CI text. If re-verification surfaces a real defect, fix it minimally + report it.

## Re-verification matrix (all must pass; report each result)

1. **Windows ctest** debug → 49/49.
2. **ASan** `windows-msvc-asan` ctest → 49/49 (run inside the VS dev shell so the ASan runtime DLL
   resolves), zero memory errors.
3. **All 49 Metal image goldens** on the Apple M4 → each DIFF 0.0000 (re-render, NO rebake). Enumerate
   from `tests/golden/metal/*.png` (the prior 46 + `editor_edit`, `netpredict`, `bindless`).
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
   `drawCalls:1, refDrawCalls:144`; `--bindless-shot` → `bindless==bound: BYTE-IDENTICAL`, `textureBinds:1,
   refTextureBinds:5`; `--game-shot` → `score:3,won:true,steps:380`; `--audio-render` twice → byte-identical;
   `--terrain-shot` → `n:128,verts:16384,tris:32258,peak:2.0972`; `--terrain-stream-shot` → `frame:45,
   resident:22, lod0:4, lod1:8, lod2:10`; `--anim-fsm-shot` → `state:walk->run, blend:0.53, step:37`;
   `--net-shot` → `replicaMatch:true, savings:43.5%`; `--netsim-shot` → `delivered:226, dropped:42,
   converged:true`; `--netpredict-shot` → `maxMisprediction:0.0655, converged:true`; `--editor-shot` →
   `selected:9, entities:10`; `--editor-edit-shot` → `edits:2, saved:true, reloadMatch:true`;
   `--poststack-shot` → `count:5`; `--ssgi-shot`/`--ssgi-denoise-shot`/`--ssgi-temporal-shot` two-run
   byte-identical.

## Docs to refresh (the only files the commit should change, besides this spec)

- **README.md** — capability summary now A–BZ: add editor live-edit + scene_io round-trip, client
  prediction + reconciliation (completing the networking trilogy), and bindless textures (completing the
  GPU-driven story with MDI). Update the showcase/flag table (+--editor-edit-shot, --netpredict-shot,
  --bindless-shot) and the golden tally (49 image + 1 material-JSON + 1 audio WAV + 1 engine-JSON).
- **docs/ARCHITECTURE.md** — add `engine/editor/edit_ops.*` (registry mutation + scene_io round-trip),
  `engine/net/prediction.*` (predict + rewind/replay reconciliation; AuthState carries the full RigidBody),
  and `engine/render/bindless.* + rhi_vulkan/vulkan_bindless.*` (descriptor-indexing array + NonUniform;
  Vulkan-true, Metal-bound-golden; pairs with MDI).
- **ci/github-actions-ci.yml** + **ci/README.md** — reflect the test count (49) + the new goldens. Keep CI
  YAML under `ci/`.

## Out of scope (YAGNI)
Any new feature, shader, golden, or engine/behavior change; any refactor. Re-verify + docs only.

## Verification gate
1. The full matrix above passes; the VERIFY report states each result explicitly.
2. `git show --stat` touches ONLY docs/CI files (README.md, docs/ARCHITECTURE.md, ci/github-actions-ci.yml,
   ci/README.md) + this spec — ZERO engine/shader/test/golden changes (`git diff master --stat --
   tests/golden` EMPTY; no engine/ or shaders/ files).
3. If any matrix item FAILS, STOP and report — no golden rebake, no layer disable.
