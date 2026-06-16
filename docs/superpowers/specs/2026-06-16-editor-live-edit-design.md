# Slice BX — Editor Live-Edit → scene_io Round-Trip (Phase 4 #23) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Completes the editor
> story: read (BT docked editor) → **EDIT** → SAVE (scene_io). Extends BT; agentic (the agent can mutate
> + persist scenes through the editor data model).

**Goal:** Make the editor mutate the scene and persist it. Add pure-CPU edit operations that change the
selected entity's components (transform + material), apply a FIXED deterministic edit in a `--editor-edit-shot`
showcase, render the result to a golden, and round-trip through `scene_io` (DumpScene) to assert the saved
scene reflects the edit. The BT inspector was read-only display; this slice adds the WRITE path
(programmatic/deterministic, not interactive — interactive mouse editing is the `--fly` path).

## Design decisions (locked)

1. **Edit operations (extend engine/editor/editor_panel_data.{h,cpp} or a new engine/editor/edit_ops.{h,cpp},
   pure CPU, no backend symbols).** Namespace `hf::editor`. Operate on the live `ecs::Registry`:
   - `void ApplyTransformEdit(ecs::Registry&, int entity, const TransformEdit&)` where `TransformEdit`
     carries optional deltas/sets for position / euler / scale (e.g. `{Vec3 position; bool setPosition;
     Vec3 euler; bool setEuler; Vec3 scale; bool setScale;}` — or delta-add semantics; pick + document).
     Mutates the entity's `TransformC`.
   - `void ApplyMaterialEdit(ecs::Registry&, int entity, const MaterialEdit&)` — optional sets for
     `metallic`, `roughness`, `baseColor` name; mutates `MaterialC`.
   - Both clamp/no-op on an out-of-range or component-less entity (safe). The edit is reflected in the
     `editor_panel_data` (the inspector would now show the new values) — verify via the data model.
   - Document the edit semantics (set vs delta).

2. **Showcase `--editor-edit-shot <out>` (Vulkan) / `--editor-edit` (Metal).** Load the default scene +
   editor state; apply a FIXED deterministic edit sequence (documented — e.g. select the duck (id 9),
   translate it +1.0 on Y; select a sphere, set its baseColor to a red swatch + metallic 1.0). Render the
   docked editor frame (BT layout) over the EDITED scene so the viewport shows the moved/recolored entity
   AND the inspector shows the edited values. Print `editor-edit: {edits:N, entity:<id>, ...}`. New golden
   `tests/golden/metal/editor_edit.png` (Metal two runs DIFF 0.0000). Existing 46 image goldens UNTOUCHED
   (incl. `editor.png` — the unedited editor — which stays byte-identical; this is a NEW showcase).

3. **scene_io round-trip (the persistence proof).** After applying the edits, call `scene_io::DumpScene`
   (or `SaveScene`) to a string/file and ASSERT the dumped JSON reflects the edits: the duck's transform
   has the new Y, the sphere's material has the new baseColor/metallic. Optionally re-`LoadScene` the dumped
   JSON into a fresh registry and assert it round-trips (the reloaded entity has the edited values). Print
   the round-trip result in the stat line (`saved:true, reloadMatch:true`). This proves edit → save →
   load is consistent — the editor can persist changes.

4. **Determinism.** Fixed edit sequence, fixed scene, fixed render (no time/input). Two runs byte-identical.

5. **Tests `tests/editor_edit_test.cpp` (pure CPU, no GPU):**
   - **ApplyTransformEdit:** sets/deltas the right TransformC fields for a known entity; leaves others
     unchanged; out-of-range entity is a safe no-op.
   - **ApplyMaterialEdit:** sets metallic/roughness/baseColor on the right MaterialC; component-less entity
     safe.
   - **Panel-data reflection:** after an edit, `editor_panel_data` reports the NEW values for the selected
     entity (the read path sees the write).
   - **scene_io round-trip:** apply edits → `DumpScene` → the JSON contains the edited values; `LoadScene`
     of that JSON into a fresh registry yields an entity with the edited transform/material (reload match).
   - **Determinism:** the edit sequence applied twice yields identical registry state + identical dump.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `editor-live-edit` (features) + `--editor-edit-shot` (showcases).

## RHI seam additions (summary)
- **None.** Pure-CPU registry mutation + the existing scene_io + the BT editor render. New/changed files
  (`engine/editor/edit_ops.*` or `editor_panel_data.*` additions, `tests/editor_edit_test.cpp`) add ZERO
  backend symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Interactive mouse/drag editing (that's the `--fly` path), undo/redo, multi-entity edits, adding/removing
entities via the editor (scene_io already supports add/remove via commands), a property-editing ImGui
widget that mutates on click (the edit here is programmatic/deterministic for the golden — wiring it to a
live ImGui widget is the interactive path), gizmo-driven edits (Slice AB has gizmos for the interactive
path), material-graph editing. One deterministic transform+material edit, golden + scene_io round-trip
asserted.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 46) + new `editor_edit_test` (transform/material
   edit, panel reflection, scene_io round-trip, determinism). Clean under `windows-msvc-asan`.
2. `--editor-edit-shot` on Windows/Vulkan: controller visual review — the docked editor shows the EDITED
   scene (the moved/recolored entity in the viewport + the new values in the inspector), coherent; the
   `editor-edit: {... saved:true, reloadMatch:true}` line is deterministic (two runs → byte-identical
   capture). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --editor-edit` → new golden `tests/golden/metal/editor_edit.png`; two runs DIFF
   0.0000; the stat line matches Vulkan.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `editor_edit.png`
   added; the other 46 byte-identical — CRITICALLY `editor.png` (the unedited editor) unchanged (proves
   the edit path is additive).
5. Introspect JSON rebaked exactly `+editor-live-edit` + `--editor-edit-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `editor_edit`
   image golden in the Mac round-trip loop.
