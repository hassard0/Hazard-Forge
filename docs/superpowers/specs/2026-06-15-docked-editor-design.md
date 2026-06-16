# Slice BT — Docked Editor UI (golden-verified) — Phase 4 #19 (depth pivot #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The third big-ticket
> gap — a docked GUI editor — completed into a golden-verified showcase by leveraging the EXISTING ImGui
> integration (`engine/editor/editor_panels.*` + `engine/editor/imgui_renderer.*` + `third_party/imgui`).

**Goal:** Render a docked, multi-panel editor frame — Scene Hierarchy + Inspector + Stats panels in an
ImGui dockspace over/beside the rendered scene viewport, with a FIXED selected entity — to an offscreen
golden. The panels already build from live ECS data (`BuildEditorUI`); this slice arranges them into a
proper DOCKED layout, adds a viewport region, and wires a deterministic `--editor-shot` capture so the
editor UI is golden-verified cross-backend. This delivers the "docked editor" milestone with the same
golden rigor as every other subsystem.

## Why this is verifiable (deterministic ImGui)

ImGui geometry is CPU-built and deterministic for fixed input: fixed `io.DisplaySize`, a fixed dockspace
layout, a FIXED selected entity, and NO time/animation/cursor input → identical vertex/index buffers every
run, and identical across backends (the geometry is backend-agnostic; the RHI just draws textured quads,
which the engine already golden-matches cross-backend at DIFF 0.0000). The font atlas is baked
deterministically. So a fixed-state editor frame is golden-stable. (Interactive editing — live mouse/drag
in a real window — remains the Vulkan `--fly` path / the Mac windowed path; THIS slice verifies the
rendered editor CHROME, deterministically.)

## Design decisions (locked)

1. **Docked layout (extend engine/editor/editor_panels.{h,cpp}, ImGui only, no backend symbols).**
   `BuildEditorUI` already builds Scene Hierarchy / Inspector / Stats from the live registry. Add/ensure a
   proper **DockSpace** layout that docks: Hierarchy (left), Inspector (right), Stats (bottom-left or an
   overlay), and a central **Viewport** panel. Build the dock layout DETERMINISTICALLY in code (ImGui
   `DockBuilder` with a fixed split ratios + a fixed dock-node id seeded the same each run — do NOT load
   an `imgui.ini` from disk, which would be machine-dependent; call `ImGui::LoadIniSettingsFromMemory` with
   a fixed string OR build the layout programmatically each frame the first time). Document the layout +
   split ratios. The Inspector shows the FIXED `EditorState.selectedEntity`'s TransformC/MaterialC/MeshC.
   The Viewport panel displays the rendered scene (the scene color RT as an ImGui image, OR the scene is
   drawn first and the viewport panel framed over it — pick the approach that the existing
   imgui_renderer/RT plumbing supports + document).

2. **Determinism harness.** A fixed `io.DisplaySize` (e.g. 1280x720), a fixed selected entity, NO time
   (`io.DeltaTime` a fixed constant, no animated widgets), no cursor/input. Disable `imgui.ini`
   persistence (`io.IniFilename = nullptr`) so layout is fully code-driven. Two runs → byte-identical.

3. **Showcase `--editor-shot <out>` (Vulkan) / `--editor` (Metal).** Set up the default scene + the
   editor state (a FIXED selected entity, e.g. the duck or a sphere), run one frame: render the scene to
   the viewport + `BuildEditorUI` (docked panels) → `imgui_renderer` draws the ImGui draw data over it →
   capture. Print `editor: {panels:[Hierarchy,Inspector,Stats,Viewport], selected:<id>, entities:<n>}`.
   New golden `tests/golden/metal/editor.png` (Metal two runs DIFF 0.0000). Existing 43 image goldens
   UNTOUCHED.

4. **Tests `tests/editor_panels_test.cpp` (pure CPU, no GPU — or extend an existing editor test):**
   - **Panel data model:** for a known registry + scene resources, the hierarchy lists the expected
     entities (ids/labels), the inspector for `selectedEntity` reports the expected Transform/Material/
     Mesh values, and the stats panel reports the expected counts (entity/mesh/light counts). Factor the
     panel DATA (what each panel WOULD display) into a testable function if it isn't already, so the test
     is GPU/ImGui-free (assert the data, not the pixels). Document the seam between data + ImGui calls.
   - **Selection:** changing `EditorState.selectedEntity` changes which entity the inspector reports;
     out-of-range selection is handled (clamped / "none").
   - **Determinism:** the panel data for a fixed registry is stable across calls.
   - Clean under `windows-msvc-asan`.

5. **Introspect.** Add exactly `docked-editor` (features) + `--editor-shot` (showcases).

## RHI seam additions (summary)
- **None expected** — the ImGui render path (`imgui_renderer`) + the scene RT already exist; the editor
  panels are ImGui-only (the header states "no rhi/backend symbols"). If the viewport-as-ImGui-image needs
  a tiny additive hook (bind the scene RT as an ImGui texture id), keep it pure-interface in `rhi.h` with
  backend impls inside the backend dirs — but PREFER drawing the scene first + the editor chrome over it
  (no new seam). New/changed files (`engine/editor/editor_panels.*`, `tests/editor_panels_test.cpp`) add
  ZERO backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Live interactive editing in a window (that's the `--fly` / Mac-windowed path — not this slice), drag-drop
reparenting, multi-select, undo/redo, asset browser, a save/load of the editor layout to disk, property
editing widgets that mutate state (the inspector is READ-ONLY display this slice — editing is the
interactive path), theming/skins, multiple viewports. One deterministic docked editor frame
(Hierarchy/Inspector/Stats/Viewport) golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 43) + new/extended `editor_panels_test` (panel data
   model, selection, determinism). Clean under `windows-msvc-asan`.
2. `--editor-shot` on Windows/Vulkan: controller visual review — a recognizable DOCKED editor (Hierarchy
   list on the left, Inspector showing the selected entity's components on the right, Stats, and the scene
   in the central viewport), legible and correctly laid out; the `editor: {...}` line is deterministic
   (two runs → byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --editor` → new golden `tests/golden/metal/editor.png`; two runs DIFF 0.0000; the
   editor stat line matches Vulkan.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `editor.png` added; the
   other 43 byte-identical.
5. Introspect JSON rebaked exactly `+docked-editor` + `--editor-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `editor` image
   golden in the Mac round-trip loop.
