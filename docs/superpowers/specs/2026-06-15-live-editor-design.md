# Slice AM — Live Editor Manipulation + Hot-Reload

Date: 2026-06-15
Branch: `slice-live-editor`
Status: design + implementation

## Goal

Complete the HUMAN editor inside the interactive `--fly` viewport:

1. **Live mouse pick** — left-click an object in the window to select it.
2. **Live gizmo drag** — left-drag a gizmo axis handle to translate/rotate/scale the
   selected entity's `TransformC`, updated every frame from the per-frame cursor ray.
3. **Scene + shader hot-reload** — poll file mtimes; when the active scene JSON changes,
   reload it via `LoadScene` and swap the live registry; when a watched shader `.spv`
   changes, recreate the affected shader module (best-effort, see scope below).

This is **ADDITIVE**. The 22 image goldens (`tests/golden/metal/*.png`) and the 1 JSON
golden (`tests/golden/introspect/default_scene.json`) are byte-untouched. Existing
rendering paths, shaders, and the capstone block are untouched.

## What already exists (REUSED, not reimplemented)

- `engine/editor/picking.{h,cpp}` — `ScreenRayThroughCamera(cam, ndcX, ndcY)`,
  `PickNearest(ray, span<PickAabb>)`. Pure C++ (hf_core), unit-tested.
- `engine/editor/gizmo.{h,cpp}` — `Selection`, `GizmoMode`, `EmitGizmo`,
  `PickGizmoAxis`, `ApplyDrag(transform, mode, axis, prevRay, curRay)`. Pure C++.
- `engine/runtime/play_state.h` — `PlayState` play/pause/step gate.
- `engine/scene/scene_io.{h,cpp}` — `LoadScene` / `DumpScene` JSON round-trip.
- `samples/hello_triangle/main.cpp` `--fly` loop (Slice AA) — live SDL window with the
  flyable camera, P/O play-pause-step, G/R/T gizmo-mode select, Ctrl+S `DumpScene` save.
- `cmake/CompileShaders.cmake` — HLSL→.spv compilation into `${SHADER_OUT}`.

## New / changed pieces

### 1. HAL: absolute cursor position (engine/hal/window.cpp + runtime/input_state.h)

`runtime::InputState` gains `float mouseX, mouseY` — the absolute cursor position in
**framebuffer pixels** (origin top-left, +x right, +y down — the SDL convention).
`Window::PumpEvents` fills them from `SDL_GetMouseState`, scaled by the
framebuffer/window-size ratio (SDL reports cursor in *window points*; on a HiDPI display
the framebuffer is larger). When relative-mouse (mouse-look) mode is engaged the absolute
position is frozen/meaningless, so the editor only consumes it while NOT flying.

`InputState` stays backend-agnostic (no SDL types). The HAL is the only place that reads
SDL, preserving the seam.

### 2. `--fly` live editing (samples/hello_triangle/main.cpp)

The fly loop already builds an editable ECS (`editReg` / `editRes`) with a small set of
representative objects. Slice AM makes that registry the *live editable scene*: it loads
the active scene JSON into it (so picking/drag/reload all act on real data) and renders
the selected object's gizmo.

Per-frame editor logic, gated so camera-fly and selection never fight:

- **Camera fly** is driven only while the **right mouse button** is held (mouse-look
  capture engages on right-press, releases on right-release). WASD still moves. This frees
  the left button for selection/drag. (Slice AA captured the mouse unconditionally; we
  switch to right-drag-to-look so the cursor is visible for picking.)
- **Left-click (press edge), not flying:** convert cursor px → NDC → `ScreenRayThroughCamera`.
  First test the ray against the *current selection's* gizmo handles via `PickGizmoAxis`;
  if an axis is hit, begin a **drag** on that axis. Otherwise `PickNearest` over the live
  entities' world AABBs → set `Selection.index` (or clear on a miss).
- **Left-drag (button held) with an axis grabbed:** each frame compute `prevRay` (last
  frame's cursor ray) and `curRay` (this frame's cursor ray) and apply
  `ApplyDrag(transform, mode, axis, prevRay, curRay)` to the selected entity's
  `TransformC.t`. prevRay is seeded to curRay on the press frame so the first frame is a
  no-op (no jump).
- **Left-release:** end the drag (clear the grabbed axis).
- The world AABB for each entity is `MeshBounds` (object-space, from `mesh->bounds()`)
  transformed by the entity's `Transform::Matrix()` and re-fit to an axis-aligned box
  (8-corner transform → min/max). Centralized in a small `WorldAabb(transform, bounds)`
  helper local to the sample.
- The selected object's gizmo is emitted via `editor::EmitGizmo` into a `debug::DebugDraw`,
  uploaded as one LINE_LIST vertex buffer, and drawn in a debug-line pass appended to the
  fly loop's render graph (depthTest on / write off, after opaque geometry — the same
  draw `--gizmo-shot` uses). The hovered/grabbed axis is brightened.
- Ctrl+S still serializes the live registry via `DumpScene` (unchanged behavior, now
  acting on the same live registry the gizmo edits).

Gizmo line rendering in the fly loop is NEW (Slice AA's fly loop had the editor *controls*
but no gizmo *render*; only the headless `--gizmo-shot` rendered the gizmo). It is added as
a new debug-line pipeline + per-frame line-buffer upload, isolated from the bloom passes.

### 3. Hot-reload (engine/runtime/hot_reload.{h,cpp}, hf_core, pure C++)

A `FileWatcher` polls a list of `{path, lastMtime}` entries and returns which changed since
the previous `Poll()`. To stay **unit-testable without real files**, the mtime source is an
injectable function `std::function<int64_t(const std::string&)>` (a "stat" callback
returning a monotonic mtime; <0 = missing). The default constructor uses a real
`std::filesystem::last_write_time`-based stat so the live loop watches real files; tests
inject a fake stat backed by an in-memory map.

API:
```
class FileWatcher {
  explicit FileWatcher(StatFn stat = DefaultStat());
  void Watch(std::string path);          // start tracking (records current mtime)
  std::vector<std::string> Poll();        // paths whose mtime increased since last Poll
};
```
First `Poll()` after `Watch` reports nothing changed (baseline captured at Watch time).
A path that newly appears (was missing, now present) counts as changed. A path that
disappears does not (best-effort; the loop logs and skips).

**Wiring into `--fly`:**
- Watch the active scene JSON. On change: re-`LoadScene` into a fresh registry and swap it
  in (clearing the live selection if the index no longer exists). Robust: a parse failure
  is caught, logged, and the old registry is kept.
- Watch the compiled shader `.spv` outputs in `${SHADER_OUT}`. On change of a `.spv` we
  reload that file's words and recreate its `IShaderModule`. **Scope note:** recreating the
  *pipeline* that references the module is heavier (it needs the full `GraphicsPipelineDesc`
  rebuild). For Slice AM we implement robust **scene** hot-reload end-to-end and, for
  shaders, detect+log the changed `.spv` and recreate the shader module for the lit/debug
  pipelines that are cheap to rebuild; a full hot-swap of every pipeline is deferred. The
  watcher + change detection (the reusable, testable part) is complete; the live shader
  application is best-effort and clearly logged.

## Testing (tests/live_editor_test.cpp, hf_core / ASan)

Pure C++, links `hf_core`, registered in `tests/CMakeLists.txt` via `hf_add_pure_test`:

(a) **cursor-px → NDC → ray** hits a known world point: take a world point in front of a
    camera, project to NDC via `ViewProj`, convert that NDC back to pixel coordinates with
    the same px↔NDC map the sample uses, then run px→NDC→`ScreenRayThroughCamera` and assert
    the ray passes within tolerance of the world point. (Round-trips the pixel mapping.)
(b) **click ray picks the expected entity** among a couple of world AABBs: aim the cursor at
    one box's projected center and assert `PickNearest` returns that box's index.
(c) **drag along an axis** via `ApplyDrag` moves the transform by the expected delta (reuse
    gizmo math — two parallel rays offset along the axis → that axis component changes).
(d) **FileWatcher.Poll** with an injected fake stat: a watched path reports unchanged on the
    first poll, reports changed exactly once after its mtime increases, then unchanged again;
    a second path is independent; a newly-appearing path counts as changed.

The px↔NDC mapping is shared between the sample and the test via a tiny free function
`editor::PixelToNdc(px, py, width, height)` added to `picking.h` (pure math, no new deps),
so the test exercises the *same* conversion the live loop uses.

## Verification honesty

- **Unit-tested (headless, deterministic):** cursor px→NDC→ray, pick selection, drag delta
  (prevRay/curRay), FileWatcher change detection.
- **Manual only (Windows/Vulkan GUI):** the actual live mouse-click pick + mouse-drag of a
  gizmo handle in the on-screen window. This cannot be golden-tested (it is interactive
  input). I will BUILD `--fly` and LAUNCH it briefly to confirm it opens + loops + exits
  without crashing — but I do NOT claim to have verified live mouse interaction by hand.
- `--fly-dry-run` (extended) headlessly exercises one synthetic pick + drag + scene-reload
  iteration with a fabricated `InputState`, proving the *logic* path the live loop runs.

## Seam audit

No new `vk*` / `MTL*` / `Backend::Metal` / `mtl::` symbols in the protected dirs. `hot_reload`
lives in hf_core (pure C++ + `<filesystem>` + `<functional>` only). The cursor-position
addition touches `engine/hal/window.cpp` (the sanctioned SDL site) and the SDL-free
`runtime::InputState` struct. The sample (`samples/`) is outside the protected seam.

## Optional `--edit-shot`

**SKIPPED.** Live interaction is not golden-able, and a synthetic edit screenshot would add a
new image golden plus a Metal mirror for marginal regression value over the headless
`--fly-dry-run` exercise, which already runs a full pick + axis-drag + scene-reload iteration
with synthetic `InputState`. The deterministic logic (cursor→ray, pick, drag delta,
file-watch) is covered by `tests/live_editor_test.cpp`. The 22 image goldens + 1 JSON golden
stay byte-untouched.
