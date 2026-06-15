# Slice AB — Editor interaction: selection + gizmos (design spec)

Date: 2026-06-14
Branch: `slice-editor-gizmos`
Track: Editor #2 (ray-cast object selection, transform gizmos, live edit→save, play/pause/step)

## Goal

Make scene objects selectable and editable from the editor viewport. The *logic*
(ray-cast picking, gizmo axis hit-testing, drag math, save-back, play/pause/step) is
pure C++ in `engine/editor/` compiled into `hf_core`, so it is fully headless-testable
even though the live mouse-drag manipulation itself is interactive/manual. Gizmos are
drawn through the existing Slice-W debug-line layer (`debug::DebugDraw`).

## Architecture / hard rules

- RHI seam invariant: NO new `vk*`/`MTL`/`Metal` literal tokens in the protected dirs.
  Baseline on master for the spec's grep is **16** matches (the spec text said "12";
  the true current baseline is 16 — all in comments/enum identifiers). We add **zero**
  new matches. The new editor logic is pure math/scene/runtime/debug; SDL stays in hal/.
- New logic lives in `engine/editor/picking.{h,cpp}` and `engine/editor/gizmo.{h,cpp}`,
  plus a tiny `engine/runtime/play_state.h`. All compiled into `hf_core` (ASan-scoped),
  depending only on math/scene/runtime/debug. They pull in NO rhi/backend symbols.
- `engine/editor/` previously held only the ImGui shell (`hf_editor`, Windows/Vulkan-only,
  links ImGui). The new pure files are added to `hf_core` instead, NOT to `hf_editor`, so
  they are sanitized + Mac-safe.

## Components

### 1. Math: `Mat4::Inverse` + ray types (`engine/math/math.h`)

- `Mat4 Mat4::Inverse() const` — general 4×4 inverse via cofactor expansion (column-major).
  Returns identity for a singular matrix (det≈0). Unit-tested `M * M.Inverse() ≈ I`.
- `struct Ray { Vec3 origin; Vec3 dir; }` (dir normalized by constructor helper `MakeRay`).
- `struct Aabb { Vec3 min, max; }`.
- `bool RayAabb(const Ray&, const Aabb&, float& tHit)` — slab test; `tHit` = nearest
  non-negative entry t. Returns false on miss / behind origin.
- `bool RaySphere(const Ray&, Vec3 center, float radius, float& tHit)` — quadratic.
- `float RayClosestParamToSegment(const Ray&, Vec3 a, Vec3 b, float& outDist)` — closest
  approach between the ray and a segment; used for gizmo axis-handle hit-testing and the
  translate-drag projection. Returns the parameter along the segment (0..1, clamped) and
  the world-space distance between the two closest points.

### 2. Picking (`engine/editor/picking.{h,cpp}`)

- `Ray ScreenRayThroughCamera(const runtime::Camera& cam, float ndcX, float ndcY)` —
  unproject NDC (x,y ∈ [-1,1]) using `(cam.Proj()*cam.View()).Inverse()`. Unproject the
  near point (z=0, Vulkan depth) and far point (z=1) to world space, ray = near→far. The
  camera's Proj bakes the Vulkan Y-flip, so passing NDC straight through inverse-VP is
  self-consistent (a point projected with VP unprojects back through VP⁻¹).
- `struct PickResult { int index = -1; float t = 0; };`
- `struct PickAabb { Aabb box; };` thin span element.
- `PickResult PickNearest(const Ray&, std::span<const PickAabb>)` — nearest ray-AABB hit;
  `index<0` on miss. Deterministic, window-free.

### 3. Gizmo (`engine/editor/gizmo.{h,cpp}`)

- `enum class GizmoMode { Translate, Rotate, Scale };`
- `struct Selection { int index = -1; GizmoMode mode = Translate; bool Has() const; };`
- `void EmitGizmo(debug::DebugDraw&, const scene::Transform& objXform, GizmoMode,
  float handleLen, int activeAxis=-1)` — draws at the object's *position* (rotation/scale
  of the object are intentionally ignored so the gizmo is world-axis-aligned, the common
  editor default):
  - Translate: 3 axis arrows (X red / Y green / Z blue) with arrowheads.
  - Rotate: 3 axis-aligned wire circles (radius = handleLen) in YZ/XZ/XY planes.
  - Scale: 3 axis lines with a small end box on each.
  The `activeAxis` (0=X,1=Y,2=Z) handle is brightened.
- `int PickGizmoAxis(const Ray&, const scene::Transform&, GizmoMode, float handleLen)` —
  which axis handle (0/1/2) the ray hits within tolerance, or -1. Translate/Scale test the
  ray vs each axis segment; Rotate tests the ray vs each axis circle (distance from the
  ray's plane-crossing to the circle radius).
- `scene::Transform ApplyDrag(const scene::Transform&, GizmoMode, int axis,
  const Ray& prev, const Ray& cur)` — pure drag math:
  - Translate: delta = (projection of cur onto axis) − (projection of prev onto axis),
    applied to position along that world axis.
  - Rotate: signed angle delta about the axis between prev/cur hit directions in the axis's
    plane, added to `eulerRadians[axis]`.
  - Scale: axis-length delta (cur vs prev closest-point param along the axis) added to
    `scale[axis]`, clamped ≥ small positive epsilon.

### 4. Play / pause / step (`engine/runtime/play_state.h`)

- `enum class RunState { Playing, Paused };`
- `class PlayState`: holds RunState + a one-shot `stepRequested_`. `StepsThisTick(int
  fixedSteps)` returns: Playing → fixedSteps; Paused → 0, unless a step was requested →
  exactly 1 (consumes the request). `Play/Pause/Toggle/RequestStep` mutators. Trivially
  unit-testable (paused → 0; one Step → 1; Playing passes through).

### 5. Live edit → save (interactive `--fly`, manual)

In the `--fly` loop: left-click casts `ScreenRayThroughCamera` through the cursor NDC and
`PickNearest` over the scene objects' world AABBs → sets `Selection`. While an axis is
grabbed, `ApplyDrag(prevRay, curRay)` edits the live `Transform`. `Ctrl+S` writes the edited
scene back via `scene::DumpScene` (+ the existing `commands` `save_scene` op). The gizmo is
emitted each frame through the debug-line pipeline added to the fly path. (Live drag is
manual; the math underneath is unit-tested + golden-verified.)

### 6. Headless verification entries (proof despite manual live-drag)

- `hello_triangle.exe --gizmo-shot <objIndex> <out.bmp>` — build a small deterministic
  multi-object scene (ground + cube + sphere + tall box), select `objIndex`, render the
  scene + that object's translate gizmo through the debug-line layer from a fixed camera,
  capture to BMP. New golden `tests/golden/metal/gizmo.png` (Mac). Self-contained LDR path
  modeled on `--debug-shot` so it never perturbs existing goldens.
- `hello_triangle.exe --pick-test` — headless: fixed camera + known scene, cast a ray
  through a scripted screen point, print which object index is picked. Also asserts the
  scripted select→translate→DumpScene round-trip moved the saved transform.

### 7. Unit test `tests/editor_test.cpp` (hf_core/ASan)

Covers: `Mat4::Inverse` (M·inv≈I, incl. a non-trivial TRS matrix); `ScreenRayThroughCamera`
hits a known world point (project a world point with VP → NDC → ray passes within tolerance
of the point); `PickNearest` picks the nearer of two AABBs + misses a non-intersecting ray;
`PickGizmoAxis` selects the correct axis for translate/scale/rotate; `ApplyDrag`
translate/rotate/scale produce the expected deltas; `PlayState` paused→0 / step→1 /
playing→passthrough; and a select→translate→DumpScene round-trip asserts the moved position
appears in the JSON.

### 8. Metal

`metal_headless/visual_test.mm`: add `--gizmo <objIndex> <out.png>` mirroring `--gizmo-shot`,
producing `tests/golden/metal/gizmo.png` (two-run DIFF 0.0000). 14 existing goldens stay
DIFF 0.0000. If the Mac is unreachable, implement the code and note "Metal golden pending".

## Non-goals / deviations

- The gizmo is world-axis-aligned (ignores object rotation) — standard editor "global" mode;
  "local" mode is out of scope.
- Live mouse-drag is manual-only by nature; everything testable is unit-tested + golden.
- Spec's stated seam baseline "12" is stale; the real grep baseline is 16. Invariant honored
  is "add zero new matches", verified == 16 after the slice.
