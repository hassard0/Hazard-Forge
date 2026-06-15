# Slice AQ — Frustum Culling (Phase 3 perf #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. Phase 3 (performance) opener.

**Goal:** Skip draw calls for objects entirely outside the camera frustum. Extract the 6 frustum planes
from the view-projection matrix, test each renderable's world-space bounding volume, and cull those fully
outside. This is a **render-invariant** optimization: the final image of any normal scene is UNCHANGED
(culled objects were off-screen), so every existing image golden + the introspect JSON stay byte-identical.
The proof of work is (a) a unit-tested pure-CPU frustum, (b) the cull partition matching a brute-force
reference, and (c) a NEW debug-visualization golden that renders the frustum + per-object bounds from an
overview camera so culling is visibly correct.

## Correctness contract (the crux)

Frustum culling must NEVER remove an object that contributes a visible pixel. Therefore:
- All 22 existing image goldens + `taa.png` (23 image goldens) MUST remain **byte-identical** — culling
  the default/showcase scenes removes nothing on-screen, so pixels don't move.
- `tests/golden/introspect/default_scene.json` UNCHANGED (introspect describes the scene, not the draw
  list).
- Use a **conservative** test (bounding SPHERE first; an object is culled only if its sphere is fully
  outside at least one plane). Conservative = may keep some off-screen objects (correct, just less
  optimal) but NEVER drops a visible one. Document that we accept false-keeps, never false-culls.

## Design decisions (locked)

1. **Pure-CPU header `engine/render/frustum.h`** (header-only, NO device/backend symbols — same pattern
   as `engine/render/ssr.h`/`taa.h`). Namespace `hf::render::frustum`. Contents:
   - `struct Plane { math::Vec3 n; float d; };` (normalized; `signedDistance(p) = dot(n,p)+d`).
   - `struct Frustum { Plane planes[6]; };`
   - `Frustum FromViewProj(const math::Mat4& viewProj)` — **Gribb-Hartmann** plane extraction (rows of
     the combined matrix: left=row3+row0, right=row3-row0, bottom=row3+row1, top=row3-row1,
     near=row3+row2, far=row3-row2, then normalize each by its xyz length). Document the matrix
     storage/row convention used by `engine/math` so the extraction matches (row-major vs column-major:
     inspect `math::Mat4` and the existing view-proj build in `main.cpp`/the render path and match it
     EXACTLY — a transpose bug here silently culls visible geometry).
   - `bool SphereOutside(const Frustum&, const math::Vec3& center, float radius)` — true iff the sphere
     is fully outside any plane (`signedDistance < -radius` for any plane). Cull when true.
   - `bool AabbOutside(const Frustum&, const math::Vec3& min, const math::Vec3& max)` — the
     "p-vertex / n-vertex" positive-vertex AABB test (optional second test for tighter culling; include
     it and unit-test it, but the renderer may use the sphere test for simplicity — document which the
     render path uses).
   - The plane convention (inside = positive signed distance) must be consistent between FromViewProj
     and the Sphere/Aabb tests; the unit test pins this with hand-checked cases.

2. **Per-mesh local bounds (engine side, additive).** Add a world-space bounding SPHERE per renderable.
   `scene::Mesh` (or the existing mesh/bounds struct) gains a local bounding sphere
   (`math::Vec3 center; float radius`) computed at mesh-build time from its vertex positions (min/max →
   center + max-distance radius). The `Mesh::Cube/Plane/Sphere` factories and the glTF/skinned loaders
   set it. Transform to world space per frame: `worldCenter = model * localCenter`,
   `worldRadius = localRadius * maxAbsScale(model)` (uniform-scale-safe; for the engine's scenes scale is
   ~uniform — document the conservative max-scale choice). If adding a field to `scene::Mesh` is invasive,
   compute bounds on the fly from the existing vertex data — pick the cleaner path and document it.

3. **Cull in the render submission path.** Where the renderer iterates renderables and records draws
   (inspect the scene-draw loop in the render graph / main render pass), build the camera frustum once
   per frame from the (UNJITTERED) view-proj, test each renderable's world sphere, and SKIP recording the
   draw when `SphereOutside`. Keep a per-frame stat `{drawn, culled}`. IMPORTANT: cull against the
   unjittered frustum (TAA jitter is sub-pixel — never cull on the jittered matrix or edge objects could
   pop between accumulation frames). Shadow/depth passes may keep their own (light) frustum or stay
   un-culled for now (document; YAGNI — main-view culling is this slice).

4. **Draw-stat exposure (no golden churn).** Expose `{drawn, culled, total}` via a stdout line on the
   showcases and/or a `cull` command/flag — NOT in the introspect JSON golden (keep it byte-stable). A
   new test asserts the partition.

5. **Debug-visualization showcase + NEW golden.** `--cull-shot <out>` renders the scene from a pulled-back
   **overview camera** that sees the actual (narrower) render camera's frustum drawn as debug LINES (reuse
   the Slice-W debug-line layer) plus each object's bounding sphere/box: **green** if the render camera
   keeps it, **red** if culled. Deterministic (fixed overview camera, fixed scene, fixed render camera).
   This makes culling visible and is the slice's golden. Mirror it in `metal_headless/visual_test.mm` as a
   `--cull` showcase; two Metal runs DIFF 0.0000; new golden `tests/golden/metal/cull.png`. Add
   `--cull-shot` to introspect `showcases` AND `frustum-culling` to `features` — meaning the introspect
   JSON IS intentionally rebaked with exactly those two additions (same pattern as TAA), documented in the
   gate. (The image goldens for normal scenes stay byte-identical; only this NEW cull.png is added.)

## RHI seam additions (summary)
- **None.** Frustum math is pure CPU; culling skips draws in the existing submission path; bounds ride
  existing mesh/scene structs; the debug viz reuses the existing debug-line layer + RT/post infra. NO
  `vk*`/`MTL*`/`Backend::Metal`/`mtl::` symbols above the backend dirs. New files
  (`engine/render/frustum.h`, `tests/frustum_test.cpp`) add ZERO backend types. Seam grep stays at the
  benign baseline (3).

## Out of scope (YAGNI)
GPU-driven culling / compute indirect draw, hierarchical Z occlusion culling, BVH/octree spatial
acceleration, per-cascade shadow-frustum culling, portal/cell culling. One main-view CPU frustum,
conservative bounding-sphere test, render-invariant, debug-viz golden. (GPU culling is the NEXT Phase 3
slice.)

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 23) + new `frustum_test`: Gribb-Hartmann
   extraction on a known proj (hand-checked planes), SphereOutside/Inside hand cases, AabbOutside p/n
   vertex cases, and a **cull-partition vs brute-force** check (random points/spheres tested against the
   6 planes vs an independent reference). `frustum_test` clean under `windows-msvc-asan`.
2. **Render-invariance proof:** `git diff master --stat -- tests/golden/metal` shows ONLY `cull.png`
   added — all 23 existing image goldens byte-identical (this is the headline correctness result: culling
   changed no visible pixel).
3. `--cull-shot` on Windows/Vulkan: controller visual review — frustum lines + green(kept)/red(culled)
   bounds look correct (objects outside the drawn frustum are red, inside are green); the drawn/culled
   stat line is printed and sane.
4. Metal: `visual_test --cull` → new golden `tests/golden/metal/cull.png`; two runs DIFF 0.0000.
5. Introspect JSON rebaked with exactly `frustum-culling` (features) + `--cull-shot` (showcases); no other
   JSON drift; introspect test updated.
6. Seam grep clean (3 benign). `scripts/verify.ps1` updated to include the new `cull` image golden in the
   Mac round-trip loop.
