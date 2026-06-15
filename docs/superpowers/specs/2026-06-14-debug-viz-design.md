# Slice W — Debug Visualization Layer / Immediate-Mode Debug Draw

Date: 2026-06-14
Branch: `slice-debug-viz`

## Goal

An immediate-mode debug-draw subsystem the engine (and an *agentic* developer) can drive
with simple data/API calls and capture headlessly. It overlays crisp colored 3D line
primitives — grids, AABB/OBB wireframes, wire spheres, gizmos, vertex normals, light
arrows, physics contacts — on top of the shaded scene, occluded correctly by opaque
geometry so the overlay reads as 3D.

Design lens: **headless, scriptable, machine-observable**. The collector is pure C++
(math + stdlib), deterministic, and unit-tested; the GPU side is a single small line
pipeline driven by a per-frame CPU-built vertex buffer (the ImGui dynamic-geometry pattern).

## Decisions

### 1. RHI line topology (`GraphicsPipelineDesc.lineList`)

- New `bool lineList = false;` field on `GraphicsPipelineDesc`. Default false →
  every existing pipeline is byte-for-byte unchanged (the field is never read unless set).
- **Vulkan** (`vulkan_pipeline.cpp`): input-assembly topology becomes
  `VK_PRIMITIVE_TOPOLOGY_LINE_LIST` when `lineList`, else the existing point/triangle
  selection. Topology is fixed at pipeline creation; the draw call needs no change.
- **Metal** (`metal_pipeline.mm` / `metal_command_buffer.mm`): Metal selects the
  primitive type at *draw* time, so `MetalPipeline` records a `lineList()` flag,
  `BindPipeline` latches it into `boundLineList_`, and `Draw()` issues
  `MTLPrimitiveTypeLine`. (Debug lines use non-indexed `Draw`, so `DrawIndexed`
  is untouched.)
- `pointList` and `lineList` are mutually exclusive; if both set, line wins (documented;
  no engine pipeline sets both).

### 2. Immediate-mode collector `engine/debug/debug_draw.{h,cpp}` (hf_core)

Pure C++; depends only on `math/math.h` + stdlib. NO RHI/backend symbols, so it compiles
into the ASan-scoped `hf_core` and is unit-tested.

```
struct LineVertex { float pos[3]; float color[3]; };   // matches debug_line.vert layout

class DebugDraw {
  void Clear();
  void Line(Vec3 a, Vec3 b, Vec3 color);
  void Ray(Vec3 origin, Vec3 dir, float len, Vec3 color);
  void Box(Vec3 min, Vec3 max, Vec3 color);             // AABB as 12 edges (24 verts)
  void Obb(const Mat4& transform, Vec3 halfExtents, Vec3 color);
  void WireSphere(Vec3 center, float radius, Vec3 color, int segments=16);
  void Grid(float halfSize, float step, Vec3 color);
  void Axes(const Mat4& transform, float len);          // X=red, Y=green, Z=blue
  const std::vector<LineVertex>& Vertices() const;
};
```

All shapes decompose to line segments appended in **deterministic order** to one vertex
list. Each segment = 2 `LineVertex`. Counts (locked by the unit test):

- `Line`/each grid line/each box edge = 2 verts.
- `Box` = 12 edges = **24 verts**.
- `Obb` = 12 edges (transformed corners) = 24 verts.
- `WireSphere(segments=S)` = 3 orthogonal circles, each S segments = `3*S*2` verts
  (S=16 → 96).
- `Grid(halfSize=H, step=s)`: lines per axis = `2*floor(H/s)+1`; total lines =
  `2*(2*floor(H/s)+1)`, verts = lines*2. (H=10,s=1 → 21 lines/axis → 42 lines → 84 verts.)
- `Axes` = 3 segments = 6 verts.

#### Convenience emitters (take engine objects)

Free functions in `hf::debug` (kept separate from the core `DebugDraw` so they may pull in
`scene::`/`physics::` types without bloating the pure class):

- `MeshAabb(DebugDraw&, span<const scene::Vertex>, const Mat4& model, Vec3 color)` —
  compute the AABB from CPU vertices, transform the 8 corners by `model`, draw as a Box in
  world space (AABB recomputed from transformed corners so it hugs the posed mesh).
- `MeshNormals(DebugDraw&, span<const scene::Vertex>, const Mat4& model, float len, Vec3
  color)` — one short hair per vertex from world pos along the world normal.
- `LightArrow(DebugDraw&, Vec3 origin, Vec3 dir, float len, Vec3 color)` — an arrow
  (shaft + 4 short head fins) showing the directional-light direction.
- `PhysicsContacts(DebugDraw&, const physics::World&, Vec3 pointColor, Vec3 normalColor)` —
  recomputes sphere/ground + sphere/sphere contacts (same fixed order as the solver, read
  only) and draws a small cross at each contact point plus a normal stub. (The solver keeps
  contacts local; the emitter recomputes them so `world.cpp` stays untouched.)

To feed `MeshAabb`/`MeshNormals`, `scene::Mesh` gains an optional CPU-side `bounds()`
accessor and the engine primitives (`Cube`/`Plane`/`Sphere`) record their min/max AABB at
build time. (CPU vertices themselves are not retained — only the AABB — which is all the
overlay needs; full CPU-vert retention is noted as future work.)

### 3. Debug line pipeline + shader

- `shaders/debug_line.vert.hlsl`: input `{ float3 pos; float3 color; }` (locations 0/1);
  transforms `pos` by `FrameData.viewProj`; passes color through. Uses the same FrameData
  cbuffer convention as `lit.vert`/`shadow.vert` (set 0 binding 0 on Vulkan; HF_MSL_GEN
  binding(1,0) for the MSL path). No push constant, no model matrix (vertices are world
  space).
- `shaders/debug_line.frag.hlsl`: outputs the interpolated color, alpha 1.
- Pipeline: `lineList=true, usesFrameUniforms=true, usesTexture=false, depthTest=true,
  depthWrite=false`. Depth-test-on/write-off means lines are correctly **occluded** by
  opaque geometry in front (more readable as 3D) but never fight the depth buffer or
  occlude each other.
- Drawn in the scene RT pass **after** opaque geometry (and after the instanced bodies),
  **before** the post/tonemap pass. (For the showcase there is no transparency; if both
  existed, debug lines go after opaque and after transparency so the overlay is always on
  top of the shaded result but still depth-tested against opaques.)
- Registered in BOTH the Vulkan DXC list (`samples/hello_triangle/CMakeLists.txt`) and the
  Metal MSL-gen list (`metal_headless/CMakeLists.txt`).

#### Dynamic vertex buffer

Follow the ImGui pattern: accumulate `LineVertex`es CPU-side via `DebugDraw`, then
`CreateBuffer` once per frame with the line data as `initialData` (per-frame (re)creation
is acceptable for the single-frame showcase capture). A persistent dynamic ring buffer is
noted as future work.

### 4. Showcase + verification

- **Vulkan**: `hello_triangle.exe --debug-shot <path>`. Renders the physics sphere-pyramid
  scene (settled, lit + shadowed, over the checkerboard ground + sky) — the same self-
  contained setup as `--physics-shot` — PLUS a debug overlay built with `DebugDraw`:
  a ground grid, a colored wireframe AABB around each settled body, a light-direction
  arrow, and a wire sphere on each body. Captured to BMP, converted to PNG, visually
  inspected.
- **NEW** golden only. Existing goldens / pipelines / shaders / scenes are UNTOUCHED.
- **Metal**: same overlay added to `metal_headless/visual_test.mm` (`--debug` arg),
  producing the new committed golden `tests/golden/metal/debug_viz.png` (two runs diff
  0.0000). The 9 existing Metal goldens must stay diff 0.0000.

### 5. Unit test `tests/debug_draw_test.cpp`

Pure C++ / hf_core / ASan. Asserts the primitive→vertex counts above and that a `Line`'s
emitted endpoints equal its inputs. Registered in `tests/CMakeLists.txt`.

## Seam

`engine/debug/` is added to the no-backend-symbols seam. `grep -rnE "vk[A-Z]|MTL|Metal"`
over the listed engine dirs stays at the master baseline of **12** (header comments worded
to avoid the literal tokens).

## Future work

- Persistent dynamic ring buffer for the line vertex stream (avoid per-frame recreate).
- Retain full CPU vertices on `scene::Mesh` for true per-vertex normal hairs on imported
  meshes (the showcase uses engine primitives + the recorded AABB).
- Depth-tested vs. always-on-top toggle (a second no-depth debug pipeline).
