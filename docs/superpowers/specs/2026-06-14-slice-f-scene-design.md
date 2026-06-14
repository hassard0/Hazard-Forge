# Hazard Forge — Slice F: Scene Layer + Multi-Object Lit Scene

**Date:** 2026-06-14
**Status:** Self-approved (autonomous session)
**Branch:** `slice-f-scene`

## Goal

Introduce the engine's first abstraction layer **above the raw RHI** — a small scene/mesh model —
and use it to render a real **multi-object lit scene**: a ground plane plus a grid of cubes, all
textured and lit. This is the step from "a renderer that draws one cube" to "an engine that draws
a scene."

**Definition of Done:** `hello_triangle.exe --shot scene.bmp` renders a ground plane + a 3×3 grid
of cubes (varied heights/rotations) under the directional light, each casting clear shading; the
sample's draw loop iterates a list of `Renderable`s rather than hardcoding one cube. Headless-
capture-verified (controller views the BMP), zero validation errors, ctest green.

## Bold decisions (architect-of-record)

1. **Start the engine layer now (`engine/scene/`).** Everything above the RHI has lived inline in
   the sample. Introduce `Transform`, `Mesh`, and `Renderable` as real engine types. This is the
   spine of the eventual gameplay/scene system and the right time to lay it down cleanly.
2. **Mesh owns its GPU buffers; factories build primitives.** `Mesh::Cube(device)` /
   `Mesh::Plane(device)` create the vertex+index buffers. The shared vertex format
   (`pos,color,uv,normal`) moves out of the sample into `engine/scene/vertex.h` as the engine's
   canonical mesh vertex — one definition, used by mesh factories and the pipeline's vertex layout.
3. **No new RHI surface.** Multi-object rendering = iterate renderables, push each model matrix,
   bind its texture, draw. The RHI already supports this. Bold = resist adding instancing/indirect
   draw before a perf reason exists (a flat draw-per-object loop is correct and clear for this scale).
4. **Transform = TRS.** `Transform{position, eulerRadians, scale}` → `Translate * Rz*Ry*Rx *
   Scale`. Needs `Mat4::Scale` (+ `RotateZ`) added to the math lib. Uniform scale keeps the
   `(float3x3)model` normal path valid; the plane uses uniform scale, cubes too.

## New engine module: `engine/scene/`

- `vertex.h` — `struct Vertex { float pos[3]; float color[3]; float uv[2]; float normal[3]; };`
  (stride 44) + a helper returning the matching `rhi::VertexLayout`.
- `transform.h` — `struct Transform { math::Vec3 position{}, eulerRadians{}, scale{1,1,1};
  math::Mat4 Matrix() const; };`
- `mesh.h/.cpp` — `class Mesh { std::unique_ptr<rhi::IBuffer> vertices, indices; uint32_t
  indexCount; }`, with static factories `Mesh::Cube(rhi::IRHIDevice&)` and
  `Mesh::Plane(rhi::IRHIDevice&)` (a large flat quad on the XZ plane, normal +Y). Cube = the
  existing 24-vertex/36-index data moved here.
- `renderable.h` — `struct Renderable { Mesh* mesh; rhi::ITexture* texture; Transform transform; };`

The scene layer depends only on `rhi/` and `math/` — never on Vulkan. (The hard rule extends:
`engine/scene/` has no `vk*` either.)

## Math additions (`engine/math/math.h`)
- `Mat4::Scale(Vec3 s)` — diagonal scale.
- `Mat4::RotateZ(float rad)`.

## Sample (`samples/hello_triangle/main.cpp`)

- Build one shared checkerboard texture and two meshes (`Mesh::Cube`, `Mesh::Plane`).
- Build a `std::vector<Renderable>`: one plane (large, y=0), and a 3×3 grid of cubes at varied
  positions/heights/rotations above it (e.g. spacing 1.6, small per-cube y offset + yaw).
- Camera positioned to see the scene (e.g. eye {4,3.5,6} looking at {0,0.5,0}); directional light
  unchanged. Per frame (or per --shot frame at fixed t): set frame uniforms once, then for each
  renderable: `PushConstants(&model,64); BindTexture(*r.texture); BindVertexBuffer(mesh.vertices);
  BindIndexBuffer(mesh.indices); DrawIndexed(mesh.indexCount);`.
- Interactive loop spins the grid (rotate each cube and/or orbit camera); `--shot` uses fixed t.
- Title "Hazard Forge — Scene".

## Testing
- `math_test` extended: assert `Mat4::Scale` diagonal and that `RotateZ(0)`/`Scale(1,1,1)` are
  identity; keep existing checks. Still headless.
- `rhi_smoke` unchanged.
- New optional headless gate stays manual: controller runs `--shot scene.bmp` and views it.

## Out of scope
Scene graph / parent-child hierarchy, frustum culling, instancing/indirect draw, material system
beyond one texture, multiple lights, model/glTF loading (next slice). Flat list of renderables,
two primitive meshes, one light, one texture.
