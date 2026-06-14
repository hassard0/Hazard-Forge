# Slice F: Scene Layer + Multi-Object Lit Scene — Implementation Plan

> subagent-driven. Toolchain proven. Env facts below.

**Goal:** Introduce `engine/scene/` (Vertex, Transform, Mesh, Renderable) and render a ground
plane + 3×3 grid of lit textured cubes by iterating a list of renderables. Headless-verified via
`--shot`.

**Builds on:** Slices A–E on `master`. Additive (refactors the sample's inline cube into the scene layer).

---

## Environment (proven — trust)
- VS BuildTools x64 dev shell: `& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64`, then cmake in that session.
- Conan installed. `cmake --preset windows-msvc-debug` (reconfigure after adding source files) then `cmake --build --preset windows-msvc-debug`. Add new engine/scene/*.cpp to the hf_engine source list in engine/CMakeLists.txt.
- `hello_triangle.exe --shot <path>.bmp` renders ONE frame and EXITS — you MAY run it to self-verify (exit 0 + non-empty BMP). Never run the no-arg interactive loop (infinite).
- Gates: ctest (rhi_smoke + math_test). 
- READ first: engine/rhi/rhi.h, engine/math/math.h, samples/hello_triangle/main.cpp (the current single-cube setup — vertex struct, cube data, vertex layout, frame-uniform + draw recording, and the --shot capture path you must preserve), engine/CMakeLists.txt.
- HARD RULE extends: NO vulkan symbols in engine/rhi/ OR engine/scene/. engine/scene/ depends only on rhi/ + math/.

## Task F1: Math additions (engine/math/math.h)
- [ ] Add `static Mat4 Scale(const Vec3& s)`: identity with m[0]=s.x, m[5]=s.y, m[10]=s.z.
- [ ] Add `static Mat4 RotateZ(float rad)`: c=cos,s=sin; m[0]=c, m[4]=-s, m[1]=s, m[5]=c (column-major: element(0,0)=c, element(0,1)=-s→m[4], element(1,0)=s→m[1], element(1,1)=c).
- [ ] math_test.cpp: assert Scale({2,3,4}) has m[0]==2,m[5]==3,m[10]==4 and m[15]==1; assert RotateZ(0) equals Identity (all 16). Keep existing checks. Commit after math_test passes: `feat(math): Scale + RotateZ`.

## Task F2: Scene module (engine/scene/)
Create these; add the .cpp to engine/CMakeLists.txt hf_engine sources.

- [ ] `engine/scene/vertex.h`:
```cpp
#pragma once
#include "rhi/rhi.h"
namespace hf::scene {
struct Vertex { float pos[3]; float color[3]; float uv[2]; float normal[3]; };
inline rhi::VertexLayout MeshVertexLayout() {
    rhi::VertexLayout l;
    l.stride = sizeof(Vertex);
    l.attributes = {
        {0, rhi::Format::RGB32_Float, 0},
        {1, rhi::Format::RGB32_Float, 12},
        {2, rhi::Format::RG32_Float,  24},
        {3, rhi::Format::RGB32_Float, 32},
    };
    return l;
}
} // namespace
```
- [ ] `engine/scene/transform.h`:
```cpp
#pragma once
#include "math/math.h"
namespace hf::scene {
struct Transform {
    math::Vec3 position{0,0,0};
    math::Vec3 eulerRadians{0,0,0};  // x=pitch, y=yaw, z=roll
    math::Vec3 scale{1,1,1};
    math::Mat4 Matrix() const {
        using namespace math;
        Mat4 r = Mat4::RotateZ(eulerRadians.z) * Mat4::RotateY(eulerRadians.y) * Mat4::RotateX(eulerRadians.x);
        return Mat4::Translate(position) * r * Mat4::Scale(scale);
    }
};
} // namespace
```
- [ ] `engine/scene/mesh.h`:
```cpp
#pragma once
#include <memory>
#include <cstdint>
#include "rhi/rhi.h"
namespace hf::scene {
class Mesh {
public:
    Mesh(std::unique_ptr<rhi::IBuffer> v, std::unique_ptr<rhi::IBuffer> i, uint32_t indexCount)
        : vertices_(std::move(v)), indices_(std::move(i)), indexCount_(indexCount) {}
    rhi::IBuffer& vertices() const { return *vertices_; }
    rhi::IBuffer& indices() const { return *indices_; }
    uint32_t indexCount() const { return indexCount_; }
    static Mesh Cube(rhi::IRHIDevice& device);
    static Mesh Plane(rhi::IRHIDevice& device);
private:
    std::unique_ptr<rhi::IBuffer> vertices_, indices_;
    uint32_t indexCount_;
};
} // namespace
```
- [ ] `engine/scene/mesh.cpp`: implement Cube (move the 24-vertex/36-index cube data from the
  sample here, using `scene::Vertex`) and Plane (4 vertices forming a large quad on XZ, y=0, normal
  {0,1,0}, uv 0..N for a tiled look e.g. uv 0..4, color {1,1,1}; 6 indices CCW when viewed from
  above). Each factory creates the vertex buffer (`BufferUsage::Vertex`) and index buffer
  (`BufferUsage::Index`) via `device.CreateBuffer`, returns `Mesh{...}`.
- [ ] `engine/scene/renderable.h`:
```cpp
#pragma once
#include "scene/mesh.h"
#include "scene/transform.h"
#include "rhi/rhi.h"
namespace hf::scene {
struct Renderable { Mesh* mesh; rhi::ITexture* texture; Transform transform; };
} // namespace
```
Commit with F3 (sample) since the scene layer isn't exercised until the sample uses it.

## Task F3: Sample renders the scene
**File:** samples/hello_triangle/main.cpp. Preserve the `--shot` capture path and the BMP writer.
- [ ] Remove the inline cube Vertex struct + cube data + inline vertex layout; instead
  `#include "scene/vertex.h"`, `scene/mesh.h`, `scene/transform.h`, `scene/renderable.h`. Use
  `scene::MeshVertexLayout()` for the pipeline's `GraphicsPipelineDesc.vertexLayout`.
- [ ] Build the shared checkerboard texture (as before). Build `scene::Mesh cube = Mesh::Cube(*device);`
  and `scene::Mesh plane = Mesh::Plane(*device);`.
- [ ] Build `std::vector<scene::Renderable> scene;`:
  - plane: `{&plane, texture.get(), Transform{ position{0,0,0}, scale{6,1,6} }}`.
  - 3×3 grid of cubes: for gx in -1..1, gz in -1..1: `Transform{ position{gx*1.8f, 0.6f, gz*1.8f},
    eulerRadians{0, (gx+gz)*0.5f, 0}, scale{0.5f,0.5f,0.5f} }`, texture = the shared texture, mesh = &cube.
- [ ] Camera: `eye={4.5f,4.0f,6.5f}`, center `{0,0.5f,0}`, up `{0,1,0}`. proj Perspective(60°,w/h,0.1,100).
  Light unchanged (dir {-0.5,-1,-0.3}). Set frame uniforms (viewProj, light, viewPos) once per frame.
- [ ] Draw loop (both interactive and --shot): after `BindPipeline` (binds frame set 0), for each
  `Renderable& r : scene`: `math::Mat4 m = r.transform.Matrix(); cmd->PushConstants(&m, 64);
  cmd->BindTexture(*r.texture); cmd->BindVertexBuffer(r.mesh->vertices());
  cmd->BindIndexBuffer(r.mesh->indices()); cmd->DrawIndexed(r.mesh->indexCount());`.
- [ ] Interactive mode: animate (e.g. orbit camera or add `t` to each cube's yaw). `--shot` mode:
  fixed `t` (cubes static), capture as before. Keep the WriteBMP + capture flow intact.
- [ ] Title "Hazard Forge — Scene".

- [ ] **Build, ctest (rhi_smoke + math_test pass), run `--shot "$env:TEMP\hf_f.bmp"` → exit 0,
  BMP > 54 bytes.** Commit (F2+F3 together): `feat(scene): scene layer (Transform/Mesh/Renderable) + multi-object lit scene`

## Definition of Done
- [ ] ctest green; `--shot` produces a valid BMP and exits 0.
- [ ] hello_triangle builds clean, zero validation errors.
- [ ] No `vk*` in engine/rhi/ OR engine/scene/ (grep both).
- [ ] Controller views the BMP: ground plane + grid of lit cubes, correct shading + depth between objects.

## Risks
- Plane winding/normal: must face up (+Y) and be CCW from above so back-face culling keeps it
  visible from the camera. If the plane is invisible, flip its winding.
- Vertex layout must match scene::Vertex exactly (stride 44, normal at offset 32).
