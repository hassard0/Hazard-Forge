# Hazard Forge

A C++20 cross-platform game engine built around a thin, explicit **Rendering Hardware Interface (RHI) seam** that renders natively on **Vulkan (Windows)** and **Apple Metal (macOS, Apple Silicon)** from one codebase. The design is Apple-native and Metal-first in philosophy, with Vulkan as the primary shipping platform; Metal parity is verified headlessly against committed golden images.

> **Status:** Active development — an in-progress engine, not a finished product. The RHI-seam thesis is proven on real hardware: the same engine code renders a full PBR/IBL/post-processed scene on a Vulkan RTX GPU (Windows) and on Apple Metal (M4, macOS), verified by **20 deterministic golden-image regression tests** that must each diff `0.0000`. Roughly slices A–AH have shipped — PBR materials, image-based lighting, HDR bloom, SSAO, alpha-blended transparency, glTF scene-graph import, skeletal animation + blending, rigid-body physics, GPU instancing, compute particles, immediate-mode debug visualization, an interactive runtime (fixed-timestep loop + flyable camera), an editor layer (ray-cast selection + transform gizmos), a full shadow set (cascaded shadow maps, spot-light shadows, omnidirectional point-light cube shadows), clustered/Forward+ lighting, and screen-space reflections.

---

## What it is

Hazard Forge is a focused, tooling-obsessed C++ engine built slice by slice. Each slice lands a specific rendering or runtime capability through the shared RHI seam — which means every feature is simultaneously a Vulkan implementation and a specification for the Metal backend to implement.

The central bet: **one engine layer above a clean seam, two GPU backends below it, verified headlessly by golden images.** No Vulkan symbols leak above `engine/rhi/`; no Metal symbols leak above `engine/rhi_metal/`. The seam invariant is the architectural spine of the project.

---

## Features (shipped, slices A–AH)

### Rendering

- **RHI abstraction** — Vulkan and Metal backends behind a pure C++ virtual-interface seam (`engine/rhi/rhi.h`). No backend symbol (`vk*` / `MTL*`) is visible to engine, scene, or application code.
- **3D math library** — own `hf::math` (Vec3, Mat4, Quat): perspective/ortho/LookAt, TRS, rotations, `Inverse`, slerp. Column-major; Vulkan `[0,1]` depth; the Metal NDC +Y flip is applied CPU-side so the shared shaders are backend-neutral.
- **PBR materials** — full glTF metallic-roughness Cook-Torrance shading: base color, metallic-roughness, tangent-space normal mapping, emissive, and ambient occlusion, bound as a 5-texture material set.
- **Image-based lighting** — both a lightweight **procedural** sky-color IBL and real **HDR equirectangular** environment IBL (`assets/env/env.hdr`, CPU box-prefiltered mip chain → roughness-aware specular + diffuse irradiance).
- **Directional shadow mapping (PCF)** — depth-only pass into a 2048² shadow map; 3×3 PCF in the lit shaders; `Mat4::Ortho` light projection. Static, skinned, and instanced shadow pipelines.
- **Cascaded shadow maps (CSM)** — the directional light's frustum is split into N cascades (increasing splits over `(near, far]`), each fit to a tight ortho volume and rendered into one tile of a shadow atlas via `SetViewport`. The lit shaders select a cascade per fragment by view depth. `render/csm.h` is header-only and unit-tested (frustum split + per-cascade ortho fit).
- **Spot lights + spot shadows** — a perspective spot-light shadow (`render/spot.h`: `spotViewProj` projection + cone attenuation smoothstep), depth-rendered into the shadow atlas and PCF-sampled in the lit pass.
- **Omnidirectional point-light shadows** — a 6-face cube shadow atlas (`render/point_shadow.h`: per-face cube view-proj, dominant-axis face selection, 3×2 tile mapping) so a point light casts shadows in every direction. Rendered into six atlas tiles via `SetViewport`.
- **Clustered / Forward+ lighting** — the view frustum is divided into a 3D cluster grid (exponential z-slices); each cluster gets the list of lights whose sphere overlaps it (`render/clustered.h`: `BuildClusters` → per-cluster offset/count + a flat light-index list). The lit pass reads the cluster set (set 3, storage buffers) and shades only the lights touching its cluster, scaling to **hundreds** of lights. Proven **byte-identical to brute-force** shading by golden (`clustered`, 192 lights).
- **Screen-space reflections (SSR)** — a depth-marched SSR pass (`render/ssr.h`: view↔screen projection round-trip + view-space reflection ray) reflects the scene off a reflective floor by ray-marching the depth buffer, with the Vulkan/Metal yFlip handled in the projection.
- **Multi-pass post-processing** — fullscreen pass with FXAA, exposure, ACES tonemap, cinematic grade, film grain, and vignette.
- **HDR bloom** — render into an `RGBA16_Float` target, soft-knee threshold bright-pass, a 5-level downsample mip chain (13-tap dual filter), tent-filter upsample/combine, and a composite that tonemaps.
- **SSAO** — view-space normal+depth g-buffer prepass, a 16-sample hemisphere-kernel AO pass (baked kernel + noise, no runtime RNG), a box blur, and a composite that multiplies the lit scene by the blurred AO (contact AO).
- **Alpha-blended transparency** — a sorted (back-to-front) translucent pass that depth-tests against opaque geometry but does not write depth (`depthWrite=false`), with Fresnel-style alpha.
- **GPU instancing** — a per-instance vertex stream (`scene::InstanceTransformLayout`) drives a single `DrawIndexedInstanced` call; transforms come from the deterministic `scene::BuildInstanceGrid` (no RNG).
- **Compute particles** — a compute kernel animates a 50k-particle storage buffer (gravity + swirl fountain, deterministic respawn), drawn as additive points.
- **Procedural skybox** — gradient sky + sun disk drawn first in the scene pass via camera-ray reconstruction from the per-frame UBO. No matrix inverse, no extra textures.
- **Render graph** — `render::RenderGraph` declares passes with imported targets (shadow map / scene color / swapchain) and executes them in order through the RHI.

### Assets, animation, physics

- **glTF loading** — header-only `cgltf`. Single-model loaders (geometry + full PBR material set) **and** a full **scene-graph import** (`LoadGltfScene`: node hierarchy walked to world transforms, one renderable per primitive, deduped materials).
- **Skeletal animation + blending** — GPU skinning (joint palette at a dedicated vertex binding); single-clip sampling and **cross-clip blending** (`BlendAnimations`: lerp T/S, slerp R per joint) on the CPU in `hf::anim`.
- **Rigid-body physics** — impulse-based `physics::World` (spheres + ground, contact resolution), deterministic fixed-step settling.

### Runtime, editor, tooling

- **Interactive runtime** — a fixed-timestep loop (`runtime::Clock`/`FixedTimestep`), a backend-agnostic flyable `runtime::Camera` (yaw/pitch → basis, `View`/`Proj`/`ViewProj`), and a `FlyCameraController`. The live windowed viewport (WASD + mouse-look) is **Vulkan/Windows** (`--fly`); the camera math itself is golden-verified on both backends.
- **Editor layer** — ray-cast **selection** (`editor::picking`: `ScreenRayThroughCamera`, `PickNearest`), transform **gizmos** (`editor::gizmo`: translate/rotate/scale axis math + `ApplyDrag`), and play/pause/step. The selection/gizmo math is pure C++ and unit-tested; live mouse-drag manipulation is Windows/Vulkan `--fly` only and is manual — the **goldens + unit tests prove the math under it**.
- **Immediate-mode debug visualization** — `debug::DebugDraw` collects grids / AABBs / wire spheres / arrows / contact markers into a LINE_LIST and draws them in one call through a debug-line pipeline (`depthTest=true, depthWrite=false`).
- **Headless GPU capture** — `CaptureNextFrame()` / `GetCapturedPixels()`: render a frame, read pixels back from the GPU, write a PNG/BMP. No visible desktop required — the primary verification path.
- **Cross-platform verification** — `scripts/verify.ps1` runs the Windows/Vulkan ctest **and** drives the bench Mac over SSH to build `metal_headless` and golden-compare all 20 Metal goldens at `DIFF 0.0000`, printing a per-golden table and an overall `VERIFY: PASS/FAIL`.
- **AddressSanitizer** — an opt-in `HF_SANITIZE=address` build (`windows-msvc-asan` preset) instruments the backend-agnostic core (`hf_core`) and the pure-C++ unit tests so they run clean under MSVC `/fsanitize=address`.

**Metal parity status:** The Metal backend renders **every** showcase headless on Apple Silicon (M4) and is verified against a committed golden at `DIFF 0.0000` for all 20 scenes (see below). The Metal shaders are **generated from the shared HLSL** at build time (HLSL → SPIR-V via `glslc` → MSL via `spirv-cross`), so there is no hand-written MSL to drift. The remaining gap is intentional and documented: the live **windowed** present loop (interactive viewport + mouse-drag editing) is Vulkan/Windows only; Metal is headless-offscreen-verified.

---

## The 20 Metal goldens

Each is produced by a distinct `metal_headless/visual_test` flag and compared against `tests/golden/metal/<name>.png` at threshold `0.0` (deterministic — two runs diff `0.0000`):

| golden            | flag                          | what it proves                                            |
| ----------------- | ----------------------------- | --------------------------------------------------------- |
| `scene_shadow`    | *(default)*                   | lit scene + PCF shadows + procedural IBL + post (Slice F) |
| `skinning`        | `--skinning`                  | GPU skeletal animation (Slice O)                          |
| `pbr_helmet`      | `--pbr`                       | full glTF metallic-roughness PBR (Slice P)                |
| `instanced`       | `--instanced`                 | GPU instancing, 144 spheres in one call (Slice Q)         |
| `ibl_helmet`      | `--ibl`                       | HDR equirect environment IBL (Slice R)                    |
| `physics`         | `--physics`                   | settled rigid-body sphere pyramid (Slice S)               |
| `transparency`    | `--transparency`              | sorted alpha-blended glass (Slice T)                      |
| `bloom`           | `--bloom`                     | HDR bloom chain (Slice U)                                 |
| `scene_import`    | `--scene`                     | full glTF scene-graph import (Slice V)                    |
| `debug_viz`       | `--debug`                     | immediate-mode debug-line overlay (Slice W)               |
| `anim_blend`      | `--blend`                     | cross-clip animation blending (Slice X)                   |
| `ssao`            | `--ssao`                      | screen-space ambient occlusion (Slice Y)                  |
| `capstone`        | `--capstone`                  | integrated scene: every feature in one frame (Slice Z)    |
| `camera_pose`     | `--camera 0.2,-0.1,0,3,10`    | scripted runtime-camera pose (Slice AA)                   |
| `gizmo`           | `--gizmo 2`                   | editor selection + transform gizmo (Slice AB)             |
| `csm`             | `--csm`                       | cascaded shadow maps over a receding plane (Slice AD)     |
| `spot`            | `--spot`                      | spot light + perspective spot shadow (Slice AE)           |
| `point_shadow`    | `--point-shadow`              | omnidirectional point-light 6-face cube shadows (Slice AF)|
| `clustered`       | `--clustered`                 | clustered/Forward+ lighting, 192 lights, byte-identical to brute force (Slice AG) |
| `ssr`             | `--ssr`                       | screen-space reflections off a reflective floor (Slice AH)|

---

## Architecture

Hazard Forge is organized in layers, all above the seam:

1. **HAL** (`engine/hal/`) — SDL3 window + Vulkan surface creation.
2. **RHI seam** (`engine/rhi/`) — pure C++ interfaces (`IRHIDevice`, `ICommandBuffer`, `IPipeline`, `IBuffer`, `ITexture`, `IRenderTarget`, `ISwapchain`, compute). Zero backend symbols.
3. **Backends** — Vulkan (`engine/rhi_vulkan/`) and Metal (`engine/rhi_metal/`). Accessed via `rhi::CreateDevice(Backend::Vulkan, window)` or `rhi::mtl::CreateMetalDeviceHeadless(w, h)`.
4. **Engine modules** — `scene/`, `render/` (render graph), `asset/` (glTF + HDR env), `anim/`, `physics/`, `runtime/`, `editor/`, `debug/`. All depend only on `rhi/` + `math/`; the backend-agnostic subset compiles into `hf_core` for the sanitized unit tests.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the seam design, descriptor model, frame structure, the shared HLSL→MSL toolchain, and per-backend notes.

---

## Repo layout

```
hazard-forge/
├── CMakeLists.txt              Top-level C++20 build; HF_SANITIZE opt-in
├── CMakePresets.json           windows-msvc-debug / -release / -asan / macos-arm64-debug
├── conanfile.py                SDL3, vk-bootstrap, VMA, vulkan-headers/loader
├── cmake/                      HLSL→SPIR-V shader-compile CMake rule
├── engine/
│   ├── hal/                    SDL3 window + Vulkan surface
│   ├── math/                   Vec3, Mat4, Quat (TRS, Inverse, slerp, ...)
│   ├── rhi/                    THE SEAM — pure interfaces, zero backend symbols
│   ├── rhi_vulkan/             Vulkan backend (vk-bootstrap, VMA, Vulkan 1.3 dynamic rendering)
│   ├── rhi_metal/              Metal backend (Obj-C++/ARC, runtime MSL compile)
│   ├── render/                 RenderGraph + csm / spot / point_shadow / clustered / ssr (header-only math)
│   ├── scene/                  Vertex, Transform, Mesh, Renderable, scene_io, commands, instancing
│   ├── asset/                  glTF loader + scene-graph import + HDR env loader
│   ├── anim/                   skeleton + animation sampling + blending
│   ├── physics/                impulse rigid-body World
│   ├── runtime/                Clock/FixedTimestep, Camera, FlyCameraController
│   ├── editor/                 picking, gizmo (+ ImGui editor shell, Vulkan-only)
│   └── debug/                  DebugDraw collector + emitters
├── shaders/                    Shared HLSL (lit/pbr/ibl/shadow/post/bloom/ssao/sky/...) → SPIR-V & MSL
├── samples/hello_triangle/     Vulkan sample: every showcase via --*-shot headless capture + --fly
├── metal_headless/             Standalone no-Conan/no-SDL Metal target (visual_test, 20 showcases)
├── tests/                      Pure unit tests (math/ecs/render_graph/scene_io/commands/anim/
│   │                           physics/runtime/editor/...) + rhi_smoke + golden/metal/*.png
│   └── golden/metal/           The 20 committed Metal goldens
├── scripts/verify.ps1          Cross-platform gate: Windows ctest + Mac 20-golden compare
├── ci/                         Staged GitHub Actions workflow (see ci/README.md)
└── docs/                       ARCHITECTURE.md + per-slice specs/plans
```

---

## Building

### Windows — Vulkan (MSVC + Ninja + Conan 2)

Prerequisites: Visual Studio Build Tools 2022 (MSVC x64), CMake ≥ 3.25, Ninja, Conan 2, Vulkan SDK.

```powershell
# 1. Install Conan dependencies and generate the CMake toolchain:
conan install . -of=build/windows-msvc-debug `
    -s build_type=Debug -s compiler.cppstd=17 `
    -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing

# 2. Configure + build + test (from a VS x64 developer shell):
cmake --preset windows-msvc-debug
cmake --build --preset windows-msvc-debug
ctest --preset windows-msvc-debug          # 18 tests

# 3. AddressSanitizer build of the pure-C++ core + tests:
conan install . -of=build/windows-msvc-asan `
    -s build_type=Debug -s compiler.cppstd=17 `
    -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing
cmake --preset windows-msvc-asan
cmake --build --preset windows-msvc-asan
ctest --preset windows-msvc-asan
```

### macOS — Metal headless (no Conan, no SDL, no full Xcode required)

`metal_headless/` is a standalone CMake project that compiles the real Metal backend + engine, generates MSL from the shared HLSL at build time, renders a showcase into an offscreen texture, and writes a PNG. Command Line Tools suffice.

```sh
cmake -S metal_headless -B build-metal -G Ninja
cmake --build build-metal          # also generates *.gen.metal from the HLSL
./build-metal/visual_test out.png  # default scene
```

### Full cross-platform verification (one command)

```powershell
scripts\verify.ps1
```

Runs the Windows/Vulkan ctest locally and drives the bench Mac over SSH to build `metal_headless` once and golden-compare **all 20** Metal goldens at threshold `0.0`. Prints a per-golden DIFF table and an overall `VERIFY: PASS/FAIL`. (`-SkipWindows` / `-SkipMac` run one half.)

---

## Build & run the showcases

The same scenes render on both backends. On **Vulkan** (Windows sample), each is a `--*-shot` headless capture (or run `--fly` for the live navigable viewport); on **Metal** (`metal_headless/visual_test`), each is the matching flag below.

| showcase            | Vulkan (`hello_triangle.exe`)      | Metal (`visual_test`)              |
| ------------------- | ---------------------------------- | ---------------------------------- |
| default lit+shadow  | `--shot out.bmp`                   | `out.png`                          |
| skinning            | `--skinning-shot out.bmp`          | `--skinning out.png`               |
| PBR helmet          | `--pbr-shot out.bmp`               | `--pbr out.png`                    |
| instancing          | `--instanced-shot out.bmp`         | `--instanced out.png`              |
| HDR IBL             | `--ibl-shot out.bmp`               | `--ibl out.png`                    |
| physics             | `--physics-shot out.bmp`           | `--physics out.png`                |
| transparency        | `--transparency-shot out.bmp`      | `--transparency out.png`           |
| bloom               | `--bloom-shot out.bmp`             | `--bloom out.png`                  |
| glTF scene import   | `--scene-shot out.bmp`             | `--scene out.png`                  |
| debug viz           | `--debug-shot out.bmp`             | `--debug out.png`                  |
| animation blend     | `--blend-shot out.bmp`             | `--blend out.png`                  |
| SSAO                | `--ssao-shot out.bmp`              | `--ssao out.png`                   |
| capstone            | `--capstone-shot out.bmp`          | `--capstone out.png`               |
| scripted camera     | `--camera-shot <yaw,pitch,x,y,z> out.bmp` | `--camera <yaw,pitch,x,y,z> out.png` |
| editor gizmo        | `--gizmo-shot <objIndex> out.bmp`  | `--gizmo <objIndex> out.png`       |
| cascaded shadows    | `--csm-shot out.bmp`               | `--csm out.png`                    |
| spot-light shadows  | `--spot-shot out.bmp`              | `--spot out.png`                   |
| point shadows       | `--point-shadow-shot out.bmp`      | `--point-shadow out.png`           |
| clustered lighting  | `--clustered-shot out.bmp`         | `--clustered out.png`              |
| screen-space reflections | `--ssr-shot out.bmp`          | `--ssr out.png`                    |
| live viewport       | `--fly` (WASD + mouse-look)        | *(Vulkan/Windows only)*            |

---

## Repository

`github.com/hassard0/Hazard-Forge`

---

## Roadmap

- Windowed Metal present loop (SDL Metal view + present) for an interactive macOS viewport — currently headless-offscreen only.
- Continued editor work (multi-select, undo/redo surfaced in the UI, asset browser).
- Broader physics (boxes/convex, joints) and animation (state machines, IK).
```
