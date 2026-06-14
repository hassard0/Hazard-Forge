# Hazard Forge

A C++20 cross-platform game engine built around a thin, explicit **Rendering Hardware Interface (RHI) seam** that renders natively on **Vulkan (Windows)** and **Apple Metal (macOS, Apple Silicon)** from one codebase. The design is Apple-native and Metal-first in philosophy, with Vulkan as the primary shipping platform while Metal parity is being completed.

> **Status:** Active development — in-progress engine, not a finished product. The RHI-seam thesis is proven on real hardware: the same engine code renders a multi-pass lit scene on a Vulkan RTX GPU (Windows) and Apple Metal (M4, macOS), verified by deterministic golden-image regression tests. Approximately 14 feature slices have shipped, and the Metal backend has reached parity on offscreen render targets, post-processing, and directional shadow mapping.

---

## What it is

Hazard Forge is a focused, tooling-obsessed C++ engine built slice by slice. Each slice lands a specific rendering capability through the shared RHI seam — which means every feature is simultaneously a Vulkan implementation and a specification for the Metal backend to implement.

The central bet: **one engine layer above a clean seam, two GPU backends below it, verified headlessly by golden images.** No Vulkan symbols leak above `engine/rhi/`; no Metal symbols leak above `engine/rhi_metal/`.

---

## Features (shipped)

- **RHI abstraction** — Vulkan and Metal backends behind a pure C++ virtual-interface seam (`engine/rhi/rhi.h`). No backend symbols visible to engine or application code.
- **3D math library** — own `hf::math` (Vec3, Mat4): perspective, orthographic, LookAt, rotation, scale, translate. Right-handed, column-major, Vulkan [0,1] depth range.
- **Depth buffering** — `D32_Float` depth, recreated on resize.
- **Textures** — staging upload (host → device-local), samplers, descriptor sets. Procedural checkerboard, extensible to loaded images.
- **Descriptor model** — frequency-based layout: per-frame UBO at set 0, per-material texture at set 1. Per-frame double-buffered UBOs (CPU never writes a buffer the GPU is reading). Model matrix via push constants.
- **Blinn-Phong lighting** — directional light + ambient + specular in the fragment shader, with normals through the scene vertex format.
- **Multiple colored point lights** — up to 3 point lights with radius-based attenuation, orbit around the scene.
- **Directional shadow mapping (PCF)** — depth-only shadow pass from the light into a 2048² shadow map; 3×3 PCF filter in the lit shader; `Mat4::Ortho` for the light projection.
- **Scene layer** — `Transform` (TRS), `Mesh` (owns GPU vertex+index buffers, factories: Cube/Plane/Sphere), `Renderable` (mesh + texture + transform). Depends only on `rhi/` and `math/`; zero Vulkan symbols.
- **Offscreen render targets** — `IRenderTarget` for rendering the scene into a sampleable color texture before compositing.
- **Post-processing** — fullscreen triangle pass (no vertex buffer): ACES-style tonemap (Reinhard), gamma, vignette. Wired as a second pipeline sampling the offscreen RT.
- **Procedural skybox** — gradient sky + sun disk drawn first in the scene pass via camera-ray reconstruction from the per-frame UBO. No matrix inverse; no extra textures.
- **Headless GPU capture** — `CaptureNextFrame()` / `GetCapturedPixels()`: renders a frame, reads pixels back from the GPU, and writes a BMP. No visible desktop required. The primary verification and regression-testing path.

**Metal status:** The Metal backend renders the full multi-object scene headless on Apple Silicon (M4), golden-tested, and has reached parity on **offscreen render targets**, **post-processing** (ACES tonemap + vignette), and **directional shadow mapping (PCF)** — each verified against a committed M4 golden at `DIFF 0.0000`. Remaining gaps: the skybox and point-light passes (sample-level, not yet wired into the Metal entry point) and a unified HLSL→MSL shader toolchain (Metal currently uses hand-written MSL alongside the HLSL).

---

## Architecture

Hazard Forge is organized around three layers:

1. **HAL** (`engine/hal/`) — SDL3 window and Vulkan surface creation.
2. **RHI seam** (`engine/rhi/`) — pure C++ interfaces (`IRHIDevice`, `ICommandBuffer`, `IPipeline`, `IBuffer`, `ITexture`, `IRenderTarget`, `ISwapchain`). Zero backend symbols.
3. **Backends** — Vulkan (`engine/rhi_vulkan/`) and Metal (`engine/rhi_metal/`). Accessed via `rhi::CreateDevice(Backend::Vulkan, window)` or `rhi::mtl::CreateMetalDeviceHeadless(w, h)`.
4. **Scene layer** (`engine/scene/`) — `Vertex`, `Transform`, `Mesh`, `Renderable`. Depends on `rhi/` and `math/` only.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a deeper treatment of the seam design, descriptor model, and per-backend differences.

---

## Repo layout

```
hazard-forge/
├── CMakeLists.txt              Top-level C++20 build
├── CMakePresets.json           windows-msvc-debug / -release / macos-arm64-debug
├── conanfile.py                SDL3, vk-bootstrap, VMA, vulkan-headers/loader
├── cmake/                      DXC shader-compile CMake rule
├── engine/
│   ├── hal/                    SDL3 window + Vulkan surface
│   ├── math/math.h             Vec3, Mat4 (perspective, ortho, LookAt, TRS)
│   ├── rhi/                    THE SEAM — pure interfaces, zero backend symbols
│   │   ├── rhi.h               IRHIDevice, ICommandBuffer, IPipeline, IBuffer, ITexture, IRenderTarget
│   │   └── rhi_factory.*       CreateDevice(Backend::Vulkan) → Vulkan backend
│   ├── rhi_vulkan/             Vulkan backend (vk-bootstrap, VMA, Vulkan 1.3 dynamic rendering)
│   ├── rhi_metal/              Metal backend (Obj-C++/ARC, runtime MSL compile)
│   └── scene/                  Transform, Mesh (Cube/Plane/Sphere), Renderable, Vertex
├── shaders/                    HLSL shaders (lit, shadow, post, sky) → SPIR-V via DXC
│                               lit.metal (MSL, for the Metal backend)
├── samples/
│   └── hello_triangle/         Vulkan sample: multi-pass scene, --shot headless capture
├── metal_headless/             Standalone no-Conan/no-SDL Metal target (visual_test)
├── tests/
│   ├── rhi_smoke.cpp           Headless device/RT/shadow-map init + teardown
│   ├── math_test.cpp           Pure math unit tests (perspective, ortho, TRS, etc.)
│   └── golden/metal/
│       └── scene.png           Reference golden image (M4, deterministic; diff → 0.0000)
└── docs/superpowers/
    ├── specs/                  Per-slice design documents
    └── plans/                  Per-slice implementation plans
```

---

## Building

### Windows — Vulkan (MSVC + Ninja + Conan 2)

Prerequisites: Visual Studio Build Tools 2022 (MSVC x64), CMake ≥ 3.25, Ninja, Conan 2, Vulkan SDK.

```powershell
# 1. Install Conan dependencies and generate the CMake toolchain:
conan install . -of=build/windows-msvc-debug `
    -s build_type=Debug `
    -s compiler.cppstd=17 `
    -c tools.cmake.cmaketoolchain:generator=Ninja `
    --build=missing

# 2. Configure (from a VS x64 developer shell):
cmake --preset windows-msvc-debug

# 3. Build:
cmake --build --preset windows-msvc-debug

# 4. Test (headless — no GPU required for math_test; rhi_smoke needs a Vulkan device):
ctest --preset windows-msvc-debug

# 5. Run the sample (interactive):
.\build\windows-msvc-debug\samples\hello_triangle\hello_triangle.exe

# 6. Headless capture (no visible desktop needed):
.\build\windows-msvc-debug\samples\hello_triangle\hello_triangle.exe --shot out.bmp
```

### macOS — Metal headless (no Conan, no SDL, no Xcode required)

The `metal_headless/` target is a standalone CMake project that compiles the real Metal backend and scene layer, renders the full Slice-F scene into an offscreen texture, and writes a PNG. Command Line Tools suffice; MSL is compiled at runtime — no `metal` CLI / no full Xcode toolchain.

```sh
cmake -S metal_headless -B build-metal -G Ninja
cmake --build build-metal
./build-metal/visual_test out.png

# Compare against the golden image (deterministic — diff should be 0.0000):
# tests/golden/metal/scene.png
```

---

## Roadmap

Near-term:
- Metal backend parity — offscreen render targets, shadow mapping, skybox/post pipeline
- Unified HLSL → MSL shader toolchain (so one shader source compiles for both backends)

Later slices:
- PBR materials, normal mapping
- glTF model loading, asset pipeline
- Compute shaders
- Render graph
- ECS runtime
- Editor and asset pipeline
