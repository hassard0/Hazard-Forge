# Hazard Forge — Slice A: Hello Triangle + RHI Seam

**Date:** 2026-06-14
**Status:** Approved design
**Repo root:** `C:\Users\ihass\dev\hazard-forge`

## Context

Hazard Forge is a C++ game engine positioned as a focused, Apple-native, Vulkan-first,
tooling-obsessed challenger to UE5 — deliberately *not* a breadth copy. The full engine is
a multi-year, multi-person effort spanning a dozen independent subsystems (RHI, shader
toolchain, asset compiler, editor, ECS runtime, plugins, package registry, etc.). It cannot
be specified or built as a single plan.

We therefore build **slice by slice**, each slice its own spec → plan → implement cycle.
**Slice A** is the spine: a runnable triangle that validates the single most important
architectural bet — the **RHI (Rendering Hardware Interface) seam** — at the lowest possible
ceremony, Vulkan-first on Windows.

### Forward note (not Slice A)
When later slices reach the **editor, testing, and visualization** subsystems, their designs
must **optimize for agentic development**: headless control, machine-readable state, and
scriptable operation so AI agents can drive, observe, and test them. Recorded here so it is
not lost; out of scope for Slice A.

## Goal / Definition of Done

A `hello_triangle.exe` that:
- opens an SDL3 window,
- renders a colored triangle through the Vulkan RHI backend,
- survives window resize (swapchain recreation),
- shuts down cleanly,
- produces **zero Vulkan validation-layer errors** in debug builds.

## Locked Decisions

| Area | Decision | Rationale |
|---|---|---|
| Dependency manager | **Conan 2** | Spec-aligned; binary caching + profiles pay off when CI/matrix arrives. |
| Windowing / HAL | **SDL3** | Broadest HAL surface (window, input, surface, future audio/gamepad) with macOS parity for the coming Metal work. |
| Vulkan bring-up | **vk-bootstrap + VMA**, raw C Vulkan for the rest | Collapses instance/device/swapchain/allocation boilerplate; we still hand-write command/pipeline/sync (the parts that matter for an engine). |
| RHI dispatch | **C++ abstract (virtual) interfaces** | RHI is internal — no C-ABI purity needed yet. Cleanest to read, trivial to add the Metal backend later. |
| Build | **MSVC + Ninja + CMake Presets** | Standard, deterministic, Conan integrates via CMakeToolchain. |
| Language | **C++20** | Modern baseline. |
| Shaders | **HLSL → SPIR-V at build time via DXC** | Aligns with the engine's eventual HLSL/Slang strategy. |

## Architecture

### Repo layout
```
hazard-forge/
├─ CMakeLists.txt             top-level, C++20, Ninja
├─ CMakePresets.json          windows-msvc-debug / windows-msvc-release
├─ conanfile.py               sdl, vk-bootstrap, vulkan-memory-allocator, vulkan-headers/loader
├─ cmake/                     helper modules (DXC shader-compile rule)
├─ engine/
│  ├─ hal/                    platform: window, input, Vulkan surface (SDL3 wrapper)
│  ├─ rhi/                    THE SEAM — pure interfaces, NO vulkan symbols
│  │  ├─ rhi.h               interfaces + descriptor structs + enums
│  │  └─ rhi_factory.cpp     CreateDevice(Backend::Vulkan) → vulkan backend
│  └─ rhi_vulkan/             ONLY place vulkan / vk-bootstrap / VMA appear
├─ shaders/                   triangle.vert.hlsl, triangle.frag.hlsl
├─ samples/
│  └─ hello_triangle/         main.cpp — the runnable target
└─ docs/superpowers/specs/
```

### The RHI seam (minimal surface — nothing speculative)
- **`IRHIDevice`** — owns instance/device; factory for swapchain, buffers, shader modules,
  pipelines; `BeginFrame()` / `EndFrame()`.
- **`ISwapchain`** — surface + images; `AcquireNext()`, `Present()`, recreate-on-resize.
- **`ICommandBuffer`** — `BeginRenderPass`, `BindPipeline`, `BindVertexBuffer`, `Draw`,
  `EndRenderPass`.
- **`IPipeline`** — graphics PSO from a small `GraphicsPipelineDesc` (shaders, vertex layout,
  color formats).
- **`IBuffer`** — vertex buffer; upload helper (map/unmap or staging).
- **`IShaderModule`** — wraps compiled SPIR-V.

**Hard rule:** no `vk*` type appears in `engine/rhi/`. Backends are handed out as
`std::unique_ptr<IRHIDevice>` from `rhi_factory`. This is the seam the future Metal backend
implements without touching front-end code.

### Vulkan backend internals (`engine/rhi_vulkan/`)
- `vk-bootstrap`: instance, physical-device selection, logical device, queues, swapchain.
- **VMA**: vertex buffer allocation.
- Raw C Vulkan: command recording, render pass (or dynamic rendering), pipeline creation,
  per-frame sync (acquire/submit/present with per-frame fences + semaphores, double-buffered).
- **Validation layers enabled in debug** — the correctness gate.

### Shaders
- Author `triangle.vert.hlsl` / `triangle.frag.hlsl` — a hardcoded-in-NDC colored triangle.
- A CMake custom command invokes **DXC** (`-spirv`) to produce `.spv` at build time.
- `IShaderModule` loads the `.spv`.
- No math library yet (vertices hardcoded in NDC); deferred until actually required.

### App loop (`samples/hello_triangle/main.cpp`)
SDL3 window → `CreateDevice(Vulkan)` → create swapchain → load shaders → build pipeline →
upload 3-vertex buffer → loop { poll events, acquire, record (clear + draw), present, handle
resize } → clean shutdown.

## Error Handling
- Vulkan validation layers on in debug; any validation error fails the DoD.
- RHI calls that can fail surface explicit error returns/results (no silent failure); the
  sample logs and exits non-zero on init failure.
- Swapchain `VK_ERROR_OUT_OF_DATE` / suboptimal triggers recreation, not a crash.

## Testing
- **CTest headless smoke test:** create RHI device + swapchain, then tear down — catches
  init/leak regressions without rendering a frame.
- **Validation layers** as the runtime correctness gate in debug.
- Real unit tests grow as the RHI surface grows; we do **not** fabricate a test asserting
  "a triangle appeared."

## Out of Scope for Slice A
Metal backend, asset pipeline / derived-data cache, editor, ECS runtime, plugin system &
C ABI, math library, profiling/telemetry, package registry. All are later slices.
