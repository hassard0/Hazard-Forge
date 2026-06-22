# Getting Started with Hazard Forge

> **Goal:** clone → build → run a showcase → write your first custom sample → understand the render pattern, in about
> 30 minutes. This is the on-ramp the README and `docs/ARCHITECTURE.md` assume you've read; it bridges "I cloned the
> repo" to "I'm rendering my own scene." When you want a specific capability (ray tracing, GI, physics, …) and the
> flag to see it, jump to [`CAPABILITIES.md`](CAPABILITIES.md).

## 1. Build (5 minutes)

### Windows / Vulkan (the primary dev target)

You need an RTX-class Vulkan GPU, the Vulkan SDK (for `glslc` + `spirv-cross`), Conan 2.x, CMake 3.25+, and a VS 2022
x64 developer shell.

```powershell
# from a VS x64 developer shell, at the repo root:
conan install . -of=build/windows-msvc-debug --build=missing -s build_type=Debug
cmake --preset windows-msvc-debug
cmake --build --preset windows-msvc-debug
ctest --preset windows-msvc-debug          # the full test suite — everything green
```

### macOS / Metal (Apple Silicon)

No Conan, no SDL, no full Xcode.app — but beyond Apple Command Line Tools you need the shader toolchain:

```sh
brew install cmake ninja shaderc spirv-cross    # shaderc provides glslc; or the LunarG Vulkan SDK
cmake -S metal_headless -B build-metal -G Ninja
cmake --build build-metal
```

## 2. Run a showcase (2 minutes)

Hazard Forge's capabilities are reached through **headless showcase flags** — each renders one feature to a PNG and is
byte-compared cross-platform. This is how you *see* what the engine does without writing code.

```powershell
# Windows / Vulkan — render a feature to a PNG:
.\build\windows-msvc-debug\samples\hello_triangle\hello_triangle.exe --pbr-shot pbr.png
.\build\windows-msvc-debug\samples\hello_triangle\hello_triangle.exe --ibl-shot ibl.png
.\build\windows-msvc-debug\samples\hello_triangle\hello_triangle.exe --gi6-hero-shot gi.png
```

```sh
# macOS / Metal — the same scenes (the flag drops the "-shot" suffix; the PNG path is the last arg):
./build-metal/visual_test --pbr pbr.png
./build-metal/visual_test --gi6-hero gi.png
```

The complete, always-current list of flags + their reference images is `scripts/verify.ps1`'s `$Goldens` table (265
showcases). The engine also emits its own machine-readable manifest:

```powershell
hello_triangle.exe --introspect manifest.json   # features + showcases + commands, as JSON
hello_triangle.exe --agent-api                   # the versioned Agent-SDK contract
```

## 3. Your first custom scene (10 minutes) — the declarative path

The fastest way to author a scene is the **declarative scene spec** (no C++): a JSON array of objects, loaded by
`scene_io`. The engine canonicalizes it deterministically (the same spec → the same scene, byte-for-byte).

```json
// my_scene.json
[
  { "mesh": "cube",  "position": [0, 0, 0], "baseColor": "0.8 0.2 0.2", "metallic": 0.0, "roughness": 0.4 },
  { "mesh": "cube",  "position": [2, 0, 0], "metallic": 1.0, "roughness": 0.1 },
  { "mesh": "duck",  "position": [-2, 0, 0], "scale": [2, 2, 2] }
]
```

```powershell
hello_triangle.exe --author-scene my_scene.json my_scene.canonical.json   # load + re-emit the canonical form
```

You can also drive the live ECS from a **command script** (mutate + capture, headlessly) — see
`engine/scene/commands.h` (`--commands cmds.json`): `add` / `remove` / `set_transform` / `set_material` / `query` /
`capture`. This is the engine's agent-facing surface; an AI agent or a CI job drives a scene entirely from a script.

## 4. Your first C++ sample (10 minutes) — the render pattern

For full control, write a sample like `samples/hello_triangle/main.cpp` (Vulkan) or `mac_window/main.mm` (Metal,
interactive). The pattern every sample follows:

1. **Create the device + window** (`rhi::IRHIDevice`, via the HAL).
2. **Build the pipelines** you need — each is a `GraphicsPipelineDesc` over a shared HLSL shader (`lit.frag.hlsl`,
   `lit_instanced.vert.hlsl`, …). The same HLSL compiles to SPIR-V (Vulkan) and MSL (Metal); you never write two
   shaders.
3. **Load assets** — `asset::LoadSkinnedGltfModel` / the scene_io loader; meshes, textures, skeletons.
4. **Per frame:** fill the shared `FrameData` cbuffer (camera `viewProj`, `lightDir`, up to `HF_MAX_POINT_LIGHTS`
   point lights, and `skyParams` — note `skyParams.zw = (time, frameIndex)` for animated shaders), record the passes
   (shadow → main → post), and present (or `CaptureNextFrame()` for a headless PNG).

The single most useful file to read end-to-end is `samples/hello_triangle/main.cpp` — it wires every showcase flag, so
it's a catalog of "how do I do X." For the interactive camera + input pattern, read `mac_window/main.mm`.

### Cookbook — common tasks

| Task | Where to look |
|---|---|
| Load a glTF mesh (static or skinned) | `asset::LoadGltfModel` / `LoadSkinnedGltfModel` (`engine/asset/gltf_loader.h`) |
| Add a light | fill `FrameData.lightDir`/`lightColor` (directional) or `ptPos[i]`/`ptColor[i]` (point, up to `HF_MAX_POINT_LIGHTS`) |
| Write a custom shader | add `your.frag.hlsl`, register the `:fs` line in `samples/hello_triangle/CMakeLists.txt` + the `hf_gen_msl` line in `metal_headless/CMakeLists.txt`; `#include "frame_data.hlsli"` for the per-frame uniforms, `#include "procedural_sky.hlsli"` for the sky/IBL |
| Animate a shader over time | read `f.skyParams.z` (seconds) / `f.skyParams.w` (frame index) — see `shaders/sky_animated.frag.hlsl` and issue #5 |
| Hook up input | `runtime::InputState` + `runtime::Key` (the platform-agnostic key set); see `runtime::FlyCameraController` |
| Bind a custom control key | use a `runtime::Key` slot (digits, A–Z, F-keys, arrows are all available) instead of intercepting raw platform keycodes |
| Capture a frame to PNG | `device->CaptureNextFrame()` then `GetCapturedPixels()` — the same path all the goldens use |
| Drive a scene from a script (no C++) | `--author-scene` (declarative) or `--commands` (mutate/query/capture); `engine/scene/commands.h` |

## 5. Understand the determinism guarantees

The engine's distinguishing property: most of it is **deterministic and bit-identical across Vulkan/Windows and
Metal/macOS**, and the simulation layers are **lockstep/rollback-replayable**. Which subsystems are deterministic, and
where the float seams are, is documented per-subsystem in `docs/ARCHITECTURE.md` (and summarized in
`CAPABILITIES.md`). The short version: the integer simulation + the agent/IO layers are bit-exact; the final
raster/shade is float (per-backend deterministic, with a documented in-band cross-vendor delta). The
`--replay-record` / `--replay-verify` / `--determinism-stress` flags let you prove a command stream replays
bit-identically.

## Where to go next

- [`CAPABILITIES.md`](CAPABILITIES.md) — the feature → flag → golden map (find any capability and how to run it).
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the deep architectural breakdown, per-subsystem.
- `scripts/verify.ps1` — the live, complete index of every showcase + its reference image.
- `samples/hello_triangle/main.cpp` — the catalog of every showcase, the best "how do I do X" reference.

---

### Docs roadmap (this is Phase 1)

This file is **Phase 1** (the getting-started on-ramp) of the documentation plan in issue #32. Planned next:
**Phase 2** — an mkdocs site built from `docs/`, deployed to GitHub Pages; **Phase 3** — Doxygen API reference over the
public headers; **Phase 4** — an engineering blog (per-flagship deep-dives). Contributions toward those phases are
welcome; the per-feature shader/path explainers (the `lit_*.frag.hlsl` variants) are tracked as part of Phase 2.
