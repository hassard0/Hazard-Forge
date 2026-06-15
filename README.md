# Hazard Forge

A C++20 cross-platform game engine built around a thin, explicit **Rendering Hardware Interface (RHI) seam** that renders natively on **Vulkan (Windows)** and **Apple Metal (macOS, Apple Silicon)** from one codebase. The design is Apple-native and Metal-first in philosophy, with Vulkan as the primary shipping platform; Metal parity is verified headlessly against committed golden images.

> **Status:** Active development — an in-progress engine, not a finished product. The RHI-seam thesis is proven on real hardware: the same engine code renders a full PBR/IBL/post-processed scene on a Vulkan RTX GPU (Windows) and on Apple Metal (M4, macOS), verified by **35 deterministic golden-image regression tests** (plus a machine-readable engine-state JSON golden **and** a byte-exact audio-WAV golden) that must each diff `0.0000`, and by a **34-test** ctest suite that runs clean under AddressSanitizer. Roughly slices A–BF have shipped — PBR materials, image-based lighting, HDR bloom, SSAO, alpha-blended transparency, glTF scene-graph import, skeletal animation + blending, rigid-body physics, GPU instancing, compute particles, immediate-mode debug visualization, an interactive runtime (fixed-timestep loop + flyable camera), a full shadow set (cascaded shadow maps, spot-light shadows, omnidirectional point-light cube shadows), clustered/Forward+ lighting, screen-space reflections, volumetric fog / light shafts, reflection + irradiance probes (local cubemap GI), temporal anti-aliasing (TAA), CPU frustum culling, **GPU-driven culling + indirect draw**, a **render graph with automatic resource-state barriers** (Vulkan-synchronization-validation-clean), **multithreaded command recording** (byte-identical 1-vs-N workers), a **data-driven material / shader graph** (expanded node set incl. a **tangent-space normal-map node** + a multi-material scene) with **live runtime material authoring**, a **baked-font text / HUD overlay renderer**, a **deterministic integer-mixer audio engine** (byte-exact WAV golden), **distance-based scene / asset streaming** (per-frame budget + hysteresis), **procedural terrain / heightmap generation** (finite-difference normals), a **playable deterministic game sample**, machine-readable engine-state JSON introspection for agents, a live editor (mouse pick + gizmo drag + shader/scene hot-reload), and a macOS windowed Metal viewport.

---

## What it is

Hazard Forge is a focused, tooling-obsessed C++ engine built slice by slice. Each slice lands a specific rendering or runtime capability through the shared RHI seam — which means every feature is simultaneously a Vulkan implementation and a specification for the Metal backend to implement.

The central bet: **one engine layer above a clean seam, two GPU backends below it, verified headlessly by golden images.** No Vulkan symbols leak above `engine/rhi/`; no Metal symbols leak above `engine/rhi_metal/`. The seam invariant is the architectural spine of the project.

---

## Features (shipped, slices A–BF)

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
- **Volumetric fog / light shafts** — a ray-marched volumetric pass (`render/volumetric.h`: Henyey-Greenstein phase function, camera-basis world-ray reconstruction, Beer-Lambert transmittance) accumulates in-scattered light through participating media (shadow-sampled god rays), composited over the lit scene. Header-only math is unit-tested.
- **Reflection + irradiance probes (local cubemap GI)** — baked 6-face environment probes (`render/probe.h`: per-face cube projection + dominant-axis face selection + a combined reflection/irradiance atlas-tile layout the `lit_probe` shader mirrors) provide local specular reflection and diffuse irradiance, bound through the seam via `BindReflectionProbe`. Header-only math is unit-tested.
- **Multi-pass post-processing** — fullscreen pass with FXAA, exposure, ACES tonemap, cinematic grade, film grain, and vignette.
- **HDR bloom** — render into an `RGBA16_Float` target, soft-knee threshold bright-pass, a 5-level downsample mip chain (13-tap dual filter), tent-filter upsample/combine, and a composite that tonemaps.
- **SSAO** — view-space normal+depth g-buffer prepass, a 16-sample hemisphere-kernel AO pass (baked kernel + noise, no runtime RNG), a box blur, and a composite that multiplies the lit scene by the blurred AO (contact AO).
- **Alpha-blended transparency** — a sorted (back-to-front) translucent pass that depth-tests against opaque geometry but does not write depth (`depthWrite=false`), with Fresnel-style alpha.
- **GPU instancing** — a per-instance vertex stream (`scene::InstanceTransformLayout`) drives a single `DrawIndexedInstanced` call; transforms come from the deterministic `scene::BuildInstanceGrid` (no RNG).
- **Compute particles** — a compute kernel animates a 50k-particle storage buffer (gravity + swirl fountain, deterministic respawn), drawn as additive points.
- **Procedural skybox** — gradient sky + sun disk drawn first in the scene pass via camera-ray reconstruction from the per-frame UBO. No matrix inverse, no extra textures.
- **Temporal anti-aliasing (TAA)** — a reprojecting TAA resolve (`render/taa.h`: deterministic Halton(2,3) sub-pixel jitter sequence + history reprojection + neighborhood clamp) accumulates jittered frames into a stable image. The jitter sequence is bit-identical across runs and backends, so the resolved golden is deterministic.
- **Frustum culling** — `render::Frustum` extracts the six Gribb–Hartmann planes from the view-projection and tests sphere/AABB bounds CPU-side (`render/frustum.h`, header-only + unit-tested); the `cull` showcase visualizes which objects survive the cull from an overview camera.
- **GPU-driven culling + indirect draw** — a compute pass culls the instance set against the frustum on the GPU and compacts the survivors into an ordered indirect-args buffer (`render/gpu_cull.h` reference CPU model, mirrored by the compute shader), which a single `DrawIndexedIndirect` consumes. The CPU reference and the GPU result agree, and the `gpu_cull` golden matches the brute-force draw.
- **Render graph + automatic barriers** — `render::RenderGraph` declares passes over imported resources (shadow map / scene color / swapchain) and a **resource-state tracker + barrier solver** automatically inserts the correct transitions between passes from each resource's prior→next state. On Vulkan each transition lowers to an explicit `vkCmdPipelineBarrier2`, **proven hazard-free by the Khronos synchronization-validation layer**; on Metal the tracked-hazard model makes them no-ops. The graph logic itself carries zero backend symbols.
- **Multithreaded command recording** — passes record into per-thread secondary command buffers from a worker pool and are replayed in deterministic creation order (`vkCmdExecuteCommands` on Vulkan; an `MTLParallelRenderCommandEncoder`'s sub-encoders on Metal). A 1-worker render and an N-worker render are **byte-identical** (proven by the `mt` golden and a 1-vs-N hash test).
- **Data-driven material / shader graph** — materials are authored as a node graph in JSON (`assets/materials/*.mat.json`); `material::ShaderGraph` + the codegen emit HLSL for the PBR inputs (base color / metallic / roughness / emissive / **normal**). The node set spans `Constant`, `UV`, `TextureSample`, `Multiply`, `Add`, `Lerp`, `Fresnel`, a **Slice-AZ expansion** — `Swizzle`, `MakeFloat3`, `MakeFloat4`, `Dot`, `Normalize`, `Power`, `OneMinus`, `Saturate` — and the **Slice-BE `NormalMap` node** (samples a tangent-space normal-map slot and decodes `c*2-1`, feeding the `PBROutput` sink's 5th `normal` input), all terminating at the `PBROutput` sink — enough to author non-trivial materials as graphs. When a graph drives `PBROutput.normal`, codegen emits the perturbed-normal `hfShadePBRN` lighting variant (TBN-transformed shading normal); graphs that leave `normal` unconnected codegen byte-identically to before, so every pre-BE golden is unchanged. A **build-time codegen tool** bakes the showcase materials into committed generated HLSL, and a **live runtime path** re-compiles an edited material on the fly (dxc subprocess → SPIR-V → pipeline) for in-editor authoring. The build-time and runtime paths render **byte-identically** (the `mat_graph` / `mat_graph2` goldens; a runtime==build-time hash check). A **multi-material scene** (`--material-multi-shot` / `--material-multi`) renders three spheres each shaded by a distinct graph material in one frame (the `mat_multi` golden), and the `NormalMap` node is pinned by the `mat_normal` golden (`--material-normal-shot` / `--material-normal`).

### Text / HUD and audio

- **Text / HUD overlay renderer** — `engine/ui/` bakes a fixed 8×8 monospace bitmap font into an RGBA atlas (`BuildFontAtlas`) and lays a string out into a batch of screen-space NDC quads (`LayoutText`) — pure CPU, zero backend symbols, no asset/clock/RNG, so the same string yields the same pixels on both backends. The draw side reuses the existing alpha-blend + sampled-texture **screen-space overlay** pass (`shaders/text.{vert,frag}.hlsl`), compositing the HUD over the scene after post. Golden-captured standalone (`--hud-shot` / `--hud`, the `hud` golden) and over the game (`--game-hud-shot` / `--game-hud`, the `game_hud` golden).
- **Deterministic audio mixer + WAV writer** — `engine/audio/` is a software mixer that is **integer / fixed-point end to end** (Q15 gains, an int32 accumulator, hard-clamp to int16 — no float in the sample loop) with sine/square/noise voices and a piecewise-linear ADSR envelope, plus a canonical 16-bit-PCM WAV writer (hand-serialized little-endian, no timestamps). Pure CPU in `hf_core` (ASan-scoped, unit-tested), it produces **bit-identical** output on every compiler/run; `--audio-render out.wav` renders a fixed scene that is byte-exact against the committed `tests/golden/audio/scene.wav` golden.

### Assets, animation, physics

- **glTF loading** — header-only `cgltf`. Single-model loaders (geometry + full PBR material set) **and** a full **scene-graph import** (`LoadGltfScene`: node hierarchy walked to world transforms, one renderable per primitive, deduped materials).
- **Skeletal animation + blending** — GPU skinning (joint palette at a dedicated vertex binding); single-clip sampling and **cross-clip blending** (`BlendAnimations`: lerp T/S, slerp R per joint) on the CPU in `hf::anim`.
- **Rigid-body physics** — impulse-based `physics::World` (spheres + ground, contact resolution), deterministic fixed-step settling.

### World streaming and terrain

- **Distance-based scene / asset streaming** — `scene::StreamingWorld` divides a large world into a fixed `N×N` grid of cells; `Update(cameraPos)` keeps cells inside `loadRadius` resident, unloads cells beyond the larger `unloadRadius` (the in-between **hysteresis** band prevents thrashing), and processes loads/unloads **nearest-first** under a per-frame **budget** so content streams in over frames. The resident set is a pure function of (camera path, radii, budget, prior state) — pure CPU, zero RHI/backend symbols, unit-tested (`streaming_test`) and golden-captured (`--stream-shot` / `--stream`, the `stream` golden). Driven by a fixed scripted camera path the per-frame residency is bit-stable; the showcase capture frame reports `frame:40, resident:24`.
- **Procedural terrain / heightmap** — `terrain::Height(x,z)` is a fixed pure function (a few sines/cosines plus a deterministic integer value-noise lattice); `terrain::BuildTerrain(n, worldSize, heightScale)` generates an `n×n` lit/PBR-ready vertex grid with **central-finite-difference per-vertex normals** and a height-based vertex tint, no new shader required. It is compiled into both `hf_core`/`hf_engine` **and** the standalone `metal_headless` target from the **same** TU, so Windows/Vulkan and Apple/Metal generate a **bit-identical** mesh (the cross-backend golden contract). Unit-tested (`terrain_test`) and golden-captured (`--terrain-shot` / `--terrain`, the `terrain` golden); the showcase reports `n:128, verts:16384, tris:32258, peak:2.0972`.

### Runtime, editor, tooling

- **Playable game sample** — `game::roll_game`: a deterministic roll-a-ball gameplay layer (player sphere on a scripted track + collectible pickups) driven by `MakeRollGame`/`StepGame` at the engine's fixed timestep. Pure C++ above the seam, unit-tested, and golden-captured (`game`) at a fixed mid-track frame; the deterministic end state is `score:3, won:true, steps:380` every run.
- **Interactive runtime** — a fixed-timestep loop (`runtime::Clock`/`FixedTimestep`), a backend-agnostic flyable `runtime::Camera` (yaw/pitch → basis, `View`/`Proj`/`ViewProj`), and a `FlyCameraController`. The live windowed viewport (WASD + mouse-look) runs on **Vulkan/Windows** (`--fly`) **and on macOS/Metal** (windowed Metal present loop, below); the camera math itself is golden-verified on both backends.
- **Live editor (pick / drag / gizmos / hot-reload)** — inside the live `--fly` viewport: mouse-ray **picking** (`editor::picking`: cursor-px → NDC → `ScreenRayThroughCamera`, `PickNearest` over world AABBs) selects entities, transform **gizmos** (`editor::gizmo`: translate/rotate/scale axis math + `ApplyDrag` driven by prev/cur rays) drag them, and a `FileWatcher` **hot-reloads** edited shaders/scenes (poll-based change detection). The deterministic logic under all of it is pure C++ and unit-tested (`editor_test`, `live_editor_test`); the interactive mouse manipulation in the window is manual, and the **goldens + unit tests prove the math beneath it**.
- **Machine-readable engine-state introspection (for agents)** — `editor::DescribeEngine` emits a deterministic JSON document of the live engine state (engine/features/showcases/commands manifest + scene entities/transforms + camera/lights + stats; `backends == ["vulkan","metal"]`). Exposed via `--introspect <out.json>`; it is backend-agnostic (pure `hf_core`) and pinned by an **exact byte-match golden** (`tests/golden/introspect/default_scene.json`) so an agent can reliably observe — and, via the commands manifest, act on — the engine.
- **Immediate-mode debug visualization** — `debug::DebugDraw` collects grids / AABBs / wire spheres / arrows / contact markers into a LINE_LIST and draws them in one call through a debug-line pipeline (`depthTest=true, depthWrite=false`).
- **Headless GPU capture** — `CaptureNextFrame()` / `GetCapturedPixels()`: render a frame, read pixels back from the GPU, write a PNG/BMP. No visible desktop required — the primary verification path.
- **Vulkan validation-clean invariant** — every showcase runs under the Khronos validation layer (core + **synchronization** validation) with **zero** `VUID-*` / `SYNC-HAZARD-*` / `UNASSIGNED-*` / `[ERROR]` lines; the render graph's auto-inserted barriers are proven hazard-free by the sync-validation layer. (Only a benign `[WARNING: Performance]` "vertex attribute not consumed" notice from the depth-only shadow pipelines remains, and is expected.) The validation layer is pulled in as a Conan dependency for debug builds.
- **Cross-platform verification** — `scripts/verify.ps1` runs the Windows/Vulkan ctest **and** the introspection JSON-golden + audio-WAV-golden byte matches, **and** drives the bench Mac over SSH to build `metal_headless` and golden-compare all 35 Metal goldens at `DIFF 0.0000`, printing a per-golden table and an overall `VERIFY: PASS/FAIL`.
- **AddressSanitizer** — an opt-in `HF_SANITIZE=address` build (`windows-msvc-asan` preset) instruments the backend-agnostic core (`hf_core`) and the pure-C++ unit tests so they run clean under MSVC `/fsanitize=address`.

**Metal parity status:** The Metal backend renders **every** showcase headless on Apple Silicon (M4) and is verified against a committed golden at `DIFF 0.0000` for all 35 scenes (see below). The Metal shaders are **generated from the shared HLSL** at build time (HLSL → SPIR-V via `glslc` → MSL via `spirv-cross`), so there is no hand-written MSL to drift. A **windowed** Metal present loop now also exists on macOS: a SDL-free native Cocoa entry builds a `MetalDevice` from a `CAMetalLayer*` and runs the same interactive `--fly` viewport (pick / drag / gizmos). It is **build-verified** in CI/`verify.ps1` (the bench Mac compiles and headless-renders all 35 goldens); the interactive on-screen window itself is exercised manually — the user confirms the live viewport, while the deterministic logic under it (camera, picking, gizmo, file-watch) is golden- and unit-tested on both backends.

---

## The 35 Metal goldens

Each is produced by a distinct `metal_headless/visual_test` flag and compared against `tests/golden/metal/<name>.png` at threshold `0.0` (deterministic — two runs diff `0.0000`). Two further non-image goldens are byte-matched on the Windows side: the engine-state JSON (`tests/golden/introspect/default_scene.json`) and the audio WAV (`tests/golden/audio/scene.wav`, from `--audio-render`):

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
| `volumetric`      | `--volumetric`                | ray-marched volumetric fog / light shafts (Slice AJ)      |
| `probe`           | `--probe`                     | reflection + irradiance probes / local cubemap GI (Slice AK) |
| `taa`             | `--taa`                       | temporal anti-aliasing, Halton jitter + reprojection (Slice AP) |
| `cull`            | `--cull`                      | CPU frustum culling, overview-camera visualization (Slice AQ) |
| `gpu_cull`        | `--gpu-cull`                  | GPU-driven culling + compacted indirect draw (Slice AR)   |
| `mt`              | `--mt`                        | multithreaded recording, byte-identical to single-threaded (Slice AU) |
| `mat_graph`       | `--material`                  | data-driven material/shader graph, build-time codegen (Slice AV) |
| `mat_graph2`      | `--material2`                 | second graph material via live runtime compile path (Slice AW) |
| `mat_multi`       | `--material-multi`            | multi-material scene: 3 spheres, 3 distinct graph materials (Slice AZ) |
| `mat_normal`      | `--material-normal`           | NormalMap graph node: tangent-space normal map → perturbed PBR shading (Slice BE) |
| `game`            | `--game`                      | playable deterministic roll-a-ball game sample (Slice AX) |
| `hud`             | `--hud`                       | baked-font text / HUD screen-space overlay (Slice BA)     |
| `game_hud`        | `--game-hud`                  | game sample with the score HUD overlaid (Slice BA)        |
| `stream`          | `--stream`                    | distance-based scene streaming: resident cell subset (Slice BD) |
| `terrain`         | `--terrain`                   | procedural terrain / heightmap, finite-difference normals (Slice BF) |

---

## Architecture

Hazard Forge is organized in layers, all above the seam:

1. **HAL** (`engine/hal/`) — SDL3 window + Vulkan surface creation.
2. **RHI seam** (`engine/rhi/`) — pure C++ interfaces (`IRHIDevice`, `ICommandBuffer`, `IPipeline`, `IBuffer`, `ITexture`, `IRenderTarget`, `ISwapchain`, compute). Zero backend symbols.
3. **Backends** — Vulkan (`engine/rhi_vulkan/`) and Metal (`engine/rhi_metal/`). Accessed via `rhi::CreateDevice(Backend::Vulkan, window)` or `rhi::mtl::CreateMetalDeviceHeadless(w, h)`.
4. **Engine modules** — `scene/` (incl. distance-based `streaming`), `render/` (render graph + barrier solver + frustum/GPU-cull/TAA), `terrain/` (procedural heightmap + mesh gen), `asset/` (glTF + HDR env), `anim/`, `physics/`, `material/` (shader graph + NormalMap node + codegen + runtime compile + live authoring), `ui/` (baked-font text / HUD overlay), `audio/` (integer mixer + WAV writer), `game/` (roll_game), `runtime/`, `editor/`, `debug/`. All depend only on `rhi/` + `math/`; the backend-agnostic subset compiles into `hf_core` for the sanitized unit tests.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the seam design, descriptor model, frame structure, the shared HLSL→MSL toolchain, and per-backend notes.

---

## Repo layout

```
hazard-forge/
├── CMakeLists.txt              Top-level C++20 build; HF_SANITIZE opt-in
├── CMakePresets.json           windows-msvc-debug / -release / -asan / macos-arm64-debug
├── conanfile.py                SDL3, vk-bootstrap, VMA, vulkan-headers/loader, validation layers (debug)
├── cmake/                      HLSL→SPIR-V shader-compile CMake rule
├── engine/
│   ├── hal/                    SDL3 window + Vulkan surface
│   ├── math/                   Vec3, Mat4, Quat (TRS, Inverse, slerp, ...)
│   ├── rhi/                    THE SEAM — pure interfaces, zero backend symbols
│   ├── rhi_vulkan/             Vulkan backend (vk-bootstrap, VMA, Vulkan 1.3 dynamic rendering)
│   ├── rhi_metal/              Metal backend (Obj-C++/ARC, runtime MSL compile)
│   ├── render/                 RenderGraph + barrier solver + frustum / gpu_cull / taa + csm / spot / point_shadow / clustered / ssr / volumetric / probe (header-only math)
│   ├── scene/                  Vertex, Transform, Mesh, Renderable, scene_io, commands, instancing, streaming
│   ├── terrain/                procedural heightmap (deterministic Height field) + finite-diff-normal mesh gen
│   ├── asset/                  glTF loader + scene-graph import + HDR env loader
│   ├── anim/                   skeleton + animation sampling + blending
│   ├── physics/                impulse rigid-body World
│   ├── material/               shader_graph (+ NormalMap node) + codegen + material_loader + runtime_compile + live_material
│   ├── ui/                     baked 8x8 font atlas + screen-space text/HUD layout (pure CPU)
│   ├── audio/                  integer/fixed-point software mixer + canonical 16-bit PCM WAV writer
│   ├── game/                   roll_game: deterministic gameplay layer (sample)
│   ├── runtime/                Clock/FixedTimestep, Camera, FlyCameraController, hot_reload (FileWatcher)
│   ├── editor/                 picking, gizmo, introspect (engine-state JSON) (+ ImGui editor shell)
│   └── debug/                  DebugDraw collector + emitters
├── shaders/                    Shared HLSL (lit/pbr/ibl/shadow/post/bloom/ssao/volumetric/probe/taa/sky/...) → SPIR-V & MSL
│   └── generated/              build-time material-codegen output (mat_showcase*.frag.hlsl)
├── tools/                      material_codegen: build-time HLSL generator from *.mat.json graphs
├── mac_window/                 SDL-free native Cocoa entry: windowed Metal viewport from a CAMetalLayer*
├── samples/hello_triangle/     Vulkan sample: every showcase via --*-shot headless capture + --fly + --introspect
├── metal_headless/             Standalone no-Conan/no-SDL Metal target (visual_test, 35 showcases)
├── tests/                      Pure unit tests (math/ecs/render_graph/scene_io/commands/anim/physics/terrain/
│   │                           streaming/runtime/editor/volumetric/probe/taa/frustum/gpu_cull/parallel_record/
│   │                           shader_graph/runtime_material/roll_game/audio/introspect/live_editor/text/...) + rhi_smoke
│   ├── golden/metal/           The 35 committed Metal goldens
│   ├── golden/introspect/      The engine-state JSON golden (default_scene.json)
│   └── golden/audio/           The audio WAV golden (scene.wav, from --audio-render)
├── scripts/verify.ps1          Cross-platform gate: Windows ctest + JSON + audio goldens + Mac 35-golden compare
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
ctest --preset windows-msvc-debug          # 34 tests

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
./build-metal/visual_test out.png  # default scene (one of 35 showcase flags)
```

### Full cross-platform verification (one command)

```powershell
scripts\verify.ps1
```

Runs the Windows/Vulkan ctest (plus the engine-state JSON-golden and audio-WAV-golden byte matches) locally and drives the bench Mac over SSH to build `metal_headless` once and golden-compare **all 35** Metal goldens at threshold `0.0`. Prints a per-golden DIFF table and an overall `VERIFY: PASS/FAIL`. (`-SkipWindows` / `-SkipMac` run one half.)

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
| volumetric fog      | `--volumetric-shot out.bmp`        | `--volumetric out.png`             |
| reflection/irradiance probes | `--probe-shot out.bmp`    | `--probe out.png`                  |
| temporal AA (TAA)   | `--taa-shot out.bmp`               | `--taa out.png`                    |
| frustum culling     | `--cull-shot out.bmp`              | `--cull out.png`                   |
| GPU-driven culling  | `--gpu-cull-shot out.bmp`          | `--gpu-cull out.png`               |
| multithreaded record| `--mt-shot out.bmp [--workers N]`  | `--mt out.png`                     |
| material graph      | `--material-shot out.bmp`          | `--material out.png`               |
| material (live compile) | `--material-live-shot out.bmp [mat.json]` | `--material2 out.png`     |
| multi-material scene | `--material-multi-shot out.bmp`   | `--material-multi out.png`         |
| material normal-map node | `--material-normal-shot out.bmp` | `--material-normal out.png`      |
| scene / asset streaming | `--stream-shot out.bmp`        | `--stream out.png`                 |
| procedural terrain  | `--terrain-shot out.bmp`           | `--terrain out.png`                |
| game sample         | `--game-shot out.bmp`              | `--game out.png`                   |
| text / HUD overlay  | `--hud-shot out.bmp`               | `--hud out.png`                    |
| game + score HUD    | `--game-hud-shot out.bmp`          | `--game-hud out.png`               |
| audio render (WAV)  | `--audio-render out.wav`           | *(pure hf_core; same bytes)*       |
| engine-state JSON   | `--introspect out.json`            | *(pure hf_core; same bytes)*       |
| live viewport       | `--fly` (WASD + mouse-look)        | `--fly` (windowed Metal viewport)  |

---

## Repository

`github.com/hassard0/Hazard-Forge`

---

## Roadmap

- ~~Windowed Metal present loop for an interactive macOS viewport~~ — **shipped** (native Cocoa `--fly` viewport from a `CAMetalLayer*`; build-verified, manual on-screen confirmation).
- Continued editor work (multi-select, undo/redo surfaced in the UI, asset browser).
- Broader physics (boxes/convex, joints) and animation (state machines, IK).
- Real-time / dynamic GI beyond the baked reflection-probe path.
```
