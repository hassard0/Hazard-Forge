# Hazard Forge

**A C++20 game engine with a modern Vulkan + Metal renderer and a fully deterministic, fixed-point simulation
stack — every feature proven byte-for-byte across platforms by golden-image tests that must diff `0.0000`.**

Build cross-platform games where the physics, animation, and AI produce *the same bytes on every machine* — the
foundation for rollback netcode, frame-perfect replays, and CI you can actually trust. One engine, two GPU
backends, no per-platform drift.

> **Status:** active development — an in-progress engine built slice by slice, with two theses already proven on
> real hardware (a Vulkan RTX GPU on Windows and Apple Silicon / M4 on macOS):
> 1. **One engine, two GPU backends, golden-verified.** The same engine code renders a full PBR / image-based-lit /
>    post-processed scene and a modern GPU-driven pipeline on both Vulkan and Metal — from a single shader source.
> 2. **Determinism as a first-class feature.** A fixed-point (Q16.16) physics / animation / AI stack produces
>    *byte-identical* state on every platform and replays from inputs alone.
>
> Backed by **252 deterministic golden-image regression tests** (each diffs `0.0000` cross-vendor), three byte-exact
> data goldens (engine-state JSON, material-graph JSON, audio WAV), a **118-test** suite that runs clean under
> AddressSanitizer, and a graphics-validation-clean (core + synchronization) invariant across every showcase.

---

## Why build on Hazard Forge

- **Determinism you can ship on.** Every simulation — rigid bodies, cloth, fluid, granular media, fracture,
  ragdolls, vehicles, crowds, convex contacts — is integer fixed-point end to end. Feed two machines the same
  inputs and they compute the same world, bit for bit. That is the hard prerequisite for **lockstep and rollback
  netcode**, deterministic **replays**, and reproducible automated tests — normally the most painful thing to
  retrofit, here built into the foundation.

- **True cross-platform parity, not "close enough."** A thin Rendering Hardware Interface (RHI) seam puts Vulkan
  and Metal behind one pure C++ interface, and the Metal shaders are *generated from the same HLSL* at build time.
  Every rendering feature is verified against a committed reference image that must match to the byte. There is no
  hand-written second backend to drift.

- **Verifiable by construction.** Headless capture is the primary path: every capability ships as a golden-image
  regression test. A single command runs the whole cross-platform gate and prints `VERIFY: PASS/FAIL`. If shared
  code changes a single pixel anywhere, a specific test catches it.

- **Automation- and agent-friendly.** The engine emits a machine-readable JSON description of its own state and a
  command manifest (`--introspect`), runs entirely headless, and has zero hidden global state in its simulation
  core — so tooling, CI, and automated agents can drive and observe it reliably.

- **Focused and legible.** It is built in small, reviewable slices; the simulation models are header-only and
  unit-tested; nothing in the engine layer leaks a backend symbol. You can read a subsystem end to end.

---

## At a glance

| | |
| --- | --- |
| **Language / std** | C++20 |
| **GPU backends** | Vulkan 1.3 (Windows) · Apple Metal (macOS, Apple Silicon) — one HLSL source → SPIR-V + MSL |
| **Determinism** | Q16.16 fixed-point simulation, bit-identical CPU / Vulkan / Metal, lockstep + rollback replayable |
| **Verification** | 252 golden images @ `DIFF 0.0000` · 125-test suite · AddressSanitizer-clean · graphics-validation-clean |
| **Footprint** | header-only simulation models, slice-by-slice growth, zero backend symbols above the RHI seam |
| **Repo** | `github.com/hassard0/Hazard-Forge` |

---

## Capabilities

### Real-time rendering (Vulkan + Metal, every feature golden-verified)

A full modern renderer — and crucially, *the identical image* on both backends, proven to the byte.

- **Physically based shading & lighting** — metallic-roughness PBR (base color, metallic-roughness, tangent-space
  normal, emissive, ambient occlusion); procedural and HDR-environment image-based lighting; clustered / Forward+
  light culling that scales to hundreds of lights (proven identical to brute-force shading).
- **Shadows** — directional PCF, cascaded shadow maps, perspective spot shadows, omnidirectional point-light cube
  shadows, and screen-space contact shadows; plus **sparse clipmap shadow caching** that renders shadow pages
  on demand for very large light footprints.
- **Global illumination** — screen-space GI delivered as the full `raw → edge-preserving denoise → temporal
  accumulation` pipeline, baked reflection/irradiance probes, and a **dynamic diffuse GI** path built on an
  irradiance-probe field (probe ray-trace → radiance capture → spherical-harmonic encode → interpolation →
  probe occlusion → multi-bounce → composite), each stage holding a GPU-equals-CPU bit-exact compute proof.
- **Reflections** — screen-space reflections, box-projected cubemap probes, dynamic cubemap-capture probes, and
  planar mirror reflections.
- **Atmosphere & volumetrics** — Gerstner-wave water, raymarched volumetric clouds with ground cloud-shadows, 2D
  light shafts, and a true 3D volumetric fog volume (sun single-scatter + per-cell light injection + shadowed fog).
- **Post-processing** — HDR bloom, two ambient-occlusion methods (hemisphere SSAO + horizon-search GTAO),
  subsurface scattering, depth of field, motion blur, a data-driven post stack (tonemap / color-grade / chromatic
  aberration / vignette / grain), histogram-based auto-exposure, temporal anti-aliasing, and adaptive sharpening.
- **Surface & geometry detail** — parallax occlusion mapping, order-independent transparency, screen-space
  projected decals, and GPU instancing.
- **GPU-driven pipeline** — frustum + Hi-Z occlusion culling, multi-draw-indirect batching, bindless textures, and
  a fully-GPU-driven *culled* combined pass (compute cull → ordered compaction → one indirect draw + one bindless
  bind), each proven byte-identical to the per-object reference.
- **GPU virtual geometry** — meshlet-based continuous cluster level-of-detail, GPU cluster culling + Hi-Z, a
  visibility buffer with deferred material resolve, and a compute software rasterizer for sub-triangle detail.
- **Streaming virtual textures** — sample a texture larger than VRAM via feedback-driven, sparse on-demand paging
  (feedback → allocate → page generation → sample → cache).
- **GPU isosurface meshing** — integer marching cubes (classify → count → emit → interpolate → render → smooth
  normals) for real-time procedural and destructible geometry.
- **Engine foundations** — a render graph with an automatic resource-state barrier solver (synchronization-
  validation clean), multithreaded command recording (byte-identical 1-vs-N workers), a data-driven material /
  shader graph with live runtime authoring and JSON/Graphviz introspection, and an own column-major math library.

Plus the services a game needs: glTF scene import, GPU skeletal animation + blending + a cross-fade state machine,
a deterministic integer audio mixer (byte-exact WAV), a deterministic networking stack (replication → lossy
transport with interpolation → client prediction + reconciliation), distance-based scene/terrain streaming with
LOD, a text/HUD layer, a CPU VFX emitter, a playable game sample, and a docked editor shell.

### Deterministic simulation (the headline)

Above the renderer sits a suite of **fixed-point (Q16.16) simulations that are bit-identical across CPU, Vulkan,
and Metal and replayable from inputs alone**. This is the capability that is normally hardest to get and that most
engines simply do not offer: a *whole-simulation* determinism guarantee strong enough to run lockstep multiplayer
and rollback netcode on.

How it holds up: integer math throughout (no floating point, no RNG, no clock in the simulation loop); every
ordering decision pinned (fixed broadphase / solver / tie-break order); and GPU compute kernels that mirror the
CPU reference exactly, proven equal by a tolerance-free byte comparison. Integer-only kernels run natively on both
backends to a strict zero-differing-pixel bar; kernels needing 64-bit intermediates run as a Vulkan GPU pass plus
a byte-identical CPU reference.

Each capability is a full arc — `core dynamics → spatial structure → solver → the distinguishing physics →
lockstep/rollback proof → lit 3D render` — and lands bit-identical on both backends:

| Capability | What you get |
| --- | --- |
| **Rigid-body physics** (`hf::sim::fpx`) | A fixed-point rigid-body integrator + broadphase + 6-DOF contact solver, **proven lockstep- and rollback-replayable** across platforms — the deterministic core every other simulation reuses |
| **Navigation & pathfinding** (`hf::nav`) | GPU navmesh generation (span-raster → walkable filter → region partition → contour → polygonize) feeding a **deterministic integer A\*** |
| **Cloth** (`hf::sim::cloth`) | Position-based-dynamics cloth — a deterministic deformable, bit-identical and replayable |
| **Fluid** (`hf::sim::fluid`) | GPU position-based fluids (density-constraint solve), deterministic and replayable |
| **Granular media / sand** (`hf::sim::grain`) | Dry granular dynamics with Coulomb friction and angle-of-repose |
| **Rigid ↔ fluid coupling** (`hf::sim::couple`) | A rigid body buoyed, dragged by, and displacing the fluid in one world |
| **Rigid ↔ granular coupling** (`hf::sim::cgrain`) | A body sinking into, supported by, and parting a frictional sand bed |
| **Granular ↔ fluid coupling** (`hf::sim::cgf`) | Wet sand / mud / slurry — two-phase particle↔particle interaction |
| **Fracture & destruction** (`hf::sim::fract`) | Voronoi cell decomposition → bonded-cluster break → falling rubble, bit-exact and replayable |
| **Articulated ragdoll** (`hf::sim::joint`) | Ball / hinge / cone joints over the rigid-body solver, driving the GPU skinning path — a replayable ragdoll |
| **Vehicles** (`hf::sim::vehicle`) | Suspension springs + wheel hinges + Coulomb traction, rollback-netcode-ready |
| **Active ragdoll / physical animation** (`hf::sim::active`) | Angular pose-drive blending physics with animation clips, deterministically |
| **Crowds** (`hf::sim::boids`) | Grid-hashed steering + flocking + pathfinding-corridor following, lockstep-replayable at scale |
| **Convex rigid-body contacts** (`hf::sim::convex`) | Separating-axis test → contact manifold → inertia-tensor angular impulse → settling stacks → lockstep → lit render: boxes that tumble, stack, and interlock byte-for-byte |

Each capability proves its determinism as a hard test — *two peers fed only an input stream converge byte-identical,
and a rollback re-simulates from a snapshot bit-for-bit* — and ships a lit 3D render as the visual payoff.

---

## How it compares to Unreal Engine 5

A capability comparison — not a verdict. Unreal Engine 5 is a vast, production-proven engine with content
tooling, an asset ecosystem, and a feature surface Hazard Forge does not attempt to match. Hazard Forge instead
makes one deliberate, uncommon set of guarantees its whole reason for being:

| Capability | Unreal Engine 5 (typical) | Hazard Forge |
| --- | --- | --- |
| Simulation determinism | Floating-point; varies run-to-run and machine-to-machine | Integer fixed-point; bit-identical every run and every platform |
| Cross-platform reproducibility | Not guaranteed at the byte level | Byte-identical CPU ↔ Vulkan ↔ Metal, proven by 252 golden images |
| Lockstep / rollback netcode | Hard — float simulations diverge across peers | Built in — every simulation replays from inputs; rollback re-sims bit-for-bit |
| Scope of determinism | Mostly gameplay logic | The whole simulation stack: rigid bodies, cloth, fluid, granular media, fracture, ragdolls, vehicles, crowds, convex contacts |
| Headless / CI verification | Manual and limited | First-class: every feature is a golden-image regression that must diff `0.0000` |
| Renderer portability | Multiple backends, engine-internal | One engine over a clean RHI seam; Vulkan + Metal from a single HLSL source |
| Footprint & legibility | Very large | Focused, slice-by-slice, header-only simulation models you can read end to end |

If you need a deterministic, lockstep-ready, cross-platform-reproducible simulation foundation — for competitive
multiplayer, replays, automated testing, or research — that is exactly what Hazard Forge is built to give you. If
you need a turnkey AAA content pipeline today, a mature engine is the pragmatic choice.

---

## How correctness is verified

- **Golden images (252).** Each is one headless `visual_test` flag compared to its committed reference at
  threshold `0.0` — deterministic, two runs diff `0.0000`. The same scene on Windows/Vulkan must match the macOS/
  Metal reference cross-vendor.
- **Byte-exact data goldens (3).** The engine-state JSON (`--introspect`), the material-graph JSON, and the audio
  WAV (`--audio-render`) are matched byte for byte.
- **Unit tests (118).** A suite over the pure-C++ core — math, ECS, render graph, every header-only render and
  simulation model, codegen, networking, audio, editor — that runs clean under AddressSanitizer.
- **Graphics-validation-clean.** Every showcase runs under the core + synchronization validation layers with zero
  errors; the render graph's auto-inserted barriers are proven hazard-free.
- **One-command cross-platform gate.** `scripts/verify.ps1` runs the Windows/Vulkan tests + the data-golden byte
  matches, then drives a macOS machine over SSH to build the Metal target once and compare **all 252** references
  at `DIFF 0.0000`, printing a per-golden table and an overall `VERIFY: PASS/FAIL`.

---

## Building

### Windows — Vulkan (MSVC + Ninja + Conan 2)

Prerequisites: Visual Studio Build Tools 2022 (MSVC x64), CMake ≥ 3.25, Ninja, Conan 2, Vulkan SDK.

```powershell
# 1. Install dependencies and generate the CMake toolchain:
conan install . -of=build/windows-msvc-debug `
    -s build_type=Debug -s compiler.cppstd=17 `
    -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing

# 2. Configure + build + test (from a VS x64 developer shell):
cmake --preset windows-msvc-debug
cmake --build --preset windows-msvc-debug
ctest --preset windows-msvc-debug          # 125 tests

# 3. AddressSanitizer build of the pure-C++ core + tests:
conan install . -of=build/windows-msvc-asan `
    -s build_type=Debug -s compiler.cppstd=17 `
    -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing
cmake --preset windows-msvc-asan
cmake --build --preset windows-msvc-asan
ctest --preset windows-msvc-asan
```

### macOS — Metal headless (no Conan, no SDL, no full Xcode required)

`metal_headless/` is a standalone CMake project that compiles the real Metal backend + engine, generates Metal
shaders from the shared HLSL at build time, renders a showcase into an offscreen texture, and writes a PNG.
Command Line Tools suffice.

```sh
cmake -S metal_headless -B build-metal -G Ninja
cmake --build build-metal           # also generates the Metal shaders from the HLSL
./build-metal/visual_test out.png   # default scene; pass a showcase flag for others
```

### Full cross-platform verification (one command)

```powershell
scripts\verify.ps1
```

Runs the Windows/Vulkan tests (plus the engine-state-JSON and audio-WAV byte matches) locally and drives a macOS
machine over SSH to build the Metal target once and compare **all 252** references at threshold `0.0`. Prints a
per-golden DIFF table and an overall `VERIFY: PASS/FAIL`. (`-SkipWindows` / `-SkipMac` run one half.)

---

## Running the showcases

The same scenes render on both backends. On **Vulkan** (the Windows `hello_triangle` sample) each is a
`--<name>-shot out.bmp` headless capture (or `--fly` for the live navigable viewport); on **Metal**
(`metal_headless/visual_test`) each is the matching `--<name> out.png` flag. There are ~190 showcase flags — the
authoritative, always-current list is machine-readable:

```sh
# the engine emits its own showcase / command / feature manifest as JSON:
hello_triangle.exe --introspect manifest.json

# verify.ps1 enumerates every reference image + its flag; its per-golden table is the live index.
```

A few representative flags: `--pbr` (PBR materials), `--ssgi-temporal` (the GI pipeline), `--gpudriven` (one
indirect draw + bindless), `--fpx-lockstep` (deterministic physics rollback), `--cloth-render` / `--fluid-render`
/ `--grain-render` (deterministic deformables & particles), `--fract-render` (destruction), `--joint-render`
(ragdoll), `--vehicle-render`, `--boids-render` (crowds), `--convex-stack` (settling rigid bodies),
`--audio-render out.wav`, `--introspect out.json`, and `--fly` (live viewport, windowed on both Vulkan and Metal).

---

## Architecture

Layered, all above the seam:

1. **HAL** (`engine/hal/`) — window + Vulkan surface creation.
2. **RHI seam** (`engine/rhi/`) — pure C++ interfaces (`IRHIDevice`, `ICommandBuffer`, `IPipeline`, `IBuffer`,
   `ITexture`, `IRenderTarget`, `ISwapchain`, compute). Zero backend symbols — no `vk*` or `MTL*` leaks above it.
3. **Backends** — Vulkan (`engine/rhi_vulkan/`, vk-bootstrap + VMA + 1.3 dynamic rendering + descriptor-indexing
   bindless) and Metal (`engine/rhi_metal/`, Obj-C++/ARC, runtime shader compile).
4. **Engine modules** — `render/` (render graph + barrier solver + every header-only render model), `sim/` (the
   deterministic fixed-point simulations: `fpx` / `cloth` / `fluid` / `grain` / `couple*` / `fract` / `joint` /
   `vehicle` / `active` / `boids` / `convex`), `nav/` (deterministic navmesh + A*), `scene/`, `terrain/`,
   `asset/`, `anim/`, `physics/`, `material/`, `ui/`, `audio/`, `vfx/`, `net/`, `game/`, `runtime/`, `editor/`,
   `debug/`. All depend only on `rhi/` + `math/`; the backend-agnostic subset compiles into `hf_core` for the
   sanitized unit tests.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the seam design, descriptor model, frame structure, the
shared HLSL→Metal toolchain, the deterministic-simulation conventions, and per-backend notes.

**Shared shader includes + the time channel.** Shaders that share math pull it from a shared `.hlsli` include (no
separate compile entry — both DXC and the `hf_gen_msl` Metal path preprocess it inline). `shaders/procedural_sky.hlsli`
holds `HFSkyColor(dir, lightDir)`, the **single source of truth** for the procedural sky: the sky pass, the lit pass's
IBL reflection (`SkyColor(R)` — what metals/glass reflect), and the material-graph IBL all call it, so retuning the sky
updates every reflection in lock-step. For animation, `FrameData.skyParams` carries a per-frame **time channel** —
`skyParams.z = time` (seconds, from `runtime::FixedTimestep`), `skyParams.w = frameIndex`. A shader animates by reading
`float time = f.skyParams.z;`; existing showcases pass `time = 0` so their goldens stay byte-identical. The worked
example is `shaders/sky_animated.frag.hlsl` (`--sky-animated-shot` / `--sky-animated`).

### Repo layout

```
hazard-forge/
├── CMakeLists.txt              Top-level C++20 build; HF_SANITIZE opt-in
├── CMakePresets.json           windows-msvc-debug / -release / -asan / macos-arm64-debug
├── conanfile.py                SDL3, vk-bootstrap, VMA, vulkan-headers/loader, validation layers (debug)
├── engine/
│   ├── rhi/                    THE SEAM — pure interfaces, zero backend symbols
│   ├── rhi_vulkan/             Vulkan backend (+ descriptor-indexing bindless array)
│   ├── rhi_metal/              Metal backend (Obj-C++/ARC, runtime shader compile)
│   ├── render/                 Render graph + barrier solver + all header-only render models
│   ├── sim/                    Deterministic fixed-point sims: fpx/cloth/fluid/grain/couple*/fract/joint/vehicle/active/boids/convex
│   ├── nav/                    Deterministic navmesh + integer A* pathfinding
│   ├── scene/ terrain/ asset/ anim/ physics/ material/ ui/ audio/ vfx/ net/ game/ runtime/ editor/ debug/
│   └── math/ hal/
├── shaders/                    Shared HLSL → SPIR-V & Metal (+ generated/ material-codegen output)
│                               Shared includes: pbr_core.hlsli (PBR core), procedural_sky.hlsli (HFSkyColor — one sky for the sky pass AND IBL reflections)
├── tools/                      material_codegen: build-time HLSL generator from *.mat.json graphs
├── mac_window/                 Native Cocoa entry: windowed Metal viewport from a CAMetalLayer*
├── samples/hello_triangle/     Vulkan sample: every showcase via --*-shot capture + --fly + --introspect
├── metal_headless/             Standalone no-Conan/no-SDL Metal target (visual_test, every showcase)
├── tests/                      Pure unit tests + golden/{metal,introspect,material,audio}
├── scripts/verify.ps1          Cross-platform gate: Windows tests + data goldens + Mac 252-image compare
├── ci/                         Staged GitHub Actions workflow (see ci/README.md)
└── docs/                       ARCHITECTURE.md + per-slice specs/plans
```

---

## Roadmap

Recent work has pushed past rendering breadth into two depth pillars: **GPU-compute rendering** (dynamic GI,
virtual geometry, sparse shadow caching, streaming virtual textures, GPU isosurface meshing) and **deterministic
fixed-point simulation** (the physics / coupling / destruction / ragdoll / vehicle / crowd / convex-contact
capabilities above). Already shipped: a windowed Metal viewport; deterministic physics across boxes, joints,
vehicles, and ragdolls; the full screen-space and dynamic GI pipeline; and a deterministic networking stack.

Open directions:

- **Real sockets** (UDP/TCP) over the existing deterministic replication / transport / prediction API.
- **World-space / reprojected temporal GI** under camera motion.
- **Simulation breadth** — friction refinements, general convex hulls beyond boxes, multi-threaded solves, and
  composing the deterministic-contact model back into the rigid-body simulations.
- **Editor depth** — multi-select, surfaced undo/redo, an asset browser.

---

## Repository

`github.com/hassard0/Hazard-Forge`
