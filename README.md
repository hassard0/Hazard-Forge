# Hazard Forge

A C++20 cross-platform game engine built around a thin, explicit **Rendering Hardware Interface (RHI) seam** that
renders natively on **Vulkan (Windows)** and **Apple Metal (macOS, Apple Silicon)** from one codebase — and, above
that, a **deterministic, fixed-point simulation suite** whose results are *bit-identical across CPU, Vulkan, and
Metal* and *lockstep/rollback-replayable*. Every capability is verified headlessly against committed golden images
that must diff `0.0000`.

> **Status:** active development — an in-progress engine, not a finished product, built slice by slice. The two
> theses are both proven on real hardware:
> 1. **One engine, two GPU backends, golden-verified.** The same engine code renders a full PBR / IBL /
>    post-processed scene — and a modern GPU-driven pipeline — on a Vulkan RTX GPU (Windows) and on Apple Metal (M4,
>    macOS).
> 2. **Deterministic simulation beyond float engines.** A Q16.16 fixed-point physics/animation/AI stack produces
>    *byte-identical* state on every platform and is replayable from inputs alone — the one guarantee UE5's float
>    Chaos cannot make.
>
> Verified by **188 deterministic golden-image regression tests** (each diffs `0.0000` cross-vendor), three
> byte-exact non-image goldens (engine-state JSON, material-graph JSON, audio WAV), a **105-test** ctest suite that
> runs clean under AddressSanitizer, and a Vulkan-validation-clean (core + synchronization) invariant across every
> showcase.

---

## The bet

**One engine layer above a clean seam, two GPU backends below it, verified headlessly by golden images.** No Vulkan
symbol (`vk*`) leaks above `engine/rhi/`; no Metal symbol (`MTL*`) leaks above `engine/rhi_metal/`. The seam
invariant is the architectural spine.

Two consequences make the project unusual:

- **Every rendering feature is simultaneously a Vulkan implementation and a specification** the Metal backend must
  match — proven by a committed golden image that must diff `0.0000`. The Metal shaders are *generated from the
  shared HLSL* at build time (HLSL → SPIR-V via `glslc` → MSL via `spirv-cross`), so there is no hand-written MSL to
  drift.
- **Every simulation feature is integer / fixed-point end to end**, so the same tick produces the same bytes on a
  CPU reference, a Vulkan compute dispatch, and a Metal one. That makes the sims not just cross-platform-identical
  but *lockstep-* and *rollback-replayable* — the foundation for deterministic netcode.

It is also **built for agentic development**: headless capture is the primary path, the engine emits a
machine-readable JSON description of its own state (`--introspect`), and a single command (`scripts/verify.ps1`)
runs the whole cross-platform gate.

---

## At a glance

| | |
| --- | --- |
| **Language / std** | C++20 |
| **GPU backends** | Vulkan 1.3 (Windows, primary) · Apple Metal (macOS, Apple Silicon) |
| **Verification** | 188 golden images @ `DIFF 0.0000` · 105 ctest tests · ASan-clean · Vulkan-validation-clean |
| **Shaders** | one HLSL source → SPIR-V (Vulkan) + MSL (Metal), generated at build time |
| **Determinism** | Q16.16 fixed-point sim, bit-identical CPU/Vulkan/Metal, lockstep + rollback replayable |
| **Repo** | `github.com/hassard0/Hazard-Forge` |

---

## Pillar 1 — Real-time rendering (Vulkan + Metal)

A full modern renderer, every feature golden-verified byte-for-byte against the Metal backend. Grouped by area:

- **PBR & lighting** — glTF metallic-roughness Cook-Torrance shading (base color / metallic-roughness / tangent-space
  normal / emissive / AO); procedural **and** HDR-equirectangular image-based lighting; **clustered / Forward+**
  light culling scaling to hundreds of lights (proven byte-identical to brute force).
- **Shadows** — directional PCF, **cascaded shadow maps**, perspective **spot** shadows, omnidirectional **point-light
  cube** shadows, and screen-space **contact shadows**.
- **Global illumination** — screen-space GI as the full `raw → edge-preserving bilateral denoise → temporal
  accumulation` trilogy; baked reflection/irradiance probes; and a **DDGI dynamic-GI pillar** (probe-grid ray-trace →
  per-probe radiance capture → 3rd-order SH encode → interpolation → probe occlusion → multi-bounce → GI composite),
  each slice holding a **GPU==CPU bit-exact** compute proof.
- **Reflections** — screen-space reflections, **box-projected** cubemap probes, **dynamic cubemap-capture** probes,
  and **planar** mirror reflections.
- **Atmosphere & volumetrics** — Gerstner-wave water, raymarched volumetric clouds + ground cloud-shadows, 2D light
  shafts, and a true **3D froxel volumetric fog** (sun single-scatter + per-froxel clustered-light injection +
  CSM-gated volumetric shadows).
- **Post-processing** — HDR bloom, SSAO + **ground-truth AO** (horizon-search), **subsurface scattering**, depth of
  field, motion blur, a data-driven post stack (tonemap / color-grade / chromatic aberration / vignette / grain),
  **auto-exposure** (integer-histogram eye adaptation), **TAA**, and **contrast-adaptive sharpening**.
- **Surface detail** — parallax occlusion mapping, order-independent transparency (Weighted Blended), screen-space
  projected decals, GPU instancing.
- **GPU-driven pipeline** — frustum + **Hi-Z occlusion** culling, **multi-draw-indirect** batching, **bindless**
  textures, and a fully-GPU-driven *culled* combined pass (compute cull → ordered compaction → one indirect draw +
  one bindless bind), each proven byte-identical to the per-object reference.
- **Flagship GPU-compute arcs** (each multi-slice, bit-exact or golden-verified on both backends):
  - **Nanite-style virtual geometry** — meshlet cluster-LOD, cluster cull + Hi-Z, a **visibility buffer** with
    deferred resolve, and a compute **software rasterizer** (packed depth|id atomic-min).
  - **Virtual Shadow Maps** — clipmap mark → render → sample → cache.
  - **Runtime Virtual Texturing** — feedback → allocate → page-gen → sample → cache.
  - **GPU Marching Cubes** — integer isosurface meshing (classify → count → emit → interp → render → smooth
    normals), real-time procedural/destructible geometry.
- **Foundations** — own `hf::math` (Vec3/Mat4/Quat, column-major, Vulkan `[0,1]` depth, CPU-side Metal NDC flip);
  a render graph with an **automatic resource-state barrier solver** (Vulkan-synchronization-validation clean);
  **multithreaded** command recording (byte-identical 1-vs-N workers); a data-driven **material/shader graph** with
  live runtime authoring + JSON/Graphviz introspection.

Plus engine services: glTF scene-graph import, GPU skeletal animation + blending + a cross-fade state machine, a
deterministic integer **audio mixer** (byte-exact WAV), a deterministic **networking trilogy** (replication →
seeded lossy transport + interpolation → client prediction + reconciliation), distance-based **scene/terrain
streaming** with LOD, a baked-font **text/HUD** layer, a CPU **VFX** emitter, a playable game sample, and a docked
**ImGui editor** shell (hierarchy / inspector / stats / viewport + live edit-ops → scene round-trip).

---

## Pillar 2 — Deterministic simulation (the differentiator)

Above the renderer sits a suite of **fixed-point (Q16.16) simulations** that are *bit-identical across CPU, Vulkan,
and Metal* and *replayable from inputs alone*. Where a typical engine's physics is float and non-deterministic
(UE5's Chaos), Hazard Forge's sims are byte-reproducible and **lockstep + rollback** ready — the missing primitive
for true cross-platform deterministic netcode.

How it holds: integer Q16.16 math throughout (no `<cmath>`, no RNG, no clock in the sim loop); every ordering
decision pinned (fixed broadphase / solver / tie-break order); GPU compute shaders that copy the CPU reference
*verbatim*, proven equal by a tolerance-free `memcmp`. Integer-only kernels run natively on both backends (strict
zero-differing-pixel); kernels needing 64-bit intermediates run as a Vulkan GPU pass plus a byte-identical Metal
CPU reference.

Each flagship is a 6-slice arc — `core physics → spatial structure → solve → the new physics → lockstep/rollback →
lit 3D render` — and lands bit-identical on both backends:

| # | Flagship | What it is |
| --- | --- | --- |
| 6 | **Fixed-point physics** (`hf::sim::fpx`) | Q16.16 rigid-body integrator + broadphase + 6-DOF contact solver; **proven lockstep + rollback** cross-platform — the deterministic core every later sim reuses |
| 7 | **GPU navmesh + pathfinding** (`hf::nav`) | Recast/Detour-style span-raster → walkable filter → watershed regions → contour → polygonize → **deterministic integer A\*** |
| 8 | **Cloth** (`hf::sim::cloth`) | Q16.16 position-based-dynamics cloth (the first deterministic deformable; UE5's Chaos Cloth is float) |
| 9 | **Fluid** (`hf::sim::fluid`) | deterministic GPU position-based fluids (density-constraint solve) |
| 10 | **Granular / sand** (`hf::sim::grain`) | deterministic granular media with dry Coulomb friction / angle-of-repose |
| 11 | **Rigid↔fluid coupling** (`hf::sim::couple`) | a rigid body buoyed/dragged by + displacing the fluid, in one Q16.16 world |
| 12 | **Rigid↔grain coupling** (`hf::sim::cgrain`) | a body sinking into + supported by + parting a frictional sand bed |
| 13 | **Grain↔fluid coupling** (`hf::sim::cgf`) | wet sand / mud / slurry — the first particle↔particle two-phase coupling |
| 14 | **Fracture / destruction** (`hf::sim::fract`) | Voronoi cell decomposition → bonded-cluster break → falling rubble (UE5's marquee Chaos feature, made bit-exact + replayable) |
| 15 | **Articulated ragdoll** (`hf::sim::joint`) | ball / hinge / cone joints over the fixed-point solver → a bit-exact lockstep-replayable ragdoll that drives the GPU skinning path |
| 16 | **Vehicle physics** (`hf::sim::vehicle`) | suspension springs + wheel hinges + Coulomb traction; rollback-netcode-ready (UE5's Chaos Vehicles are float) |
| 17 | **Active ragdoll / physical animation** (`hf::sim::active`) | angular pose-drive blending physics with anim clips (UE5's Physical Animation Component, made deterministic) |
| 18 | **GPU crowds** (`hf::sim::boids`) | grid-hashed steering + 3-rule flocking + A\*-corridor path-following, lockstep-replayable |
| 19 | **Convex rigid-body contacts** (`hf::sim::convex`) | box-box SAT → contact manifold → inertia-tensor angular impulse → settling stack → lockstep → lit render *(in progress)* |

Each lands its determinism headline as a hard proof — e.g. *two peers fed only an input stream converge
byte-identical, and a rollback re-sims from a snapshot bit-for-bit* — and a lit 3D render capstone as the visual
payoff.

---

## How correctness is verified

- **Golden images (188).** Each is one `metal_headless/visual_test` flag compared to `tests/golden/metal/<name>.png`
  at threshold `0.0` — deterministic, two runs diff `0.0000`. The same scene rendered on Windows/Vulkan must match
  the committed Mac/Metal golden cross-vendor.
- **Byte-exact non-image goldens (3).** The engine-state JSON (`--introspect`), the material-graph introspection
  JSON, and the audio WAV (`--audio-render`) are matched byte-for-byte.
- **Unit tests (105).** A ctest suite over the pure-C++ core (math, ECS, render-graph, every header-only render +
  sim model, codegen, net, audio, editor). Runs clean under MSVC AddressSanitizer (`windows-msvc-asan`).
- **Vulkan-validation-clean.** Every showcase runs under the Khronos core + synchronization validation layers with
  zero `VUID-*` / `SYNC-HAZARD-*` / `UNASSIGNED-*` / `[ERROR]` lines; the render graph's auto-inserted barriers are
  proven hazard-free.
- **One-command cross-platform gate.** `scripts/verify.ps1` runs the Windows/Vulkan ctest + the JSON/audio byte
  matches, then drives the bench Mac over SSH to build `metal_headless` once and golden-compare **all 188** Metal
  goldens at `DIFF 0.0000`, printing a per-golden table and an overall `VERIFY: PASS/FAIL`.

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
ctest --preset windows-msvc-debug          # 105 tests

# 3. AddressSanitizer build of the pure-C++ core + tests:
conan install . -of=build/windows-msvc-asan `
    -s build_type=Debug -s compiler.cppstd=17 `
    -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing
cmake --preset windows-msvc-asan
cmake --build --preset windows-msvc-asan
ctest --preset windows-msvc-asan
```

### macOS — Metal headless (no Conan, no SDL, no full Xcode required)

`metal_headless/` is a standalone CMake project that compiles the real Metal backend + engine, generates MSL from
the shared HLSL at build time, renders a showcase into an offscreen texture, and writes a PNG. Command Line Tools
suffice.

```sh
cmake -S metal_headless -B build-metal -G Ninja
cmake --build build-metal           # also generates *.gen.metal from the HLSL
./build-metal/visual_test out.png   # default scene; pass a showcase flag for others
```

### Full cross-platform verification (one command)

```powershell
scripts\verify.ps1
```

Runs the Windows/Vulkan ctest (plus the engine-state JSON-golden and audio-WAV-golden byte matches) locally and
drives the bench Mac over SSH to build `metal_headless` once and golden-compare **all 188** Metal goldens at
threshold `0.0`. Prints a per-golden DIFF table and an overall `VERIFY: PASS/FAIL`. (`-SkipWindows` / `-SkipMac`
run one half.)

---

## Running the showcases

The same scenes render on both backends. On **Vulkan** (the Windows `hello_triangle` sample) each is a `--<name>-shot
out.bmp` headless capture (or `--fly` for the live navigable viewport); on **Metal** (`metal_headless/visual_test`)
each is the matching `--<name> out.png` flag. There are ~190 showcase flags — the authoritative, always-current list
is machine-readable:

```sh
# the engine emits its own showcase / command / feature manifest as JSON:
hello_triangle.exe --introspect manifest.json

# verify.ps1 enumerates every golden + its flag; its per-golden table is the live index.
```

A few representative flags: `--pbr` (glTF PBR), `--ssgi-temporal` (the GI trilogy), `--gpudriven` (one indirect
draw + bindless), `--fpx-lockstep` (deterministic physics rollback), `--cloth-render` / `--fluid-render` /
`--grain-render` (the deterministic deformable / particle sims), `--fract-render` (destruction), `--joint-render`
(ragdoll), `--vehicle-render`, `--boids-render` (crowds), `--convex-stack` (settling rigid bodies),
`--audio-render out.wav`, `--introspect out.json`, and `--fly` (live viewport, windowed on both Vulkan and Metal).

---

## Architecture

Layered, all above the seam:

1. **HAL** (`engine/hal/`) — SDL3 window + Vulkan surface creation.
2. **RHI seam** (`engine/rhi/`) — pure C++ interfaces (`IRHIDevice`, `ICommandBuffer`, `IPipeline`, `IBuffer`,
   `ITexture`, `IRenderTarget`, `ISwapchain`, compute). Zero backend symbols.
3. **Backends** — Vulkan (`engine/rhi_vulkan/`, vk-bootstrap + VMA + 1.3 dynamic rendering + descriptor-indexing
   bindless) and Metal (`engine/rhi_metal/`, Obj-C++/ARC, runtime MSL compile).
4. **Engine modules** — `render/` (render graph + barrier solver + every header-only render model), `sim/` (the
   deterministic fixed-point simulation flagships: `fpx` / `cloth` / `fluid` / `grain` / `couple*` / `fract` /
   `joint` / `vehicle` / `active` / `boids` / `convex`), `nav/` (deterministic GPU navmesh + A*), `scene/`,
   `terrain/`, `asset/`, `anim/`, `physics/`, `material/`, `ui/`, `audio/`, `vfx/`, `net/`, `game/`, `runtime/`,
   `editor/`, `debug/`. All depend only on `rhi/` + `math/`; the backend-agnostic subset compiles into `hf_core`
   for the sanitized unit tests.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the seam design, descriptor model, frame structure, the
shared HLSL→MSL toolchain, the deterministic-sim conventions, and per-backend notes.

### Repo layout

```
hazard-forge/
├── CMakeLists.txt              Top-level C++20 build; HF_SANITIZE opt-in
├── CMakePresets.json           windows-msvc-debug / -release / -asan / macos-arm64-debug
├── conanfile.py                SDL3, vk-bootstrap, VMA, vulkan-headers/loader, validation layers (debug)
├── engine/
│   ├── rhi/                    THE SEAM — pure interfaces, zero backend symbols
│   ├── rhi_vulkan/             Vulkan backend (+ vulkan_bindless: VK_EXT_descriptor_indexing array)
│   ├── rhi_metal/              Metal backend (Obj-C++/ARC, runtime MSL compile)
│   ├── render/                 RenderGraph + barrier solver + all header-only render models
│   ├── sim/                    Deterministic Q16.16 sims: fpx/cloth/fluid/grain/couple*/fract/joint/vehicle/active/boids/convex
│   ├── nav/                    Deterministic GPU navmesh + integer A* pathfinding
│   ├── scene/ terrain/ asset/ anim/ physics/ material/ ui/ audio/ vfx/ net/ game/ runtime/ editor/ debug/
│   └── math/ hal/
├── shaders/                    Shared HLSL → SPIR-V & MSL (+ generated/ material-codegen output)
├── tools/                      material_codegen: build-time HLSL generator from *.mat.json graphs
├── mac_window/                 SDL-free native Cocoa entry: windowed Metal viewport from a CAMetalLayer*
├── samples/hello_triangle/     Vulkan sample: every showcase via --*-shot capture + --fly + --introspect
├── metal_headless/             Standalone no-Conan/no-SDL Metal target (visual_test, every showcase)
├── tests/                      Pure unit tests + golden/{metal,introspect,material,audio}
├── scripts/verify.ps1          Cross-platform gate: Windows ctest + JSON/audio goldens + Mac 188-golden compare
├── ci/                         Staged GitHub Actions workflow (see ci/README.md)
└── docs/                       ARCHITECTURE.md + per-slice specs/plans
```

---

## Roadmap

Recent work has pushed well past the rendering breadth phase into two depth pillars: **GPU-compute rendering**
(DDGI dynamic GI, Nanite-style virtual geometry, virtual shadow maps, runtime virtual texturing, GPU marching
cubes) and **deterministic fixed-point simulation** (the physics / coupling / destruction / ragdoll / vehicle /
crowd / convex-contact flagships above). Shipped since the early roadmap: a windowed Metal `--fly` viewport;
broader physics (boxes/convex, joints, vehicles, ragdolls — all deterministic); the full SSGI + DDGI GI arc; and
the deterministic networking trilogy.

Open directions:

- **Real sockets** (UDP/TCP) over the existing deterministic replication / transport / prediction API.
- **World-space / reprojected temporal GI** under camera motion (beyond the static-camera SSGI accumulation).
- **Simulation breadth** — friction refinements, general convex hulls (beyond boxes), multi-threaded solves, and
  composing the deterministic-contact model back into the existing rigid-body flagships.
- **Editor depth** — multi-select, surfaced undo/redo, an asset browser.

---

## Repository

`github.com/hassard0/Hazard-Forge`
