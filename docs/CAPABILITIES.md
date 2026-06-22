# Hazard Forge — Capability Map

> **Why this doc exists.** Hazard Forge ships a large, deterministic, cross-platform engine, but much of its power is
> reached through **headless showcase flags** rather than a GUI, so it can read as a black box if you don't know the
> flag. This is the index: for each capability, *what ships*, *the flag to see it*, *the golden image*, *the
> ARCHITECTURE section*, and an *honest note on what's not done*.
>
> **How to run any showcase.** Each capability has a Vulkan flag (`--<name>-shot <out.png>`, on
> `samples/hello_triangle`) and a Metal flag (`--<name>`, on `metal_headless/visual_test`). Every flag in this doc has
> a committed reference render under `tests/golden/metal/` (byte-compared cross-platform at DIFF 0.0000 in
> `scripts/verify.ps1`). Run `hello_triangle --help` or read `scripts/verify.ps1`'s `$Goldens` table for the complete,
> always-current list (265 showcases). The determinism that underlies all of this — bit-identical Vulkan/Windows ==
> Metal/macOS, lockstep/rollback-replayable — is the moat described in `docs/ARCHITECTURE.md`.

## Rendering & lighting

| Capability | Ships? | See it (flag) | ARCHITECTURE / notes |
|---|---|---|---|
| Deferred + forward+ rendering, clustered lights | ✅ | `--clustered-lights`, `--cull`, `--clustered` | Forward+ clustered light culling, byte-identical to brute force |
| Shadows: CSM, spot, point-cube, contact | ✅ | `--csm`, `--spot`, `--point-shadow`, `--contact-shadows` | Shadow-atlas tiling |
| **Virtual Shadow Maps (single-texel-per-pixel, any distance)** | ✅ | `--vsm-pages`, `--vsm-atlas`, `--vsm-shadow`, `--vsm-cache` | Clipmap VSM flagship — *replaces the fixed cascade* (issue #14) |
| SSR / planar / SSGI reflections + GI | ✅ | `--ssr`, `--planar-reflection`, `--ssgi`, `--ssgi-temporal` | Screen-space + planar reflections |
| **Deterministic Lumen-class global illumination** | ✅ | `--gi1-probe` … `--gi6-hero`, plus DDGI `--probegi`/`--probe-capture`/`--probe-sh` | RT-traced integer-SH irradiance probe volume + multi-bounce + Chebyshev occlusion + Cornell color-bleed (issue #10) |
| Volumetrics, fog, clouds, cloud shadows | ✅ | `--volumetric`, `--froxel-fog`, `--clouds`, `--cloud-shadows` | Raymarched volumetric clouds + froxel fog (issue #12; atmospheric *scattering* — Rayleigh/Mie — is a genuine future add) |
| PBR + IBL, GTAO, SSS, bloom, DoF, TAA, motion blur, CAS, auto-exposure, color grade | ✅ | `--pbr`, `--ibl`, `--gtao`, `--sss`, `--bloom`, `--dof`, `--taa`, `--motion-blur`, `--cas`, `--auto-exposure`, `--color-grade` | Full post stack |
| Animated shaders (time-driven sky/water/foliage/VFX) | ✅ | `--sky-animated`; `FrameData.skyParams.zw = (time, frameIndex)` | The per-frame time channel (issue #5); shared `procedural_sky.hlsli` (issue #4) |

## Hardware ray tracing (deterministically reconciled) — issues #7, #13

✅ **Ships as a full RHI accel-structure seam.** Flagship #28 added `IAccelerationStructure` / BLAS / TLAS + inline ray
query over `VK_KHR_ray_query` (Vulkan) and `MTLAccelerationStructure` + `intersection_query` (Metal) — *not* a single
hint shader. The public path:

- `--rt1-trace` — the deterministic Q16.16 **software reference tracer** (the "ground-truth reference mode" of issue
  #13 — every HW path is validated to agree with it).
- `--rt2-query` (Vulkan HW inline ray query) / `--rt2-hwquery` (Metal HW intersection query) — real hardware RT on
  *both* vendors, reconciled to a bit-identical result.
- `--rt3-shadow`, `--rt4-reflect`, `--rt5-simrender`, `--rt6-hero` — RT shadows, reflections, sim-render, hero scene.

The moat: the HW BVH is used only as a candidate generator; our integer intersection owns the closest hit, so the
result is **bit-identical NVIDIA + Apple** — a deterministic RT reference no float RT engine offers. See ARCHITECTURE
"Hardware ray tracing, deterministically reconciled". *Genuine remaining work:* a temporal RT **denoiser** for noisy
1-spp paths (a screen-space denoiser exists for SSGI: `--ssgi-denoise`).

## Virtualized geometry (Nanite-style) — issue #9

✅ **The core ships:** a **visibility-buffer** renderer (`--visbuffer`), a **software rasterizer** for sub-pixel
triangles (`--swraster`, `--swraster-gpu`, `--swraster-resolve`), and **cluster-LOD** selection with no popping
(`--cluster-lod`, `--cluster-hiz`, `--cluster-cull`). Meshlet decomposition is Morton-ordered + deterministic. See
ARCHITECTURE's virtual-geometry sections. *Genuine remaining work:* **world-scale meshlet streaming** (paging clusters
from disk for billion-triangle scenes) — the core rasterizer + LOD are done; the streaming tier is a tracked
enhancement, an opportunistic roadmap item.

## Physics — Chaos-class, deterministic — issue #29

✅ **Ships broadly, and bit-identical/replayable in a way Chaos is not.** The deterministic fixed-point (Q16.16) sim
stack:

- **Cloth:** `--cloth-integrate` … `--cloth-render` (PBD cloth, lockstep-replayable).
- **Fluids:** `--fluid-integrate` … `--fluid-render` (position-based fluids).
- **Granular/sand:** `--grain-integrate` … `--grain-render` (Coulomb friction / angle-of-repose).
- **Destruction/fracture:** `--fract-cells` … `--fract-render` (Voronoi fracture + rigid-body rubble).
- **Rigid bodies + contacts:** convex GJK/EPA, hull friction + joints (`--hf1-points` … `--hf6-hull`), warm-started
  stacking, CCD, ragdoll, vehicle.
- **Two-way material coupling:** rigid↔fluid, rigid↔grain, grain↔fluid (mud/slurry/wet-sand).

Every one is bit-identical CPU/Vulkan/Metal AND lockstep/rollback-replayable. See ARCHITECTURE's deterministic-sim
sections. *Genuine remaining work:* **hair/strand** simulation (the one Chaos pillar not yet built) — a natural
extension of the existing PBD solvers.

## Particles / VFX — issue #19

🟡 **Partial.** A CPU particle/VFX emitter ships (`--vfx`, additive billboard fountain). A full **GPU-driven
Niagara-class** system (mesh emission, force fields, GPU collisions, a node graph) is a **genuine gap** — though the
deterministic GPU sim flagships (grain/fluid/cloth) already provide the GPU-particle *substrate*. Tracked as a future
flagship.

## Animation — issues (foundation for #17, #25)

✅ Skeletal **skinning** (`--skinning`, the GPU-skinned Fox), an animation **state machine** cross-fade (`--anim-fsm`),
animation **blending** (`--anim-blend`), and a full **deterministic IK control-rig** (`--ik1-angle` … `--ik6-render`:
two-bone + FABRIK + look-at + skeleton bridge + lockstep + a lit skinned capstone) — bit-identical, rollback-replayable
IK. See ARCHITECTURE "Deterministic IK control-rig".

## Agent / developer experience

✅ A versioned, golden-verifiable **Agent SDK**: `--agent-api` (the versioned contract), `--agent-query` (scene read),
`--author-scene` (declarative spec → canonical scene), `--hot-reload` (deterministic reload == cold load),
`--replay-record`/`--replay-verify` (record→replay→assert-determinism), `--determinism-stress` (the rollback fuzzer).
See ARCHITECTURE "The Agent Experience (AX) product".

---

### Genuinely not yet built (honest gaps / roadmap)

These are real features Hazard Forge does **not** yet ship, tracked as future flagships: a visual-scripting / Blueprint
layer, a UMG-class retained-mode UI framework, a cinematic Sequencer, a GPU profiler / frame-debugger UI, spatial /
graph-based audio (a deterministic integer mixer ships, but no HRTF/graph), a production networking layer (dedicated
server / RPC / replication graph — a deterministic lockstep *substrate* ships beneath it), an AI behaviour-tree + EQS
layer (navmesh + deterministic A* ship: `--nav-path`), a PCG framework, foliage-at-scale, temporal upscaling
(TSR/FSR/DLSS-class), broader platform targets (Linux / mobile / console), wider asset import (FBX/OBJ/USD; glTF ships),
and a layered (Substrate-style) material BRDF. See the roadmap in the project notes.
