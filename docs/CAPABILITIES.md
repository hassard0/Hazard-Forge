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
> always-current list (364 showcases). The determinism that underlies all of this — bit-identical Vulkan/Windows ==
> Metal/macOS, lockstep/rollback-replayable — is the moat described in `docs/ARCHITECTURE.md`.

## Rendering & lighting

| Capability | Ships? | See it (flag) | ARCHITECTURE / notes |
|---|---|---|---|
| Deferred + forward+ rendering, clustered lights | ✅ | `--clustered-lights`, `--cull`, `--clustered` | Forward+ clustered light culling, byte-identical to brute force |
| **Many-light rendering (128 lights, past the 8-light FrameData cap)** | ✅ | `--manylight-shot` / `--manylight` | SSBO light array through the existing set-3 cluster bindings (`engine/render/manylight.h`, golden `manylight`); pinned assignment digest `0x42d5535632f152ea`. *Honest note:* the cluster-slice math uses `std::pow`/`std::log`, so the digest is pinned per-toolchain-family |
| Shadows: CSM, spot, point-cube, contact | ✅ | `--csm`, `--spot`, `--point-shadow`, `--contact-shadows` | Shadow-atlas tiling |
| **Virtual Shadow Maps (single-texel-per-pixel, any distance)** | ✅ | `--vsm-pages`, `--vsm-atlas`, `--vsm-shadow`, `--vsm-cache` | Clipmap VSM flagship — *replaces the fixed cascade* (issue #14) |
| SSR / planar / SSGI reflections + GI | ✅ | `--ssr`, `--planar-reflection`, `--ssgi`, `--ssgi-temporal` | Screen-space + planar reflections |
| **Deterministic Lumen-class global illumination** | ✅ | `--gi1-probe` … `--gi6-hero`, plus DDGI `--probegi`/`--probe-capture`/`--probe-sh` | RT-traced integer-SH irradiance probe volume + multi-bounce + Chebyshev occlusion + Cornell color-bleed (issue #10) |
| Volumetrics, fog, clouds, cloud shadows | ✅ | `--volumetric`, `--froxel-fog`, `--clouds`, `--cloud-shadows` | Raymarched volumetric clouds + froxel fog (issue #12) |
| **Physical sky — Rayleigh + Mie atmospheric scattering** | ✅ | `--at1-sky-shot` / `--at1-sky` | Nishita-class single-scattering (`engine/render/atmosphere.h`, golden `at1_sky`); the CPU digest `0x5792dedf3715ec68` is exact MSVC == Win-clang (a libm-free `DetExp`). *Honest limits:* single scattering only — no multiple scattering (dark twilight), no ozone, no ground-albedo bounce; HG Mie phase |
| PBR + IBL, GTAO, SSS, bloom, DoF, TAA, motion blur, CAS, auto-exposure, color grade | ✅ | `--pbr`, `--ibl`, `--gtao`, `--sss`, `--bloom`, `--dof`, `--taa`, `--motion-blur`, `--cas`, `--auto-exposure`, `--color-grade` | Full post stack |
| Animated shaders (time-driven sky/water/foliage/VFX) | ✅ | `--sky-animated`; `FrameData.skyParams.zw = (time, frameIndex)` | The per-frame time channel (issue #5); shared `procedural_sky.hlsli` (issue #4) |

### HDR image-based lighting (IBL) — issue #8

✅ **The `--ibl` showcase *is* the HDR-IBL sample** (the `ibl_helmet` golden): it loads the HDR environment, builds the
IBL, and renders the DamagedHelmet catching real environmental detail in its metal — not the procedural-sky gradient.
The pieces, and the minimal wiring:

- `engine/asset/env_loader.cpp` loads `assets/env/env.hdr` (the path is `HF_ENV_PATH`, defined in the sample
  CMakeLists).
- `shaders/lit_pbr_ibl.frag.hlsl` consumes the HDR environment cube for both the diffuse irradiance and the specular
  reflection; `shaders/sky_hdr.frag.hlsl` is the matching HDR sky sampler.
- Wire it like the `--ibl` handler: load the env cube → bind it to the IBL slot → use the `lit_pbr_ibl` pipeline
  instead of `lit.frag`. The `--ibl` showcase in `samples/hello_triangle/main.cpp` / `metal_headless/visual_test.mm`
  is the copy-paste reference. (To make this even more extractable as a standalone `samples/hdr_ibl_demo/`, that's a
  welcome follow-up — the capability + golden already ship.)

### Reflections — the options — issue #6

The *simple* `lit.frag.hlsl` reflects the procedural sky only (fast, no scene-radiance input — a metallic surface
shows the sky, not the car driving past). Dynamic scene-object reflections **are** available, through dedicated paths
— each a public showcase:

- **Screen-space reflections** — `--ssr` (+ `--ssgi`/`--ssgi-temporal` for screen-space GI): reflects on-screen
  geometry, cheapest dynamic-object reflection.
- **Planar mirror reflections** — `--planar` (golden `planar_reflection`): a true mirror plane (floor/glass) that
  reflects the scene.
- **Cubemap probes** — `--refl-probe` (box-projected static) and `--capture-probe` (dynamic cubemap capture): local
  reflections for a room/region.
- **Ray-traced reflections** — `--rt4-reflect`: ground-truth RT reflections (see Hardware ray tracing below).

Pick by need: SSR for cheap on-screen, planar for a mirror surface, probes for a localized environment, RT for
ground truth. *Honest note:* composing these into the *simple* lit pipeline **by default** (so any `lit.frag` surface
auto-reflects dynamic geometry without opting into a path) is a genuine architectural enhancement, not yet done — today
you select the reflection path explicitly, as above.

## Hardware ray tracing (deterministically reconciled) — issues #7, #13

✅ **Ships as a full RHI accel-structure seam on BOTH backends.** The RT flagship added `IAccelStructure` / BLAS / TLAS
+ inline ray query over `VK_KHR_ray_query` (Vulkan) to the RHI — *not* a single hint shader — and (#34) extended the
accel binding to graphics/fragment pipelines. **Metal now implements the same seam (issues #42/#35, CLOSED):**
`engine/rhi_metal/metal_accel.{h,mm}` builds a real `MTLAccelerationStructure` and `MetalDevice` overrides
`CreateBlas`/`CreateTlas`/`BindAccelStructure`/`SupportsHardwareRayQuery` (→ **true on Apple-silicon HW**). The
Metal RT showcases run **real hardware ray tracing through the RHI seam**, not the CPU reference. The int64→MSL
blocker (HLSL+int64 `RayQuery` is Vulkan-SPIR-V-only) is sidestepped by **native MSL** kernels
(`shaders/rt_query.metal`/`rt_shadow.metal`/`rt_reflect.metal`/`rt_hero.metal`) using `metal::raytracing::
intersection_query<>` with int64 fx via MSL `long`. The public path:

- `--rt1-trace` — the deterministic Q16.16 **software reference tracer** (the "ground-truth reference mode" of issue
  #13 — every HW path is validated to agree with it).
- `--rt2-query` (Vulkan HW inline ray query, in the RHI) — real Vulkan hardware RT through the engine RHI.
- `--rt2-query-rhi` (Metal HW, in the RHI) — real Apple-Silicon hardware ray query **through the engine RHI seam**
  (`device->CreateBlas`/`CreateTlas` + `cmd->BindAccelStructure`), proven byte-equal to the CPU reference + the
  `rt2_query` golden on an M4. (`--rt2-query-hw` is the original raw-`MTL*` standalone proof, kept for comparison.)
- `--rt3-shadow`, `--rt4-reflect`, `--rt6-hero` — RT hard shadows, mirror reflections, and the lit hero capstone —
  run **real Metal HW ray tracing through the RHI seam** on Apple-silicon (and Vulkan HW on Windows), each
  byte-equal to the CPU `rtrace::` reference and its committed golden (`rt3_shadow`/`rt4_reflect`/`rt6_hero`). On a
  Mac without HW ray tracing (M1/M2), they fall back to the CPU reference. (`--rt5-simrender` / `--rt-reflect-graphics`
  remain CPU-reference on Metal — see the honest scope below.)
- `--rt7-instanced-shot` / `--rt7-instanced` — **real multi-instance TLAS on Metal** (golden `rt7_instanced`): N ≥ 2
  instances build a true `MTLInstanceAccelerationStructureDescriptor` (per-instance transforms + `userID` matching
  Vulkan's `instanceCustomIndex`), proven HW == CPU **byte-equal** with the per-pixel winning-instance map identical
  Vulkan/Metal (per-instance hit pixels 1156/716/1157; pinned digests `0x4bff69aafcf23871` / `0x70112b520d270e6b`).

The moat: the HW BVH is used only as a candidate generator (margin-inflated AABBs make the float overlap a strict
superset); our integer Q16.16 intersection owns the closest hit, so the HW result is **bit-identical to the CPU
reference on BOTH vendors** — Metal HW == Vulkan HW == CPU, byte-for-byte — a deterministic RT no float RT engine
offers. **Honest scope (issues #42/#35):** the v1 "degenerate single-instance TLAS" caveat is CLOSED by RT7 (above —
a 1-instance TLAS keeps the original path byte-for-byte, so the rt2/rt3/rt4/rt6 goldens are unchanged);
fragment-stage RT (the graphics-pipeline `accelStructureBinding` / `--rt-reflect-graphics`) is wired on Vulkan but
not yet on Metal. See ARCHITECTURE "Hardware ray tracing, deterministically reconciled".

✅ **Deterministic stochastic RT soft shadows + an SVGF-lite temporal denoiser now also ship** (superiority-run S9:
`--rtd1-softshadow-shot` / `--rtd1-softshadow`, golden `rtd1_softshadow`, `engine/render/rtd.h`): area-light soft
shadows via one jittered shadow ray per pixel per frame off a fixed 64-entry host-baked golden-angle Vogel spiral
(no RNG/clock; ranged any-hit occlusion — an occluder beyond the light does not occlude), denoised by fixed-N=8
temporal accumulation + the `--ssgi-denoise` bilateral called verbatim, edge-guarded by the RT hits' own t/normal.
The integer half (visibility counts + accumulated image) is **HW == CPU byte-equal** (digests
`0xf38663819426b236`/`0xc93c6d134aa489aa`, cross-platform exact); the bilateral + denoised shade is the float
visresolve class (two full runs byte-identical, float buffers included). *Honest notes:* the pinned
`RtdDenoiseParams` tightens spatialSigma 2.0→1.0 because the ssgi default over-blurs the penumbra ramp; at the
96×72 test resolution the ~4 px penumbra still over-blurs vs the raw 8-sample (reported, not hidden — the strict
gates are vs the raw 1-sample); static camera, no motion reprojection yet (composes with TSR US3).

## Virtualized geometry (Nanite-style) — issue #9

✅ **The core ships:** a **visibility-buffer** renderer (`--visbuffer`), a **software rasterizer** for sub-pixel
triangles (`--swraster`, `--swraster-gpu`, `--swraster-resolve`), and **cluster-LOD** selection with no popping
(`--cluster-lod`, `--cluster-hiz`, `--cluster-cull`). Meshlet decomposition is Morton-ordered + deterministic.
**Automatic LOD generation now also ships** (`--lod-gen-shot`, golden `lod_gen`): deterministic integer-QEM
decimation (`engine/render/lod_gen.h`) builds the ~50%/~25% LOD chain feeding the byte-untouched `cluster_lod`
selection, with a **pinned digest per level** (`0xeade209d8e61838f` / `0xbedec704837c71f4` / `0x1a9821628ab77e80`,
identical MSVC/Win-clang/Mac-clang) — a deterministic LOD build a nondeterministic Nanite-class build pipeline
explicitly is not. *Honest note:* attributes carry by nearest-source-vertex, boundary verts are locked outright, and
the reported geometric error is an upper-estimate heuristic, not a certified Hausdorff bound. See
ARCHITECTURE's virtual-geometry sections. *Genuine remaining work:* **world-scale meshlet streaming** (paging clusters
from disk for billion-triangle scenes) — the core rasterizer + LOD are done; the streaming tier is a tracked
enhancement, an opportunistic roadmap item.

## Physics — Chaos-class, deterministic — issue #29

✅ **Ships broadly, and bit-identical/replayable in a way Chaos is not.** The deterministic fixed-point (Q16.16) sim
stack:

- **Cloth:** `--cloth-integrate` … `--cloth-render` (PBD cloth, lockstep-replayable) + **self-collision**
  (`--cl7-self`, golden `cl7_self` — a folding cloth no longer passes through itself; *honest gap:* the settled
  residual is exactly 1 LSB under thickness, and a vert moving > ~thickness per step can still tunnel — no CCD yet).
- **Fluids:** `--fluid-integrate` … `--fluid-render` (position-based fluids) + **XSPH viscosity** (`--fl7-visc`,
  golden `fl7_visc` — water that flows, not foams; *honest gap:* the integer momentum drift is a pinned nonzero
  bound ≤ 256 LSB/axis, not exactly zero).
- **Granular/sand:** `--grain-integrate` … `--grain-render` (Coulomb friction / angle-of-repose) + **polydisperse
  sizes** (`--gr7-poly`, golden `gr7_poly` — mixed gravel/sand with exact r³ mass splits; *honest gap:* Brazil-nut
  segregation is an emergent shaking-driven statistic, near-equal heights in a small settled pour).
- **Destruction/fracture:** `--fract-cells` … `--fract-render` (Voronoi fracture + rigid-body rubble) +
  **convex-shard rubble** (`--fr8-hull-shot`, golden `fr8_hull` — oriented boxes through the SAT+friction solver,
  angular rubble that rests rotated; *honest gap:* shards are centroid-AABB boxes, not exact Voronoi cell hulls).
- **Rigid bodies + contacts:** convex GJK/EPA, hull friction + joints (`--hf1-points` … `--hf6-hull`), warm-started
  stacking, CCD, ragdoll, vehicle; **spatial islands + sleeping hull piles** (`--ps7-hullsleep`, golden
  `ps7_hullsleep` — O(n·k) island discovery byte-identical to all-pairs); **high-energy impact hardening**
  (`--wh7-harddrop`, golden `wh7_harddrop` — the WH7/R13 fix: 2.5-unit hard drops onto large floors settle
  fully asleep; the former "not validated for hard drops" gap was two int32 narrowphase bugs, root-caused +
  fixed, and the fix revealed the warm-start's real ~400× residual win the depth bug had masked — the old
  warm-vs-cold "no benefit" equality pin was an artifact; *honest gap:* the plain GJK dot products still
  saturate int32 at a CSO diameter of ~104 world units);
  **hinge/prismatic/motorized joints**
  (`--jt7-machine`, golden `jt7_machine` — a lockstep-replayable motorized crank-slider; *honest gap:* the
  crank-slider is over-constrained with a pinned deterministic residual, and the bench is zero-g).
- **Hair/strands:** `--hr1-hair-shot` (golden `hr1_hair`) — Q16.16 PBD rods with bending + strand↔strand collision,
  lockstep-replayable (*honest gap:* PBD bending saturates — a "stiff" strand still droops visibly; the droop is
  pinned, not hidden).
- **Soft body (volumetric):** `--sb1-soft-shot` (golden `sb1_soft`) — a PBD lattice with per-cell int64 volume
  preservation; squashes to ~75% and recovers while the no-volume control stays crushed (*honest gap:* a pinned
  deterministic settled micro-jitter ~0.04 u/s, and per-cell volume carries a small deterministic residual).
- **Two-way material coupling:** rigid↔fluid — now with **Archimedes submerged-volume buoyancy + sealed
  containment** (`--cp7-float-shot`, golden `cp7_float`; a ρ=0.5 body floats half-submerged, 0 particles escape the
  sealed basin vs 343 leaked before; *honest gap:* the equilibrium depth is a pinned band around the analytic depth,
  not exactly on it) — rigid↔grain, grain↔fluid (mud/slurry/wet-sand), and **cloth↔fluid** (`--cf1-couple-shot`,
  golden `cf1_couple` — the wet cloth: a pinned hammock catches a falling stream 64/64 while sagging; *honest gap:*
  the vert-sphere barrier is porous by construction and not a validated FSI continuum model).

Every one is bit-identical CPU/Vulkan/Metal AND lockstep/rollback-replayable. See ARCHITECTURE's deterministic-sim
sections. With hair/strands and the volumetric soft body shipped, the deterministic-sim family now spans rigid /
cloth / fluid / grain / hair / soft-body plus four two-way couplings.

## Navigation & pathfinding

✅ **A deterministic Recast/Detour-class navmesh + integer A\* ships** (`engine/nav/navmesh.h`, `--nav-raster` …
`--nav-render`): span rasterization → walkable filter → watershed regions → contours/polygonization → integer A\* →
lit render, every build stage bit-identical cross-backend. **Multi-layer navmesh now also ships** (`--nav7-ml`,
golden `nav7_ml`): every walkable span top becomes its own surface layer, so an agent paths **under** a bridge and
**over** it as distinct never-connected surfaces (pinned digest `0x14cf524e6e089c37`) — closing the flagship's
one-surface-per-column caveat. *Honest gaps (documented, deferred):* hole-carving, polygon merge
(triangles-as-polys), inter-region portals, and the ML path runs on the surface grid graph directly (not an ML
polymesh); vertical distance is not costed. See ARCHITECTURE "Deterministic GPU navmesh + pathfinding".

## Audio — deterministic integer DSP

✅ **A deterministic audio stack ships, byte-identical cross-platform:** the Q15 integer mixer + WAV goldens
(`--audio-render`), the node-DSP flagship (`--dsp-song`, `song.wav`), and now the **declarative audio graph with 3D
spatialization** (`--au1-graph-shot` / `--au1-graph`, `engine/audio/audio_graph.h`): osc/gain/mix/delay/adsr/pan
nodes plus a `kSpatial` node (integer inverse-square distance attenuation + azimuth pan, no trig/float anywhere in
the sample path), wired by index and evaluated in one canonical topological order. The golden is the strongest
cross-platform proof tier in the engine: `tests/golden/audio/au1_scene.wav`, a 384 KB WAV **byte-identical
Windows/MSVC vs macOS/Apple-clang** (pinned scene digest `0x9ab527c7a057c06b`). *Honest gaps:* pan/ILD
spatialization (not HRTF convolution), offline buffer synthesis (no real-time device output), and the Q15
truncation means a feedback delay decays to **exact silence** in finitely many echoes — pinned, by design. See
ARCHITECTURE "Deterministic audio graph + 3D spatialization".

## Particles / VFX — issue #19

✅ **A deterministic GPU particle system ships** (`engine/sim/particles.h`, `--pt1-emit` … `--pt6-render`): a
free-list emitter (no atomics, no `rand` — spawn is a pure function of the command stream), **force fields**
(point/vortex/wind), **particle-vs-world collision** (ground + spheres + restitution bounce), the composed
`StepParticles` tick, **lockstep/rollback replay**, and a **lit 3D capstone** (the spark-fountain money-shot). The
entire sim is Q16.16, **bit-identical CPU/Vulkan/Metal AND lockstep/rollback-replayable** — which float Niagara
cannot do (two machines re-derive the exact same fountain, every spark, from inputs alone). **Particle AUTHORING now
also ships** (superiority-run S11: `--pa1-fountain-shot` / `--pa1-fountain`, golden `pa1_fountain`,
`engine/sim/particle_author.h`): the deterministic flow VM drives the emitter/force-field parameters — Niagara-class
emitter graphs, authored via the editor edit-ops, not C++ structs. The showcase pulsing fountain is built exclusively
through `AddFlowNode`/`ConnectFlow` (24 nodes, 3 bindings; the pinned `SerializeGraph` digest `0xca55eabb27042ba8` is
the authored-not-hardcoded proof), and the lockstep snapshot includes the flow `GraphState` (omitting it provably
diverges — graph state IS sim state). Honest v1 scope: a pure emitter (no particle-particle/SPH — the grid-hash is
reuse-ready for a future slice), sphere-instanced particles (not sprites/ribbons), no mesh/skinned emission. The older
CPU billboard emitter (`--vfx`) also remains. See ARCHITECTURE "Deterministic GPU particle system".

## Animation — issue #17

✅ **Skinned glTF animation ships, and `--skinning` is the public animated sample** (the GPU-skinned **Fox.glb**,
which is an animated model — the Fox is posed by sampling its animation, then GPU-skinned and rendered). The full set:
**skinning** (`--skinning`), an animation **state-machine** cross-fade (`--anim-fsm`), animation **blending**
(`--anim-blend`), a **deterministic IK control-rig** (`--ik1-angle` … `--ik6-render`: two-bone + FABRIK + look-at +
skeleton bridge + lockstep + a lit skinned capstone, bit-identical + rollback-replayable), and now **deterministic
MOTION MATCHING** (superiority-run S4 / flagship #33: `--mm1-locomotion-shot`, golden `mm1_locomotion`,
`engine/anim/motion_match.h`) — a Q16.16 20-dim pose/trajectory feature database + an int64 weighted-L1
nearest-neighbor search with the strict (cost, index) tie-break, lockstep/rollback-replayable from input commands
alone (trace digest `0xb37aa47b7bc7d69b`); UE5's motion matching is float/non-replayable. *Honest v1:* brute-force
search, hard-switch transitions (a render-side cross-fade layers on top), root-position-relative (not
facing-relative) features, small fixture database. See ARCHITECTURE "Deterministic IK control-rig" and
"Deterministic motion matching".

**The loader *does* surface skeleton + animation** (the issue's premise that it only handles static meshes is out of
date): `asset::LoadSkinnedGltfModel(device, path)` returns a `SkinnedModel` with `.skeleton` + `.animations`
(`FindAnimation("Survey"/"Walk"/"Run")`). The load → play path:

```cpp
auto fox = asset::LoadSkinnedGltfModel(*device, HF_FOX_MODEL_PATH);   // skeleton + animations
const anim::Animation* clip = fox.FindAnimation("Survey");
auto pose    = anim::SampleLocalPose(fox.skeleton, *clip, timeSeconds);   // sample at any time
auto palette = anim::PaletteFromLocalPose(fox.skeleton, pose);           // joint matrices for the GPU
device->SetJointPalette(palette.data(), palette.size() * sizeof(Mat4));  // bind, draw with lit_skinned.vert
```

Drive `timeSeconds` from your frame clock to play it; use `anim::StateMachine::Evaluate` for a cross-faded FSM, or
`BlendAnimations` for a blend. The `--skinning` / `--anim-fsm` handlers in `samples/hello_triangle/main.cpp` /
`metal_headless/visual_test.mm` are the copy-paste references. **Genuine gap:** **morph targets** (blend-shapes — for
facial animation / vehicle wheel-deform) are not yet extracted by the glTF loader; tracked as a follow-up. (Rotating
*wheels* are a skeletal/transform animation, which the path above already supports.)

## Editor — interactive, headlessly verified

✅ **The docked editor now EDITS, not just displays** (superiority-run R11: ED1/ED2/ED5 — the former "panels
read-only, inspector text-only" state is closed). Every piece of interactivity is proven headlessly by a
**synthetic-input dry-run gate** in `scripts/verify.ps1` (real ImGui io events aimed at probe-recorded widget
rects; two full passes must be byte-identical):

| Capability | Ships? | See it (flag) | Notes |
|---|---|---|---|
| **Interactive Inspector editing** (Transform/Material DragFloats + a texture-name combo, through the existing pure edit ops) | ✅ | `--ed1-dry-run` (+ the live `--fly` editor; goldens `editor`/`editor_edit` rebaked) | Typed values land bit-exact, persist through save/reload, second pass byte-identical |
| **Interactive authoring panels** — flow (palette-add / slot-wire / delete), sequencer (lane-click add-key on-curve, arrow-key move), widget (add-child / delete / Ctrl+click-typed style edits) | ✅ | `--ed2-dry-run` (static shots `--flow-editor-shot`/`--seq-editor-shot`/`--widget-editor-shot` byte-unchanged) | Every action bit-compared against a hand-called edit-op twin; all edit chrome is edit-mode-only, so the static goldens stayed pixel-identical |
| **Deterministic undo/redo + a replayable edit-session artifact** (`engine/editor/edit_history.h`) | ✅ | `--ed5-dry-run` (+ Ctrl+Z/Ctrl+Y in `--fly`) | Flat-POD reversible commands; undo N + redo N restores BIT-IDENTICAL state; the serialized session is pointer-free (name-resolved textures) and `ReplayHistory` reproduces the edited scene byte-for-byte on a fresh registry — UE5's `FTransaction` is neither serializable nor replay-stable. Pinned cross-compiler digest `0x3cb790d71e9f35d2` |

*Honest gaps (open on the R11 row):* ED3 docking/layout, the ED4 remainder (multi-select + snapping), ED6
asset/content browser; the undo stack enrolls the scene Transform/Material + flow op families (seq/widget
enrollment is documented follow-up work; scene entity create/delete is not enrolled). See ARCHITECTURE
"Interactive editing — inspector edits, authoring panels, undo/redo".

## Agent / developer experience

✅ A versioned, golden-verifiable **Agent SDK**: `--agent-api` (the versioned contract), `--agent-query` (scene read),
`--author-scene` (declarative spec → canonical scene), `--hot-reload` (deterministic reload == cold load),
`--replay-record`/`--replay-verify` (record→replay→assert-determinism), `--determinism-stress` (the rollback fuzzer).
See ARCHITECTURE "The Agent Experience (AX) product".

## Parity++ audit — fourteen new capabilities

A focused gap-closing audit against a modern-engine feature checklist. Each is built to the house determinism discipline
(integer fixed-point core, host-baked LUTs, no runtime transcendentals / RNG / clock on the state path; a float bridge
only for the lit render) and backed by a committed golden. Thirteen register a new golden; the fourteenth upgrades the
interactive Sponza fly-through. Every honest caveat below is carried verbatim from the slice header. See ARCHITECTURE
"Parity++ capability slices" for the full write-ups.

| Capability | Ships? | See it (flag) | ARCHITECTURE / notes |
|---|---|---|---|
| **Deterministic gameplay ability system** (attribute sets, gameplay effects, cooldowns, ability activation) | ✅ | `--gas1-duel` | `engine/game/ability.h`, golden `gas1_duel`. Pure integer (Q16.16 / `uint32`, no float, no wall clock — durations/cooldowns are TICKS); base-vs-current attribute split with a pinned fold order; stacking (stack/refresh/ignore); periodic damage-over-time; deterministic failure enums; lockstep + rollback (duel-trace `0xf74f7e4198440670`). A peer re-derives every damage number and cooldown bit-for-bit |
| **Water gameplay volume** (analytic-Gerstner surface driving Archimedes buoyancy + drag) | ✅ | `--wv1-float` | `engine/sim/water_body.h`, golden `wv1_float`. The render ocean and the physics ocean are one integer-Gerstner equation; **zero-byte** analytic/stateless water snapshot; lockstep-replayable. *Honest gaps:* height-only physics surface (horizontal crest-pinch kept in the render only); render-vs-sim agree as the same *equation*, not the same *bits* (within-LSB band) |
| **First-class deterministic splines** (Catmull-Rom eval + arc-length + scatter + swept road-strip + camera track) | ✅ | `--sp1-road` | `engine/spline/spline.h`, golden `sp1_road`. Pure int32/int64 (no float/`<cmath>`); the 2-point spline is an exact integer lerp; stateless (lockstep inherited by consumers). *Honest gap:* **uniform** Catmull-Rom can overshoot/loop/cusp under very uneven control-point spacing — author roughly-even spacing (centripetal-via-LUT is future); `\|coord\| ≤ 2²⁹` |
| **Deterministic blend spaces** (phase-synced 1D/2D parametric animation blending) | ✅ | `--an1-blend` | `engine/anim/blend_space.h`, golden `an1_blend`. Integer params/weights/barycentric math + tick-based slew (replayable, not frame-rate-coupled); **normalized-phase** foot-plant alignment; the reused float pose-blend seam; cross-compiler digests. *Honest gap:* a fixed **authored** 2D triangulation (no Delaunay) |
| **Rigid-body force-field volumes** (radial / vortex / wind AABB volumes) | ✅ | `--ff1-fields` | `engine/sim/force_field.h`, golden `ff1_fields`. The PT2 particle field math applied to `fpx` bodies pre-step; bit-exact + lockstep with a toggle command; linear-falloff force pinned **bit-exact** to the particle math. *Honest gaps:* linear force at the center of mass only (no field-gradient torque); center-gated volumes; frictionless ground contact (bodies coast on exit) |
| **Landscape authoring** (sculpt/flatten/smooth brushes + splat paint + spline-carved roads + bit-exact undo) | ✅ | `--la1-landscape-shot` / `--la1-landscape` | `engine/terrain/terrain_author.h`, golden `la1_landscape`. Every edit a pure integer op over a Q16.16 + `uint8×4`-splat `AuthoredTerrain`; flat-core brush ("flatten hits exactly"); splat re-sums to exactly 255; the road carve composes SP1; bit-exact reversible history |
| **VFX renderer variety** (ribbon trails + beams + mesh emitters + particle lights) | ✅ | `--vr1-vfx` | `engine/vfx/vfx_render.h`, golden `vr1_vfx`. Four deterministic geometry generators over the bit-exact particle sim; particle lights feed the existing clustered many-light path (`manylight.h`/`clustered.h` byte-untouched). *Honest notes:* velocity-aligned ribbons kink at a direction reversal; beams are **visual-only** (no gameplay hit query) |
| **Hair strand renderer** (tangent-aligned camera-facing ribbons + Kajiya-Kay anisotropic shading) | ✅ | `--hrr1-groom-shot` / `--hrr1-groom` | `engine/render/hair_render.h` + `shaders/hair_kajiya.frag.hlsl`, golden `hrr1_groom`. Turns the bit-exact HR1 strand sim (formerly debug dots) into a lit groom; the pure-integer groom scene (sim digest `0x2fe41334235921fd`) + the one documented render-only float ribbon crossing (the `ClothToRenderMesh` twin) |
| **Deterministic animation retargeting** (bind-delta rotation transfer + root-motion height scaling) | ✅ | `--an2-retarget` | `engine/anim/retarget.h`, golden `an2_retarget`. Plays one skeleton's clip on a differently-proportioned skeleton; one `QuantizeFx` float→integer boundary then integer quaternion algebra (cross-compiler by construction); **self-retarget is a bit-exact identity**. *Honest gap:* assumes unit quaternions (integer-sqrt normalize for non-unit inputs is future) |
| **Deterministic crowd simulation at 10,000+ agents** (O(N) grid-neighbor separation, archetypes, goal-seek) | ✅ | `--cr1-crowd` | `engine/sim/crowd.h`, golden `cr1_crowd`. Scales where the O(N²) boids separation could not; the **key pin:** the O(N) grid separation sum == the O(N²) all-pairs sum **bit-for-bit** (integer add is associative); lockstep-replayable; proven **10,000 agents × 200 ticks** (digest `0xc80b1b212fadfeb0`). Pure CPU, no new shader |
| **Deterministic convolution reverb + submix bus graph** (integer Q15 MAC, synthetic room impulse response) | ✅ | `--au2-reverb-shot` / `--au2-reverb` | `engine/audio/reverb.h`, golden `au2_reverb`. Integer Q15 MAC (int32 IR so a unit impulse is a bit-exact identity); wet/dry sends through the submix bus math; byte-identical int16 samples MSVC == clang. *Honest gaps:* **room / space** reverb, **NOT binaural HRTF**; a Q15 feedback path decays to exact silence; offline (no real-time device output) |
| **Deterministic USD/UsdSkel skeletal-animation import** (bone hierarchy + skin weights + TRS clips) | ✅ | `--sk1-import-shot` / `--sk1-import` | `engine/asset/usd_skel.h`, golden `sk1_import`. A **second, device-free** skeletal importer alongside glTF (which already imports skeletal but is device-coupled); pure-CPU, dependency-free, byte-identical text parse; composes with the retarget/blend stack with zero new anim code. *Honest scope:* the binary-FBX and USDA-geometry importers remain geometry-only |
| **Deterministic signed-distance-field text** (integer SDF glyph gen + proportional layout with kerning) | ✅ | `--uf1-text-shot` / `--uf1-text` | `engine/ui/sdf_text.h`, golden `uf1_text`. Scales crisply past the fixed 8×8 monospace bitmap; integer point-to-segment SDF + advances + pair kerning + line-break; bit-identical every compiler/backend. *Honest scope:* hand-authored **monoline stroke** glyphs, a ~40-glyph Latin subset (A–Z / 0–9 / punct), **NOT a TTF/FreeType loader**; no Unicode/bidi/shaping; shader-free (a GPU SDF-text shader is the future capstone) |
| **Upgraded interactive Sponza fly-through** (live HDR + bloom + physical Rayleigh/Mie sky) | ✅ | `--sponza-explore` (+ `--sponza-explore-shot`) | `samples/hello_triangle/main.cpp`. The real multi-material PBR Sponza fly-through now runs the full HDR `RGBA16F` → bloom → ACES post stack + the physical AT1 sky live. **No new golden** — anchored by the existing SC1 / AT1 / bloom goldens it reuses |

*Honest note — an attempted slice deferred:* a decal **material-channel** blending slice (DM1) was prototyped but is **not
shipped** — its Vulkan path worked, but a Metal-specific reconstruction bug blocked the cross-platform golden, so
material-channel decals are **not** claimed. The shipped screen-space decal remains texture/color projection + alpha
blend (`--decal`, golden `decal`).

---

### Genuinely not yet built (honest gaps / roadmap)

These are real features Hazard Forge does **not** yet ship, tracked as future flagships: a visual-scripting / Blueprint
layer (#24), a UMG-class retained-mode UI framework (#30), a cinematic Sequencer (#25), a GPU profiler / frame-debugger
UI (#31), HRTF audio (#26 — the deterministic integer mixer, node-DSP graph, 3D spatialization, AND now **room
convolution reverb + submix buses** (`--au2-reverb`) all ship, see "Audio" and "Parity++ audit" above; true binaural HRTF
convolution and real-time device output remain), a production
networking layer (#27 — dedicated server / RPC / replication graph; a deterministic lockstep *substrate* ships beneath
it), a PCG framework (#22), foliage-at-scale (#21), temporal upscaling (#20, TSR/FSR/DLSS-class), broader platform
targets (#23, Linux / mobile / console), and wider asset import (#15/#16 — **device-free USD/UsdSkel skeletal import now
ships** (`--sk1-import`) alongside device-coupled glTF skeletal; binary-FBX and the beyond-first-mesh USDA importers remain
geometry-only).
(Metal hardware ray tracing *through the RHI* — #42/#35 — now SHIPS: `engine/rhi_metal/` implements the accel seam,
`SupportsHardwareRayQuery()` is true on Apple-silicon, and `--rt2-query-rhi`/`--rt3-shadow`/`--rt4-reflect`/`--rt6-hero`
run real Metal HW RT byte-equal to the CPU reference.) See the roadmap in the project notes.

**Recently shipped — moved OUT of this list (the superiority run):** ✅ **strand/hair sim** (`--hr1-hair-shot`),
✅ **volumetric soft body** (`--sb1-soft-shot`), ✅ **audio graph + 3D spatialization** (`--au1-graph`), ✅ **physical
Rayleigh+Mie sky** (`--at1-sky`), ✅ **auto-LOD (integer QEM)** (`--lod-gen-shot`), ✅ **many-light rendering**
(`--manylight`), ✅ **cloth self-collision** (`--cl7-self`), ✅ **cloth↔fluid coupling** (`--cf1-couple-shot`),
✅ **XSPH viscosity** (`--fl7-visc`), ✅ **convex-shard rubble** (`--fr8-hull-shot`), ✅ **Archimedes buoyancy +
sealed containment** (`--cp7-float-shot`), ✅ **polydisperse grains** (`--gr7-poly`), ✅ **spatial islands + hull
sleep** (`--ps7-hullsleep`), ✅ **multi-layer navmesh** (`--nav7-ml`), ✅ **multi-instance TLAS on Metal**
(`--rt7-instanced`), ✅ **hinge/prismatic/motorized joints** (`--jt7-machine`), ✅ **stochastic RT soft shadows +
SVGF-lite denoiser** (`--rtd1-softshadow`), ✅ **particle authoring via the flow VM** (`--pa1-fountain`),
✅ **deterministic motion matching** (`--mm1-locomotion-shot`), ✅ **warm-hull high-energy-impact hardening**
(`--wh7-harddrop`), ✅ **interactive editor: inspector editing / authoring panels / undo+redo with a replayable
edit session** (`--ed1-dry-run` / `--ed2-dry-run` / `--ed5-dry-run`).

**Earlier ships moved out:** ✅ **deterministic GPU particles** (#19, `--pt1-emit`…`--pt6-render`),
✅ **Substrate-lite layered materials** (#11, `--sb1-clearcoat`…`--sb6-substrate` — clearcoat/sheen/iridescence/aniso/
SSS), ✅ **deterministic AI** (#28, `--ai1-tree`…`--ai6-render` — decision/behaviour trees + environment queries (EQS) +
integer line-of-sight, on the navmesh + deterministic A*).
