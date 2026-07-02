# Engine Superiority Roadmap

*The build-out plan: make Hazard Forge's functionality vastly superior to UE5, on two tracks —
**Track R** refines already-shipped flagships to close every documented caveat (each flagship
honestly recorded its v1 gaps; that trail is the refinement backlog), and **Track S** ships new
capabilities UE5 does not have at all. Everything stays deterministic + golden-verified; the moat
is never spent, only deepened.*

**Status (2026-07): the superiority run is largely COMPLETE — 17 slices merged (`572bf25`..`9cae497`).
R1–R10 + R12 and S1–S3 + S5–S8 + S10 are ✅ SHIPPED (per-row commits + goldens below). Open: R11
(editor), R13 (the PS7-discovered warm-hull hardening, NEW), S4 (motion matching — unverified), S9
(RT temporal denoiser), S11 (particle authoring).*

---

## Track R — Refinement: close the documented caveats in shipped work

Every flagship shipped with an honest caveat list. Closing them converts "we have X" into
"our X is best-in-class." Ordered by (correctness first, then headline value).

| # | System | Documented v1 caveat (verbatim from ship notes) | Refinement slice |
|---|---|---|---|
| **R1** | **RHI (BUG)** ✅ **SHIPPED** (`572bf25`) | Latent set-3 cluster **binding-13** graphics push-descriptor reads wrong data (found during #34; compute path fine; may silently affect the clustered-lighting graphics path). Flagged, never fixed. | Root cause: TWO push-descriptor set layouts in one pipeline layout (the #34 accel set + the cluster set) — `VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293` UB. Fixed: the graphics accel set is now a regular pooled set; guarded by the `--probe-binding13` regression probe (binding 13 FAILed before, ALL PASS after; clustered/froxel/VSM goldens byte-identical). |
| **R2** | **Fluid** ✅ **SHIPPED** (`a32530b`, golden `fl7_visc`) | FL4 solve is net-repulsive with **no viscosity** — spreads like dry foam, not liquid. | **FL7 viscosity/XSPH** shipped: velocity-smoothing pass (the FL3 host-snapped kernel LUT reused, Jacobi multi-thread — no TDR), bit-exact, `c==0` an exact no-op. Honest residual: momentum drift pinned ≤ 256 LSB/axis (fxmul truncation), not exactly zero. |
| **R3** | **Cloth** | **No self-collision**, no dynamic colliders, no friction; single-projection contact. | **CL7 self-collision** ✅ **SHIPPED** (`b2e8129`, golden `cl7_self`): exclusion 1-ring + thickness grid + Jacobi projection, 108 penetrating pairs → 0, settled residual exactly 1 LSB under thickness (pinned). **CL8 dynamic colliders + Coulomb friction remains OPEN.** |
| **R4** | **Fracture** ✅ **SHIPPED** (`fa93d4c`, golden `fr8_hull`) | Fragments solved as bounding **spheres** → rounded rubble; convex manifolds deferred. | **FR8 convex-shard solve** shipped: oriented `FxBox` shards through the shipped `fric.h` SAT+friction solver — angular rubble resting rotated (maxQuatDev ≈ 0.74, digest `0xa5a9b8f5158108d2`). Honest: shards are centroid-AABB boxes, not exact Voronoi cell hulls (a `gjk::FxHull` per fragment stays future). |
| **R5** | **Couple (rigid↔fluid)** ✅ **SHIPPED** (`73f3e7f`, golden `cp7_float`) | Linear buoyancy only (not Archimedes-depth-exact); static-wall containment **leaks**. | **CP7** shipped: exact spherical-cap submerged-volume buoyancy (`SphereCapVolume`, ρ=0.5 floats half-submerged) + a hard AABB seal (0 escapes vs 343 leaked before). Honest: the equilibrium depth is a pinned band around the analytic depth (splash-quantized surface estimator). |
| **R6** | **Grain** ✅ **SHIPPED** (`a44b596`, golden `gr7_poly`) | Monodisperse only; Jacobi residual nonzero. | **GR7 polydisperse** shipped: index-hash size classes + exact r³ inverse mass + a widened broadphase contract, identity-at-uniform byte-exact to GR4 (digest `0xb3325f416aff93ce`). Honest: Brazil-nut segregation is emergent/weak in a small settled pour (pinned, not forced); the Jacobi residual caveat stands. |
| **R7** | **Persist/sleep** ✅ **SHIPPED** (`7b8217e`, golden `ps7_hullsleep`) | Boxes-only; feature-ID approximate under sliding; islands are all-pairs. | **PS7** shipped: spatial island discovery (broad.h grid + order-independent union-find) byte-identical to all-pairs, extended to general-hull warm+sleep (lockstep digest `0x3a49757d1f7d6750`). Honest: the sliding warm-start-miss caveat is WH1's, inherited; see the NEW R13 finding below. |
| **R8** | **Navmesh** ✅ **SHIPPED (multi-layer)** (`b070e78`, golden `nav7_ml`) | One surface per column (no overhangs), no hole-carving, triangles-as-polys, no inter-region portals. | **NAV7 multi-layer heightfield** shipped: every walkable span top a surface layer — under-the-bridge AND over-it paths sharing zero surfaces (digest `0x14cf524e6e089c37`). **The rest of R8 remains OPEN:** hole-carving, polygon merge, inter-region portals, ML watershed/polymesh (NAV7 paths on the surface grid graph directly). |
| **R9** | **Metal RT** ✅ **SHIPPED** (`ec85940`, golden `rt7_instanced`) | v1 TLAS is degenerate single-instance; fragment-stage RT is Vulkan-only. | **RT7 real multi-instance TLAS on Metal** shipped: true `MTLInstanceAccelerationStructureDescriptor` instances (userID == Vulkan `instanceCustomIndex`), HW == CPU byte-equal, per-instance hit counts 1156/716/1157 identical Vulkan/Metal; the 1-instance path byte-unchanged. Fragment-stage RT on Metal remains OPEN. |
| **R10** | ~~Headless robustness~~ **ALREADY SHIPPED** | `engine/platform/crash_dialogs.h` (`DisableCrashDialogs()`) exists and is wired into main.cpp, all tests (`test_main.h`), and visual_test.mm — asserts/aborts/faults route to stderr + non-zero exit. The roadmap TODO was stale. | — |
| **R11** | **Editor** | 60% built — panels read-only, inspector text-only, no docking. | ED1–ED6 (specced in `GAP_CLOSING_ROADMAP.md` Tier 1). **OPEN.** |
| **R12** | **Ragdoll/joints** ✅ **SHIPPED** (`6c4aa9a`, golden `jt7_machine`) | Trailing-contacts composition; ball+cone limits only. | **JT7 hinge/prismatic/motorized joints** shipped: cross/dot axis-alignment hinge + line-clamp prismatic + accumulated-impulse-clamped motors, a lockstep-replayable motorized crank-slider (digest `0xf947a5e58a21d4ac` both backends). Honest: the crank-slider is over-constrained (pinned deterministic residual ≤ 0.6); zero-g machine bench. |
| **R13** | **Warm-hull solver (NEW — the PS7 finding)** | Verbatim from PS7's ship notes: "the frozen accumulated warm hull solver is NOT validated for hard drops (a large fall pumps energy through the documented gjk iteration-cap near-field band — floor half >4 or a 2.5-unit drop makes hulls hop; scenes use near-rest drops + angDamp 0.3)." | Harden the accumulated warm hull solve for high-energy impacts (the gjk iteration-cap near-field band), then re-pin the PS7 scenes without the near-rest-drop restriction. **OPEN.** |

## Track S — Superiority: capabilities UE5 does not have

Each is a new flagship in the proven 6-slice cadence (integer core → the new physics → lockstep →
lit render), bit-exact CPU/Vulkan/Metal unless marked as the float render exception.

| # | Flagship | Why it beats UE5 outright |
|---|---|---|
| **S1** | **Cloth↔fluid coupling** (wet cloth) ✅ **SHIPPED** (`ac2f5db`, golden `cf1_couple`) | 4th pairing in the material-interaction matrix (rigid↔fluid, rigid↔grain, grain↔fluid shipped). **CF1 shipped:** a pinned hammock catches a falling stream 64/64 while sagging 4264 LSB, two-way drag + barrier, lockstep-replayable (digests `0x1a97a446acc1ec52`/`0xc1ef0d4d5332c7e5`). Honest: the vert-sphere barrier is porous by construction; not a validated FSI continuum model. |
| **S2** | **Strand/hair sim** ✅ **SHIPPED** (`d135d83`, golden `hr1_hair`) | The one deterministic-sim family still missing (noted at every scout since GR). **HR1 shipped:** Q16.16 PBD rods (stretch + bending + strand collision, the cloth mold), lockstep-replayable (digest `0x4f241b7fbdadbadc`). Honest: PBD bending saturates — a "stiff" strand still droops (pinned honestly). |
| **S3** | **Soft body (volumetric PBD)** ✅ **SHIPPED** (`ffcafdb`, golden `sb1_soft`) | **SB1 shipped:** an N³ PBD lattice with per-cell int64-determinant volume preservation — squashes to ~75% and recovers while the kVol=0 control stays crushed; lockstep-replayable (digest `0x10a66a8b0e20bfb5`). Honest: a pinned deterministic settled micro-jitter (~0.04 u/s). |
| **S4** | **Motion matching** | Queued flagship #33: pose-database search over the anim/skeleton + IK stack, made *deterministic* (integer feature vectors, pinned search order) — UE5's is float/non-replayable. **OPEN (unverified).** |
| **S5** | ~~Deterministic TSR~~ **ALREADY SHIPPED** | Flagship #20 (US1–US5, merged 2026-06-23, ARCHITECTURE.md "Temporal super-resolution"): half-res render → history reprojection + disocclusion → N=8 temporal resolve → RCAS → the 2× hero (tsrDiff 1.05 < naiveDiff 2.67, goldens both backends). The gap roadmap's "no upscaling" claim was stale — corrected. Optional future hardening: a strict-integer CPU==GPU byte-exact upscale variant ("US1-strict"), only if the moat-grade proof is judged worth a slice. |
| **S6** | **Auto-LOD (QEM)** ✅ **SHIPPED** (`9f4ef09`, golden `lod_gen`) | **LOD1 shipped:** integer-quantized quadric-error-metric decimation with per-level pinned digests (`0xeade209d8e61838f`/`0xbedec704837c71f4`/`0x1a9821628ab77e80`, identical MSVC/Win-clang/Mac-clang) feeding the byte-untouched cluster-LOD pipeline — the deterministic LOD build UE5's Nanite build is not. Honest: nearest-source-vertex attribute carry; heuristic (not Hausdorff-certified) error bound. |
| **S7** | **Many-light** ✅ **SHIPPED** (`91d481e`, golden `manylight`) | **ML1 shipped:** 128 lights (cap 1024) through the existing set-3 cluster SSBOs — `frame_data.hlsli` byte-untouched, pinned assignment digest `0x42d5535632f152ea`. Honest: the slice math uses `std::pow`/`std::log`, so the digest is pinned per-toolchain-family. |
| **S8** | **Atmospheric scattering** ✅ **SHIPPED** (`7ecfd0d`, golden `at1_sky`) | **AT1 shipped:** Nishita-class Rayleigh + HG-Mie single scattering, the CPU digest exact MSVC==Win-clang via a libm-free `DetExp` (`0x5792dedf3715ec68`; Apple-ARM last-ULP diverges, the clouds/water float class). Honest: single scattering only — no multiple scattering / ozone / ground bounce. |
| **S9** | **Temporal RT denoiser** | The documented RT gap: SVGF-class temporal accumulation over the RT output, deterministic variant. **OPEN.** |
| **S10** | **Audio graph** ✅ **SHIPPED** (`e8e55e8`, golden `tests/golden/audio/au1_scene.wav`) | **AU1 shipped:** a MetaSounds-class node graph over the frozen Q15/dsp primitives + a `kSpatial` 3D node (integer inverse-square + azimuth pan, no trig/float) — a 384 KB WAV byte-identical Windows/MSVC vs macOS/Apple-clang (digest `0x9ab527c7a057c06b`), a category UE5's float DSP cannot enter. Honest: pan/ILD (not HRTF), offline synthesis, Q15 delays decay to exact silence. |
| **S11** | **GPU-particle authoring** | The #19 note: the deterministic GPU particle sim exists; add the authoring layer (emitter graphs via the flow VM). **OPEN.** |

## Execution order (interleaved, highest value ÷ risk first)

1. ~~**R1 binding-13 bug**~~ ✅ — fixed first, as planned (`572bf25`).
2. ~~**R2 FL7 viscosity**~~ ✅ (`a32530b`).
3. ~~**R3 CL7 cloth self-collision**~~ ✅ (`b2e8129`) — CL8 friction/dynamic-colliders still open.
4. ~~**S1 cloth↔fluid**~~ ✅ (`ac2f5db`).
5. ~~**R4 FR8 convex shards**~~ ✅ (`fa93d4c`).
6. ~~**S5 TSR + S6 auto-LOD + S7 many-light**~~ ✅ (S5 pre-shipped; `9f4ef09`, `91d481e`).
7. ~~**S2 hair → S3 soft body**~~ ✅ (`d135d83`, `ffcafdb`).
8. **R5–R9, S8, S10** ✅ shipped (`73f3e7f`, `a44b596`, `7b8217e`, `b070e78`, `ec85940`, `7ecfd0d`, `e8e55e8`) + **R12** ✅ (`6c4aa9a`); **S4, S9, S11 remain** — steady-state loop, one flagship arc at a time.
9. **R10** ✅ (already shipped) / **R11** open (tracks the gap roadmap) / **R13** open (the PS7 finding).

**Cadence unchanged:** scout → spec on master → branch → implementer agent → controller audits
(append-only, frozen-header discipline) → Windows verify → Mac bake/prove → ff-merge → next.
Every slice lands golden-verified on both backends or it doesn't land.

**The standard:** UE5 ships features; we ship *proofs*. Superiority means every one of these is
bit-exact, replayable, and verifiable on three platforms — properties UE5 cannot claim for a single
one of its equivalents.
