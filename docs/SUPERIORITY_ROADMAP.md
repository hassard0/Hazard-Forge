# Engine Superiority Roadmap

*The build-out plan: make Hazard Forge's functionality vastly superior to UE5, on two tracks —
**Track R** refines already-shipped flagships to close every documented caveat (each flagship
honestly recorded its v1 gaps; that trail is the refinement backlog), and **Track S** ships new
capabilities UE5 does not have at all. Everything stays deterministic + golden-verified; the moat
is never spent, only deepened.*

---

## Track R — Refinement: close the documented caveats in shipped work

Every flagship shipped with an honest caveat list. Closing them converts "we have X" into
"our X is best-in-class." Ordered by (correctness first, then headline value).

| # | System | Documented v1 caveat (verbatim from ship notes) | Refinement slice |
|---|---|---|---|
| **R1** | **RHI (BUG)** | Latent set-3 cluster **binding-13** graphics push-descriptor reads wrong data (found during #34; compute path fine; may silently affect the clustered-lighting graphics path). Flagged, never fixed. | Diagnose root cause (off-by-one in the graphics cluster push OR placeholder set-1/2 layouts shifting binding indices), fix, regression-golden both paths. **Correctness > features — first.** |
| **R2** | **Fluid** | FL4 solve is net-repulsive with **no viscosity** — spreads like dry foam, not liquid. | **FL7 viscosity/XSPH**: velocity-smoothing pass (host-snapped kernel LUT, Jacobi multi-thread like FL4 — no TDR), bit-exact. Water that behaves like water. |
| **R3** | **Cloth** | **No self-collision**, no dynamic colliders, no friction; single-projection contact. | **CL7 self-collision** (grid-hash the cloth verts against themselves — the fluid FL2 grid reused) + **CL8 dynamic colliders + Coulomb friction** (grain GR4 friction mold). *Deterministic self-colliding cloth is a headline no engine has.* |
| **R4** | **Fracture** | Fragments solved as bounding **spheres** → rounded rubble; convex manifolds deferred. | **FR8 convex-shard solve**: the hull/SAT machinery (fric.h convex) already exists — solve shards as convex hulls. Real angular rubble. |
| **R5** | **Couple (rigid↔fluid)** | Linear buoyancy only (not Archimedes-depth-exact); static-wall containment **leaks**. | **CP7** submerged-volume buoyancy + sealed containment. |
| **R6** | **Grain** | Monodisperse only; Jacobi residual nonzero. | **GR7 polydisperse** radii (mixed gravel/sand) + iteration-count sweep golden. |
| **R7** | **Persist/sleep** | Boxes-only; feature-ID approximate under sliding; islands are all-pairs. | **PS7** convex-hull contact caching + spatial island partitioning. |
| **R8** | **Navmesh** | One surface per column (no overhangs), no hole-carving, triangles-as-polys, no inter-region portals. | **NAV7+** multi-layer heightfield + polygon merge + portals. |
| **R9** | **Metal RT** | v1 TLAS is degenerate single-instance; fragment-stage RT is Vulkan-only. | **RT7** real multi-instance TLAS on Metal (instance transforms in `MTLAccelerationStructure`). |
| **R10** | **Headless robustness** | Windows modal CRT/assert/WER dialogs can hang headless agents (known TODO). | Suppress via `SetErrorMode`/`_CrtSetReportMode` in every exe entry. |
| **R11** | **Editor** | 60% built — panels read-only, inspector text-only, no docking. | ED1–ED6 (specced in `GAP_CLOSING_ROADMAP.md` Tier 1). |
| **R12** | **Ragdoll/joints** | Trailing-contacts composition; ball+cone limits only. | **JT7** hinge/prismatic joint types + motorized joints (deterministic actuation). |

## Track S — Superiority: capabilities UE5 does not have

Each is a new flagship in the proven 6-slice cadence (integer core → the new physics → lockstep →
lit render), bit-exact CPU/Vulkan/Metal unless marked as the float render exception.

| # | Flagship | Why it beats UE5 outright |
|---|---|---|
| **S1** | **Cloth↔fluid coupling** (wet cloth) | 4th pairing in the material-interaction matrix (rigid↔fluid, rigid↔grain, grain↔fluid shipped). A soaked flag dripping into a puddle, deterministic + rollback-replayable — no engine has *any* deterministic two-way coupling; we'd have four. |
| **S2** | **Strand/hair sim** | The one deterministic-sim family still missing (noted at every scout since GR). PBD rods (cloth constraint-graph mold, 1D). Deterministic hair UE5's float Groom can't match. |
| **S3** | **Soft body (volumetric PBD)** | Tet-lattice shape-matching/volume constraints over the cloth solver mold — deformable flesh/jelly, lockstep-replayable. |
| **S4** | **Motion matching** | Queued flagship #33: pose-database search over the anim/skeleton + IK stack, made *deterministic* (integer feature vectors, pinned search order) — UE5's is float/non-replayable. |
| **S5** | **Deterministic TSR** | Temporal super-resolution (UP1–3 in the gap roadmap): render-at-lower-res + motion reprojection + upsample resolve. The *deterministic* upscaler (golden-able) — DLSS/FSR/TSR are all non-reproducible. |
| **S6** | **Auto-LOD (QEM)** | Integer-quantized quadric-error-metric decimation → the Nanite-class pipeline gets automatic LOD generation with a *pinned digest* (UE5's Nanite build is nondeterministic across versions). |
| **S7** | **Many-light** | Lift the 8-light cap via the existing froxel cluster grid → 100+ dynamic lights, light-assignment digest golden. |
| **S8** | **Atmospheric scattering** | Rayleigh/Mie sky (documented gap from the clouds flagship) — LUT-based, deterministic. |
| **S9** | **Temporal RT denoiser** | The documented RT gap: SVGF-class temporal accumulation over the RT output, deterministic variant. |
| **S10** | **Audio graph** | MetaSounds-class node DSP over the Q15 integer mixer (AU1–4): spatialization, buses, effects — *bit-exact procedural audio*, a category UE5's float DSP cannot enter. |
| **S11** | **GPU-particle authoring** | The #19 note: the deterministic GPU particle sim exists; add the authoring layer (emitter graphs via the flow VM). |

## Execution order (interleaved, highest value ÷ risk first)

1. **R1 binding-13 bug** — correctness debt in shipped RHI; fix before building on it.
2. **R2 FL7 viscosity** — small, bounded, converts the fluid from tech-demo to *liquid*.
3. **R3 CL7/CL8 cloth self-collision + friction** — the single strongest new headline in Track R.
4. **S1 cloth↔fluid** — builds directly on R3; the wet-cloth money shot.
5. **R4 FR8 convex shards** — the visual credibility of destruction.
6. **S5 TSR + S6 auto-LOD + S7 many-light** — the rendering-superiority cluster (feeds the Sponza bake).
7. **S2 hair → S3 soft body** — complete the deterministic-sim material families.
8. **R5–R9, S4, S8–S11** — steady-state loop, one flagship arc at a time.
9. **R10/R11** — infrastructure interleave (R10 is an hour; R11 tracks the gap roadmap).

**Cadence unchanged:** scout → spec on master → branch → implementer agent → controller audits
(append-only, frozen-header discipline) → Windows verify → Mac bake/prove → ff-merge → next.
Every slice lands golden-verified on both backends or it doesn't land.

**The standard:** UE5 ships features; we ship *proofs*. Superiority means every one of these is
bit-exact, replayable, and verifiable on three platforms — properties UE5 cannot claim for a single
one of its equivalents.
