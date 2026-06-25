# Hazard Forge — Gap-Closing Roadmap

*Closing the distance to a production engine (the UE5 parity axis), as opposed to widening the
determinism moat. Grounded in a codebase survey (2026-06-25), not aspiration.*

## Framing

Hazard Forge already wins on one axis no shipping engine matches: **bit-exact cross-platform
deterministic simulation + lockstep/rollback netcode**, verified on Windows/MSVC, macOS/Apple-clang,
and Linux/gcc with identical golden digests. The whole GitHub issue backlog is closed except the
environment-blocked platform issue.

The *gap* to a general-purpose production engine is almost never a missing core system. The survey
found that the gap is overwhelmingly **productization**: solid, golden-verified primitives that lack
an authoring/integration layer, plus a few genuinely-missing systems. This roadmap closes the gap in
the order of (leverage × tractability), and — crucially — it preserves the moat: **almost every slice
below stays golden-verifiable**, using the same cadence (a deterministic data model + a render-bake
or a synthetic-input replay), because the interactivity itself can be driven by recorded input events
(the `--fly-dry-run` harness already does this).

**Verification legend per slice:**
`🟢 golden` = bit-exact digest or pixel-diff golden, the standard cadence ·
`🟡 synthetic-input` = deterministic recorded-input replay → asserted state/render (the dry-run pattern) ·
`🔵 integration` = real I/O / real platform, validated by behavior not byte-identity (the honest exception).

---

## Tier 1 — The interactive editor (highest leverage, mostly UI plumbing)

**The gap, honestly:** the "no real editor app" critique is ~40% true, not 100%. `--fly`
(`samples/hello_triangle/main.cpp`) already boots an SDL3 window + live main loop + 50-key input
(`engine/runtime/input_state.h`) + fly camera + T/R/S gizmos (`engine/editor/gizmo.h`, wired to G/R/T)
+ click-pick + gizmo-drag + Ctrl+S scene save (`engine/scene/scene_io.h`) + hot-reload + a clickable
outliner. The edit-ops (`engine/editor/edit_ops.h::ApplyTransformEdit/ApplyMaterialEdit`) work but are
only called from test code; the panels (`engine/editor/{flow,seq,widget,profiler}_editor_panels.cpp`)
draw read-only geometry and never check `ImGui::IsItemClicked()`. This is the single biggest
UE5-adoption-driver gap, and it's the cheapest to close.

| Slice | Work | Verify |
|---|---|---|
| **ED1 Inspector editing** | Replace the text-only inspector (`editor_panels.cpp`) with `DragFloat`/`SliderFloat`/color-edit fields that call the existing `ApplyTransformEdit`/`ApplyMaterialEdit`. | 🟡 record a synthetic drag on a field → assert the ECS component + a render-bake diff |
| **ED2 Panel interactivity** | Wire `IsItemClicked`/drag in the flow/seq/widget panels to the *existing* edit-ops (AddFlowNode/ConnectFlow, AddKeyframe/MoveKeyframe, AddChildWidget). The ops + golden data-models already exist. | 🟡 replay an input script (palette-click → AddNode) → assert the post-edit view digest |
| **ED3 Docking + layout** | The vendored ImGui (1.91.8) lacks the docking branch. Swap to the docking-enabled ImGui, replace the fixed-position tiling with a DockSpace; or formalize the fixed layout as the "stable golden layout" and keep both. | 🟢 panel render-bakes unchanged (golden layout) + 🔵 manual docking smoke |
| **ED4 Gizmo→inspector live link** | Gizmo drag already mutates the transform; reflect it live in the inspector + multi-select + snapping. | 🟡 synthetic drag → assert inspector reads back the new transform |
| **ED5 Undo/redo** | A deterministic command stack over the edit-ops (every edit is already a pure op — wrap them in a recorded, reversible command list; this *is* a moat feature: replayable edit history). | 🟢 op-stack digest; apply N, undo N → original digest bit-identical |
| **ED6 Asset/content browser** | A panel listing `assets/` + the reference library (#18 fetch), drag-to-place into the scene via the loader. | 🟡 synthetic place → assert scene graph + bake |

**Why first:** ~2–3 days per slice on infrastructure that already exists; turns "a headless bake harness"
into "a launchable scene editor" — the thing newcomers actually judge an engine by. The moat is
*preserved*, because every edit is a deterministic op and ED5 makes the entire edit history replayable.

---

## Tier 2 — Real content at scale (techniques exist, scale unproven)

**The gap, honestly:** every technique is built and golden-verified — meshlet/cluster-LOD/cluster-cull
(`engine/render/`), software raster, VSM + virtual-texturing page tables (`vsm.h`/`vt.h`),
clustered/froxel lighting, substrate materials, distance streaming (`engine/scene/streaming.h`) — but
proven on **fixtures**: ~144 instances of ~12k-tri spheres, 8 point lights, **hand-authored** 3-level
LODs, homogeneous grids. Nothing has rendered a real Sponza/Bistro through the full stack. #18's fetch
loop now works (proven with a live CC0 asset), which unblocks this tier.

| Slice | Work | Verify |
|---|---|---|
| **SC1 Hero-scene bake** | A `--hero-shot` that fetches + loads the Khronos PBR Sponza (or a substantial CC0 subset) and renders it through PBR+IBL+shadows. First real measure of draw-calls/tris/materials. | 🟢 two-run byte-identical bake + cross-vendor pixel-diff |
| **SC2 Automatic LOD generation** | Quadric-error-metric decimation to generate LOD1/LOD2 from a high-poly LOD0 (today's 3 LODs are hand-tessellated spheres). Deterministic integer-quantized collapse order. | 🟢 decimated-mesh digest pinned, cross-platform |
| **SC3 Full-stack at scale** | Push SC1's scene through meshlet→cluster-cull→cluster-LOD→VSM shadow→VT sample → measure it holds (the techniques' first heterogeneous, million-tri, multi-material workout). | 🟢 per-stage digests + 🔵 perf budget |
| **SC4 Many-light path** | Lift the 8-light cap: a clustered/tiled deferred (or Forward+ light-list) path for 100+ dynamic lights, reusing the existing `clustered.h` froxel cluster grid. | 🟢 light-assignment digest + lit bake |
| **SC5 Foliage scatter at scale** | Wire the PCG scatter (`pcg.h` PCG1-6) + foliage wind/LOD (`foliage.h`) into a 10k+ instance GPU-driven bake with the integer wind applied on-GPU. | 🟢 placement digest (already shuffle-invariant) + instanced bake |
| **SC6 Texture-residency integration** | Feed the VT/VSM page-feedback into the streaming budget (today the page math is proven but not driven by a real scene's residency). | 🟢 page-set digest under a deterministic camera path |

**Why second:** proves the rendering tech is *real*, not just unit-correct — the credibility gap between
"has a Nanite-style pipeline" and "renders Sponza." All slices stay golden (deterministic content +
fixed camera path = byte-stable bake).

---

## Tier 3 — Authoring layers on solid primitives

Each of these is a **solid, golden-verified deterministic core** missing only the authoring/integration
layer. The cores don't need rework; they need an editor surface (which Tier 1 makes possible) and a few
runtime bridges.

| Area | What exists (golden) | The gap slices |
|---|---|---|
| **Animation** | skeleton, channel clips, cross-fade blend, FSM (`anim_fsm`), IK solvers (FABRIK/two-bone, `ik.h` — already shipped) | **AN1** blend spaces (1D/2D param→clip blend) 🟢 · **AN2** retargeting (skeleton→skeleton pose) 🟢 · **AN3** anim-graph authoring in the editor (Tier 1) 🟡 |
| **PCG** | full PCG1-6 (hash-PRNG, jittered scatter, density mask, transforms, overlap-prune, render bridge) | **PC1** spline-driven scatter 🟢 · **PC2** terrain-aware placement (slope/mask/normal-align) 🟢 · **PC3** PCG node-graph editor (Tier 1) 🟡 |
| **Foliage** | deterministic wind field, distance LOD, PCG placement | **FO-A** GPU wind animation (integer bend→quaternion on-GPU) 🟢 bake · **FO-B** impostor/card LOD at distance 🟢 · **FO-C** in-editor paint tool 🟡 |
| **Audio** | Q15 integer mixer, ADSR, wavetable osc, WAV out | **AU1** 3D spatialization (distance/pan/occlusion, integer) 🟢 · **AU2** mixing buses + submix graph 🟢 · **AU3** DSP effects (EQ/delay/reverb, integer) 🟢 · **AU4** node-graph audio authoring 🟡 |

**Why third:** high value, but each depends on Tier 1's editor surface to be *authorable*. The
deterministic cores mean every slice here is still a 🟢 golden (a MetaSounds-class audio graph, evaluated
in fixed-point, is bit-exact — a moat feature UE5's float audio can't claim).

---

## Tier 4 — Genuinely missing systems

These are real holes, not productization. They're later because they're larger and (for networking)
partly break the headless-golden paradigm.

### TSR — temporal super-resolution (real gap)
Today's `taa.h` is anti-aliasing at *native* res (Halton jitter + neighborhood clamp + EMA blend) — the
jitter beachhead exists, but there is no *upscaling*. Slices: **UP1** render-at-lower-res plumbing
(render-target scale factor) 🟢 · **UP2** motion-vector reprojection (warp history by camera+object
motion) 🟢 · **UP3** the upsample resolve (disocclusion handling, sharpening) 🟢. All deterministic →
golden. Closes the TSR/FSR/DLSS-class gap on the moat's terms (a *deterministic* temporal upscaler).

### Production networking (the honest paradigm exception)
The deterministic substrate is *done* and golden: snapshots, delta-encode, jitter buffer, interpolation,
and a seeded `SimChannel` (`engine/net/`). What's missing is the **real transport**: **NW1** a UDP socket
channel implementing the same `Channel` interface as `SimChannel` (the abstraction already exists) 🔵 ·
**NW2** a dedicated-server harness (headless authoritative sim loop) 🔵 · **NW3** RPC over the snapshot
stream 🟡 · **NW4** a replication graph (interest management / relevance culling — the assignment logic is
deterministic) 🟢. Note: NW1/NW2 are the **first 🔵 integration-tested slices** — real sockets are
nondeterministic timing, so they're validated by behavior (convergence, no-desync) while the sim *under*
them stays bit-exact. This is the one place "close the gap" genuinely trades some golden coverage for
production reality.

### Vehicles (already complete — listed for completeness)
`engine/sim/vehicle.h` is a **complete** deterministic system (spring suspension, Coulomb wheel traction,
drive/steer commands, lockstep+rollback, VH1-VH5 golden). Only polish remains (aero, transmission, visual
feedback) — not a gap.

---

## Tier 5 — Platforms

| Platform | Status | Path |
|---|---|---|
| **Linux (core)** | ✅ proven bit-exact (gcc 14.3.0, Docker) | done — `master 5f8fecc` |
| **Linux (renderer)** | portable Vulkan, blocked only on windowing | **PL1** `VK_EXT_headless_surface` (or surfaceless offscreen) device-init path 🟢 → full engine build under Docker + lavapipe → a Linux render-bake golden. Tractable now (Docker proven reachable). |
| **iOS / Android / consoles** | genuinely blocked (no SDKs/devkits/signing in this env) | needs a maintainer-provided build environment; the core is already proven portable, so the surface is the platform/windowing + RHI-surface layer, not engine logic. **Won't stub.** |

---

## Recommended sequence

1. **Tier 1 (editor)** — biggest perceived-quality jump per unit effort; unblocks Tier 3 authoring. Start **ED1/ED2** (inspector + panel interactivity on existing ops).
2. **Tier 2 SC1** — one real Sponza bake is the highest-credibility single artifact in the whole list; do it early even out of tier order.
3. **Tier 4 TSR (UP1-3)** — self-contained, fully golden, closes a marquee checklist gap cleanly.
4. **Tier 2 SC2-SC6 + Tier 3 authoring** — interleave as the editor surface lands.
5. **Tier 5 PL1 (Linux render)** — tractable via the proven Docker+lavapipe path.
6. **Tier 4 networking** — last of the tractable work (the 🔵 paradigm exception); biggest single system.

## What stays out (honest)

- Mobile/console runtime (SDK/hardware-blocked).
- A marketplace/asset-store ecosystem (organizational, not technical).
- Matching UE5's *content scale & battle-hardening* — that's years of shipped titles, not a roadmap.

**The throughline:** closing the gap here means *productizing verified primitives into authorable,
real-content-scale systems* — and because the cores are deterministic, the productization stays
golden-verifiable almost everywhere. The gap closes without spending the moat.
