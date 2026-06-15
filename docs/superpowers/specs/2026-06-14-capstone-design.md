# Slice Z — Integrated Capstone Showcase (Design)

Date: 2026-06-14
Branch: `slice-capstone`

## Goal

Compose the per-feature showcases already built in Slices O–Y into ONE coherent,
deterministic rendered frame. This is both a milestone demo and an INTEGRATION
TEST: rendering IBL + glTF scene import + skinned animation + rigid-body physics
+ full-PBR + sorted transparency + directional shadows + HDR bloom all in a
single frame surfaces descriptor-set / pipeline-state / push-constant coexistence
bugs that the isolated showcases never hit.

This is ORCHESTRATION. No new RHI/shader/loader code. Reuse the existing
pipelines, shaders, loaders, render graph, and capture path. (Seam grep must stay
== 12.)

## Entry points

- Vulkan: `hello_triangle.exe --capstone-shot <path>` — renders the composed frame
  to a BMP and exits. PRIMARY deliverable.
- Metal: `visual_test --capstone <path>` — mirrors the SAME scene construction
  (deterministic physics steps, same fox blend times, same transforms) and writes
  the golden `tests/golden/metal/capstone.png`.

## The composed "showroom" scene

A single ground plane under the HDR sky environment (`assets/env/env.hdr`), with
a fixed deterministic camera and directional light. Laid out so ALL elements are
visible together and legible:

- **HDR environment** (`sky_hdr` + `LoadHdrEnvironment`) — background sky AND the
  IBL lighting/reflection source for the PBR helmet.
- **CesiumMilkTruck** (`LoadGltfScene`) — node hierarchy, deduped PBR materials,
  placed on the ground to the LEFT.
- **Animated Fox** (`LoadSkinnedGltfModel` + `BlendAnimations`) — a fixed 50/50
  Walk(t=0.3)+Run(t=0.2) blended pose, standing on the ground to the RIGHT.
- **DamagedHelmet** (`LoadPbrGltfModel`, full PBR + IBL via `lit_pbr_ibl`) — on a
  small pedestal cube at CENTER-BACK so its metal reflects the environment.
- **Settled physics sphere stack** (`physics::World`, a 3-layer pyramid stepped
  240× @ dt=1/120 to rest) — rendered via the instanced lit pipeline, off to the
  ground-front-left.
- **Translucent glass panel** — a couple of tinted glass spheres in the
  transparent pipeline (depthTest on / depthWrite off), sorted back-to-front,
  placed IN FRONT so the PBR/scene objects read through them.
- **Directional shadows** — shadow pass over all opaque casters (ground, truck,
  fox skinned, helmet, physics instances).
- **HDR bloom** — the whole scene renders into an RGBA16F target; the bloom chain
  (prefilter → 5 down → up → composite/tonemap) blooms the HDR sun + helmet
  emissive.

## Single-frame render order

1. **shadow pass** — all opaque casters into the shadow map (static plane,
   static truck primitives, skinned fox, helmet, instanced physics spheres).
   Needs static + skinned + instanced shadow pipelines coexisting.
2. **scene pass** into the HDR RGBA16F target:
   - `sky_hdr` fullscreen background (binds the equirect env).
   - opaque, all lit + shadowed: ground (lit) → truck (lit_pbr) → helmet
     (lit_pbr_ibl, binds env) → fox (lit_skinned, binds joint palette) →
     physics spheres (lit_instanced).
   - sorted transparent: glass spheres (transparent pipeline).
3. **bloom chain**: prefilter → down[1..4] → upTop → up[3..0] → composite/tonemap
   into the swapchain.

## Known integration risks (why this is a test, not just a demo)

- **State leakage between consecutive pipelines in ONE render pass.** The scene
  pass binds 6+ different pipelines (sky_hdr, lit, lit_pbr, lit_pbr_ibl,
  lit_skinned, lit_instanced, transparent), each with different descriptor-set
  layouts (frame UBO, material textures, PBR 5-tex set, environment, joint
  palette, instance buffer). Must re-bind every required descriptor before each
  draw. The per-feature showcases never exercise this many in one pass.
- **Env / joint-palette / instance buffer must each be (re)bound** right before
  the draw that needs them, since an intervening pipeline change can invalidate
  the bound set.
- **HDR target.** Every opaque pipeline must use `colorFormat = RGBA16_Float`
  (like the bloom showcase), NOT the swapchain format, or the scene pass
  framebuffer is incompatible. The transparent pipeline likewise.
- **Shadow map contention** — single shadow map shared by all casters across
  static/skinned/instanced shadow pipelines in one pass.

## Determinism

Fixed camera/light, fixed transforms, physics stepped a fixed count, fox blended
at fixed times. The Metal mirror uses byte-identical scene construction so the two
renders match.

## Verification

- Vulkan: capture BMP → PNG, visually inspect that ALL elements appear together
  correctly lit/shadowed with bloom, glass see-through, nothing black/exploded.
- Metal: NEW committed golden `tests/golden/metal/capstone.png` (two-run DIFF
  0.0000). The 12 existing goldens stay untouched.
- `ctest` stays green; seam grep stays == 12.

## Bonus (only if it doesn't risk the primary)

- `--capstone-debug-shot` adding the DebugDraw overlay (AABBs + grid + light
  arrow). Deferred unless time permits.
