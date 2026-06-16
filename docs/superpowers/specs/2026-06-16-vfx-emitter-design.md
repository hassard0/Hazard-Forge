# Slice CC — CPU Particle / VFX Emitter System (Phase 4 #28) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. An authorable VFX
> EMITTER layer (data-driven emitter params + lifetime curves) — DISTINCT from the existing fixed
> `gpu-particles` compute fountain (that's one hard-coded GPU sim; this is a configurable CPU emitter
> system, the VFX analogue of the material graph being authorable materials).

**Goal:** A deterministic CPU particle system: configurable emitters (spawn rate, lifetime, initial
velocity + spread, gravity, drag, size-over-life, color-over-life) simulated at a fixed timestep and
rendered as camera-facing, alpha-blended billboards. A `--vfx-shot` showcase simulates a fixed emitter for
N fixed steps and captures one frame; golden-verified on both backends.

## Why fixed-seed + fixed-step (determinism)

Particle spawn jitter normally uses RNG; sim normally uses real time. To stay golden-verifiable, the
spawn jitter is a FIXED-SEED integer LCG (no real RNG) and the sim runs a FIXED number of FIXED-dt steps
(no clock). Same seed + same steps + same emitter config ⇒ identical particle state ⇒ identical billboards
⇒ goldens match. Two runs bit-identical.

## Design decisions (locked)

1. **Particle sim (engine/vfx/particles.{h,cpp}, pure CPU, no backend symbols).** Namespace `hf::vfx`.
   - `struct EmitterConfig { math::Vec3 origin; float spawnRate /*particles/sec*/; float lifetime;
     math::Vec3 initVel; float velSpread; math::Vec3 gravity; float drag; float startSize, endSize;
     math::Vec4 startColor, endColor; uint32_t seed; int maxParticles; };`
   - `struct Particle { math::Vec3 pos; math::Vec3 vel; float age; float lifetime; /* derived size/color
     computed from age/lifetime at render */ };`
   - `class ParticleSystem` — owns a fixed-capacity particle pool. `Step(const EmitterConfig&, float dt)`:
     spawn `floor(spawnRate*dt + accumulator)` new particles (each with `initVel` perturbed by a SEEDED
     LCG draw scaled by `velSpread`), integrate live particles (`vel += gravity*dt; vel *= (1-drag*dt);
     pos += vel*dt; age += dt`), retire particles with `age >= lifetime`. Deterministic (seeded LCG,
     fixed dt). Accessors: `span<const Particle> Alive()`, counts spawned/alive.
   - `float SizeAt(const EmitterConfig&, const Particle&)` = lerp(startSize,endSize,age/lifetime);
     `Vec4 ColorAt(...)` = lerp(startColor,endColor,age/lifetime). Shared with the shader/billboard gen +
     unit-tested.

2. **Billboard generation (CPU, shared with the render).** `BuildBillboards(span<Particle>, EmitterConfig,
   cameraRight, cameraUp, outVerts)` — for each live particle, emit a camera-facing quad (2 tris, 6 verts)
   at `pos`, sized by `SizeAt`, colored by `ColorAt`, with corner UVs for a soft round sprite. Reuse the
   existing alpha-blend pipeline + a billboard/quad pattern (inspect how the text/decal/debug overlays
   build + draw screen/world quads). The sprite texture: a procedural soft circle (computed in the frag
   from UV, like the decal's procedural texture — no asset) OR a small baked alpha sprite; pick + document.

3. **Render path.** A new VFX pass draws the billboard quads alpha-blended, additively or
   over-blended (document), AFTER the opaque scene (sorted back-to-front by view depth for correct
   over-blend, OR additive so order-independent — pick + document; additive is simplest + order-free).
   New `shaders/vfx.{vert,frag}.hlsl` (vert: world quad → clip; frag: soft-sprite alpha * particle color).
   HLSL→SPIR-V→MSL via the existing toolchain.

4. **Showcase `--vfx-shot <out>` (Vulkan) / `--vfx` (Metal).** A fixed FOUNTAIN emitter (origin on the
   ground, upward `initVel` + spread, gravity pulling particles back down, color fading hot→cool,
   size shrinking) simulated for a FIXED number of fixed-dt steps so a full plume is in flight, rendered
   over the lit+shadowed scene from a fixed camera. Print `vfx: {emitters:1, alive:K, spawned:S}`
   (deterministic). New golden `tests/golden/metal/vfx.png` (Metal two runs DIFF 0.0000). Existing 50
   image goldens UNTOUCHED.

5. **Tests `tests/vfx_test.cpp` (pure CPU, no GPU):**
   - **Spawn rate:** over T seconds at rate R, ~`R*T` particles are spawned (within the accumulator
     rounding); capped at `maxParticles`.
   - **Lifetime/retirement:** a particle retires exactly when `age >= lifetime`; alive count is correct
     after stepping past lifetimes.
   - **Integration:** gravity + drag applied correctly over a step (hand-checked a known particle's
     pos/vel after one/two steps); no-gravity no-drag = linear motion.
   - **Curves:** `SizeAt`/`ColorAt` lerp endpoints (age 0 → start, age==lifetime → end) + midpoint.
   - **Determinism:** the same seed + step sequence → bit-identical particle state across two runs.
   - **Billboard gen:** 6 verts/particle, camera-facing (quad spans cameraRight/cameraUp), centered on
     pos, sized by SizeAt.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `particle-vfx` (features) + `--vfx-shot` (showcases).

## RHI seam additions (summary)
- **None.** The particle sim + billboard gen are pure CPU; the VFX pass reuses the existing alpha-blend +
  textured-quad infra. New files (`engine/vfx/particles.{h,cpp}`, `shaders/vfx.*.hlsl`,
  `tests/vfx_test.cpp`) add ZERO backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
GPU particle simulation (the existing `gpu-particles` covers that), collision/physics-coupled particles,
sub-emitters / particle trails, mesh particles, ribbon/beam effects, a node-graph VFX editor, soft-particle
depth-fade, texture-sheet animation, lights from particles, GPU sorting. One CPU emitter with lifetime
curves rendered as alpha-blended billboards, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 50) + new `vfx_test` (spawn rate, lifetime,
   integration, curves, determinism, billboard gen). Clean under `windows-msvc-asan`.
2. `--vfx-shot` on Windows/Vulkan: controller visual review — a recognizable particle FOUNTAIN/plume
   (upward spray arcing under gravity, fading color/size), alpha-blended over the scene, coherent; the
   `vfx: {...}` line is deterministic (two runs → byte-identical capture). Run under the AT Vulkan-
   validation gate → ZERO errors.
3. Metal: `visual_test --vfx` → new golden `tests/golden/metal/vfx.png`; two runs DIFF 0.0000; the vfx stat
   line matches Vulkan.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `vfx.png` added; the
   other 50 byte-identical.
5. Introspect JSON rebaked exactly `+particle-vfx` + `--vfx-shot`; introspect test updated; no other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `vfx` image
   golden in the Mac round-trip loop.
