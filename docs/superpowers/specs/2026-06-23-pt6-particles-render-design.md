# Slice PT6 — Lit 3D render capstone (the FLOAT money-shot, COMPLETES flagship #19) (Issue #19, 6th slice)

APPEND to `engine/sim/particles.h` (PT1-PT5 BYTE-FROZEN). PT6 renders the bit-exact particle pool as lit 3D
instanced spheres — a deterministic spark fountain. The money-shot completing flagship #19. Mirrors GR6/FPX6
VERBATIM (grain::GrainToRenderInstances → the existing instanced-lit pipeline). NO new shader, NO new RHI.

## THE GATE IS DIFFERENT FROM PT1-PT5 (read carefully)
PT6 is a FLOAT visresolve-bar slice. The SIM (the pool) is bit-exact integer + byte-identical cross-backend
(run the SAME StepParticlesN on both → identical pool). But the final raster/shade is FLOAT, so the rendered
image is NOT bit-identical Vulkan-vs-Metal. THEREFORE:
- The gate is **Metal two-run BYTE-IDENTICAL (DIFF 0.0000)** (deterministic on the same GPU) + **provenance**
  (every instance transform IS a bit-exact FxParticle::pos) + a **documented cross-vendor mean** (~30-60,
  NOT zero — the GR6/FPX6 float baseline).
- DO NOT expect/require the strict-zero cross-backend pixel compare (it will NOT be 0 — that's expected for a
  float render). The golden is the Metal-baked render.

## Append to particles.h
```cpp
inline float ParticleToFloat(fx v) { return (float)v / (float)kOne; }                 // the GrainToFloat twin
inline math::Vec3 ParticleVertToWorld(const FxVec3& p);                               // pos/(float)kOne
inline math::Mat4 ParticleTransform(const FxParticle& p, float radius);               // translate(pos/kOne)*scale(radius)
// ParticleToRenderInstances(pool, radius): one column-major mat4 per ALIVE particle (dead slots skipped), a
// PURE FUNCTION of the pool (two calls byte-equal). The ONE host float crossing, render-only, OUT of the
// bit-exact sim path. Mirrors grain::GrainToRenderInstances (grain.h:1015) VERBATIM.
inline std::vector<math::Mat4> ParticleToRenderInstances(const ParticlePool& pool, float radius);
```

## Showcase `--pt6-render-shot` (Vulkan) / `--pt6-render` (Metal)
Build a deterministic spark FOUNTAIN: the PT4 StepParticlesN scene (emitter + a force field + gravity + ground
+ a sphere) advanced to a representative settled/steady tick — **the SAME integer scene + recipe on BOTH
backends so the pool is byte-identical by construction** (the GR6 pattern). Then ParticleToRenderInstances →
render as lit 3D INSTANCED spheres through the EXISTING instanced-lit pipeline (lit_instanced.vert + lit.frag +
scene::InstanceTransformLayout, the FrameData camera/light UBO, sky + instanced/static shadow + post — REUSED
VERBATIM from `--grain-render-shot`/`--fpx-render-shot`; find that block and clone it) from a fixed 3/4 camera
+ directional light over the ground, a warm spark color. Optional: scale/fade particles by age (deterministic
float of integer age/lifetime) — keep it simple if it risks the determinism.

EXACT proof lines (fail loudly):
```
pt6-render: {particles:A, instances:A, shaded:P} (deterministic particles -> lit 3D render)
pt6-render determinism: two renders BYTE-IDENTICAL
pt6-render coverage: P shaded (coherent lit fountain), provenance instances == ParticleToRenderInstances
pt6-render empty: base scene (no-op)  (zero alive -> cleared ground/sky, no instances)
```
Assertions: (1) instance count == alive count AND instances == a recomputed ParticleToRenderInstances (the
provenance — every transform derives from the bit-exact pool); (2) two GPU renders BYTE-IDENTICAL (Metal-side
determinism is the golden gate); (3) shaded pixel count > 0 and non-uniform (a coherent lit fountain);
(4) an empty pool → the base scene (ground+sky), no instances. Register `pt6_render` in verify.ps1 $Goldens +
`--pt6-render-shot` in $vkShots. Add a PT6 case to particles_test.cpp (ParticleToRenderInstances provenance:
instance count == alive, two calls byte-equal, dead slots skipped).

## Constraints (HARD)
- particles.h APPEND-ONLY (PT1-PT5 byte-frozen). fpx.h/grain.h READ-ONLY. The render helpers are the ONLY
  float in particles.h and are render-only (OUT of the sim path); the sim stays strictly integer.
- NO new shader, NO new RHI — reuse the instanced-lit pipeline VERBATIM (the --grain-render-shot wiring).
- **Scene + recipe IDENTICAL in main.cpp AND visual_test.mm so the POOL is byte-identical (the render then
  differs cross-vendor by the documented float baseline — that's fine).**
- Branch `fix-issue-19-pt6`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`).
- COMPLETION CRITERIA — do NOT commit until: `--pt6-render-shot` runs exit 0, all 4 proof lines print, the
  provenance holds, two-run determinism holds, shaded>0. (The CONTROLLER bakes the Metal golden, checks Metal
  two-run 0.0000, documents the cross-vendor mean, and eyeballs the PNG to confirm a coherent lit spark
  fountain. NO strict-zero cross-backend compare for this FLOAT slice.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.

This slice COMPLETES the 6-slice flagship #19 (deterministic GPU particles). After merge, close GitHub issue #19.
