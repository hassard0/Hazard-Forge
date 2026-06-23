# Slice PT4 — The composed StepParticles tick (Issue #19, flagship #19, 4th slice)

APPEND to `engine/sim/particles.h` (PT1-PT3 BYTE-FROZEN — do NOT modify any existing line). PT4 composes the
PT1 emitter + PT2 force fields + PT3 collision into ONE deterministic tick: emit → forces+integrate →
collide → recycle. A full VFX effect (a fountain with wind + ground/sphere collision) in one call. v1 is a
PURE EMITTER — NO particle-particle interaction (justified: Niagara's default modules are emitter + global
fields + world collision; per-particle-independent = stronger determinism, NO grid/atomics/TDR; the grid-hash
is reuse-ready for a future P-P slice). Pure Q16.16/integer. int64 → shader Vulkan-only, Metal CPU ref.

## Append to particles.h
```cpp
// One deterministic PT4 world tick, composing PT1-PT3:
//   (1) Emit(pool, cfg)                                              — host single-thread (free-list LIFO)
//   (2) IntegrateParticlesWithForces(pool, fields, count, g, dragK, dt) — PT2 (force accumulate + integrate)
//   (3) CollideParticleWorld(pool, groundY, radius, e, spheres, sc)  — PT3 (plane + spheres + bounce)
//   (4) RecycleDead(pool)                                            — PT1 (ascending-slot free-list maint.)
//   (5) ++pool.tick
// Every sub-stage is the EXISTING PTn function called VERBATIM (PT4 only orders them). Per-particle
// independent (2)+(3) -> multi-thread, NO atomics, NO TDR. Returns the contact count (a coverage stat).
inline int StepParticles(ParticlePool& pool, const EmitterConfig& cfg, const ForceField* fields, uint32_t count,
                         const FxVec3& gravity, fx dragK, fx dt, fx groundY, fx radius, fx e,
                         const ParticleSphereCollider* spheres, uint32_t sphereCount);
inline void StepParticlesN(ParticlePool& pool, ... , int steps);   // run `steps` StepParticles
```

## Shader `shaders/particles_step.comp.hlsl` (NEW, int64, Vulkan-only)
The GPU per-particle pass for (2)+(3): copy the PT2 AccumulateForce + IntegrateParticleWithForce body THEN
the PT3 collide body (plane + colliders loop) — i.e. a clone of particles_collide.comp but with the FORCES
accumulate added before the integrate (so it composes both, not just gravity). Both a fields StructuredBuffer
AND a colliders StructuredBuffer input. Register in the shader manifest (NOT Metal MSL). The host runs Emit +
RecycleDead between dispatches (the PT1 loop); the GPU dispatch is forces+integrate+collide per particle. The
CPU StepParticles is the bit-exact reference the GPU memcmp's against.
(If reusing two dispatches per tick — particles_forces.comp then a collide-only pass — is cleaner and stays
bit-exact, that's acceptable; but a single particles_step.comp avoids double-integrate and is preferred.)

## Showcase `--pt4-step-shot` (Vulkan) / `--pt4-step` (Metal)
**Scene/recipe/fields/colliders BYTE-IDENTICAL in BOTH main.cpp and visual_test.mm (the PT2 lesson — verified
by the controller's cross-backend pixel compare).** A full effect: emitter raining DOWN, gravity, a WIND or
VORTEX field deflecting the stream sideways, onto a ground plane + 1-2 static spheres — the stream curves,
falls, bounces, and pools. K large (lifetime kOne*3 ≈ 180 ticks, K=240 → emit/death steady-state + churn).
Viz: REUSE the PT1/PT2/PT3 fixed-point worldToPx VERBATIM + draw the ground line + sphere outlines (the PT3
viz). hashColor(seed) dots.

EXACT proof lines (fail loudly):
```
pt4-step: {alive:A, fields:F, colliders:C, contacts:X, steps:K} GPU==CPU BIT-EXACT
pt4-step determinism: two runs BYTE-IDENTICAL
pt4-step steady-state: alive A converged to ~rate*lifetime/dt-bounded (emit/death balanced, died>0)
pt4-step composition: each sub-stage == its PT1/PT2/PT3 reference in isolation
```
Assertions: (1) full-buffer memcmp(gpu,cpu)==0 NO tol; (2) two GPU runs memcmp==0; (3) track spawned/died,
assert died>0 AND alive within the steady-state band AND the free-list invariant (freeList.size()==capacity-
alive) holds; (4) run ONE StepParticles tick and assert the pool equals applying Emit→IntegrateParticlesWith
Forces→CollideParticleWorld→RecycleDead by hand (the composition is exactly the PTn stages in order — proves
PT4 is pure composition, nothing new in the math). Strict-zero golden `pt4_step` in verify.ps1 $Goldens +
`--pt4-step-shot` in $vkShots. Add a PT4 case to particles_test.cpp (StepParticles == hand-composed PTn
stages, steady-state churn, determinism).

## Constraints (HARD)
- particles.h APPEND-ONLY (PT1-PT3 byte-frozen). fpx.h/grain.h READ-ONLY. Pure Q16.16/integer (NO float/rand/
  clock). NO particle-particle interaction (v1). The sub-stages are the EXISTING PTn functions called verbatim.
- **Scene IDENTICAL in main.cpp AND visual_test.mm (set the SAME literal values).**
- Branch `fix-issue-19-pt4`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
  Register the golden in verify.ps1.
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`).
- COMPLETION CRITERIA — do NOT commit until: `--pt4-step-shot` runs exit 0, all 4 proof lines print with
  GPU==CPU BIT-EXACT, died>0 (churn), composition holds. (The CONTROLLER will pixel-compare Metal vs Vulkan —
  MUST be 0-diff — and eyeball the PNG to confirm the composed effect: a curved, bouncing, pooling stream.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
