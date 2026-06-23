# Slice PT3 — Deterministic GPU particle collisions (Issue #19, flagship #19, 3rd slice)

APPEND to `engine/sim/particles.h` (PT1+PT2 BYTE-FROZEN — do NOT modify any existing line). PT3 adds
particle-vs-world collision: a ground plane + static spheres, with a deterministic velocity bounce
(restitution). Ports the grain collision MATH (grain.h:533-561) onto FxParticle (fixed particle radius) +
adds the velocity reflection grains don't need (grains are position-based; particles carry explicit vel).
Pure Q16.16/integer, per-particle independent (no grid, NO TDR). int64 → shader Vulkan-only, Metal CPU ref.

## Append to particles.h
```cpp
// A static sphere collider (the GrainSphereCollider shape). center/radius Q16.16.
struct ParticleSphereCollider { FxVec3 center{}; fx radius = 0; };

inline constexpr fx kParticleRadius   = kOne / 4;   // 0.25 world units (the particle's collision radius)
inline constexpr fx kParticleRestitution = kOne / 2; // 0.5 bounce (Q16.16); 0 = stick, kOne = elastic
inline constexpr fx kCollideEps = kOne / 256;       // containment tolerance for the proof

// CollideParticlePlane: clamp the particle SURFACE to the ground (pos.y >= groundY + kParticleRadius) AND
// reflect the downward velocity: if (pos.y < restY) { pos.y = restY; if (vel.y < 0) vel.y = -fxmul(e, vel.y); }
// (frictionless — tangential vel.x/vel.z unchanged). Pure integer.
inline void CollideParticlePlane(FxParticle& p, fx groundY, fx radius, fx e);

// CollideParticleSphere: project the particle CENTRE out of a static sphere (surfaces touch at sphereR+radius)
// AND reflect the inward velocity. int32 AABB reject (against surf) -> if dist < surf: nrm = FxNormalize(d)
// (d = pos - center; d==0 -> +Y fallback); pos = center + nrm*surf; vn = FxDot(vel, nrm); if (vn < 0)
// vel -= FxScale(nrm, fxmul(kOne + e, vn)). Returns true iff it was a contact. (Mirrors CollideGrainSphere +
// the bounce.)
inline bool CollideParticleSphere(FxParticle& p, const ParticleSphereCollider& s, fx radius, fx e);

// CollideParticleWorld: for each ALIVE particle (index order), CollideParticlePlane then each sphere (fixed
// order) CollideParticleSphere. Returns the contact count. Per-particle independent.
inline int CollideParticleWorld(ParticlePool& pool, fx groundY, fx radius, fx e,
                                const ParticleSphereCollider* spheres, uint32_t sphereCount);

// One PT3 tick: Emit (host single-thread) -> IntegrateParticles (gravity+drag, PT1) -> CollideParticleWorld
// (multi-thread) -> RecycleDead -> ++tick. (NO force fields here — PT3 is collision; PT4 composes all.)
inline void StepEmitIntegrateCollide(ParticlePool& pool, const EmitterConfig& cfg, const FxVec3& gravity,
                                     fx dragK, fx dt, fx groundY, fx radius, fx e,
                                     const ParticleSphereCollider* spheres, uint32_t sphereCount);
```

## Shader `shaders/particles_collide.comp.hlsl` (NEW, int64, Vulkan-only)
Clone particles_integrate.comp; after the integrate, run the collide math (plane + a loop over a colliders
StructuredBuffer) per particle, copying CollideParticlePlane/CollideParticleSphere VERBATIM. Register in the
shader-compile manifest (NOT Metal MSL list). The GPU per step: integrate + collide (host does Emit/RecycleDead
between dispatches). The CPU StepEmitIntegrateCollide is the bit-exact reference.

## Showcase `--pt3-collide-shot` (Vulkan) / `--pt3-collide` (Metal)
**The scene + recipe MUST be byte-identical in BOTH main.cpp and visual_test.mm (the PT2 lesson — a mismatch
passes GPU==CPU + two-run but breaks the cross-backend golden).** Scene: the PT1 emitter raining DOWN onto a
ground plane (groundY = -kOne*2) + 1-2 static spheres below the emitter, so particles fall, bounce, and pool.
Emitter origin above the spheres; gravity (0,-9.8,0); restitution kOne/2. **K large enough to show pooling/
bounce AND lifetime churn** (e.g. lifetime kOne*3=180 ticks, K=220 — so particles die + recycle: died>0). Viz:
REUSE the PT1 fixed-point worldToPx VERBATIM (identical both renderers), PLUS draw the ground line + each
sphere as an integer circle outline, so the colliders are visible. hashColor(seed) dots for particles.

EXACT proof lines (fail loudly):
```
pt3-collide: {alive:A, colliders:C, contacts:X, steps:K} GPU==CPU BIT-EXACT
pt3-collide determinism: two runs BYTE-IDENTICAL
pt3-collide containment: no particle below groundY+radius or inside a collider (within kCollideEps)
pt3-collide no-op: a particle clear of all colliders == PT1 free-fall
```
Assertions: (1) full-buffer memcmp(gpu,cpu)==0 NO tol; (2) two GPU runs memcmp==0; (3) scan all ALIVE
particles: assert pos.y >= groundY + radius - kCollideEps AND for each sphere dist(pos,center) >= sphereR +
radius - kCollideEps (nothing penetrates); (4) a control particle launched away from all colliders matches a
PT1 StepEmitIntegrate reference (collision idle when clear). Strict-zero 2D golden `pt3_collide` in verify.ps1
$Goldens + `--pt3-collide-shot` in $vkShots. Add a PT3 case to particles_test.cpp (bounce reflects vel, sphere
projection, containment, no-op-when-clear, determinism).

## Constraints (HARD)
- particles.h APPEND-ONLY (PT1/PT2 byte-frozen). fpx.h/grain.h READ-ONLY. Pure Q16.16/integer (NO float/rand/
  clock). Per-particle independent. Velocity reflection uses FxDot/FxScale (the PT2 locals / fpx) — int64.
- **The scene/recipe/colliders MUST be IDENTICAL in main.cpp AND visual_test.mm (set the SAME literal values).**
- Branch `fix-issue-19-pt3`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
  Register the golden in verify.ps1.
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`).
- COMPLETION CRITERIA — do NOT commit until: `--pt3-collide-shot` runs exit 0, all 4 proof lines print with
  GPU==CPU BIT-EXACT, containment holds, no-op control holds. (The CONTROLLER will pixel-compare the Metal
  golden vs the Vulkan render — they MUST be 0-diff, so keep the two scenes byte-identical — AND eyeball the
  PNG to confirm particles visibly bounce/pool on the ground + spheres.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop.
