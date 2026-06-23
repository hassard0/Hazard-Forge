# Slice PT2 — Deterministic integer force fields (Issue #19, flagship #19, 2nd slice)

APPEND to `engine/sim/particles.h` (PT1 is BYTE-FROZEN — do NOT modify any existing PT1 line; sim headers
fpx.h/grain.h READ-ONLY). PT2 adds deterministic integer FORCE FIELDS (point attractor/repeller, vortex,
uniform wind) accumulated per-particle and folded into the integrate. Per-particle INDEPENDENT (no grid, no
atomics, NO TDR). int64 → the new shader is Vulkan-only; Metal runs the CPU reference byte-identical by
construction (the GR3/FL4/PT1 split).

## Append to particles.h
```cpp
inline constexpr uint32_t kFieldPoint  = 0u;  // attractor (strength>0) / repeller (strength<0), radial falloff
inline constexpr uint32_t kFieldVortex = 1u;  // swirl about `axis` through `center`, tangential, radial falloff
inline constexpr uint32_t kFieldWind   = 2u;  // uniform constant force = axis*strength (no falloff, no radius)

struct ForceField {
    uint32_t kind = kFieldPoint;
    FxVec3   center{};     // Q16.16 field origin (point/vortex)
    FxVec3   axis{};       // Q16.16 unit-ish: vortex spin axis, OR wind direction (host-snapped unit)
    fx       strength = 0; // Q16.16 force magnitude (signed for point: + attract, - repel)
    fx       radius = 0;   // Q16.16 influence radius (point/vortex); ignored for wind
};

// AccumulateForce(p, fields, count): sum the force on particle p over the fields in FIXED ARRAY ORDER.
// PURE int64 Q16.16. Per field:
//   kFieldPoint:  d = center - p.pos; dist = FxLength(d); if (0 < dist < radius) {
//                   dir = FxNormalize(d); falloff = fxdiv(radius - dist, radius);   // 1 at center -> 0 at edge
//                   force += FxScale(dir, fxmul(strength, falloff)); }              // strength sign = attract/repel
//   kFieldVortex: r = p.pos - center; rPerp = r - FxScale(axis, fxdot(r, axis));    // component perpendicular to axis
//                 dist = FxLength(rPerp); if (0 < dist < radius) {
//                   tan = FxCross(axis, rPerp); if (FxLength(tan) > 0) {
//                     dir = FxNormalize(tan); falloff = fxdiv(radius - dist, radius);
//                     force += FxScale(dir, fxmul(strength, falloff)); } }
//   kFieldWind:   force += FxScale(axis, strength);                                  // constant, no falloff
// Returns the accumulated FxVec3. (Use fpx::FxCross/FxDot/FxLength/FxNormalize/fxdiv/fxmul — confirm names.)
inline FxVec3 AccumulateForce(const FxParticle& p, const ForceField* fields, uint32_t count);

// IntegrateParticleWithForce(p, force, gravity, dragK, dt): the PT1 integrate with `force` added to gravity:
//   vel += (gravity + force)*dt; vel -= FxScale(vel, dragK); pos += vel*dt; age += dt; death. The shader
//   copies THIS body VERBATIM. (force==0 -> EXACTLY IntegrateParticle -> the no-op control holds.)
inline void IntegrateParticleWithForce(FxParticle& p, const FxVec3& force, const FxVec3& gravity, fx dragK, fx dt);

inline void IntegrateParticlesWithForces(ParticlePool& pool, const ForceField* fields, uint32_t count,
                                         const FxVec3& gravity, fx dragK, fx dt);   // per-particle: accum + integrate
// One PT2 tick: Emit (host single-thread) -> IntegrateParticlesWithForces (multi-thread) -> RecycleDead -> ++tick.
inline void StepEmitForcesIntegrate(ParticlePool& pool, const EmitterConfig& cfg, const ForceField* fields,
                                    uint32_t count, const FxVec3& gravity, fx dragK, fx dt);
```

## Shader `shaders/particles_forces.comp.hlsl` (NEW, int64, Vulkan-only)
Copy `particles_integrate.comp.hlsl` and insert the AccumulateForce body before the integrate; add a fields
input (a small structured/uniform buffer of ForceField + a count push-constant or a count field). Register it
in the shader-compile manifest like particles_integrate (NOT in the Metal MSL list). The GPU runs accumulate
+ integrate per particle per step (host does Emit/RecycleDead between dispatches, the PT1 loop).

## Showcase `--pt2-forces-shot` (Vulkan) / `--pt2-forces` (Metal)
Clone the PT1 `--pt1-emit-shot` block. Scene: the PT1 fountain (capacity 256, rate 8, speed kOne*4, gravity
(0,-9.8,0), dragK kOne/50, dt kOne/60) PLUS 2 fields: a **vortex** (axis +Y, center a bit above origin,
strength + , radius ~kOne*4) and a **point attractor** (center off to one side, strength + , radius ~kOne*5)
so the stream visibly SWIRLS/curves — distinct from PT1's straight plume. **K so the effect is clear** (e.g.
K=140). REUSE the PT1 viz transform VERBATIM (the fixed-point `worldToPx`: scale RAW Q16.16 pos by
kPxPerUnit=40 via int64, originPxX=imgW/2, originPxY=imgH/3, 240x240, hashColor(seed) dots) — IDENTICAL in
both renderers so the golden is bit-identical cross-backend.

EXACT proof lines (fail loudly):
```
pt2-forces: {fields:F, alive:A, steps:K} GPU==CPU BIT-EXACT
pt2-forces determinism: two runs BYTE-IDENTICAL
pt2-forces no-op: zero fields == PT1 free-fall (forces idle)
pt2-forces effect: vortex deflected the stream by <D> (non-trivial vs no-field control)
```
Assertions: (1) full-buffer memcmp(gpu,cpu)==0 NO tol; (2) two GPU runs memcmp==0; (3) run the SAME scene with
count=0 fields and assert the pool == a PT1 StepEmitIntegrate reference (forces idle when absent); (4) compare
the mean/max particle x-displacement (or a representative deflection metric) WITH vs WITHOUT the fields and
assert the fielded run differs by a non-trivial threshold (the effect is real, not a vacuous no-op).
Strict-zero 2D side-view golden `pt2_forces` registered in verify.ps1 $Goldens + `--pt2-forces-shot` in $vkShots.

## Constraints (HARD)
- particles.h APPEND-ONLY (PT1 byte-frozen). fpx.h/grain.h READ-ONLY. Pure Q16.16/integer (NO float/rand/clock).
  Fields accumulated in FIXED array order. Per-particle independent (multi-thread, NO atomics).
- Branch `fix-issue-19-pt2`, commit there, do NOT merge, do NOT commit `tests/golden/metal/*` (controller bakes).
  Register the golden name in verify.ps1. Add a PT2 case to particles_test.cpp (forces no-op == PT1, vortex
  deflection non-trivial, determinism).
- Build Windows (`cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`,
  vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`).
- COMPLETION CRITERIA — do NOT commit until: `--pt2-forces-shot` runs exit 0, all 4 proof lines print with
  GPU==CPU BIT-EXACT, the no-op control holds, AND the deflection metric is non-trivial. If GPU!=CPU the int64
  force math or field-order is wrong — fix first. (The CONTROLLER will additionally eyeball the golden PNG to
  confirm the swirl is visible — make sure the field strengths produce a clearly curved, on-screen stream.)
- If main.cpp arg-parse hits MSVC C1061, give the flag its own parse loop (do NOT add an else-if to the chain).
