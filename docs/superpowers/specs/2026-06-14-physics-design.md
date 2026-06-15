# Slice S — Minimal Rigid-Body Physics (design spec)

Date: 2026-06-14
Branch: `slice-physics`

## Goal
Broaden Hazard Forge beyond rendering into simulation: a minimal, **real**, **deterministic**,
impulse-based rigid-body engine. Pure C++20, backend-agnostic, depends ONLY on `engine/math/` +
stdlib. Lives in the `hf_core` static lib so it is AddressSanitizer-scoped and unit-testable. A new
showcase steps a world of spheres to rest and renders the settled pile with the EXISTING instanced
pipeline (Slice Q). Existing goldens, pipelines, shaders and scenes are UNTOUCHED.

## Architecture / seam
- `engine/physics/{body.h, world.h, world.cpp}` — NO `vk*`/`MTL`/Metal symbols. The seam grep
  (`vk[A-Z]|MTL|Metal` over rhi/scene/math/physics/asset/render) stays at the master baseline of 12.
- Sources compiled into `hf_core` (sanitized, pure) AND `hf_engine` (live build). Tests link `hf_core`.

## Data model (`body.h`)
```
enum class Shape { Sphere, Box };
struct RigidBody {
  math::Vec3 position;          // center of mass, world space
  math::Quat orientation;       // body->world
  math::Vec3 linVel, angVel;    // world space
  float invMass = 0;            // 0 == static/infinite mass
  math::Vec3 invInertiaDiag;    // body-space inverse inertia (diagonal); 0 == no angular response
  Shape shape = Shape::Sphere;
  float radius = 0.5f;          // sphere collider radius
  math::Vec3 halfExtents{0.5,0.5,0.5}; // box collider half-extents (bonus)
  float restitution = 0.2f;     // bounciness (small so the pile settles)
  float friction    = 0.5f;     // Coulomb friction coefficient
  Mat4 Transform() const;       // FromTRS(position, orientation, scale) for rendering
};
```
- Sphere render scale: the engine sphere mesh has radius 0.5, so the render scale factor is
  `radius / 0.5 == 2*radius` (uniform). Box uses `halfExtents` (cube mesh is a unit ±0.5 cube → scale
  `2*halfExtents`). `Transform()` returns `math::FromTRS`.
- Inertia helpers: `MakeDynamicSphere(pos, radius, density)` fills invMass + solid-sphere
  invInertiaDiag = 1/(2/5 m r^2). Static bodies set invMass=0, invInertiaDiag=0.

## World / integrator (`world.{h,cpp}`)
```
struct World {
  math::Vec3 gravity{0,-9.81,0};
  float groundY = 0.0f;                 // single static plane y=groundY, normal +Y
  std::vector<RigidBody> bodies;
  void Step(float dt);
};
```
`Step(dt)` runs in a FIXED iteration order, no RNG / no time / no `Date`:
1. **Integrate velocities**: dynamic bodies (`invMass>0`) get `linVel += gravity*dt`.
2. **Detect contacts** → manifold list `{a, b(-1 for ground), normal(world, points from A toward B
   for pairs / +Y for ground), penetration, contact point}`:
   - sphere–plane: ground at y=groundY normal +Y. penetration = radius - (pos.y - groundY).
   - sphere–sphere: all i<j pairs; penetration = (rA+rB) - dist; normal = (posB-posA)/dist.
   Fixed order: ground contacts first (body index ascending), then sphere–sphere pairs (i ascending,
   j>i ascending).
3. **Resolve** with sequential impulse, **kIter = 8** iterations over the manifold list in order:
   - Relative velocity at contact (incl. angular: `v + ω×r`). Normal impulse:
     `jn = -(1+e)*vn / (invMassSum + n·(Iinv·(r×n))×r ... )` (full angular effective mass).
   - Restitution `e` is zeroed when the approach speed `|vn|` is below `kRestitutionSlop = 1.0 m/s`
     (Baumgarte-free bounce gate) so resting stacks don't jitter.
   - Accumulated normal impulse clamped `>= 0` (non-penetrating, no sticking).
   - **Friction**: tangent impulse along the tangential relative velocity, Coulomb-clamped to
     `|jt| <= mu * jn` (per-contact accumulated normal). mu = min of the two bodies' friction.
4. **Positional correction** (split-impulse style Baumgarte): after the velocity solve, push bodies
   apart along the normal by `beta * max(penetration - slop, 0)` split by inverse mass,
   `beta = 0.2`, `slop = 0.005`. This removes penetration without injecting velocity (energy-safe).
5. **Integrate positions** (semi-implicit Euler): `position += linVel*dt`. Orientation integrated from
   angular velocity: `q += 0.5 * (ω as quat) * q * dt`, then `Normalize`. Static bodies never move
   (guarded by invMass==0).

## Determinism
No RNG, no clock, fixed dt, fixed iteration order, fixed iteration counts. Two runs on the same
machine/build are **bit-identical**. Cross-platform (MSVC vs clang) need only be *visually* identical
at the settled rest state — the scenario is chosen so every body comes fully to rest (velocities → 0),
which is robust to tiny FP ordering differences.

## Math additions (`math/math.h`)
- `Vec3 operator-(const Vec3&)` (unary negate), `Vec3 operator/(Vec3,float)`.
- `Quat` integration helper `IntegrateOrientation(q, omega, dt)` → normalized quaternion advanced by
  the body angular velocity (`q + 0.5*(0,ω)*q*dt`, renormalized). Keeps the integrator out of physics
  internals and reusable. (Only added if not already present.)

## Unit tests (`tests/physics_test.cpp`, pure C++ / hf_core / ASan)
- A sphere dropped from height falls under gravity and comes to rest at `y ≈ radius` (tol) and STAYS
  (|linVel| ≈ 0) after N steps.
- Two overlapping spheres separate (final distance ≥ rA+rB - slop) and do NOT explode (final kinetic
  energy bounded — no energy injection).
- A static body (invMass 0) never moves regardless of contacts.
- Determinism: two identical worlds stepped the same number of times are bit-identical.

## Showcase + verification (NEW golden only)
- Vulkan: `hello_triangle.exe --physics-shot <path>`. Build a `physics::World` with ground plane +
  a deterministic pyramid/grid of spheres dropped from a modest height; `Step` a FIXED 240 times at
  `dt = 1/120` (fully settles); upload one instance transform per body (`body.Transform()`), render
  with the EXISTING instanced lit + instanced shadow pipelines over ground + skybox. Capture → BMP.
  Visually verify: spheres rest in a plausible pile ON the ground (not floating / sunk / exploded).
- Metal: same scenario in `metal_headless/visual_test.mm` (`--physics`). New committed golden
  `tests/golden/metal/physics.png` (two runs DIFF 0.0000). Existing 5 goldens UNCHANGED.

## Constants
gravity (0,-9.81,0); dt 1/120; steps 240; restitution 0.2; restitution slop 1.0 m/s; friction 0.5;
solver iterations 8; Baumgarte beta 0.2; penetration slop 0.005.
