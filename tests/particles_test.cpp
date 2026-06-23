// Slice PT1 — Deterministic GPU Particles: the Q16.16 PARTICLE POOL EMITTER + INTEGRATOR core
// (engine/sim/particles.h) that the GPU shaders/particles_integrate.comp.hlsl copies VERBATIM + proves
// bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::particles.
// The GR1 integer-beachhead twin, with the EMITTER + FREE-LIST delta: a deterministic free-list (LIFO spawn,
// ascending recycle, NO atomic cursor) on top of the per-particle semi-implicit-Euler + linear-drag integrate.
//
// What this test PINS (the contracts the GPU particles_integrate.comp + the GPU==CPU proof build on):
//   * FxParticle: a 48-byte std430 packing (memcmp-able).
//   * InitParticlePool: all slots empty (seed 0, flags 0), the free-list DESCENDING so the first pop yields slot 0.
//   * ParticleHash: deterministic + non-trivial (distinct spawn indices -> distinct hashes).
//   * EmitParticle: pops the LIFO free-list (slot 0 first), writes an ALIVE particle (origin/vel/lifetime/seed),
//     advances spawnCursor exactly once; a full pool -> kNoSlot no-op.
//   * IntegrateParticle: one-step + K-step semi-implicit-Euler + linear-drag closed form (hand-checked integer
//     recurrence); age advances; age >= lifetime clears ALIVE (death).
//   * IntegrateParticles ORDER-INDEPENDENCE (shuffled vs in-order -> identical, particles independent in PT1).
//   * RecycleDead: ASCENDING-slot order, only dead-this-step slots returned, seed cleared.
//   * StepEmitIntegrate: determinism (two runs byte-identical) + the free-list invariant (alive == spawned-died,
//     freeList.size() == capacity - alive) + the no-emit no-op (pool stays empty).
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "sim/particles.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace pt = hf::sim::particles;
using pt::fx;
using pt::kOne;
using pt::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << kFrac); }

// Count alive particles in a packed vector (the showcase's CountAlive twin).
static uint32_t Alive(const pt::ParticlePool& pool) { return pt::CountAlive(pool); }

int main() {
    HF_TEST_MAIN_INIT();

    // ===== the std430 packing: 48 bytes, NO padding holes (memcmp-able) =====
    {
        check(sizeof(pt::FxParticle) == 48, "FxParticle is 48 bytes (the 12 x int32 std430 packing)");
    }

    // ================= InitParticlePool: all slots empty, free-list DESCENDING (first pop -> slot 0) =====
    {
        const uint32_t cap = 8;
        pt::ParticlePool pool = pt::InitParticlePool(cap);
        check(pool.particles.size() == cap, "InitParticlePool: capacity slots allocated");
        check(pool.freeList.size() == cap, "InitParticlePool: free-list has capacity entries");
        check(pool.spawnCursor == 0 && pool.tick == 0, "InitParticlePool: spawnCursor/tick zeroed");
        bool allEmpty = true;
        for (const pt::FxParticle& p : pool.particles)
            if (p.seed != 0 || p.flags != 0) allEmpty = false;
        check(allEmpty, "InitParticlePool: every slot empty (seed 0, flags 0)");
        // Free-list is [cap-1, cap-2, ..., 1, 0]; back() == 0 so the first pop yields slot 0 (ascending birth).
        check(pool.freeList.back() == 0, "InitParticlePool: free-list back is slot 0 (first spawn -> slot 0)");
        check(pool.freeList.front() == cap - 1, "InitParticlePool: free-list front is slot cap-1");
    }

    // ================= ParticleHash: deterministic + distinct spawn indices give distinct hashes ==========
    {
        check(pt::ParticleHash(1, 7) == pt::ParticleHash(1, 7), "ParticleHash: deterministic (same input -> same)");
        // A handful of distinct spawn indices -> distinct hashes (no trivial collisions in a small sweep).
        bool distinct = true;
        for (uint32_t a = 0; a < 16 && distinct; ++a)
            for (uint32_t b = a + 1; b < 16; ++b)
                if (pt::ParticleHash(1, a) == pt::ParticleHash(1, b)) { distinct = false; break; }
        check(distinct, "ParticleHash: distinct spawn indices -> distinct hashes (small sweep)");
    }

    // ================= EmitParticle: LIFO free-list pop (slot 0 first), ALIVE write, cursor++ =============
    {
        pt::ParticlePool pool = pt::InitParticlePool(4);
        pt::EmitterConfig cfg;
        cfg.origin = pt::FxVec3{FromInt(2), FromInt(5), FromInt(-1)};
        cfg.ratePerTick = (fx)1;
        cfg.lifetime = kOne * 2;
        cfg.speed = kOne * 3;
        cfg.emitterId = 1u;

        const uint32_t s0 = pt::EmitParticle(pool, cfg);
        check(s0 == 0, "EmitParticle: first spawn lands in slot 0 (LIFO back of descending free-list)");
        check(pool.spawnCursor == 1, "EmitParticle: spawnCursor advanced exactly once");
        check(pool.freeList.size() == 3, "EmitParticle: one slot popped from the free-list");
        const pt::FxParticle& p0 = pool.particles[0];
        check(p0.flags & pt::kFlagAlive, "EmitParticle: spawned particle is ALIVE");
        check(p0.seed != 0, "EmitParticle: spawned particle has a non-zero seed");
        check(p0.pos.x == cfg.origin.x && p0.pos.y == cfg.origin.y && p0.pos.z == cfg.origin.z,
              "EmitParticle: spawned at the emitter origin");
        check(p0.age == 0 && p0.lifetime == cfg.lifetime, "EmitParticle: age 0, lifetime from config");
        // The initial velocity is EmitInitialVelocity(seed, speed) — the +Y-biased fountain (vel.y > 0).
        check(p0.vel.y > 0, "EmitParticle: initial velocity biased +Y (a fountain jet)");

        const uint32_t s1 = pt::EmitParticle(pool, cfg);
        check(s1 == 1, "EmitParticle: second spawn lands in slot 1 (ascending birth order)");

        // Fill the pool, then a further emit is a deterministic no-op returning kNoSlot.
        pt::EmitParticle(pool, cfg);
        pt::EmitParticle(pool, cfg);
        check(pool.freeList.empty(), "EmitParticle: pool full -> free-list empty");
        const uint32_t sFull = pt::EmitParticle(pool, cfg);
        check(sFull == pt::kNoSlot, "EmitParticle: full pool -> kNoSlot no-op");
        check(Alive(pool) == 4, "EmitParticle: exactly capacity particles alive when full");
    }

    // ================= a FREE particle: one-step semi-implicit-Euler + linear-drag closed form ============
    {
        const fx g  = FromInt(-10);    // gravity -10 (exact in Q16.16)
        const fx dt = kOne / 2;        // dt = 0.5 (exact)
        const fx drag = 0;             // NO drag first (pure integrate, matches the grain closed form)
        const pt::FxVec3 grav{0, g, 0};

        // One step (no drag): vel.y = 0 + g*dt = -10*0.5 = -5; pos.y = 100 + vel.y*dt = 100 + (-5)*0.5 = 97.5.
        pt::FxParticle p;
        p.pos = pt::FxVec3{0, FromInt(100), 0};
        p.vel = pt::FxVec3{0, 0, 0};
        p.age = 0; p.lifetime = kOne * 100; p.seed = 1u; p.flags = pt::kFlagAlive;
        pt::IntegrateParticle(p, grav, drag, dt);
        check(p.vel.y == FromInt(-5), "IntegrateParticle: one step vel.y == g*dt (no drag)");
        check(p.pos.y == FromInt(100) - kOne / 2 * 5, "IntegrateParticle: one step pos.y == 97.5 (no drag)");
        check(p.age == dt, "IntegrateParticle: age advanced by dt");
        check(p.flags & pt::kFlagAlive, "IntegrateParticle: still alive (age < lifetime)");
    }

    // ================= linear drag: vel -= vel*dragK after the gravity add =============================
    {
        const fx dt = kOne;            // dt = 1.0 (exact)
        const fx drag = kOne / 4;      // dragK 0.25
        const pt::FxVec3 grav{0, 0, 0};  // NO gravity -> isolate the drag term
        pt::FxParticle p;
        p.pos = pt::FxVec3{0, 0, 0};
        p.vel = pt::FxVec3{FromInt(8), 0, 0};   // vel.x = 8
        p.age = 0; p.lifetime = kOne * 100; p.seed = 1u; p.flags = pt::kFlagAlive;
        pt::IntegrateParticle(p, grav, drag, dt);
        // vel.x = 8 - 8*0.25 = 6; pos.x = 0 + 6*1 = 6.
        check(p.vel.x == FromInt(6), "IntegrateParticle: linear drag vel.x == vel*(1-dragK)");
        check(p.pos.x == FromInt(6), "IntegrateParticle: pos integrates the post-drag velocity");
    }

    // ================= death: age >= lifetime clears ALIVE =============================================
    {
        const fx dt = kOne;
        const pt::FxVec3 grav{0, 0, 0};
        pt::FxParticle p;
        p.pos = pt::FxVec3{0, 0, 0}; p.vel = pt::FxVec3{0, 0, 0};
        p.age = 0; p.lifetime = kOne * 2; p.seed = 1u; p.flags = pt::kFlagAlive;
        pt::IntegrateParticle(p, grav, 0, dt);   // age 1, alive
        check(p.flags & pt::kFlagAlive, "IntegrateParticle: alive at age 1 < lifetime 2");
        pt::IntegrateParticle(p, grav, 0, dt);   // age 2 >= lifetime 2 -> dead
        check(!(p.flags & pt::kFlagAlive), "IntegrateParticle: dead at age == lifetime");
        // A dead particle is a no-op on the next integrate (the alive-bit guard) -> byte-stable.
        pt::FxParticle before = p;
        pt::IntegrateParticle(p, grav, 0, dt);
        check(std::memcmp(&before, &p, sizeof(pt::FxParticle)) == 0,
              "IntegrateParticle: dead particle is a no-op (byte-stable)");
    }

    // ================= IntegrateParticles ORDER-INDEPENDENCE (particles independent in PT1) ==============
    {
        const fx dt = kOne / 60;
        const fx drag = kOne / 50;
        const pt::FxVec3 grav{0, FromInt(-10), 0};
        pt::ParticlePool pool = pt::InitParticlePool(32);
        pt::EmitterConfig cfg;
        cfg.origin = pt::FxVec3{0, 0, 0}; cfg.ratePerTick = (fx)16; cfg.lifetime = kOne * 5;
        cfg.speed = kOne * 4; cfg.emitterId = 3u;
        pt::Emit(pool, cfg);
        // in-order integrate
        pt::ParticlePool a = pool;
        pt::IntegrateParticles(a, grav, drag, dt);
        // shuffled-order integrate (same set, different visitation) — must match per-particle.
        pt::ParticlePool b = pool;
        std::vector<uint32_t> idx(b.particles.size());
        for (uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
        // a fixed deterministic shuffle (reverse) — particles are independent so the result per slot is identical.
        for (uint32_t i = (uint32_t)idx.size(); i-- > 0;)
            pt::IntegrateParticle(b.particles[(size_t)i], grav, drag, dt);
        check(std::memcmp(a.particles.data(), b.particles.data(),
                          a.particles.size() * sizeof(pt::FxParticle)) == 0,
              "IntegrateParticles: order-independent (reverse == forward, particles independent in PT1)");
    }

    // ================= RecycleDead: ASCENDING-slot, only dead-this-step slots, seed cleared ==============
    {
        pt::ParticlePool pool = pt::InitParticlePool(4);
        pt::EmitterConfig cfg;
        cfg.origin = pt::FxVec3{0, 0, 0}; cfg.ratePerTick = (fx)4; cfg.lifetime = kOne; cfg.speed = kOne * 2;
        cfg.emitterId = 5u;
        pt::Emit(pool, cfg);                       // slots 0..3 alive, free-list empty
        check(pool.freeList.empty(), "RecycleDead setup: pool full");
        // Kill slots 1 and 3 (clear ALIVE, keep seed != 0 -> a died-this-step slot).
        pool.particles[1].flags &= ~pt::kFlagAlive;
        pool.particles[3].flags &= ~pt::kFlagAlive;
        pt::RecycleDead(pool);
        // ASCENDING order: slot 1 pushed before slot 3 -> free-list (LIFO) is [1, 3], back() == 3.
        check(pool.freeList.size() == 2, "RecycleDead: two dead slots returned to the free-list");
        check(pool.freeList[0] == 1 && pool.freeList[1] == 3,
              "RecycleDead: ascending-slot push order (1 before 3)");
        check(pool.particles[1].seed == 0 && pool.particles[3].seed == 0,
              "RecycleDead: recycled slots have seed cleared (marked empty)");
        check(pool.particles[0].seed != 0 && pool.particles[2].seed != 0,
              "RecycleDead: live slots untouched");
        // The next spawn re-uses slot 3 (LIFO back) deterministically.
        const uint32_t reuse = pt::EmitParticle(pool, cfg);
        check(reuse == 3, "RecycleDead: next spawn re-uses the LIFO back (slot 3)");
    }

    // ================= StepEmitIntegrate: determinism + the free-list invariant + the no-emit no-op =======
    {
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        const fx dt = kOne / 60, drag = kOne / 50;
        const pt::FxVec3 grav{0, kGravY, 0};
        const uint32_t cap = 256;
        const int K = 90;
        pt::EmitterConfig cfg;
        cfg.origin = pt::FxVec3{0, 0, 0}; cfg.ratePerTick = (fx)8; cfg.lifetime = kOne * 2;
        cfg.speed = kOne * 4; cfg.emitterId = 1u;

        auto run = [&](uint32_t& spawned, uint32_t& died) {
            pt::ParticlePool pool = pt::InitParticlePool(cap);
            spawned = 0; died = 0;
            for (int s = 0; s < K; ++s) {
                const uint32_t a0 = Alive(pool);
                pt::Emit(pool, cfg);
                spawned += Alive(pool) - a0;
                const uint32_t a1 = Alive(pool);
                pt::IntegrateParticles(pool, grav, drag, dt);
                died += a1 - Alive(pool);
                pt::RecycleDead(pool);
                ++pool.tick;
            }
            return pool;
        };
        uint32_t sp1 = 0, di1 = 0, sp2 = 0, di2 = 0;
        pt::ParticlePool p1 = run(sp1, di1);
        pt::ParticlePool p2 = run(sp2, di2);
        check(std::memcmp(p1.particles.data(), p2.particles.data(),
                          p1.particles.size() * sizeof(pt::FxParticle)) == 0,
              "StepEmitIntegrate: two runs byte-identical (deterministic)");
        const uint32_t alive = Alive(p1);
        check(alive == sp1 - di1, "StepEmitIntegrate: alive == spawned - died (lifecycle balanced)");
        check((uint32_t)p1.freeList.size() == cap - alive,
              "StepEmitIntegrate: freeList.size() == capacity - alive (free-list invariant)");

        // No-emit no-op: an empty pool stepped K times (NO Emit) stays all-empty + byte-stable.
        pt::ParticlePool empty = pt::InitParticlePool(cap);
        pt::ParticlePool empty0 = empty;
        for (int s = 0; s < K; ++s) {
            pt::IntegrateParticles(empty, grav, drag, dt);
            pt::RecycleDead(empty);
            ++empty.tick;
        }
        check(Alive(empty) == 0, "StepEmitIntegrate: no-emit -> 0 alive");
        check(std::memcmp(empty0.particles.data(), empty.particles.data(),
                          empty0.particles.size() * sizeof(pt::FxParticle)) == 0,
              "StepEmitIntegrate: no-emit pool byte-stable (the no-op control)");
    }

    // ================= Slice PT2: FORCE FIELDS — zero-fields==PT1, vortex deflection, determinism =========
    {
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        const fx dt = kOne / 60, drag = kOne / 50;
        const pt::FxVec3 grav{0, kGravY, 0};
        const uint32_t cap = 256;
        const int K = 140;
        pt::EmitterConfig cfg;
        cfg.origin = pt::FxVec3{0, 0, 0}; cfg.ratePerTick = (fx)8; cfg.lifetime = kOne * 2;
        cfg.speed = kOne * 4; cfg.emitterId = 1u;

        // The two PT2 fields (== the showcase): a vortex (axis +Y) + a point attractor off to +x.
        std::vector<pt::ForceField> fields(2);
        fields[0].kind = pt::kFieldVortex;
        fields[0].center = pt::FxVec3{0, kOne * 3 / 2, 0};
        fields[0].axis = pt::FxVec3{0, kOne, 0};
        fields[0].strength = kOne * 6;
        fields[0].radius = kOne * 4;
        fields[1].kind = pt::kFieldPoint;
        fields[1].center = pt::FxVec3{kOne * 2, kOne * 2, 0};
        fields[1].axis = pt::FxVec3{0, 0, 0};
        fields[1].strength = kOne * 5;
        fields[1].radius = kOne * 5;

        // --- IntegrateParticleWithForce with force==0 is BYTE-IDENTICAL to PT1's IntegrateParticle ---
        {
            pt::FxParticle a;
            a.pos = pt::FxVec3{FromInt(1), FromInt(3), FromInt(-2)};
            a.vel = pt::FxVec3{FromInt(2), FromInt(1), 0};
            a.age = 0; a.lifetime = kOne * 100; a.seed = 7u; a.flags = pt::kFlagAlive;
            pt::FxParticle b = a;
            pt::IntegrateParticle(a, grav, drag, dt);
            pt::IntegrateParticleWithForce(b, pt::FxVec3{0, 0, 0}, grav, drag, dt);
            check(std::memcmp(&a, &b, sizeof(pt::FxParticle)) == 0,
                  "PT2 IntegrateParticleWithForce(force=0) == PT1 IntegrateParticle (byte-identical)");
        }

        // --- AccumulateForce idle when count==0 (no fields -> zero force) ---
        {
            pt::FxParticle p;
            p.pos = pt::FxVec3{kOne, kOne, 0}; p.flags = pt::kFlagAlive; p.seed = 1u;
            const pt::FxVec3 f = pt::AccumulateForce(p, fields.data(), 0u);
            check(f.x == 0 && f.y == 0 && f.z == 0, "PT2 AccumulateForce(count=0) is zero (forces idle when absent)");
        }

        // --- zero fields == PT1 free-fall: StepEmitForcesIntegrate(count=0) == StepEmitIntegrate ---
        {
            pt::ParticlePool zeroPool = pt::InitParticlePool(cap);
            for (int s = 0; s < K; ++s)
                pt::StepEmitForcesIntegrate(zeroPool, cfg, fields.data(), 0u, grav, drag, dt);
            pt::ParticlePool pt1Pool = pt::InitParticlePool(cap);
            for (int s = 0; s < K; ++s)
                pt::StepEmitIntegrate(pt1Pool, cfg, grav, drag, dt);
            check(std::memcmp(zeroPool.particles.data(), pt1Pool.particles.data(),
                              (size_t)cap * sizeof(pt::FxParticle)) == 0,
                  "PT2 StepEmitForcesIntegrate(count=0) == PT1 StepEmitIntegrate (zero fields == free-fall)");
        }

        // --- determinism: two fielded runs byte-identical ---
        auto runFielded = [&]() {
            pt::ParticlePool pool = pt::InitParticlePool(cap);
            for (int s = 0; s < K; ++s)
                pt::StepEmitForcesIntegrate(pool, cfg, fields.data(), (uint32_t)fields.size(), grav, drag, dt);
            return pool;
        };
        pt::ParticlePool f1 = runFielded();
        pt::ParticlePool f2 = runFielded();
        check(std::memcmp(f1.particles.data(), f2.particles.data(),
                          (size_t)cap * sizeof(pt::FxParticle)) == 0,
              "PT2 StepEmitForcesIntegrate: two runs byte-identical (deterministic)");

        // --- vortex deflection non-trivial: fielded mean |x| clearly exceeds the no-field control ---
        {
            pt::ParticlePool ctrl = pt::InitParticlePool(cap);
            for (int s = 0; s < K; ++s)
                pt::StepEmitIntegrate(ctrl, cfg, grav, drag, dt);
            auto meanAbsX = [](const pt::ParticlePool& p) -> double {
                int64_t sum = 0; uint32_t n = 0;
                for (const pt::FxParticle& q : p.particles)
                    if (q.flags & pt::kFlagAlive) { sum += (q.pos.x < 0 ? -(int64_t)q.pos.x : q.pos.x); ++n; }
                if (n == 0) return 0.0;
                return (double)sum / (double)n / (double)kOne;
            };
            const double deflect = meanAbsX(f1) - meanAbsX(ctrl);
            check(deflect > 0.10, "PT2 vortex+attractor deflection non-trivial (mean |x| > no-field control + 0.1)");
        }
    }

    // ================= Slice PT3: COLLISIONS — bounce, sphere projection, containment, no-op, determinism ==
    {
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        const fx dt = kOne / 60, drag = kOne / 50;
        const pt::FxVec3 grav{0, kGravY, 0};
        const fx groundY = -kOne * 2;
        const fx radius = pt::kParticleRadius;       // 0.25
        const fx e = pt::kParticleRestitution;       // 0.5

        // --- CollideParticlePlane: clamps the surface to groundY+radius AND reflects+damps the downward vel ---
        {
            pt::FxParticle p{};
            p.pos = pt::FxVec3{0, groundY - kOne, 0};   // a full unit BELOW the plane
            p.vel = pt::FxVec3{kOne, -kOne * 2, 0};     // moving down at 2 (+ a tangential x)
            p.flags = pt::kFlagAlive; p.seed = 1u; p.lifetime = kOne * 100;
            const fx vxBefore = p.vel.x;
            pt::CollideParticlePlane(p, groundY, radius, e);
            check(p.pos.y == groundY + radius, "PT3 plane: surface clamped to groundY+radius");
            check(p.vel.y == -pt::fxmul(e, -kOne * 2), "PT3 plane: downward vel reflected+damped (vel.y = -e*vy)");
            check(p.vel.y > 0, "PT3 plane: reflected vel points UP");
            check(p.vel.x == vxBefore, "PT3 plane: tangential vel unchanged (frictionless)");
        }

        // --- CollideParticlePlane: an UPWARD-moving particle below the plane is clamped but vel is NOT flipped ---
        {
            pt::FxParticle p{};
            p.pos = pt::FxVec3{0, groundY - kOne / 2, 0};
            p.vel = pt::FxVec3{0, kOne, 0};             // already moving UP
            p.flags = pt::kFlagAlive; p.seed = 1u; p.lifetime = kOne * 100;
            pt::CollideParticlePlane(p, groundY, radius, e);
            check(p.pos.y == groundY + radius, "PT3 plane: upward particle still clamped");
            check(p.vel.y == kOne, "PT3 plane: upward vel NOT reflected (vel.y >= 0)");
        }

        // --- CollideParticleSphere: projects the centre out to sphereR+radius AND reflects the inward vel ---
        {
            pt::ParticleSphereCollider s; s.center = pt::FxVec3{0, 0, 0}; s.radius = kOne;
            pt::FxParticle p{};
            p.pos = pt::FxVec3{kOne / 2, 0, 0};         // inside: dist 0.5 < surf 1.25
            p.vel = pt::FxVec3{-kOne, 0, 0};            // moving INTO the sphere (toward center along -x)
            p.flags = pt::kFlagAlive; p.seed = 1u; p.lifetime = kOne * 100;
            const bool hit = pt::CollideParticleSphere(p, s, radius, e);
            check(hit, "PT3 sphere: contact detected (particle inside the expanded sphere)");
            // Projected out along +x to surf = 1.25 -> pos.x == 1.25.
            check(p.pos.x == kOne + radius, "PT3 sphere: centre projected to sphereR+radius along the normal");
            // dist(pos, center) == surf now (on the surface).
            const fx surf = s.radius + radius;
            const pt::FxVec3 d{p.pos.x - s.center.x, p.pos.y - s.center.y, p.pos.z - s.center.z};
            check(pt::FxLength(d) >= surf - pt::kCollideEps, "PT3 sphere: projected particle on/outside the surface");
            // The inward vel (-x, vn<0) is reflected -> now points OUTWARD (+x).
            check(p.vel.x > 0, "PT3 sphere: inward vel reflected to point outward");
        }

        // --- CollideParticleSphere: a particle CLEAR of the sphere is untouched (no contact) ---
        {
            pt::ParticleSphereCollider s; s.center = pt::FxVec3{0, 0, 0}; s.radius = kOne;
            pt::FxParticle p{};
            p.pos = pt::FxVec3{kOne * 5, 0, 0};         // far outside
            p.vel = pt::FxVec3{-kOne, 0, 0};
            p.flags = pt::kFlagAlive; p.seed = 1u; p.lifetime = kOne * 100;
            const pt::FxParticle before = p;
            const bool hit = pt::CollideParticleSphere(p, s, radius, e);
            check(!hit, "PT3 sphere: a clear particle is not a contact");
            check(std::memcmp(&p, &before, sizeof(pt::FxParticle)) == 0, "PT3 sphere: clear particle byte-stable");
        }

        // The full PT3 scene (== the showcase): a fountain raining onto a plane + two spheres.
        const uint32_t cap = 220;
        const int K = 220;
        pt::EmitterConfig cfg;
        cfg.origin = pt::FxVec3{0, kOne * 3, 0}; cfg.ratePerTick = (fx)2; cfg.lifetime = kOne * 3;
        cfg.speed = kOne * 2; cfg.emitterId = 1u;
        std::vector<pt::ParticleSphereCollider> spheres(2);
        spheres[0].center = pt::FxVec3{-kOne, 0, 0}; spheres[0].radius = kOne;
        spheres[1].center = pt::FxVec3{kOne * 5 / 4, -kOne / 2, 0}; spheres[1].radius = kOne * 3 / 4;
        const uint32_t sc = (uint32_t)spheres.size();

        auto runScene = [&]() {
            pt::ParticlePool pool = pt::InitParticlePool(cap);
            for (int s = 0; s < K; ++s)
                pt::StepEmitIntegrateCollide(pool, cfg, grav, drag, dt, groundY, radius, e, spheres.data(), sc);
            return pool;
        };

        // --- containment: no ALIVE particle below groundY+radius or inside a collider (within kCollideEps) ---
        {
            pt::ParticlePool pool = runScene();
            bool ok = true;
            const fx restY = groundY + radius;
            for (const pt::FxParticle& p : pool.particles) {
                if (!(p.flags & pt::kFlagAlive)) continue;
                if (p.pos.y < restY - pt::kCollideEps) { ok = false; break; }
                for (uint32_t s = 0; s < sc; ++s) {
                    const fx surf = spheres[s].radius + radius;
                    const pt::FxVec3 d{p.pos.x - spheres[s].center.x, p.pos.y - spheres[s].center.y,
                                       p.pos.z - spheres[s].center.z};
                    if (pt::FxLength(d) < surf - pt::kCollideEps) { ok = false; break; }
                }
                if (!ok) break;
            }
            check(ok, "PT3 containment: no particle below ground or inside a collider (within kCollideEps)");
            check(Alive(pool) > 0, "PT3 scene: particles are alive + pooled");
        }

        // --- determinism: two scene runs byte-identical ---
        {
            pt::ParticlePool a = runScene();
            pt::ParticlePool b = runScene();
            check(std::memcmp(a.particles.data(), b.particles.data(),
                              (size_t)cap * sizeof(pt::FxParticle)) == 0,
                  "PT3 StepEmitIntegrateCollide: two runs byte-identical (deterministic)");
        }

        // --- no-op control: a particle CLEAR of all colliders == PT1 free-fall (collision idle when clear) ---
        {
            pt::FxParticle seed{};
            seed.pos = pt::FxVec3{kOne * 8, kOne * 100, 0};   // drag caps the fall ~y=78 over K, never near ground
            seed.vel = pt::FxVec3{kOne, kOne, 0};
            seed.age = 0; seed.lifetime = kOne * 1000; seed.seed = 123u; seed.flags = pt::kFlagAlive;
            pt::ParticlePool collidePool = pt::InitParticlePool(1);
            collidePool.particles[0] = seed;
            pt::ParticlePool freefallPool = pt::InitParticlePool(1);
            freefallPool.particles[0] = seed;
            pt::EmitterConfig noEmit; noEmit.ratePerTick = 0; noEmit.lifetime = kOne * 1000;
            for (int s = 0; s < K; ++s) {
                pt::StepEmitIntegrateCollide(collidePool, noEmit, grav, drag, dt, groundY, radius, e,
                                             spheres.data(), sc);
                pt::StepEmitIntegrate(freefallPool, noEmit, grav, drag, dt);
            }
            check(std::memcmp(collidePool.particles.data(), freefallPool.particles.data(),
                              sizeof(pt::FxParticle)) == 0,
                  "PT3 no-op: a clear particle == PT1 free-fall (collision idle when clear)");
        }
    }

    // ================= Slice PT4: StepParticles — composition, steady-state churn, determinism ============
    {
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        const fx dt = kOne / 60, drag = kOne / 50;
        const pt::FxVec3 grav{0, kGravY, 0};
        const fx groundY = -kOne * 2;
        const fx radius = pt::kParticleRadius;       // 0.25
        const fx e = pt::kParticleRestitution;       // 0.5

        // The full PT4 scene (== the showcase): a fountain + a vortex field + a plane + two spheres.
        const uint32_t cap = 240;
        const int K = 240;
        pt::EmitterConfig cfg;
        cfg.origin = pt::FxVec3{0, kOne * 3, 0}; cfg.ratePerTick = (fx)2; cfg.lifetime = kOne * 3;
        cfg.speed = kOne * 2; cfg.emitterId = 1u;

        std::vector<pt::ForceField> fields(1);
        fields[0].kind = pt::kFieldVortex;
        fields[0].center = pt::FxVec3{0, kOne, 0};
        fields[0].axis = pt::FxVec3{0, kOne, 0};
        fields[0].strength = kOne * 5;
        fields[0].radius = kOne * 5;
        const uint32_t fc = (uint32_t)fields.size();

        std::vector<pt::ParticleSphereCollider> spheres(2);
        spheres[0].center = pt::FxVec3{-kOne, 0, 0}; spheres[0].radius = kOne;
        spheres[1].center = pt::FxVec3{kOne * 5 / 4, -kOne / 2, 0}; spheres[1].radius = kOne * 3 / 4;
        const uint32_t sc = (uint32_t)spheres.size();

        auto runScene = [&](uint32_t& diedOut) {
            pt::ParticlePool pool = pt::InitParticlePool(cap);
            uint32_t died = 0;
            for (int s = 0; s < K; ++s) {
                pt::Emit(pool, cfg);
                const uint32_t a = pt::CountAlive(pool);
                pt::IntegrateParticlesWithForces(pool, fields.data(), fc, grav, drag, dt);
                pt::CollideParticleWorld(pool, groundY, radius, e, spheres.data(), sc);
                died += a - pt::CountAlive(pool);
                pt::RecycleDead(pool);
                ++pool.tick;
            }
            diedOut = died;
            return pool;
        };

        // --- composition: ONE StepParticles tick == applying the four PTn stages by hand (after a warm-up) ---
        {
            pt::ParticlePool warm = pt::InitParticlePool(cap);
            for (int s = 0; s < K / 2; ++s)
                pt::StepParticles(warm, cfg, fields.data(), fc, grav, drag, dt, groundY, radius, e,
                                  spheres.data(), sc);
            pt::ParticlePool composed = warm;   // copy
            pt::ParticlePool byHand = warm;     // copy
            const int contacts = pt::StepParticles(composed, cfg, fields.data(), fc, grav, drag, dt, groundY,
                                                   radius, e, spheres.data(), sc);
            pt::Emit(byHand, cfg);
            pt::IntegrateParticlesWithForces(byHand, fields.data(), fc, grav, drag, dt);
            const int handContacts = pt::CollideParticleWorld(byHand, groundY, radius, e, spheres.data(), sc);
            pt::RecycleDead(byHand);
            ++byHand.tick;
            check(contacts == handContacts, "PT4 StepParticles: returned contact count == CollideParticleWorld");
            check(composed.particles.size() == byHand.particles.size() &&
                  std::memcmp(composed.particles.data(), byHand.particles.data(),
                              (size_t)cap * sizeof(pt::FxParticle)) == 0 &&
                  composed.freeList == byHand.freeList && composed.spawnCursor == byHand.spawnCursor &&
                  composed.tick == byHand.tick,
                  "PT4 StepParticles == Emit->IntegrateParticlesWithForces->CollideParticleWorld->RecycleDead");
        }

        // --- steady-state churn: died>0 AND the free-list invariant (freeList.size()==capacity-alive) ---
        {
            uint32_t died = 0;
            pt::ParticlePool pool = runScene(died);
            const uint32_t alive = pt::CountAlive(pool);
            check(died > 0, "PT4 steady-state: died>0 (emit/death churn over K ticks)");
            check(alive > 0 && alive <= cap, "PT4 steady-state: alive within the band [1, capacity]");
            check(pool.freeList.size() == (size_t)cap - (size_t)alive,
                  "PT4 free-list invariant: freeList.size() == capacity - alive");
        }

        // --- determinism: two StepParticlesN runs byte-identical ---
        {
            pt::ParticlePool a = pt::InitParticlePool(cap);
            pt::StepParticlesN(a, cfg, fields.data(), fc, grav, drag, dt, groundY, radius, e,
                               spheres.data(), sc, K);
            pt::ParticlePool b = pt::InitParticlePool(cap);
            pt::StepParticlesN(b, cfg, fields.data(), fc, grav, drag, dt, groundY, radius, e,
                               spheres.data(), sc, K);
            check(a.particles.size() == b.particles.size() &&
                  std::memcmp(a.particles.data(), b.particles.data(),
                              (size_t)cap * sizeof(pt::FxParticle)) == 0,
                  "PT4 StepParticlesN: two runs byte-identical (deterministic)");
        }
    }

    // ================= Slice PT5: LOCKSTEP + ROLLBACK — inputs-only equal, rollback corrects, snapshot ====
    // round-trip exact, snapshot-completeness control diverges (the NETCODE HEADLINE). PURE CPU.
    {
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        const fx dt = kOne / 60, drag = kOne / 50;
        const pt::FxVec3 grav{0, kGravY, 0};
        const fx groundY = -kOne * 2;
        const fx radius = pt::kParticleRadius;
        const fx e = pt::kParticleRestitution;

        const uint32_t cap = 240;
        const uint32_t T = 240;
        const uint32_t rollbackAt = 60;
        pt::EmitterConfig cfg0;
        cfg0.origin = pt::FxVec3{0, kOne * 3, 0}; cfg0.ratePerTick = (fx)2; cfg0.lifetime = kOne * 3;
        cfg0.speed = kOne * 2; cfg0.emitterId = 1u;

        std::vector<pt::ForceField> fields(1);
        fields[0].kind = pt::kFieldVortex;
        fields[0].center = pt::FxVec3{0, kOne, 0};
        fields[0].axis = pt::FxVec3{0, kOne, 0};
        fields[0].strength = kOne * 5;
        fields[0].radius = kOne * 5;
        const uint32_t fc = (uint32_t)fields.size();

        std::vector<pt::ParticleSphereCollider> spheres(2);
        spheres[0].center = pt::FxVec3{-kOne, 0, 0}; spheres[0].radius = kOne;
        spheres[1].center = pt::FxVec3{kOne * 5 / 4, -kOne / 2, 0}; spheres[1].radius = kOne * 3 / 4;
        const uint32_t sc = (uint32_t)spheres.size();

        pt::ParticlePool pool0 = pt::InitParticlePool(cap);
        const pt::ParticleSnapshot init = pt::SnapshotParticles(pool0, cfg0);

        const std::vector<pt::ParticleCommand> authStream = {
            pt::ParticleCommand{40,  pt::kCmdGust,        pt::FxVec3{kOne * 3, 0, 0}, 0},
            pt::ParticleCommand{80,  pt::kCmdBurst,       pt::FxVec3{kOne, kOne * 4, 0}, 16},
            pt::ParticleCommand{120, pt::kCmdMoveEmitter, pt::FxVec3{kOne, kOne * 3, 0}, 0},
            pt::ParticleCommand{160, pt::kCmdGust,        pt::FxVec3{0, 0, kOne * 2}, 0},
        };
        const uint32_t cc = (uint32_t)authStream.size();
        std::vector<pt::ParticleCommand> mispredictStream = authStream;
        mispredictStream.push_back(pt::ParticleCommand{rollbackAt, pt::kCmdGust, pt::FxVec3{kOne * 40, 0, 0}, 0});

        auto poolOf = [](const pt::ParticleSnapshot& s) {
            pt::ParticlePool p;
            p.particles = s.particles; p.freeList = s.freeList;
            p.spawnCursor = s.spawnCursor; p.tick = s.tick;
            return p;
        };

        const pt::ParticleSnapshot authority = pt::RunParticleLockstep(
            init, authStream.data(), cc, T, fields.data(), fc, grav, drag, dt, groundY, radius, e,
            spheres.data(), sc);

        // --- (1) lockstep: replica (inputs-only) == authority BIT-EXACT ---
        {
            const pt::ParticleSnapshot replica = pt::RunParticleLockstep(
                init, authStream.data(), cc, T, fields.data(), fc, grav, drag, dt, groundY, radius, e,
                spheres.data(), sc);
            check(pt::ParticleStatesEqual(poolOf(authority), poolOf(replica)),
                  "PT5 lockstep: replica == authority BIT-EXACT (inputs-only re-derivation)");
            check(pt::CountAlive(poolOf(authority)) > 0, "PT5 lockstep: authority pool is alive (non-degenerate)");
        }

        // --- (2) rollback: corrected == authority AND the mispredicted state DIFFERED ---
        {
            const pt::ParticleSnapshot rolledBack = pt::RunParticleRollback(
                init, authStream.data(), cc, mispredictStream.data(), (uint32_t)mispredictStream.size(),
                T, rollbackAt, fields.data(), fc, grav, drag, dt, groundY, radius, e, spheres.data(), sc);
            const pt::ParticleSnapshot mispredicted = pt::RunParticleLockstep(
                init, mispredictStream.data(), (uint32_t)mispredictStream.size(), T, fields.data(), fc, grav,
                drag, dt, groundY, radius, e, spheres.data(), sc);
            check(pt::ParticleStatesEqual(poolOf(rolledBack), poolOf(authority)),
                  "PT5 rollback: corrected == authority BIT-EXACT");
            check(!pt::ParticleStatesEqual(poolOf(mispredicted), poolOf(authority)),
                  "PT5 rollback: mispredicted state DIFFERED from authority (real divergence fixed)");
        }

        // --- (3) snapshot round-trip exact: Restore(Snapshot(p)) == p ---
        {
            pt::ParticleSnapshot p = pt::RunParticleLockstep(
                init, authStream.data(), cc, rollbackAt, fields.data(), fc, grav, drag, dt, groundY, radius, e,
                spheres.data(), sc);
            pt::ParticlePool pool; pt::EmitterConfig cfg;
            pt::RestoreParticles(pool, cfg, p);
            const pt::ParticleSnapshot snap = pt::SnapshotParticles(pool, cfg);
            pt::SimParticleTick(pool, cfg, authStream.data(), cc, fields.data(), fc, grav, drag, dt, groundY,
                                radius, e, spheres.data(), sc);   // mutate
            pt::RestoreParticles(pool, cfg, snap);
            check(pt::ParticleStatesEqual(pool, poolOf(snap)),
                  "PT5 snapshot: RestoreParticles(SnapshotParticles(p)) == p BIT-EXACT");
        }

        // --- (4) snapshot-completeness control: omitting freeList/spawnCursor DIVERGES ---
        {
            const uint32_t mid = 100, tail = 60;
            pt::ParticleSnapshot atMid = pt::RunParticleLockstep(
                init, authStream.data(), cc, mid, fields.data(), fc, grav, drag, dt, groundY, radius, e,
                spheres.data(), sc);
            pt::ParticlePool fullPool; pt::EmitterConfig fullCfg;
            pt::RestoreParticles(fullPool, fullCfg, atMid);
            for (uint32_t t = mid; t < mid + tail; ++t)
                pt::SimParticleTick(fullPool, fullCfg, authStream.data(), cc, fields.data(), fc, grav, drag, dt,
                                    groundY, radius, e, spheres.data(), sc);
            pt::ParticlePool badPool; pt::EmitterConfig badCfg;
            pt::RestoreParticles(badPool, badCfg, atMid);
            badPool.freeList = init.freeList;        // STALE (omit the free-list)
            badPool.spawnCursor = init.spawnCursor;  // STALE (omit the cursor)
            for (uint32_t t = mid; t < mid + tail; ++t)
                pt::SimParticleTick(badPool, badCfg, authStream.data(), cc, fields.data(), fc, grav, drag, dt,
                                    groundY, radius, e, spheres.data(), sc);
            check(!pt::ParticleStatesEqual(fullPool, badPool),
                  "PT5 snapshot-completeness: omit freeList/spawnCursor -> DIVERGES (the crux control)");
        }
    }

    // ================= PT6 ParticleToRenderInstances: provenance (count==alive, pure fn, dead skipped) ====
    // The render helper turns the bit-exact pool into one float model matrix per ALIVE particle (dead/empty
    // slots skipped), a PURE FUNCTION of the pool. The render-only float crossing — render-only, NO sim mutation.
    {
        const uint32_t cap = 256;
        const fx grav_y = -(fx)(98 * kOne / 10);          // -9.8 (exact-ish; the sign/magnitude is irrelevant here)
        const pt::FxVec3 grav{0, grav_y, 0};
        const fx drag = kOne / 64, dt = kOne / 60, groundY = 0;
        const fx radius = pt::kParticleRadius, e = pt::kParticleRestitution;
        const float renderRadius = pt::ParticleToFloat(radius);

        pt::EmitterConfig cfg;
        cfg.origin = pt::FxVec3{0, kOne / 4, 0}; cfg.ratePerTick = (fx)8;
        cfg.lifetime = kOne * 5 / 2; cfg.speed = kOne * 12; cfg.emitterId = 1u;

        std::vector<pt::ForceField> fields(1);
        fields[0].kind = pt::kFieldVortex; fields[0].center = pt::FxVec3{0, 0, 0};
        fields[0].axis = pt::FxVec3{0, kOne, 0}; fields[0].strength = kOne * 6; fields[0].radius = kOne * 8;
        const uint32_t fc = (uint32_t)fields.size();

        std::vector<pt::ParticleSphereCollider> spheres(1);
        spheres[0].center = pt::FxVec3{kOne * 3, kOne * 3 / 2, 0}; spheres[0].radius = kOne * 3 / 2;
        const uint32_t sc = (uint32_t)spheres.size();

        pt::ParticlePool pool = pt::InitParticlePool(cap);
        pt::StepParticlesN(pool, cfg, fields.data(), fc, grav, drag, dt, groundY, radius, e,
                           spheres.data(), sc, 60);
        const uint32_t alive = pt::CountAlive(pool);
        check(alive > 0, "PT6 ParticleToRenderInstances: the test pool has alive particles (non-degenerate)");

        // (1) instance count == alive count (one transform per ALIVE particle, dead/empty slots skipped).
        const std::vector<math::Mat4> mats = pt::ParticleToRenderInstances(pool, renderRadius);
        check((uint32_t)mats.size() == alive,
              "PT6 ParticleToRenderInstances: instance count == alive count (dead/empty slots skipped)");

        // (2) PURE FUNCTION: two calls byte-equal (no RNG, no clock, no sim mutation).
        const std::vector<math::Mat4> mats2 = pt::ParticleToRenderInstances(pool, renderRadius);
        check(mats.size() == mats2.size() &&
              std::memcmp(mats.data(), mats2.data(), mats.size() * sizeof(math::Mat4)) == 0,
              "PT6 ParticleToRenderInstances: two calls BYTE-IDENTICAL (pure function of the pool)");

        // (2b) the call did not mutate the pool (render-only — the sim path stays bit-exact).
        {
            pt::ParticlePool poolCopy = pool;
            (void)pt::ParticleToRenderInstances(poolCopy, renderRadius);
            check(pt::ParticleStatesEqual(pool, poolCopy),
                  "PT6 ParticleToRenderInstances: render call does NOT mutate the pool (render-only)");
        }

        // (3) dead/empty slots skipped: an all-dead pool -> ZERO instances (the empty no-op).
        {
            pt::ParticlePool emptyPool = pt::InitParticlePool(cap);
            check(pt::ParticleToRenderInstances(emptyPool, renderRadius).empty(),
                  "PT6 ParticleToRenderInstances: empty (all-dead) pool -> zero instances (the empty no-op)");
        }

        // (3b) provenance: each transform's translation == ParticleVertToWorld of the corresponding ALIVE pos.
        {
            bool provOk = true;
            size_t mi = 0;
            for (const pt::FxParticle& p : pool.particles) {
                if (!(p.flags & pt::kFlagAlive)) continue;
                const math::Vec3 t = pt::ParticleVertToWorld(p.pos);
                const math::Mat4& m = mats[mi++];
                // column-major translate*scale: translation in m.m[12..14].
                if (m.m[12] != t.x || m.m[13] != t.y || m.m[14] != t.z) { provOk = false; break; }
            }
            check(provOk && mi == mats.size(),
                  "PT6 ParticleToRenderInstances: every transform's translation IS the bit-exact ParticleVertToWorld(pos)");
        }
    }

    if (g_fail == 0) std::printf("particles_test: ALL CHECKS PASSED\n");
    else std::printf("particles_test: %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
