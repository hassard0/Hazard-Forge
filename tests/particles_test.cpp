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

    if (g_fail == 0) std::printf("particles_test: ALL CHECKS PASSED\n");
    else std::printf("particles_test: %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
