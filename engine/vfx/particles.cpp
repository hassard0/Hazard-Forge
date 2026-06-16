// Slice CC — CPU particle / VFX emitter system implementation. See particles.h for the design + the
// determinism contract (fixed-seed LCG jitter + fixed-dt steps => bit-identical state).
#include "vfx/particles.h"

#include <algorithm>

namespace hf::vfx {

// --- Lifetime curves ----------------------------------------------------------------------------
// t = clamp(age/lifetime, 0, 1); guard lifetime<=0 (degenerate emitter) by treating it as "already
// at the end" (t==1) so SizeAt/ColorAt stay finite + deterministic.
static float LifeT(const EmitterConfig&, const Particle& p) {
    if (p.lifetime <= 0.0f) return 1.0f;
    float t = p.age / p.lifetime;
    return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}

float SizeAt(const EmitterConfig& cfg, const Particle& p) {
    return math::lerp(cfg.startSize, cfg.endSize, LifeT(cfg, p));
}

Vec4 ColorAt(const EmitterConfig& cfg, const Particle& p) {
    return math::lerp(cfg.startColor, cfg.endColor, LifeT(cfg, p));
}

// --- Billboard generation -----------------------------------------------------------------------
// Per particle: a camera-facing quad centered on pos, half-extent = SizeAt, spanning cameraRight /
// cameraUp. Corners (in right/up units): (-,-), (+,-), (+,+), (-,+) with matching [0,1] UVs. Two
// triangles, CCW: (0,1,2) and (0,2,3). Color is ColorAt (constant across the quad).
void BuildBillboards(std::span<const Particle> particles, const EmitterConfig& cfg, Vec3 cameraRight,
                     Vec3 cameraUp, std::vector<BillboardVertex>& outVerts) {
    outVerts.reserve(outVerts.size() + particles.size() * 6);
    for (const Particle& p : particles) {
        float h = SizeAt(cfg, p);
        Vec4 col = ColorAt(cfg, p);
        Vec3 r = cameraRight * h;
        Vec3 u = cameraUp * h;
        // Four corners + their UVs.
        const Vec3 c0 = p.pos - r - u;  // (0,0)
        const Vec3 c1 = p.pos + r - u;  // (1,0)
        const Vec3 c2 = p.pos + r + u;  // (1,1)
        const Vec3 c3 = p.pos - r + u;  // (0,1)
        const Vec2 uv0{0.0f, 0.0f};
        const Vec2 uv1{1.0f, 0.0f};
        const Vec2 uv2{1.0f, 1.0f};
        const Vec2 uv3{0.0f, 1.0f};
        // Triangle 1: c0, c1, c2.
        outVerts.push_back({c0, uv0, col});
        outVerts.push_back({c1, uv1, col});
        outVerts.push_back({c2, uv2, col});
        // Triangle 2: c0, c2, c3.
        outVerts.push_back({c0, uv0, col});
        outVerts.push_back({c2, uv2, col});
        outVerts.push_back({c3, uv3, col});
    }
}

// --- ParticleSystem -----------------------------------------------------------------------------
void ParticleSystem::Reset() {
    particles_.clear();
    accumulator_ = 0.0f;
    spawned_ = 0;
    lcg_ = 0;
    seeded_ = false;
}

// LCG advance (same constants as audio::Wave::Noise): state = state*1664525 + 1013904223.
uint32_t ParticleSystem::NextRand(uint32_t seed) {
    if (!seeded_) { lcg_ = seed; seeded_ = true; }
    lcg_ = lcg_ * 1664525u + 1013904223u;
    return lcg_;
}

// Map a fresh LCG draw to [-1, 1]: take the high 24 bits as a [0,1) fraction, then scale to [-1,1).
float ParticleSystem::RandSpread(uint32_t seed) {
    uint32_t r = NextRand(seed) >> 8;            // 24-bit
    float unit = static_cast<float>(r) / 16777216.0f;  // [0,1)
    return unit * 2.0f - 1.0f;                   // [-1,1)
}

void ParticleSystem::Step(const EmitterConfig& cfg, float dt) {
    // 1. SPAWN. nSpawn = floor(spawnRate*dt + accumulator); the fractional remainder carries.
    accumulator_ += cfg.spawnRate * dt;
    int nSpawn = static_cast<int>(accumulator_);  // floor for non-negative
    if (nSpawn < 0) nSpawn = 0;
    accumulator_ -= static_cast<float>(nSpawn);
    const int cap = std::max(0, cfg.maxParticles);
    for (int i = 0; i < nSpawn; ++i) {
        // Draw the 3 jitter values UNCONDITIONALLY (even when capped) so the LCG stream — and thus
        // the spawned particles' velocities — stay deterministic regardless of the cap. Order:
        // x, y, z.
        float jx = RandSpread(cfg.seed) * cfg.velSpread;
        float jy = RandSpread(cfg.seed) * cfg.velSpread;
        float jz = RandSpread(cfg.seed) * cfg.velSpread;
        if (static_cast<int>(particles_.size()) >= cap) continue;  // pool full: drop (jitter consumed)
        Particle p;
        p.pos = cfg.origin;
        p.vel = Vec3{cfg.initVel.x + jx, cfg.initVel.y + jy, cfg.initVel.z + jz};
        p.age = 0.0f;
        p.lifetime = cfg.lifetime;
        particles_.push_back(p);
        ++spawned_;
    }

    // 2. INTEGRATE every live particle (semi-implicit Euler).
    const float dragFactor = 1.0f - cfg.drag * dt;
    for (Particle& p : particles_) {
        p.vel = p.vel + cfg.gravity * dt;
        p.vel = p.vel * dragFactor;
        p.pos = p.pos + p.vel * dt;
        p.age += dt;
    }

    // 3. RETIRE particles with age >= lifetime (swap-remove, deterministic for a fixed sequence).
    size_t w = 0;
    for (size_t r = 0; r < particles_.size(); ++r) {
        if (particles_[r].age < particles_[r].lifetime) {
            if (w != r) particles_[w] = particles_[r];
            ++w;
        }
    }
    particles_.resize(w);
}

}  // namespace hf::vfx
