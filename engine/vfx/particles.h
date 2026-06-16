#pragma once
// Slice CC — CPU particle / VFX emitter system (Phase 4 #28).
//
// PURE CPU: this module has ZERO RHI / graphics-backend symbols (no vk*/MTL*/mtl::/Backend::Metal).
// It is an AUTHORABLE emitter layer — data-driven emitter parameters + lifetime curves — DISTINCT
// from the fixed `gpu-particles` compute fountain (shaders/particles.comp.hlsl, one hard-coded GPU
// sim). This is the VFX analogue of the authorable material graph: the emitter is configurable
// (spawn rate, lifetime, initial velocity + spread, gravity, drag, size-over-life, color-over-life),
// simulated on the CPU at a FIXED timestep, and rendered as camera-facing alpha-blended billboards.
//
// DETERMINISM (golden-verifiable on both backends): particle spawn jitter normally uses RNG and the
// sim normally uses real time. To stay byte-stable we instead use:
//   * a FIXED-SEED integer LCG (the same 1664525/1013904223 constants the audio Noise voice uses) for
//     the velocity-spread jitter — NO std::rand / <random>, NO clock; and
//   * a FIXED number of FIXED-dt Step() calls — NO wall clock.
// Same seed + same step sequence + same EmitterConfig => identical particle state => identical
// billboards => goldens match, and two runs are bit-identical. The CPU sim is compiled into BOTH
// hf_core (ASan-scoped, unit-tested by tests/vfx_test.cpp) and hf_engine (the live --vfx-shot
// showcase + the Metal --vfx path), exactly like engine/audio + engine/physics.
//
// The render side (samples/hello_triangle/main.cpp --vfx-shot + shaders/vfx.{vert,frag}.hlsl) reuses
// the EXISTING alpha-blend + dynamic-textured-quad infra (the same pattern the HUD/text overlay and
// the transparency pass use) — it adds NO new RHI seam symbols. BuildBillboards() below produces the
// world-space camera-facing quad vertices that pass uploads as a dynamic vertex buffer.
#include <cstdint>
#include <span>
#include <vector>

#include "math/math.h"

namespace hf::vfx {

using math::Vec2;
using math::Vec3;
using math::Vec4;

// --- Emitter configuration (authorable) ---------------------------------------------------------
// One data-driven emitter. All fields are plain values (no handles); a fixed config + seed + step
// count fully determines the particle state. `maxParticles` caps the live pool (spawns beyond the
// cap are dropped, NOT queued — see ParticleSystem::Step).
struct EmitterConfig {
    Vec3 origin{0.0f, 0.0f, 0.0f};   // world-space spawn point
    float spawnRate = 100.0f;        // particles per second (continuous; fractional accumulated)
    float lifetime = 2.0f;           // seconds each particle lives before retiring
    Vec3 initVel{0.0f, 4.0f, 0.0f};  // base initial velocity (before per-particle spread jitter)
    float velSpread = 1.0f;          // jitter magnitude added to each velocity component (+/- spread)
    Vec3 gravity{0.0f, -9.8f, 0.0f}; // constant acceleration applied each step
    float drag = 0.1f;               // velocity damping per second: vel *= (1 - drag*dt) each step
    float startSize = 0.3f;          // billboard half-extent at age 0
    float endSize = 0.05f;           // billboard half-extent at age == lifetime (lerp over life)
    Vec4 startColor{1.0f, 0.8f, 0.2f, 1.0f};  // RGBA at age 0 (premultiply-free; alpha used directly)
    Vec4 endColor{0.6f, 0.1f, 0.0f, 0.0f};    // RGBA at age == lifetime
    uint32_t seed = 1u;              // LCG seed for the velocity-spread jitter (determinism)
    int maxParticles = 4096;         // hard cap on the live pool
};

// --- One live particle --------------------------------------------------------------------------
// Size + color are NOT stored — they are derived from age/lifetime via SizeAt/ColorAt at render time
// (shared with the billboard generation + the shader), so the curve params can be authored without
// touching the per-particle state.
struct Particle {
    Vec3 pos{0.0f, 0.0f, 0.0f};
    Vec3 vel{0.0f, 0.0f, 0.0f};
    float age = 0.0f;       // seconds since spawn
    float lifetime = 0.0f;  // seconds this particle lives (copied from config at spawn)
};

// --- Lifetime curves (shared CPU/GPU semantics) -------------------------------------------------
// SizeAt = lerp(startSize, endSize, age/lifetime); ColorAt = lerp(startColor, endColor, age/lifetime).
// Clamped to [0,1] in t so an exactly-at-lifetime particle reads the end value. Pure; unit-tested and
// mirrored by the billboard generation below (the shader receives the per-vertex color/size directly).
float SizeAt(const EmitterConfig& cfg, const Particle& p);
Vec4  ColorAt(const EmitterConfig& cfg, const Particle& p);

// --- Billboard vertex ---------------------------------------------------------------------------
// One vertex of a camera-facing particle quad: world position, atlas-corner UV ([0,1]^2 — the frag
// computes a procedural soft round sprite from it, like the decal's in-shader texture), and the
// per-particle RGBA color (from ColorAt). Tightly packed for a dynamic vertex buffer; the layout the
// VFX pipeline declares is { pos: RGB32F @0, uv: RG32F @12, color: RGBA32F @20 } = 36 bytes.
struct BillboardVertex {
    Vec3 pos;    // world-space
    Vec2 uv;     // [0,1] sprite corner
    Vec4 color;  // RGBA (alpha used by the over/additive blend)
};

// Append 6 vertices (2 triangles) per particle, a camera-facing quad centered on the particle's
// world position. The quad spans `cameraRight` and `cameraUp` (assumed unit, orthogonal) scaled by
// SizeAt(cfg, p) so it always faces the camera, with corner UVs (0,0)/(1,0)/(1,1)/(0,1). Color is
// ColorAt(cfg, p), constant across the quad. `outVerts` is appended to (NOT cleared). Deterministic.
void BuildBillboards(std::span<const Particle> particles, const EmitterConfig& cfg, Vec3 cameraRight,
                     Vec3 cameraUp, std::vector<BillboardVertex>& outVerts);

// --- The particle system ------------------------------------------------------------------------
// Owns a fixed-capacity live pool (cap == EmitterConfig::maxParticles at the first Step). Step():
//   1. SPAWN: nSpawn = floor(spawnRate*dt + accumulator); accumulator keeps the fractional remainder
//      so the long-run rate is exactly spawnRate. Each new particle starts at cfg.origin with
//      vel = initVel + jitter, where jitter.{x,y,z} = (LCG draw in [-1,1]) * velSpread (3 seeded LCG
//      draws per particle). Spawns that would exceed maxParticles are DROPPED (the accumulator still
//      advances — capacity, not spawn rate, is the limiter).
//   2. INTEGRATE every live particle (semi-implicit Euler): vel += gravity*dt; vel *= (1 - drag*dt);
//      pos += vel*dt; age += dt.
//   3. RETIRE every particle with age >= lifetime (swap-remove; order is not part of the contract,
//      but is deterministic for a fixed step sequence).
// Deterministic: a fixed seed + fixed dt + fixed step count => bit-identical state across two runs.
class ParticleSystem {
public:
    // Advance the simulation one fixed step of `dt` seconds for the given emitter config.
    void Step(const EmitterConfig& cfg, float dt);

    // The live particles (valid until the next Step). Size == AliveCount().
    std::span<const Particle> Alive() const { return {particles_.data(), particles_.size()}; }

    // Live particle count after the most recent Step.
    int AliveCount() const { return static_cast<int>(particles_.size()); }

    // Total particles spawned over the system's lifetime (cumulative; counts only actually-created
    // particles, i.e. excludes spawns dropped at the maxParticles cap).
    uint64_t SpawnedCount() const { return spawned_; }

    // Reset to the empty initial state (no particles, zero accumulator, LCG re-seeded on next Step).
    void Reset();

private:
    // Advance + return the next LCG value (same constants as audio::Wave::Noise). Lazily seeds from
    // cfg.seed on the first draw of a fresh/reset system.
    uint32_t NextRand(uint32_t seed);
    // Map a fresh LCG draw to a float in [-1, 1].
    float RandSpread(uint32_t seed);

    std::vector<Particle> particles_;
    float accumulator_ = 0.0f;   // fractional spawn carry
    uint64_t spawned_ = 0;       // cumulative spawned (excludes cap-dropped)
    uint32_t lcg_ = 0;           // LCG state
    bool seeded_ = false;        // lazy-seed guard
};

}  // namespace hf::vfx
