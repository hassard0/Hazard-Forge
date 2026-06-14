// GPU particle simulation. One thread per particle; reads+writes the particle storage buffer in
// place each frame. A simple deterministic fountain: gravity pulls particles down, and when a
// particle drops below the floor (or ages out) it respawns at the emitter with an upward+swirl
// velocity derived from its index (no RNG state -> deterministic for golden-image runs).
//
// Particle layout (32 bytes = 2x float4), shared with particle.vert.hlsl:
//   float4 posLife;  // xyz = world position, w = remaining life (seconds)
//   float4 velSeed;  // xyz = velocity,       w = per-particle seed (stable index hash)
struct Particle {
    float4 posLife;
    float4 velSeed;
};

RWStructuredBuffer<Particle> gParticles : register(u0);

// Push constant: dt (seconds since last frame), time (absolute), particleCount, _pad.
struct Params { float dt; float time; uint count; uint _pad; };
[[vk::push_constant]] Params gp;

// Cheap hash -> [0,1).
float hash11(float n) { return frac(sin(n) * 43758.5453123); }

// Emitter respawn: place at a tight origin disc, launch upward with index-derived swirl. Fully
// determined by the seed, so the same frame index always yields the same state (golden-stable).
void respawn(inout Particle pt, float seed) {
    // Decorrelate trajectories per-particle so 50k points fan into a dense volumetric fountain
    // rather than tracing one overlapping ribbon. Azimuth + elevation + speed all come from
    // independent hashes of the (stable) seed -> deterministic but well-spread.
    float a   = hash11(seed * 7.13) * 6.2831853;        // launch azimuth (full circle)
    float r   = hash11(seed * 1.70) * 0.3;              // spawn disc radius
    pt.posLife.xyz = float3(cos(a) * r, 0.05, sin(a) * r);
    float up    = 3.5 + hash11(seed * 2.30) * 3.5;      // vertical launch 3.5..7
    float outR  = 0.5 + hash11(seed * 5.70) * 2.8;      // radial fan 0.5..3.3
    float aVel  = hash11(seed * 9.41) * 6.2831853;      // velocity azimuth (independent of pos)
    pt.velSeed.xyz = float3(cos(aVel) * outR, up, sin(aVel) * outR);
    pt.posLife.w = 1.5 + hash11(seed * 3.10) * 2.5;     // 1.5..4s life
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= gp.count) return;

    Particle pt = gParticles[i];
    float seed = pt.velSeed.w;

    // Integrate: gravity + gentle swirl about the Y axis (curl-ish), then advance position.
    float3 vel = pt.velSeed.xyz;
    vel.y -= 5.0 * gp.dt;                                  // gravity
    float3 pos = pt.posLife.xyz;
    // Tangential swirl so the fountain twists as it rises (reads nicely as motion).
    vel.x += -pos.z * 1.5 * gp.dt;
    vel.z +=  pos.x * 1.5 * gp.dt;
    pos += vel * gp.dt;

    pt.posLife.xyz = pos;
    pt.velSeed.xyz = vel;
    pt.posLife.w -= gp.dt;

    // Respawn on death or when it falls below the floor.
    if (pt.posLife.w <= 0.0 || pos.y < -0.1) {
        respawn(pt, seed);
    }

    gParticles[i] = pt;
}
