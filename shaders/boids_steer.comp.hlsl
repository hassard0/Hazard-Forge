// Slice BD1 — Deterministic GPU Crowds: the Q16.16 INTEGER STEERING compute pass (the BEACHHEAD of FLAGSHIP
// #18: DETERMINISTIC GPU CROWDS — boids + steering + path-following). ONE thread per AGENT i (i < agentCount).
// Each thread computes agent i's Reynolds STEERING force from the FROZEN input buffer (SteerSeek toward the
// shared target + a brute-force all-pairs SteerSeparation over EVERY agent, skip self by index), per-axis
// clamps the force, integrates velocity (per-axis clamped) + position, and writes agent i into the OUTPUT
// buffer. The seek + separation + per-axis clamps + integrate are copied VERBATIM from
// engine/sim/boids.h::StepBoids — so the GPU exercises the EXACT integer ops the CPU reference runs, and the
// host's GPU==CPU memcmp catches any divergence.
//
// JACOBI / PING-PONG (the cross-backend crux): the step is Jacobi — every agent reads the PREVIOUS step's
// positions from gAgentsIn (a SEPARATE read-only input) and writes the next step's into gAgentsOut. So the
// per-thread write is INDEPENDENT of thread order (no thread reads another thread's freshly-written value) ->
// race-free, NO atomics, bit-identical GPU==CPU AND two-run byte-identical. The host ping-pongs the two
// buffers between steps (out -> in). The brute-force separation loop iterates ALL agents in the SAME fixed
// index order (skip self by index) as boids.h -> bit-exact.
//
// INTEGER WIDTH (the determinism crux, the FPX1/GR1 lesson): the steer/integrate path scales by Q16.16 gains
// + dt — `force = (target-pos)*seekGain`, `vel += force*dt`, `pos += vel*dt` — each an fxmul ((int64)a*b >>
// 16). Q16.16 world-scale products overflow int32, so fxmul uses an int64 INTERMEDIATE (the
// fpx_integrate.comp / grain_integrate.comp form). The separation squared-distance compare (d² < sepRadius²)
// is also int64. HLSL SM6 supports int64_t (DXC -spirv with the Int64 capability — the Vulkan path); glslc
// (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY
// (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --boids-steer showcase runs the CPU
// boids::StepBoids (byte-identical by construction — the fpx_integrate.comp/grain_integrate.comp convention).
//
// stepEnabled push/param flag: 0 -> copy the input agent back UNCHANGED (the disabled-path no-op; gAgentsOut
// stays byte-identical to gAgentsIn).
//
// Buffers (storage, bound at compute bindings 0..2; Vulkan-only):
//   b0 gAgentsIn  : the Q16.16 Agent array (pos.xyz, vel.xyz — std430 6 x int32, 24 bytes), READ (the frozen
//                   Jacobi input).
//   b1 gAgentsOut : the Q16.16 Agent array (24 bytes), WRITE (the next step).
//   b2 gParams    : the BoidsParams (seekGain, sepGain, sepRadius, maxForce, maxSpeed, target.xyz,
//                   gravity.xyz, dt, agentCount, stepEnabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// fpx_integrate.comp / grain_integrate.comp), not backend CODE symbols.

#define HF_BOIDS_THREADS 64
#define HF_BOIDS_FRAC 16     // MUST match boids.h::kFrac (== fpx.h::kFrac)

// std430 Agent mirror (engine/sim/boids.h::Agent): pos.xyz, vel.xyz. 6 x 4-byte = 24 bytes (memcmp-able).
struct Agent {
    int px, py, pz;   // Q16.16 position
    int vx, vy, vz;   // Q16.16 velocity
};

// std430 BoidsParams (the C++ upload mirror).
//   p0 : x=seekGain, y=sepGain, z=sepRadius, w=maxForce  (all Q16.16)
//   p1 : x=maxSpeed, y=targetX, z=targetY, w=targetZ      (all Q16.16)
//   p2 : x=gravityX, y=gravityY, z=gravityZ, w=dt         (all Q16.16)
//   p3 : x=agentCount, y=stepEnabled, z=unused, w=unused
struct BoidsParams {
    int4 p0;
    int4 p1;
    int4 p2;
    int4 p3;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Agent>       gAgentsIn  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<Agent>       gAgentsOut : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<BoidsParams> gParams    : register(u2);

// VERBATIM fpx.h::fxmul / boids.h fxmul — (a*b) >> kFrac with an int64 intermediate (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_BOIDS_FRAC);
}

// VERBATIM boids.h::ClampAxis — clamp v to [-limit, limit] (limit >= 0). Pure int32.
int ClampAxis(int v, int limit) {
    if (v >  limit) return  limit;
    if (v < -limit) return -limit;
    return v;
}

[numthreads(HF_BOIDS_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int seekGain  = gParams[0].p0.x;
    int sepGain   = gParams[0].p0.y;
    int sepRadius = gParams[0].p0.z;
    int maxForce  = gParams[0].p0.w;
    int maxSpeed  = gParams[0].p1.x;
    int targetX   = gParams[0].p1.y;
    int targetY   = gParams[0].p1.z;
    int targetZ   = gParams[0].p1.w;
    int gravX     = gParams[0].p2.x;
    int gravY     = gParams[0].p2.y;
    int gravZ     = gParams[0].p2.z;
    int dt        = gParams[0].p2.w;
    int agentCount  = gParams[0].p3.x;
    int stepEnabled = gParams[0].p3.y;

    uint i = gid.x;
    if ((int)i >= agentCount) return;

    Agent a = gAgentsIn[i];

    // Disabled -> copy the input agent back UNCHANGED (the byte-identical no-op proof).
    if (stepEnabled == 0) { gAgentsOut[i] = a; return; }

    // (a) SteerSeek: desired = target - pos; force = desired * seekGain (proportional / un-normalized).
    int fx = fxmul(targetX - a.px, seekGain);
    int fy = fxmul(targetY - a.py, seekGain);
    int fz = fxmul(targetZ - a.pz, seekGain);

    // (a) SteerSeparation: brute-force all-pairs; accumulate (a.pos - o.pos) for neighbors within sepRadius²
    // (raw integer adds, NO per-pair normalize), then scale by sepGain. Skip self by index.
    int64_t r2 = (int64_t)sepRadius * (int64_t)sepRadius;
    int sx = 0, sy = 0, sz = 0;
    for (int j = 0; j < agentCount; ++j) {
        if (j == (int)i) continue;                       // skip self by INDEX (fixed order, bit-exact)
        Agent o = gAgentsIn[j];
        int dx = a.px - o.px;
        int dy = a.py - o.py;
        int dz = a.pz - o.pz;
        int64_t d2 = (int64_t)dx * (int64_t)dx + (int64_t)dy * (int64_t)dy + (int64_t)dz * (int64_t)dz;
        if (d2 < r2) { sx += dx; sy += dy; sz += dz; }
    }
    fx += fxmul(sx, sepGain);
    fy += fxmul(sy, sepGain);
    fz += fxmul(sz, sepGain);

    // (a) + gravity (a constant acceleration; 0 in BD1).
    fx += gravX; fy += gravY; fz += gravZ;

    // (b) per-axis clamp the force (the axis-box magnitude limit).
    fx = ClampAxis(fx, maxForce);
    fy = ClampAxis(fy, maxForce);
    fz = ClampAxis(fz, maxForce);

    // (c) integrate velocity: vel += force * dt; then per-axis clamp to ±maxSpeed.
    int vx = a.vx + fxmul(fx, dt);
    int vy = a.vy + fxmul(fy, dt);
    int vz = a.vz + fxmul(fz, dt);
    vx = ClampAxis(vx, maxSpeed);
    vy = ClampAxis(vy, maxSpeed);
    vz = ClampAxis(vz, maxSpeed);

    // (d) integrate position: pos += vel * dt.
    a.px += fxmul(vx, dt);
    a.py += fxmul(vy, dt);
    a.pz += fxmul(vz, dt);
    a.vx = vx; a.vy = vy; a.vz = vz;

    gAgentsOut[i] = a;   // per-agent independent write into the OUTPUT buffer (NO atomics)
}
