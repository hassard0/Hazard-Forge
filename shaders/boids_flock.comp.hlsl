// Slice BD3 — Deterministic GPU Crowds: THE FULL FLOCK STEP compute pass (the 3rd slice of FLAGSHIP #18:
// DETERMINISTIC GPU CROWDS — boids + steering + path-following, hf::sim::boids). ONE thread per AGENT i (i <
// agentCount). Each thread accumulates the THREE Reynolds rules over agent i's BD2 NEIGHBOR LIST (the indices
// in neighbors[neighborStart[i]..neighborStart[i+1]], built THIS tick by the MSL-native BD2 grid passes) —
// SEPARATION (Σ (pos_i - pos_j) * sepGain) + ALIGNMENT ((Σ vel_j)/count - vel_i) * alignGain + COHESION
// ((Σ pos_j)/count - pos_i) * cohGain — adds the optional seek-to-target + gravity, per-axis clamps the force,
// integrates velocity (per-axis clamped) + position, and writes agent i into the OUTPUT buffer. The 3 rules +
// the means + the per-axis clamps + the integrate are copied VERBATIM from engine/sim/boids.h::AccumFlock +
// StepFlock — so the GPU exercises the EXACT integer ops the CPU reference runs, and the host's GPU==CPU
// memcmp catches any divergence.
//
// JACOBI / PING-PONG (the cross-backend crux): the step is Jacobi — every agent reads the PREVIOUS step's
// positions/velocities from gAgentsIn (a SEPARATE read-only input) + the neighbor list built over THAT frozen
// snapshot, and writes the next step's into gAgentsOut. So the per-thread write is INDEPENDENT of thread order
// (no thread reads another thread's freshly-written value) -> race-free, NO atomics, bit-identical GPU==CPU AND
// two-run byte-identical. The host ping-pongs the two agent buffers between steps (out -> in) and rebuilds the
// BD2 neighbor list (the MSL-native boids_cell_*/boids_neighbor_* passes) on the new positions each tick.
//
// THE MEANS ARE INTEGER DIVIDES + the integrate is int64 (the determinism crux, the FPX1/GR1/BD1 lesson): the
// alignment/cohesion means divide the neighbor velocity/position SUM by the integer neighbor COUNT (truncate
// toward zero, the C++/HLSL/MSL `/` semantics, identical on every vendor — VERBATIM the CPU AccumFlock). The
// integrate scales by Q16.16 gains + dt — `force = sum*gain`, `vel += force*dt`, `pos += vel*dt` — each an
// fxmul ((int64)a*b >> 16): Q16.16 world-scale products overflow int32, so fxmul uses an int64 INTERMEDIATE
// (the fpx_integrate.comp / boids_steer.comp form). HLSL SM6 supports int64_t (DXC -spirv with the Int64
// capability — the Vulkan path); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so
// this shader is VULKAN-SPIR-V-ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the
// --boids-flock showcase runs the CPU boids::StepFlock (byte-identical by construction — the boids_steer.comp
// convention), while the MSL-native BD2 grid passes (boids_cell_*/boids_neighbor_*) stay a true GPU pass on
// both backends. The flock-integrate pass is the only int64 split.
//
// stepEnabled push/param flag: 0 -> copy the input agent back UNCHANGED (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..4; Vulkan-only):
//   b0 gAgentsIn      : the Q16.16 Agent array (pos.xyz, vel.xyz — std430 6 x int32, 24 bytes), READ (frozen).
//   b1 gAgentsOut     : the Q16.16 Agent array (24 bytes), WRITE (the next step).
//   b2 neighborStart  : the BD2 neighbor-list prefix-sum (agentCount+1), READ.
//   b3 neighbors      : the BD2 neighbor j-index array grouped by i, READ.
//   b4 gParams        : the FlockParams (seek/sep/align/coh gains, maxForce, maxSpeed, target, gravity, dt,
//                       agentCount, stepEnabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// boids_steer.comp / fpx_integrate.comp), not backend CODE symbols.

#define HF_BOIDS_THREADS 64
#define HF_BOIDS_FRAC 16     // MUST match boids.h::kFrac (== fpx.h::kFrac)

// std430 Agent mirror (engine/sim/boids.h::Agent): pos.xyz, vel.xyz. 6 x 4-byte = 24 bytes (memcmp-able).
struct Agent {
    int px, py, pz;   // Q16.16 position
    int vx, vy, vz;   // Q16.16 velocity
};

// std430 FlockParams (the C++ upload mirror).
//   p0 : x=seekGain, y=sepGain, z=alignGain, w=cohGain   (all Q16.16)
//   p1 : x=maxForce, y=maxSpeed, z=targetX,  w=targetY   (all Q16.16)
//   p2 : x=targetZ,  y=gravityX, z=gravityY, w=gravityZ  (all Q16.16)
//   p3 : x=dt, y=agentCount, z=stepEnabled, w=unused
struct FlockParams {
    int4 p0;
    int4 p1;
    int4 p2;
    int4 p3;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Agent>       gAgentsIn     : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<Agent>       gAgentsOut    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>        neighborStart : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>        neighbors     : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<FlockParams> gParams       : register(u4);

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
    int alignGain = gParams[0].p0.z;
    int cohGain   = gParams[0].p0.w;
    int maxForce  = gParams[0].p1.x;
    int maxSpeed  = gParams[0].p1.y;
    int targetX   = gParams[0].p1.z;
    int targetY   = gParams[0].p1.w;
    int targetZ   = gParams[0].p2.x;
    int gravX     = gParams[0].p2.y;
    int gravY     = gParams[0].p2.z;
    int gravZ     = gParams[0].p2.w;
    int dt          = gParams[0].p3.x;
    int agentCount  = gParams[0].p3.y;
    int stepEnabled = gParams[0].p3.z;

    uint i = gid.x;
    if ((int)i >= agentCount) return;

    Agent a = gAgentsIn[i];

    // Disabled -> copy the input agent back UNCHANGED (the byte-identical no-op proof).
    if (stepEnabled == 0) { gAgentsOut[i] = a; return; }

    // === The 3 Reynolds rules over agent i's BD2 neighbor slice (VERBATIM boids.h::AccumFlock) ===
    uint s0 = neighborStart[i];
    uint s1 = neighborStart[i + 1u];
    int sepX = 0, sepY = 0, sepZ = 0;   // Σ (pos_i - pos_j)
    int velX = 0, velY = 0, velZ = 0;   // Σ vel_j
    int posX = 0, posY = 0, posZ = 0;   // Σ pos_j
    int count = 0;
    for (uint s = s0; s < s1; ++s) {
        Agent o = gAgentsIn[neighbors[s]];
        sepX += a.px - o.px; sepY += a.py - o.py; sepZ += a.pz - o.pz;
        velX += o.vx;        velY += o.vy;        velZ += o.vz;
        posX += o.px;        posY += o.py;        posZ += o.pz;
        ++count;
    }
    // separation: scale the away-sum by sepGain (zero if no neighbors).
    int fx = fxmul(sepX, sepGain);
    int fy = fxmul(sepY, sepGain);
    int fz = fxmul(sepZ, sepGain);
    if (count > 0) {
        // alignment: (mean neighbor velocity) - own velocity, scaled by alignGain. Integer divide by count.
        int mvx = velX / count, mvy = velY / count, mvz = velZ / count;
        fx += fxmul(mvx - a.vx, alignGain);
        fy += fxmul(mvy - a.vy, alignGain);
        fz += fxmul(mvz - a.vz, alignGain);
        // cohesion: (mean neighbor position) - own position, scaled by cohGain. Integer divide by count.
        int mpx = posX / count, mpy = posY / count, mpz = posZ / count;
        fx += fxmul(mpx - a.px, cohGain);
        fy += fxmul(mpy - a.py, cohGain);
        fz += fxmul(mpz - a.pz, cohGain);
    }

    // (b) the optional seek toward the shared target (un-normalized; skipped when seekGain == 0).
    if (seekGain != 0) {
        fx += fxmul(targetX - a.px, seekGain);
        fy += fxmul(targetY - a.py, seekGain);
        fz += fxmul(targetZ - a.pz, seekGain);
    }

    // (c) + gravity (a constant acceleration; 0 in the planar showcase).
    fx += gravX; fy += gravY; fz += gravZ;

    // (d) per-axis clamp the force (the axis-box magnitude limit).
    fx = ClampAxis(fx, maxForce);
    fy = ClampAxis(fy, maxForce);
    fz = ClampAxis(fz, maxForce);

    // (e) integrate velocity: vel += force * dt; then per-axis clamp to ±maxSpeed.
    int vx = a.vx + fxmul(fx, dt);
    int vy = a.vy + fxmul(fy, dt);
    int vz = a.vz + fxmul(fz, dt);
    vx = ClampAxis(vx, maxSpeed);
    vy = ClampAxis(vy, maxSpeed);
    vz = ClampAxis(vz, maxSpeed);

    // (f) integrate position: pos += vel * dt.
    a.px += fxmul(vx, dt);
    a.py += fxmul(vy, dt);
    a.pz += fxmul(vz, dt);
    a.vx = vx; a.vy = vy; a.vz = vz;

    gAgentsOut[i] = a;   // per-agent independent write into the OUTPUT buffer (NO atomics)
}
