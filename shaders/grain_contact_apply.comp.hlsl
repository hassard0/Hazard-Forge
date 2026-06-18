// Slice GR3 — Deterministic GPU Granular/Sand: the JACOBI Δp APPLY pass (the 3rd slice of FLAGSHIP #10, the
// FL4 fluid_apply.comp twin). ONE thread per GRAIN i: p_i.pos += dp_i (the Jacobi double-buffer apply —
// reads the Δp_i computed by grain_contact_dp.comp from the iteration-start positions, writes p_i.pos).
// Per-grain independent (each thread writes only its OWN grain) -> NO race, NO atomics, deterministic
// regardless of thread order. STATIC grains (flags & STATIC bit) are untouched. Copied VERBATIM from
// grain.h::StepGrainContact's "pos_i += dp_i" apply step.
//
// Pure int32 add (NO int64) — so this pass COULD MSL-generate natively. But, like the FL4 fluid_apply.comp,
// the GR3 solve is driven as a HOST sequence on Vulkan (the int64 grain_contact_dp.comp + grain_collide.comp
// are Vulkan-only) and the Metal --grain-contact showcase runs the CPU grain::StepGrainContact (byte-
// identical by construction), so this apply pass is kept VULKAN-SPIR-V-ONLY alongside its siblings (NOT in
// hf_gen_msl) — the simplest correct split (the whole solve trio is one Vulkan-only group).
//
// Buffers (storage, Vulkan-only):
//   b0 gGrains : the Q16.16 GrainParticle array (48 bytes), READ+WRITE (pos += dp; flags read).
//   b1 dp      : one FxVec3Gpu (3 x int32) per grain (the Jacobi correction Δp_i), READ.
//   b2 gParams : the GrainContactParams (grainCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_GRAIN_THREADS 64
#define HF_GRAIN_STATIC 1u   // == grain.h::kFlagStatic (bit0)

struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

struct FxVec3Gpu { int x, y, z; };

struct GrainContactParams { int4 cfg; };

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle>      gGrains : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FxVec3Gpu>         dp      : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<GrainContactParams> gParams : register(u2);

[numthreads(HF_GRAIN_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int grainCount = gParams[0].cfg.x;
    int enabled    = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= grainCount) return;
    if (enabled == 0) return;

    GrainParticle p = gGrains[i];
    if (p.flags & HF_GRAIN_STATIC) return;   // static -> never moves
    FxVec3Gpu d = dp[i];
    p.px += d.x; p.py += d.y; p.pz += d.z;
    gGrains[i] = p;                          // per-grain independent write (NO atomics)
}
