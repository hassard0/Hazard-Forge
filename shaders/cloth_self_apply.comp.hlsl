// Slice CL7 — Deterministic GPU Cloth: the SELF-COLLISION pass 2 (the Jacobi correction apply; the
// Track-R R3 refinement of FLAGSHIP #8). ONE thread per cloth VERT — the JACOBI apply (the
// fluid_visc_apply.comp twin): runs ONLY after pass 1 (cloth_self.comp) computed EVERY vert's correction
// into gCorr (a compute->compute barrier between them), so no position is rewritten before every
// correction was gathered. Copied VERBATIM from cloth.h::SolveSelfCollision PASS 2: every NON-pinned
// vert: pos += corr[i]; then the ground clamp (pos.y >= groundY). PINNED verts are untouched. Per-vert
// independent write -> race-free, deterministic, NO atomics, NO single-thread, NO TDR.
//
// INTEGER WIDTH: pure int32 adds + compares here, but the pass pairs with the int64 cloth_self.comp so
// both live in the same VULKAN-SPIR-V-ONLY list (NOT in the Metal hf_gen_msl list); on Metal the
// --cl7-self showcase runs the CPU cloth::StepClothSelfSteps (byte-identical by construction, the
// cloth_solve/cloth_collide convention).
//
// Buffers (storage, Vulkan-only):
//   b0 gParticles : the Q16.16 ClothParticle array, READ+WRITE (pos updated).
//   b1 gCorr      : one FxVec3Gpu per vert (pass 1's gathered correction), READ.
//   b2 gParams    : the ClothSelfParams (thickness, groundY, particleCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_CLOTH_THREADS 64
#define HF_CLOTH_FLAG_PINNED 1u

// std430 ClothParticle mirror (engine/sim/cloth.h::ClothParticle): 11 x 4-byte = 44 bytes (== CL1-CL4).
struct ClothParticle {
    int  px, py, pz;     // Q16.16 current position
    int  prx, pry, prz;  // Q16.16 previous position
    int  vx, vy, vz;     // Q16.16 velocity
    int  invMass;        // Q16.16 inverse mass (0 => pinned)
    uint flags;          // bit0 = PINNED
};

struct FxVec3Gpu { int x, y, z; };

// ClothSelfParams (std430). cfg {thickness, groundY, particleCount, enabled} (== cloth_self.comp).
struct ClothSelfParams {
    int4 cfg;   // x=thickness (Q16.16), y=groundY (Q16.16), z=particleCount, w=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ClothParticle>   gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FxVec3Gpu>       gCorr      : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<ClothSelfParams> gParams    : register(u2);

[numthreads(HF_CLOTH_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int thickness     = gParams[0].cfg.x;
    int groundY       = gParams[0].cfg.y;
    int particleCount = gParams[0].cfg.z;
    int enabled       = gParams[0].cfg.w;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    if (enabled == 0 || thickness <= 0) return;                 // identity-at-zero: touch nothing

    ClothParticle p = gParticles[i];
    if (p.flags & HF_CLOTH_FLAG_PINNED) return;                 // pinned untouched
    FxVec3Gpu c = gCorr[i];
    p.px += c.x;
    p.py += c.y;
    p.pz += c.z;
    if (p.py < groundY) p.py = groundY;                         // the ground clamp (pass-2 tail)
    gParticles[i] = p;
}
