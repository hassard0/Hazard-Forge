// Slice CL3 — Deterministic GPU Cloth: the PBD DISTANCE-CONSTRAINT SOLVER compute pass (the MAKE-OR-BREAK of
// FLAGSHIP #8). A SINGLE THREAD ([numthreads(1,1,1)], the fpx_solve.comp / cloth_edge_scan.comp pattern;
// gid.x!=0 -> return) runs `steps` iterations of StepCloth = IntegrateParticles (CL1 semi-implicit-Euler)
// then `iters` Gauss-Seidel constraint passes (each iterating ALL CL2 constraints in the FIXED emit order
// applying SolveDistanceConstraint) then a ground floor-clamp. The constraint projection is inherently
// SEQUENTIAL (each constraint reads the LATEST positions, already updated by earlier constraints this pass)
// -> one thread -> bit-exact GPU==CPU + cross-backend, NO atomics, NO race. The fxmul/fxdiv/FxISqrt/
// FxNormalize + integrate + distance-constraint projection + floor clamp are copied VERBATIM from
// engine/sim/cloth.h (SolveDistanceConstraint / StepCloth, which reuse fpx.h) so tests/cloth_test.cpp + the
// GPU pass exercise the EXACT integer ops -> a divergence is exactly what the host GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux): fxmul/fxdiv/FxISqrt use int64_t (IDENTICAL to cloth.h/fpx.h). DXC
// -spirv compiles int64 (the Int64 capability, the fpx_solve.comp / cloth_integrate.comp pattern); glslc
// (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY
// (in the Vulkan compile list, NOT in the Metal hf_gen_msl list). The Metal --cloth-solve showcase runs the
// CPU cloth::StepCloth over the same sheet -> byte-identical to this GPU result by construction, while the
// Vulkan side carries the GPU==CPU bit-identity proof.
//
// solveEnabled=0 -> write the input particles back UNCHANGED (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..2):
//   b0 gParticles  : the Q16.16 ClothParticle array (pos.xyz, prev.xyz, vel.xyz, invMass, flags — std430
//                    ints, 44 bytes), READ+WRITE.
//   b1 gConstraints: the CL2 Constraint list (i, j, restLen, kind — std430 ints, 16 bytes), READ.
//   b2 gParams     : { gravity.xyz, dt, groundY, particleCount, constraintCount, steps, iters,
//                      solveEnabled }, READ.

#define HF_CLOTH_FRAC 16   // MUST match cloth.h::kFrac (== fpx.h::kFrac)
#define HF_CLOTH_ONE  (1 << HF_CLOTH_FRAC)
#define HF_CLOTH_FLAG_PINNED 1u

// std430 ClothParticle mirror (engine/sim/cloth.h::ClothParticle): pos.xyz, prev.xyz, vel.xyz, invMass,
// flags. 11 x 4-byte = 44 bytes, no padding holes (memcmp-able; == the CL1 cloth_integrate.comp mirror).
struct ClothParticle {
    int  px, py, pz;     // Q16.16 current position
    int  prx, pry, prz;  // Q16.16 previous position
    int  vx, vy, vz;     // Q16.16 velocity
    int  invMass;        // Q16.16 inverse mass (0 => pinned)
    uint flags;          // bit0 = PINNED
};

// std430 Constraint mirror (engine/sim/cloth.h::Constraint): i, j, restLen, kind. 4 x 4-byte = 16 bytes
// (== the CL2 cloth_edge_emit.comp mirror).
struct Constraint {
    uint i, j;     // endpoint particle indices (i<j)
    int  restLen;  // Q16.16 rest length
    uint kind;     // structural / shear / bend
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=groundY (Q16.16), y=particleCount, z=constraintCount, w=steps
//   cfg2 : x=iters, y=solveEnabled, z=_, w=_
struct ClothSolveParams {
    int4 grav;
    int4 cfg;
    int4 cfg2;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ClothParticle>    gParticles   : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<Constraint>       gConstraints : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<ClothSolveParams> gParams      : register(u2);

// VERBATIM cloth.h::fxmul / fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_CLOTH_FRAC);
}

// VERBATIM fpx.h::fxdiv — (a << kFrac) / b in Q16.16 (int64 shift + truncating divide; guard b==0).
int fxdiv(int a, int b) {
    if (b == 0) return 0;
    return (int)(((int64_t)a << HF_CLOTH_FRAC) / (int64_t)b);
}

// VERBATIM fpx.h::FxISqrt — floor(sqrt) of a non-negative int64 (binary digit-by-digit).
int64_t FxISqrt(int64_t v) {
    if (v <= 0) return 0;
    int64_t bit = (int64_t)1 << 62;
    while (bit > v) bit >>= 2;
    int64_t res = 0;
    while (bit != 0) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else { res >>= 1; }
        bit >>= 2;
    }
    return res;
}

// VERBATIM fpx.h::FxLength — sqrt(x^2+y^2+z^2) in Q16.16 (sum of int64 squares -> Q32.32 -> floor-sqrt).
int FxLength(int vx, int vy, int vz) {
    int64_t sx = (int64_t)vx * (int64_t)vx;
    int64_t sy = (int64_t)vy * (int64_t)vy;
    int64_t sz = (int64_t)vz * (int64_t)vz;
    return (int)FxISqrt(sx + sy + sz);
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    // SINGLE THREAD: only thread 0 runs the serial integrate+solve (guard defensively).
    if (gid.x != 0u) return;

    int gravx = gParams[0].grav.x;
    int gravy = gParams[0].grav.y;
    int gravz = gParams[0].grav.z;
    int dt    = gParams[0].grav.w;
    int groundY         = gParams[0].cfg.x;
    int particleCount   = gParams[0].cfg.y;
    int constraintCount = gParams[0].cfg.z;
    int steps           = gParams[0].cfg.w;
    int iters           = gParams[0].cfg2.x;
    int solveEnabled    = gParams[0].cfg2.y;

    // Disabled -> write the input back UNCHANGED (the byte-identical no-op proof).
    if (solveEnabled == 0) return;

    for (int s = 0; s < steps; ++s) {
        // (1) IntegrateParticles: per-particle semi-implicit-Euler + ground floor-clamp (VERBATIM
        //     cloth.h::IntegrateParticle). PINNED particles untouched.
        for (int i = 0; i < particleCount; ++i) {
            ClothParticle p = gParticles[i];
            if ((p.flags & HF_CLOTH_FLAG_PINNED) == 0u) {
                p.vx += fxmul(gravx, dt);
                p.vy += fxmul(gravy, dt);
                p.vz += fxmul(gravz, dt);
                p.prx = p.px;
                p.pry = p.py;
                p.prz = p.pz;
                p.px += fxmul(p.vx, dt);
                p.py += fxmul(p.vy, dt);
                p.pz += fxmul(p.vz, dt);
                if (p.py < groundY) {
                    p.py = groundY;
                    if (p.vy < 0) p.vy = 0;
                }
            }
            gParticles[i] = p;
        }

        // (2) `iters` Gauss-Seidel constraint passes in the FIXED CL2 emit order. SEQUENTIAL — each
        //     constraint reads + writes gParticles so the next constraint sees the update (VERBATIM
        //     cloth.h::SolveDistanceConstraint).
        for (int it = 0; it < iters; ++it) {
            for (int e = 0; e < constraintCount; ++e) {
                uint ci = gConstraints[e].i;
                uint cj = gConstraints[e].j;
                int  restLen = gConstraints[e].restLen;
                ClothParticle pi = gParticles[ci];
                ClothParticle pj = gParticles[cj];
                int wsum = pi.invMass + pj.invMass;
                if (wsum != 0) {
                    int dx = pj.px - pi.px;
                    int dy = pj.py - pi.py;
                    int dz = pj.pz - pi.pz;
                    int len = FxLength(dx, dy, dz);
                    if (len != 0) {
                        int pen = len - restLen;
                        // FxNormalize(d) (len != 0 here so the fallback branch is dead, kept verbatim).
                        int nx = fxdiv(dx, len);
                        int ny = fxdiv(dy, len);
                        int nz = fxdiv(dz, len);
                        int wi = fxdiv(pi.invMass, wsum);
                        int wj = fxdiv(pj.invMass, wsum);
                        int ai = fxmul(pen, wi);
                        int aj = fxmul(pen, wj);
                        pi.px += fxmul(nx, ai); pi.py += fxmul(ny, ai); pi.pz += fxmul(nz, ai);
                        pj.px -= fxmul(nx, aj); pj.py -= fxmul(ny, aj); pj.pz -= fxmul(nz, aj);
                        gParticles[ci] = pi;
                        gParticles[cj] = pj;
                    }
                }
            }
        }

        // (3) ground floor clamp AFTER the constraint passes (a constraint may push a particle below).
        for (int g = 0; g < particleCount; ++g) {
            ClothParticle p = gParticles[g];
            if (p.py < groundY) { p.py = groundY; gParticles[g] = p; }
        }
    }
}
