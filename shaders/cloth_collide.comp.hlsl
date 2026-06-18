// Slice CL4 — Deterministic GPU Cloth: INTEGER COLLISION compute pass (cloth-vs-FPX rigid sphere + ground).
// FOLDS the CL3 PBD step (integrate + Gauss-Seidel constraint passes + ground floor-clamp) AND per-step
// COLLISION (project each non-pinned particle out of a small STATIC set of SPHERE colliders, then a ground
// plane clamp) into ONE single-thread ([numthreads(1,1,1)], the cloth_solve.comp / fpx_solve.comp pattern;
// gid.x!=0 -> return) `steps` loop. The constraint projection is inherently SEQUENTIAL (Gauss-Seidel, each
// constraint reads the LATEST positions) -> one thread -> bit-exact GPU==CPU + cross-backend, NO atomics.
// The integrate + distance-constraint projection are copied VERBATIM from cloth_solve.comp (== cloth.h);
// the collision (the int32 AABB reject + the int64 FxLength/FxNormalize push to the surface) is copied
// VERBATIM from engine/sim/cloth.h::CollideParticleSphere — so the host GPU==CPU memcmp exercises the EXACT
// integer ops a divergence would break.
//
// INTEGER WIDTH (the determinism crux): fxmul/fxdiv/FxISqrt use int64_t (IDENTICAL to cloth.h/fpx.h). DXC
// -spirv compiles int64 (the Int64 capability, the cloth_solve.comp / fpx_solve.comp pattern); glslc (the
// Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (in
// the Vulkan compile list, NOT in the Metal hf_gen_msl list). The Metal --cloth-collide showcase runs the
// CPU cloth::StepClothCollide over the same sheet -> byte-identical to this GPU result by construction,
// while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// collideEnabled=0 -> write the input particles back UNCHANGED (the disabled-path no-op). spheres=0 (an
// EMPTY collider set) -> the collision loop is a no-op -> byte-identical to the CL3 cloth_solve (the
// zero-collider equivalence).
//
// Buffers (storage, bound at compute bindings 0..3):
//   b0 gParticles  : the Q16.16 ClothParticle array (44 bytes, the CL1 mirror), READ+WRITE.
//   b1 gConstraints: the CL2 Constraint list (i, j, restLen, kind — 16 bytes), READ.
//   b2 gSpheres    : the CL4 SphereCollider array (center.xyz, radius — 16 bytes), READ.
//   b3 gParams     : { gravity.xyz, dt, groundY, particleCount, constraintCount, steps, iters,
//                      sphereCount, collideEnabled }, READ.

#define HF_CLOTH_FRAC 16   // MUST match cloth.h::kFrac (== fpx.h::kFrac)
#define HF_CLOTH_ONE  (1 << HF_CLOTH_FRAC)
#define HF_CLOTH_FLAG_PINNED 1u

// std430 ClothParticle mirror (engine/sim/cloth.h::ClothParticle): pos.xyz, prev.xyz, vel.xyz, invMass,
// flags. 11 x 4-byte = 44 bytes, no padding holes (memcmp-able; == the CL1/CL3 mirror).
struct ClothParticle {
    int  px, py, pz;     // Q16.16 current position
    int  prx, pry, prz;  // Q16.16 previous position
    int  vx, vy, vz;     // Q16.16 velocity
    int  invMass;        // Q16.16 inverse mass (0 => pinned)
    uint flags;          // bit0 = PINNED
};

// std430 Constraint mirror (engine/sim/cloth.h::Constraint): i, j, restLen, kind. 4 x 4-byte = 16 bytes
// (== the CL2/CL3 mirror).
struct Constraint {
    uint i, j;     // endpoint particle indices (i<j)
    int  restLen;  // Q16.16 rest length
    uint kind;     // structural / shear / bend
};

// std430 SphereCollider mirror (engine/sim/cloth.h::SphereCollider): center.xyz, radius. 4 x 4-byte = 16
// bytes, no padding holes (memcmp-able). Reuses fpx::FxBody's pos+radius semantics.
struct SphereCollider {
    int cx, cy, cz;   // Q16.16 sphere center
    int radius;       // Q16.16 sphere radius
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=groundY (Q16.16), y=particleCount, z=constraintCount, w=steps
//   cfg2 : x=iters, y=sphereCount, z=collideEnabled, w=_
struct ClothCollideParams {
    int4 grav;
    int4 cfg;
    int4 cfg2;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ClothParticle>      gParticles   : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<Constraint>         gConstraints : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<SphereCollider>     gSpheres     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<ClothCollideParams> gParams      : register(u3);

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
    // SINGLE THREAD: only thread 0 runs the serial integrate+solve+collide (guard defensively).
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
    int sphereCount     = gParams[0].cfg2.y;
    int collideEnabled  = gParams[0].cfg2.z;

    // Disabled -> write the input back UNCHANGED (the byte-identical no-op proof).
    if (collideEnabled == 0) return;

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

        // (4) COLLISION: clamp to the ground plane (idempotent with (3)) THEN project each non-pinned
        //     particle out of every STATIC sphere (VERBATIM cloth.h::CollidePlane + CollideParticleSphere:
        //     int32 AABB reject -> int64 FxLength -> if inside, snap to the surface along FxNormalize(d),
        //     dist==0 -> the fixed +Y fallback). Deterministic (fixed particle order, fixed sphere order).
        for (int cp = 0; cp < particleCount; ++cp) {
            ClothParticle p = gParticles[cp];
            if (p.py < groundY) p.py = groundY;          // CollidePlane (idempotent ground clamp)
            if ((p.flags & HF_CLOTH_FLAG_PINNED) == 0u) {
                for (int sp = 0; sp < sphereCount; ++sp) {
                    int rcx = gSpheres[sp].cx;
                    int rcy = gSpheres[sp].cy;
                    int rcz = gSpheres[sp].cz;
                    int rad = gSpheres[sp].radius;
                    int dx = p.px - rcx;
                    int dy = p.py - rcy;
                    int dz = p.pz - rcz;
                    int ax = dx < 0 ? -dx : dx;
                    int ay = dy < 0 ? -dy : dy;
                    int az = dz < 0 ? -dz : dz;
                    if (ax > rad || ay > rad || az > rad) continue;   // int32 AABB reject -> no overlap
                    int dist = FxLength(dx, dy, dz);
                    if (dist >= rad) continue;                        // outside the sphere -> untouched
                    // FxNormalize(d): dist==0 -> the fixed +Y fallback {0,kOne,0}; else fxdiv per axis.
                    int nx, ny, nz;
                    if (dist == 0) { nx = 0; ny = HF_CLOTH_ONE; nz = 0; }
                    else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
                    // p.pos = center + n * radius (snap to the surface).
                    p.px = rcx + fxmul(nx, rad);
                    p.py = rcy + fxmul(ny, rad);
                    p.pz = rcz + fxmul(nz, rad);
                }
            }
            gParticles[cp] = p;
        }
    }
}
