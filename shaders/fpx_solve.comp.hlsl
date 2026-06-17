// Slice FPX3 — Deterministic Fixed-Point Physics: the PBD POSITIONAL collision-response compute pass (the
// MAKE-OR-BREAK of FLAGSHIP #6). A SINGLE THREAD ([numthreads(1,1,1)], the mc_scan/vt_alloc allocator
// pattern; gid.x!=0 -> return) runs `steps` iterations of StepWorld = IntegrateStep then SolveContacts
// (K=solveIters Gauss-Seidel sweeps): each sweep resolves ALL ground contacts (in body order) then ALL
// sphere-sphere pair contacts in the FIXED FPX2 pair order (ascending). The contact resolution is
// inherently SEQUENTIAL (each contact reads the LATEST positions) -> one thread -> bit-exact GPU==CPU +
// cross-backend, NO atomics, NO race. The fxmul/fxdiv/FxISqrt/FxNormalize + ground + sphere-sphere math is
// copied VERBATIM from engine/sim/fpx.h so tests/fpx_test.cpp + the GPU pass exercise the EXACT integer ops
// -> a divergence is exactly what the host GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux): fxmul/fxdiv use int64_t (IDENTICAL to fpx.h). DXC -spirv compiles
// int64 (the Int64 capability, the swraster.comp / fpx_integrate.comp pattern); glslc (the Metal
// HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (in the
// Vulkan compile list, NOT in the Metal hf_gen_msl list). The Metal --fpx-solve showcase runs the CPU
// fpx::StepWorld over the same world -> byte-identical to this GPU result by construction, while the Vulkan
// side carries the GPU==CPU bit-identity proof.
//
// solveEnabled=0 -> write the input bodies back UNCHANGED (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these would land at buffer(0..2)):
//   b0 gBodies : the Q16.16 FxBody array (pos.xyz, vel.xyz, invMass, flags, radius — std430 ints), R+W.
//   b1 gPairs  : the FPX2 candidate-pair list (i,j per pair), READ.
//   b2 gParams : { gravity.xyz, dt, groundY, bodyCount, pairCount, steps, solveIters, solveEnabled }, READ.

#define HF_FPX_FRAC 16   // MUST match fpx.h::kFrac
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)

// std430 FxBody mirror (engine/sim/fpx.h::FxBody): pos.xyz, vel.xyz, invMass, flags, radius. 9 x 4-byte =
// 36 bytes (matches the FPX2 FpxBody layout, memcmp-able).
struct FpxBody {
    int  px, py, pz;   // Q16.16 position
    int  vx, vy, vz;   // Q16.16 velocity
    int  invMass;      // Q16.16 inverse mass (0 => static)
    uint flags;        // bit0 = dynamic
    int  radius;       // Q16.16 sphere radius
};

struct FxPair {
    uint i, j;
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=groundY (Q16.16), y=bodyCount, z=pairCount, w=steps
//   cfg2 : x=solveIters, y=solveEnabled, z=_, w=_
struct FpxParams {
    int4 grav;
    int4 cfg;
    int4 cfg2;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FpxBody>   gBodies : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FxPair>    gPairs  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FpxParams> gParams : register(u2);

// VERBATIM fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_FPX_FRAC);
}

// VERBATIM fpx.h::fxdiv — (a << kFrac) / b in Q16.16 (int64 shift + truncating divide; guard b==0).
int fxdiv(int a, int b) {
    if (b == 0) return 0;
    return (int)(((int64_t)a << HF_FPX_FRAC) / (int64_t)b);
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
    int groundY      = gParams[0].cfg.x;
    int bodyCount    = gParams[0].cfg.y;
    int pairCount    = gParams[0].cfg.z;
    int steps        = gParams[0].cfg.w;
    int solveIters   = gParams[0].cfg2.x;
    int solveEnabled = gParams[0].cfg2.y;

    // Disabled -> write the input back UNCHANGED (the byte-identical no-op proof).
    if (solveEnabled == 0) return;

    const uint kFlagDynamic = 1u;

    for (int s = 0; s < steps; ++s) {
        // (A) IntegrateStep: per-body semi-implicit-Euler + ground floor-clamp (VERBATIM fpx.h::IntegrateBody).
        for (int i = 0; i < bodyCount; ++i) {
            FpxBody b = gBodies[i];
            if ((b.flags & kFlagDynamic) != 0u) {
                b.vx += fxmul(gravx, dt);
                b.vy += fxmul(gravy, dt);
                b.vz += fxmul(gravz, dt);
                b.px += fxmul(b.vx, dt);
                b.py += fxmul(b.vy, dt);
                b.pz += fxmul(b.vz, dt);
                if (b.py < groundY) {
                    b.py = groundY;
                    if (b.vy < 0) b.vy = 0;
                }
            }
            gBodies[i] = b;
        }

        // (B) SolveContacts: K Gauss-Seidel sweeps (ground then pairs in FPX2 order). VERBATIM fpx.h.
        for (int it = 0; it < solveIters; ++it) {
            // (B1) ALL ground contacts (body order).
            for (int gi = 0; gi < bodyCount; ++gi) {
                FpxBody b = gBodies[gi];
                if (b.invMass != 0) {
                    int pen = groundY + b.radius - b.py;   // ResolveGround
                    if (pen > 0) { b.py += pen; gBodies[gi] = b; }
                }
            }
            // (B2) ALL sphere-sphere pair contacts (FIXED ascending FPX2 order; Gauss-Seidel).
            for (int pi = 0; pi < pairCount; ++pi) {
                uint ia = gPairs[pi].i;
                uint ib = gPairs[pi].j;
                FpxBody a = gBodies[ia];
                FpxBody bb = gBodies[ib];
                int invSum = a.invMass + bb.invMass;       // ResolvePair
                if (invSum != 0) {
                    int dx = bb.px - a.px;
                    int dy = bb.py - a.py;
                    int dz = bb.pz - a.pz;
                    int dist = FxLength(dx, dy, dz);
                    int pen = (a.radius + bb.radius) - dist;
                    if (pen > 0) {
                        // FxNormalize(d).
                        int nx, ny, nz;
                        if (dist == 0) { nx = 0; ny = HF_FPX_ONE; nz = 0; }
                        else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
                        int wi = fxdiv(a.invMass, invSum);
                        int wj = HF_FPX_ONE - wi;
                        int ai = fxmul(pen, wi);
                        int aj = fxmul(pen, wj);
                        a.px -= fxmul(nx, ai); a.py -= fxmul(ny, ai); a.pz -= fxmul(nz, ai);
                        bb.px += fxmul(nx, aj); bb.py += fxmul(ny, aj); bb.pz += fxmul(nz, aj);
                        gBodies[ia] = a;
                        gBodies[ib] = bb;
                    }
                }
            }
        }
    }
}
