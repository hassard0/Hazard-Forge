// Slice JT1 — Deterministic Articulated-Body Ragdoll: the BALL-JOINT CONSTRAINT solve compute pass (the
// BEACHHEAD of FLAGSHIP #15). A SINGLE THREAD ([numthreads(1,1,1)], the fpx_solve.comp / cloth_solve.comp
// pattern; gid.x!=0 -> return) runs `steps` iterations of StepJointWorld = IntegrateBodyFull each body (the
// FPX4 6-DOF semi-implicit-Euler: dynamic bodies get vel += gravity*dt; pos += vel*dt; every body's
// orientation integrates from angVel) then `iters` Gauss-Seidel ball passes (each iterating ALL joints in
// the FIXED joint order applying SolveBallJoint = the cloth distance constraint with restLen 0 over the
// world anchors, the correction applied to the body centres) then a ground floor-clamp. The ball projection
// is inherently SEQUENTIAL (each joint reads the body centres earlier joints THIS pass already moved) ->
// one thread -> bit-exact GPU==CPU + cross-backend, NO atomics, NO race. The fxmul/fxdiv/FxISqrt/FxNormalize
// + FxRotate + IntegrateBodyFull + the ball-joint projection + floor clamp are copied VERBATIM from
// engine/sim/joint.h (SolveBallJoint / StepJointWorld, which reuse fpx.h) so tests/joint_test.cpp + the GPU
// pass exercise the EXACT integer ops -> a divergence is exactly what the host GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux): fxmul/fxdiv/FxISqrt + the FxRotate/quaternion math use int64_t
// (IDENTICAL to joint.h/fpx.h). DXC -spirv compiles int64 (the Int64 capability, the fpx_solve.comp /
// cloth_solve.comp pattern); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so
// this shader is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl list). The
// Metal --joint-ball showcase runs the CPU joint::StepJointWorld over the same chain -> byte-identical to
// this GPU result by construction, while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// JT1 SIMPLIFICATION (documented): the ball joint TRANSLATES the body centres (the cloth positional split),
// it does NOT rotate a body to align an off-centre anchor (no lever-arm / angular coupling — JT2/JT3).
//
// solveEnabled=0 -> write the input bodies back UNCHANGED (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..2):
//   b0 gBodies : the Q16.16 FxBody array (pos.xyz, vel.xyz, invMass, flags, radius, orient.xyzw,
//                angVel.xyz — std430 ints, 64 bytes), READ+WRITE.
//   b1 gJoints : the FxJoint list (bodyA, bodyB, anchorA.xyz, anchorB.xyz, kind, limit — std430 ints,
//                40 bytes), READ.
//   b2 gParams : { gravity.xyz, dt, groundY, bodyCount, jointCount, steps, iters, solveEnabled }, READ.

#define HF_JOINT_FRAC 16   // MUST match joint.h::kFrac (== fpx.h::kFrac)
#define HF_JOINT_ONE  (1 << HF_JOINT_FRAC)
#define HF_JOINT_FLAG_DYNAMIC 1u

// std430 FxBody mirror (engine/sim/fpx.h::FxBody): pos.xyz, vel.xyz, invMass, flags, radius, orient.xyzw,
// angVel.xyz. 16 x 4-byte = 64 bytes, no padding holes (memcmp-able; == the host FxBodyGpu mirror).
struct FxBody {
    int  px, py, pz;          // Q16.16 position
    int  vx, vy, vz;          // Q16.16 velocity
    int  invMass;             // Q16.16 inverse mass (0 => static/pinned)
    uint flags;               // bit0 = dynamic
    int  radius;              // Q16.16 broadphase half-extent (unused by JT1)
    int  ox, oy, oz, ow;      // Q16.16 orientation quaternion
    int  ax, ay, az;          // Q16.16 angular velocity
};

// std430 FxJoint mirror (engine/sim/joint.h::FxJoint): bodyA, bodyB, anchorA.xyz, anchorB.xyz, kind, limit.
// 10 x 4-byte = 40 bytes, no padding holes.
struct FxJoint {
    uint bodyA, bodyB;        // body indices
    int  aax, aay, aaz;       // anchorA (Q16.16 body-local offset on bodyA)
    int  abx, aby, abz;       // anchorB (Q16.16 body-local offset on bodyB)
    uint kind;                // kJointBall (JT1)
    int  limit;               // angular limit (unused at JT1)
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=groundY (Q16.16), y=bodyCount, z=jointCount, w=steps
//   cfg2 : x=iters, y=solveEnabled, z=_, w=_
struct JointSolveParams {
    int4 grav;
    int4 cfg;
    int4 cfg2;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxBody>          gBodies : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FxJoint>         gJoints : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<JointSolveParams> gParams : register(u2);

// VERBATIM fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_JOINT_FRAC);
}

// VERBATIM fpx.h::fxdiv — (a << kFrac) / b in Q16.16 (int64 shift + truncating divide; guard b==0).
int fxdiv(int a, int b) {
    if (b == 0) return 0;
    return (int)(((int64_t)a << HF_JOINT_FRAC) / (int64_t)b);
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

// VERBATIM fpx.h::FxQuatMul — Hamilton product, each term an int64 fxmul.
void FxQuatMul(int ax2, int ay2, int az2, int aw2, int bx, int by, int bz, int bw,
               out int rx, out int ry, out int rz, out int rw) {
    rw = fxmul(aw2, bw) - fxmul(ax2, bx) - fxmul(ay2, by) - fxmul(az2, bz);
    rx = fxmul(aw2, bx) + fxmul(ax2, bw) + fxmul(ay2, bz) - fxmul(az2, by);
    ry = fxmul(aw2, by) - fxmul(ax2, bz) + fxmul(ay2, bw) + fxmul(az2, bx);
    rz = fxmul(aw2, bz) + fxmul(ax2, by) - fxmul(ay2, bx) + fxmul(az2, bw);
}

// VERBATIM fpx.h::FxQuatNormalize — unit quaternion via FxISqrt of the Q32.32 sum-of-squares; len==0 ->
// identity.
void FxQuatNormalize(inout int qx, inout int qy, inout int qz, inout int qw) {
    int64_t sx = (int64_t)qx * (int64_t)qx;
    int64_t sy = (int64_t)qy * (int64_t)qy;
    int64_t sz = (int64_t)qz * (int64_t)qz;
    int64_t sw = (int64_t)qw * (int64_t)qw;
    int len = (int)FxISqrt(sx + sy + sz + sw);
    if (len == 0) { qx = 0; qy = 0; qz = 0; qw = HF_JOINT_ONE; return; }
    qx = fxdiv(qx, len); qy = fxdiv(qy, len); qz = fxdiv(qz, len); qw = fxdiv(qw, len);
}

// VERBATIM fpx.h::FxRotate — rotate v by the (unit) quaternion q via v' = v + 2*cross(u, cross(u,v)+w*v).
void FxRotate(int qx, int qy, int qz, int qw, int vx, int vy, int vz,
              out int ox, out int oy, out int oz) {
    // u = q.xyz; c1 = cross(u, v).
    int c1x = fxmul(qy, vz) - fxmul(qz, vy);
    int c1y = fxmul(qz, vx) - fxmul(qx, vz);
    int c1z = fxmul(qx, vy) - fxmul(qy, vx);
    // t = c1 + q.w*v.
    int tx = c1x + fxmul(qw, vx);
    int ty = c1y + fxmul(qw, vy);
    int tz = c1z + fxmul(qw, vz);
    // c2 = cross(u, t).
    int c2x = fxmul(qy, tz) - fxmul(qz, ty);
    int c2y = fxmul(qz, tx) - fxmul(qx, tz);
    int c2z = fxmul(qx, ty) - fxmul(qy, tx);
    // v' = v + 2*c2.
    ox = vx + 2 * c2x; oy = vy + 2 * c2y; oz = vz + 2 * c2z;
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
    int jointCount   = gParams[0].cfg.z;
    int steps        = gParams[0].cfg.w;
    int iters        = gParams[0].cfg2.x;
    int solveEnabled = gParams[0].cfg2.y;

    // Disabled -> write the input back UNCHANGED (the byte-identical no-op proof).
    if (solveEnabled == 0) return;

    int kHalf = HF_JOINT_ONE / 2;   // 0.5 in Q16.16 (== fpx.h::kHalf)

    for (int s = 0; s < steps; ++s) {
        // (1) IntegrateBodyFull each body (VERBATIM fpx.h::IntegrateBodyFull): dynamic bodies get the
        //     translational semi-implicit-Euler; every body's orientation integrates from angVel.
        for (int i = 0; i < bodyCount; ++i) {
            FxBody b = gBodies[i];
            if ((b.flags & HF_JOINT_FLAG_DYNAMIC) != 0u) {
                b.vx += fxmul(gravx, dt);
                b.vy += fxmul(gravy, dt);
                b.vz += fxmul(gravz, dt);
                b.px += fxmul(b.vx, dt);
                b.py += fxmul(b.vy, dt);
                b.pz += fxmul(b.vz, dt);
            }
            // IntegrateOrientation: dq = omega⊗q; orient += 0.5*dq*dt; renormalize.
            int dqx, dqy, dqz, dqw;
            FxQuatMul(b.ax, b.ay, b.az, 0, b.ox, b.oy, b.oz, b.ow, dqx, dqy, dqz, dqw);
            b.ox += fxmul(fxmul(dqx, kHalf), dt);
            b.oy += fxmul(fxmul(dqy, kHalf), dt);
            b.oz += fxmul(fxmul(dqz, kHalf), dt);
            b.ow += fxmul(fxmul(dqw, kHalf), dt);
            FxQuatNormalize(b.ox, b.oy, b.oz, b.ow);
            gBodies[i] = b;
        }

        // (2) `iters` Gauss-Seidel ball passes in the FIXED joint order. SEQUENTIAL — each joint reads +
        //     writes gBodies so the next joint sees the update (VERBATIM joint.h::SolveBallJoint).
        for (int it = 0; it < iters; ++it) {
            for (int e = 0; e < jointCount; ++e) {
                uint ja = gJoints[e].bodyA;
                uint jb = gJoints[e].bodyB;
                if (ja >= (uint)bodyCount || jb >= (uint)bodyCount) continue;
                FxBody a = gBodies[ja];
                FxBody b = gBodies[jb];
                int wsum = a.invMass + b.invMass;
                if (wsum != 0) {
                    // WorldAnchor(a, anchorA) = a.pos + FxRotate(a.orient, anchorA).
                    int rax, ray, raz;
                    FxRotate(a.ox, a.oy, a.oz, a.ow, gJoints[e].aax, gJoints[e].aay, gJoints[e].aaz,
                             rax, ray, raz);
                    int pax = a.px + rax, pay = a.py + ray, paz = a.pz + raz;
                    int rbx, rby, rbz;
                    FxRotate(b.ox, b.oy, b.oz, b.ow, gJoints[e].abx, gJoints[e].aby, gJoints[e].abz,
                             rbx, rby, rbz);
                    int pbx = b.px + rbx, pby = b.py + rby, pbz = b.pz + rbz;
                    int dx = pbx - pax;
                    int dy = pby - pay;
                    int dz = pbz - paz;
                    int len = FxLength(dx, dy, dz);
                    if (len != 0) {
                        int pen = len;   // restLen 0
                        // FxNormalize(d) (len != 0 here so the fallback branch is dead, kept verbatim).
                        int nx = fxdiv(dx, len);
                        int ny = fxdiv(dy, len);
                        int nz = fxdiv(dz, len);
                        int wa = fxdiv(a.invMass, wsum);
                        int wb = fxdiv(b.invMass, wsum);
                        int aa = fxmul(pen, wa);
                        int ab = fxmul(pen, wb);
                        a.px += fxmul(nx, aa); a.py += fxmul(ny, aa); a.pz += fxmul(nz, aa);
                        b.px -= fxmul(nx, ab); b.py -= fxmul(ny, ab); b.pz -= fxmul(nz, ab);
                        gBodies[ja] = a;
                        gBodies[jb] = b;
                    }
                }
            }
        }

        // (3) ground floor clamp AFTER the constraint passes (a joint may pull a body below ground).
        for (int g = 0; g < bodyCount; ++g) {
            FxBody b = gBodies[g];
            if ((b.flags & HF_JOINT_FLAG_DYNAMIC) != 0u && b.py < groundY) {
                b.py = groundY;
                gBodies[g] = b;
            }
        }
    }
}
