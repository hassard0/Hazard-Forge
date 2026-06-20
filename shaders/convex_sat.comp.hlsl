// Slice CX1 — Deterministic Convex Rigid-Body Contacts: THE BOX-BOX SAT compute pass (the BEACHHEAD of
// FLAGSHIP #19: hf::sim::convex). ONE THREAD PER BOX PAIR (per-pair INDEPENDENT — each thread reads its
// pair's two bodies+boxes and writes its OWN SatResult slot; race-free, NO atomics) runs the 15-axis
// box-box Separating-Axis Test, copying engine/sim/convex.h::BoxSat's body VERBATIM (the SAME FxRotate
// box-axes, FxNormalize/FxDot/FxCross int64 ops, the SAME fixed 15-axis order, the SAME near-zero edge-
// cross skip, the SAME strict-< min-pen lowest-index tie-break) so the GPU SatResult[] is byte-identical
// to the CPU BoxSat -> the host GPU==CPU memcmp catches any divergence.
//
// INTEGER WIDTH (the determinism crux, the FPX3/FPX1 lesson): fxmul/fxdiv/FxISqrt + the FxDot/FxCross
// Q16.16 products use int64_t. DXC -spirv compiles int64 (the Int64 capability, the fpx_solve.comp /
// boids_steer.comp pattern); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so
// this shader is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl list). The
// Metal --convex-sat showcase runs the CPU convex::BoxSat over the same pairs -> byte-identical to this GPU
// result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// satEnabled=0 -> write a cleared SatResult (overlap=0) for every pair (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these would land at buffer(0..2)):
//   b0 gPairs   : the box-pair array (FxBody A + FxBox A + FxBody B + FxBox B per pair), READ.
//   b1 gResults : the SatResult array (overlap, axisIndex, penetration, axis.xyz) per pair, WRITE.
//   b2 gParams  : { pairCount, satEnabled, _, _ }, READ.

#define HF_FPX_FRAC 16   // MUST match fpx.h::kFrac
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)

// The near-zero edge-cross epsilon (MUST match convex.h::kEdgeEps == kOne/256).
#define HF_CX_EDGE_EPS (HF_FPX_ONE / 256)

// std430 FxBody mirror (engine/sim/fpx.h::FxBody): 16 x int32 (64 bytes) — pos.xyz, vel.xyz, invMass,
// flags, radius, orient.xyzw, angVel.xyz. The SAT only reads pos + orient; the rest rides along (the
// memcmp-able FxBody pack).
struct FxBody {
    int  px, py, pz;   // Q16.16 position
    int  vx, vy, vz;   // Q16.16 velocity
    int  invMass;      // Q16.16 inverse mass
    uint flags;        // bit0 = dynamic
    int  radius;       // Q16.16 broadphase radius
    int  ox, oy, oz, ow;   // Q16.16 orientation quaternion (x,y,z,w)
    int  ax, ay, az;       // Q16.16 angular velocity
};

// std430 SatPair mirror (engine/sim/convex.h::SatPair): FxBody A (16) + FxBox A (3) + FxBody B (16) +
// FxBox B (3) = 38 x int32 (152 bytes). FxBox is halfExtents.xyz.
struct SatPair {
    FxBody bodyA;
    int    hAx, hAy, hAz;   // box A half-extents
    FxBody bodyB;
    int    hBx, hBy, hBz;   // box B half-extents
};

// std430 SatResult mirror (the host packs convex::SatResult into THIS 6 x int32 / 24-byte form for the
// memcmp): overlap (0/1), axisIndex, penetration (Q16.16), axis.xyz (Q16.16, signed toward B).
struct SatResult {
    uint overlap;
    uint axisIndex;
    int  penetration;
    int  axisx, axisy, axisz;
};

struct SatParams {
    int4 cfg;   // x=pairCount, y=satEnabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<SatPair>   gPairs   : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<SatResult> gResults : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<SatParams> gParams  : register(u2);

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
int FxLength(int3 v) {
    int64_t sx = (int64_t)v.x * (int64_t)v.x;
    int64_t sy = (int64_t)v.y * (int64_t)v.y;
    int64_t sz = (int64_t)v.z * (int64_t)v.z;
    return (int)FxISqrt(sx + sy + sz);
}

// VERBATIM fpx.h::FxNormalize — unit vector via FxLength (int64); len==0 -> the fixed (0,1,0) fallback.
int3 FxNormalize(int3 v) {
    int len = FxLength(v);
    if (len == 0) return int3(0, HF_FPX_ONE, 0);
    return int3(fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len));
}

// VERBATIM convex.h::FxDot — (ax*bx + ay*by + az*bz) >> kFrac, int64 intermediate.
int FxDot(int3 a, int3 b) {
    int64_t d = (int64_t)a.x * (int64_t)b.x + (int64_t)a.y * (int64_t)b.y + (int64_t)a.z * (int64_t)b.z;
    return (int)(d >> HF_FPX_FRAC);
}

// VERBATIM convex.h::FxCross — the Q16.16 cross product (the fpx::FxRotate internal cross).
int3 FxCross(int3 a, int3 b) {
    return int3(fxmul(a.y, b.z) - fxmul(a.z, b.y),
                fxmul(a.z, b.x) - fxmul(a.x, b.z),
                fxmul(a.x, b.y) - fxmul(a.y, b.x));
}

// VERBATIM fpx.h::FxRotate — rotate v by the unit quaternion q (q.xyzw). v' = v + 2*cross(u, cross(u,v)+q.w*v).
int3 FxRotate(int4 q, int3 v) {
    int3 u = int3(q.x, q.y, q.z);
    int3 c1 = int3(fxmul(u.y, v.z) - fxmul(u.z, v.y),
                   fxmul(u.z, v.x) - fxmul(u.x, v.z),
                   fxmul(u.x, v.y) - fxmul(u.y, v.x));
    int3 t = int3(c1.x + fxmul(q.w, v.x), c1.y + fxmul(q.w, v.y), c1.z + fxmul(q.w, v.z));
    int3 c2 = int3(fxmul(u.y, t.z) - fxmul(u.z, t.y),
                   fxmul(u.z, t.x) - fxmul(u.x, t.z),
                   fxmul(u.x, t.y) - fxmul(u.y, t.x));
    return int3(v.x + 2 * c2.x, v.y + 2 * c2.y, v.z + 2 * c2.z);
}

int absI(int v) { return v < 0 ? -v : v; }

// VERBATIM convex.h::ProjectedRadius — |L·ax0|·h.x + |L·ax1|·h.y + |L·ax2|·h.z.
int ProjectedRadius(int3 L, int3 ax0, int3 ax1, int3 ax2, int3 h) {
    int d0 = absI(FxDot(L, ax0));
    int d1 = absI(FxDot(L, ax1));
    int d2 = absI(FxDot(L, ax2));
    return fxmul(d0, h.x) + fxmul(d1, h.y) + fxmul(d2, h.z);
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int pairCount  = gParams[0].cfg.x;
    int satEnabled = gParams[0].cfg.y;
    int idx = (int)gid.x;
    if (idx >= pairCount) return;

    // Disabled -> write a cleared result (the byte-identical no-op).
    if (satEnabled == 0) {
        SatResult z;
        z.overlap = 0u; z.axisIndex = 0u; z.penetration = 0; z.axisx = 0; z.axisy = 0; z.axisz = 0;
        gResults[idx] = z;
        return;
    }

    SatPair p = gPairs[idx];

    // The 3 world face axes of each box (FxRotate of the local X/Y/Z by the body orient) — VERBATIM BoxAxes.
    int4 qA = int4(p.bodyA.ox, p.bodyA.oy, p.bodyA.oz, p.bodyA.ow);
    int4 qB = int4(p.bodyB.ox, p.bodyB.oy, p.bodyB.oz, p.bodyB.ow);
    int3 axA0 = FxRotate(qA, int3(HF_FPX_ONE, 0, 0));
    int3 axA1 = FxRotate(qA, int3(0, HF_FPX_ONE, 0));
    int3 axA2 = FxRotate(qA, int3(0, 0, HF_FPX_ONE));
    int3 axB0 = FxRotate(qB, int3(HF_FPX_ONE, 0, 0));
    int3 axB1 = FxRotate(qB, int3(0, HF_FPX_ONE, 0));
    int3 axB2 = FxRotate(qB, int3(0, 0, HF_FPX_ONE));
    int3 axA[3] = { axA0, axA1, axA2 };
    int3 axB[3] = { axB0, axB1, axB2 };
    int3 hA = int3(p.hAx, p.hAy, p.hAz);
    int3 hB = int3(p.hBx, p.hBy, p.hBz);
    int3 t  = int3(p.bodyB.px - p.bodyA.px, p.bodyB.py - p.bodyA.py, p.bodyB.pz - p.bodyA.pz);

    bool found   = false;
    int  minPen  = 0;
    uint minIndex = 0u;
    int3 minAxis = int3(0, 0, 0);
    bool separated = false;

    // Iterate the 15 axes in the FIXED order: 3 A-faces (0..2), 3 B-faces (3..5), 9 edge-crosses (6..14).
    // The HLSL has no closures, so the testAxis body is inlined per group. skipDeg=true for edge-crosses.
    // The whole loop runs to completion (no early return) so the per-thread control flow is uniform-ish;
    // once `separated` is set, no min is updated and the final write is the separated result.
    for (int phase = 0; phase < 15 && !separated; ++phase) {
        int3 rawL;
        bool skipDeg = false;
        if (phase < 3) {
            rawL = axA[phase];
        } else if (phase < 6) {
            rawL = axB[phase - 3];
        } else {
            int e = phase - 6;          // 0..8
            int i = e / 3, j = e % 3;
            rawL = FxCross(axA[i], axB[j]);
            skipDeg = true;
        }
        if (skipDeg) {
            int rawLen = FxLength(rawL);
            if (rawLen < HF_CX_EDGE_EPS) continue;   // degenerate edge-cross -> ignore this axis
        }
        int3 L = FxNormalize(rawL);
        int rA = ProjectedRadius(L, axA0, axA1, axA2, hA);
        int rB = ProjectedRadius(L, axB0, axB1, axB2, hB);
        int dotLt = FxDot(L, t);
        int s = absI(dotLt);
        int sum = rA + rB;
        if (s > sum) { separated = true; continue; }
        int pen = sum - s;
        if (!found || pen < minPen) {
            found    = true;
            minPen   = pen;
            minIndex = (uint)phase;
            minAxis  = (dotLt < 0) ? int3(-L.x, -L.y, -L.z) : L;
        }
    }

    SatResult r;
    if (separated) {
        r.overlap = 0u; r.axisIndex = 0u; r.penetration = 0; r.axisx = 0; r.axisy = 0; r.axisz = 0;
    } else {
        r.overlap = 1u; r.axisIndex = minIndex; r.penetration = minPen;
        r.axisx = minAxis.x; r.axisy = minAxis.y; r.axisz = minAxis.z;
    }
    gResults[idx] = r;
}
