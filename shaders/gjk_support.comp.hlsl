// Slice GJ1 — General Convex-Hull Contacts: THE HULL + SUPPORT FUNCTION compute pass (the BEACHHEAD of
// FLAGSHIP #22: hf::sim::gjk). ONE THREAD PER (hull, direction) QUERY (per-query INDEPENDENT — each thread
// reads its query's hull + body + direction and writes its OWN support slot; race-free, NO atomics) runs the
// world-space support function, copying engine/sim/gjk.h::Support's body VERBATIM (the SAME conjugate-rotate
// the direction into local, the SAME FIXED strict-greater lowest-index argmax SupportLocal, the SAME FxRotate
// out + FxAdd pos) so the GPU support buffer is byte-identical to the CPU gjk::Support -> the host GPU==CPU
// memcmp catches any divergence.
//
// INTEGER WIDTH (the determinism crux, the CX1/FPX3 lesson): fxmul + the FxDot/FxRotate Q16.16 products use
// int64_t. DXC -spirv compiles int64 (the Int64 capability, the convex_sat.comp / fpx_solve.comp pattern);
// glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-
// ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl list). The Metal --gjk-support showcase runs
// the CPU gjk::Support over the same queries -> byte-identical to this GPU result BY CONSTRUCTION, while the
// Vulkan side carries the GPU==CPU bit-identity proof.
//
// queryEnabled=0 -> write a cleared support (0,0,0; extent 0) for every query (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..3):
//   b0 gHulls   : the FxHull array (kMaxHullVerts x int3 verts + uint count per hull), READ.
//   b1 gQueries : the query array (hullIndex, FxBody, dir.xyz) per query, READ.
//   b2 gOut     : the SupportOut array (support.xyz + extent) per query, WRITE.
//   b3 gParams  : { queryCount, queryEnabled, _, _ }, READ.

#define HF_FPX_FRAC 16   // MUST match fpx.h::kFrac
#define HF_GJK_MAX_VERTS 20   // MUST match gjk.h::kMaxHullVerts

// std430 FxHull mirror (engine/sim/gjk.h::FxHull): kMaxHullVerts x int3 verts (each int3 padded to int4 in a
// StructuredBuffer's std430 array-of-struct? No — HLSL packs an int3 as 12 bytes inside a struct only if the
// next member aligns; to be SAFE + match the host pack we store the verts as a flat int array (3*N) + count).
struct FxHull {
    int  vx[HF_GJK_MAX_VERTS];
    int  vy[HF_GJK_MAX_VERTS];
    int  vz[HF_GJK_MAX_VERTS];
    uint count;
};

// std430 FxBody mirror (engine/sim/fpx.h::FxBody): 16 x int32 (64 bytes). The support only reads pos + orient.
struct FxBody {
    int  px, py, pz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
    int  radius;
    int  ox, oy, oz, ow;
    int  ax, ay, az;
};

// std430 Query mirror: a hull index + the placed body + the world direction.
struct Query {
    uint   hullIndex;
    uint   _pad0, _pad1, _pad2;
    FxBody body;
    int    dx, dy, dz;
    int    _pad3;
};

// std430 SupportOut mirror: the world support vertex + its extent (FxDot(support, dir)).
struct SupportOut {
    int sx, sy, sz;
    int extent;
};

struct Params {
    int4 cfg;   // x=queryCount, y=queryEnabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxHull>     gHulls   : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<Query>      gQueries : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<SupportOut> gOut     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<Params>     gParams  : register(u3);

// VERBATIM fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_FPX_FRAC);
}

// VERBATIM convex.h::FxDot — (ax*bx + ay*by + az*bz) >> kFrac, int64 intermediate.
int FxDot(int3 a, int3 b) {
    int64_t d = (int64_t)a.x * (int64_t)b.x + (int64_t)a.y * (int64_t)b.y + (int64_t)a.z * (int64_t)b.z;
    return (int)(d >> HF_FPX_FRAC);
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

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int queryCount   = gParams[0].cfg.x;
    int queryEnabled = gParams[0].cfg.y;
    int idx = (int)gid.x;
    if (idx >= queryCount) return;

    // Disabled -> write a cleared support (the byte-identical no-op).
    if (queryEnabled == 0) {
        SupportOut z;
        z.sx = 0; z.sy = 0; z.sz = 0; z.extent = 0;
        gOut[idx] = z;
        return;
    }

    Query q = gQueries[idx];
    FxHull hull = gHulls[q.hullIndex];
    int4 orient = int4(q.body.ox, q.body.oy, q.body.oz, q.body.ow);
    int3 dir = int3(q.dx, q.dy, q.dz);

    // VERBATIM gjk::Support: rotate dir into body-local by the CONJUGATE of orient ({-x,-y,-z,w}).
    int4 conj = int4(-orient.x, -orient.y, -orient.z, orient.w);
    int3 localDir = FxRotate(conj, dir);

    // VERBATIM gjk::SupportLocal: FIXED scan, STRICT-greater update -> ties keep the LOWEST index.
    int3 localV = int3(0, 0, 0);
    if (hull.count > 0u) {
        uint best = 0u;
        int3 v0 = int3(hull.vx[0], hull.vy[0], hull.vz[0]);
        int bestDot = FxDot(v0, localDir);
        for (uint i = 1u; i < hull.count; ++i) {
            int3 vi = int3(hull.vx[i], hull.vy[i], hull.vz[i]);
            int d = FxDot(vi, localDir);
            if (d > bestDot) { bestDot = d; best = i; }
        }
        localV = int3(hull.vx[best], hull.vy[best], hull.vz[best]);
    }

    // VERBATIM gjk::Support: map the local support vertex back to world (FxRotate(orient, v) + pos).
    int3 worldV = FxRotate(orient, localV);
    int3 support = int3(worldV.x + q.body.px, worldV.y + q.body.py, worldV.z + q.body.pz);

    SupportOut r;
    r.sx = support.x; r.sy = support.y; r.sz = support.z;
    r.extent = FxDot(support, dir);
    gOut[idx] = r;
}
