// Slice CL2 — Deterministic GPU Cloth: the DISTANCE-CONSTRAINT GRAPH per-particle EDGE-EMIT compute pass
// (the FPX2 fpx_pair_emit / MC3 mc_emit analog on lattice particles). ONE thread per particle p = r*W + c
// (p < particleCount). The thread reads its write base gEdgeOffset[p] (the CL2 prefix-sum), re-derives its
// OWNED forward edges in the FIXED order [right, down, diag1 (down-right), diag2 (down-left), bend-right,
// bend-down], and EMITS each as a Constraint{i, j, restLen, kind} into gEdges at gEdgeOffset[p] + local++.
// Each particle writes into its OWN DISJOINT [offset[p], offset[p]+count[p]) range -> race-free, NO
// atomics. enabled=0 -> emit nothing (gEdges stays the pre-cleared upload; the byte-identical no-op).
//
// WHY BIT-IDENTICAL to the CPU cloth.h::BuildConstraints (the make-or-break): every value written is PURE
// INT32 — the same owned-edge enumeration in the same FIXED order the CPU uses, and restLen is the
// host-snapped int32 per-kind rest length read from params (restStruct/restShear/restBend = the CPU's
// FxLength of the flat sheet, computed host-side — the only int64 use, on the CPU). The grouped-by-owner
// (ascending p) then fixed-per-particle order matches BuildConstraints exactly. A divergence vs the header
// is exactly what the host's GPU==CPU memcmp (gEdges) catches. INT32 only, NO sqrt, NO int64 -> MSL-gens
// natively on Metal (unlike CL1's int64 cloth_integrate).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gEdgeOffset : one uint per particle (the CL2 prefix-sum write base), READ.
//   b1 gEdges      : the output Constraint array {i, j, restLen, kind} per slot, WRITE (pre-cleared).
//   b2 gParams     : { W, H, particleCount, enabled, restStruct, restShear, restBend, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_CLOTH_THREADS 64

// std430 Constraint mirror (engine/sim/cloth.h::Constraint): 4 x int32 (16 bytes) = i, j, restLen, kind.
struct Constraint {
    uint i;        // lower endpoint index
    uint j;        // upper endpoint index
    int  restLen;  // Q16.16 rest length (host-snapped int32)
    uint kind;     // 0=STRUCTURAL, 1=SHEAR, 2=BEND
};

// std430 ClothEdgeParams mirror (IDENTICAL to cloth_edge_count.comp's). grid {W,H,particleCount,enabled}
// + rest {restStruct, restShear, restBend, _}.
struct ClothEdgeParams {
    int4 grid;   // x=W, y=H, z=particleCount, w=enabled
    int4 rest;   // x=restStruct, y=restShear, z=restBend, w=unused
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            gEdgeOffset : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<Constraint>      gEdges      : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<ClothEdgeParams> gParams     : register(u2);

// Constants matching cloth.h::kConstraint{Structural,Shear,Bend}.
static const uint kStructural = 0u;
static const uint kShear      = 1u;
static const uint kBend       = 2u;

[numthreads(HF_CLOTH_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int W             = gParams[0].grid.x;
    int H             = gParams[0].grid.y;
    int particleCount = gParams[0].grid.z;
    int enabled       = gParams[0].grid.w;
    int restStruct    = gParams[0].rest.x;
    int restShear     = gParams[0].rest.y;
    int restBend      = gParams[0].rest.z;

    uint p = gid.x;
    if ((int)p >= particleCount) return;

    // Disabled -> emit nothing (gEdges stays the pre-cleared upload; the byte-identical no-op).
    if (enabled == 0) return;

    int r = (int)p / W;
    int c = (int)p % W;
    uint base  = gEdgeOffset[p];   // this particle's disjoint write base (from the prefix-sum)
    uint local = 0u;

    // FIXED per-particle emit order — VERBATIM cloth.h::BuildConstraints (right, down, diag1, diag2,
    // bend-right, bend-down). restLen by kind from the host-snapped params.
    if (c + 1 < W) {                                  // STRUCTURAL right
        Constraint e; e.i = p; e.j = (uint)(r * W + (c + 1));       e.restLen = restStruct; e.kind = kStructural;
        gEdges[base + local] = e; local += 1u;
    }
    if (r + 1 < H) {                                  // STRUCTURAL down
        Constraint e; e.i = p; e.j = (uint)((r + 1) * W + c);       e.restLen = restStruct; e.kind = kStructural;
        gEdges[base + local] = e; local += 1u;
    }
    if (r + 1 < H && c + 1 < W) {                     // SHEAR diag1 down-right
        Constraint e; e.i = p; e.j = (uint)((r + 1) * W + (c + 1)); e.restLen = restShear;  e.kind = kShear;
        gEdges[base + local] = e; local += 1u;
    }
    if (r + 1 < H && c - 1 >= 0) {                    // SHEAR diag2 down-left
        Constraint e; e.i = p; e.j = (uint)((r + 1) * W + (c - 1)); e.restLen = restShear;  e.kind = kShear;
        gEdges[base + local] = e; local += 1u;
    }
    if (c + 2 < W) {                                  // BEND right
        Constraint e; e.i = p; e.j = (uint)(r * W + (c + 2));       e.restLen = restBend;   e.kind = kBend;
        gEdges[base + local] = e; local += 1u;
    }
    if (r + 2 < H) {                                  // BEND down
        Constraint e; e.i = p; e.j = (uint)((r + 2) * W + c);       e.restLen = restBend;   e.kind = kBend;
        gEdges[base + local] = e; local += 1u;
    }
}
