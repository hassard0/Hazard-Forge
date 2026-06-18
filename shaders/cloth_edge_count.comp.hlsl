// Slice CL2 — Deterministic GPU Cloth: the DISTANCE-CONSTRAINT GRAPH per-particle EDGE-COUNT compute pass
// (the 2nd CL slice, after CL1's integrator; the FPX2 fpx_pair_count / MC3 mc_count analog on lattice
// particles). ONE thread per particle p = r*W + c (p < particleCount). The thread counts the edges OWNED
// by particle (r,c) — each unordered edge is owned by its LOWER linear index, so a particle owns the
// FORWARD edges that stay in-bounds, in the FIXED order [right, down, diag1 (down-right), diag2
// (down-left), bend-right, bend-down] — and writes gEdgeCount[p]. NO atomics (each thread writes its own
// count); the scan pass sums. enabled=0 -> write 0 (the byte-identical no-op).
//
// WHY BIT-IDENTICAL to the CPU cloth.h::CountOwnedEdges (the make-or-break): the whole count is PURE INT32
// — only integer index arithmetic + in-bounds compares, NO products, NO int64, NO float. So this
// MSL-generates NATIVELY on Metal (no int64, no --msl-version 20200), unlike CL1's int64 fxmul integrator
// which is Vulkan-only. A divergence vs the header is exactly what the host's GPU==CPU memcmp catches.
//
// Buffers (storage, bound at compute bindings 0..1; on Metal these land at buffer(0..1)):
//   b0 gEdgeCount : one uint per particle (the owned-edge count), WRITE.
//   b1 gParams    : { W, H, particleCount, enabled, restStruct, restShear, restBend, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations (same as
// fpx_pair_count.comp / nav_raster_count.comp), not backend CODE symbols.

#define HF_CLOTH_THREADS 64

// std430 ClothEdgeParams mirror (matches the C++ upload struct): two int4 — grid {W, H, particleCount,
// enabled} + rest {restStruct, restShear, restBend, _}. The per-kind rest lengths are the host-snapped
// int32 FxLength of the flat sheet (structural = spacing, shear = FxLength(spacing,spacing,0), bend =
// 2*spacing) — computed host-side (the only int64 use, on the CPU) so the shaders stay PURE int32.
struct ClothEdgeParams {
    int4 grid;   // x=W, y=H, z=particleCount, w=enabled
    int4 rest;   // x=restStruct, y=restShear, z=restBend, w=unused
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            gEdgeCount : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ClothEdgeParams> gParams    : register(u1);

// CountOwnedEdges(W,H,r,c): VERBATIM cloth.h::CountOwnedEdges — the owned forward edges in bounds.
uint CountOwnedEdges(int W, int H, int r, int c) {
    uint n = 0u;
    if (c + 1 < W)                ++n;   // STRUCTURAL right
    if (r + 1 < H)                ++n;   // STRUCTURAL down
    if (r + 1 < H && c + 1 < W)   ++n;   // SHEAR diag1 down-right
    if (r + 1 < H && c - 1 >= 0)  ++n;   // SHEAR diag2 down-left
    if (c + 2 < W)                ++n;   // BEND right
    if (r + 2 < H)                ++n;   // BEND down
    return n;
}

[numthreads(HF_CLOTH_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int W             = gParams[0].grid.x;
    int H             = gParams[0].grid.y;
    int particleCount = gParams[0].grid.z;
    int enabled       = gParams[0].grid.w;

    uint p = gid.x;
    if ((int)p >= particleCount) return;

    // Disabled -> write 0 (gEdgeCount stays the cleared all-zero upload; the byte-identical no-op).
    if (enabled == 0) { gEdgeCount[p] = 0u; return; }

    int r = (int)p / W;
    int c = (int)p % W;
    gEdgeCount[p] = CountOwnedEdges(W, H, r, c);   // order-independent per-particle write (NO atomics)
}
