// Slice FR2 — Deterministic Rigid-Body Fracture/Destruction: FRAGMENT EXTRACTION per-fragment REDUCTION
// compute pass (the 4th and final FR2 pass; the per-fragment mass-property reduce over the FR2 CSR). ONE
// thread per FRAGMENT (f < fragCount). Thread f reads its source cell = fragmentToCell[f], walks its CSR
// slice fragSamples[fragStart[cell] .. fragStart[cell+1]) in ASCENDING sample order, and reduces it to its
// own FractFragment record (centroid, AABB, boundRadiusSq, boundRadius, volume, cellId, invMass). Each
// fragment writes ONLY its own record -> per-fragment DISJOINT, race-free, NO atomics. The fixed ascending
// reduction order makes the integer sums/min/max bit-identical CPU<->GPU<->cross-vendor.
//
// WHY BIT-IDENTICAL to the CPU fract.h::ExtractFragments reduce (the make-or-break): the whole reduce is
// PURE INT32 — sums of small lattice coords (< ~1024), per-axis min/max, squared distances (< ~3.1M, fit
// int32), a truncating integer divide for the centroid, an int32 binary ISqrt for boundRadius, and a
// pure-integer kOne/volume reciprocal for invMass. NO int64, NO float, NO Q16.16 world scaling -> this
// MSL-generates NATIVELY on Metal (the GR2/FR1 precedent), a TRUE GPU pass on both backends. Any divergence
// vs the header is exactly what the host's GPU==CPU memcmp catches.
//
// THE RECORD LAYOUT (std430, 14 x 4-byte = 56 bytes, memcmp-able with C++ fract.h::FractFragment):
//   cx,cy,cz (centroid) | minx,miny,minz,maxx,maxy,maxz (AABB) | boundRadiusSq | boundRadius | volume |
//   cellId | invMass (Q16.16, the ONLY field that is conceptually fixed-point, but derived by a pure
//   integer kOne/volume).
//
// SampleCoord decompose (VERBATIM fract.h::SampleCoord, the inverse of SampleIndex):
//   x = idx % nx; y = (idx / nx) % ny; z = idx / (nx*ny).
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 fragmentToCell : F uints (fragment -> source cell index), READ.
//   b1 fragStart      : seedCount+1 uints (the CSR row pointers), READ.
//   b2 fragSamples    : sampleCount uints (sample indices grouped by cell, ascending), READ.
//   b3 gFragments     : F FractFragment records (56 bytes each), WRITE (per-fragment disjoint).
//   b4 gParams        : the FR2 params (nx, ny, nz, fragCount), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_FRACT_THREADS 64

static const int kFrac = 16;
static const int kOne  = 1 << kFrac;   // 1.0 in Q16.16 (65536) — fract.h::FractInvMass uses kOne/volume

// std430 FractFragment mirror (engine/sim/fract.h::FractFragment): 14 x 4-byte = 56 bytes (memcmp-able).
struct FractFragment {
    int  cx, cy, cz;                 // integer centroid
    int  minx, miny, minz;           // AABB min
    int  maxx, maxy, maxz;           // AABB max
    int  boundRadiusSq;              // max member squared-dist to centroid
    int  boundRadius;                // ISqrt32(boundRadiusSq)
    uint volume;                     // integer voxel count
    uint cellId;                     // source FR1 seed index
    int  invMass;                    // FractInvMass(volume) = kOne / volume (Q16.16)
};

struct FractFragParams {
    int4 cfg0;   // x=sampleCount, y=seedCount, z=nx, w=ny
    int4 cfg1;   // x=nz, y=fragCount, z=enabled, w=unused
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            fragmentToCell : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            fragStart      : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            fragSamples    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<FractFragment>   gFragments     : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<FractFragParams> gParams        : register(u4);

// ISqrt32 — VERBATIM fract.h::ISqrt32 (int32 binary digit-by-digit floor-sqrt; pure int32, NO int64).
int ISqrt32(int v) {
    if (v <= 0) return 0;
    int bit = 1 << 30;
    while (bit > v) bit >>= 2;
    int res = 0;
    while (bit != 0) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else { res >>= 1; }
        bit >>= 2;
    }
    return res;
}

[numthreads(HF_FRACT_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int nx = gParams[0].cfg0.z;
    int ny = gParams[0].cfg0.w;
    int fragCount = gParams[0].cfg1.y;

    uint f = gid.x;
    if ((int)f >= fragCount) return;

    uint cell  = fragmentToCell[f];
    uint begin = fragStart[cell];
    uint end   = fragStart[cell + 1u];
    uint count = end - begin;
    if (count == 0u) return;   // guard: a padding thread (f >= compact fragCount) over an empty cell

    // Pass A: centroid sum + AABB.
    int sx = 0, sy = 0, sz = 0;
    int minx = 0x7FFFFFFF, miny = 0x7FFFFFFF, minz = 0x7FFFFFFF;
    int maxx = -0x7FFFFFFF - 1, maxy = -0x7FFFFFFF - 1, maxz = -0x7FFFFFFF - 1;
    for (uint k = begin; k < end; ++k) {
        int idx = (int)fragSamples[k];
        int px =  idx % nx;
        int py = (idx / nx) % ny;
        int pz =  idx / (nx * ny);
        sx += px; sy += py; sz += pz;
        if (px < minx) minx = px; if (px > maxx) maxx = px;
        if (py < miny) miny = py; if (py > maxy) maxy = py;
        if (pz < minz) minz = pz; if (pz > maxz) maxz = pz;
    }
    int cx = sx / (int)count;   // truncating integer divide (deterministic)
    int cy = sy / (int)count;
    int cz = sz / (int)count;

    // Pass B: boundRadiusSq = max member squared-dist to centroid.
    int brSq = 0;
    for (uint k = begin; k < end; ++k) {
        int idx = (int)fragSamples[k];
        int px =  idx % nx;
        int py = (idx / nx) % ny;
        int pz =  idx / (nx * ny);
        int dx = px - cx, dy = py - cy, dz = pz - cz;
        int d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > brSq) brSq = d2;
    }

    FractFragment fr;
    fr.cx = cx; fr.cy = cy; fr.cz = cz;
    fr.minx = minx; fr.miny = miny; fr.minz = minz;
    fr.maxx = maxx; fr.maxy = maxy; fr.maxz = maxz;
    fr.boundRadiusSq = brSq;
    fr.boundRadius = ISqrt32(brSq);
    fr.volume = count;
    fr.cellId = cell;
    fr.invMass = (count > 0u) ? (int)((uint)kOne / count) : 0;
    gFragments[f] = fr;
}
