// Slice FR1 — Deterministic Rigid-Body Fracture/Destruction Slice 1: the per-sample NEAREST-SEED CELL
// CLASSIFICATION compute pass (the BEACHHEAD of FLAGSHIP #14). ONE thread per lattice SAMPLE
// (sample < sampleCount). Each thread decomposes its flat sample id -> (x,y,z), runs the nearest-seed
// loop copied VERBATIM from engine/sim/fract.h::NearestSeed over the M host-supplied seeds (gSeeds, b0),
// and writes gCells[sample] = bestId (b1). The per-sample cell is computed independently from the SAME
// read-only seed array, so the write is ORDER-INDEPENDENT — race-free, NO atomics, bit-identical
// GPU==CPU + cross-backend (the MC1/VT1 set-write argument applied to nearest-seed classification).
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it runs
// the SAME pure-int32 squared-distance compares (dx*dx+dy*dy+dz*dz, STRICTLY-less lowest-index tie-break)
// fract.h::NearestSeed runs. A divergence here vs the header is exactly what the host's GPU==CPU memcmp
// catches.
//
// THE INT32 CRUX (MSL-NATIVE on BOTH backends): the lattice coords are small (each axis < ~1024) so the
// max squared distance 3·1023² ≈ 3.1M fits int32 with headroom — NO int64. A pure integer SSBO compute
// (like mc_classify.comp / cgf_gf_count.comp — no int64, no atomics, no integer texture.read) -> the
// default MSL gen suffices (NO --msl-version 20200) and fract_classify.comp is in hf_gen_msl, a TRUE GPU
// pass on Vulkan AND Metal.
//
// classifyEnabled push/param flag: 0 -> every thread writes 0 (the disabled-path no-op; gCells stays the
// cleared all-zero upload, byte-identical).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gSeeds  : the int32 seed lattice coords (Seed{x,y,z,_} std430), READ.
//   b1 gCells  : one uint per lattice sample (the nearest-seed index), WRITE.
//   b2 gParams : { nx, ny, nz, _ } + { seedCount, classifyEnabled, sampleCount, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations
// (same as mc_classify.comp / cgf_gf_count.comp), not backend CODE symbols.

#define HF_FRACT_THREADS 64

// A seed lattice coord (std430). Mirrors the C++ upload struct (FractSeed padded to int4).
//   coord : x=seed.x, y=seed.y, z=seed.z, w=unused
struct Seed {
    int4 coord;
};

// Params (std430). Mirrors the C++ upload struct.
//   dims  : x=nx, y=ny, z=nz, w=unused (SAMPLE counts per axis)
//   cfg   : x=seedCount, y=classifyEnabled, z=sampleCount, w=unused
struct Params {
    int4 dims;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Seed>   gSeeds  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gCells  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams : register(u2);

// The nearest-seed loop — VERBATIM engine/sim/fract.h::NearestSeed. Pure int32 squared distance, STRICTLY
// less (<, not <=) so the lowest-index seed at the minimum distance wins (the deterministic tie-break).
uint NearestSeed(int x, int y, int z, int seedCount) {
    uint bestId = 0u;
    int bestD2 = 0x7FFFFFFF;  // INT32_MAX
    for (int k = 0; k < seedCount; ++k) {
        int dx = x - gSeeds[(uint)k].coord.x;
        int dy = y - gSeeds[(uint)k].coord.y;
        int dz = z - gSeeds[(uint)k].coord.z;
        int d2 = dx * dx + dy * dy + dz * dz;  // pure int32 squared distance
        if (d2 < bestD2) { bestD2 = d2; bestId = (uint)k; }
    }
    return bestId;
}

[numthreads(HF_FRACT_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int nx = gParams[0].dims.x;
    int ny = gParams[0].dims.y;
    int seedCount       = gParams[0].cfg.x;
    int classifyEnabled = gParams[0].cfg.y;
    int sampleCount     = gParams[0].cfg.z;

    uint sample = gid.x;
    if ((int)sample >= sampleCount) return;

    // Disabled -> write 0 (gCells stays the cleared all-zero upload; the byte-identical no-op proof).
    if (classifyEnabled == 0) { gCells[sample] = 0u; return; }

    // Decompose sample -> (x,y,z). sampleIndex = (z*ny + y)*nx + x (VERBATIM engine/sim/fract.h).
    int x =  (int)sample % nx;
    int y = ((int)sample / nx) % ny;
    int z =  (int)sample / (nx * ny);

    gCells[sample] = NearestSeed(x, y, z, seedCount);
}
