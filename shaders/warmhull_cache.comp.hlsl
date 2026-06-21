// Slice WH2 — Warm-Started Hull Contacts: THE KEYED MANIFOLD + PERSISTENT CACHE compute pass (the 2nd slice of
// FLAGSHIP #26: WARM-STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, hf::sim::warmhull). ONE THREAD PER
// HULL PAIR (per-pair INDEPENDENT — each thread reads its pair's HOST-BUILT keyed manifold (the int64-derived
// convex::ContactManifold positions/depths/normal + the WH1 HullContactKey per point, all computed host-side by
// manifold::HullContactMulti + warmhull::BuildKeyedHullManifold) then MATCHES the persistent cache (a small SSBO
// scanned in FIXED order) so a matched point INHERITS its prior accumulated normal impulse + writes its OWN
// gManifolds slot; race-free, NO atomics). The cache match copies engine/sim/warmhull.h::MatchHullCache VERBATIM
// (the SAME fixed scan/match order, the SAME HullContactKeysEqual field compare) so the GPU inherited-impulse
// keyed manifold is byte-identical to the CPU warmhull::BuildKeyedHullManifold + MatchHullCache -> the host
// GPU==CPU memcmp catches any divergence.
//
// INTEGER WIDTH (the int64 reality, the MF3/PS2 lesson): the MANIFOLD this caches is built HOST-SIDE by the
// int64 manifold::HullContactMulti (GJK/EPA + the Sutherland-Hodgman clip — int64 fxmul/fxdiv/FxDot). DXC -spirv
// compiles int64; glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so even though THIS
// shader's match body is pure int32 the SLICE caches int64-derived data and this shader is VULKAN-SPIR-V-ONLY
// (in the Vulkan compile list, NOT in the Metal hf_gen_msl list — UNLIKE WH1's warmhull_key.comp which IS
// MSL-native because the KEY alone is pure int32). The Metal --wh2-cache showcase runs the CPU
// warmhull::BuildKeyedHullManifold + MatchHullCache over the same pairs -> byte-identical to this GPU result BY
// CONSTRUCTION (the persist_cache.comp convention), while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// cacheEnabled=0 -> write a cleared KeyedHullManifoldGpu (count=0) for every pair (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..3):
//   b0 gInputs    : the host-built keyed manifold per pair (count + 4 ManifoldPoint {pos.xyz, depth} + normal.xyz
//                   + 4 HullContactKey), READ. The positions/depths/normal/keys are the int64-derived host result.
//   b1 gCache     : the persistent cache — CachedHullContactGpu {HullContactKey key; int normalImpulse}, READ.
//   b2 gManifolds : the KeyedHullManifoldGpu array (count + 4 ManifoldPoint + normal + 4 key + 4 normalImpulse)
//                   per pair, WRITE.
//   b3 gParams    : { pairCount, cacheEnabled, cacheCount, _ }, READ.

// std430 HullContactKey mirror (engine/sim/warmhull.h::HullContactKey): 4 x uint32 (16 bytes).
struct HullContactKey {
    uint bodyA;
    uint bodyB;
    uint refFaceId;
    uint incVertId;
};

// std430 ManifoldPoint mirror (a convex::ContactManifold point packed: pos.xyz + depth): 4 x int32 (16 bytes).
struct ManifoldPoint {
    int px, py, pz;
    int depth;
};

// std430 input keyed manifold (the host-built convex::ContactManifold + the parallel keys): count (uint) + 4
// ManifoldPoint (64) + normal.xyz (12) + pad (4) + 4 HullContactKey (64) = 4 + 64 + 16 + 64 = 148 bytes.
struct KeyedHullInputGpu {
    uint count;
    ManifoldPoint pts[4];
    int nx, ny, nz;
    int _pad0;
    HullContactKey keys[4];
};

// std430 CachedHullContact mirror (engine/sim/warmhull.h::CachedHullContact): HullContactKey (4xu32) + 1 x int32
// impulse = 5 x int32 (20 bytes).
struct CachedHullContactGpu {
    HullContactKey key;
    int normalImpulse;
};

// std430 output keyed manifold (the matched result, packed for the memcmp): count (uint) + 4 ManifoldPoint (64)
// + normal.xyz (12) + pad (4) + 4 HullContactKey (64) + 4 normalImpulse (16) = 4 + 64 + 16 + 64 + 16 = 164 bytes.
struct KeyedHullManifoldGpu {
    uint count;
    ManifoldPoint pts[4];
    int nx, ny, nz;
    int _pad0;
    HullContactKey keys[4];
    int normalImpulse[4];
};

struct WarmhullCacheParams {
    int4 cfg;   // x=pairCount, y=cacheEnabled, z=cacheCount, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<KeyedHullInputGpu>     gInputs    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<CachedHullContactGpu>  gCache     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<KeyedHullManifoldGpu>  gManifolds : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<WarmhullCacheParams>   gParams    : register(u3);

// VERBATIM engine/sim/warmhull.h::HullContactKeysEqual — field-by-field equality (the cache match predicate).
bool HullContactKeysEqual(HullContactKey a, HullContactKey b) {
    return a.bodyA == b.bodyA && a.bodyB == b.bodyB
        && a.refFaceId == b.refFaceId && a.incVertId == b.incVertId;
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int pairCount    = gParams[0].cfg.x;
    int cacheEnabled = gParams[0].cfg.y;
    int cacheCount   = gParams[0].cfg.z;
    int idx = (int)gid.x;
    if (idx >= pairCount) return;

    // Cleared KeyedHullManifoldGpu (count=0, every field zero) — the disabled no-op + the empty default.
    KeyedHullManifoldGpu mz;
    mz.count = 0u;
    {
        ManifoldPoint pz; pz.px = 0; pz.py = 0; pz.pz = 0; pz.depth = 0;
        HullContactKey kz; kz.bodyA = 0u; kz.bodyB = 0u; kz.refFaceId = 0u; kz.incVertId = 0u;
        mz.pts[0] = pz; mz.pts[1] = pz; mz.pts[2] = pz; mz.pts[3] = pz;
        mz.nx = 0; mz.ny = 0; mz.nz = 0; mz._pad0 = 0;
        mz.keys[0] = kz; mz.keys[1] = kz; mz.keys[2] = kz; mz.keys[3] = kz;
        mz.normalImpulse[0] = 0; mz.normalImpulse[1] = 0; mz.normalImpulse[2] = 0; mz.normalImpulse[3] = 0;
    }

    if (cacheEnabled == 0) { gManifolds[idx] = mz; return; }

    KeyedHullInputGpu inp = gInputs[idx];

    // Copy the host-built manifold (positions/depths/normal/keys come from the int64 host narrowphase).
    KeyedHullManifoldGpu km = mz;
    km.count = inp.count;
    km.nx = inp.nx; km.ny = inp.ny; km.nz = inp.nz;
    for (uint i = 0u; i < 4u; ++i) {
        km.pts[i] = inp.pts[i];
        km.keys[i] = inp.keys[i];
        km.normalImpulse[i] = 0;   // cold-start (BuildKeyedHullManifold contract)
    }

    // ===== WH2: the cache MATCH (VERBATIM warmhull.h::MatchHullCache) — FIXED-order scan, first match wins =====
    for (uint p = 0u; p < inp.count; ++p) {
        for (int ce = 0; ce < cacheCount; ++ce) {
            if (HullContactKeysEqual(gCache[ce].key, inp.keys[p])) {
                km.normalImpulse[p] = gCache[ce].normalImpulse;
                break;   // the FIRST matching entry wins (fixed scan order)
            }
        }
    }

    gManifolds[idx] = km;
}
