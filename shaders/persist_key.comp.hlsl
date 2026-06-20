// Slice PS1 — Deterministic Persistent Contacts: THE CONTACT FEATURE ID compute pass (the BEACHHEAD of
// FLAGSHIP #21: DETERMINISTIC WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, hf::sim::persist). ONE THREAD
// PER CONTACT POINT (per-contact INDEPENDENT — each thread reads its contact's (bodyA, bodyB, axisIndex,
// pointIndex) input and writes its OWN ContactKey slot + hash; race-free, NO atomics) runs the order-
// normalized integer key + the fixed hash mix, copying engine/sim/persist.h::MakeContactKey +
// ContactKeyHash's bodies VERBATIM (the SAME bodyA<bodyB swap, the SAME (bodyA<<20)^(bodyB<<8)^(axisIndex<<4)^
// featureIndex packing + the SAME avalanche xor-shift) so the GPU ContactKey[]+hash[] is byte-identical to
// the CPU reference -> the host GPU==CPU memcmp (Vulkan) + the cross-vendor strict-zero (Metal) catch any
// divergence.
//
// THE PURE-INT32 WIN (the STRONGEST proof tier): the key is PURE int32 — compares + bit shifts + xors, NO
// Q16.16 products, NO int64, NO float. So UNLIKE the int64 contact MATH (convex_*/fric_* are VULKAN-SPIR-V-
// ONLY because glslc cannot parse int64_t in HLSL), this shader is MSL-NATIVE — it IS in the metal_headless
// hf_gen_msl list, a TRUE GPU pass on BOTH backends (the fract_emit_count / boids_cell_count tier), with
// strict zero-differing-pixel cross-vendor. The Metal --persist-key runs THIS shader (NOT a CPU reference).
//
// keyEnabled=0 -> write a cleared ContactKey + hash 0 for every contact (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gContacts : one ContactInput per contact { bodyA, bodyB, axisIndex, featureIndex } (the RAW caller
//                  order — bodyA may be > bodyB; the shader order-normalizes), READ.
//   b1 gKeys     : one ContactKey per contact { bodyA, bodyB, axisIndex, featureIndex } (NORMALIZED), WRITE.
//   b2 gHashes   : one uint per contact = ContactKeyHash(key), WRITE.
//   b3 gParams   : { contactCount, keyEnabled, _, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_PERSIST_THREADS 64

// The RAW contact-feature input (the caller's per-contact-point tuple; bodyA/bodyB NOT yet normalized).
struct ContactInput {
    uint bodyA;
    uint bodyB;
    uint axisIndex;
    uint featureIndex;
};

// std430 ContactKey mirror (engine/sim/persist.h::ContactKey): 4 x uint32 (16 bytes) — bodyA<bodyB
// NORMALIZED, axisIndex, featureIndex. The host packs convex/persist into THIS form for the memcmp.
struct ContactKey {
    uint bodyA;
    uint bodyB;
    uint axisIndex;
    uint featureIndex;
};

struct PersistParams {
    int4 cfg;   // x=contactCount, y=keyEnabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ContactInput> gContacts : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ContactKey>   gKeys     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>         gHashes   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<PersistParams> gParams  : register(u3);

// VERBATIM engine/sim/persist.h::MakeContactKey — order-normalize bodyA<=bodyB (swap the stored fields only),
// axisIndex/featureIndex stored as-is. PURE INT32 (a compare + a swap + a copy).
ContactKey MakeContactKey(uint bodyAIdx, uint bodyBIdx, uint axisIndex, uint featureIndex) {
    ContactKey k;
    if (bodyAIdx <= bodyBIdx) { k.bodyA = bodyAIdx; k.bodyB = bodyBIdx; }
    else                      { k.bodyA = bodyBIdx; k.bodyB = bodyAIdx; }   // swap -> bodyA < bodyB
    k.axisIndex    = axisIndex;
    k.featureIndex = featureIndex;
    return k;
}

// VERBATIM engine/sim/persist.h::ContactKeyHash — the fixed (bodyA<<20)^(bodyB<<8)^(axisIndex<<4)^featureIndex
// packing + the avalanche xor-shift. PURE INT32 — only shifts + xor + add, NO products.
uint ContactKeyHash(ContactKey k) {
    uint h = (k.bodyA << 20) ^ (k.bodyB << 8) ^ (k.axisIndex << 4) ^ k.featureIndex;
    h ^= h >> 15;
    h += (h << 7);
    h ^= h >> 11;
    return h;
}

[numthreads(HF_PERSIST_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int contactCount = gParams[0].cfg.x;
    int keyEnabled   = gParams[0].cfg.y;
    int idx = (int)gid.x;
    if (idx >= contactCount) return;

    // Disabled -> write a cleared key + hash 0 (the byte-identical no-op).
    if (keyEnabled == 0) {
        ContactKey z;
        z.bodyA = 0u; z.bodyB = 0u; z.axisIndex = 0u; z.featureIndex = 0u;
        gKeys[idx] = z;
        gHashes[idx] = 0u;
        return;
    }

    ContactInput c = gContacts[idx];
    ContactKey k = MakeContactKey(c.bodyA, c.bodyB, c.axisIndex, c.featureIndex);
    gKeys[idx] = k;
    gHashes[idx] = ContactKeyHash(k);
}
