// Slice WH1 — Warm-Started Hull Contacts: THE HULL CONTACT FEATURE ID compute pass (the int32 MSL-native
// BEACHHEAD of FLAGSHIP #26: WARM-STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, hf::sim::warmhull).
// ONE THREAD PER CONTACT POINT (per-contact INDEPENDENT — each thread reads its contact's
// (bodyA, bodyB, refIsA, refFace, incTag) input and writes its OWN HullContactKey slot + hash; race-free, NO
// atomics) runs the order-normalized integer key + the fixed hash mix, copying engine/sim/warmhull.h::
// MakeHullContactKey + HullContactKeyHash's bodies VERBATIM (the SAME bodyA<bodyB swap, the SAME refIsA/role
// fold into refFaceId's high bit, the SAME (bodyA<<20)^(bodyB<<8)^(refFaceId<<4)^incVertId packing + the SAME
// avalanche xor-shift) so the GPU HullContactKey[]+hash[] is byte-identical to the CPU reference -> the host
// GPU==CPU memcmp (Vulkan) + the cross-vendor strict-zero (Metal) catch any divergence.
//
// THE PURE-INT32 WIN (the STRONGEST proof tier, persist.h's PS1 split): the KEY is PURE int32 — compares + bit
// shifts + xors, NO Q16.16 products, NO int64, NO float. So UNLIKE the int64 hull MATH (hull_manifold /
// hull_step_hardened are VULKAN-SPIR-V-ONLY because glslc cannot parse int64_t in HLSL), this shader is
// MSL-NATIVE — it IS in the metal_headless hf_gen_msl list (the persist_key / fract_emit_count tier), a TRUE
// GPU pass on BOTH backends, with strict zero-differing-pixel cross-vendor. The MANIFOLD that PRODUCED the
// (refFace, incTag) indices is int64, but it runs HOST-SIDE; this shader only PACKS the already-computed int32
// indices into keys (persist_key.comp's exact role). The Metal --wh1-keys runs THIS shader (NOT a CPU ref).
//
// keyEnabled=0 -> write a cleared HullContactKey + hash 0 for every contact (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gContacts : one HullKeyInput per contact { bodyA, bodyB, refIsA, refFace, incTag } (the RAW caller
//                  order — bodyA may be > bodyB; the shader order-normalizes), READ.
//   b1 gKeys     : one HullContactKey per contact { bodyA, bodyB, refFaceId, incVertId } (NORMALIZED), WRITE.
//   b2 gHashes   : one uint per contact = HullContactKeyHash(key), WRITE.
//   b3 gParams   : { contactCount, keyEnabled, _, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_WARMHULL_THREADS 64

// The high-bit tag constants (== engine/sim/warmhull.h). PURE int32.
static const uint kRefFaceRoleBit = 0x80000000u;   // high bit of refFaceId: 1 == the STORED bodyB owns the ref face

// The RAW hull-contact feature input (the caller's per-contact-point tuple; bodyA/bodyB NOT yet normalized).
// refIsA = "the RAW-order body A owns the reference face" (0/1); refFace = the reference face index (small);
// incTag = the incident source-feature provenance tag (an original LOCAL vertex idx OR a packed
// (refEdge,incEdge) crossing under the intersect high bit).
struct HullKeyInput {
    uint bodyA;
    uint bodyB;
    uint refIsA;
    uint refFace;
    uint incTag;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// std430 HullContactKey mirror (engine/sim/warmhull.h::HullContactKey): 4 x uint32 (16 bytes) — bodyA<bodyB
// NORMALIZED, refFaceId (role bit | face index), incVertId. The host packs warmhull into THIS form for memcmp.
struct HullContactKey {
    uint bodyA;
    uint bodyB;
    uint refFaceId;
    uint incVertId;
};

struct WarmhullParams {
    int4 cfg;   // x=contactCount, y=keyEnabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<HullKeyInput>   gContacts : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<HullContactKey> gKeys     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>           gHashes   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<WarmhullParams> gParams   : register(u3);

// VERBATIM engine/sim/warmhull.h::MakeHullContactKey — order-normalize bodyA<=bodyB (swap the stored fields),
// fold the stored ref-face owner into refFaceId's high bit, store incTag. PURE INT32 (compares + shifts + ors).
HullContactKey MakeHullContactKey(uint bodyAIdx, uint bodyBIdx, uint refIsA, uint refFace, uint incTag) {
    HullContactKey k;
    bool swapped;
    if (bodyAIdx <= bodyBIdx) { k.bodyA = bodyAIdx; k.bodyB = bodyBIdx; swapped = false; }
    else                      { k.bodyA = bodyBIdx; k.bodyB = bodyAIdx; swapped = true; }
    // storedOwnerIsB == (refIsA == swapped). Fold that single bit into refFaceId's high bit; low bits = index.
    bool refIsABool = (refIsA != 0u);
    bool storedOwnerIsB = (refIsABool == swapped);
    k.refFaceId = (storedOwnerIsB ? kRefFaceRoleBit : 0u) | (refFace & 0x7FFFFFFFu);
    k.incVertId = incTag;
    return k;
}

// VERBATIM engine/sim/warmhull.h::HullContactKeyHash — the fixed (bodyA<<20)^(bodyB<<8)^(refFaceId<<4)^incVertId
// packing + the avalanche xor-shift. PURE INT32 — only shifts + xor + add, NO products.
uint HullContactKeyHash(HullContactKey k) {
    uint h = (k.bodyA << 20) ^ (k.bodyB << 8) ^ (k.refFaceId << 4) ^ k.incVertId;
    h ^= h >> 15;
    h += (h << 7);
    h ^= h >> 11;
    return h;
}

[numthreads(HF_WARMHULL_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int contactCount = gParams[0].cfg.x;
    int keyEnabled   = gParams[0].cfg.y;
    int idx = (int)gid.x;
    if (idx >= contactCount) return;

    // Disabled -> write a cleared key + hash 0 (the byte-identical no-op).
    if (keyEnabled == 0) {
        HullContactKey z;
        z.bodyA = 0u; z.bodyB = 0u; z.refFaceId = 0u; z.incVertId = 0u;
        gKeys[idx] = z;
        gHashes[idx] = 0u;
        return;
    }

    HullKeyInput c = gContacts[idx];
    HullContactKey k = MakeHullContactKey(c.bodyA, c.bodyB, c.refIsA, c.refFace, c.incTag);
    gKeys[idx] = k;
    gHashes[idx] = HullContactKeyHash(k);
}
