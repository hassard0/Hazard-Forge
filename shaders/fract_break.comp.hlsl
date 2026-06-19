// Slice FR3 — Deterministic Rigid-Body Fracture/Destruction: the BONDED-CLUSTER BREAK pass (THE NEW
// PHYSICS — the 3rd slice of FLAGSHIP #14). Runs the int64 Jacobi load-diffusion break over the host-built
// bond graph VERBATIM from engine/sim/fract.h::ApplyImpactBreak: inject the impact load at one fragment,
// diffuse it through the INTACT bonds over K Jacobi iterations (read iteration-START per-fragment load,
// write a SEPARATE delta buffer, apply — the FL4/GR3 read-start/write-separate discipline, race-free), each
// bond accumulating the transmitted magnitude into loadAccum, then SEVER a bond iff loadAccum exceeds its
// strength-scaled threshold fxmul(kBreakThreshold, faceArea<<kFrac). The output per-bond {loadAccum,
// severed} is memcmp'd BIT-EXACT against the CPU ApplyImpactBreak (the make-or-break).
//
// INTEGER WIDTH: the load is Q16.16 and the transmit/threshold use int64 (fxmul). DXC compiles int64
// (Vulkan); glslc CANNOT parse int64 in HLSL -> this shader is VULKAN-SPIR-V-ONLY (NOT in
// metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --fract-break showcase runs the CPU
// fract::ApplyImpactBreak (byte-identical by construction — the FPX1/GR3/CL3 split). The bond-graph BUILD
// (BuildFractBonds) is pure int32 + host-built; the GPU-PROVEN pass is THIS int64 break.
//
// THREADING: the K-iter diffusion shares the per-fragment load[] state across iterations and the per-bond
// loadAccum is a monotone serial accumulation, so this is a SINGLE THREAD ([numthreads(1,1,1)], the
// cloth_solve.comp / fract_emit_scan.comp single-thread mirror): one thread walks the K iters over the
// fixed bond order, reading load-start into a local-ish scratch and writing a separate delta, exactly the
// CPU body. The break is small (M fragments, B bonds — tens), so single-thread is ample and the
// order-sensitive shared diffusion stays bit-exact to the sequential CPU reference.
//
// Buffers (storage, bound at compute bindings 0..3; Vulkan-only):
//   b0 gBonds   : the FractBondGpu array (40 bytes each) — READ fragA/fragB/faceArea, WRITE loadAccum.
//   b1 gSevered : one uint per bond (the 0/1 severed flag), WRITE.
//   b2 gLoad    : M int64 per-fragment load scratch (read/write), pre-zeroed by the host except the
//                 injected impact fragment. (A separate M int64 delta scratch is bound at b3.)
//   b3 gDelta   : M int64 per-fragment Jacobi delta scratch (the read-start/write-separate buffer).
//   b4 gParams  : the FR3 break params (fragmentCount, bondCount, K, impactFragment, impulse, threshold).
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_FR3_FRAC 16

// FractBondGpu std430 mirror: fragA,fragB,faceArea (uint,uint,int) + pad + midpoint (3 x int32 Q16.16) +
// pad + loadAccum (int64). 40 bytes (loadAccum at offset 32, 8-aligned); the host lays it out to match.
struct FractBondGpu {
    uint fragA;
    uint fragB;
    int  faceArea;
    int  _pad0;          // align the int64 loadAccum below to 8 bytes
    int  midx, midy, midz;   // Q16.16 midpoint (viz-only; not read by the break)
    int  _pad1;
    int64_t loadAccum;   // Q16.16 diffused load (WRITE)
};

// FR3 break params (std430). cfg0 {fragmentCount, bondCount, K, impactFragment}; cfg1 {impulse, threshold,
// flow, enabled} — impulse/threshold/flow are Q16.16.
struct FractBreakParams { int4 cfg0; int4 cfg1; };

[[vk::binding(0, 0)]] RWStructuredBuffer<FractBondGpu>      gBonds   : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>             gSevered : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<int64_t>          gLoad    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<int64_t>          gDelta   : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<FractBreakParams> gParams  : register(u4);

// fxmul — VERBATIM fpx.h::fxmul (int64 intermediate, arithmetic shift).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_FR3_FRAC); }

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // single-thread (the order-sensitive shared diffusion mirror)

    int M       = gParams[0].cfg0.x;
    int B       = gParams[0].cfg0.y;
    int K       = gParams[0].cfg0.z;
    int impactF = gParams[0].cfg0.w;
    int impulse   = gParams[0].cfg1.x;   // Q16.16
    int threshold = gParams[0].cfg1.y;   // Q16.16 (kBreakThreshold)
    int flow      = gParams[0].cfg1.z;   // Q16.16 (kFlow)
    int enabled   = gParams[0].cfg1.w;

    // Reset loadAccum + severed (host pre-zeroes too, but mirror the CPU which zeroes loadAccum).
    for (int i = 0; i < B; ++i) { gBonds[i].loadAccum = (int64_t)0; gSevered[i] = 0u; }
    if (enabled == 0 || M == 0 || B == 0) return;

    // (1) Inject the impact load at its fragment (all others start at 0).
    for (int f = 0; f < M; ++f) gLoad[f] = (int64_t)0;
    if (impactF >= 0 && impactF < M) gLoad[impactF] = (int64_t)impulse;

    int64_t kFlow = (int64_t)flow;

    // (2) K Jacobi diffusion iterations (read-start load / write-separate gDelta / apply).
    for (int it = 0; it < K; ++it) {
        for (int d = 0; d < M; ++d) gDelta[d] = (int64_t)0;
        for (int bi = 0; bi < B; ++bi) {
            uint a = gBonds[bi].fragA;
            uint b = gBonds[bi].fragB;
            int64_t la = gLoad[a];
            int64_t lb = gLoad[b];
            int64_t diff = la - lb;
            int64_t mag  = diff < (int64_t)0 ? -diff : diff;
            int64_t transmit = (kFlow * mag) >> HF_FR3_FRAC;   // Q16.16 fxmul(kFlow, mag)
            gBonds[bi].loadAccum = gBonds[bi].loadAccum + transmit;
            if (diff > (int64_t)0) { gDelta[a] -= transmit; gDelta[b] += transmit; }
            else if (diff < (int64_t)0) { gDelta[a] += transmit; gDelta[b] -= transmit; }
        }
        for (int f2 = 0; f2 < M; ++f2) gLoad[f2] = gLoad[f2] + gDelta[f2];
    }

    // (3) Sever iff loadAccum > fxmul(kBreakThreshold, faceArea<<kFrac).
    for (int bi2 = 0; bi2 < B; ++bi2) {
        int faceQ = gBonds[bi2].faceArea << HF_FR3_FRAC;
        int64_t thresh = (int64_t)fxmul(threshold, faceQ);
        gSevered[bi2] = (gBonds[bi2].loadAccum > thresh) ? 1u : 0u;
    }
}
