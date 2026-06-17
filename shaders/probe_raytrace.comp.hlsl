// Slice DH — DDGI Beachhead: the PROBE-GRID RAY-TRACE compute pass. ONE thread per probe
// (dimX*dimY*dimZ), dispatched as enough HF_PROBE_THREADS workgroups to cover the grid. Each thread
// decodes its probe's (px,py,pz) from the flat index, reconstructs the probe's WORLD position
// (origin + (px,py,pz)*spacing), and traces kRaysPerProbe deterministic Fibonacci-sphere rays against
// the scene's VIEW-SPACE depth field — a FLAT float[depthW*depthH] SSBO (gDepth) read back from the
// rendered G-buffer's .w channel (NOT a sampled depth texture, mirroring froxel_inject which never
// samples a depth texture in compute). Per ray it records the world-space HIT position + the hit
// distance (or kRayMiss) into the flat ProbeRayHit[probeCount*kRaysPerProbe] SSBO at gRayHits[p*16+r].
//
// THE BIT-EXACT GPU==CPU PROOF: FibonacciSphere + ViewToScreenUV/ReconstructViewPos + TraceRayToDepth
// are copied VERBATIM from engine/render/probe_gi.h (+ ssr.h), so this pass and the CPU reference
// (probegi::TraceRayToDepth over the SAME flat depthField, same nearest index math) produce a
// BIT-EXACT ray-hit buffer (memcmp == 0). THE probeCount=0 NO-OP PROOF: dimX==0 -> probeCount()==0 ->
// the host dispatches 0 groups -> this main() never runs -> the ray-hit SSBO is untouched (== the
// cleared upload, all kRayMiss).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal they land at buffer(kCsStorage+0..2)):
//   b0 gRayHits : the flat ProbeRayHit[probeCount*16] ray-hit buffer, WRITE (xyz hit pos, w dist/miss).
//   b1 gParams  : the probe grid (origin/dims/spacing) + camera (view/tanHalfFovY/aspect/yFlip) +
//                 march (maxDist/steps/thickness) + the depth field dims (depthW/H), READ.
//   b2 gDepth   : the FLAT view-linear depth field float[depthW*depthH] (the G-buffer .w readback), READ.
//                 Indexed NEAREST: gDepth[(uint)(v*depthH)*depthW + (uint)(u*depthW)].
// (NO sampled textures — same style as froxel_inject/cluster_assign. The depth is a flat SSBO so the GPU
//  and CPU index it identically.)
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL are the HF_MSL_GEN
// generation-path guard + [[vk::binding]] decorations (same as cluster_assign / froxel_inject), not
// backend CODE symbols. spirv-cross maps the SPIR-V bindings to Metal's flat compute buffer indices so
// the SAME HLSL feeds Vulkan (DXC) and Metal (glslang->spirv-cross): bit-identical math.

#define HF_PROBE_THREADS  64
#define HF_RAYS_PER_PROBE 16

static const float HF_GOLDEN_ANGLE = 2.39996322972865332;
static const float HF_RAY_MISS     = 1e30;

// One ray hit (std430, 16 bytes): mirrors render::probegi::ProbeRayHit. xyz = world hit position, w =
// hit distance or HF_RAY_MISS.
struct ProbeRayHit {
    float4 hitPosDist;
};

// Probe params (std430). Mirrors the C++ upload struct (ProbeParamsCPU): grid + camera + march + depth
// dims. All packed into float4/uint4 + a column-major view matrix.
struct ProbeParams {
    float4   originSpacing;   // xyz = grid origin (world), w = spacing
    uint4    dims;            // x=dimX, y=dimY, z=dimZ, w=probeCount
    float4   camera;          // x=tanHalfFovY, y=aspect, z=yFlip, w=unused
    float4   march;           // x=maxDist, y=steps, z=thickness, w=unused
    uint4    depthDims;       // x=depthW, y=depthH, zw unused
    float4x4 view;            // world -> view, column-major (matches engine Mat4)
    // The kRaysPerProbe Fibonacci-sphere directions, PRECOMPUTED on the host (from the verbatim
    // render::probegi::FibonacciSphere below) and uploaded as exact float32 bits. The shader reads these
    // for the trace so the per-ray direction is BIT-IDENTICAL to the CPU reference's dir — cos/sin/sqrt
    // are NOT bit-identical between the CPU libm and the GPU, so computing FibonacciSphere on BOTH sides
    // would diverge by a few ULP and break the bit-exact stored hit position. The FibonacciSphere helper
    // is kept verbatim (the canonical mirror of the header) but is NOT the trace's dir source. xyz = dir.
    float4   rayDirs[HF_RAYS_PER_PROBE];
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ProbeRayHit> gRayHits : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ProbeParams> gParams  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float>       gDepth   : register(u2);

// FibonacciSphere(i,N): the i-th of N golden-spiral directions on the unit sphere. Copied VERBATIM from
// render::probegi::FibonacciSphere (incl. the N==1 -> north-pole single-sample convention).
float3 FibonacciSphere(int i, int N) {
    if (N < 1) N = 1;
    if (i < 0) i = 0;
    if (i >= N) i = N - 1;
    if (N == 1) return float3(0.0, 0.0, 1.0);
    float z   = 1.0 - (2.0 * (float)i + 1.0) / (float)N;
    float r   = sqrt(max(0.0, 1.0 - z * z));
    float phi = (float)i * HF_GOLDEN_ANGLE;
    return float3(r * cos(phi), r * sin(phi), z);
}

// ViewToScreenUV: forward-project a VIEW-SPACE position to a screen UV in [0,1] + view-linear depth.
// Copied VERBATIM from render::ssr::ViewToScreenUV (re-exported by probegi).
float3 ViewToScreenUV(float3 viewPos, float tanHalfFovY, float aspect, float yFlip) {
    float invZ = 1.0 / max(-viewPos.z, 1e-4);
    float ndcx = viewPos.x / (aspect * tanHalfFovY) * invZ;
    float ndcy = yFlip * viewPos.y / tanHalfFovY * invZ;
    return float3(ndcx * 0.5 + 0.5, ndcy * 0.5 + 0.5, -viewPos.z);
}

// ReconstructViewPos: the inverse of ViewToScreenUV (copied VERBATIM from render::ssr::ReconstructViewPos
// for shader<->header parity; not used by the march but kept verbatim per the slice's seam discipline).
float3 ReconstructViewPos(float u, float v, float linDepth, float tanHalfFovY, float aspect, float yFlip) {
    float ndcx = u * 2.0 - 1.0;
    float ndcy = v * 2.0 - 1.0;
    float vx = ndcx * (aspect * tanHalfFovY) * linDepth;
    float vy = yFlip * ndcy * (tanHalfFovY) * linDepth;
    float vz = -linDepth;
    return float3(vx, vy, vz);
}

// Sample the FLAT view-linear depth field at screen UV (u,v), NEAREST. Mirrors the CPU DepthFn the
// showcase reference uses: index gDepth[(uint)(v*H)*W + (uint)(u*W)], clamped to the field.
float SampleDepth(float u, float v, uint depthW, uint depthH) {
    int ix = (int)(u * (float)depthW);
    int iy = (int)(v * (float)depthH);
    if (ix < 0) ix = 0; if (ix > (int)depthW - 1) ix = (int)depthW - 1;
    if (iy < 0) iy = 0; if (iy > (int)depthH - 1) iy = (int)depthH - 1;
    return gDepth[(uint)iy * depthW + (uint)ix];
}

// TraceRayToDepth: march the ray in WORLD space, projecting each marched point to a screen UV +
// view-linear depth, comparing against the sampled scene depth (the SSR-style thickness band). Copied
// VERBATIM from render::probegi::TraceRayToDepth (the march body), with sampleDepth resolved to gDepth.
bool TraceRayToDepth(float3 originWorld, float3 dirWorld, float maxDist, int steps, float thickness,
                     float4x4 view, float tanHalfFovY, float aspect, float yFlip,
                     uint depthW, uint depthH, out ProbeRayHit outHit) {
    if (steps < 1) steps = 1;
    float dt = maxDist / (float)steps;
    for (int k = 1; k <= steps; ++k) {
        float t = dt * (float)k;
        // `precise` so DXC does NOT contract origin + dir*t into a single FMA — the CPU reference does the
        // separate multiply-then-add, so the stored hit position must be the SAME rounding to be bit-exact.
        precise float3 pWorld = originWorld + dirWorld * t;
        float3 pView = mul(view, float4(pWorld, 1.0)).xyz;   // world -> view (column-major)
        float3 uvd = ViewToScreenUV(pView, tanHalfFovY, aspect, yFlip);
        if (uvd.x < 0.0 || uvd.x > 1.0 || uvd.y < 0.0 || uvd.y > 1.0) continue;   // off-screen
        float surf = SampleDepth(uvd.x, uvd.y, depthW, depthH);
        if (uvd.z >= surf - thickness && uvd.z <= surf + thickness) {
            outHit.hitPosDist = float4(pWorld, t);
            return true;
        }
    }
    outHit.hitPosDist = float4(0.0, 0.0, 0.0, HF_RAY_MISS);
    return false;
}

[numthreads(HF_PROBE_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint dimX = gParams[0].dims.x;
    uint dimY = gParams[0].dims.y;
    uint dimZ = gParams[0].dims.z;
    uint probeCount = gParams[0].dims.w;   // == dimX*dimY*dimZ (host-computed)

    uint p = gid.x;
    if (p >= probeCount) return;

    // Decode (px,py,pz) from the flat index: idx = px + py*dimX + pz*(dimX*dimY).
    uint pz  = p / (dimX * dimY);
    uint rem = p % (dimX * dimY);
    uint py  = rem / dimX;
    uint px  = rem % dimX;

    float3 origin = gParams[0].originSpacing.xyz;
    float  spacing = gParams[0].originSpacing.w;
    float3 probeWorld = origin + float3((float)px, (float)py, (float)pz) * spacing;

    float  tanHalfFovY = gParams[0].camera.x;
    float  aspect      = gParams[0].camera.y;
    float  yFlip       = gParams[0].camera.z;
    float  maxDist     = gParams[0].march.x;
    int    steps       = (int)gParams[0].march.y;
    float  thickness   = gParams[0].march.z;
    float4x4 view      = gParams[0].view;
    uint   depthW      = gParams[0].depthDims.x;
    uint   depthH      = gParams[0].depthDims.y;

    [loop] for (int r = 0; r < HF_RAYS_PER_PROBE; ++r) {
        // The PRECOMPUTED Fibonacci dir (uploaded as exact bits; the verbatim FibonacciSphere(r,16) above
        // is its canonical mirror but cos/sin/sqrt diverge by ULPs across CPU/GPU, so the trace reads the
        // host-computed dir to keep the stored hit position BIT-EXACT to the CPU reference).
        float3 dir = gParams[0].rayDirs[r].xyz;
        ProbeRayHit hit;
        TraceRayToDepth(probeWorld, dir, maxDist, steps, thickness, view, tanHalfFovY, aspect, yFlip,
                        depthW, depthH, hit);
        gRayHits[p * HF_RAYS_PER_PROBE + r] = hit;
    }
}
