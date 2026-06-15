// SSAO pass (Slice Y). Fullscreen pass over the g-buffer (view-space normal in xyz, view-space linear
// depth in w). For each pixel it reconstructs the view-space position from linear depth + screen UV +
// the projection params (tanHalfFovY, aspect) — NO matrix inverse needed — orients a baked 16-sample
// hemisphere kernel to the pixel's normal via a TBN built from a tiled rotation-noise vector, samples
// neighbor depths, and counts occluders with a smooth range check. Outputs AO (1 = open, 0 = fully
// occluded) in R. The kernel + noise are BAKED CONSTANTS (no runtime RNG) so two runs are byte-equal.
[[vk::binding(0, 0)]] Texture2D    gGbuf : register(t0);   // view-space normal.xyz + linDepth.w
[[vk::binding(1, 0)]] SamplerState gSmp  : register(s0);

struct SsaoParams {
    float2 texel;       // 1/gbufferSize (gbuffer is full-res here)
    float  radius;      // sampling radius in view-space units
    float  bias;        // depth bias to avoid self-occlusion acne
    float  intensity;   // AO strength multiplier
    float  tanHalfFovY; // tan(0.5*fovY) for view-space reconstruction
    float  aspect;      // width/height
    float  pad;
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer SsaoPC { SsaoParams sp; };
#define HF_SP sp
#else
[[vk::push_constant]] struct { SsaoParams p; } pc;
#define HF_SP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// Baked deterministic hemisphere kernel (tangent space, +Z = surface normal axis). Clustered near the
// origin (accelerating scale) so most taps probe the near field. Generated CPU-side from a fixed
// integer hash; see docs/superpowers/specs/2026-06-14-ssao-design.md.
static const float3 kKernel[16] = {
    float3(0.100795f, 0.000000f, 0.228780f),
    float3(0.091955f, -0.033122f, 0.233282f),
    float3(0.081891f, -0.067790f, 0.239155f),
    float3(-0.079710f, -0.079066f, 0.252534f),
    float3(0.090877f, -0.108247f, 0.261072f),
    float3(0.148477f, -0.036402f, 0.284807f),
    float3(-0.079297f, -0.147176f, 0.313703f),
    float3(-0.123456f, 0.139877f, 0.346523f),
    float3(0.123145f, 0.051105f, 0.416689f),
    float3(-0.015629f, -0.160221f, 0.459947f),
    float3(0.024918f, -0.012231f, 0.542259f),
    float3(-0.178843f, 0.018104f, 0.577147f),
    float3(-0.233236f, 0.111806f, 0.620094f),
    float3(-0.204048f, 0.335162f, 0.633428f),
    float3(0.169120f, 0.242681f, 0.769312f),
    float3(-0.119215f, -0.147209f, 0.889227f),
};
// Baked 4x4 (16) rotation-noise vectors (unit xy). Tiled across the screen to rotate the kernel basis
// per-pixel, decorrelating the 16 taps so the blur can resolve them into smooth AO.
static const float2 kNoise[16] = {
    float2(0.670217f, 0.742166f),
    float2(0.799867f, -0.600178f),
    float2(-0.688025f, -0.725687f),
    float2(0.775070f, -0.631876f),
    float2(-0.194749f, 0.980853f),
    float2(-0.975357f, -0.220634f),
    float2(0.894866f, 0.446335f),
    float2(-0.419847f, -0.907595f),
    float2(0.754884f, 0.655858f),
    float2(-0.340564f, -0.940221f),
    float2(-0.492609f, -0.870251f),
    float2(-0.913368f, -0.407135f),
    float2(0.120550f, -0.992707f),
    float2(0.959111f, 0.283031f),
    float2(-0.953719f, -0.300701f),
    float2(-0.979557f, -0.201165f),
};

// Reconstruct view-space position for a UV + linear depth. The projection bakes a Y-flip
// (Mat4::Perspective m[5] = -1/t), and post.vert gives a V-down UV on Vulkan, so view-space Y is
// recovered with the matching sign so it agrees with the g-buffer's stored view-space normal.
// Y-flip sign that maps screen UV.y <-> view-space Y. On Vulkan the projection bakes a Y-flip
// (Mat4::Perspective m[5] = -1/t) and post.vert gives a V-down UV, so view-space +Y maps to a
// SMALLER UV.y (sign -1). On Metal the backend applies FlipProjY to the projection AND post.vert
// flips V; the two compositions land view-space +Y at a LARGER UV.y (sign +1). Reconstruct and
// project share this sign so they stay mutual inverses AND match the g-buffer rasterization.
#ifdef HF_MSL_GEN
static const float HF_YS = 1.0;
#else
static const float HF_YS = -1.0;
#endif

float3 ReconstructViewPos(float2 uv, float linDepth) {
    float2 ndc = uv * 2.0 - 1.0;
    float vx = ndc.x * (HF_SP.aspect * HF_SP.tanHalfFovY) * linDepth;
    float vy = HF_YS * ndc.y * (HF_SP.tanHalfFovY) * linDepth;
    float vz = -linDepth;                   // RH view space: -Z forward
    return float3(vx, vy, vz);
}

// Forward-project a view-space position to UV (inverse of ReconstructViewPos). Used to look up the
// stored depth at each kernel sample's screen location.
float2 ProjectToUV(float3 vp) {
    float invZ = 1.0 / max(-vp.z, 1e-4);
    float ndcx = vp.x / (HF_SP.aspect * HF_SP.tanHalfFovY) * invZ;
    float ndcy = HF_YS * vp.y / (HF_SP.tanHalfFovY) * invZ;
    return float2(ndcx, ndcy) * 0.5 + 0.5;
}

float4 main(PSInput i) : SV_Target {
    float4 g = gGbuf.Sample(gSmp, i.uv);
    float  linDepth = g.w;
    // Background / no geometry (cleared w == 0): fully unoccluded.
    if (linDepth <= 0.0001) return float4(1.0, 1.0, 1.0, 1.0);

    float3 P = ReconstructViewPos(i.uv, linDepth);
    float3 N = normalize(g.xyz);                 // view-space normal
    // Orient the hemisphere toward the camera. The g-buffer stores the geometric view-space normal,
    // which can face away from the camera for back-facing or non-uniformly-scaled geometry (e.g. the
    // wide ground plane), in which case a +N kernel would dig INTO the surface and over-occlude. For
    // a VISIBLE fragment the AO hemisphere must lie on the camera-facing side, so flip N to face the
    // viewer (view-space camera is at the origin looking down -Z; the view direction is -P, and N
    // should have a positive component along it).
    if (dot(N, -normalize(P)) < 0.0) N = -N;

    // Per-pixel rotation vector from the tiled 4x4 noise (integer pixel coords -> 0..15).
    int2 px = int2(i.uv / HF_SP.texel);
    int  ni = (px.x & 3) + ((px.y & 3) << 2);
    float3 randV = float3(kNoise[ni], 0.0);

    // Gram-Schmidt TBN: tangent = randV projected off N, then bitangent.
    float3 T = normalize(randV - N * dot(randV, N));
    float3 B = cross(N, T);
    float3x3 TBN = float3x3(T, B, N);            // rows = T,B,N: mul(k, TBN) = k.x*T + k.y*B + k.z*N

    float occlusion = 0.0;
    [unroll] for (int s = 0; s < 16; ++s) {
        // Orient the baked tangent-space kernel sample to the surface, scale by radius, offset P.
        float3 dir = mul(kKernel[s], TBN);
        float3 samplePos = P + dir * HF_SP.radius;
        // Project to UV and read the stored surface depth there.
        float2 sUV = ProjectToUV(samplePos);
        if (sUV.x < 0.0 || sUV.x > 1.0 || sUV.y < 0.0 || sUV.y > 1.0) continue;
        float storedDepth = gGbuf.Sample(gSmp, sUV).w;
        if (storedDepth <= 0.0001) continue;     // sample landed on background
        // The sample is OCCLUDED if the real surface at sUV is closer to the camera (smaller linDepth)
        // than the sample point, by more than the bias. Range check fades the contribution when the
        // occluder is far from P (so distant geometry behind doesn't darken).
        float sampleDepth = -samplePos.z;        // sample point's view-space linear depth
        // Range check: the occluder only counts if it is within `radius` of the shaded fragment, with
        // a smooth falloff (distant background behind the silhouette does not darken the surface).
        float rangeCheck = smoothstep(0.0, 1.0, HF_SP.radius / max(abs(linDepth - storedDepth), 1e-4));
        occlusion += ((storedDepth < sampleDepth - HF_SP.bias) ? 1.0 : 0.0) * rangeCheck;
    }

    float ao = 1.0 - (occlusion / 16.0) * HF_SP.intensity;
    ao = saturate(ao);
    return float4(ao, ao, ao, 1.0);
}
