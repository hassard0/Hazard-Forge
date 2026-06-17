// Slice DR — DDGI MULTI-BOUNCE capture-with-GI fragment shader. A SIBLING COPY of probe_bake.frag.hlsl
// (the DI probe-radiance capture fragment that writes the directly-lit room color into the cube face)
// with ONE addition: the DN single-bounce DDGI indirect-diffuse term added as the LAST contribution, so
// the bounce-1 capture sees the 1st-bounce GI on the surfaces. SH-encoding this bounce-1 radiance yields
// SH1 (the 2nd-bounce irradiance) -> the composite gets a brighter, more filled-in 2nd bounce (toward
// Lumen infinite-bounce). Used ONLY by the bounce-1 capture in --ddgimb-shot. probe_bake.frag.hlsl +
// lit_ddgi.frag.hlsl + lit.frag.hlsl + ALL their goldens stay BYTE-IDENTICAL (this is a NEW shader, like
// lit_ddgi/lit_ddgi_occ isolate their variants).
//
// THE INDIRECT TERM (VERBATIM the DN lit_ddgi.frag InterpolateIrradianceSH): the 8 nearest probes' SH0
// irradiance is trilinearly blended toward the captured surface normal (NearestProbes + InterpolateSH +
// SHEvaluate copied VERBATIM, mad for every accumulation — the DH cross-backend FP discipline) and added
// as indirect*albedo*(1-metallic)*giStrength. SH0 (the 1st-bounce direct capture) rides in a fragment-
// stage storage buffer bound via the EXISTING usesLightClusters/BindLightClusters path (binding 13/space3,
// single SSBO via dummies — NO new RHI). The ProbeGrid params + giStrength ride in the capture's OWN
// FrameData UBO (the SAME 208-byte DN lit_ddgi FrameData layout so InterpolateIrradianceSH is verbatim).
//
// THE DISABLED PATH (bounceCount=1): --ddgimb-shot SKIPS this shader entirely at bounceCount=1 (it never
// runs the bounce-1 capture), so the composite uses SH0 -> BYTE-IDENTICAL to the DN single-bounce render.
// This shader contributes only when bounceCount>=2. A probeCount=0 grid -> InterpolateIrradianceSH returns
// {0,0,0} (the DN zero-SH fallback) so even then the indirect term is a literal +0.0.

// DDGI FrameData layout — the SAME 208-byte struct lit_ddgi.frag uses (so the indirect term is verbatim).
// Only giOrigin/giDims + lightDir/lightColor are read here; viewProj/viewPos/lightViewProj ride unused.
struct FrameData {
    float4x4 viewProj;     //   0
    float4   lightDir;     //  64
    float4   lightColor;   //  80
    float4   viewPos;      //  96
    float4x4 lightViewProj;// 112
    float4   giOrigin;     // 176  x=originX, y=originY, z=originZ, w=spacing
    float4   giDims;       // 192  x=dimX, y=dimY, z=dimZ, w=giStrength
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };

// The wall albedo texture rides the material set (set 1 b0/b0), bound via BindMaterial — the SAME slot
// lit_ddgi.frag's gTex uses, so spirv-cross --msl-decoration-binding lands it on Metal fragment
// texture(0)/sampler(0) and there is no set-0 binding-0 collision with the FrameData UBO.
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);

// The per-probe SH0 irradiance records (std430, 108 bytes each: 9 coeffs x 3 channels). Bound to the
// FRAGMENT stage via the cluster set (binding 13/space3, like lit_ddgi's gProbeSH) -> spirv-cross
// --msl-decoration-binding maps it to Metal fragment buffer slot 13. Mirrors render::probesh::ProbeSH.
struct ProbeSH { float coeff[9][3]; };
[[vk::binding(13, 3)]] StructuredBuffer<ProbeSH> gProbeSH : register(t13, space3);

struct PSInput {
    float4 clip : SV_Position;
    [[vk::location(0)]] float2 uv      : TEXCOORD0;
    [[vk::location(1)]] float3 wnormal : NORMAL;
    [[vk::location(2)]] float3 wpos    : POSITION0;
};

float3 SrgbToLinear(float3 c) { return pow(saturate(c), 2.2); }

// --- 3rd-order real-SH constants (VERBATIM probesh:: constants — the DN copy). ---
static const float kY00 = 0.282094791773878140;   // 0.5 * sqrt(1/pi)
static const float kY1  = 0.488602511902919920;   // 0.5 * sqrt(3/pi)
static const float kY2a = 1.092548430592079200;   // 0.5 * sqrt(15/pi)
static const float kY2b = 0.315391565252520000;   // 0.25 * sqrt(5/pi)
static const float kY2c = 0.546274215296039600;   // 0.25 * sqrt(15/pi)
static const float kA0  = 3.14159265358979324;     // pi
static const float kA1  = 2.09439510239319549;     // 2*pi/3
static const float kA2  = 0.785398163397448310;    // pi/4

// --- The 9 real SH basis functions at a UNIT direction (VERBATIM probesh::SHBasis9 / the DN copy). ---
void SHBasis9(float3 dir, out float o[9]) {
    float x = dir.x, y = dir.y, z = dir.z;
    o[0] = kY00;
    o[1] = kY1 * y;
    o[2] = kY1 * z;
    o[3] = kY1 * x;
    o[4] = kY2a * (x * y);
    o[5] = kY2a * (y * z);
    o[6] = kY2b * (3.0 * z * z - 1.0);
    o[7] = kY2a * (x * z);
    o[8] = kY2c * (x * x - y * y);
}

// --- The DDGI indirect-diffuse term: trilinearly-blended probe SH irradiance toward N (VERBATIM the DN
// lit_ddgi.frag InterpolateIrradianceSH — probegi::NearestProbes + probesh::InterpolateSH +
// probesh::SHEvaluate, mad for every accumulation). Returns {0,0,0} on the disabled path (spacing<=0 /
// any dim<=0 -> the probeCount=0 fallback), so the indirect term is a literal +0.0 there too. ---
float3 InterpolateIrradianceSH(float3 worldPos, float3 N) {
    int dimX = (int)f.giDims.x;
    int dimY = (int)f.giDims.y;
    int dimZ = (int)f.giDims.z;
    float ox = f.giOrigin.x, oy = f.giOrigin.y, oz = f.giOrigin.z;
    float spacing = f.giOrigin.w;

    float acc[9][3];
    [unroll] for (int i = 0; i < 9; ++i) { acc[i][0] = 0.0; acc[i][1] = 0.0; acc[i][2] = 0.0; }

    bool valid = (spacing > 0.0) && (dimX > 0) && (dimY > 0) && (dimZ > 0);
    if (valid) {
        int bx, by, bz;
        float fx, fy, fz;
        if (dimX <= 1) { bx = 0; fx = 0.0; }
        else {
            float g = (worldPos.x - ox) / spacing;
            float fb = floor(g);
            fb = clamp(fb, 0.0, (float)(dimX - 2));
            bx = (int)fb;
            fx = clamp(g - fb, 0.0, 1.0);
        }
        if (dimY <= 1) { by = 0; fy = 0.0; }
        else {
            float g = (worldPos.y - oy) / spacing;
            float fb = floor(g);
            fb = clamp(fb, 0.0, (float)(dimY - 2));
            by = (int)fb;
            fy = clamp(g - fb, 0.0, 1.0);
        }
        if (dimZ <= 1) { bz = 0; fz = 0.0; }
        else {
            float g = (worldPos.z - oz) / spacing;
            float fb = floor(g);
            fb = clamp(fb, 0.0, (float)(dimZ - 2));
            bz = (int)fb;
            fz = clamp(g - fb, 0.0, 1.0);
        }

        [unroll] for (int c = 0; c < 8; ++c) {
            int sx = (c & 1), sy = ((c >> 1) & 1), sz = ((c >> 2) & 1);
            int cx = min(bx + sx, dimX - 1);
            int cy = min(by + sy, dimY - 1);
            int cz = min(bz + sz, dimZ - 1);
            int flat = cx + cy * dimX + cz * (dimX * dimY);
            float wx = sx ? fx : (1.0 - fx);
            float wy = sy ? fy : (1.0 - fy);
            float wz = sz ? fz : (1.0 - fz);
            float wc = mad(wx * wy, wz, 0.0);
            [unroll] for (int j = 0; j < 9; ++j) {
                acc[j][0] = mad(wc, gProbeSH[flat].coeff[j][0], acc[j][0]);
                acc[j][1] = mad(wc, gProbeSH[flat].coeff[j][1], acc[j][1]);
                acc[j][2] = mad(wc, gProbeSH[flat].coeff[j][2], acc[j][2]);
            }
        }
    }

    float basis[9];
    SHBasis9(normalize(N), basis);
    float bw[9];
    bw[0] = basis[0] * kA0;
    bw[1] = basis[1] * kA1; bw[2] = basis[2] * kA1; bw[3] = basis[3] * kA1;
    bw[4] = basis[4] * kA2; bw[5] = basis[5] * kA2; bw[6] = basis[6] * kA2;
    bw[7] = basis[7] * kA2; bw[8] = basis[8] * kA2;
    float3 r = float3(0.0, 0.0, 0.0);
    [unroll] for (int k = 0; k < 9; ++k) {
        r.x += acc[k][0] * bw[k];
        r.y += acc[k][1] * bw[k];
        r.z += acc[k][2] * bw[k];
    }
    return r;
}

float4 main(PSInput i) : SV_Target {
    // --- The DI direct-capture term (VERBATIM probe_bake.frag.hlsl). ---
    float3 albedo = SrgbToLinear(gTex.Sample(gSmp, i.uv).rgb);
    float3 N = normalize(i.wnormal);
    // Fixed fill light from above-front so walls are gently shaded (a flat fill keeps color dominant).
    float3 L = normalize(float3(-0.3, 0.9, 0.4));
    float ndl = saturate(dot(N, L)) * 0.6 + 0.4;   // 0.4 ambient + up to 0.6 diffuse
    float3 radiance = albedo * ndl;

    // --- The 2nd-bounce feed (the DR addition, added LAST): add the 1st-bounce GI (SH0 trilinear
    // irradiance) to the captured surface radiance. The captured walls are dielectric (metallic=0) so
    // (1-metallic)==1. giStrength==0 / probeCount=0 -> a literal +0.0 -> identical to the DI capture. ---
    float metallic = 0.0;
    float3 indirect = InterpolateIrradianceSH(i.wpos, N) * (1.0 - metallic);
    radiance += indirect * albedo * f.giDims.w;

    return float4(radiance, 1.0);
}
