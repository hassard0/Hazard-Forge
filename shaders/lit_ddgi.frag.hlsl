// Slice DN — DDGI Slice 5: the GI COMPOSITE lit fragment (the VISIBLE payoff). A SIBLING COPY of
// lit.frag.hlsl with ONE addition: a single-bounce DDGI indirect-diffuse term added as the LAST
// contribution, so a Cornell box shows colored indirect light bleed (the red wall bleeds reddish onto the
// floor, the green wall bleeds green). Used ONLY by the --ddgi-shot showcase. lit.frag.hlsl + ALL its
// goldens stay BYTE-IDENTICAL (this is a NEW shader, like lit_clustered/lit_csm/lit_contactshadow isolate
// their variants).
//
// THE GI COMPOSITE (the DDGI visible result): per pixel, the 8 nearest irradiance probes are trilinearly
// blended (NearestProbes) into a single 3rd-order real-SH irradiance record, reconstructed toward the
// surface normal (SHEvaluate with the cosine-lobe convolution), and added as indirect diffuse modulated by
// albedo + (1-metallic) + a giStrength scalar. ALL the math (NearestProbes + InterpolateSH + SHEvaluate)
// is copied VERBATIM from engine/render/probe_gi.h + probe_sh.h (the shader-copied-shared-math rule), with
// mad for every accumulation (the DH cross-backend FP discipline: a plain a+b*c contracts to fma on
// Metal's fast-math but not on Vulkan/DXC, so the indirect term must mad to match the CPU mirror exactly).
//
// THE giStrength==0 NO-OP PROOF: the indirect term is `rgb += indirect * albedo * gGi.giStrength`. With
// giStrength==0 this is a LITERAL +0.0 (a multiply by the exact float 0), so the accumulated rgb is
// BYTE-IDENTICAL to the no-GI render through the SAME pipeline. The indirect term is added LAST (after the
// env block) so the float accumulation ORDER matches a no-GI render that stops at the env block. THE
// probeCount=0 FALLBACK: dimX==0 -> valid==false -> InterpolateIrradianceSH returns {0,0,0} -> the indirect
// term is +0.0 too (== the giStrength=0 frame).
//
// THE ProbeSH SSBO BINDING (NO new RHI): the ProbeSH[] rides in a fragment-stage storage buffer bound via
// the EXISTING usesLightClusters / BindLightClusters path (set 3 = three fragment storage buffers, as
// lit_clustered does); DN needs ONE buffer, bound via dummies: BindLightClusters(*probeShBuf, *dummy,
// *dummy). [[vk::binding(13,3)]] -> spirv-cross maps it to Metal fragment buffer 13. The ProbeGrid params
// (origin/dims/spacing) + giStrength ride in the showcase's OWN FrameData UBO (the lit_clustered.frag
// precedent — its own FrameData layout with spare fields).

// DDGI FrameData layout (own struct; the --ddgi-shot showcase fills the per-frame UBO with THIS layout).
// lit.vert reads viewProj at offset 0 (the standard lit vertex contract).
struct FrameData {
    float4x4 viewProj;     //   0  (lit.vert reads this at offset 0)
    float4   lightDir;     //  64  directional sun
    float4   lightColor;   //  80
    float4   viewPos;      //  96  world-space camera position (for V)
    float4x4 lightViewProj;// 112  directional shadow projection
    float4   giOrigin;     // 176  x=originX, y=originY, z=originZ, w=spacing
    float4   giDims;       // 192  x=dimX, y=dimY, z=dimZ, w=giStrength
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Shadow map lives in the per-frame set (set 0): binding 1 = depth image, binding 2 = sampler.
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);

// The per-probe SH irradiance records (std430, 108 bytes each: 9 coeffs x 3 channels). Bound to the
// FRAGMENT stage via the cluster set (binding 13/space3, like lit_clustered's gClusters) -> spirv-cross
// --msl-decoration-binding maps it to Metal fragment buffer slot 13. Mirrors render::probesh::ProbeSH.
struct ProbeSH { float coeff[9][3]; };
[[vk::binding(13, 3)]] StructuredBuffer<ProbeSH> gProbeSH : register(t13, space3);

struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    [[vk::location(4)]] nointerpolation float2 material : TEXCOORD1; // x=metallic, y=roughness
    [[vk::location(5)]] float3 wtangent : TANGENT;
};
static const float HF_PI = 3.14159265358979323846;

// --- 3rd-order real-SH constants (VERBATIM probesh:: constants). ---
static const float kY00 = 0.282094791773878140;   // 0.5 * sqrt(1/pi)
static const float kY1  = 0.488602511902919920;   // 0.5 * sqrt(3/pi)
static const float kY2a = 1.092548430592079200;   // 0.5 * sqrt(15/pi)
static const float kY2b = 0.315391565252520000;   // 0.25 * sqrt(5/pi)
static const float kY2c = 0.546274215296039600;   // 0.25 * sqrt(15/pi)
static const float kA0  = 3.14159265358979324;     // pi
static const float kA1  = 2.09439510239319549;     // 2*pi/3
static const float kA2  = 0.785398163397448310;    // pi/4

// GGX / Trowbridge-Reitz normal distribution. alpha = roughness^2.
float hfDistributionGGX(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float d  = (NoH * NoH) * (a2 - 1.0) + 1.0;
    return a2 / max(HF_PI * d * d, 1e-7);
}
float hfGeometrySchlickGGX(float NoX, float k) {
    return NoX / (NoX * (1.0 - k) + k);
}
float hfGeometrySmith(float NoV, float NoL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return hfGeometrySchlickGGX(NoV, k) * hfGeometrySchlickGGX(NoL, k);
}
float3 hfFresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (float3(1.0, 1.0, 1.0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}
float3 SkyColor(float3 dir) {
    float3 d = normalize(dir);
    float  h       = saturate(d.y * 0.5 + 0.5);
    float3 zenith  = float3(0.18, 0.30, 0.62);
    float3 horizon = float3(0.65, 0.72, 0.82);
    float3 sky     = lerp(horizon, zenith, pow(h, 0.8));
    float3 ground = float3(0.12, 0.11, 0.10);
    if (d.y < 0.0) {
        float g = saturate(-d.y * 2.0);
        sky = lerp(sky, ground, g);
    }
    float3 sunDir = normalize(-f.lightDir.xyz);
    float  s = pow(max(dot(d, sunDir), 0.0), 256.0);
    sky += float3(1.0, 0.95, 0.8) * s * 2.0;
    return sky;
}
float3 hfCookTorrance(float3 N, float3 V, float3 L, float3 radiance,
                      float3 albedo, float metallic, float roughness, float3 F0) {
    float3 H   = normalize(L + V);
    float  NoV = max(dot(N, V), 1e-4);
    float  NoL = max(dot(N, L), 0.0);
    float  NoH = max(dot(N, H), 0.0);
    float  VoH = max(dot(V, H), 0.0);
    float  alpha = roughness * roughness;

    float  D = hfDistributionGGX(NoH, alpha);
    float  G = hfGeometrySmith(NoV, NoL, roughness);
    float3 F = hfFresnelSchlick(VoH, F0);

    float3 spec = (D * G) * F / max(4.0 * NoV * NoL, 1e-4);
    float3 kd = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);
    float3 diff = kd * albedo / HF_PI;
    return (diff + spec) * radiance * NoL;
}

// --- The 9 real SH basis functions at a UNIT direction (VERBATIM probesh::SHBasis9). ---
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

// --- The DDGI indirect-diffuse term: trilinearly-blended probe SH irradiance toward N (VERBATIM
// probegi::NearestProbes + probesh::InterpolateSH + probesh::SHEvaluate, mad for every accumulation). ---
// Returns {0,0,0} on the disabled path (spacing<=0 / any dim<=0 -> the probeCount=0 fallback), so the
// indirect term is a literal +0.0 there too.
float3 InterpolateIrradianceSH(float3 worldPos, float3 N) {
    int dimX = (int)f.giDims.x;
    int dimY = (int)f.giDims.y;
    int dimZ = (int)f.giDims.z;
    float ox = f.giOrigin.x, oy = f.giOrigin.y, oz = f.giOrigin.z;
    float spacing = f.giOrigin.w;

    // Zeroed SH accumulator (the blended SH).
    float acc[9][3];
    [unroll] for (int i = 0; i < 9; ++i) { acc[i][0] = 0.0; acc[i][1] = 0.0; acc[i][2] = 0.0; }

    bool valid = (spacing > 0.0) && (dimX > 0) && (dimY > 0) && (dimZ > 0);
    if (valid) {
        // Per-axis FLOOR base + fractional position; a dim==1 axis collapses to base 0, frac 0.
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

        // The 8 corners: blend each corner's ProbeSH with the per-corner weight via mad (matching the
        // header's std::fma). The corner coordinate clamps to [0,dim-1]; w = mad(wx*wy, wz, 0).
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
    // !valid -> acc stays zero (the probeCount=0 fallback == probesh::InterpolateSH's zero SH).

    // --- SHEvaluate (VERBATIM probesh::SHEvaluate, cosineLobe=true): reconstruct the irradiance toward N. ---
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
    // Geometric (interpolated) world normal.
    float3 Ng = normalize(i.wnormal);

    float3 T = normalize(i.wtangent - Ng * dot(Ng, i.wtangent));
    float3 B = cross(Ng, T);                       // handedness baked into T at authoring time
    float3x3 TBN = float3x3(T, B, Ng);             // rows = T,B,N
    float3 nTS = gNormalMap.Sample(gNormalSmp, i.uv).xyz * 2.0 - 1.0;
    float3 N = normalize(mul(nTS, TBN));
    float3 V = normalize(f.viewPos.xyz - i.wpos);
    float3 tex = gTex.Sample(gSmp, i.uv).rgb * i.color;

    float3 albedo   = tex;
    float  metallic = saturate(i.material.x);
    float  roughness = clamp(i.material.y, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // --- Directional shadow: project world pos into the light's clip space, compare depth. ---
    float shadow = 1.0;
    {
        float4 lp = mul(f.lightViewProj, float4(i.wpos, 1.0));
        float3 proj = lp.xyz / lp.w;
        float2 smUV = proj.xy * 0.5 + 0.5;
#ifdef HF_MSL_GEN
        smUV.y = 1.0 - smUV.y;
#endif
        float  curDepth = proj.z;
        if (smUV.x >= 0.0 && smUV.x <= 1.0 && smUV.y >= 0.0 && smUV.y <= 1.0 &&
            curDepth >= 0.0 && curDepth <= 1.0) {
            float bias = 0.0025;
            float s = 0.0;
            float texel = 1.0 / 2048.0;
            [unroll] for (int sx = -1; sx <= 1; ++sx)
            [unroll] for (int sy = -1; sy <= 1; ++sy) {
                float d = gShadow.Sample(gShadowSmp, smUV + float2(sx, sy) * texel).r;
                s += (curDepth - bias > d) ? 0.0 : 1.0;
            }
            shadow = s / 9.0;
        }
    }

    // Small ambient term (unshadowed) so unlit areas aren't pure black.
    float3 rgb = albedo * 0.03;

    // Directional light: Cook-Torrance, with the directional radiance attenuated by the PCF shadow factor.
    {
        float3 L = normalize(-f.lightDir.xyz);
        float3 radiance = f.lightColor.rgb * shadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }

    // --- Procedural image-based lighting (environment reflection), same as lit.frag. ---
    {
        float3 R = reflect(-V, N);
        float  NoV = max(dot(N, V), 0.0);
        float3 Fr  = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);
        float3 envColor = SkyColor(R);
        envColor = lerp(envColor, SkyColor(float3(0.0, 1.0, 0.0)), roughness * 0.7);
        float3 iblSpecular = envColor * Fr;
        rgb += iblSpecular;
        rgb += (1.0 - metallic) * albedo * SkyColor(N) * 0.15;
    }

    // --- DDGI indirect diffuse (the GI composite, added LAST). Trilinearly-blended probe SH irradiance
    // toward N, modulated by albedo + the non-metallic fraction. giStrength==0 -> + 0.0 EXACTLY (a literal-
    // zero multiply) -> byte-identical to the no-GI render through this same pipeline. ---
    {
        float3 indirect = InterpolateIrradianceSH(i.wpos, N) * (1.0 - metallic);
        rgb += indirect * albedo * f.giDims.w;
    }

    return float4(rgb, 1.0);
}
