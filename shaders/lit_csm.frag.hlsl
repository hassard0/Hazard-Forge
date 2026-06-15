// Cascaded-shadow lit fragment shader (Slice AD). Identical lighting to lit.frag, but the directional
// shadow is sampled from an N-cascade ATLAS instead of a single map:
//   * pick a cascade by the fragment's view-space (camera-forward) distance vs the split distances,
//   * project wpos by that cascade's lightViewProj,
//   * remap the [0,1] cascade UV into the cascade's atlas TILE,
//   * 3x3 PCF clamped to the tile bounds (so the kernel never bleeds into a neighbour tile).
//
// CSM FrameData layout (own struct; fits in kFrameUboSize=1024). The --csm-shot showcase fills the
// per-frame UBO with THIS layout; other showcases keep their own. See docs/.../csm-design.md.
struct FrameData {
    float4x4 viewProj;        //   0
    float4   lightDir;        //  64
    float4   lightColor;      //  80
    float4   viewPos;         //  96
    float4   csmSplits;       // 112  x,y,z,w = view-space far distance of cascade 0..3
    float4x4 cascadeVP[4];    // 128  per-cascade lightViewProj (256B) -> ends 384
    float4   camFwd;          // 384
    float4   camRight;        // 400
    float4   camUp;           // 416
    float4   skyParams;       // 432
    float4   csmAtlas;        // 448  x=tilesPerRow, y=tileUVScale(1/tilesPerRow), z=numCascades
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);

// Per-cascade debug tint — OFF in the committed golden. Define HF_CSM_DEBUG_TINT to colorize
// cascade bands (red/green/blue/yellow) for visual verification of the split boundaries.
// #define HF_CSM_DEBUG_TINT

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

// Sample one cascade's atlas tile with 3x3 PCF, clamped to the tile so the kernel never bleeds into
// a neighbour. cascadeUV is the [0,1] UV within the cascade; (col,row) selects the tile.
float SampleCascadePCF(float2 cascadeUV, float curDepth, int col, int row, float tileScale) {
    float2 tileOrigin = float2((float)col, (float)row) * tileScale;
    float2 atlasUV = tileOrigin + cascadeUV * tileScale;
    // Tile bounds in atlas UV (inset by half a texel so taps stay inside the tile).
    float  atlasTexel = 1.0 / 4096.0;
    float2 tileMin = tileOrigin + atlasTexel;
    float2 tileMax = tileOrigin + tileScale - atlasTexel;
    float  bias = 0.0025;
    float  s = 0.0;
    [unroll] for (int sx = -1; sx <= 1; ++sx)
    [unroll] for (int sy = -1; sy <= 1; ++sy) {
        float2 tap = clamp(atlasUV + float2(sx, sy) * atlasTexel, tileMin, tileMax);
        float d = gShadow.Sample(gShadowSmp, tap).r;
        s += (curDepth - bias > d) ? 0.0 : 1.0;
    }
    return s / 9.0;
}

float4 main(PSInput i) : SV_Target {
    float3 Ng = normalize(i.wnormal);
    float3 T = normalize(i.wtangent - Ng * dot(Ng, i.wtangent));
    float3 B = cross(Ng, T);
    float3x3 TBN = float3x3(T, B, Ng);
    float3 nTS = gNormalMap.Sample(gNormalSmp, i.uv).xyz * 2.0 - 1.0;
    float3 N = normalize(mul(nTS, TBN));
    float3 V = normalize(f.viewPos.xyz - i.wpos);
    float3 tex = gTex.Sample(gSmp, i.uv).rgb * i.color;

    float3 albedo   = tex;
    float  metallic = saturate(i.material.x);
    float  roughness = clamp(i.material.y, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // --- CASCADE SELECTION: view-space distance along the camera forward. ---
    int   numCascades = (int)f.csmAtlas.z;
    int   tilesPerRow = (int)f.csmAtlas.x;
    float tileScale   = f.csmAtlas.y;
    float viewDepth = dot(i.wpos - f.viewPos.xyz, normalize(f.camFwd.xyz));
    int cascade = numCascades - 1;
    [unroll] for (int ci = 0; ci < 4; ++ci) {
        if (ci < numCascades && viewDepth <= f.csmSplits[ci]) { cascade = ci; break; }
    }

    float shadow = 1.0;
    {
        float4 lp = mul(f.cascadeVP[cascade], float4(i.wpos, 1.0));
        float3 proj = lp.xyz / lp.w;
        float2 smUV = proj.xy * 0.5 + 0.5;
        // Same Metal-only texture-origin flip as lit.frag: the cascade VP has the Vulkan Y-flip baked
        // in; on Metal it's flipped CPU-side, so V must be flipped to hit the matching texel. RENDER
        // and SAMPLE both derive from the same CPU-side matrix, so they stay self-consistent.
#ifdef HF_MSL_GEN
        smUV.y = 1.0 - smUV.y;
#endif
        float curDepth = proj.z;
        if (smUV.x >= 0.0 && smUV.x <= 1.0 && smUV.y >= 0.0 && smUV.y <= 1.0 &&
            curDepth >= 0.0 && curDepth <= 1.0) {
            int col = cascade % tilesPerRow;
            int row = cascade / tilesPerRow;
            shadow = SampleCascadePCF(smUV, curDepth, col, row, tileScale);
        }
    }

    float3 rgb = albedo * 0.03;
    {
        float3 L = normalize(-f.lightDir.xyz);
        float3 radiance = f.lightColor.rgb * shadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }
    {
        float3 R = reflect(-V, N);
        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);
        float3 envColor = SkyColor(R);
        envColor = lerp(envColor, SkyColor(float3(0.0, 1.0, 0.0)), roughness * 0.7);
        float3 iblSpecular = envColor * F;
        rgb += iblSpecular;
        rgb += (1.0 - metallic) * albedo * SkyColor(N) * 0.15;
    }

#ifdef HF_CSM_DEBUG_TINT
    float3 tint[4] = { float3(1,0.3,0.3), float3(0.3,1,0.3), float3(0.3,0.5,1), float3(1,1,0.3) };
    rgb = lerp(rgb, tint[cascade], 0.35);
#endif

    return float4(rgb, 1.0);
}
