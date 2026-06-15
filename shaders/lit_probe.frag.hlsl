// Slice AK — reflection + irradiance PROBE lit fragment shader. Base lit PBR (same Cook-Torrance
// machinery as lit_pbr_ibl.frag), but the image-based lighting comes from a BAKED LOCAL cubemap GI
// probe instead of the global HDR environment: metals reflect the ACTUAL surrounding room walls and
// diffuse surfaces pick up color bleed from them.
//
// The probe is ONE RGBA16F atlas bound at the environment slot (gEnv/gEnvSmp) with two blocks:
//   * REFLECTION block: 3x2 grid of 512 tiles -> 1536x1024, at atlas origin.
//   * IRRADIANCE block: 3x2 grid of  64 tiles ->  192x128, below it (y starts at 1024px).
// To sample a direction `dir`: face = SelectFace(dir) (dominant abs axis, EXACT mirror of
// render::probe::SelectFace / point_shadow), project (probePos + dir) through faceVP[face] to a
// [0,1] face UV, then remap into that face's tile in the chosen block. Tile UVs are clamped to the
// tile interior (texel inset) so linear filtering never bleeds across tile borders.
//
// FrameData (own layout, < kFrameUboSize=1024; 624 bytes).
struct FrameData {
    float4x4 viewProj;     //   0
    float4   lightDir;     //  64  directional fill
    float4   lightColor;   //  80
    float4   viewPos;      //  96
    float4x4 faceVP[6];    // 112  6 probe-face view-projs (384B) -> ends 496
    float4   probePos;     // 496  xyz probe bake position
    float4   atlasParams;  // 512  x=reflTileU y=reflTileV z=irrTileU w=irrTileV  (tile size in atlas UV)
    float4   atlasParams2; // 528  x=irrBlockV0 (irradiance block V start) y=texelU z=texelV w=tilesPerRow
    float4   camFwd;       // 544
    float4   camRight;     // 560
    float4   camUp;        // 576
    float4   skyParams;    // 592  unused (kept for layout symmetry)
    float4   pad0;         // 608
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// (the directional shadow map slot exists in set 0 for layout parity but is unused here)
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
// Material set (set 1): base-color + tangent-space normal (2-texture set, like lit_point).
[[vk::binding(0, 1)]] Texture2D    gBase      : register(t0);
[[vk::binding(1, 1)]] SamplerState gBaseSmp   : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);
// Probe atlas (set 3, binding 11/12 -> Metal texture(11)/sampler(12)); bound via BindReflectionProbe.
[[vk::binding(11, 3)]] Texture2D    gProbe    : register(t11);
[[vk::binding(12, 3)]] SamplerState gProbeSmp : register(s11);
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
float hfGeometrySchlickGGX(float NoX, float k) { return NoX / (NoX * (1.0 - k) + k); }
float hfGeometrySmith(float NoV, float NoL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return hfGeometrySchlickGGX(NoV, k) * hfGeometrySchlickGGX(NoL, k);
}
float3 hfFresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (float3(1.0, 1.0, 1.0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
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
float3 SrgbToLinear(float3 c) { return pow(saturate(c), 2.2); }

// EXACT mirror of render::probe::SelectFace (== point_shadow::SelectFace): dominant abs axis -> face.
//   0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z.
int SelectFace(float3 dir) {
    float ax = abs(dir.x), ay = abs(dir.y), az = abs(dir.z);
    if (ax >= ay && ax >= az) return dir.x >= 0.0 ? 0 : 1;
    if (ay >= ax && ay >= az) return dir.y >= 0.0 ? 2 : 3;
    return dir.z >= 0.0 ? 4 : 5;
}

// Sample the probe atlas for a world direction. `block`==0 -> reflection block, 1 -> irradiance.
// Returns float3 color. Projects (probePos + dir) through the selected face's view-proj to a [0,1]
// face UV, remaps into that face's tile, clamps to the tile interior, samples.
float3 SampleProbe(float3 dir, int block) {
    int face = SelectFace(dir);
    float4 cp = mul(f.faceVP[face], float4(f.probePos.xyz + normalize(dir), 1.0));
    if (cp.w <= 0.0) return float3(0, 0, 0);
    float2 faceUV = (cp.xy / cp.w) * 0.5 + 0.5;
#ifdef HF_MSL_GEN
    faceUV.y = 1.0 - faceUV.y;
#endif
    faceUV = saturate(faceUV);

    int   tilesPerRow = (int)f.atlasParams2.w;     // 3
    int   col = face % tilesPerRow;
    int   row = face / tilesPerRow;
    float2 tileSize = (block == 0) ? f.atlasParams.xy : f.atlasParams.zw;
    float2 tileOrigin = float2((float)col, (float)row) * tileSize;
    if (block == 1) tileOrigin.y += f.atlasParams2.x;   // irradiance block V offset

    float2 texel = f.atlasParams2.yz;
    float2 atlasUV = tileOrigin + faceUV * tileSize;
    float2 tileMin = tileOrigin + texel;
    float2 tileMax = tileOrigin + tileSize - texel;
    atlasUV = clamp(atlasUV, tileMin, tileMax);
    return gProbe.SampleLevel(gProbeSmp, atlasUV, 0.0).rgb;
}

float4 main(PSInput i) : SV_Target {
    float3 Ng = normalize(i.wnormal);
    float3 T = normalize(i.wtangent - Ng * dot(Ng, i.wtangent));
    float3 B = cross(Ng, T);
    float3x3 TBN = float3x3(T, B, Ng);
    float3 nTS = gNormalMap.Sample(gNormalSmp, i.uv).xyz * 2.0 - 1.0;
    float3 N = normalize(mul(nTS, TBN));
    float3 V = normalize(f.viewPos.xyz - i.wpos);

    float3 baseSrgb = gBase.Sample(gBaseSmp, i.uv).rgb;
    // Albedo comes from the bound material texture ONLY (the colored wall / hero textures). Vertex
    // color is intentionally ignored so the shared cube mesh's per-face tints don't contaminate the
    // hero box / walls.
    float3 albedo = SrgbToLinear(baseSrgb);
    float  metallic  = saturate(i.material.x);
    float  roughness = clamp(i.material.y, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // Small ambient so unlit areas aren't pure black.
    float3 rgb = albedo * 0.03;

    // Directional fill (Cook-Torrance, no shadow — the probe carries the local lighting story).
    {
        float3 L = normalize(-f.lightDir.xyz);
        rgb += hfCookTorrance(N, V, L, f.lightColor.rgb, albedo, metallic, roughness, F0);
    }

    // --- PROBE image-based lighting. Specular = reflection atlas at reflect(-V,N); a rougher
    //     surface blends toward the (diffuse-convolved) irradiance so the reflection softens.
    //     Diffuse ambient = irradiance atlas at N. This is LOCAL GI: it reflects/bleeds the actual
    //     room walls baked into the atlas, not a global sky. ---
    {
        float3 R = reflect(-V, N);
        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);

        float3 sharpRefl = SampleProbe(R, 0);     // reflection block
        float3 irradiance = SampleProbe(N, 1);    // irradiance block (diffuse-convolved)
        // Roughen the reflection by fading the sharp mirror toward the convolved irradiance.
        float3 prefiltered = lerp(sharpRefl, irradiance, roughness);
        float3 iblSpecular = prefiltered * F;
        rgb += iblSpecular;

        // Diffuse color bleed. Scaled so neutral surfaces stay mostly neutral while still picking up
        // a clear warm(red)/cool(green) tint from the wall they face — the local-GI proof.
        rgb += (1.0 - metallic) * albedo * irradiance * 0.55;
    }

    return float4(rgb, 1.0);
}
