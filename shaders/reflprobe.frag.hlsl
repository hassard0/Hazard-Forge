// Slice DA — Box-Projected Cubemap Reflections (local reflection probe) fragment shader. A NEW
// reflective-surface variant of lit_probe.frag: same base lit PBR (Cook-Torrance) + same probe-atlas
// cubemap sampling (SampleProbe — REUSES the EXISTING reflection cubemap the engine binds via
// BindReflectionProbe), but the SPECULAR reflection direction is PARALLAX-CORRECTED to the local probe
// BOX before sampling. The reflected room walls/floor line up with the actual room geometry (the
// hallmark of a local reflection probe) instead of appearing infinitely distant.
//
// The box-projection math (reflprobe::RayBoxExitT + reflprobe::BoxProject) is copied VERBATIM from
// engine/render/reflection_probe.h, so tests/reflection_probe_test.cpp exercises the EXACT correction
// this shader runs. THE parallaxStrength=0 NO-OP PROOF: with parallaxStrength == 0 BoxProject returns
// normalize(R) EXACTLY (the early-out below), so the box-projected reflection is BYTE-IDENTICAL to the
// standard infinite-cubemap reflection (the same SampleProbe(R,0) the plain probe shader does). The
// showcase asserts this internally (SHA). HLSL->SPIR-V->MSL via the existing toolchain makes it
// bit-identical cross-backend.
//
// This is a NEW shader behind the --reflprobe-shot showcase flag; the existing lit_probe / IBL / lit
// paths + their goldens stay BYTE-IDENTICAL.
//
// FrameData: the SAME 624-byte layout as lit_probe.frag (so the showcase reuses ProbeFrameData), but
// the previously-unused skyParams + pad0 carry the probe BOX: skyParams.xyz = boxMin,
// skyParams.w = parallaxStrength, pad0.xyz = boxMax. The probe box CENTER is probePos.xyz (the probe
// was baked there, exactly like lit_probe's SampleProbe origin).
struct FrameData {
    float4x4 viewProj;     //   0
    float4   lightDir;     //  64  directional fill
    float4   lightColor;   //  80
    float4   viewPos;      //  96
    float4x4 faceVP[6];    // 112  6 probe-face view-projs (384B) -> ends 496
    float4   probePos;     // 496  xyz probe bake position (== box CENTER)
    float4   atlasParams;  // 512  x=reflTileU y=reflTileV z=irrTileU w=irrTileV
    float4   atlasParams2; // 528  x=irrBlockV0 y=texelU z=texelV w=tilesPerRow
    float4   camFwd;       // 544
    float4   camRight;     // 560
    float4   camUp;        // 576
    float4   skyParams;    // 592  xyz = box MIN, w = parallaxStrength
    float4   pad0;         // 608  xyz = box MAX
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gBase      : register(t0);
[[vk::binding(1, 1)]] SamplerState gBaseSmp   : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);
// Probe atlas (set 3, binding 11/12); bound via BindReflectionProbe — the EXISTING reflection cubemap.
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
int SelectFace(float3 dir) {
    float ax = abs(dir.x), ay = abs(dir.y), az = abs(dir.z);
    if (ax >= ay && ax >= az) return dir.x >= 0.0 ? 0 : 1;
    if (ay >= ax && ay >= az) return dir.y >= 0.0 ? 2 : 3;
    return dir.z >= 0.0 ? 4 : 5;
}

// Sample the probe atlas for a world direction (block 0 = reflection, 1 = irradiance). IDENTICAL to
// lit_probe.frag's SampleProbe — this is the EXISTING reflection-cubemap sampling, reused verbatim.
float3 SampleProbe(float3 dir, int block) {
    int face = SelectFace(dir);
    float4 cp = mul(f.faceVP[face], float4(f.probePos.xyz + normalize(dir), 1.0));
    if (cp.w <= 0.0) return float3(0, 0, 0);
    float2 faceUV = (cp.xy / cp.w) * 0.5 + 0.5;
#ifdef HF_MSL_GEN
    faceUV.y = 1.0 - faceUV.y;
#endif
    faceUV = saturate(faceUV);

    int   tilesPerRow = (int)f.atlasParams2.w;
    int   col = face % tilesPerRow;
    int   row = face / tilesPerRow;
    float2 tileSize = (block == 0) ? f.atlasParams.xy : f.atlasParams.zw;
    float2 tileOrigin = float2((float)col, (float)row) * tileSize;
    if (block == 1) tileOrigin.y += f.atlasParams2.x;

    float2 texel = f.atlasParams2.yz;
    float2 atlasUV = tileOrigin + faceUV * tileSize;
    float2 tileMin = tileOrigin + texel;
    float2 tileMax = tileOrigin + tileSize - texel;
    atlasUV = clamp(atlasUV, tileMin, tileMax);
    return gProbe.SampleLevel(gProbeSmp, atlasUV, 0.0).rgb;
}

// --- Box-projection math — copied VERBATIM from engine/render/reflection_probe.h (shader port). ---
// RayBoxExitT: slab-method smallest POSITIVE exit t of the ray origin + t*dir from [boxMin,boxMax].
float RayBoxExitT(float3 origin, float3 dir, float3 boxMin, float3 boxMax) {
    const float kBig = 1e30;
    float tFar = kBig;
    [unroll] for (int i = 0; i < 3; ++i) {
        if (dir[i] > -1e-9 && dir[i] < 1e-9) continue;   // parallel slab: no finite bound on this axis
        float inv = 1.0 / dir[i];
        float t1 = (boxMin[i] - origin[i]) * inv;
        float t2 = (boxMax[i] - origin[i]) * inv;
        float tExit = max(t1, t2);                       // this slab is exited at the larger crossing
        if (tExit < tFar) tFar = tExit;                  // box exited at the smallest per-slab exit
    }
    if (tFar <= 0.0) return kBig;
    return tFar;
}

// BoxProject: parallax-corrected UNIT cubemap sample direction. parallaxStrength==0 -> normalize(R)
// EXACTLY (the byte-identical no-op proof — early-out, no dependence on the box/ray result).
float3 BoxProject(float3 reflDir, float3 worldPos, float3 center, float3 boxMin, float3 boxMax,
                  float parallaxStrength) {
    if (parallaxStrength == 0.0) return normalize(reflDir);
    float tExit = RayBoxExitT(worldPos, reflDir, boxMin, boxMax);
    float3 hit = worldPos + reflDir * tExit;
    float3 toHit = hit - center;
    float3 corrected = lerp(reflDir, toHit, parallaxStrength);
    return normalize(corrected);
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
    float3 albedo = SrgbToLinear(baseSrgb);
    float  metallic  = saturate(i.material.x);
    float  roughness = clamp(i.material.y, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    float3 rgb = albedo * 0.03;

    // Directional fill (Cook-Torrance, no shadow — the probe carries the local lighting story).
    {
        float3 L = normalize(-f.lightDir.xyz);
        rgb += hfCookTorrance(N, V, L, f.lightColor.rgb, albedo, metallic, roughness, F0);
    }

    // --- PROBE image-based lighting with BOX-PROJECTED specular. The specular reflection direction R
    //     is parallax-corrected to the local probe box BEFORE sampling the reflection cubemap, so the
    //     reflected walls line up with the room geometry (the local-reflection-probe hallmark). With
    //     parallaxStrength=0 the corrected dir == R -> byte-identical to the plain probe reflection. ---
    {
        float3 R = reflect(-V, N);
        // Box-project R to the probe box. center = probePos, box = [skyParams.xyz, pad0.xyz],
        // parallaxStrength = skyParams.w.
        float3 Rc = BoxProject(R, i.wpos, f.probePos.xyz, f.skyParams.xyz, f.pad0.xyz, f.skyParams.w);

        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);

        float3 sharpRefl = SampleProbe(Rc, 0);    // reflection block, along the CORRECTED direction
        float3 irradiance = SampleProbe(N, 1);    // irradiance block (diffuse-convolved, NOT corrected)
        float3 prefiltered = lerp(sharpRefl, irradiance, roughness);
        float3 iblSpecular = prefiltered * F;
        rgb += iblSpecular;

        // Diffuse color bleed (unchanged from lit_probe — diffuse IBL is NOT box-corrected per the spec).
        rgb += (1.0 - metallic) * albedo * irradiance * 0.55;
    }

    return float4(rgb, 1.0);
}
