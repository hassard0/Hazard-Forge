// Slice DD — Runtime Cubemap-Capture Reflection Probe fragment shader. A reflective-surface variant
// of reflprobe.frag (DA): same base lit PBR (Cook-Torrance) + the SAME box-projection
// (reflprobe::RayBoxExitT + BoxProject, copied VERBATIM from engine/render/reflection_probe.h), but
// the reflection source is a CAPTURED hardware cubemap (a real TextureCube the runtime probe rendered
// the scene into, 6 faces from the probe center) — the DYNAMIC counterpart to DA's STATIC env
// cubemap. The reflective surface computes R = reflect(-V, N), box-projects it to the probe box, and
// samples the captured cube along the corrected direction, so it mirrors the ACTUAL surrounding
// scene geometry.
//
// The capture is BYTE-IDENTICAL to the scene rendered directly with each face's view/proj (the
// capture-correctness proof asserted by the showcase), so this samples a faithful scene render. The
// cube's face convention + the per-face capture view/proj come from engine/render/cubemap.h; the cube
// lookup here is just the hardware TextureCube.Sample(dir), which uses the IDENTICAL major-axis
// convention cubemap.h documents — so capture + sample agree.
//
// This is a NEW shader behind the --captureprobe-shot showcase flag; the existing lit_probe / IBL /
// reflprobe paths + their goldens stay BYTE-IDENTICAL.
//
// FrameData: the SAME 624-byte layout as lit_probe / reflprobe (so the showcase reuses ProbeFrameData);
// skyParams.xyz = boxMin, skyParams.w = parallaxStrength, pad0.xyz = boxMax; probePos.xyz = the probe
// CENTER (== box center, where the cube was captured from). The faceVP[6] / atlas params are unused
// here (the cube is sampled directly, not via an atlas), kept only for layout compatibility.
struct FrameData {
    float4x4 viewProj;     //   0
    float4   lightDir;     //  64  directional fill
    float4   lightColor;   //  80
    float4   viewPos;      //  96
    float4x4 faceVP[6];    // 112  (unused here) -> ends 496
    float4   probePos;     // 496  xyz probe capture center (== box CENTER)
    float4   atlasParams;  // 512  (unused here)
    float4   atlasParams2; // 528  (unused here)
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
// Captured cubemap (set 3, binding 11/12); bound via BindCubemapProbe — a real TextureCube.
[[vk::binding(11, 3)]] TextureCube  gCube      : register(t11);
[[vk::binding(12, 3)]] SamplerState gCubeSmp   : register(s11);
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

// Sample the CAPTURED cubemap along a world direction. A hardware TextureCube does the major-axis
// face select + face UV internally — the IDENTICAL convention engine/render/cubemap.h documents, so
// the captured faces line up with how this samples them. On Metal the captured V is flipped exactly
// like the other showcases (the host wraps the per-face projection); the cube sample matches.
float3 SampleCube(float3 dir) {
    return gCube.SampleLevel(gCubeSmp, normalize(dir), 0.0).rgb;
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

    // Directional fill (Cook-Torrance, no shadow — the captured cube carries the scene's appearance).
    {
        float3 L = normalize(-f.lightDir.xyz);
        rgb += hfCookTorrance(N, V, L, f.lightColor.rgb, albedo, metallic, roughness, F0);
    }

    // --- Captured-cubemap reflection with BOX-PROJECTED specular. R is parallax-corrected to the
    //     probe box BEFORE sampling the CAPTURED cube, so the reflected scene objects line up with the
    //     room geometry. With parallaxStrength=0 the corrected dir == R (BoxProject's early-out). ---
    {
        float3 R = reflect(-V, N);
        float3 Rc = BoxProject(R, i.wpos, f.probePos.xyz, f.skyParams.xyz, f.pad0.xyz, f.skyParams.w);

        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);

        float3 sharpRefl  = SampleCube(Rc);   // captured scene reflection along the CORRECTED direction
        float3 irradiance = SampleCube(N);    // crude ambient from the captured cube along N
        float3 prefiltered = lerp(sharpRefl, irradiance, roughness);
        float3 iblSpecular = prefiltered * F;
        rgb += iblSpecular;

        // Diffuse color bleed from the captured environment (NOT box-corrected — diffuse is broad).
        rgb += (1.0 - metallic) * albedo * irradiance * 0.55;
    }

    return float4(rgb, 1.0);
}
