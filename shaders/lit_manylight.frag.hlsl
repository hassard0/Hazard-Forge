// Many-light lit fragment shader (Slice ML1, Track-S S7) — the SSBO-backed path that lifts the
// HF_MAX_POINT_LIGHTS=8 FrameData cap to 100+ dynamic lights. The base is the lit.frag-family
// clustered fragment (lit_clustered.frag, Slice AG) — same PBR + dim directional fill + procedural
// sky IBL machinery — with the point lights coming from a runtime-sized StructuredBuffer instead of
// the FrameData array (shaders/frame_data.hlsli is NOT included here and stays byte-untouched;
// this pipeline is ADDITIVE — no existing shader/pipeline/golden changes):
//   * compute this fragment's CLUSTER index from its screen tile (SV_Position.xy) and its EXPONENTIAL
//     view-depth slice (-viewSpacePos.z), using the EXACT formulas from render::clustered (CPU),
//   * read clusters[idx] = {offset, count},
//   * iterate ONLY that cluster's `count` lights (indices lightIndices[offset..offset+count)),
//   * accumulate each point light's PBR contribution with smooth distance falloff by its radius.
// The light SSBO is runtime-sized (N up to render::manylight::kMaxManyLights = 1024) — the loop is
// bounded by each cluster's count, so there is NO shader-side light cap.
//
// The three buffers are bound to the FRAGMENT stage via the dedicated cluster set (set 3 on Vulkan,
// bindings 13/14/15 — binding 13 proven by the R1 --probe-binding13 fix; flat fragment buffer slots
// 13/14/15 on Metal via spirv-cross --msl-decoration-binding). Lights are stored in VIEW space (CPU
// pre-transforms them), so this shader does the clustered point lighting in view space — it already
// has `view`.
//
// CPU<->shader agreement is critical: a mismatch in the cluster-index formula or the sphere/AABB
// assignment shows up as rectangular tile banding or lights popping at cluster boundaries. The
// formulas here are the verbatim mirror of engine/render/clustered.h, whose CPU assignment over the
// SAME hashed light field is digest-pinned by tests/manylight_test.cpp.
//
// Many-light FrameData layout (own struct — the lit_clustered 224-byte layout, NOT frame_data.hlsli;
// fits kFrameUboSize=1024). The --manylight-shot showcase fills the per-frame UBO with THIS layout.
struct FrameData {
    float4x4 viewProj;     //   0  (lit.vert reads this at offset 0)
    float4x4 view;         //  64  world -> view (for view-space position + depth)
    float4   lightDir;     // 128  directional fill (dim)
    float4   lightColor;   // 144
    float4   viewPos;      // 160  world-space camera position (for V)
    float4   clusterParams;  // 176  x=CX, y=CY, z=CZ, w=znear
    float4   clusterParams2; // 192  x=zfar, y=screenW, z=screenH, w=tanX
    float4   clusterParams3; // 208  x=tanY, (yzw unused)
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);

// Clustered-lighting storage buffers — dedicated set 3 (Vulkan), flat fragment buffers (Metal).
// The binding numbers (13/14/15) are chosen past the full-PBR material's 0..10 and the env's 11/12
// so spirv-cross --msl-decoration-binding maps them to Metal FRAGMENT BUFFER slots 13/14/15 (a
// separate index space from textures/samplers), never colliding with the fragment FrameData buffer
// (0) or the bloom push-const buffer (1). On Vulkan these are bindings 13/14/15 of descriptor set 3.
struct Cluster { uint offset; uint count; };
struct GpuLight { float4 posRadius; float4 color; };  // xyz=viewPos w=radius ; rgb=color w=intensity
[[vk::binding(13, 3)]] StructuredBuffer<Cluster>  gClusters     : register(t13, space3);
[[vk::binding(14, 3)]] StructuredBuffer<uint>     gLightIndices : register(t14, space3);
[[vk::binding(15, 3)]] StructuredBuffer<GpuLight> gLights       : register(t15, space3);

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

// --- Cluster index from a fragment (EXACT mirror of render::clustered). ---
// Screen tile from SV_Position.xy (top-left origin); exponential z-slice from view-space depth.
int ClusterIndex(float2 screenXY, float viewZ) {
    int   CX = (int)f.clusterParams.x;
    int   CY = (int)f.clusterParams.y;
    int   CZ = (int)f.clusterParams.z;
    float znear = f.clusterParams.w;
    float zfar  = f.clusterParams2.x;
    float W = f.clusterParams2.y;
    float H = f.clusterParams2.z;

    int cx = (int)floor(screenXY.x / W * (float)CX);
    int cy = (int)floor(screenXY.y / H * (float)CY);
    cx = clamp(cx, 0, CX - 1);
    cy = clamp(cy, 0, CY - 1);

    int cz;
    if (viewZ <= znear)      cz = 0;
    else if (viewZ >= zfar)  cz = CZ - 1;
    else {
        float t = log(viewZ / znear) / log(zfar / znear);   // [0,1)
        cz = clamp((int)floor(t * (float)CZ), 0, CZ - 1);
    }
    return cx + cy * CX + cz * (CX * CY);
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

    float3 rgb = albedo * 0.03;  // small ambient

    // --- Dim directional fill (no shadow): keeps unlit areas from going pure black. ---
    {
        float3 L = normalize(-f.lightDir.xyz);
        rgb += hfCookTorrance(N, V, L, f.lightColor.rgb, albedo, metallic, roughness, F0);
    }

    // --- Clustered point lights ---------------------------------------------------------------
    // View-space position of this fragment (lights are stored view-space; pick the cluster by the
    // same view-depth the CPU culler used).
    float3 vpos = mul(f.view, float4(i.wpos, 1.0)).xyz;
    float  viewZ = -vpos.z;   // positive distance in front of the camera

    int idx = ClusterIndex(i.clip.xy, viewZ);
    Cluster cl = gClusters[idx];

    for (uint j = 0; j < cl.count; ++j) {
        uint li = gLightIndices[cl.offset + j];
        GpuLight gl = gLights[li];
        float3 lvpos = gl.posRadius.xyz;
        float  radius = gl.posRadius.w;
        float3 lcolor = gl.color.rgb;
        float  intensity = gl.color.w;

        // View-space light -> fragment.
        float3 Lv = lvpos - vpos;
        float  dist = length(Lv);
        if (dist >= radius) continue;            // outside the light's influence (and its cull sphere)
        // Light direction in WORLD space (lighting uses world N/V). Reconstruct from the inverse of
        // the rotational part of `view`: world dir = transpose(view3x3) * viewDir. The view matrix's
        // upper-left 3x3 is orthonormal (rotation), so its inverse is its transpose.
        float3 Lview = Lv / max(dist, 1e-4);
        float3x3 viewRot = (float3x3)f.view;
        float3 Lworld = normalize(mul(Lview, viewRot));  // mul(row, M) == M^T * row == view^-1 dir

        // Smooth distance falloff over the radius (matches the CPU cull radius so nothing pops at a
        // cluster boundary). sat(1 - d/r)^2.
        float att = saturate(1.0 - dist / radius);
        att *= att;

        float3 radiance = lcolor * intensity * att;
        rgb += hfCookTorrance(N, V, Lworld, radiance, albedo, metallic, roughness, F0);
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

    return float4(rgb, 1.0);
}
