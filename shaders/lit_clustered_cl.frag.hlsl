// Slice CL — Clustered Light Culling (Forward+): the CLUSTERED LIT fragment shader. The SAME
// directional sun + ambient + procedural-sky IBL machinery as lit.frag, PLUS a clustered point-light
// loop that reads ONLY this fragment's cluster's light list:
//   * compute this fragment's CLUSTER index from its screen tile (SV_Position.xy) and its EXPONENTIAL
//     view-depth slice (-viewPos.z), using the EXACT formulas from render::cluster (CPU),
//   * read gClusterList[idx] = {count, idx[64]} (the fixed-slot ordered list the compute pass filled),
//   * iterate ONLY that cluster's `count` lights, accumulating each with a WINDOWED HARD-RADIUS
//     attenuation: atten = (1/(d²+ε)) * clamp(1 - (d/radius)⁴, 0, 1)². A light contributes EXACTLY 0
//     at and beyond `radius` — the property that makes clustered shading BYTE-IDENTICAL to brute-force.
//
// (Distinct from Slice AG's lit_clustered.frag.hlsl: AG uses a SMOOTH falloff + a separate flat
// lightIndices buffer; THIS shader uses the hard-windowed atten + the fixed-slot ClusterList the CL
// compute pass writes, and is asserted byte-identical to a brute-force all-lights render. Separate
// golden (clustered_lights.png); AG's path + golden untouched.)
//
// In BRUTE-FORCE mode the host uploads a 1x1x1 grid whose single cluster lists ALL lights -> the SAME
// shader loops over every light. Confirming the clustered (16x9x24) capture == the brute-force capture
// (byte-identical SHA) proves no contributing light was ever wrongly culled.
//
// The three cluster storage buffers reuse the Slice-AG cluster binding path (set 3 on Vulkan; flat
// fragment buffers on Metal) via BindLightClusters / usesLightClusters — binding 13 = the ClusterList
// array, binding 14 = an unused dummy (keeps the 3-buffer layout), binding 15 = the lights array.
struct FrameData {
    float4x4 viewProj;     //   0  (lit.vert reads this at offset 0)
    float4x4 view;         //  64  world -> view (for view-space position + depth)
    float4   lightDir;     // 128  directional sun
    float4   lightColor;   // 144
    float4   viewPos;      // 160  world-space camera position (for V)
    float4   clusterParams;  // 176  x=dimX, y=dimY, z=dimZ, w=zNear
    float4   clusterParams2; // 192  x=zFar, y=screenW, z=screenH, w=ambient
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);

// Clustered-lighting storage buffers — the Slice-AG cluster set (set 3 Vulkan; flat fragment buffers
// Metal) at bindings 13/14/15. ClusterList = {count, pad, idx[64]} the CL compute pass filled.
#define HF_MAX_LIGHTS_PER_CLUSTER 96
struct ClusterList { uint count; uint pad0; uint pad1; uint pad2; uint idx[HF_MAX_LIGHTS_PER_CLUSTER]; };
struct GpuLight { float4 posRadius; float4 color; };  // xyz=viewPos w=radius ; rgb=color w=intensity
[[vk::binding(13, 3)]] StructuredBuffer<ClusterList> gClusterList  : register(t13, space3);
[[vk::binding(14, 3)]] StructuredBuffer<uint>        gUnused       : register(t14, space3);
[[vk::binding(15, 3)]] StructuredBuffer<GpuLight>    gLights       : register(t15, space3);

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

// Cluster index from a fragment. EXACT mirror of render::cluster: screen tile from SV_Position.xy
// (top-left origin); exponential z-slice from positive view-distance. idx = cx + cy*dimX + cz*dimX*dimY.
int ClusterIndex(float2 screenXY, float viewZ) {
    int   dimX = (int)f.clusterParams.x;
    int   dimY = (int)f.clusterParams.y;
    int   dimZ = (int)f.clusterParams.z;
    float zNear = f.clusterParams.w;
    float zFar  = f.clusterParams2.x;
    float W = f.clusterParams2.y;
    float H = f.clusterParams2.z;

    int cx = clamp((int)floor(screenXY.x / W * (float)dimX), 0, dimX - 1);
    int cy = clamp((int)floor(screenXY.y / H * (float)dimY), 0, dimY - 1);
    int cz;
    if (viewZ <= zNear)      cz = 0;
    else if (viewZ >= zFar)  cz = dimZ - 1;
    else {
        float t = log(viewZ / zNear) / log(zFar / zNear);   // [0,1)
        cz = clamp((int)floor(t * (float)dimZ), 0, dimZ - 1);
    }
    return cx + cy * dimX + cz * (dimX * dimY);
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

    float3 rgb = albedo * 0.03;  // small constant ambient (same as lit.frag)

    // --- Directional SUN (no shadow): identical to the default lit path's directional term. ---
    {
        float3 L = normalize(-f.lightDir.xyz);
        rgb += hfCookTorrance(N, V, L, f.lightColor.rgb, albedo, metallic, roughness, F0);
    }

    // --- Clustered point lights (the ONLY term that differs from the default lit path) -----------
    float3 vpos  = mul(f.view, float4(i.wpos, 1.0)).xyz;
    float  viewZ = -vpos.z;   // positive distance in front of the camera

    int idx = ClusterIndex(i.clip.xy, viewZ);
    ClusterList cl = gClusterList[idx];
    float3x3 viewRot = (float3x3)f.view;

    [loop] for (uint j = 0; j < cl.count; ++j) {
        uint li = cl.idx[j];
        GpuLight gl = gLights[li];
        float3 lvpos     = gl.posRadius.xyz;   // view-space light position
        float  radius    = gl.posRadius.w;
        float3 lcolor    = gl.color.rgb;
        float  intensity = gl.color.w;

        float3 Lv   = lvpos - vpos;            // view-space light -> fragment
        float  dist = length(Lv);

        // WINDOWED HARD-RADIUS attenuation: (1/(d²+ε)) * clamp(1-(d/r)⁴,0,1)². EXACTLY 0 at d>=radius,
        // so a light assigned iff its sphere intersects the cluster AABB == clustered identical to
        // brute-force. (ε keeps the 1/d² finite at d->0.)
        float r4   = (dist / radius); r4 = r4 * r4; r4 = r4 * r4;   // (d/r)^4
        float win  = saturate(1.0 - r4); win = win * win;
        float atten = (1.0 / (dist * dist + 0.01)) * win;

        // World-space light dir (lighting uses world N/V). view^-1 of a direction == transpose(view3x3)
        // applied as mul(rowVec, M). The upper-left 3x3 of `view` is orthonormal.
        float3 Lview  = Lv / max(dist, 1e-4);
        float3 Lworld = normalize(mul(Lview, viewRot));

        float3 radiance = lcolor * intensity * atten;
        rgb += hfCookTorrance(N, V, Lworld, radiance, albedo, metallic, roughness, F0);
    }

    // --- Procedural image-based lighting (environment reflection), identical to lit.frag. ---
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
