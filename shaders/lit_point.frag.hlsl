// Point-light omnidirectional shadow lit fragment shader (Slice AF). Same PBR/IBL/ambient machinery
// as lit.frag, but the shadowing light is a POINT light that casts shadows in ALL directions via a
// 6-face cube rendered into one shadow ATLAS (3x2 grid of 1024 tiles in a 3072 map). No cubemap.
//
//   * L = ptPos - wpos; pick the cube face by the DOMINANT ABSOLUTE AXIS of (wpos - ptPos),
//   * project wpos by that face's view-proj,
//   * remap the [0,1] face UV into that face's atlas TILE (col=face%3, row=face/3),
//   * 3x3 PCF clamped to the tile bounds (so the kernel never bleeds into a neighbour tile),
//   * distance attenuation over `range`.
//
// The face-selection rule is the EXACT mirror of render::point_shadow::SelectFace (CPU) so the
// render face and the sampled face always agree (continuous shadows across face boundaries).
//
// A DIM directional fill keeps unlit areas off pure-black; the point light dominates so its radial
// shadows read clearly.
//
// Point FrameData layout (own struct; fits kFrameUboSize=1024 -> 608 bytes). The --point-shadow-shot
// showcase fills the per-frame UBO with THIS layout. See docs/.../point-shadows-design.md.
struct FrameData {
    float4x4 viewProj;     //   0
    float4   lightDir;     //  64  directional fill (dim)
    float4   lightColor;   //  80
    float4   viewPos;      //  96
    float4x4 faceVP[6];    // 112  6 face view-projs (384B) -> ends 496
    float4   ptPos;        // 496  xyz position, w=range
    float4   ptColor;      // 512  rgb color, w=intensity
    float4   atlasParams;  // 528  x=tilesPerRow(3), y=tilesPerCol(2), z=1/atlasSize, w=near
    float4   camFwd;       // 544
    float4   camRight;     // 560
    float4   camUp;        // 576
    float4   skyParams;    // 592  x=tan(0.5*fovY), y=aspect
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Point shadow atlas lives in the per-frame set (set 0): binding 1 = depth image, binding 2 = sampler.
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);
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

// EXACT mirror of render::point_shadow::SelectFace(wpos - ptPos): dominant absolute axis -> face.
//   0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z.
int SelectFace(float3 dirLightToFrag) {
    float ax = abs(dirLightToFrag.x);
    float ay = abs(dirLightToFrag.y);
    float az = abs(dirLightToFrag.z);
    if (ax >= ay && ax >= az) return dirLightToFrag.x >= 0.0 ? 0 : 1;
    if (ay >= ax && ay >= az) return dirLightToFrag.y >= 0.0 ? 2 : 3;
    return dirLightToFrag.z >= 0.0 ? 4 : 5;
}

// Sample one cube face's atlas tile with 3x3 PCF, clamped to the tile so the kernel never bleeds
// into a neighbour. faceUV is the [0,1] UV within the face; (col,row) selects the tile in the 3x2
// grid. tileScale = (1/tilesPerRow, 1/tilesPerCol). atlasTexel = 1/atlasSize.
float SampleFacePCF(float2 faceUV, float curDepth, int col, int row, float2 tileScale,
                    float atlasTexel) {
    float2 tileOrigin = float2((float)col, (float)row) * tileScale;
    float2 atlasUV = tileOrigin + faceUV * tileScale;
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

    float3 rgb = albedo * 0.03;  // small ambient

    // --- Dim directional fill (no shadow): keeps unlit areas from going pure black. ---
    {
        float3 L = normalize(-f.lightDir.xyz);
        rgb += hfCookTorrance(N, V, L, f.lightColor.rgb, albedo, metallic, roughness, F0);
    }

    // --- POINT LIGHT: 6-face cube shadow atlas + distance falloff. ---
    {
        float3 Lv   = f.ptPos.xyz - i.wpos;          // light -> fragment (negated)
        float  dist = length(Lv);
        float3 L    = Lv / max(dist, 1e-4);          // unit dir fragment -> light
        float  range     = f.ptPos.w;
        float  intensity = f.ptColor.w;

        // Distance falloff over the range (smooth, like the other point lights).
        float att = saturate(1.0 - dist / range);
        att *= att;

        // Pick the cube face from the dominant axis of (wpos - ptPos) == -Lv.
        int face = SelectFace(-Lv);

        float shadow = 1.0;
        {
            float4 lp = mul(f.faceVP[face], float4(i.wpos, 1.0));
            if (lp.w > 0.0) {
                float3 proj = lp.xyz / lp.w;
                float2 faceUV = proj.xy * 0.5 + 0.5;
#ifdef HF_MSL_GEN
                faceUV.y = 1.0 - faceUV.y;
#endif
                float curDepth = proj.z;
                if (faceUV.x >= 0.0 && faceUV.x <= 1.0 && faceUV.y >= 0.0 && faceUV.y <= 1.0 &&
                    curDepth >= 0.0 && curDepth <= 1.0) {
                    int   tilesPerRow = (int)f.atlasParams.x;   // 3
                    int   col = face % tilesPerRow;
                    int   row = face / tilesPerRow;
                    float2 tileScale = float2(1.0 / f.atlasParams.x, 1.0 / f.atlasParams.y);
                    float  atlasTexel = f.atlasParams.z;        // 1/atlasSize
                    shadow = SampleFacePCF(faceUV, curDepth, col, row, tileScale, atlasTexel);
                }
            }
        }

        float3 radiance = f.ptColor.rgb * intensity * att * shadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }

    // --- Procedural image-based lighting (environment reflection), same as lit.frag. ---
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

    return float4(rgb, 1.0);
}
