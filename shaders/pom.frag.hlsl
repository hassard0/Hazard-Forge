// Slice CP — Parallax Occlusion Mapping fragment shader (NEW lit surface variant). A lit surface that
// gives a FLAT tangent-plane quad apparent per-pixel DEPTH by ray-marching a HEIGHT texture in tangent
// space (steep parallax + binary refine, render/pom.h ParallaxUV mirrored VERBATIM below), then shading
// the DISPLACED texel with the EXACT same directional-sun + shadow + point-light + procedural-IBL model
// as lit.frag — plus a soft SelfShadow toward the sun (the "occlusion" in POM).
//
// THE ZERO-HEIGHT EQUIVALENCE PROOF: at heightScale == 0 the march's degenerate exit returns the BASE
// uv EXACTLY and SelfShadow returns 1.0 EXACTLY, so every sample (height/normal/albedo) is taken at the
// SAME uv as the plain normal-mapped lit.frag and the shading math below is byte-for-byte identical to
// lit.frag. The --pom-shot showcase renders this surface with heightScale==0 and asserts SHA-equality
// to the plain lit render. NEW path behind --pom-shot; the default lit path + its goldens are UNTOUCHED.
//
// HEIGHT CONVENTION (must match render/pom.h EXACTLY): h(uv) ∈ [0,1], h==1 = surface top (flat base
// plane, nearest the eye), h==0 = deepest recess; the ray descends a "depth" 0->1 and intersects where
// depth >= (1 - h). viewDirTangent points from the surface TOWARD the eye.
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL here are the HF_MSL_GEN
// generation-path guards + [[vk::binding]] decorations (same as lit.frag / lit_pbr.frag), not backend
// CODE symbols. spirv-cross --msl-decoration-binding maps these SPIR-V bindings to the engine's flat
// Metal texture/sampler indices so the SAME HLSL feeds Vulkan (DXC) and Metal (glslang->spirv-cross).
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);          // albedo
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);    // tangent-space normal (0..1)
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);
// Height map at the SAME material slot lit_pbr.frag uses for metallic-roughness (t5/s5 -> Metal
// texture(5)/sampler(5)). Bound via the existing BindMaterialPBR texture-pair path (height in the
// metalRough slot). R channel = h ∈ [0,1], 1 = top. A flat height of 1.0 leaves the surface flat.
[[vk::binding(5, 1)]] Texture2D    gHeight    : register(t5);
[[vk::binding(6, 1)]] SamplerState gHeightSmp : register(s5);
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    // x=metallic, y=roughness, z=heightScale, w=numSteps
    [[vk::location(4)]] nointerpolation float4 material : TEXCOORD1;
    [[vk::location(5)]] float3 wtangent : TANGENT;
};
static const float HF_PI = 3.14159265358979323846;

// ---- render/pom.h ParallaxUV, copied VERBATIM (steep-parallax linear march + fixed binary refine).
// sampleHeight is inlined as gHeight.Sample(...).r. Returns baseUV EXACTLY at heightScale==0 / straight-
// on view (the equivalence proof). ----
float SampleHeight(float2 uv) { return saturate(gHeight.Sample(gHeightSmp, uv).r); }

float2 PomParallaxUV(float2 baseUV, float3 viewDirTangent, float heightScale, int numSteps) {
    if (heightScale == 0.0) return baseUV;                                    // zero amplitude -> flat
    if (viewDirTangent.x == 0.0 && viewDirTangent.y == 0.0) return baseUV;    // straight-on view
    float vz = viewDirTangent.z;
    if (abs(vz) < 1e-4) vz = (vz < 0.0 ? -1e-4 : 1e-4);                       // guard the .z divide

    int steps = numSteps > 0 ? numSteps : 1;
    float2 maxOffset = float2(viewDirTangent.x / vz, viewDirTangent.y / vz) * heightScale;
    float dt = 1.0 / (float)steps;

    float prevSurfDepth = 1.0 - SampleHeight(baseUV);
    if (0.0 >= prevSurfDepth) return baseUV;                                  // flat top -> entry texel

    float prevRayDepth = 0.0;
    float curRayDepth = 0.0;
    float2 curUV = baseUV;
    bool hit = false;
    [loop] for (int k = 1; k <= steps; ++k) {
        float rayDepth = (float)k * dt;
        float2 uv = baseUV - maxOffset * rayDepth;
        float surfDepth = 1.0 - SampleHeight(uv);
        if (rayDepth >= surfDepth) {
            prevRayDepth = (float)(k - 1) * dt;
            curRayDepth = rayDepth;
            curUV = uv;
            hit = true;
            break;
        }
    }
    if (!hit) return curUV;                                                   // no crossing -> deepest

    float loDepth = prevRayDepth;     // above (gap < 0)
    float hiDepth = curRayDepth;      // below (gap >= 0)
    float2 refinedUV = curUV;
    [unroll] for (int r = 0; r < 8; ++r) {
        float midDepth = 0.5 * (loDepth + hiDepth);
        float2 uv = baseUV - maxOffset * midDepth;
        float surfDepth = 1.0 - SampleHeight(uv);
        float gap = midDepth - surfDepth;
        if (gap >= 0.0) { hiDepth = midDepth; refinedUV = uv; }
        else            { loDepth = midDepth; }
    }
    return refinedUV;
}

// ---- render/pom.h SelfShadow, copied VERBATIM. Returns 1.0 EXACTLY at heightScale==0 / overhead /
// below-plane light (so heightScale==0 is byte-identical to lit.frag, which has no self-shadow). ----
float PomSelfShadow(float2 uv, float3 lightDirTangent, float heightScale, int numSteps) {
    if (heightScale == 0.0) return 1.0;
    if (lightDirTangent.x == 0.0 && lightDirTangent.y == 0.0) return 1.0;
    if (lightDirTangent.z <= 0.0) return 1.0;

    int steps = numSteps > 0 ? numSteps : 1;
    float h0 = SampleHeight(uv);
    float lz = lightDirTangent.z;
    float2 lstep = float2(lightDirTangent.x / lz, lightDirTangent.y / lz) * heightScale;
    float dt = 1.0 / (float)steps;

    float maxBlock = 0.0;
    [loop] for (int k = 1; k <= steps; ++k) {
        float t = (float)k * dt;
        float2 suv = uv + lstep * t;
        float rayHeight = h0 + t * (1.0 - h0);
        float sh = SampleHeight(suv);
        float block = sh - rayHeight;
        if (block > maxBlock) maxBlock = block;
    }
    return 1.0 - saturate(maxBlock);
}

// ---- The lit.frag shading model, copied VERBATIM (GGX / Smith / Fresnel / SkyColor / Cook-Torrance)
// so the heightScale==0 output equals lit.frag bit-for-bit. ----
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
float3 SkyColor(float3 dir) {
    float3 d = normalize(dir);
    float  h       = saturate(d.y * 0.5 + 0.5);
    float3 zenith  = float3(0.18, 0.30, 0.62);
    float3 horizon = float3(0.65, 0.72, 0.82);
    float3 sky     = lerp(horizon, zenith, pow(h, 0.8));
    float3 ground = float3(0.12, 0.11, 0.10);
    if (d.y < 0.0) { float g = saturate(-d.y * 2.0); sky = lerp(sky, ground, g); }
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

float4 main(PSInput i) : SV_Target {
    // Geometric (interpolated) world normal. (Same as lit.frag.)
    float3 Ng = normalize(i.wnormal);

    // --- Orthonormal TBN (Gram-Schmidt), IDENTICAL to lit.frag. Rows = T,B,N: mul(vec, TBN) maps a
    // TANGENT-space vector to WORLD; mul(TBN, vec) maps a WORLD vector to TANGENT (transpose == inverse
    // for an orthonormal basis). ---
    float3 T = normalize(i.wtangent - Ng * dot(Ng, i.wtangent));
    float3 B = cross(Ng, T);
    float3x3 TBN = float3x3(T, B, Ng);

    // --- Parallax march in tangent space. View dir = surface->eye (world), rotated into tangent space.
    // heightScale==0 / straight-on -> the march returns i.uv EXACTLY (the equivalence proof). ---
    float  heightScale = i.material.z;
    int    numSteps    = (int)(i.material.w + 0.5);
    float3 Vw = normalize(f.viewPos.xyz - i.wpos);
    float3 Vt = mul(TBN, Vw);                               // world->tangent (eye dir in tangent space)
    float2 uv = PomParallaxUV(i.uv, Vt, heightScale, numSteps);

    // --- Sample the displaced texel + perturb the normal (same decode as lit.frag). ---
    float3 nTS = gNormalMap.Sample(gNormalSmp, uv).xyz * 2.0 - 1.0;
    float3 N = normalize(mul(nTS, TBN));
    float3 V = Vw;
    float3 tex = gTex.Sample(gSmp, uv).rgb * i.color;

    float3 albedo   = tex;
    float  metallic = saturate(i.material.x);
    float  roughness = clamp(i.material.y, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // --- Directional shadow (PCF), IDENTICAL to lit.frag. ---
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

    // --- POM self-shadow toward the sun. Light dir (surface->light) rotated into tangent space; the
    // soft occlusion attenuates the DIRECT sun only (matching lit.frag's shadow application). At
    // heightScale==0 PomSelfShadow returns 1.0 EXACTLY -> no change vs lit.frag. ---
    float3 Lw = normalize(-f.lightDir.xyz);
    float3 Lt = mul(TBN, Lw);
    float  selfShadow = PomSelfShadow(uv, Lt, heightScale, numSteps);

    float3 rgb = albedo * 0.03;
    {
        float3 L = Lw;
        float3 radiance = f.lightColor.rgb * shadow * selfShadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }

    int n = (int)f.ptCount.x;
    for (int li = 0; li < n; ++li) {
        float3 lp = f.ptPos[li].xyz;
        float  radius = f.ptPos[li].w;
        float3 lc = f.ptColor[li].rgb;
        float  intensity = f.ptColor[li].w;
        float3 Lv = lp - i.wpos;
        float  dist = length(Lv);
        float3 Ld = Lv / max(dist, 1e-4);
        float  att = saturate(1.0 - dist / radius);
        att *= att;
        float3 radiance = lc * intensity * att;
        rgb += hfCookTorrance(N, V, Ld, radiance, albedo, metallic, roughness, F0);
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

    return float4(rgb, 1.0);
}
