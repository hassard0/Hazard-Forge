// Slice BZ — bindless-textures lit fragment shader. VULKAN-ONLY. Byte-for-byte identical to lit.frag
// in EVERY lighting/shadow/IBL computation; the ONLY difference is the BASE-COLOR TEXTURE SOURCE:
//
//   lit.frag:           float3 tex = gTex.Sample(gSmp, i.uv).rgb * i.color;   // per-material bound set 1
//   lit_bindless.frag:  float3 tex = gTextures[NonUniformResourceIndex(i.texIndex)]
//                                        .Sample(gBindlessSmp, i.uv).rgb * i.color;  // ONE bindless array
//
// `gTextures[]` is an UNBOUNDED runtime descriptor array (space 4) of every scene texture, bound ONCE;
// `i.texIndex` is the per-draw material index (push constant via lit_bindless.vert). NonUniformResourceIndex
// marks the index as potentially non-uniform across a wavefront (descriptor-indexing requirement; DXC
// lowers it to SPV_EXT_descriptor_indexing). The bindless shared sampler (gBindlessSmp) is created
// LINEAR / REPEAT — IDENTICAL to the device default sampler the bound material set uses — so the SAME
// texel of the SAME texture is fetched, and the bindless image is BYTE-IDENTICAL to the bound reference
// image. The normal map / shadow / frame bindings are unchanged from lit.frag.
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Shadow map lives in the per-frame set (set 0): binding 1 = depth image, binding 2 = sampler.
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
// Tangent-space normal map at the material set (set 1, binding 3/4) — UNCHANGED from lit.frag. The
// base-color texture (set 1, binding 0/1 in lit.frag) is replaced by the bindless array below, so the
// material set's binding 0/1 are present (in the layout) but unused by this shader.
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);
// The BINDLESS texture array: an unbounded Texture2D[] at set 4 binding 1, filled with every scene
// texture (index i -> textures[i]). + a shared sampler at set 4 binding 0. A draw samples its base
// color by INDEX (texIndex) instead of binding a per-material set. The ARRAY is the HIGHEST-numbered
// binding (binding 1 > the sampler's binding 0) because the VARIABLE_DESCRIPTOR_COUNT array binding
// must be highest (VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-03004).
[[vk::binding(0, 4)]] SamplerState gBindlessSmp : register(s0, space4);
[[vk::binding(1, 4)]] Texture2D    gTextures[]  : register(t0, space4);
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    [[vk::location(4)]] nointerpolation float2 material : TEXCOORD1; // x=metallic, y=roughness
    [[vk::location(5)]] float3 wtangent : TANGENT;
    [[vk::location(6)]] nointerpolation uint texIndex : TEXCOORD2;    // bindless array index
};
static const float HF_PI = 3.14159265358979323846;

// GGX / Trowbridge-Reitz normal distribution. alpha = roughness^2.
float hfDistributionGGX(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float d  = (NoH * NoH) * (a2 - 1.0) + 1.0;
    return a2 / max(HF_PI * d * d, 1e-7);
}
// Smith geometry with Schlick-GGX (direct-lighting k = (r+1)^2 / 8).
float hfGeometrySchlickGGX(float NoX, float k) {
    return NoX / (NoX * (1.0 - k) + k);
}
float hfGeometrySmith(float NoV, float NoL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return hfGeometrySchlickGGX(NoV, k) * hfGeometrySchlickGGX(NoL, k);
}
// Fresnel-Schlick.
float3 hfFresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (float3(1.0, 1.0, 1.0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}
// Procedural sky color for a world-space direction. Replicates the gradient + sun glow from
// sky.frag.hlsl (same zenith/horizon/ground colors and the same sun term keyed off the directional
// light). Used for image-based lighting: metals reflect this in the reflection direction.
float3 SkyColor(float3 dir) {
    float3 d = normalize(dir);
    // Horizon -> zenith gradient.
    float  h       = saturate(d.y * 0.5 + 0.5);
    float3 zenith  = float3(0.18, 0.30, 0.62);
    float3 horizon = float3(0.65, 0.72, 0.82);
    float3 sky     = lerp(horizon, zenith, pow(h, 0.8));
    // Dim ground haze for the lower hemisphere.
    float3 ground = float3(0.12, 0.11, 0.10);
    if (d.y < 0.0) {
        float g = saturate(-d.y * 2.0);
        sky = lerp(sky, ground, g);
    }
    // Sun glow toward the (incoming) directional light direction.
    float3 sunDir = normalize(-f.lightDir.xyz);
    float  s = pow(max(dot(d, sunDir), 0.0), 256.0);
    sky += float3(1.0, 0.95, 0.8) * s * 2.0;
    return sky;
}

// Cook-Torrance contribution for a single light of given radiance.
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
    // Energy conservation: diffuse only from the non-reflected, non-metallic fraction.
    float3 kd = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);
    float3 diff = kd * albedo / HF_PI;
    return (diff + spec) * radiance * NoL;
}

float4 main(PSInput i) : SV_Target {
    // Geometric (interpolated) world normal.
    float3 Ng = normalize(i.wnormal);

    // --- Tangent-space normal mapping (UNCHANGED from lit.frag). A flat normal map leaves N == Ng. ---
    float3 T = normalize(i.wtangent - Ng * dot(Ng, i.wtangent));
    float3 B = cross(Ng, T);                       // handedness baked into T at authoring time
    float3x3 TBN = float3x3(T, B, Ng);             // rows = T,B,N
    float3 nTS = gNormalMap.Sample(gNormalSmp, i.uv).xyz * 2.0 - 1.0;
    float3 N = normalize(mul(nTS, TBN));
    float3 V = normalize(f.viewPos.xyz - i.wpos);
    // --- THE ONLY DIFFERENCE FROM lit.frag: sample the base color from the BINDLESS ARRAY by index
    // (NonUniformResourceIndex) instead of the per-material bound base-color set. Same texture, same
    // (LINEAR/REPEAT) sampler, same uv -> same texel -> byte-identical to the bound reference. ---
    float3 tex = gTextures[NonUniformResourceIndex(i.texIndex)].Sample(gBindlessSmp, i.uv).rgb * i.color;

    // --- Per-material PBR (UNCHANGED from lit.frag). ---
    float3 albedo   = tex;
    float  metallic = saturate(i.material.x);
    float  roughness = clamp(i.material.y, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic); // dielectric vs. metal base reflectance

    // --- Directional shadow (UNCHANGED from lit.frag). ---
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

    // Directional light.
    {
        float3 L = normalize(-f.lightDir.xyz);
        float3 radiance = f.lightColor.rgb * shadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }

    // Colored point lights.
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
        att *= att;                              // smooth falloff to the radius
        float3 radiance = lc * intensity * att;
        rgb += hfCookTorrance(N, V, Ld, radiance, albedo, metallic, roughness, F0);
    }

    // Procedural image-based lighting (UNCHANGED from lit.frag).
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
