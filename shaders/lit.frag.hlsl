struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Shadow map lives in the per-frame set (set 0): binding 1 = depth image, binding 2 = sampler.
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
};
float4 main(PSInput i) : SV_Target {
    float3 N = normalize(i.wnormal);
    float3 L = normalize(-f.lightDir.xyz);
    float  diff = max(dot(N, L), 0.0);
    float3 V = normalize(f.viewPos.xyz - i.wpos);
    float3 H = normalize(L + V);
    float  spec = pow(max(dot(N, H), 0.0), 32.0);
    float  ambient = 0.12;
    float3 tex = gTex.Sample(gSmp, i.uv).rgb * i.color;

    // --- Directional shadow: project world pos into the light's clip space, compare depth. ---
    // lightViewProj uses Ortho (same Y-flip as Perspective), so smUV = proj.xy*0.5+0.5 matches
    // the shadow map's texel layout (Vulkan NDC y-down -> texture row 0 at NDC y=-1) directly.
    float shadow = 1.0;
    {
        float4 lp = mul(f.lightViewProj, float4(i.wpos, 1.0));
        float3 proj = lp.xyz / lp.w;
        float2 smUV = proj.xy * 0.5 + 0.5;
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

    // Ambient is unshadowed; the directional diffuse+spec is multiplied by the shadow factor.
    float3 rgb = tex * (ambient * f.lightColor.rgb)
               + shadow * (tex * (diff * f.lightColor.rgb) + spec * f.lightColor.rgb * 0.4);

    // Accumulate colored point lights with smooth radius-based attenuation.
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
        float  d2 = max(dot(N, Ld), 0.0);
        float3 H2 = normalize(Ld + V);
        float  s2 = pow(max(dot(N, H2), 0.0), 32.0);
        rgb += (tex * d2 + s2 * 0.4) * lc * intensity * att;
    }
    return float4(rgb, 1.0);
}
