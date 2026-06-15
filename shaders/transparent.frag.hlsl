// Transparent ("glass") fragment shader (Slice T). Self-contained glassy translucent surface — it
// does NOT depend on the PBR/IBL material sets and binds NO textures. It produces a tinted RGB from a
// simplified lighting model (directional half-Lambert diffuse + a Blinn-Phong specular highlight + a
// procedural SkyColor reflection) and a Fresnel-style alpha that rises at grazing angles so edges read
// as more opaque than the head-on center. Pair with the alphaBlend + depthWrite=false pipeline so it
// blends OVER the opaque scene while still depth-testing against it.
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };

struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 wnormal : NORMAL;
    [[vk::location(1)]] float3 wpos    : POSITION0;
    [[vk::location(2)]] nointerpolation float4 tintAlpha : TEXCOORD0; // rgb = tint, w = base alpha
};

// Procedural sky color for a world-space direction (same gradient + sun glow as sky.frag.hlsl /
// lit.frag.hlsl SkyColor). Copied locally so the transparent shader stays self-contained.
float3 SkyColor(float3 dir) {
    float3 d = normalize(dir);
    float  h       = saturate(d.y * 0.5 + 0.5);
    float3 zenith  = float3(0.18, 0.30, 0.62);
    float3 horizon = float3(0.65, 0.72, 0.82);
    float3 sky     = lerp(horizon, zenith, pow(h, 0.8));
    float3 ground  = float3(0.12, 0.11, 0.10);
    if (d.y < 0.0) {
        float g = saturate(-d.y * 2.0);
        sky = lerp(sky, ground, g);
    }
    float3 sunDir = normalize(-f.lightDir.xyz);
    float  s = pow(max(dot(d, sunDir), 0.0), 256.0);
    sky += float3(1.0, 0.95, 0.8) * s * 2.0;
    return sky;
}

float4 main(PSInput i) : SV_Target {
    float3 N = normalize(i.wnormal);
    float3 V = normalize(f.viewPos.xyz - i.wpos);
    // Make the lit hemisphere face the camera (double-sided glass: back faces would otherwise shade
    // dark). Flip the normal toward the viewer so both faces are lit consistently.
    if (dot(N, V) < 0.0) N = -N;

    float3 tint      = i.tintAlpha.rgb;
    float  baseAlpha = i.tintAlpha.w;

    float3 L = normalize(-f.lightDir.xyz);

    // Half-Lambert diffuse: soft wrap so the tint stays readable even in shadow-facing areas.
    float ndl  = dot(N, L);
    float diff = ndl * 0.5 + 0.5;
    diff = diff * diff;

    // Blinn-Phong specular highlight (sharp, glassy).
    float3 H    = normalize(L + V);
    float  spec = pow(max(dot(N, H), 0.0), 96.0);

    // Procedural sky reflection in the mirror direction (the "through the glass you see sky" sheen).
    float3 R       = reflect(-V, N);
    float3 skyRefl = SkyColor(R);

    // Fresnel term: grazing angles reflect more sky AND become more opaque (glass rim).
    float  NoV     = saturate(dot(N, V));
    float  fresnel = pow(1.0 - NoV, 5.0);

    float3 ambient = tint * 0.10;
    float3 body    = tint * (f.lightColor.rgb * diff + ambient);
    float3 rgb     = body + skyRefl * (0.15 + 0.6 * fresnel) + spec * f.lightColor.rgb;

    // Fresnel-style alpha: head-on stays at baseAlpha (see-through), grazing edges approach 1.0.
    float alpha = lerp(baseAlpha, 1.0, fresnel);

    return float4(rgb, alpha);
}
