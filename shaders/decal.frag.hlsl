// Screen-space projected decals — composite + tonemap (Slice BH). The FINAL pass for the --decal-shot
// showcase. For each pixel it reconstructs the VIEW-SPACE position from the g-buffer (the SAME g-buffer
// SSR/SSAO read: view normal.xyz + linear depth.w) EXACTLY as ssr.frag/ssao.frag do (ReconstructViewPos
// + the yFlip convention), then maps view->world with the camera's view->world matrix (invView). The
// world position is transformed into the decal box's LOCAL space (worldToDecal, the inverse of
// engine/render/decal.h BuildDecalTransform): if it is InsideUnitBox (|xyz|<=0.5) the decal is sampled
// at DecalUV (top-down along local -Y: uv = local.xz + 0.5), alpha = decalCoverage * EdgeFade, and
// outColor = lerp(scene, decalAlbedo*decalRGB, alpha). Otherwise the scene passes through. Then the SAME
// displayed-image pipeline as post.frag/ssr_composite (exposure, ACES, gamma, grade, grain, vignette) is
// applied and the LDR swapchain is written. Background pixels (g-buffer w==0, no geometry) are EXCLUDED
// so the decal never bleeds onto the sky. The projection math is pinned by tests/decal_test.
//
// DECAL TEXTURE: a PROCEDURAL crack/cross computed in-shader from the decal UV (no extra texture bind),
// so the pass needs ONLY the lit scene + g-buffer — bound exactly as the SSR composite binds them
// (BindTexturePair: scene at t0/s0, g-buffer at t3/s3). This keeps the RHI seam at baseline (no third
// texture slot, no new bind path). The pattern is a bright "cross + diagonal crack" mask: visibly a
// decal, deterministic, byte-stable across runs and backends.
//
// Existing gbuffer/lit/ssao/ssr shaders + goldens are UNTOUCHED.
[[vk::binding(0, 0)]] Texture2D    gScene : register(t0);   // lit HDR scene color
[[vk::binding(1, 0)]] SamplerState gSmp   : register(s0);
[[vk::binding(3, 0)]] Texture2D    gGbuf  : register(t3);   // view normal.xyz + linDepth.w
[[vk::binding(4, 0)]] SamplerState gGSmp  : register(s3);

struct DecalParams {
    float2 texel;        // 1/size
    float  tanHalfFovY;  // view-space reconstruction (matches ssr.frag)
    float  aspect;       // width/height
    float4 albedo;       // decal albedo TINT (rgb); a unused
    float4 fadeIntensity;// x = edge fade band (local units), y = exposure intensity, zw pad
    float4x4 worldToDecal; // world -> decal-local (inverse of BuildDecalTransform)
    float4x4 invView;      // view -> world (camera's inverse view matrix)
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer DecalPC { DecalParams dp; };
#define HF_DP dp
#else
[[vk::push_constant]] struct { DecalParams p; } pc;
#define HF_DP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
static const float3 kLuma = float3(0.299, 0.587, 0.114);

// Y-flip sign mapping screen UV.y <-> view-space Y — IDENTICAL convention to ssr.frag/ssao.frag.
// Vulkan: -1 (projection bakes a Y-flip + post.vert V-down UV); Metal: +1 (FlipProjY + post.vert V-flip).
#ifdef HF_MSL_GEN
static const float HF_YS = 1.0;
#else
static const float HF_YS = -1.0;
#endif

// Reconstruct a VIEW-SPACE position from a UV + linear depth — IDENTICAL to ssr.frag.
float3 ReconstructViewPos(float2 uv, float linDepth) {
    float2 ndc = uv * 2.0 - 1.0;
    float vx = ndc.x * (HF_DP.aspect * HF_DP.tanHalfFovY) * linDepth;
    float vy = HF_YS * ndc.y * (HF_DP.tanHalfFovY) * linDepth;
    float vz = -linDepth;                   // RH view space: -Z forward
    return float3(vx, vy, vz);
}

// engine/render/decal.h equivalents (the unit test pins these on the CPU side).
bool   InsideUnitBox(float3 l) { return all(abs(l) <= 0.5); }
float2 DecalUV(float3 l)       { return float2(l.x + 0.5, l.z + 0.5); }   // top-down along local -Y
float  EdgeFade(float3 l, float fade) {
    if (fade <= 0.0) return 1.0;
    float dist = 0.5 - max(abs(l.x), max(abs(l.y), abs(l.z)));
    return smoothstep(0.0, 1.0, saturate(dist / fade));
}

// Procedural decal coverage (alpha) from the [0,1]^2 decal UV: a centered cross (a thick + sign) plus a
// diagonal crack streak, with a soft circular vignette so it reads as a placed decal rather than a tile.
// Returns coverage in [0,1]. Deterministic; no texture fetch.
float DecalCoverage(float2 uv) {
    float2 c = uv - 0.5;                         // centered [-0.5,0.5]
    // Cross: a horizontal + vertical bar through the centre.
    float bar = max(smoothstep(0.10, 0.06, abs(c.x)),
                    smoothstep(0.10, 0.06, abs(c.y)));
    // Diagonal crack: distance to the line y = x.
    float crack = smoothstep(0.06, 0.02, abs(c.x - c.y));
    float pattern = max(bar, crack);
    // Soft circular boundary so the cross/crack fade before the box edge.
    float disc = smoothstep(0.5, 0.42, length(c));
    return saturate(pattern * disc);
}

float3 ACES(float3 x) {
    float a = 2.51, b = 0.03, c2 = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c2 * x + d) + e));
}
float3 ColorGrade(float3 c) {
    float luma = dot(c, kLuma);
    const float3 shadowTint = float3(0.90, 1.00, 1.06);
    const float3 highTint   = float3(1.06, 1.00, 0.92);
    const float  gradeAmt   = 0.08;
    float3 tint = lerp(shadowTint, highTint, luma);
    c = lerp(c, c * tint, gradeAmt);
    c = lerp(c, smoothstep(0.0, 1.0, c), 0.15);
    float gluma = dot(c, kLuma);
    c = lerp(gluma.xxx, c, 1.08);
    return c;
}
float3 FilmGrain(float3 c, float2 uv, float2 res) {
    float n = frac(sin(dot(uv * res, float2(12.9898, 78.233))) * 43758.5453);
    float grain = (n - 0.5) * 2.0 * 0.025;
    float luma = dot(c, kLuma);
    grain *= 1.0 - smoothstep(0.5, 1.0, luma);
    return c + grain;
}

float4 main(PSInput i) : SV_Target {
    float w, h;
    gScene.GetDimensions(w, h);

    float3 scene = gScene.Sample(gSmp, i.uv).rgb;   // linear HDR lit scene
    float4 g     = gGbuf.Sample(gGSmp, i.uv);
    float  linDepth = g.w;

    float3 c = scene;
    // Only project the decal onto real geometry; background (g-buffer w==0, the sky) is EXCLUDED so the
    // decal never bleeds onto it.
    if (linDepth > 0.0001) {
        float3 vpos  = ReconstructViewPos(i.uv, linDepth);
        float3 world = mul(HF_DP.invView, float4(vpos, 1.0)).xyz;       // view -> world
        float3 local = mul(HF_DP.worldToDecal, float4(world, 1.0)).xyz; // world -> decal-local
        if (InsideUnitBox(local)) {
            float2 duv  = DecalUV(local);
            float coverage = DecalCoverage(duv);
            float alpha = coverage * EdgeFade(local, HF_DP.fadeIntensity.x);
            c = lerp(scene, HF_DP.albedo.rgb, alpha);
        }
    }

    c *= HF_DP.fadeIntensity.y;                      // exposure, matches post.frag/ssr_composite
    c = ACES(c);
    c = pow(c, 1.0 / 2.2);
    c = ColorGrade(c);
    c = FilmGrain(c, i.uv, float2(w, h));

    float2 dd = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(dd));
    return float4(saturate(c * vig), 1.0);
}
