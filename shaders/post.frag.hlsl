[[vk::binding(0, 0)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 0)]] SamplerState gSmp : register(s0);
struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

static const float3 kLuma = float3(0.299, 0.587, 0.114);

// Lightweight FXAA (NVIDIA FXAA3 "console" style): detect luma edges from the 4 corner
// neighbours, then blend along the edge direction. Smooths geometry aliasing.
float3 Fxaa(float2 uv, float2 texel) {
    float3 rgbM  = gTex.Sample(gSmp, uv).rgb;
    float3 rgbNW = gTex.Sample(gSmp, uv + float2(-1.0, -1.0) * texel).rgb;
    float3 rgbNE = gTex.Sample(gSmp, uv + float2( 1.0, -1.0) * texel).rgb;
    float3 rgbSW = gTex.Sample(gSmp, uv + float2(-1.0,  1.0) * texel).rgb;
    float3 rgbSE = gTex.Sample(gSmp, uv + float2( 1.0,  1.0) * texel).rgb;

    float lumaM  = dot(rgbM,  kLuma);
    float lumaNW = dot(rgbNW, kLuma);
    float lumaNE = dot(rgbNE, kLuma);
    float lumaSW = dot(rgbSW, kLuma);
    float lumaSE = dot(rgbSE, kLuma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.125, 1.0 / 128.0);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, -8.0, 8.0) * texel;

    float3 rgbA = 0.5 * (gTex.Sample(gSmp, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
                         gTex.Sample(gSmp, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    float3 rgbB = rgbA * 0.5 + 0.25 * (gTex.Sample(gSmp, uv + dir * -0.5).rgb +
                                       gTex.Sample(gSmp, uv + dir *  0.5).rgb);
    float lumaB = dot(rgbB, kLuma);
    return (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
}

float4 main(PSInput i) : SV_Target {
    float w, h;
    gTex.GetDimensions(w, h);
    float2 texel = 1.0 / float2(w, h);

    float3 c = Fxaa(i.uv, texel);              // anti-alias
    c = c / (c + 1.0);                         // Reinhard tonemap
    c = pow(c, 1.0 / 2.2);                     // gamma
    float2 d = i.uv - 0.5;
    float vig = smoothstep(0.8, 0.35, length(d));  // darken corners
    return float4(c * vig, 1.0);
}
