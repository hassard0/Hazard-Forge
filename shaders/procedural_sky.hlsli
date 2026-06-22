// Hazard Forge — shared PROCEDURAL SKY (the single source of truth for the clear-day gradient).
//
// This factors the procedural sky gradient (horizon->zenith lerp + dim ground haze + a directional
// sun glow) out of the THREE places it used to be hand-duplicated — sky.frag.hlsl (the sky pass),
// lit.frag.hlsl's SkyColor() (the IBL reflection metals/glass reflect), and pbr_core.hlsli's
// hfSkyColor() (the data-driven material graph's IBL) — into ONE HFSkyColor(dir, lightDir) function.
//
// WHY IT MATTERS: the sky PASS and the lit pass's IBL must stay in sync. SkyColor(R) is what every
// metallic / glass surface reflects; if the sky look changes but the IBL copy doesn't, every metallic
// building reflects the OLD sky. With this header there is one edit point: retune HFSkyColor here and
// BOTH the sky dome and the IBL reflection update together (fixes issue #4).
//
// CROSS-COMPILE: included exactly like pbr_core.hlsli — a plain `#include` resolved relative to the
// including .hlsl. The Vulkan path (DXC) and the Metal path (glslc -> spirv-cross, hf_gen_msl) both
// preprocess the include before lowering, so NO new compile entry / CMake registration is needed; the
// include just has to sit next to the shaders that pull it in (shaders/). Pure refactor: the math here
// is byte-for-byte the computation that was inline, so every existing golden stays bit-identical.
//
// The `dir` is the world-space lookup direction (the sky pass passes the view ray; the lit/material IBL
// passes pass the reflection vector R, the up vector, and the surface normal N — each call site keeps
// its own argument). `lightDir` is FrameData.lightDir.xyz (the INCOMING directional-light direction);
// the sun glow keys off normalize(-lightDir).
#ifndef HF_PROCEDURAL_SKY_HLSLI
#define HF_PROCEDURAL_SKY_HLSLI

float3 HFSkyColor(float3 dir, float3 lightDir) {
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
    float3 sunDir = normalize(-lightDir);
    float  s = pow(max(dot(d, sunDir), 0.0), 256.0);
    sky += float3(1.0, 0.95, 0.8) * s * 2.0;
    return sky;
}

#endif  // HF_PROCEDURAL_SKY_HLSLI
