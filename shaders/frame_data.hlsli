// Hazard Forge — the SINGLE source of truth for the per-frame uniform block (FrameData).
//
// Every lit/sky/water/etc. shader that reads the camera, the directional light, the point-light
// array, the shadow matrix, or the procedural-sky params used to hand-copy this struct (issue #3 —
// the `float4 ptPos[3]; float4 ptColor[3]` literal was duplicated across ~20+ shaders, and the
// 3-light cap forced point lights to pop in any scene with more than a few). This header replaces
// every one of those hand-copied declarations with a single #include, and names the cap so it can
// be raised in ONE place. Works exactly like pbr_core.hlsli / procedural_sky.hlsli: a plain
// preprocessor #include, so it cross-compiles to BOTH Vulkan-SPIR-V (DXC) and Metal-MSL (glslc /
// hf_gen_msl) with no codegen registration.
//
// IMPORTANT — the CPU-side mirror MUST match this layout byte-for-byte (std140-style: a float4[N]
// array occupies N*16 bytes). The C++ FrameData structs in samples/hello_triangle/main.cpp,
// metal_headless/visual_test.mm, and mac_window/main.mm grow ptPos/ptColor to HF_MAX_POINT_LIGHTS
// in lockstep, so the cbuffer upload and the shader read agree on every field offset.
#ifndef HF_FRAME_DATA_HLSLI
#define HF_FRAME_DATA_HLSLI

// Point-light cap (issue #3). Raised from the old hardcoded 3 to 8 — still well inside the
// uniform-buffer budget (kFrameUboSize = 1024 B; FrameData with 8 lights is 416 B in the shaders),
// no new pipeline class, no new shader variant. The point-light loops are bounded by
// (int)f.ptCount.x at runtime, so scenes that set fewer lights are unaffected (the extra slots are
// zero-initialized and never read).
#define HF_MAX_POINT_LIGHTS 8

struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[HF_MAX_POINT_LIGHTS]; float4 ptColor[HF_MAX_POINT_LIGHTS]; float4x4 lightViewProj;
    // skyParams: x=tanHalfFov, y=aspect, z=time(seconds), w=frameIndex (issue #5 time channel —
    // now uncontested; the IBL maxLod that used to overload skyParams.z moved to iblParams.x below).
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
    // prevViewProj: TAA reprojection matrix. The C++ FrameData carries it after skyParams (offset 512),
    // so it is mirrored here byte-for-byte to keep iblParams at the SAME offset (576) on both sides —
    // even though no frame_data.hlsli consumer reads it (motion_blur / taa_resolve roll their own).
    // iblParams.x = env maxLod for the IBL pass (issue #33): a DEDICATED slot so HDR-IBL objects and an
    // animated sky (which needs skyParams.z=time, issue #5) can coexist in one frame without colliding.
    float4x4 prevViewProj; float4 iblParams;
};

#endif // HF_FRAME_DATA_HLSLI
