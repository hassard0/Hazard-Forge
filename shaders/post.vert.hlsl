struct VSOutput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
VSOutput main(uint vid : SV_VertexID) {
    VSOutput o;
    float2 p = float2((vid << 1) & 2, vid & 2);   // (0,0),(2,0),(0,2)
    // Render-target sampling V origin differs between backends. The lit pass renders the scene into
    // the RT right-side-up in each backend's NDC; Vulkan's texture origin (top-left, V down) makes
    // uv = p sample correctly there, but Metal's RT texture stores row 0 = top with V up, so the
    // Metal path samples with V flipped. This is the ONE backend-specific line in the post pass; it
    // is guarded so the shared HLSL still produces the correct Vulkan image. (NDC Y for the geometry
    // passes is handled CPU-side in the Metal backend; this is purely a texture-origin difference.)
#ifdef HF_MSL_GEN
    o.uv = float2(p.x, 1.0 - p.y);                 // Metal: flip V (texture row 0 = top)
#else
    o.uv = p;                                      // Vulkan
#endif
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);       // covers clip space with a big triangle
    return o;
}
