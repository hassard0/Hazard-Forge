// Fullscreen sky triangle from SV_VertexID (same pattern as post.vert); passes uv (0..1).
struct VSOutput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };
VSOutput main(uint vid : SV_VertexID) {
    VSOutput o;
    float2 p = float2((vid << 1) & 2, vid & 2);   // (0,0),(2,0),(0,2)
    o.uv = p;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);       // covers clip space with a big triangle
    return o;
}
