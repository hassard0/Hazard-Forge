// Slice W — immediate-mode debug-line fragment shader. Outputs the interpolated per-vertex color
// at full opacity. No textures, no lighting (debug overlays are flat, self-evident colors).
struct PSInput {
    float4 pos   : SV_Position;
    float3 color : COLOR;
};
float4 main(PSInput i) : SV_Target {
    return float4(i.color, 1.0);
}
