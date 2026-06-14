// Particle fragment: emissive point, slightly boosted so the additive draw glows over the scene.
struct PSInput {
    float4 pos : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};

float4 main(PSInput i) : SV_Target {
    return float4(i.color * 1.4, 1.0);
}
