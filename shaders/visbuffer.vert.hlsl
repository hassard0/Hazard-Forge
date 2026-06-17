// Slice DW — visibility-buffer vertex shader. VULKAN-ONLY (uses the SPIR-V DrawIndex builtin), a THIN
// variant of cluster_viz.vert: it drops the Lambert / hash-color output entirely and instead forwards
// the FLAT survivor-draw index (gl_DrawID == the clusterID the visibility buffer packs) as a
// `nointerpolation uint` to the fragment shader. The model matrix is still reassembled from the
// compacted PerDraw[gl_DrawID] record (the SAME survivor draw the DT/DV cluster MDI uses), so the
// rasterized geometry is identical to --cluster-cull — only the per-fragment OUTPUT changes from a
// color to an integer (clusterID, triID) pair.
//
// NOT in the Metal MSL-gen list: glslc/spirv-cross cannot lower the DrawIndex builtin (same as
// cluster_viz.vert). The Metal showcase renders the identical survivors via its per-cluster-instance
// BOUND path (one DrawIndexed per survivor, clusterID pushed as a constant) so the integer IDs match.
struct VSInput {
    [[vk::location(0)]] float3 pos : POSITION;  // only position is needed (no shading -> no normal)
};
struct VSOutput {
    float4 clip : SV_Position;
    // The survivor-draw index (== clusterID). nointerpolation: a flat per-primitive integer, never
    // interpolated across the triangle (interpolating an integer ID would be meaningless).
    [[vk::location(0)]] nointerpolation uint clusterID : CLUSTERID;
};
struct FrameData {
    float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos;
    float4 ptCount; float4 ptPos[3]; float4 ptColor[3]; float4x4 lightViewProj;
    float4 camFwd; float4 camRight; float4 camUp; float4 skyParams;
};

// Per-draw record (matches the cluster_cull.comp.hlsl PerDrawOut byte layout exactly: column-major
// mat4 + float4 color = 80 bytes, std430). DW ignores `color` (no shading) but keeps the layout so the
// SAME compacted PerDraw SSBO the cluster cull writes is reused verbatim.
struct PerDraw {
    float4 modelCol0;
    float4 modelCol1;
    float4 modelCol2;
    float4 modelCol3;
    float4 color;  // ignored by the visibility pass
};

[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
[[vk::binding(0, 2)]] StructuredBuffer<PerDraw> gPerDraw;

VSOutput main(VSInput i, [[vk::builtin("DrawIndex")]] uint drawId : DRAWID) {
    VSOutput o;
    PerDraw pd = gPerDraw[drawId];
    // Reassemble the column-major model matrix from the four per-draw columns (identical to
    // cluster_viz.vert / lit_gpudriven.vert so the rasterized geometry matches bit-for-bit).
    float4x4 model = float4x4(
        float4(pd.modelCol0.x, pd.modelCol1.x, pd.modelCol2.x, pd.modelCol3.x),
        float4(pd.modelCol0.y, pd.modelCol1.y, pd.modelCol2.y, pd.modelCol3.y),
        float4(pd.modelCol0.z, pd.modelCol1.z, pd.modelCol2.z, pd.modelCol3.z),
        float4(pd.modelCol0.w, pd.modelCol1.w, pd.modelCol2.w, pd.modelCol3.w));

    float4 world = mul(model, float4(i.pos, 1.0));
    o.clip = mul(f.viewProj, world);
    o.clusterID = drawId;  // the survivor-draw index = the clusterID the visibility buffer packs
    return o;
}
