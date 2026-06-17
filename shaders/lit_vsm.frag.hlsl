// Slice VC — Virtual Shadow Maps Slice 3: the LIT-PASS VSM INDIRECTION SAMPLE (the shadowed scene). A
// SIBLING of lit_csm.frag.hlsl (NOT an edit of lit.frag/lit_csm.frag — those + their goldens stay
// BYTE-IDENTICAL, exactly as lit_clustered/lit_csm/lit_ddgi isolate their variants). Identical lighting to
// lit.frag, but the directional shadow is sampled from the VIRTUAL SHADOW MAP via the indirection table:
//   * per receiver pixel: clipmap level = SelectClipmapLevel(length(wpos - cameraPos)) (the VA integer
//     threshold-ladder, copied VERBATIM from render/vsm.h — NO transcendental, bit-exact GPU==CPU),
//   * project wpos -> the level's top-down clipmap page (px,py) (the VA MarkPage projection, verbatim),
//   * pageId = PageId(level,px,py); tile = gIndirection[pageId] (the VB virtual->physical SSBO),
//   * if tile == kNoTile -> page not resident -> shadow = 1.0 (lit; documented),
//   * else project wpos into THAT page's light-space (PageWorldOrtho's ortho — the SAME ortho the VB depth
//     pass rendered the page depth with) -> lightUV + receiverDepth, sample the physical tile in the atlas
//     with tile-clamped 3x3 PCF (SampleCascadePCF adapted: the tile origin comes from PhysicalTileOrigin,
//     not a cascade index).
//
// THE make-or-break NO-OP: shadow = lerp(1.0, shadow, gVsm.vsmEnabled). With vsmEnabled==0 this is the
// EXACT float 1.0 (lerp(1,x,0)==1 bit-exact) -> the render is BYTE-IDENTICAL to the same scene through THIS
// shader with shadowing off == the unshadowed lit (the showcase's frame-A proof). rgb = directLighting *
// shadow + ambient (shadow modulates ONLY the direct term, like lit_csm).
//
// THE DEPTH ENCODE/DECODE (documented so the compare is correct): VC binds the VSM atlas as a DEPTH-only
// shadow map (CreateShadowMap), rendered by the depth-only shadow pass (shadow_csm.vert + the depth-only
// pipeline) with each resident page's PageWorldOrtho VP. So gShadow stores the PLAIN post-divide NDC depth
// SV_Position.z in [0,1] (near=small, far=large) — the SAME convention lit_csm samples, NOT the VB
// vsm_depth.frag grayscale `1-z` color encode (that 1-z trick was VB's readback-golden colorization only).
// Therefore the compare is the standard depth test: `receiverDepth - bias > storedDepth` -> the receiver is
// BEHIND the nearest caster depth -> SHADOWED (return 0), else LIT (return 1). No 1-z reconstruction here.
//
// THE INDIRECTION SSBO (NO new RHI): gIndirection rides the EXISTING fragment-storage path —
// [[vk::binding(13,3)]] StructuredBuffer<uint> -> bound via usesLightClusters / BindLightClusters(*indir,
// *dummy, *dummy) (the DX/DN single-SSBO-via-dummies idiom; spirv-cross maps it to Metal fragment buffer
// 13). The clipmap + atlas params + vsmEnabled ride the showcase's OWN FrameData UBO. NO new RHI symbol.
//
// THE DETERMINISM CRUX (mirrors render/vsm.h verbatim): the level/page/tile selection is pure INTEGER
// compare/subtract/divide/floor (no log2) -> bit-identical CPU<->Vulkan<->Metal (the GPU==CPU page-lookup
// proof). The light-space transform + PCF reuse lit_csm.frag's cross-backend-stable math + the Metal V-flip
// behind HF_MSL_GEN. mad/fma for the FP accumulation (the DH cross-backend FP discipline).

// VSM FrameData layout (own struct; fits kFrameUboSize=1024). The --vsm-sample-shot showcase fills the
// per-frame UBO with THIS layout. lit.vert reads viewProj at offset 0 (the standard lit vertex contract).
struct FrameData {
    float4x4 viewProj;     //   0  (lit.vert reads this at offset 0)
    float4   lightDir;     //  64  directional sun
    float4   lightColor;   //  80
    float4   viewPos;      //  96  world-space camera position (for V)
    float4   vsmCamera;    // 112  x,y,z = clipmap cameraPos; w = level0WorldExtent
    float4   vsmClip;      // 128  x=levels, y=virtualPagesPerSide, z=atlas.tilesPerSide, w=atlas.tileSize
    float4   vsmCtrl;      // 144  x=vsmEnabled (0/1), y=atlasSize(px), z=lightHeight, w=ortho(near|far)
    float4   vsmOrtho;     // 160  x=orthoNear, y=orthoFar (the SAME near/far the page depth was rendered with)
};
[[vk::binding(0, 0)]] cbuffer Frame { FrameData f; };
// Shadow map (the VSM physical atlas, DEPTH-only) lives in the per-frame set (set 0): binding 1 = depth
// image, binding 2 = sampler. Bound via SetShadowMap. gShadow.Sample(...).r = stored NDC depth.
[[vk::binding(1, 0)]] Texture2D    gShadow    : register(t1);
[[vk::binding(2, 0)]] SamplerState gShadowSmp : register(s1);
[[vk::binding(0, 1)]] Texture2D    gTex : register(t0);
[[vk::binding(1, 1)]] SamplerState gSmp : register(s0);
[[vk::binding(3, 1)]] Texture2D    gNormalMap : register(t3);
[[vk::binding(4, 1)]] SamplerState gNormalSmp : register(s3);

// The VB virtual->physical indirection table: gIndirection[pageId] = physical tile index (or kNoTile).
// Bound to the FRAGMENT stage via the cluster set (binding 13/space3, like lit_ddgi's gProbeSH) ->
// spirv-cross --msl-decoration-binding maps it to Metal fragment buffer slot 13. uint per page.
[[vk::binding(13, 3)]] StructuredBuffer<uint> gIndirection : register(t13, space3);

static const uint kNoTile = 0xFFFFFFFFu;   // == render::vsm::kNoTile (non-resident / overflowed page).

struct PSInput {
    float4 clip      : SV_Position;
    [[vk::location(0)]] float3 color  : COLOR;
    [[vk::location(1)]] float2 uv     : TEXCOORD0;
    [[vk::location(2)]] float3 wnormal: NORMAL;
    [[vk::location(3)]] float3 wpos    : POSITION0;
    [[vk::location(4)]] nointerpolation float2 material : TEXCOORD1; // x=metallic, y=roughness
    [[vk::location(5)]] float3 wtangent : TANGENT;
};
static const float HF_PI = 3.14159265358979323846;

float hfDistributionGGX(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float d  = (NoH * NoH) * (a2 - 1.0) + 1.0;
    return a2 / max(HF_PI * d * d, 1e-7);
}
float hfGeometrySchlickGGX(float NoX, float k) {
    return NoX / (NoX * (1.0 - k) + k);
}
float hfGeometrySmith(float NoV, float NoL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return hfGeometrySchlickGGX(NoV, k) * hfGeometrySchlickGGX(NoL, k);
}
float3 hfFresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (float3(1.0, 1.0, 1.0) - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}
float3 SkyColor(float3 dir) {
    float3 d = normalize(dir);
    float  h       = saturate(d.y * 0.5 + 0.5);
    float3 zenith  = float3(0.18, 0.30, 0.62);
    float3 horizon = float3(0.65, 0.72, 0.82);
    float3 sky     = lerp(horizon, zenith, pow(h, 0.8));
    float3 ground = float3(0.12, 0.11, 0.10);
    if (d.y < 0.0) {
        float g = saturate(-d.y * 2.0);
        sky = lerp(sky, ground, g);
    }
    float3 sunDir = normalize(-f.lightDir.xyz);
    float  s = pow(max(dot(d, sunDir), 0.0), 256.0);
    sky += float3(1.0, 0.95, 0.8) * s * 2.0;
    return sky;
}
float3 hfCookTorrance(float3 N, float3 V, float3 L, float3 radiance,
                      float3 albedo, float metallic, float roughness, float3 F0) {
    float3 H   = normalize(L + V);
    float  NoV = max(dot(N, V), 1e-4);
    float  NoL = max(dot(N, L), 0.0);
    float  NoH = max(dot(N, H), 0.0);
    float  VoH = max(dot(V, H), 0.0);
    float  alpha = roughness * roughness;
    float  D = hfDistributionGGX(NoH, alpha);
    float  G = hfGeometrySmith(NoV, NoL, roughness);
    float3 F = hfFresnelSchlick(VoH, F0);
    float3 spec = (D * G) * F / max(4.0 * NoV * NoL, 1e-4);
    float3 kd = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);
    float3 diff = kd * albedo / HF_PI;
    return (diff + spec) * radiance * NoL;
}

// ============================== VSM page lookup (VERBATIM render/vsm.h) ==============================
// SelectClipmapLevel: the level is the NUMBER of host thresholds (level0WorldExtent*2^L) distToCamera
// EXCEEDS, clamped to [0, levels-1]. Pure compare/count -> bit-identical CPU<->GPU (vsm.h:100-110). The
// threshold level0WorldExtent*2^L is an exact float32 power-of-two scale (transcendental-free), == the
// CPU BuildLevelThresholds bits.
int SelectClipmapLevel(float distToCamera, float level0WorldExtent, int levels) {
    int level = 0;
    [unroll] for (int L = 0; L < 16; ++L) {
        if (L < levels) {
            float threshold = level0WorldExtent * (float)(1u << (uint)L);
            if (distToCamera > threshold) level = L + 1;
        }
    }
    if (level < 0) level = 0;
    if (level > levels - 1) level = levels - 1;
    return level;
}

// PageId(level,px,py) = level*(vpps*vpps) + py*vpps + px (vsm.h:67-70, the flat-index discipline).
int PageId(int level, int px, int py, int vpps) { return level * (vpps * vpps) + py * vpps + px; }

float4 main(PSInput i) : SV_Target {
    float3 Ng = normalize(i.wnormal);
    float3 T = normalize(i.wtangent - Ng * dot(Ng, i.wtangent));
    float3 B = cross(Ng, T);
    float3x3 TBN = float3x3(T, B, Ng);
    float3 nTS = gNormalMap.Sample(gNormalSmp, i.uv).xyz * 2.0 - 1.0;
    float3 N = normalize(mul(nTS, TBN));
    float3 V = normalize(f.viewPos.xyz - i.wpos);
    float3 tex = gTex.Sample(gSmp, i.uv).rgb * i.color;

    float3 albedo   = tex;
    float  metallic = saturate(i.material.x);
    float  roughness = clamp(i.material.y, 0.05, 1.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // --- VSM clipmap / atlas params (unpacked from the UBO; the same ints the CPU ref uses). ---
    float3 vsmCam    = f.vsmCamera.xyz;
    float  level0    = f.vsmCamera.w;
    int    levels    = (int)f.vsmClip.x;
    int    vpps      = (int)f.vsmClip.y;
    int    tps       = (int)f.vsmClip.z;          // atlas tilesPerSide
    int    tileSize  = (int)f.vsmClip.w;          // atlas tileSize (texels)
    float  atlasSize = f.vsmCtrl.y;               // atlas size (px) == tps*tileSize
    float  lightH    = f.vsmCtrl.z;               // light eye height above the page center
    float  orthoNear = f.vsmOrtho.x;
    float  orthoFar  = f.vsmOrtho.y;

    // ===================== THE VSM SHADOW FACTOR =====================
    float shadow = 1.0;
    {
        // 1) Clipmap level via the VA threshold-ladder (integer, bit-exact).
        float distCam = length(i.wpos - vsmCam);
        int   level   = SelectClipmapLevel(distCam, level0, levels);

        // 2) Project wpos into the level's top-down clipmap page (VERBATIM MarkPage projection): origin =
        //    floor((cameraPos - levelExtent/2)/pageWorldSize)*pageWorldSize; (px,py) = floor((worldXZ -
        //    origin)/pageWorldSize). World X -> px, world Z -> py (the directional clipmap ground axes).
        float levelExtent   = level0 * (float)(1u << (uint)level);
        float pageWorldSize = levelExtent / (float)vpps;
        float originX = floor((vsmCam.x - levelExtent * 0.5) / pageWorldSize) * pageWorldSize;
        float originZ = floor((vsmCam.z - levelExtent * 0.5) / pageWorldSize) * pageWorldSize;
        int   px = (int)floor((i.wpos.x - originX) / pageWorldSize);
        int   py = (int)floor((i.wpos.z - originZ) / pageWorldSize);
        // Clamp to the page grid (MarkPage's edge clamp).
        px = clamp(px, 0, vpps - 1);
        py = clamp(py, 0, vpps - 1);

        // 3) pageId -> physical tile via the indirection table.
        int  pageId = PageId(level, px, py, vpps);
        uint tile   = gIndirection[pageId];

        if (tile != kNoTile) {
            // 4) Project wpos into THAT page's light-space — the SAME PageWorldOrtho the depth pass used.
            //    Page center = origin + (page + 0.5)*pageWorldSize (mad, DH). Light looks straight down
            //    (-Y) from lightH above, +Z 'up'; ortho half-width = pageWorldSize/2, near/far = the VB
            //    render's near/far. We project in light-space the SAME way Mat4::LookAt+Ortho would, but
            //    inline (axis-aligned top-down ortho): u = (wx - cx)/half, v = (wz - cz)/half in [-1,1],
            //    NDC depth = (lightH - wy - near)/(far - near) (linear ortho: viewZ = lightH - wy, mapped
            //    [near,far]->[0,1]). lightUV = u,v*0.5+0.5.
            float cx = mad((float)px + 0.5, pageWorldSize, originX);
            float cz = mad((float)py + 0.5, pageWorldSize, originZ);
            float half = pageWorldSize * 0.5;

            float u = (i.wpos.x - cx) / half;          // [-1,1] across the page
            float v = (i.wpos.z - cz) / half;
            float2 lightUV = float2(u, v) * 0.5 + 0.5; // [0,1]

            // Linear ortho depth: the light eye is at y = vsmCam.y + lightH looking down; view-space Z (to
            // the eye) of a point at world y is (eyeY - wy). The VB ortho maps [near,far] -> [0,1].
            float eyeY = vsmCam.y + lightH;
            float viewZ = eyeY - i.wpos.y;
            float receiverDepth = (viewZ - orthoNear) / (orthoFar - orthoNear);

#ifdef HF_MSL_GEN
            // Metal NDC +Y up: the VB depth pass flipped the page ortho's clip-space Y (FlipProjY), so the
            // stored tile is V-flipped relative to the Vulkan layout. Flip V to hit the matching texel —
            // the SAME self-consistent RENDER/SAMPLE flip lit_csm uses (vsm-sample render flips too).
            lightUV.y = 1.0 - lightUV.y;
#endif

            // 5) Tile-clamped 3x3 PCF (SampleCascadePCF adapted: the tile origin comes from the indirection
            //    table's physical tile, atlas-pixel -> UV; the kernel is clamped to the tile so it never
            //    bleeds into a neighbour page's depth). col = tile % tps, row = tile / tps.
            int   col = (int)tile % tps;
            int   row = (int)tile / tps;
            float tileScale = (float)tileSize / atlasSize;   // a tile's UV size in the atlas
            float2 tileOrigin = float2((float)col, (float)row) * tileScale;
            float2 atlasUV = tileOrigin + lightUV * tileScale;
            float  atlasTexel = 1.0 / atlasSize;
            float2 tileMin = tileOrigin + atlasTexel;
            float2 tileMax = tileOrigin + tileScale - atlasTexel;
            float  bias = 0.0025;

            // Only sample if the receiver projects inside the page (else leave lit).
            if (lightUV.x >= 0.0 && lightUV.x <= 1.0 && lightUV.y >= 0.0 && lightUV.y <= 1.0 &&
                receiverDepth >= 0.0 && receiverDepth <= 1.0) {
                float s = 0.0;
                [unroll] for (int sx = -1; sx <= 1; ++sx)
                [unroll] for (int sy = -1; sy <= 1; ++sy) {
                    float2 tap = clamp(atlasUV + float2(sx, sy) * atlasTexel, tileMin, tileMax);
                    float storedDepth = gShadow.Sample(gShadowSmp, tap).r;   // plain NDC depth (near=small)
                    // receiver BEHIND the nearest caster depth -> shadowed. (Standard depth test, NO 1-z.)
                    s += (receiverDepth - bias > storedDepth) ? 0.0 : 1.0;
                }
                shadow = s / 9.0;
            }
        }
        // tile == kNoTile -> page not resident -> shadow stays 1.0 (lit; documented YAGNI fallback).
    }

    // THE make-or-break NO-OP: vsmEnabled==0 -> lerp(1, shadow, 0) == 1.0 EXACTLY -> unshadowed lit.
    shadow = lerp(1.0, shadow, f.vsmCtrl.x);

    // --- Lighting (IDENTICAL to lit_csm.frag; shadow modulates ONLY the direct radiance). ---
    float3 rgb = albedo * 0.03;
    {
        float3 L = normalize(-f.lightDir.xyz);
        float3 radiance = f.lightColor.rgb * shadow;
        rgb += hfCookTorrance(N, V, L, radiance, albedo, metallic, roughness, F0);
    }
    {
        float3 R = reflect(-V, N);
        float  NoV = max(dot(N, V), 0.0);
        float3 F   = F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - NoV, 5.0);
        float3 envColor = SkyColor(R);
        envColor = lerp(envColor, SkyColor(float3(0.0, 1.0, 0.0)), roughness * 0.7);
        float3 iblSpecular = envColor * F;
        rgb += iblSpecular;
        rgb += (1.0 - metallic) * albedo * SkyColor(N) * 0.15;
    }
    return float4(rgb, 1.0);
}
