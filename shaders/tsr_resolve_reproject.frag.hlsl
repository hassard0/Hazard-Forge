// Temporal SUPER-RESOLUTION resolve with MOVING-CAMERA history reprojection + disocclusion (Slice US3,
// issue #20). A fork of tsr_resolve.frag (US2): identical running-average supersample + neighborhood
// clamp, but the FULL-res history is no longer sampled at the identity UV. Under a moving (orbiting)
// camera the history must be REPROJECTED through the camera motion so it lines up with the current
// frame — otherwise it ghosts. This pass copies motion_blur.frag's analytic camera-motion reprojection
// VERBATIM (the MbParams prevClip0..3 == prevViewProj * inverse(curView), the HF_YS Y-flip, the
// view-pos reconstruction from linear depth, and the prevClip.xy/prevClip.w -> prevNdc -> prevUV
// convention incl. the prevUV.y flip). A pixel whose history reprojects OFF-SCREEN is DISOCCLUDED: we
// take the fresh current sample (alpha=1 locally) instead of a stale/garbage off-screen history. The
// inherited 3x3 neighborhood clamp suppresses the residual ghosting on reprojected-but-drifted history.
//
// New inputs vs US2: the history is now a PACKED FULL-res RGBA16F (rgb = accumulated history color, a =
// the current-frame G-buffer LINEAR depth, positive in front of the camera — folded in by the
// tsr_pack_histdepth pass so this stays within the RHI's 2-sampled-image material set, NO RHI change) +
// tanHalfFovY + aspect + prevClip0..3. NEW shader (fork of tsr_resolve.frag + the motion_blur
// reprojection copied verbatim); tsr_resolve.frag / taa_resolve.frag / taa.h / motion_blur.frag and all
// existing goldens UNTOUCHED.
[[vk::binding(0, 0)]] Texture2D    gCurrent : register(t0);   // current HALF-res jittered HDR scene
[[vk::binding(1, 0)]] SamplerState gSmp     : register(s0);
[[vk::binding(3, 0)]] Texture2D    gHistory : register(t3);   // FULL-res PACKED history.rgb + linDepth.a
[[vk::binding(4, 0)]] SamplerState gHSmp    : register(s3);

// US2's curTexel/histTexel/alpha/firstFrame, PLUS the motion_blur reprojection params (prevClip0..3 =
// prevViewProj * inverse(curView), a column-major mat4 as 4 float4s — IDENTICAL to motion_blur's
// MbParams — + tanHalfFovY + aspect for the view-pos reconstruction).
struct TsrParams {
    float2 curTexel; float2 histTexel; float alpha; float firstFrame;
    float  tanHalfFovY; float aspect;
    float4 prevClip0;    // prevViewProj * inverse(curView), column 0
    float4 prevClip1;    // column 1
    float4 prevClip2;    // column 2
    float4 prevClip3;    // column 3
};
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer TsrPC { TsrParams tp; };
#define HF_TP tp
#else
[[vk::push_constant]] struct { TsrParams p; } pc;
#define HF_TP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

// Y-flip sign mapping screen UV.y <-> view-space Y — IDENTICAL convention to motion_blur.frag /
// ssr.frag / ssao.frag (1.0 on Vulkan / -1.0 wait: motion_blur is -1.0 Vulkan / +1.0 Metal). Copy
// motion_blur's exact #ifdef.
#ifdef HF_MSL_GEN
static const float HF_YS = 1.0;
#else
static const float HF_YS = -1.0;
#endif

float4 main(PSInput i) : SV_Target {
    float3 current = gCurrent.Sample(gSmp, i.uv).rgb;

    // First frame (empty history): output the current frame unblended so accumulation has a defined
    // start (matches tsr_resolve / render::taa::ResolveBlend with alpha=1.0).
    if (HF_TP.firstFrame != 0.0) return float4(current, 1.0);

    // Build the 3x3 neighborhood color AABB over the LOW-res current frame, using curTexel = 1/halfRes
    // (the clamp source is the low-res image, FSR2 convention). IDENTICAL to tsr_resolve.
    float3 boxMin = current;
    float3 boxMax = current;
    [unroll] for (int dy = -1; dy <= 1; ++dy) {
        [unroll] for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float2 off = float2((float)dx, (float)dy) * HF_TP.curTexel;
            float3 n = gCurrent.Sample(gSmp, i.uv + off).rgb;
            boxMin = min(boxMin, n);
            boxMax = max(boxMax, n);
        }
    }

    // --- REPROJECT the FULL-res history through the camera motion (motion_blur.frag VERBATIM). ---
    // Reconstruct THIS pixel's view-space position from the G-buffer linear depth (exactly like
    // motion_blur.frag's ReconstructViewPos: RH view space, -Z forward, so vz = -linDepth), map it
    // through prevClip == prevViewProj * inverse(curView) to the PREVIOUS frame's clip position, and
    // derive the previous-frame UV (the same prevNdc -> prevUV convention incl. the prevUV.y flip).
    float linDepth = gHistory.Sample(gHSmp, i.uv).a;     // current pixel linear depth (packed in history.a)
    float2 prevUV = i.uv;                                // default: identity (safe)
    bool disoccluded = false;
    if (linDepth > 0.0001) {
        float2 ndc = i.uv * 2.0 - 1.0;
        float  vx = ndc.x * (HF_TP.aspect * HF_TP.tanHalfFovY) * linDepth;
        float  vy = HF_YS * ndc.y * (HF_TP.tanHalfFovY) * linDepth;
        float  vz = -linDepth;                            // RH view space: -Z forward (motion_blur convention)
        float3 vpos = float3(vx, vy, vz);
        // MulPrevClip: column-major mat4 * homogeneous point (w==1 folded into column 3).
        float4 prevClip = HF_TP.prevClip0 * vpos.x + HF_TP.prevClip1 * vpos.y
                        + HF_TP.prevClip2 * vpos.z + HF_TP.prevClip3;
        if (prevClip.w > 1e-6) {
            float2 prevNdc = prevClip.xy / prevClip.w;
            prevUV = prevNdc * 0.5 + 0.5;
            // TEXTURE-SAMPLE Y convention (distinct from motion_blur, which uses prevUV as a velocity DELTA,
            // not a texture sample): the history RT samples Y-flipped on Metal relative to Vulkan. Empirically
            // HF_YS=-1 on Vulkan (sample prevUV directly) and HF_YS=+1 on Metal (sample 1-y). Flip on Metal:
            prevUV.y = (HF_YS > 0.0) ? 1.0 - prevUV.y : prevUV.y;
            // History reprojected off-screen -> disocclusion: there is no valid history here.
            disoccluded = any(prevUV < 0.0) || any(prevUV > 1.0);
        } else {
            disoccluded = true;                           // behind the camera -> no valid history
        }
    } else {
        // Background (no surface): keep identity UV (camera-distant sky reprojects ~to itself).
        prevUV = i.uv;
    }

    // Disocclusion fallback: where the history is off-screen take the fresh current sample (effective
    // alpha 1.0) instead of a stale/garbage history. Otherwise sample the reprojected history.
    float effAlpha = disoccluded ? 1.0 : HF_TP.alpha;
    float3 history = gHistory.Sample(gHSmp, prevUV).rgb;

    // Neighborhood clamp (per-channel AABB) — suppress ghosting by pulling drifted history back into the
    // locally-plausible color range. IDENTICAL to tsr_resolve / render::taa::ClipHistoryToNeighborhood.
    float3 clampedHistory = clamp(history, boxMin, boxMax);

    // Running-average blend: lerp(clampedHistory, current, effAlpha). disoccluded => effAlpha 1 => the
    // fresh current. IDENTICAL blend to tsr_resolve.
    float3 resolved = lerp(clampedHistory, current, effAlpha);
    return float4(resolved, 1.0);
}
