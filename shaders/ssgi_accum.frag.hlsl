// Temporal SSGI accumulation — running-mean pass (Slice BV). Fullscreen pass that folds the CURRENT
// jittered SSGI indirect frame (gCurrent, binding 0/1) into the accumulated MEAN held in the history /
// accumulation RT (gHistory, binding 3/4 — the same second material slot the SSGI composite / TAA
// resolve use). For a STATIC scene + STATIC camera, temporal accumulation is simply averaging N frames
// each rendered with a DIFFERENT golden-angle-rotated hemisphere kernel (no motion-vector reprojection
// — the camera doesn't move, so frame i and frame j sample the same surface). This is the exact
// fixed-N deterministic accumulation TAA established.
//
// We accumulate an exact running mean so frame N-1's output is (1/N) * sum of all N frames:
//   weight = 1 / (frame + 1)
//   accum  = lerp(history, current, weight)
// frame 0 -> weight 1 -> accum = current (defined start, history ignored);
// frame 1 -> weight 1/2 -> (history + current)/2; ... frame f -> the mean of frames 0..f. With fixed N,
// a fixed per-frame kernel rotation and a fixed accumulation order, two runs are byte-identical.
//
// The ONE new shader for the slice; the raw --ssgi-shot gather/composite + their goldens are untouched
// (this pass only exists on the --ssgi-temporal-shot path).
[[vk::binding(0, 0)]] Texture2D    gCurrent : register(t0);   // current jittered SSGI indirect (HDR)
[[vk::binding(1, 0)]] SamplerState gSmp     : register(s0);
[[vk::binding(3, 0)]] Texture2D    gHistory : register(t3);   // accumulated mean so far (HDR)
[[vk::binding(4, 0)]] SamplerState gHSmp    : register(s3);

// weight: the running-mean blend weight for THIS frame = 1/(frame+1). firstFrame!=0 forces the current
// frame through unblended (frame 0 has no valid history).
struct AccumParams { float2 texel; float weight; float firstFrame; };
#ifdef HF_MSL_GEN
[[vk::binding(1, 0)]] cbuffer AccumPC { AccumParams ap; };
#define HF_AP ap
#else
[[vk::push_constant]] struct { AccumParams p; } pc;
#define HF_AP pc.p
#endif

struct PSInput { float4 pos : SV_Position; [[vk::location(0)]] float2 uv : TEXCOORD0; };

float4 main(PSInput i) : SV_Target {
    float3 current = gCurrent.Sample(gSmp, i.uv).rgb;
    if (HF_AP.firstFrame != 0.0) return float4(current, 1.0);   // frame 0: no history
    float3 history = gHistory.Sample(gHSmp, i.uv).rgb;
    float3 accum = lerp(history, current, HF_AP.weight);        // exact running mean
    return float4(accum, 1.0);
}
