// Slice BV — Temporal SSGI accumulation. Pure CPU math: the per-frame jittered hemisphere kernel
// (HemisphereDirJittered = the base SSGI kernel + a FIXED per-frame golden-angle azimuth rotation,
// frame 0 == HemisphereDir), the fixed-N accumulation count, and a synthetic variance-reduction
// mini-model that proves averaging N jittered SSGI frames of an unbiased noisy estimator lowers the
// variance while preserving the mean (converges toward ground truth). No device, ASan-eligible (links
// hf_core). Mirrors the math the --ssgi-temporal-shot showcase and ssgi.frag (frame param) use
// (engine/render/ssgi.h). DETERMINISTIC: no time/RNG — two runs byte-identical.
#include "render/ssgi.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace hf::math;
namespace ssgi = hf::render::ssgi;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
    const int K = 16;
    const Vec3 normals[] = {
        normalize(Vec3{0, 1, 0}), normalize(Vec3{0, 0, 1}),
        normalize(Vec3{1, 0, 0}), normalize(Vec3{-0.3f, 0.8f, 0.5f}),
        normalize(Vec3{0.2f, -0.1f, -0.97f}),
    };

    // ---- Frame 0 == the base SSGI kernel (documented clean relationship). HemisphereDirJittered
    // applies a per-frame azimuth rotation of frame*goldenAngle, which is 0 for frame 0, so the
    // returned direction is byte-identical to HemisphereDir. This keeps the single-frame raw
    // --ssgi-shot path (frame 0) byte-identical. ----
    {
        for (const Vec3& N : normals) {
            for (int i = 0; i < K; ++i) {
                Vec3 base = ssgi::HemisphereDir(i, K, N);
                Vec3 f0   = ssgi::HemisphereDirJittered(i, K, N, 0);
                check(base.x == f0.x && base.y == f0.y && base.z == f0.z,
                      "HemisphereDirJittered frame 0 == HemisphereDir (byte-identical base kernel)");
            }
        }
    }

    // ---- frame > 0: unit length, still in the hemisphere of N, and DIFFERS from frame 0 (so each
    // accumulation frame contributes genuinely new samples — otherwise averaging would not reduce
    // variance). ----
    {
        const ssgi::SsgiTemporalParams tp;
        check(tp.accumFrames == 8, "SsgiTemporalParams default accumFrames == 8");
        for (const Vec3& N : normals) {
            for (int frame = 1; frame < tp.accumFrames; ++frame) {
                for (int i = 0; i < K; ++i) {
                    Vec3 d  = ssgi::HemisphereDirJittered(i, K, N, frame);
                    Vec3 f0 = ssgi::HemisphereDirJittered(i, K, N, 0);
                    check(approx(length(d), 1.0f), "HemisphereDirJittered(frame>0) is unit length");
                    check(dot(d, N) >= -1e-4f,
                          "HemisphereDirJittered(frame>0) lies in the hemisphere of the normal");
                    // The rotation is a nonzero azimuth offset, so the direction must move.
                    check(!(d.x == f0.x && d.y == f0.y && d.z == f0.z),
                          "HemisphereDirJittered(frame>0) differs from frame 0 (new samples)");
                }
            }
        }
    }

    // ---- Determinism: same (i,K,normal,frame) -> byte-identical direction across calls. ----
    {
        Vec3 N = normalize(Vec3{0.1f, 0.9f, 0.2f});
        for (int frame = 0; frame < 8; ++frame)
            for (int i = 0; i < K; ++i) {
                Vec3 a = ssgi::HemisphereDirJittered(i, K, N, frame);
                Vec3 b = ssgi::HemisphereDirJittered(i, K, N, frame);
                check(a.x == b.x && a.y == b.y && a.z == b.z,
                      "HemisphereDirJittered is deterministic (identical bytes across calls)");
            }
    }

    // ---- Per-frame rotation is a continuous function of frame: the azimuth advances by a FIXED
    // golden-angle step, so the whole jittered set for frame f is the base set rotated about N. The
    // average of the rotated set over a full hemisphere still aligns with N (no lateral bias is
    // introduced by the rotation). ----
    {
        const int Kbig = 256;
        for (const Vec3& N : {normalize(Vec3{0, 1, 0}), normalize(Vec3{-0.3f, 0.8f, 0.5f})}) {
            for (int frame = 0; frame < 8; ++frame) {
                Vec3 avg{0, 0, 0};
                for (int i = 0; i < Kbig; ++i) avg = avg + ssgi::HemisphereDirJittered(i, Kbig, N, frame);
                avg = avg / (float)Kbig;
                check(dot(normalize(avg), N) > 0.97f,
                      "averaged jittered kernel (any frame) aligns with the normal (rotation adds no bias)");
            }
        }
    }

    // ---- Variance-reduction mini-model. ----
    // Model: a single SSGI frame estimates the indirect irradiance E of a pixel with K Monte-Carlo
    // rays. Treat the per-frame estimate as an UNBIASED random variable: each frame's K rays are a
    // different jittered subset of the hemisphere, so the frame estimate is GROUND_TRUTH plus a
    // zero-mean per-frame noise term n_f. Temporal accumulation averages N such frame estimates:
    //   accum_N = (1/N) * sum_{f=0..N-1} (GROUND_TRUTH + n_f)
    // For independent zero-mean noise, Var(accum_N) = Var(single)/N and E[accum_N] = GROUND_TRUTH
    // (same mean, lower variance — the classic 1/N Monte-Carlo convergence the showcase exploits).
    // We drive the noise with a deterministic pseudo-sequence keyed by (frame) so the test is
    // reproducible without RNG — only its statistical SHAPE (zero-mean, comparable spread) matters.
    {
        const float GROUND_TRUTH = 0.40f;
        const int   N = 8;
        const int   kPixels = 4096;       // many pixels => stable sample statistics

        // Deterministic zero-mean noise for (pixel p, frame f): a hashed value folded into [-A, A].
        auto noise = [](int p, int f) -> float {
            uint32_t h = (uint32_t)p * 374761393u + (uint32_t)f * 668265263u;
            h = (h ^ (h >> 13)) * 1274126177u;
            h ^= h >> 16;
            float u = (float)h * 2.3283064365386963e-10f;   // [0,1)
            return (u - 0.5f) * 0.40f;                       // zero-mean in [-0.2, 0.2)
        };

        // Single-frame (frame 0) estimates vs N-frame accumulated estimates, per pixel.
        double meanSingle = 0.0, varSingle = 0.0;
        double meanAccum = 0.0, varAccum = 0.0;
        std::vector<float> single(kPixels), accum(kPixels);
        for (int p = 0; p < kPixels; ++p) {
            single[p] = GROUND_TRUTH + noise(p, 0);
            float s = 0.0f;
            for (int f = 0; f < N; ++f) s += GROUND_TRUTH + noise(p, f);
            accum[p] = s / (float)N;
            meanSingle += single[p];
            meanAccum += accum[p];
        }
        meanSingle /= kPixels;
        meanAccum /= kPixels;
        for (int p = 0; p < kPixels; ++p) {
            double ds = single[p] - meanSingle; varSingle += ds * ds;
            double da = accum[p] - meanAccum;   varAccum += da * da;
        }
        varSingle /= kPixels;
        varAccum /= kPixels;

        // Same mean (both unbiased; both ~ GROUND_TRUTH).
        check(std::fabs(meanSingle - (double)GROUND_TRUTH) < 0.01,
              "variance model: single-frame mean ~ ground truth (unbiased)");
        check(std::fabs(meanAccum - (double)GROUND_TRUTH) < 0.01,
              "variance model: N-frame accumulated mean ~ ground truth (same mean, still unbiased)");
        check(std::fabs(meanAccum - meanSingle) < 0.01,
              "variance model: accumulation PRESERVES the mean");
        // Lower variance — accumulation drives the noise down (toward Var/N).
        check(varAccum < varSingle * 0.5,
              "variance model: N-frame accumulation has materially LOWER variance than a single frame");
        // And it's in the ballpark of the ideal 1/N reduction (independent zero-mean noise).
        check(varAccum < varSingle * (1.0 / (double)N) * 2.0,
              "variance model: variance reduction is near the ideal 1/N Monte-Carlo rate");
    }

    if (g_fail == 0) { std::printf("ssgi_temporal_test: all checks passed\n"); return 0; }
    std::printf("ssgi_temporal_test: %d FAILURES\n", g_fail);
    return 1;
}
