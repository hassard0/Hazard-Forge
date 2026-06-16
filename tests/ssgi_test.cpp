// Slice BP — screen-space global illumination (SSGI). Pure CPU math: the cosine-weighted hemisphere
// kernel, the SHARED SSR view<->screen reconstruction round-trip (guards the reuse), the indirect
// accumulation estimator, and a tiny color-bleed mini-model. No device, ASan-eligible (links
// hf_core). Mirrors the math the --ssgi-shot showcase and ssgi.frag use (engine/render/ssgi.h).
#include "render/ssgi.h"
#include "render/ssr.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace hf::math;
namespace ssgi = hf::render::ssgi;
namespace ssr = hf::render::ssr;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) < eps; }

int main() {
    const int K = 16;

    // ---- Hemisphere kernel: all K dirs unit length + in the hemisphere of the normal. ----
    {
        const Vec3 normals[] = {
            normalize(Vec3{0, 1, 0}), normalize(Vec3{0, 0, 1}),
            normalize(Vec3{1, 0, 0}), normalize(Vec3{-0.3f, 0.8f, 0.5f}),
            normalize(Vec3{0.2f, -0.1f, -0.97f}),
        };
        for (const Vec3& N : normals) {
            for (int i = 0; i < K; ++i) {
                Vec3 d = ssgi::HemisphereDir(i, K, N);
                check(approx(length(d), 1.0f), "HemisphereDir is unit length");
                check(dot(d, N) >= -1e-4f, "HemisphereDir lies in the hemisphere of the normal");
            }
        }
    }

    // ---- Determinism: same (i,K,normal) -> byte-identical direction. ----
    {
        Vec3 N = normalize(Vec3{0.1f, 0.9f, 0.2f});
        for (int i = 0; i < K; ++i) {
            Vec3 a = ssgi::HemisphereDir(i, K, N);
            Vec3 b = ssgi::HemisphereDir(i, K, N);
            check(a.x == b.x && a.y == b.y && a.z == b.z,
                  "HemisphereDir is deterministic (identical bytes across calls)");
        }
    }

    // ---- normal=+Z: the TBN frame is built from N=+Z; the returned dirs match the hand-computed
    // cosine-mapped local samples rotated by that exact basis (guards the kernel formula). ----
    {
        Vec3 N{0, 0, 1};
        Vec3 T, Bv; ssgi::BuildTangentBasis(N, T, Bv);
        for (int i = 0; i < K; ++i) {
            float u1 = ((float)i + 0.5f) / (float)K;
            float u2 = ssgi::RadicalInverse2((uint32_t)i);
            float r = std::sqrt(u1);
            float phi = 6.2831853071795864769f * u2;
            float lx = r * std::cos(phi), ly = r * std::sin(phi), lz = std::sqrt(1.0f - u1);
            Vec3 expected = normalize(Vec3{
                T.x*lx + Bv.x*ly + N.x*lz,
                T.y*lx + Bv.y*ly + N.y*lz,
                T.z*lx + Bv.z*ly + N.z*lz});
            Vec3 d = ssgi::HemisphereDir(i, K, N);
            check(approx(d.x, expected.x) && approx(d.y, expected.y) && approx(d.z, expected.z),
                  "HemisphereDir(+Z) matches the hand-computed cosine-mapped sample");
            // For +Z the local z (cos theta) is positive, so the dir's z must be > 0.
            check(d.z > 0.0f, "HemisphereDir(+Z) has positive z (upper hemisphere)");
        }
    }

    // ---- Average direction is ~normal (cosine-weighted set has no lateral bias). ----
    {
        const int Kbig = 256;
        const Vec3 normals[] = {
            normalize(Vec3{0, 1, 0}), normalize(Vec3{0, 0, 1}),
            normalize(Vec3{-0.3f, 0.8f, 0.5f}),
        };
        for (const Vec3& N : normals) {
            Vec3 avg{0, 0, 0};
            for (int i = 0; i < Kbig; ++i) avg = avg + ssgi::HemisphereDir(i, Kbig, N);
            avg = avg / (float)Kbig;
            Vec3 d = normalize(avg);
            // The mean points along N (cosine lobe is axially symmetric about N).
            check(dot(d, N) > 0.97f, "average HemisphereDir aligns with the normal (no lateral bias)");
        }
    }

    // ---- Reconstruction reuse: the SHARED ssr::ReconstructViewPos <-> ViewToScreenUV still round-trips
    // (ssgi re-exports these; this guards the reuse). ----
    {
        const float fovY = 1.04719755f;
        const float tanHalf = std::tan(0.5f * fovY);
        const float aspect = 1280.0f / 720.0f;
        for (float yFlip : {-1.0f, 1.0f}) {
            Vec3 p{1.1f, -0.6f, -5.5f};
            Vec3 uvd = ssgi::ViewToScreenUV(p, tanHalf, aspect, yFlip);
            Vec3 r = ssgi::ReconstructViewPos(uvd.x, uvd.y, uvd.z, tanHalf, aspect, yFlip);
            check(approx(r.x, p.x) && approx(r.y, p.y) && approx(r.z, p.z),
                  "ssgi reuse: ViewToScreenUV<->ReconstructViewPos round-trips");
        }
        // The re-export is literally the SSR function.
        Vec3 a = ssgi::ViewToScreenUV(Vec3{0.2f, 0.3f, -4.0f}, tanHalf, aspect, -1.0f);
        Vec3 b = ssr::ViewToScreenUV(Vec3{0.2f, 0.3f, -4.0f}, tanHalf, aspect, -1.0f);
        check(a.x == b.x && a.y == b.y && a.z == b.z, "ssgi::ViewToScreenUV == ssr::ViewToScreenUV");
    }

    // ---- Accumulate: mean of K hit radiances. ----
    {
        std::vector<Vec3> hits = {{0.4f, 0.2f, 0.1f}, {0.6f, 0.4f, 0.3f}, {0.2f, 0.0f, 0.2f}};
        Vec3 e = ssgi::AccumulateIndirect(hits);
        check(approx(e.x, (0.4f+0.6f+0.2f)/3.0f) &&
              approx(e.y, (0.2f+0.4f+0.0f)/3.0f) &&
              approx(e.z, (0.1f+0.3f+0.2f)/3.0f), "AccumulateIndirect == mean of hit radiances");
    }
    // ---- All-miss -> 0 (documented fallback). ----
    {
        std::vector<Vec3> none;
        Vec3 e = ssgi::AccumulateIndirect(none);
        check(approx(e.x, 0.0f) && approx(e.y, 0.0f) && approx(e.z, 0.0f),
              "AccumulateIndirect of no hits == 0 (miss = no indirect)");
        // Misses modeled as zero radiances also average toward 0.
        std::vector<Vec3> zeros(K, Vec3{0, 0, 0});
        Vec3 z = ssgi::AccumulateIndirect(zeros);
        check(approx(z.x, 0.0f) && approx(z.y, 0.0f) && approx(z.z, 0.0f),
              "AccumulateIndirect of all-zero (all-miss) hits == 0");
    }

    // ---- Color-bleed mini-model: a synthetic scene where the hemisphere rays that point toward a
    // strongly RED emitter (here: the rays whose direction has a large +X component, i.e. toward a red
    // panel on the +X side) return red radiance, and the rest return ~black. AccumulateIndirect must
    // then yield a REDDISH indirect color (R clearly dominant) — i.e. color bleeds from the panel.
    // Mini-model: emitter color = red where dir.x > 0.25 (panel subtends part of the hemisphere),
    // black elsewhere; this stands in for the screen-space march hitting the lit red panel. ----
    {
        Vec3 N{0, 1, 0};                       // a neutral floor facing up
        const Vec3 red{1.0f, 0.05f, 0.05f};
        const Vec3 black{0.0f, 0.0f, 0.0f};
        std::vector<Vec3> hits;
        hits.reserve(K);
        int redCount = 0;
        for (int i = 0; i < K; ++i) {
            Vec3 d = ssgi::HemisphereDir(i, K, N);
            if (d.x > 0.25f) { hits.push_back(red); ++redCount; }
            else             { hits.push_back(black); }
        }
        check(redCount > 0, "color-bleed mini-model: some rays reach the red panel");
        Vec3 indirect = ssgi::AccumulateIndirect(hits);
        check(indirect.x > indirect.y + 0.05f && indirect.x > indirect.z + 0.05f,
              "color-bleed: indirect is REDDISH (R dominates G and B)");
        check(indirect.x > 0.0f, "color-bleed: nonzero red indirect (panel bled onto the floor)");
    }

    if (g_fail == 0) { std::printf("ssgi_test: all checks passed\n"); return 0; }
    std::printf("ssgi_test: %d FAILURES\n", g_fail);
    return 1;
}
