// Slice CW — auto-exposure (histogram eye adaptation). Pure CPU math: the Rec.709 Luminance, the
// log2-luminance bin mapping + its inverse, the black-bin-excluded weighted AverageLuminance, and the
// key-value ExposureFromAverage. No device, ASan-eligible (links hf_core). Mirrors the EXACT math the
// --autoexposure-shot showcase and the autoexposure_histogram/reduce compute shaders run
// (engine/render/auto_exposure.h).
//
// Properties pinned (per the spec):
//   * Luminance: Rec.709 luma — white -> 1, black -> 0, pure green -> 0.7152.
//   * Bin mapping inverse: BinToLum(LumToBin(lum)) ~= lum within one bin width across the range; lum<=0
//     -> bin 0; LumToBin monotone non-decreasing; BinToLum monotone increasing.
//   * Average analytic: all pixels in ONE bin -> AverageLuminance == that bin's center luminance; a
//     known two-bin split -> the correct weighted average; the black bin (bin 0) is EXCLUDED.
//   * Exposure: ExposureFromAverage == keyValue/avg; a brighter avg -> a smaller exposure (monotone
//     decreasing); a uniform-luminance image -> the exact hand-computed EV.
//   * Determinism: same histogram -> same exposure (pure functions, no RNG/time).
#include "render/auto_exposure.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace ae = hf::render::autoexp;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

int main() {
    // Fixed histogram params (the showcase/shader family): a log2-luminance range over [2^-8, 2^4] ==
    // [~0.0039, 16] across 256 bins. The exact range is documented in auto_exposure.h; the test relies
    // on the relations + the analytic average.
    const int   kBins        = 256;
    const float kMinLogLum   = -8.0f;   // log2 floor -> 2^-8 ~= 0.0039
    const float kLogLumRange = 12.0f;   // up to 2^(−8+12) == 2^4 == 16
    const float kKeyValue    = 0.18f;   // 18%-grey middle-grey key

    // ---- Luminance: Rec.709 luma for known colors. ----
    {
        check(approx(ae::Luminance(Vec3{1, 1, 1}), 1.0f, 1e-6f), "Luminance(white) == 1");
        check(ae::Luminance(Vec3{0, 0, 0}) == 0.0f, "Luminance(black) == 0 exactly");
        check(approx(ae::Luminance(Vec3{0, 1, 0}), 0.7152f, 1e-6f), "Luminance(green) == 0.7152");
        check(approx(ae::Luminance(Vec3{1, 0, 0}), 0.2126f, 1e-6f), "Luminance(red) == 0.2126");
        check(approx(ae::Luminance(Vec3{0, 0, 1}), 0.0722f, 1e-6f), "Luminance(blue) == 0.0722");
        // Scaling the color scales the luminance linearly.
        check(approx(ae::Luminance(Vec3{0.5f, 0.5f, 0.5f}), 0.5f, 1e-6f), "Luminance is linear (0.5 grey)");
    }

    // ---- Bin mapping: lum <= 0 -> bin 0 (the black bin). ----
    {
        check(ae::LumToBin(0.0f, kMinLogLum, kLogLumRange, kBins) == 0, "LumToBin(0) == bin 0");
        check(ae::LumToBin(-1.0f, kMinLogLum, kLogLumRange, kBins) == 0, "LumToBin(negative) == bin 0");
        // A luminance below the 2^minLogLum floor also maps to bin 0.
        check(ae::LumToBin(1e-6f, kMinLogLum, kLogLumRange, kBins) == 0, "LumToBin(below floor) == bin 0");
        // A luminance above the ceiling clamps to the top bin.
        check(ae::LumToBin(1000.0f, kMinLogLum, kLogLumRange, kBins) == kBins - 1,
              "LumToBin(above ceiling) == top bin");
    }

    // ---- Bin mapping is monotone non-decreasing in luminance. ----
    {
        int prev = -1;
        bool monotone = true;
        for (float lum = 0.01f; lum <= 16.0f; lum *= 1.2f) {
            int b = ae::LumToBin(lum, kMinLogLum, kLogLumRange, kBins);
            if (b < prev) monotone = false;
            prev = b;
        }
        check(monotone, "LumToBin is monotone non-decreasing in luminance");
    }

    // ---- BinToLum is monotone increasing in bin. ----
    {
        float prev = -1.0f;
        bool monotone = true;
        for (int b = 0; b < kBins; ++b) {
            float l = ae::BinToLum(b, kMinLogLum, kLogLumRange, kBins);
            if (!(l > prev)) monotone = false;
            prev = l;
        }
        check(monotone, "BinToLum is monotone increasing in bin");
    }

    // ---- Bin mapping inverse: BinToLum(LumToBin(lum)) ~= lum within one bin width. ----
    {
        bool allClose = true;
        for (float lum = 0.02f; lum <= 12.0f; lum *= 1.3f) {
            int b = ae::LumToBin(lum, kMinLogLum, kLogLumRange, kBins);
            float lr = ae::BinToLum(b, kMinLogLum, kLogLumRange, kBins);
            // Within one bin width in LOG2 space (the bin spans logLumRange/bins stops).
            float binStops = kLogLumRange / (float)kBins;
            float d = std::fabs(std::log2(lr) - std::log2(lum));
            if (d > binStops) allClose = false;
        }
        check(allClose, "BinToLum(LumToBin(lum)) ~= lum within one bin width");
    }

    // ---- Average analytic: all pixels in ONE bin -> AverageLuminance == that bin's center luminance. ----
    {
        const int kTestBin = 100;   // a non-black bin
        std::vector<uint32_t> hist((size_t)kBins, 0u);
        hist[kTestBin] = 5000u;     // every contributing pixel in one bin
        float avg = ae::AverageLuminance(hist.data(), kBins, kMinLogLum, kLogLumRange, 5000u);
        float expect = ae::BinToLum(kTestBin, kMinLogLum, kLogLumRange, kBins);
        check(approx(avg, expect, expect * 1e-5f),
              "AverageLuminance(single bin) == that bin's center luminance");
    }

    // ---- Average analytic: a known TWO-bin split -> the correct weighted average. ----
    {
        const int binA = 60, binB = 180;
        const uint32_t cA = 3000u, cB = 1000u;
        std::vector<uint32_t> hist((size_t)kBins, 0u);
        hist[binA] = cA; hist[binB] = cB;
        float avg = ae::AverageLuminance(hist.data(), kBins, kMinLogLum, kLogLumRange, cA + cB);
        float lA = ae::BinToLum(binA, kMinLogLum, kLogLumRange, kBins);
        float lB = ae::BinToLum(binB, kMinLogLum, kLogLumRange, kBins);
        float expect = (lA * cA + lB * cB) / (float)(cA + cB);
        check(approx(avg, expect, expect * 1e-5f),
              "AverageLuminance(two bins) == the count-weighted average of the bin luminances");
    }

    // ---- Average analytic: the BLACK BIN (bin 0) is EXCLUDED from the average. ----
    {
        const int litBin = 120;
        std::vector<uint32_t> hist((size_t)kBins, 0u);
        hist[0]      = 1000000u;   // a huge pile of black pixels (sky punch-through / letterbox)
        hist[litBin] = 2000u;      // a little lit content
        float avg = ae::AverageLuminance(hist.data(), kBins, kMinLogLum, kLogLumRange, 1002000u);
        float litLum = ae::BinToLum(litBin, kMinLogLum, kLogLumRange, kBins);
        // The black pile is excluded -> the average is exactly the lit bin's luminance (not dragged ~0).
        check(approx(avg, litLum, litLum * 1e-5f),
              "AverageLuminance excludes the black bin (bin 0) -> meters the lit content");
    }

    // ---- Average analytic: an all-black histogram -> a finite non-zero floor (no divide-by-zero). ----
    {
        std::vector<uint32_t> hist((size_t)kBins, 0u);
        hist[0] = 5000u;   // everything black
        float avg = ae::AverageLuminance(hist.data(), kBins, kMinLogLum, kLogLumRange, 5000u);
        check(std::isfinite(avg) && avg > 0.0f, "AverageLuminance(all black) is finite + > 0 (floor)");
    }

    // ---- Exposure: ExposureFromAverage == keyValue / avg (the middle-grey key-value formula). ----
    {
        float avg = 0.36f;   // twice middle-grey
        float e = ae::ExposureFromAverage(avg, kKeyValue);
        check(approx(e, kKeyValue / avg, 1e-6f), "ExposureFromAverage == keyValue/avg");
        // A middle-grey (0.18) average exposes to exactly 1.0 (the key-value identity).
        float eMid = ae::ExposureFromAverage(0.18f, kKeyValue);
        check(approx(eMid, 1.0f, 1e-5f), "ExposureFromAverage(avg==keyValue) == 1 (the key-value identity)");
    }

    // ---- Exposure: monotone DECREASING in average (a brighter scene -> a smaller exposure). ----
    {
        float prev = 1e30f;
        bool monotone = true;
        for (float avg = 0.02f; avg <= 4.0f; avg *= 1.25f) {
            float e = ae::ExposureFromAverage(avg, kKeyValue);
            if (e > prev) monotone = false;   // strictly non-increasing as the scene brightens
            prev = e;
        }
        check(monotone, "ExposureFromAverage is monotone decreasing in average luminance");
        // A bright scene darkens (exposure < 1); a dark scene brightens (exposure > 1).
        check(ae::ExposureFromAverage(2.0f, kKeyValue) < 1.0f, "bright avg -> exposure < 1 (darkens)");
        check(ae::ExposureFromAverage(0.02f, kKeyValue) > 1.0f, "dark avg -> exposure > 1 (brightens)");
    }

    // ---- Exposure robustness: a near-black average yields a large-but-finite exposure (eps floor). ----
    {
        float e = ae::ExposureFromAverage(0.0f, kKeyValue);
        check(std::isfinite(e) && e > 0.0f, "ExposureFromAverage(0) is finite + > 0 (eps floor)");
    }

    // ---- Uniform-luminance image -> the exact hand-computed EV. ----
    // Every pixel has luminance L -> they all land in one bin -> AverageLuminance == that bin's center
    // -> the exposure is keyValue / binCenter, a fully deterministic hand-checkable value.
    {
        const float L = 0.5f;
        int b = ae::LumToBin(L, kMinLogLum, kLogLumRange, kBins);
        std::vector<uint32_t> hist((size_t)kBins, 0u);
        hist[b] = 1280u * 720u;   // a full uniform image
        float avg = ae::AverageLuminance(hist.data(), kBins, kMinLogLum, kLogLumRange, 1280u * 720u);
        float binCenter = ae::BinToLum(b, kMinLogLum, kLogLumRange, kBins);
        check(approx(avg, binCenter, binCenter * 1e-5f), "uniform image -> avg == the bin center");
        float e = ae::ExposureFromAverage(avg, kKeyValue);
        check(approx(e, kKeyValue / binCenter, kKeyValue / binCenter * 1e-4f),
              "uniform image -> exact EV == keyValue / binCenter");
    }

    // ---- Determinism: the same histogram -> bit-identical average + exposure. ----
    {
        std::vector<uint32_t> hist((size_t)kBins, 0u);
        hist[40] = 700u; hist[90] = 1500u; hist[200] = 320u;
        float a1 = ae::AverageLuminance(hist.data(), kBins, kMinLogLum, kLogLumRange, 2520u);
        float a2 = ae::AverageLuminance(hist.data(), kBins, kMinLogLum, kLogLumRange, 2520u);
        check(a1 == a2, "AverageLuminance is deterministic (identical bits)");
        check(ae::ExposureFromAverage(a1, kKeyValue) == ae::ExposureFromAverage(a2, kKeyValue),
              "ExposureFromAverage is deterministic (identical bits)");
    }

    if (g_fail == 0) { std::printf("auto_exposure_test: all checks passed\n"); return 0; }
    std::printf("auto_exposure_test: %d FAILURES\n", g_fail);
    return 1;
}
