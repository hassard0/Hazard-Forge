// Unit test for the DSP1 wavetable oscillator NODE (engine/audio/dsp.h, flagship #23 DETERMINISTIC AUDIO,
// beachhead). Pure CPU (hf_core), ASan-eligible like the other pure tests.
//
// The oscillator is INTEGER / FIXED-POINT (Q15 int16 PCM, uint32 phase): no float anywhere, so the
// rendered buffer is bit-identical run-to-run AND platform-to-platform (MSVC vs Apple clang). The golden
// is a PINNED FNV-1a-64 DigestBuffer value (NO image, NO render-bake).
//
// What this pins:
//   * Block-boundary determinism (MAKE-OR-BREAK) — a 440Hz sine of N*256 frames rendered as ONE buffer
//     (RenderOsc) is BYTE-IDENTICAL to N separate 256-frame RenderBlock calls on ONE persistent OscNode
//     (the 32-bit phase accumulator carries across block boundaries).
//   * Pinned hash — DigestBuffer(one-buffer) == a hard-pinned uint64_t (the regression anchor / golden).
//   * Replay-stable — two RenderOsc calls byte-identical.
//   * Per-wave — Sine/Saw/Square each a deterministic pinned hash; a 440Hz Square is hard +-32767 only.
//   * Freq-sensitive — 220Hz vs 440Hz differ.

#include "audio/dsp.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::audio;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();
    const int    kSR        = 48000;
    const int    kBlock     = 256;
    const int    kN         = 16;
    const int    kTotal     = kN * kBlock;   // 4096 frames
    const uint32_t kFreq    = 440;

    // ---- (1) BLOCK-BOUNDARY DETERMINISM (make-or-break): one big buffer == N persistent blocks. -----
    const std::vector<int16_t> oneBuf = dsp::RenderOsc(dsp::Wave::Sine, kFreq, kSR, kTotal);
    std::vector<int16_t> nBlocks;
    {
        dsp::OscNode osc;
        osc.wave = dsp::Wave::Sine;
        osc.freqHz = kFreq;
        osc.phase = 0;
        for (int b = 0; b < kN; ++b) dsp::RenderBlock(osc, kSR, kBlock, nBlocks);
    }
    const uint64_t hOne    = dsp::DigestBuffer(oneBuf);
    const uint64_t hBlocks = dsp::DigestBuffer(nBlocks);
    check(oneBuf.size() == static_cast<size_t>(kTotal), "RenderOsc fills totalFrames");
    check(nBlocks.size() == static_cast<size_t>(kTotal), "N RenderBlock calls fill N*block frames");
    check(oneBuf == nBlocks, "block-boundary: one-buffer == N-blocks (byte-identical)");
    check(hOne == hBlocks, "block-boundary: DigestBuffer(one) == DigestBuffer(N-blocks)");

    // ---- (2) PINNED HASH — the regression anchor / golden. ------------------------------------------
    const uint64_t kPinnedSine440 = 0x61f818d005aa9762ull;
    check(hOne == kPinnedSine440, "pinned hash: DigestBuffer(440Hz sine, 4096f) matches golden");

    // ---- (3) REPLAY-STABLE — two RenderOsc calls byte-identical. ------------------------------------
    {
        const std::vector<int16_t> again = dsp::RenderOsc(dsp::Wave::Sine, kFreq, kSR, kTotal);
        check(dsp::DigestBuffer(again) == hOne, "replay-stable: two RenderOsc calls byte-identical");
    }

    // ---- (4) PER-WAVE — Sine/Saw/Square each a deterministic pinned hash. ---------------------------
    const std::vector<int16_t> sineBuf   = oneBuf;
    const std::vector<int16_t> sawBuf    = dsp::RenderOsc(dsp::Wave::Saw,    kFreq, kSR, kTotal);
    const std::vector<int16_t> squareBuf = dsp::RenderOsc(dsp::Wave::Square, kFreq, kSR, kTotal);
    const uint64_t hSine   = dsp::DigestBuffer(sineBuf);
    const uint64_t hSaw    = dsp::DigestBuffer(sawBuf);
    const uint64_t hSquare = dsp::DigestBuffer(squareBuf);
    const uint64_t kPinnedSaw440    = 0xa26107d87d53ef03ull;
    const uint64_t kPinnedSquare440 = 0xcedfc6d4113edf24ull;
    check(hSine   == kPinnedSine440,    "per-wave: sine hash pinned");
    check(hSaw    == kPinnedSaw440,     "per-wave: saw hash pinned");
    check(hSquare == kPinnedSquare440,  "per-wave: square hash pinned");
    check(hSine != hSaw && hSaw != hSquare && hSine != hSquare, "per-wave: three waves all differ");
    {
        bool onlyPm = true;
        for (const int16_t s : squareBuf) if (s != 32767 && s != -32767) { onlyPm = false; break; }
        check(onlyPm, "per-wave: 440Hz Square samples are only +-32767");
    }

    // ---- (5) FREQ-SENSITIVE — 220Hz vs 440Hz differ. ------------------------------------------------
    {
        const std::vector<int16_t> s220 = dsp::RenderOsc(dsp::Wave::Sine, 220, kSR, kTotal);
        check(dsp::DigestBuffer(s220) != hSine, "freq-sensitive: 220Hz != 440Hz");
    }

    // ---- Showcase / --dsp1-osc numeric proof (printed in the test; no image golden). ---------------
    std::printf("dsp1-osc: wavetable oscillator (440Hz sine, sampleRate 48000, frames 4096)\n");
    std::printf("dsp1-osc: block-boundary determinism: one-buffer == N-blocks {hash:0x%016llx}\n",
                static_cast<unsigned long long>(hOne));
    std::printf("dsp1-osc: per-wave hashes {sine:0x%016llx, saw:0x%016llx, square:0x%016llx}\n",
                static_cast<unsigned long long>(hSine),
                static_cast<unsigned long long>(hSaw),
                static_cast<unsigned long long>(hSquare));
    std::printf("dsp1-osc: provenance {frames:4096, sampleRate:48000, blockSize:256}\n");

    if (g_fail == 0) { std::printf("dsp_test: ALL CHECKS PASSED\n"); return 0; }
    std::printf("dsp_test: %d failures\n", g_fail);
    return 1;
}
