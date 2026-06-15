// Unit test for the deterministic software audio mixer (engine/audio/mixer.{h,cpp} +
// engine/audio/wav.{h,cpp}, Slice BB). Pure CPU (hf_core), ASan-eligible like the other pure tests.
//
// The mixer is INTEGER / FIXED-POINT end to end (Q15 gains, int32 accumulation, hard-clamp to int16):
// no float/double in the sample loop, so the rendered buffer is bit-identical run-to-run AND
// platform-to-platform (MSVC vs Apple clang) — that is what makes the byte-exact WAV golden safe.
//
// What this pins:
//   * Determinism      — Render twice on the same scene => bit-identical buffers.
//   * Mix math         — two known constant voices sum to the expected int16; overflow hard-clamps to
//                        +-32767; a single full-gain voice hits the expected peak.
//   * Pan law (linear) — full-left puts all energy in L and 0 in R; center splits; full-right mirrors;
//                        integer pan is exact at the endpoints.
//   * Envelope (ADSR)  — Q15 value at offset 0 (attack start), end-of-attack (peak), the sustain
//                        region (sustainLevel), and post-release (silent) match the expected Q15.
//   * WAV header       — the 44-byte PCM header fields (chunk sizes, sampleRate, byteRate, blockAlign,
//                        bits) are correct for a known buffer; data chunk size == samples*channels*2.

#include "audio/mixer.h"
#include "audio/wav.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A flat DC "voice" helper: a Square wave evaluates to a CONSTANT +amplitude on the first half of its
// phase. With freqHz==0 the phase never advances, so a Square voice is a constant +full-scale source —
// the cleanest way to assert mix math / pan / clamp without depending on a wavetable value.
static audio::Voice ConstSquare(int16_t gainQ15, int16_t panQ15, int totalSamples) {
    audio::Voice v;
    v.wave = audio::Wave::Square;
    v.freqHz = 0;                 // phase stays at 0 => Square == +amplitude (constant)
    v.startSample = 0;
    v.durSample = totalSamples;
    v.gain = gainQ15;
    v.pan = panQ15;
    v.env = audio::Adsr{0, 0, 32767, 0};   // no attack/decay/release => full Q15 the whole duration
    v.noiseSeed = 0;
    return v;
}

int main() {
    const int kSR = 44100;
    const int16_t kFull = 32767;     // Q15 1.0 (as close as int16 gets)
    const int16_t kCenter = 0;       // pan center
    const int16_t kLeft = -32767;    // pan full-left
    const int16_t kRight = 32767;    // pan full-right

    // ---- Determinism: render the same scene twice -> bit-identical. -----------------------------
    {
        audio::MixConfig cfg{kSR, 2, 256};
        std::vector<audio::Voice> voices = {
            ConstSquare(20000, kCenter, 256),
            [] { audio::Voice v; v.wave = audio::Wave::Sine; v.freqHz = 440; v.startSample = 8;
                 v.durSample = 200; v.gain = 18000; v.pan = -12000;
                 v.env = audio::Adsr{16, 16, 24000, 32}; v.noiseSeed = 0; return v; }(),
            [] { audio::Voice v; v.wave = audio::Wave::Noise; v.freqHz = 0; v.startSample = 32;
                 v.durSample = 64; v.gain = 9000; v.pan = 20000;
                 v.env = audio::Adsr{0, 0, 32767, 0}; v.noiseSeed = 0xC0FFEEu; return v; }(),
        };
        std::vector<int16_t> a, b;
        audio::Render(cfg, voices, a);
        audio::Render(cfg, voices, b);
        check(a.size() == static_cast<size_t>(256 * 2), "Render fills totalSamples*channels");
        check(a == b, "Render is deterministic (bit-identical across runs)");
    }

    // ---- Mix math: two constant voices sum to the expected int16. -------------------------------
    {
        // Two centered const-square voices at gain 10000 (Q15). Per voice, the source is +32767 scaled
        // by env(=Q15 full) then gain: (32767*10000)>>15 = 9999. Centered pan halves to each channel:
        // (9999*16384)>>15 = 4999 per channel per voice; two voices => 9998 per channel.
        audio::MixConfig cfg{kSR, 2, 4};
        std::vector<audio::Voice> voices = { ConstSquare(10000, kCenter, 4), ConstSquare(10000, kCenter, 4) };
        std::vector<int16_t> out;
        audio::Render(cfg, voices, out);
        check(out[0] == 9998 && out[1] == 9998, "two centered const voices sum to expected int16 (L==R==9998)");
        // Every frame is identical (constant sources).
        bool allEq = true;
        for (int s = 0; s < 4; ++s) if (out[s * 2] != 9998 || out[s * 2 + 1] != 9998) allEq = false;
        check(allEq, "constant source => every frame identical");
    }

    // ---- Mix math: hard-clamp at +-32767 on overflow. ------------------------------------------
    {
        // Four full-gain, full-LEFT const voices: each contributes ~32767 to L; the int32 accumulator
        // reaches ~131068 and must hard-clamp to +32767 in L (R stays 0 at full-left).
        audio::MixConfig cfg{kSR, 2, 2};
        std::vector<audio::Voice> voices = {
            ConstSquare(kFull, kLeft, 2), ConstSquare(kFull, kLeft, 2),
            ConstSquare(kFull, kLeft, 2), ConstSquare(kFull, kLeft, 2),
        };
        std::vector<int16_t> out;
        audio::Render(cfg, voices, out);
        check(out[0] == 32767, "overflow hard-clamps L to +32767");
        check(out[1] == 0, "full-left puts ~0 in R even under overflow");
    }

    // ---- Mix math: a single full-gain centered voice hits the expected peak. --------------------
    {
        audio::MixConfig cfg{kSR, 2, 2};
        std::vector<audio::Voice> voices = { ConstSquare(kFull, kCenter, 2) };
        std::vector<int16_t> out;
        audio::Render(cfg, voices, out);
        // s = ((((32767*32767)>>15)*32767)>>15) = 32765; centered = (32765*16384)>>15 = 16382.
        check(out[0] == 16382 && out[1] == 16382, "single full-gain centered voice peak == 16382 per channel");
    }

    // ---- Pan law (linear): endpoints are integer-exact. ----------------------------------------
    {
        audio::MixConfig cfg{kSR, 2, 2};
        // Full LEFT: all energy in L (s=32765 scaled by panL=32767 => 32764), 0 in R.
        {
            std::vector<audio::Voice> v = { ConstSquare(kFull, kLeft, 2) };
            std::vector<int16_t> out; audio::Render(cfg, v, out);
            check(out[0] == 32764 && out[1] == 0, "full-left: energy in L (32764), R == 0");
        }
        // Full RIGHT: all energy in R, 0 in L.
        {
            std::vector<audio::Voice> v = { ConstSquare(kFull, kRight, 2) };
            std::vector<int16_t> out; audio::Render(cfg, v, out);
            check(out[0] == 0 && out[1] == 32764, "full-right: L == 0, energy in R (32764)");
        }
        // CENTER: equal split (panL == panR == 16384).
        {
            std::vector<audio::Voice> v = { ConstSquare(kFull, kCenter, 2) };
            std::vector<int16_t> out; audio::Render(cfg, v, out);
            check(out[0] == out[1] && out[0] == 16382, "center: L == R == 16382 (equal split)");
        }
    }

    // ---- Envelope (ADSR) Q15 at known offsets. -------------------------------------------------
    {
        // attack=100, decay=100, sustainLevel=16384 (Q15 0.5), release=100, over a voice of dur 1000.
        audio::Adsr env{100, 100, 16384, 100};
        const int dur = 1000;
        check(audio::EnvelopeAt(env, 0, dur) == 0, "ADSR at offset 0 == 0 (attack start, silent)");
        check(audio::EnvelopeAt(env, 100, dur) == 32767, "ADSR at end-of-attack == 32767 (peak)");
        check(audio::EnvelopeAt(env, 200, dur) == 16384, "ADSR at end-of-decay == sustainLevel (16384)");
        check(audio::EnvelopeAt(env, 500, dur) == 16384, "ADSR mid-sustain == sustainLevel (16384)");
        // Release starts at dur-release == 900, ends at dur==1000 reaching 0.
        check(audio::EnvelopeAt(env, 900, dur) == 16384, "ADSR at release start == sustainLevel (16384)");
        check(audio::EnvelopeAt(env, 1000, dur) == 0, "ADSR at end-of-release == 0 (silent)");
        check(audio::EnvelopeAt(env, 1001, dur) == 0, "ADSR past duration == 0 (silent)");
        check(audio::EnvelopeAt(env, 950, dur) == 8192, "ADSR mid-release halfway == 8192");
    }

    // ---- WAV header fields for a known buffer. -------------------------------------------------
    {
        const int sampleRate = 44100, channels = 2;
        std::vector<int16_t> interleaved = {1, -1, 2, -2, 3, -3};  // 3 frames, stereo
        std::vector<uint8_t> bytes = audio::EncodeWav(sampleRate, channels, interleaved);

        const uint32_t dataBytes = static_cast<uint32_t>(interleaved.size()) * 2;  // 12
        check(bytes.size() == 44u + dataBytes, "WAV total size == 44 + data bytes");

        auto u32 = [&](size_t off) -> uint32_t {
            return  static_cast<uint32_t>(bytes[off]) |
                   (static_cast<uint32_t>(bytes[off + 1]) << 8) |
                   (static_cast<uint32_t>(bytes[off + 2]) << 16) |
                   (static_cast<uint32_t>(bytes[off + 3]) << 24);
        };
        auto u16 = [&](size_t off) -> uint16_t {
            return static_cast<uint16_t>(bytes[off] | (bytes[off + 1] << 8));
        };
        auto fourcc = [&](size_t off, const char* tag) {
            return bytes[off] == (uint8_t)tag[0] && bytes[off + 1] == (uint8_t)tag[1] &&
                   bytes[off + 2] == (uint8_t)tag[2] && bytes[off + 3] == (uint8_t)tag[3];
        };

        check(fourcc(0, "RIFF"), "WAV: 'RIFF' magic");
        check(u32(4) == 36u + dataBytes, "WAV: RIFF chunk size == 36 + data bytes");
        check(fourcc(8, "WAVE"), "WAV: 'WAVE' format");
        check(fourcc(12, "fmt "), "WAV: 'fmt ' subchunk id");
        check(u32(16) == 16u, "WAV: fmt subchunk size == 16 (PCM)");
        check(u16(20) == 1u, "WAV: audio format == 1 (PCM)");
        check(u16(22) == (uint16_t)channels, "WAV: numChannels == 2");
        check(u32(24) == (uint32_t)sampleRate, "WAV: sampleRate == 44100");
        check(u32(28) == (uint32_t)(sampleRate * channels * 2), "WAV: byteRate == sr*ch*2");
        check(u16(32) == (uint16_t)(channels * 2), "WAV: blockAlign == ch*2");
        check(u16(34) == 16u, "WAV: bitsPerSample == 16");
        check(fourcc(36, "data"), "WAV: 'data' subchunk id");
        check(u32(40) == dataBytes, "WAV: data subchunk size == samples*channels*2");

        // The samples follow the header, little-endian int16.
        check((int16_t)u16(44) == 1 && (int16_t)u16(46) == -1, "WAV: first frame bytes == {1,-1}");
    }

    if (g_fail == 0) { std::printf("audio_test OK\n"); return 0; }
    std::printf("audio_test: %d failures\n", g_fail);
    return 1;
}
