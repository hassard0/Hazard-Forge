#pragma once
// Slice AU2 — DETERMINISTIC CONVOLUTION REVERB + SUBMIX BUSES (the audio parity++ gap). Header-only,
// namespace hf::audio::reverb. Pure C++ INTEGER / FIXED-POINT: NO <cmath>, NO float/double anywhere,
// NO clock/RNG, NO RHI/backend/GPU symbols. Audio has NO GPU path — cross-platform bit-exactness IS
// the product: the same dry signal through the same impulse response produces the IDENTICAL int16
// samples on every machine/compiler (MSVC == clang), a category a float convolution engine (UE5's
// convolution reverb / a MetaSounds submix graph) structurally cannot enter.
//
// WHAT THIS ADDS over the existing audio primitives (all READ-ONLY composed, never modified):
//   * engine/audio/dsp.h         — MulQ15 / ClampI16 / DigestBuffer / the kSineTable LUT discipline
//                                  (host-baked integer literals, NO runtime trig) reused VERBATIM.
//   * engine/audio/audio_graph.h — the graph::kQ15One EXACT-unity constant + the mixer's int32-
//                                  accumulate-then-single-clamp discipline (the submix bus math).
//   Both headers stay BYTE-FROZEN; reverb.h is a composition/routing layer, not a fork.
//
// ===================================================================================================
// THE DETERMINISM DECISION (REQUIRED — audio is typically float; we choose INTEGER):
// ---------------------------------------------------------------------------------------------------
// AU2 is INTEGER Q15 fixed-point, NOT pinned-quantized-float. The whole audio stack (mixer.cpp,
// dsp.h, audio_graph.h) is already integer Q15 int16 PCM, so the natural, strongest choice is to keep
// the convolution an INTEGER multiply-accumulate: wet[n] = clamp( ( Σ_k dry[n-k] * ir[k] ) >> 15 ).
// dry/wet are int16 PCM (the audio-stack sample format); the IR COEFFICIENTS are Q15 stored as int32.
//   WHY int32 (not int16) for the IR: EXACT unity is 1.0 == 32768 in Q15, which does NOT fit int16
//   (max 32767). Storing the IR as int32 Q15 lets a unit-impulse IR be EXACTLY {32768,0,0,...}, so
//   Process is a BIT-EXACT identity (wet == dry) — the convolution-identity pin. An int16 IR could
//   only hold 32767 (0.99997), truncating every sample and breaking the identity. The MAC uses an
//   int64 accumulator (dry ±32768 * ir ±2^31 over ≤4096 taps stays far inside int64), ONE arithmetic
//   >>15 (C++20 guarantees arithmetic right shift on signed types), ONE clamp — associative integer
//   addition in a PINNED ascending-k order => byte-identical run-to-run AND compiler-to-compiler. No
//   float summation-order hazard exists because there is no float. This is the whole point.
//
// HONEST SCOPE (fidelity, not smoothed):
//   * This is ROOM / SPACE convolution reverb (a mono impulse response), NOT binaural HRTF. dsp.h's
//     "NOT true HRTF convolution" note still stands: AU2 convolves a room IR, it does not synthesize
//     a head-related transfer function. Binaural HRTF remains future work.
//   * Direct-form (time-domain) convolution is O(dryLen · irLen). The IR length is CAPPED at
//     kMaxIrLen (4096 samples, ~85 ms @ 48 kHz) so a short dry click through the room IR is trivially
//     feasible (≤ a few million MACs). A production engine would block/FFT-partition a multi-second
//     IR; that is a performance refinement, not a determinism one — the integer MAC is the invariant.
//   * The synthetic room IR is a plausible MODEL (a few discrete early reflections + a decaying-noise
//     diffuse tail), not a measured space. The decay is geometric with Q15 truncation, so the tail
//     reaches EXACT silence in finitely many samples (the honest integer-DSP character, like the
//     audio_graph feedback delay).
// ===================================================================================================
#include <cstddef>
#include <cstdint>
#include <vector>

#include "audio/dsp.h"          // READ-ONLY: MulQ15 / ClampI16 / DigestBuffer / IsqrtU / the LUT discipline
#include "audio/audio_graph.h"  // READ-ONLY: graph::kQ15One (EXACT Q15 unity) — the submix bus gain unit

namespace hf::audio::reverb {

// EXACT Q15 unity (== 1.0). MulQ15(x, kQ15One) == x bit-exactly; the IR/gain identity anchor.
inline constexpr int32_t kQ15One   = 32768;
inline constexpr int      kMaxIrLen = 4096;     // O(N·M) convolution feasibility cap (see header note)

// ---- ClampI16From64: clamp a wide int64 accumulate to int16 PCM (the convolution/mix sink) ----------
// The MAC accumulator is int64; after the >>15 it can still exceed int32 in pathological hot cases, so
// we clamp on int64 (dsp::ClampI16 takes int32). Bit-identical on every compiler.
inline int16_t ClampI16From64(int64_t v) {
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return static_cast<int16_t>(v);
}

// =====================================================================================================
// (1) CONVOLUTION — the core integer MAC. wet = dry (*) ir, full linear convolution.
// -----------------------------------------------------------------------------------------------------
// wet has length dryLen + irLen - 1. For each output n, acc = Σ_k dry[n-k]*ir[k] over the valid k
// range in ASCENDING k order (the pinned accumulation order — integer add is associative so the order
// is a documentation/clarity choice, not a correctness one, but we pin it anyway). ir[k] is Q15, so
// the sum is a Q15-scaled dry; >>15 renormalizes, ClampI16From64 saturates to PCM. Pure integer.
inline std::vector<int16_t> Convolve(const std::vector<int16_t>& dry, const std::vector<int32_t>& ir) {
    std::vector<int16_t> wet;
    const int dryLen = static_cast<int>(dry.size());
    const int irLen  = static_cast<int>(ir.size());
    if (dryLen <= 0 || irLen <= 0) return wet;   // empty in -> empty out (silence)

    const int wetLen = dryLen + irLen - 1;
    wet.assign(static_cast<std::size_t>(wetLen), 0);
    for (int n = 0; n < wetLen; ++n) {
        int64_t acc = 0;
        // k ranges so that both dry[n-k] (0..dryLen-1) and ir[k] (0..irLen-1) are in range.
        const int kLo = (n - (dryLen - 1) > 0) ? (n - (dryLen - 1)) : 0;
        const int kHi = (n < irLen - 1) ? n : (irLen - 1);
        for (int k = kLo; k <= kHi; ++k)                     // ASCENDING k — the pinned order
            acc += static_cast<int64_t>(dry[static_cast<std::size_t>(n - k)]) *
                   static_cast<int64_t>(ir[static_cast<std::size_t>(k)]);
        wet[static_cast<std::size_t>(n)] = ClampI16From64(acc >> 15);   // arithmetic >>15 (C++20)
    }
    return wet;
}

// A stateless convolution-reverb effect: an impulse response + a Process that convolves a dry buffer.
struct ConvolveReverb {
    std::vector<int32_t> ir;   // Q15 impulse-response coefficients (int32 so unity 32768 is exact)
    std::vector<int16_t> Process(const std::vector<int16_t>& dry) const { return Convolve(dry, ir); }
};

// =====================================================================================================
// (2) DETERMINISTIC ROOM-IR SYNTHESIS — a plausible impulse response, pinned by construction.
// -----------------------------------------------------------------------------------------------------
// THE MODEL (documented + pinned): a few DISCRETE EARLY REFLECTIONS (the direct path + a handful of
// wall bounces at fixed sample offsets with decreasing Q15 gains) followed by a DIFFUSE TAIL of
// geometrically-decaying deterministic NOISE. NO runtime rand: the noise is a fixed integer avalanche
// hash of the sample index (the pcg.h/particles.h ParticleHash SHAPE, kept LOCAL so reverb.h stays
// self-contained clang-compilable like dsp.h). The tail envelope is a Q15 geometric decay (repeated
// MulQ15), which is NON-INCREASING by construction and reaches EXACT silence in finitely many samples.

// --- Au2Hash: a fixed uint32 wrapping avalanche (the ParticleHash shape, verbatim ops) ---------------
inline uint32_t Au2Hash(uint32_t seed, uint32_t index) {
    uint32_t h = seed * 2654435761u;                       // Knuth multiplicative
    h ^= (index + 0x9E3779B9u + (h << 6) + (h >> 2));
    h += index * 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
// A signed Q15 noise sample in [-32768, 32767] — the low 16 bits of the hash read as int16.
inline int32_t Au2NoiseQ15(uint32_t seed, uint32_t index) {
    return static_cast<int32_t>(static_cast<int16_t>(Au2Hash(seed, index) & 0xFFFFu));
}

// --- The FIXED room model (pinned literals — the kSineTable discipline) ------------------------------
inline constexpr int      kIrLen        = 2048;    // default IR length (~43 ms @ 48 kHz), <= kMaxIrLen
inline constexpr uint32_t kIrSeed       = 0x00A02DE7u;  // the pinned diffuse-tail noise seed
inline constexpr int      kTailStart    = 900;     // sample offset where the diffuse noise tail begins
inline constexpr int32_t  kTailGainQ15  = 17000;   // the tail envelope's starting Q15 amplitude
inline constexpr int32_t  kTailDecayQ15 = 32690;   // per-sample geometric decay (~0.99762), NON-INCREASING

// The discrete early reflections: {sample offset, Q15 gain}. Direct path first, then decreasing bounces
// (all offsets < kTailStart so the early cluster and the diffuse tail occupy disjoint IR regions).
struct EarlyTap { int offset; int32_t gainQ15; };
inline constexpr int kEarlyTapCount = 5;
inline constexpr EarlyTap kEarlyTaps[kEarlyTapCount] = {
    {   0, 32767},   // direct path (near-unity)
    { 149, 22000},   // first reflection
    { 293, 16000},
    { 521, 11000},
    { 787,  7000},   // last early reflection (< kTailStart = 900)
};

// SynthRoomIR: build the Q15 IR (int32 coefficients). Early taps written at their offsets, then the
// decaying-noise tail. Pure integer; deterministic in (irLen, seed).
inline std::vector<int32_t> SynthRoomIR(int irLen = kIrLen, uint32_t seed = kIrSeed) {
    if (irLen < 1) irLen = 1;
    if (irLen > kMaxIrLen) irLen = kMaxIrLen;
    std::vector<int32_t> ir(static_cast<std::size_t>(irLen), 0);

    // Early reflections (discrete taps).
    for (int t = 0; t < kEarlyTapCount; ++t)
        if (kEarlyTaps[t].offset < irLen)
            ir[static_cast<std::size_t>(kEarlyTaps[t].offset)] += kEarlyTaps[t].gainQ15;

    // Diffuse decaying-noise tail: env is a Q15 geometric decay (non-increasing, truncating to silence).
    int32_t env = kTailGainQ15;
    for (int i = kTailStart; i < irLen; ++i) {
        const int32_t nz = Au2NoiseQ15(seed, static_cast<uint32_t>(i));
        ir[static_cast<std::size_t>(i)] += dsp::MulQ15(nz, env);
        env = dsp::MulQ15(env, kTailDecayQ15);
    }
    return ir;
}

// SynthRoomIRTailEnvelope: the tail's Q15 decay envelope (0 before kTailStart, the geometric decay
// after) — exposed so the test can PIN its monotonic non-increasing decay profile without recomputing.
inline std::vector<int32_t> SynthRoomIRTailEnvelope(int irLen = kIrLen) {
    if (irLen < 1) irLen = 1;
    if (irLen > kMaxIrLen) irLen = kMaxIrLen;
    std::vector<int32_t> env(static_cast<std::size_t>(irLen), 0);
    int32_t e = kTailGainQ15;
    for (int i = kTailStart; i < irLen; ++i) {
        env[static_cast<std::size_t>(i)] = e;
        e = dsp::MulQ15(e, kTailDecayQ15);
    }
    return env;
}

// =====================================================================================================
// (3)+(4) SUBMIX BUS GRAPH — named voice buses -> a shared reverb send -> a master sum. The submix-graph
// gap. Each bus has a Q15 output gain, a mute, and a Q15 REVERB-SEND level; the buses route in id-
// ASCENDING order into (a) a DRY master accumulator and (b) a shared reverb-SEND accumulator that is
// convolved ONCE with the room IR into the WET signal; the master sums dry + wet, scaled by masterGain.
// -----------------------------------------------------------------------------------------------------
// DETERMINISTIC MIX ORDER: buses are summed in id-ascending order (a stable sort by id). Because each
// bus contributes via int32 ACCUMULATION with a SINGLE clamp at the end (the mixer.cpp discipline),
// the sum is order-INDEPENDENT — reordering the input buses (same set) yields a BIT-IDENTICAL master.
// We pin BOTH the canonical id-order digest AND that permutation invariance.

struct SubmixBus {
    uint32_t             id            = 0;          // routing id (ascending == the canonical mix order)
    int32_t              gainQ15       = kQ15One;    // the bus output gain (Q15; kQ15One == unity)
    int32_t              reverbSendQ15 = 0;          // send level to the shared reverb bus (0 == dry-only)
    bool                 muted         = false;      // a muted bus contributes NOTHING (dry or send)
    std::vector<int16_t> voice;                      // the bus's mono input signal (PCM)
};

struct SubmixResult {
    std::vector<int16_t> master;   // the final mixed master (dry + wet, master-gain scaled)
    std::vector<int16_t> dry;      // the summed dry path (clamped), length dryLen
    std::vector<int16_t> wet;      // the convolved reverb send, length dryLen + irLen - 1
    uint32_t             buses  = 0;
    int                  dryLen = 0;
    int                  wetLen = 0;
    int                  irLen  = 0;
    uint64_t             digest = 0;   // dsp::DigestBuffer(master) — the pinned golden currency
};

// StableSortBusesById: a tiny deterministic insertion sort by id (stable, no <algorithm> to keep the
// standalone-clang compile minimal). Ascending id == the canonical deterministic mix order.
inline void StableSortBusesById(std::vector<SubmixBus>& buses) {
    for (std::size_t i = 1; i < buses.size(); ++i) {
        SubmixBus key = buses[i];
        std::size_t j = i;
        while (j > 0 && buses[j - 1].id > key.id) { buses[j] = buses[j - 1]; --j; }
        buses[j] = key;
    }
}

// MixSubmix: route the buses -> dry + reverb-send -> convolve -> master. Pure integer; deterministic.
inline SubmixResult MixSubmix(std::vector<SubmixBus> buses, const std::vector<int32_t>& ir,
                              int32_t masterGainQ15 = kQ15One) {
    SubmixResult r;
    r.buses = static_cast<uint32_t>(buses.size());
    r.irLen = static_cast<int>(ir.size());

    StableSortBusesById(buses);   // canonical id-ascending order

    // dryLen = the longest voice among the buses (shorter voices read silence past their end).
    int dryLen = 0;
    for (const SubmixBus& b : buses)
        if (static_cast<int>(b.voice.size()) > dryLen) dryLen = static_cast<int>(b.voice.size());
    r.dryLen = dryLen;

    if (dryLen <= 0) { r.digest = dsp::DigestBuffer(r.master); return r; }   // empty submix == silence

    // int32 accumulators (the mixer.cpp discipline: accumulate wide, clamp ONCE) — order-independent.
    std::vector<int32_t> dryAcc(static_cast<std::size_t>(dryLen), 0);
    std::vector<int32_t> sendAcc(static_cast<std::size_t>(dryLen), 0);
    for (const SubmixBus& b : buses) {
        if (b.muted) continue;
        const int vn = static_cast<int>(b.voice.size());
        for (int i = 0; i < vn; ++i) {
            const int32_t s = dsp::MulQ15(b.voice[static_cast<std::size_t>(i)], b.gainQ15);  // bus gain
            dryAcc[static_cast<std::size_t>(i)]  += s;
            sendAcc[static_cast<std::size_t>(i)] += dsp::MulQ15(s, b.reverbSendQ15);          // send tap
        }
    }

    // The dry path (clamped once) + the reverb send (clamped once, then convolved with the IR).
    r.dry.assign(static_cast<std::size_t>(dryLen), 0);
    std::vector<int16_t> sendBuf(static_cast<std::size_t>(dryLen), 0);
    for (int i = 0; i < dryLen; ++i) {
        r.dry[static_cast<std::size_t>(i)]  = dsp::ClampI16(dryAcc[static_cast<std::size_t>(i)]);
        sendBuf[static_cast<std::size_t>(i)] = dsp::ClampI16(sendAcc[static_cast<std::size_t>(i)]);
    }
    r.wet    = Convolve(sendBuf, ir);                 // wetLen = dryLen + irLen - 1 (or empty if irLen==0)
    r.wetLen = static_cast<int>(r.wet.size());

    // Master = clamp( (dry + wet) * masterGain ). At masterGain == unity and one unity bus with send==0,
    // master == the bus voice BIT-EXACTLY (the dry/identity pin). Length = max(dryLen, wetLen).
    const int outLen = (r.wetLen > dryLen) ? r.wetLen : dryLen;
    r.master.assign(static_cast<std::size_t>(outLen), 0);
    for (int n = 0; n < outLen; ++n) {
        const int32_t d = (n < dryLen)   ? r.dry[static_cast<std::size_t>(n)] : 0;
        const int32_t w = (n < r.wetLen) ? r.wet[static_cast<std::size_t>(n)] : 0;
        r.master[static_cast<std::size_t>(n)] = dsp::ClampI16(dsp::MulQ15(d + w, masterGainQ15));
    }
    r.digest = dsp::DigestBuffer(r.master);
    return r;
}

// =====================================================================================================
// (5) IDENTITY helpers (exposed for the pinned tests):
//   * UnitImpulseIR(len): {32768, 0, 0, ...} — Convolve(dry, that) == dry (bit-exact, up to the tail
//     of zeros). THE convolution-identity pin.
//   * A zero-send bus (reverbSendQ15 == 0) -> the master is the dry path (no wet). THE bus-identity pin.
//   * An empty submix (no buses / all muted) -> silence.
// -----------------------------------------------------------------------------------------------------
inline std::vector<int32_t> UnitImpulseIR(int len = 8) {
    if (len < 1) len = 1;
    std::vector<int32_t> ir(static_cast<std::size_t>(len), 0);
    ir[0] = kQ15One;   // EXACT 1.0 — the identity coefficient (needs int32; 32768 > int16 max)
    return ir;
}

// =====================================================================================================
// THE AU2 SHOWCASE SCENARIO (shared by the Vulkan --au2-reverb-shot and the Metal --au2-reverb-shot so
// the viz bytes are IDENTICAL cross-backend BY CONSTRUCTION — the crowd.h RunCrowdShotScenario pattern):
// a dry CLICK convolved through the synthetic room IR into a decaying reverb tail, plus a 3-voice submix
// (voices -> a shared reverb send -> master). Pure integer; two runs byte-identical.
// =====================================================================================================
struct Au2ShotRun {
    std::vector<int16_t> dry;      // the dry click (top viz lane)
    std::vector<int32_t> ir;       // the room IR, Q15 (middle viz lane)
    std::vector<int16_t> wet;      // the reverb of the click (bottom viz lane)
    std::vector<int16_t> master;   // the 3-voice submix master (not drawn; feeds the digest + stats)
    int      irLen  = 0;
    int      dryLen = 0;
    int      wetLen = 0;
    uint32_t buses  = 0;
    uint64_t digest = 0;           // combined digest over dry ++ ir(as int16) ++ wet ++ master
};

inline constexpr int kShotDryLen = 96;   // the dry click length (a sharp impulse + a fast-decaying ring)

// MakeShotClick: a deterministic click — a full-scale impulse followed by a fast Q15-decaying ring.
inline std::vector<int16_t> MakeShotClick() {
    std::vector<int16_t> dry(static_cast<std::size_t>(kShotDryLen), 0);
    dry[0] = 32767;
    for (int i = 1; i < kShotDryLen; ++i)
        dry[static_cast<std::size_t>(i)] = dsp::ClampI16(dsp::MulQ15(dry[static_cast<std::size_t>(i - 1)], 29000));
    return dry;
}

inline Au2ShotRun RunAu2ShotScenario() {
    Au2ShotRun run;
    run.dry = MakeShotClick();
    run.ir  = SynthRoomIR();                 // the pinned room IR
    run.wet = Convolve(run.dry, run.ir);     // the reverb of the click

    // A 3-voice submix: three copies of the click at descending gains and ascending reverb sends, all
    // routed through the shared room IR to a unity master. buses == 3 (the submix-graph stat).
    std::vector<SubmixBus> buses;
    buses.push_back(SubmixBus{0, 28000,  4000, false, run.dry});   // near/dry-ish voice
    buses.push_back(SubmixBus{1, 20000, 12000, false, run.dry});   // mid, more send
    buses.push_back(SubmixBus{2, 14000, 24000, false, run.dry});   // far, wettest
    const SubmixResult sm = MixSubmix(buses, run.ir, kQ15One);
    run.master = sm.master;

    run.irLen  = static_cast<int>(run.ir.size());
    run.dryLen = static_cast<int>(run.dry.size());
    run.wetLen = static_cast<int>(run.wet.size());
    run.buses  = sm.buses;

    // Combined digest: concatenate dry, ir (clamped to int16 for the digest currency), wet, master.
    std::vector<int16_t> all;
    all.reserve(run.dry.size() + run.ir.size() + run.wet.size() + run.master.size());
    for (int16_t s : run.dry) all.push_back(s);
    for (int32_t c : run.ir)  all.push_back(dsp::ClampI16(c));
    for (int16_t s : run.wet) all.push_back(s);
    for (int16_t s : run.master) all.push_back(s);
    run.digest = dsp::DigestBuffer(all);
    return run;
}

// =====================================================================================================
// THE SHARED INTEGER WAVEFORM VIZ (RenderReverbShot) — three stacked lanes: top = the dry click, middle
// = the room IR (early taps + decaying tail), bottom = the wet convolved output. Pure integer BGRA
// (top-row-first), so both backends' shots are BYTE-IDENTICAL by construction (strict zero-differing-
// pixel). NO shader, NO float. Each lane peak-picks its buffer into W columns and draws a vertical bar
// from the lane center to the (auto-scaled) sample amplitude.
// =====================================================================================================
inline constexpr uint32_t kShotW     = 960;
inline constexpr uint32_t kShotLaneH = 200;
inline constexpr uint32_t kShotH     = kShotLaneH * 3;   // 600

// A packed BGRA color.
struct RgbaC { uint8_t b, g, r, a; };

inline void RenderReverbShot(const Au2ShotRun& run, std::vector<uint8_t>& img,
                             uint32_t& outW, uint32_t& outH) {
    const uint32_t W = kShotW, H = kShotH, laneH = kShotLaneH;
    outW = W; outH = H;
    img.assign(static_cast<std::size_t>(W) * H * 4u, 0);

    auto put = [&](uint32_t x, uint32_t y, RgbaC c) {
        if (x >= W || y >= H) return;
        const std::size_t p = (static_cast<std::size_t>(y) * W + x) * 4u;
        img[p + 0] = c.b; img[p + 1] = c.g; img[p + 2] = c.r; img[p + 3] = c.a;
    };

    const RgbaC kBg{18, 16, 14, 255};       // dark background
    const RgbaC kCenter{70, 70, 70, 255};   // lane center line (gray)
    const RgbaC kSep{40, 40, 40, 255};      // lane separators
    const RgbaC kDry{200, 200, 80, 255};    // dry lane (cyan-ish: high B+G)
    const RgbaC kIr{40, 170, 240, 255};     // IR lane (amber: high R+G)
    const RgbaC kWet{90, 230, 120, 255};    // wet lane (green)

    // Background.
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) put(x, y, kBg);

    // Draw one lane: laneIdx in [0,3), reading sample i via readFn (returns int32), length len, color.
    auto drawLane = [&](uint32_t laneIdx, int len, int32_t maxAbs, RgbaC col,
                        auto readFn) {
        const uint32_t y0 = laneIdx * laneH;
        const uint32_t yc = y0 + laneH / 2;                 // lane center row
        const int32_t half = static_cast<int32_t>(laneH / 2 - 4);   // amplitude half-height
        if (maxAbs < 1) maxAbs = 1;

        // Center line + top separator.
        for (uint32_t x = 0; x < W; ++x) { put(x, yc, kCenter); put(x, y0, kSep); }

        if (len <= 0) return;
        for (uint32_t x = 0; x < W; ++x) {
            // Column x maps to sample window [s0, s1).
            const int s0 = static_cast<int>((static_cast<int64_t>(x)     * len) / W);
            int       s1 = static_cast<int>((static_cast<int64_t>(x + 1) * len) / W);
            if (s1 <= s0) s1 = s0 + 1;
            if (s1 > len) s1 = len;
            int32_t peakPos = 0, peakNeg = 0;
            for (int s = s0; s < s1; ++s) {
                const int32_t v = readFn(s);
                if (v > peakPos) peakPos = v;
                if (v < peakNeg) peakNeg = v;
            }
            // Scale peaks to pixels (integer): pix = peak * half / maxAbs.
            const int32_t up = static_cast<int32_t>((static_cast<int64_t>(peakPos) * half) / maxAbs);
            const int32_t dn = static_cast<int32_t>((static_cast<int64_t>(-peakNeg) * half) / maxAbs);
            for (int32_t dy = 0; dy <= up; ++dy) put(x, yc - static_cast<uint32_t>(dy), col);
            for (int32_t dy = 0; dy <= dn; ++dy) put(x, yc + static_cast<uint32_t>(dy), col);
        }
    };

    // maxAbs per lane (auto-scale each waveform to its lane).
    auto maxAbs16 = [](const std::vector<int16_t>& v) -> int32_t {
        int32_t m = 1;
        for (int16_t s : v) { int32_t a = s < 0 ? -static_cast<int32_t>(s) : s; if (a > m) m = a; }
        return m;
    };
    auto maxAbs32 = [](const std::vector<int32_t>& v) -> int32_t {
        int32_t m = 1;
        for (int32_t s : v) { int32_t a = s < 0 ? -s : s; if (a > m) m = a; }
        return m;
    };

    drawLane(0, run.dryLen, maxAbs16(run.dry), kDry,
             [&](int s) -> int32_t { return run.dry[static_cast<std::size_t>(s)]; });
    drawLane(1, run.irLen, maxAbs32(run.ir), kIr,
             [&](int s) -> int32_t { return run.ir[static_cast<std::size_t>(s)]; });
    drawLane(2, run.wetLen, maxAbs16(run.wet), kWet,
             [&](int s) -> int32_t { return run.wet[static_cast<std::size_t>(s)]; });
}

}  // namespace hf::audio::reverb
