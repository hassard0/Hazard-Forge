#pragma once
// Slice AC1 — SERVER-AUTHORITATIVE RE-SIMULATION VERIFIER (provable anti-cheat), hf::net.
//
// THE STRATEGIC MOAT: because the whole Hazard Forge simulation is BIT-EXACT deterministic (every sim +
// the VD1-VD6 whole-world gameplay/physics tick is bit-identical CPU/Vulkan/Metal AND lockstep/rollback-
// replayable from an input stream ALONE), a SERVER can INGEST a suspect client's input stream, RE-SIMULATE
// it authoritatively, and PROVE — via a per-tick DIGEST comparison — whether the client faked its
// physics/gameplay outcomes. A diverging peer is rejected at the EXACT tick of divergence. UE5 (float
// Chaos/Mass, FPU-order-dependent) STRUCTURALLY CANNOT do this: a server cannot re-derive a client's float
// physics bit-for-bit, so "provable anti-cheat by re-simulation" is a category UE5 is disqualified from.
//
// THE SUBSTRATE IS ALREADY SHIPPED — AC1 only PACKAGES it into the adversarial verifier product, composing
// three frozen pieces READ-ONLY (this header adds NO field and edits NO existing function):
//   * verdict::RunVerdictLockstep / SimVerdictTick — the WHOLE-WORLD deterministic re-simulation (VD5): a
//     peer fed only commands re-derives the entities + every component pool + the embedded physics sim
//     byte-for-byte every tick. AC1's server re-sim IS SimVerdictTick over the client's submitted stream.
//   * verdict::DigestSnapshot — the order[]-keyed 16-hex FNV-1a fingerprint of the WHOLE final world (DX5).
//     AC1 computes ONE per authoritative tick -> a per-tick server digest TRACE.
//   * net::DesyncDetector / RecordLocal / IngestRemote (NS5) — the per-tick digest-exchange desync locator
//     that latches the EARLIEST tick whose local (authoritative) and remote (client-claimed) digests
//     disagree. AC1 feeds the server trace as "local" and the client's claimed digests as "remote", so the
//     located-divergence machinery is LITERALLY the NS5 detector — no new comparison logic.
//
// THE CHEAT MODEL (stated honestly): a client SUBMITS (a) the initial world snapshot, (b) the input stream
// it claims it played, and (c) its CLAIMED per-tick outcome digests. The server re-simulates (a)+(b) and
// compares its OWN per-tick digests against (c). A cheat is modeled as a CLAIMED-DIGEST stream that the
// submitted inputs CANNOT produce — either a hand-tampered single digest, OR (the realistic form) the
// digest trace of a DIFFERENT ("cheated") stream in which the client gave itself an impossible outcome
// (extra health / a hit that never landed / a teleport). Either way the server re-sim of the SUBMITTED
// inputs diverges from the claim at the EXACT tick the lie was injected -> REJECTED with the located tick.
//
// THE BOUNDARY (do not oversell): this proves SIMULATION INTEGRITY — that the client ran the REAL
// deterministic sim on the STATED inputs and reported the true outcomes. It does NOT by itself prevent
// INPUT-LEVEL cheats (an aimbot that picks superhuman-but-legal inputs): those produce a stream the server
// re-sim AGREES with. Input-level cheat detection is a separate (statistical/heuristic) layer. AC1's claim
// is precise and category-defining: outcomes cannot be faked, because they are re-derivable and checked.
//
// PURE CPU INTEGER (strict determinism tier). NO new render RHI, NO new shader, NO new compute. session.h /
// verdict.h are #included READ-ONLY / BYTE-UNCHANGED; authority_verify.h is a brand-new additive sibling.

#include <cstdint>
#include <cstdio>     // std::snprintf (the 16-hex commitment)
#include <cstdlib>    // std::strtoull (hex digest -> the NS5 uint64 currency)
#include <string>
#include <vector>

#include "net/session.h"     // read-only: NS5 DesyncDetector / ChecksumPacket / RecordLocal / IngestRemote +
                             // DigestBytes (the located-divergence machinery AC1 composes)
#include "game/verdict.h"    // read-only: the VD1-VD6 whole-world deterministic sim — ClonePeer /
                             // SimVerdictTick / SnapshotWorld / DigestSnapshot / VerdictParams / Command /
                             // BuildCanonicalReplay (the re-simulation AC1 verifies against)

namespace hf {
namespace net {

// Alias the frozen gameplay/netcode namespace locally (READ-ONLY compose).
namespace verdict = hf::game::verdict;

// ----- HexToDigestU64: parse a 16-hex DigestSnapshot string into the NS5 uint64 digest currency ----------
// verdict::DigestSnapshot returns a lowercase 16-hex string (64 bits, an exact bijection with uint64). NS5's
// DesyncDetector compares uint64 digests, so AC1 converts each per-tick digest string back to its uint64 to
// feed RecordLocal / IngestRemote VERBATIM. A malformed/short string parses to a deterministic value (0 for
// empty) -> the compare then simply flags a divergence (never a crash). Pure integer, no float.
inline uint64_t HexToDigestU64(const std::string& hex) {
    if (hex.empty()) return 0ull;
    return (uint64_t)std::strtoull(hex.c_str(), nullptr, 16);
}

// ----- VerdictDigestTrace: the AUTHORITATIVE per-tick server digest trace (the NS5 DigestTrace twin) ------
// RE-SIMULATE the whole world from `initial` applying `inputStream` for `ticks` ticks, recording
// verdict::DigestSnapshot AFTER each tick -> a per-tick digest trace of length `ticks`. This is the server's
// GROUND TRUTH: what the world's outcome digest MUST be at each tick given the inputs. It mirrors NS5's
// DigestTrace (run + digest-after-each-tick) but is specialized for the NON-COPYABLE VerdictWorld — the peer
// is materialized by verdict::ClonePeer(initial, params) (the VD4 determinism-faithful clone, NOT a copy),
// then advanced by verdict::SimVerdictTick VERBATIM. Deterministic of (initial, params, inputStream, ticks)
// alone, so two servers (or two runs, or two backends) produce the IDENTICAL trace. Pure CPU integer.
inline std::vector<std::string> VerdictDigestTrace(const verdict::VerdictSnapshot& initial,
                                                   const verdict::VerdictParams& params,
                                                   const std::vector<verdict::Command>& inputStream,
                                                   uint32_t ticks) {
    verdict::VerdictWorld w = verdict::ClonePeer(initial, params);   // the VD4 clone (NOT a copy; hulls seeded)
    std::vector<std::string> trace;
    trace.reserve((std::size_t)ticks);
    for (uint32_t t = 0; t < ticks; ++t) {
        verdict::SimVerdictTick(w, params, inputStream, t);          // the FROZEN VD5 whole-world tick, verbatim
        trace.push_back(verdict::DigestSnapshot(verdict::SnapshotWorld(w)));  // the outcome digest AFTER tick t
    }
    return trace;
}

// ----- CommitInputStream: a digest COMMITMENT over the client's input stream (integrity pin) --------------
// A 16-hex FNV-1a fold over the FROZEN Command wire fields {tick, kind, target, arg} in array order (reusing
// verdict::DigestFnv — the same FNV as DigestSnapshot). A client publishes this commitment BEFORE the match;
// it then CANNOT retroactively change what it "played" without flipping the commitment (a changed/added/
// removed/reordered command yields a different digest). Pure integer, endianness-independent (DigestFnv
// folds bytes LSB-first). This pins WHICH inputs were submitted; the re-sim then pins the OUTCOMES.
inline std::string CommitInputStream(const std::vector<verdict::Command>& stream) {
    verdict::DigestFnv d;
    d.mix32((uint32_t)stream.size());
    for (std::size_t i = 0; i < stream.size(); ++i) {
        const verdict::Command& c = stream[i];
        d.mix32(c.tick);
        d.mix32(c.kind);
        d.mix32((uint32_t)c.target);
        verdict::DigestVec3(d, c.arg);
        d.sep();
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)d.h);
    return std::string(buf);
}

// ----- VerifyResult: the server's adversarial verdict on one client submission ----------------------------
// `ok` — the client is HONEST (its claimed outcomes match the authoritative re-sim on its stated inputs).
// `firstDivergentTick` — the EXACT tick of the FIRST lie (-1 if clean); the "caught you at tick T" proof.
// `serverDigest` / `clientDigest` — the two diverging digests at that tick ("" if clean) — the evidence.
// `ticks` — the number of authoritative ticks verified.
struct VerifyResult {
    bool        ok                 = false;
    int         firstDivergentTick = -1;
    std::string serverDigest;
    std::string clientDigest;
    uint32_t    ticks              = 0;
};

inline bool VerifyResultsEqual(const VerifyResult& a, const VerifyResult& b) {
    return a.ok == b.ok && a.firstDivergentTick == b.firstDivergentTick &&
           a.serverDigest == b.serverDigest && a.clientDigest == b.clientDigest && a.ticks == b.ticks;
}

// ----- AuthorityVerifier::Verify — RE-SIMULATE the client's inputs + LOCATE the first faked outcome -------
// Given (a) the authoritative initial world snapshot, (b) the client-SUBMITTED input stream, (c) the
// client-CLAIMED per-tick outcome digests: RE-SIMULATE (a)+(b) authoritatively (VerdictDigestTrace), then
// COMPOSE the NS5 DesyncDetector — RecordLocal the server (authoritative) trace, IngestRemote each client
// ChecksumPacket in tick order — to latch the EARLIEST divergent tick. The NS5 detector's !desynced guard +
// the tick-ascending ingest give the FIRST (causal) divergence for free.
//
//   * HONEST: claim == the server trace -> no divergence -> ok, firstDivergentTick == -1.
//   * CHEAT: a tampered/impossible claim -> the re-sim of the SUBMITTED inputs diverges from the claim at the
//     EXACT tick the lie was injected -> !ok, firstDivergentTick == that tick, both digests reported.
//   * CAUSAL: a lie at tick T leaves ticks < T matching (the sim is deterministic; the true outcome at t<T
//     is unaffected by a later fabricated claim) -> the located tick is exactly T, never earlier.
//   * INCOMPLETE CLAIM: a client that claims FEWER than `ticks` digests cannot dodge detection — the first
//     UNCLAIMED tick counts as a divergence (its outcome is unverifiable == unproven). An EMPTY stream
//     (ticks == 0) verifies trivially (nothing to prove -> ok, -1).
// Deterministic of (initial, params, submittedInputs, claimedDigests, ticks) alone -> a third party re-runs
// Verify and gets the IDENTICAL verdict (the reproducible-fairness property UE5 cannot offer). Pure CPU.
inline VerifyResult Verify(const verdict::VerdictSnapshot& initial, const verdict::VerdictParams& params,
                           const std::vector<verdict::Command>& submittedInputs,
                           const std::vector<std::string>& claimedDigests, uint32_t ticks) {
    VerifyResult res;
    res.ticks = ticks;

    // (1) the AUTHORITATIVE re-simulation: the server's ground-truth per-tick outcome digests.
    const std::vector<std::string> server = VerdictDigestTrace(initial, params, submittedInputs, ticks);

    // (2) locate the first divergence by COMPOSING the NS5 desync detector: the server trace is "local"
    // (our authoritative record), the client's claimed digests are the "remote" ChecksumPacket stream.
    DesyncDetector det;
    for (uint32_t t = 0; t < ticks; ++t)
        RecordLocal(det, t, HexToDigestU64(server[(std::size_t)t]));
    const uint32_t claimN = (uint32_t)claimedDigests.size();
    const uint32_t common = (claimN < ticks) ? claimN : ticks;
    for (uint32_t t = 0; t < common; ++t)
        IngestRemote(det, ChecksumPacket{t, HexToDigestU64(claimedDigests[(std::size_t)t])});

    if (det.desynced) {
        res.ok = false;
        res.firstDivergentTick = (int)det.desyncTick;
        res.serverDigest = server[(std::size_t)det.desyncTick];
        res.clientDigest = claimedDigests[(std::size_t)det.desyncTick];
        return res;
    }
    // No claimed digest diverged. If the client under-claimed (fewer than `ticks`), the first UNCLAIMED tick
    // is unproven -> a rejection located at that tick (its true outcome vs an absent claim).
    if (claimN < ticks) {
        res.ok = false;
        res.firstDivergentTick = (int)claimN;
        res.serverDigest = server[(std::size_t)claimN];
        res.clientDigest = std::string();   // no claim for this tick
        return res;
    }
    // Every tick claimed AND matched -> the client is HONEST (it ran the real sim on the stated inputs).
    res.ok = true;
    res.firstDivergentTick = -1;
    return res;
}

// =================================================================================================
// THE CANONICAL ADVERSARIAL SCENARIO — the shared HEADLINE PROOF (the showcase + the test both build it).
// Two clients submit input streams for the SAME match against the SAME authoritative initial world:
//   * client 0 (HONEST): claims the true per-tick outcome digests -> VERIFIED (firstDivergentTick == -1).
//   * client 1 (CHEATER): submits the SAME honest inputs but CLAIMS the digest trace of a DIFFERENT
//     ("cheated") stream in which it gave its player an impossible Health bump at kAc1CheatTick (a
//     "claimed more health than the inputs produce" lie) -> the server re-sim of the submitted inputs
//     diverges from the claim at EXACTLY kAc1CheatTick -> REJECTED @ that tick.
// The verification is itself bit-exact/reproducible (a third party re-runs it -> the identical verdict),
// cross-platform. This is provable fairness UE5 structurally cannot offer.
// =================================================================================================

inline constexpr uint32_t kAc1CheatTick = 6u;   // the tick the cheater injects its impossible outcome

// Ac1Scenario: the fully-built canonical scenario (both clients' inputs + claims + the authoritative pieces).
struct Ac1Scenario {
    verdict::VerdictParams        params;
    std::vector<verdict::Command> honestInputs;    // the stream BOTH clients SUBMIT (the real inputs)
    std::vector<verdict::Command> cheatInputs;     // the DIFFERENT stream the cheater's claim is computed from
    std::vector<std::string>      honestClaim;     // client 0's claimed digests (the true trace)
    std::vector<std::string>      cheaterClaim;    // client 1's claimed digests (the cheated trace)
    std::string                   inputCommitment; // CommitInputStream(honestInputs) — the integrity pin
    uint32_t                      ticks  = 0;
    verdict::EntityId             player = verdict::kNoEntity;
};

// BuildAc1Scenario(world0): assemble the canonical adversarial scenario over the FROZEN
// verdict::BuildCanonicalReplay world (the same composed gameplay+physics scene DX5/DX6 use, so EVERY
// component type + the embedded sim participate in the digests). world0 is filled in place (VerdictWorld is
// non-copyable). Returns the scenario; the returned snapshot is taken by the caller from world0.
inline Ac1Scenario BuildAc1Scenario(verdict::VerdictWorld& world0) {
    const verdict::CanonicalReplay cr = verdict::BuildCanonicalReplay(world0);
    const verdict::VerdictSnapshot w0Snap = verdict::SnapshotWorld(world0);

    Ac1Scenario sc;
    sc.params = cr.params;
    sc.ticks  = cr.ticks;
    sc.player = cr.player;
    sc.honestInputs = cr.stream;

    // The cheater SUBMITS the honest inputs but COMPUTES its claim from a cheated stream: an extra
    // kCmdAbility at kAc1CheatTick that bumps the player's Health by an impossible +50 (arg.x un-fixed >>16
    // by kCmdAbility). Its claimed digests therefore diverge from the honest inputs' true outcome AT that
    // tick and thereafter -> the "claimed more health than my inputs produce" lie.
    sc.cheatInputs = cr.stream;
    verdict::Command bump;
    bump.tick   = kAc1CheatTick;
    bump.kind   = verdict::kCmdAbility;
    bump.target = cr.player;
    bump.arg    = verdict::FxVec3{(verdict::fx)((int64_t)50 * (int64_t)verdict::kOne), 0, 0};  // +50 hp
    sc.cheatInputs.push_back(bump);

    // The two clients' CLAIMED per-tick digest traces (each client's asserted outcomes).
    sc.honestClaim  = VerdictDigestTrace(w0Snap, sc.params, sc.honestInputs, sc.ticks);   // the TRUE trace
    sc.cheaterClaim = VerdictDigestTrace(w0Snap, sc.params, sc.cheatInputs, sc.ticks);    // the CHEATED trace

    sc.inputCommitment = CommitInputStream(sc.honestInputs);
    return sc;
}

// =================================================================================================
// THE SHOWCASE VIZ — a strict-integer VERIFICATION-REPORT image (NO shader, NO float, NO <cmath>).
// Two lanes (honest client, cheater client) over the ticks; each tick a cell colored by server-vs-client
// digest match (green == match, red == divergence); a verdict BANNER swatch per client (green VERIFIED /
// red REJECTED) + a bright marker column at the caught tick; and a per-tick SERVER digest strip (each tick
// tinted by its digest bytes — a visual fingerprint of the authoritative outcome). Rendered by pure integer
// pixel writes into an RGBA8 buffer, shared VERBATIM by the Vulkan --ac1-verify-shot and the Metal
// --ac1-verify so the pixels are byte-identical cross-backend BY CONSTRUCTION (the sq2/pt1 precedent).
// =================================================================================================

struct Ac1VizStats {
    uint32_t    ticks             = 0;   // authoritative ticks verified
    uint32_t    clients           = 0;   // clients checked (2: honest + cheater)
    bool        honestVerdict     = false;  // client 0 VERIFIED?
    int         cheaterCaughtTick = -1;  // client 1 caught @ this tick (-1 if not caught)
    uint32_t    width             = 0;
    uint32_t    height            = 0;
    uint64_t    pixDigest         = 0;   // net::DigestBytes over the RGBA8 pixels (the strict-zero proof)
    std::string inputCommitment;         // the input-stream integrity commitment (pinned)
};

// Fixed viz geometry (all integer).
inline constexpr int kAc1ImgW   = 560;
inline constexpr int kAc1ImgH   = 300;
inline constexpr int kAc1StripX = 120;   // per-tick cell strip left edge
inline constexpr int kAc1StripW = 408;   // strip pixel width (spans all ticks)
inline constexpr int kAc1Lane0Y = 60;    // honest lane top
inline constexpr int kAc1Lane1Y = 150;   // cheater lane top
inline constexpr int kAc1LaneH  = 54;    // lane cell height
inline constexpr int kAc1BanX   = 12;    // verdict banner swatch left
inline constexpr int kAc1BanW   = 96;    // verdict banner swatch width

// RenderAc1VerifyViz: fill `out` (RGBA8, kAc1ImgW x kAc1ImgH) + the stat block. Builds the canonical
// adversarial scenario, verifies both clients, and draws the verification report. Deterministic + pure
// integer -> two calls (and two backends) byte-identical.
inline void RenderAc1VerifyViz(std::vector<uint8_t>& out, Ac1VizStats& stats) {
    const int W = kAc1ImgW, H = kAc1ImgH;
    out.assign((std::size_t)W * H * 4u, 0);

    auto px = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        uint8_t* d = &out[((std::size_t)y * W + x) * 4u];
        d[0] = r; d[1] = g; d[2] = b; d[3] = 255;
    };
    auto fillRect = [&](int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y < y0 + h; ++y)
            for (int x = x0; x < x0 + w; ++x) px(x, y, r, g, b);
    };
    auto vline = [&](int x, int y0, int y1, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y <= y1; ++y) px(x, y, r, g, b);
    };
    auto border = [&](int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int x = x0; x < x0 + w; ++x) { px(x, y0, r, g, b); px(x, y0 + h - 1, r, g, b); }
        for (int y = y0; y < y0 + h; ++y) { px(x0, y, r, g, b); px(x0 + w - 1, y, r, g, b); }
    };

    // Background: dark report gray.
    fillRect(0, 0, W, H, 16, 18, 24);

    // ---- Build + verify the canonical adversarial scenario. -----------------------------------------
    verdict::VerdictWorld world0;
    const Ac1Scenario sc = BuildAc1Scenario(world0);
    const verdict::VerdictSnapshot w0Snap = verdict::SnapshotWorld(world0);
    // The server's authoritative ground-truth trace (over the SUBMITTED honest inputs both clients sent).
    const std::vector<std::string> server = VerdictDigestTrace(w0Snap, sc.params, sc.honestInputs, sc.ticks);
    const VerifyResult honestRes  = Verify(w0Snap, sc.params, sc.honestInputs, sc.honestClaim,  sc.ticks);
    const VerifyResult cheaterRes = Verify(w0Snap, sc.params, sc.honestInputs, sc.cheaterClaim, sc.ticks);

    const uint32_t T = sc.ticks;
    // Cell X mapping (deterministic integer division): tick t -> [x0, x1).
    auto cellX0 = [&](uint32_t t) { return kAc1StripX + (int)(((int64_t)t * kAc1StripW) / (int64_t)(T ? T : 1)); };

    // ---- Header band: a thin title bar. -------------------------------------------------------------
    fillRect(0, 0, W, 40, 24, 27, 36);
    fillRect(kAc1StripX, 6, kAc1StripW, 4, 60, 66, 80);   // a strip-aligned title rule

    // ---- Per-lane render: (banner swatch) + (per-tick match strip) + (caught-tick marker). ----------
    struct Lane { int y; const VerifyResult* res; const std::vector<std::string>* claim; };
    const Lane lanes[2] = {
        { kAc1Lane0Y, &honestRes,  &sc.honestClaim  },
        { kAc1Lane1Y, &cheaterRes, &sc.cheaterClaim },
    };
    for (int li = 0; li < 2; ++li) {
        const Lane& ln = lanes[li];
        const int y0 = ln.y;

        // Verdict banner swatch: green VERIFIED (ok) / red REJECTED (caught).
        if (ln.res->ok) fillRect(kAc1BanX, y0, kAc1BanW, kAc1LaneH, 40, 150, 70);      // green
        else            fillRect(kAc1BanX, y0, kAc1BanW, kAc1LaneH, 170, 50, 50);      // red
        border(kAc1BanX, y0, kAc1BanW, kAc1LaneH, 230, 230, 230);
        // A verdict glyph-ish mark: a check (green) vs a cross (red), drawn as thick integer strokes.
        if (ln.res->ok) {
            const int cx = kAc1BanX + kAc1BanW / 2, cy = y0 + kAc1LaneH / 2;
            for (int i = 0; i < 12; ++i) { px(cx - 14 + i, cy + i - 2, 240, 255, 240); px(cx - 13 + i, cy + i - 2, 240, 255, 240); }
            for (int i = 0; i < 20; ++i) { px(cx - 2 + i, cy + 8 - i, 240, 255, 240); px(cx - 1 + i, cy + 8 - i, 240, 255, 240); }
        } else {
            const int cx = kAc1BanX + kAc1BanW / 2, cy = y0 + kAc1LaneH / 2;
            for (int i = -14; i <= 14; ++i) { px(cx + i, cy + i, 255, 240, 240); px(cx + i, cy + i + 1, 255, 240, 240);
                                              px(cx + i, cy - i, 255, 240, 240); px(cx + i, cy - i + 1, 255, 240, 240); }
        }

        // Lane lane-bg.
        fillRect(kAc1StripX, y0, kAc1StripW, kAc1LaneH, 30, 33, 42);

        // Per-tick match strip: cell green (server == claim) / red (divergence). The claim may be shorter
        // than ticks (an under-claim); an unclaimed tick is drawn amber (unproven).
        for (uint32_t t = 0; t < T; ++t) {
            const int x0 = cellX0(t), x1 = cellX0(t + 1);
            const int cw = (x1 - x0 > 1) ? (x1 - x0 - 1) : 1;
            uint8_t r, g, b;
            if (t >= ln.claim->size())                      { r = 200; g = 160; b = 40; }   // amber: unclaimed
            else if (server[(std::size_t)t] == (*ln.claim)[(std::size_t)t]) { r = 46; g = 150; b = 70; } // green: match
            else                                            { r = 175; g = 48; b = 48; }    // red: divergence
            fillRect(x0, y0 + 6, cw, kAc1LaneH - 12, r, g, b);
        }
        border(kAc1StripX, y0, kAc1StripW, kAc1LaneH, 70, 76, 92);

        // Caught-tick marker: a bright vertical divider at the located divergence (the "caught @ T" proof).
        if (ln.res->firstDivergentTick >= 0) {
            const int mx = cellX0((uint32_t)ln.res->firstDivergentTick);
            vline(mx, y0 - 8, y0 + kAc1LaneH + 8, 250, 230, 60);
            vline(mx + 1, y0 - 8, y0 + kAc1LaneH + 8, 250, 230, 60);
            // a small down-arrow head above the lane.
            for (int dy = 0; dy < 6; ++dy)
                for (int dx = -dy; dx <= dy; ++dx) px(mx + dx, y0 - 8 + dy, 250, 230, 60);
        }
    }

    // ---- Per-tick SERVER digest strip (bottom): each tick tinted by its authoritative digest bytes. ---
    {
        const int y0 = 236, hgt = 40;
        fillRect(kAc1StripX, y0, kAc1StripW, hgt, 22, 24, 32);
        for (uint32_t t = 0; t < T; ++t) {
            const int x0 = cellX0(t), x1 = cellX0(t + 1);
            const int cw = (x1 - x0 > 1) ? (x1 - x0 - 1) : 1;
            const uint64_t d = HexToDigestU64(server[(std::size_t)t]);
            const uint8_t r = (uint8_t)(d & 0xFF);
            const uint8_t g = (uint8_t)((d >> 21) & 0xFF);
            const uint8_t b = (uint8_t)((d >> 42) & 0xFF);
            fillRect(x0, y0 + 4, cw, hgt - 8, r, g, b);
        }
        border(kAc1StripX, y0, kAc1StripW, hgt, 70, 76, 92);
    }

    // ---- Stats. --------------------------------------------------------------------------------------
    stats.ticks             = T;
    stats.clients           = 2u;
    stats.honestVerdict     = honestRes.ok;
    stats.cheaterCaughtTick = cheaterRes.firstDivergentTick;
    stats.width             = (uint32_t)W;
    stats.height            = (uint32_t)H;
    stats.pixDigest         = hf::net::DigestBytes(out.data(), out.size());
    stats.inputCommitment   = sc.inputCommitment;
}

}  // namespace net
}  // namespace hf
