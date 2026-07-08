#pragma once
// engine/net/udp_transport.h — Slice NW1: REAL UDP TRANSPORT for the deterministic rollback netcode
// (flagship #24 NETCODE; the BEAT_UE5 "Rollback Kit" post-P0 step). hf::net.
//
// The whole rollback-netcode substrate is shipped and golden-gated: session.h (NS1-6 lockstep / input-
// delay / prediction+rollback / transport-agnostic ScriptedTransport / desync detector / late-join), the
// GAME1 duel (game/duel.h), transport.h (the SimChannel jitter buffer). BUT every transport so far is an
// EXPLICITLY-MODELED IN-PROCESS channel — a seeded LCG that *simulates* latency/jitter/loss, "still NO
// real sockets/UDP/TCP" (transport.h banner, session.h NS4 ScriptedTransport "with NO real sockets").
//
// NW1 adds a REAL UDP transport behind the SAME interface shape so the deterministic rollback netcode
// actually runs over real datagrams. The moat claim UE5's float architecture structurally CANNOT make:
// two peers, exchanging inputs over REAL UDP (real send/recv syscalls, real reorder/loss), each
// INDEPENDENTLY re-derive the IDENTICAL match/world digest — deterministic rollback OVER A REAL NETWORK.
//
// 🔵 HONEST DETERMINISM BOUNDARY (this is an INTEGRATION slice — real sockets are nondeterministic I/O):
//   * The SIMULATION is deterministic: inputs -> identical world (session.h / duel.h, bit-exact integer).
//   * The TRANSPORT is nondeterministic I/O: real UDP may reorder/drop/delay datagrams run-to-run. That
//     is the ONE documented nondeterministic-I/O exception in this codebase.
//   * The OUTCOME is bit-exact: the rollback layer (NS3-6) already tolerates arbitrary arrival order +
//     loss-with-resend, so BOTH peers converge to the SAME pinned digest regardless of transport timing.
//   That is the exact "beyond-UE5" claim: nondeterministic network, bit-exact deterministic outcome.
//
// TEST-RELIABILITY (non-flaky by construction): the gate is a SAME-PROCESS two-real-UDP-socket loopback —
// two sockets bound to 127.0.0.1 on OS-assigned ports exchanging real datagrams — which exercises the
// real socket send/recv path WITHOUT flaky cross-process orchestration. A bounded ACK+resend reliability
// loop guarantees eventual delivery (drop-then-resend converges), so the DIGEST assertions never flake. A
// separate documented script (benchmarks/udp_two_process.md) demonstrates the true two-process run.
//
// PLATFORM: Winsock on _WIN32 (WSAStartup/socket/bind/sendto/recvfrom/closesocket, link ws2_32); BSD
// sockets elsewhere. Non-blocking; a bounded drain-recv loop. Real net = unordered/lossy -> the receiver
// hands whatever arrives to the deterministic reassembly / rollback layer, already robust to it.
//
// This header composes game/duel.h + net/session.h READ-ONLY / BYTE-UNTOUCHED (udp_transport.h is a NEW
// impl of the transport interface; transport.h/session.h/duel.h are untouched). The pure-integer showcase
// report lives in net/nw1_report.h (NO socket code) so the giant sample TUs never pull <winsock2.h>.

#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
#else
  #include <arpa/inet.h>
  #include <fcntl.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <unistd.h>
  #include <cerrno>
#endif

#include "game/duel.h"      // read-only: the GAME1 duel (MakeRoundScript / BuildDuelScene / RunDuelRound)
#include "net/session.h"    // read-only: the NS3 RollbackSession (StepPredicted / ConfirmRemote)

namespace hf {
namespace net {
namespace udp {

// ============================ THE REAL SOCKET LAYER (Winsock / BSD behind one interface) =================
#ifdef _WIN32
using SockT = SOCKET;
inline constexpr SockT kInvalidSock = INVALID_SOCKET;
#else
using SockT = int;
inline constexpr SockT kInvalidSock = -1;
#endif

// WinsockGuard: RAII WSAStartup/WSACleanup (Windows only; a no-op elsewhere). Ref-counted so nested
// UdpTransport instances share ONE startup and clean up exactly once. Guarded so double-init is safe.
struct WinsockGuard {
    WinsockGuard() {
#ifdef _WIN32
        if (refs_++ == 0) {
            WSADATA wsa;
            ok_ = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
        }
#endif
    }
    ~WinsockGuard() {
#ifdef _WIN32
        if (--refs_ == 0) WSACleanup();
#endif
    }
    bool ok() const { return ok_; }
#ifdef _WIN32
    static inline int  refs_ = 0;
    bool ok_ = true;
#else
    bool ok_ = true;
#endif
};

// UdpTransport: ONE real UDP endpoint. Open() creates a socket, binds to 127.0.0.1:0 (OS-assigned port),
// and switches to NON-BLOCKING. SetPeer points sends at the other endpoint's port. Send() is a real
// sendto(); Poll() drains every datagram currently available (a bounded recvfrom loop), returning each as
// a byte vector. RAII: the socket is closed in the dtor. This is the ONE nondeterministic-I/O object.
class UdpTransport {
public:
    UdpTransport() = default;
    UdpTransport(const UdpTransport&) = delete;
    UdpTransport& operator=(const UdpTransport&) = delete;
    ~UdpTransport() { Close(); }

    // Create the socket, bind 127.0.0.1:0 (kernel picks a free port), make it non-blocking. Returns false
    // (and leaves the transport closed) on any failure — the caller reports it rather than flaking.
    bool Open() {
        if (!guard_.ok()) return false;
        sock_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == kInvalidSock) return false;

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // 127.0.0.1
        addr.sin_port        = 0;                          // OS-assigned
        if (::bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) { Close(); return false; }

        // Read back the actual bound port.
        sockaddr_in bound{};
#ifdef _WIN32
        int len = (int)sizeof bound;
#else
        socklen_t len = sizeof bound;
#endif
        if (::getsockname(sock_, reinterpret_cast<sockaddr*>(&bound), &len) != 0) { Close(); return false; }
        port_ = ntohs(bound.sin_port);

        // Non-blocking so Poll() never hangs when the recv queue is empty.
#ifdef _WIN32
        u_long nb = 1;
        if (::ioctlsocket(sock_, FIONBIO, &nb) != 0) { Close(); return false; }
#else
        int fl = ::fcntl(sock_, F_GETFL, 0);
        if (fl < 0 || ::fcntl(sock_, F_SETFL, fl | O_NONBLOCK) < 0) { Close(); return false; }
#endif
        return true;
    }

    // The OS-assigned bound port (host byte order). Valid after a successful Open().
    uint16_t Port() const { return port_; }

    // Point all future Send()s at 127.0.0.1:peerPort (the other loopback endpoint).
    void SetPeer(uint16_t peerPort) {
        peer_ = sockaddr_in{};
        peer_.sin_family      = AF_INET;
        peer_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        peer_.sin_port        = htons(peerPort);
        hasPeer_ = true;
    }

    // Real sendto() to the peer. Returns true if the datagram was handed to the kernel. A transient
    // would-block is treated as "not sent" (the reliability loop resends) — never fatal.
    bool Send(const uint8_t* data, std::size_t n) {
        if (sock_ == kInvalidSock || !hasPeer_) return false;
        const int sent = (int)::sendto(sock_, reinterpret_cast<const char*>(data), (int)n, 0,
                                       reinterpret_cast<const sockaddr*>(&peer_), (int)sizeof peer_);
        ++sentCount_;
        return sent == (int)n;
    }
    bool Send(const std::vector<uint8_t>& bytes) { return Send(bytes.data(), bytes.size()); }

    // Drain every datagram currently available (a bounded recvfrom loop), appending each to `out`. Returns
    // the number drained this call. Stops cleanly on would-block (empty queue) — never blocks.
    std::size_t Poll(std::vector<std::vector<uint8_t>>& out) {
        if (sock_ == kInvalidSock) return 0;
        std::size_t got = 0;
        uint8_t buf[2048];
        for (int guard = 0; guard < 100000; ++guard) {
            const int n = (int)::recvfrom(sock_, reinterpret_cast<char*>(buf), (int)sizeof buf, 0, nullptr, nullptr);
            if (n > 0) {
                out.emplace_back(buf, buf + n);
                ++got; ++recvCount_;
                continue;
            }
            break;   // n<=0: would-block (empty) or a transient error — stop draining this call
        }
        return got;
    }

    bool     IsOpen()    const { return sock_ != kInvalidSock; }
    uint64_t SentCount() const { return sentCount_; }
    uint64_t RecvCount() const { return recvCount_; }

    void Close() {
        if (sock_ != kInvalidSock) {
#ifdef _WIN32
            ::closesocket(sock_);
#else
            ::close(sock_);
#endif
            sock_ = kInvalidSock;
        }
    }

private:
    WinsockGuard guard_;                 // RAII WSAStartup/WSACleanup (Windows)
    SockT        sock_    = kInvalidSock;
    uint16_t     port_    = 0;
    sockaddr_in  peer_{};
    bool         hasPeer_ = false;
    uint64_t     sentCount_ = 0;
    uint64_t     recvCount_ = 0;
};

// ============================ FAULT INJECTION (deterministic, eventually-delivering) =====================
// To PROVE the rollback/reassembly layer tolerates real-network conditions WITHOUT making the test flaky,
// the harness can inject reorder/duplicate/drop-THEN-resend at the send boundary. Every fault is
// eventually-delivering: a dropped datagram is dropped ONLY on its first sight (so the reliability
// resend delivers it), a reorder holds one datagram for one pump, a duplicate sends a harmless extra copy
// (idempotent per-key data / a resend the rollback layer no-ops). So the loop always converges.
struct FaultConfig {
    bool enabled   = false;
    bool reorder   = true;   // hold every 3rd first-seen datagram one pump later (arrives out of order)
    bool duplicate = true;   // send a redundant copy of every 4th datagram (a harmless resend)
    bool dropOnce  = true;   // drop every 5th datagram the FIRST time only (drop-then-resend)
};

// FaultStats: what the injector actually did (for the report / honesty).
struct FaultStats {
    uint64_t reordered   = 0;
    uint64_t duplicated  = 0;
    uint64_t droppedOnce = 0;
};

// ============================ HARNESS 1 — THE GAME1 DUEL OVER REAL UDP ===================================
// Two same-process peers on 127.0.0.1: peer 0 owns PLAYER-0's per-round DuelInput scripts, peer 1 owns
// PLAYER-1's. They EXCHANGE their scripts over real UDP (one datagram per (round,tick), ACK+resend), then
// each INDEPENDENTLY reconstructs both scripts and computes the best-of-3 match digest (MatchDigestFrom
// Scripts, mirroring RunDuelMatch's fold over duel.h READ-ONLY). Both digests == the pinned GAME1
// matchDigest -> the real GAME1 match ran over the wire and converged bit-for-bit.

namespace gduel = hf::game::duel;

// One duel-input wire datagram: type(0=data,1=ack), round, tick, moveX(int32 LE), shove(0/1). Fixed 8B.
inline std::vector<uint8_t> EncodeDuelData(uint8_t round, uint8_t tick, const gduel::DuelInput& in) {
    std::vector<uint8_t> b(8);
    b[0] = 0;                                   // data
    b[1] = round;
    b[2] = tick;
    const int32_t mv = (int32_t)in.moveX;
    std::memcpy(&b[3], &mv, 4);                 // moveX (LE on both target platforms)
    b[7] = in.shove ? 1u : 0u;
    return b;
}
inline std::vector<uint8_t> EncodeDuelAck(uint8_t round, uint8_t tick) {
    std::vector<uint8_t> b(3);
    b[0] = 1; b[1] = round; b[2] = tick;        // ack
    return b;
}

// The per-peer script store: scripts[round] is a length-kRoundTicks DuelInput vector.
using DuelScripts = std::vector<std::vector<gduel::DuelInput>>;

// MakeCanonicalScripts(player): the FIXED per-round scripts RunDuelMatch plays for `player` (via
// MakeRoundScript, duel.h read-only). Peer p transmits MakeCanonicalScripts(p) over the wire.
inline DuelScripts MakeCanonicalScripts(int player) {
    DuelScripts s(gduel::kRounds);
    for (uint32_t r = 0; r < gduel::kRounds; ++r) s[r] = gduel::MakeRoundScript(r, player);
    return s;
}

// MatchDigestFromScripts: recompute RunDuelMatch's matchDigest from EXPLICIT per-player scripts (composing
// BuildDuelScene / RunDuelRound / the md fold VERBATIM from duel.h). With the canonical scripts this
// reproduces RunDuelMatch().matchDigest bit-for-bit — so a peer that received the scripts over UDP derives
// the identical pinned digest. Pure integer; duel.h byte-untouched.
inline uint64_t MatchDigestFromScripts(const DuelScripts& sA, const DuelScripts& sB) {
    const hf::game::tags::TagRegistry reg = gduel::MakeDuelRegistry();
    gduel::verdict::DigestFnv md;
    uint32_t score[2] = {0, 0};
    for (uint32_t r = 0; r < gduel::kRounds; ++r) {
        gduel::verdict::VerdictWorld world0;
        const gduel::DuelScene scene = gduel::BuildDuelScene(world0);
        const gduel::RoundResult rr = gduel::RunDuelRound(scene, reg, sA[r], sB[r], gduel::kRoundTicks);
        const int win = rr.verdictOut.winner;
        if (win == 0 || win == 1) ++score[win];
        md.mix(rr.traceDigest);
        md.mix32(score[0]); md.mix32(score[1]); md.sep();
        if (score[0] >= gduel::kWinThreshold) break;
        if (score[1] >= gduel::kWinThreshold) break;
    }
    return md.h;
}

// The result of a UDP duel-match run (both peers' reconstructed digest + real packet stats).
struct UdpDuelResult {
    bool      ran           = false;   // sockets opened + the exchange completed
    bool      converged     = false;   // both peers reconstructed the SAME digest
    uint64_t  peer0Digest   = 0;
    uint64_t  peer1Digest   = 0;
    uint64_t  packetsSent   = 0;       // real sendto() calls across both endpoints
    uint64_t  packetsRecv   = 0;       // real recvfrom() datagrams across both endpoints
    uint64_t  reordered     = 0;
    uint64_t  duplicated    = 0;
    uint64_t  droppedOnce   = 0;
    uint32_t  pumps         = 0;       // driver iterations to full exchange
};

// A tiny per-peer reliable-exchange endpoint state (over a UdpTransport). Sends its own script entries
// (ACK+resend) and reassembles the peer's; a reorder-hold buffer + drop-once set implement the faults.
namespace detail {
struct RelPeer {
    UdpTransport                                          sock;
    DuelScripts                                           mine;        // my player's scripts (to send)
    std::map<std::pair<uint8_t,uint8_t>, gduel::DuelInput> got;         // reassembled remote entries
    std::map<std::pair<uint8_t,uint8_t>, bool>            ackedByPeer;  // my entries the peer acked
    std::vector<std::vector<uint8_t>>                     ackQueue;     // acks to emit next pump
    std::vector<std::vector<uint8_t>>                     holdBuf;      // reorder-held datagrams (one pump)
    std::map<std::pair<int,int>, bool>                    dropSeen;     // (kind, hash) already drop-skipped
    uint32_t                                              expect = 0;   // remote entries expected
    uint32_t                                              minecount = 0;// my entries to get acked
};

// Send with optional fault injection. Returns via the peer sock; updates FaultStats. The datagram key
// (type,round,tick) drives the deterministic drop-once decision so a resend of the same key delivers.
inline void FaultedSend(RelPeer& p, const std::vector<uint8_t>& dg, const FaultConfig& fc, FaultStats& fs,
                        uint64_t seqForFaults) {
    if (fc.enabled) {
        // drop-once: drop the FIRST send of every 5th datagram; a later resend of the same key delivers.
        if (fc.dropOnce && (seqForFaults % 5u) == 0u) {
            const std::pair<int,int> key{(int)dg[0], (int)((dg.size() >= 3) ? (dg[1] * 256 + dg[2]) : 0)};
            if (!p.dropSeen[key]) { p.dropSeen[key] = true; ++fs.droppedOnce; return; }
        }
        if (fc.duplicate && (seqForFaults % 4u) == 0u) { p.sock.Send(dg); ++fs.duplicated; }  // harmless resend
    }
    p.sock.Send(dg);
}
}  // namespace detail

// RunUdpDuelMatch: the loopback proof. Opens two real UDP sockets on 127.0.0.1, exchanges the canonical
// per-round DuelInput scripts over real datagrams (ACK+resend; optional reorder/dup/drop-once faults),
// and returns both peers' independently-reconstructed match digests + real packet stats. NON-FLAKY: the
// ACK-driven resend loop runs until BOTH peers fully exchanged (bounded by maxPumps), and every fault is
// eventually-delivering. If sockets can't open on this box, `ran=false` (the caller falls back + reports).
inline UdpDuelResult RunUdpDuelMatch(const FaultConfig& fc = FaultConfig{}, uint32_t maxPumps = 200000) {
    UdpDuelResult res;
    detail::RelPeer p0, p1;
    if (!p0.sock.Open() || !p1.sock.Open()) return res;   // ran=false
    p0.sock.SetPeer(p1.sock.Port());
    p1.sock.SetPeer(p0.sock.Port());
    p0.mine = MakeCanonicalScripts(0);
    p1.mine = MakeCanonicalScripts(1);

    const uint32_t entries = gduel::kRounds * gduel::kRoundTicks;   // entries each peer sends/expects
    p0.expect = p1.expect = entries;
    p0.minecount = p1.minecount = entries;

    FaultStats fs;
    uint64_t faultSeq = 0;

    auto pumpSend = [&](detail::RelPeer& p) {
        // (Re)send every one of my entries the peer has NOT yet acked.
        for (uint32_t r = 0; r < gduel::kRounds; ++r)
            for (uint32_t t = 0; t < gduel::kRoundTicks; ++t) {
                const std::pair<uint8_t,uint8_t> key{(uint8_t)r, (uint8_t)t};
                if (p.ackedByPeer.count(key)) continue;
                detail::FaultedSend(p, EncodeDuelData((uint8_t)r, (uint8_t)t, p.mine[r][t]), fc, fs, faultSeq++);
            }
        // Emit any queued acks.
        for (auto& a : p.ackQueue) detail::FaultedSend(p, a, fc, fs, faultSeq++);
        p.ackQueue.clear();
    };
    auto pumpRecv = [&](detail::RelPeer& p) {
        // Process the previously-HELD (reordered) datagrams first, then newly-arrived ones — the hold is a
        // LOCAL one-pump processing delay (NOT a re-send to the peer), so a datagram genuinely arrives out
        // of order at THIS receiver while its bytes stay intact.
        std::vector<std::vector<uint8_t>> in;
        in.swap(p.holdBuf);
        p.sock.Poll(in);
        for (auto& dg : in) {
            // Reorder: defer every 3rd first-seen datagram by one pump (processed next pumpRecv).
            if (fc.enabled && fc.reorder && (dg.size() >= 3) && ((dg[1] * 256 + dg[2] + dg[0]) % 3u == 0u)) {
                const std::pair<int,int> rk{(int)dg[0] + 100, (int)(dg[1] * 256 + dg[2])};
                if (!p.dropSeen[rk]) { p.dropSeen[rk] = true; p.holdBuf.push_back(dg); ++fs.reordered; continue; }
            }
            if (dg.empty()) continue;
            if (dg[0] == 0 && dg.size() >= 8) {                 // DATA
                const uint8_t r = dg[1], t = dg[2];
                int32_t mv; std::memcpy(&mv, &dg[3], 4);
                gduel::DuelInput di; di.moveX = (gduel::fx)mv; di.shove = (dg[7] != 0);
                p.got[{r, t}] = di;
                p.ackQueue.push_back(EncodeDuelAck(r, t));       // ack it
            } else if (dg[0] == 1 && dg.size() >= 3) {          // ACK
                p.ackedByPeer[{dg[1], dg[2]}] = true;
            }
        }
    };
    auto done = [&](detail::RelPeer& p) {
        return (uint32_t)p.got.size() >= p.expect && (uint32_t)p.ackedByPeer.size() >= p.minecount;
    };

    uint32_t pumps = 0;
    for (; pumps < maxPumps; ++pumps) {
        pumpSend(p0); pumpSend(p1);
        pumpRecv(p0); pumpRecv(p1);
        if (done(p0) && done(p1)) { pumpRecv(p0); pumpRecv(p1); break; }   // one extra drain for trailing acks
    }
    res.pumps = pumps + 1;

    // Reassemble both peers' received scripts + compute each peer's independent match digest.
    auto rebuild = [&](detail::RelPeer& p) -> DuelScripts {
        DuelScripts s(gduel::kRounds, std::vector<gduel::DuelInput>(gduel::kRoundTicks));
        for (auto& kv : p.got) s[kv.first.first][kv.first.second] = kv.second;
        return s;
    };
    // Peer 0 has its OWN scripts (player 0) + received player 1's; peer 1 vice versa.
    const DuelScripts p0RemoteB = rebuild(p0);   // player-1 scripts as received by peer 0
    const DuelScripts p1RemoteA = rebuild(p1);   // player-0 scripts as received by peer 1
    res.peer0Digest = MatchDigestFromScripts(p0.mine,   p0RemoteB);
    res.peer1Digest = MatchDigestFromScripts(p1RemoteA, p1.mine);
    res.converged   = (res.peer0Digest == res.peer1Digest);
    res.ran         = (done(p0) && done(p1));
    res.packetsSent = p0.sock.SentCount() + p1.sock.SentCount();
    res.packetsRecv = p0.sock.RecvCount() + p1.sock.RecvCount();
    res.reordered   = fs.reordered;
    res.duplicated  = fs.duplicated;
    res.droppedOnce = fs.droppedOnce;
    return res;
}

// ============================ HARNESS 2 — PER-TICK PREDICT+ROLLBACK OVER REAL UDP ========================
// The genuine "deterministic ROLLBACK over a real network" proof: two peers each run an NS3
// RollbackSession, exchanging per-tick inputs over real UDP MID-SIMULATION. Because real datagrams arrive
// late/out-of-order, each peer PREDICTS the missing remote (StepPredicted) and ROLLS BACK when the truth
// arrives and differs (ConfirmRemote) — over real sockets. A final reliable drain confirms every tick.
// Both peers re-derive the IDENTICAL world digest == the pinned NS6 authority (0x1aa9738bcc0c7001) — the
// rollback layer converges over a real network. Uses the SAME W6 Horner toy + canonical (A,B) fold as
// session.h NS6 so the pinned digest coincides by design.

using In6 = int32_t;
struct W6 { int64_t acc = 0; };
inline void StepAB(W6& w, In6 a, In6 b) { w.acc = w.acc * 6 + (int64_t)a * 3 + (int64_t)b * 5; }  // K=6,Aw=3,Bw=5
inline uint64_t Digest6(const W6& w) { return DigestBytes(&w.acc, sizeof w.acc); }

// The NS6 canonical streams (deterministic; both VARY so predictions mispredict -> real rollbacks fire).
inline void MakeNs6Streams(std::vector<In6>& a, std::vector<In6>& b, uint32_t n) {
    a.resize(n); b.resize(n);
    for (uint32_t t = 0; t < n; ++t) {
        a[t] = (In6)(1 + (t * 7) % 11);
        b[t] = (In6)(-3 + (int)((t * 5) % 13) - (int)(t % 4));
    }
}

// The pinned canonical authority digest over the 24-tick NS6 streams (== session_test NS6 / NS3 anchor).
inline uint64_t Ns6AuthorityDigest(uint32_t n) {
    std::vector<In6> a, b; MakeNs6Streams(a, b, n);
    W6 w; for (uint32_t t = 0; t < n; ++t) StepAB(w, a[t], b[t]);
    return Digest6(w);
}

struct UdpRollbackResult {
    bool     ran          = false;
    bool     converged    = false;   // both peers' world digest equal
    bool     matchedPin    = false;  // and == the pinned NS6 authority
    bool     peer0Rolled  = false;   // peer 0 actually rolled back over the wire
    bool     peer1Rolled  = false;
    uint64_t peer0Digest  = 0;
    uint64_t peer1Digest  = 0;
    uint64_t packetsSent  = 0;
    uint64_t packetsRecv  = 0;
};

// One per-tick input datagram: forTick(u16 LE) + input(int32 LE). No ack; the receiver ConfirmRemotes each
// (a duplicate/late/reordered delivery is a harmless no-op in ConfirmRemote), and a final drain resends
// any not-yet-confirmed tick until both peers have confirmed all n ticks.
inline std::vector<uint8_t> EncodeTickInput(uint16_t forTick, In6 v) {
    std::vector<uint8_t> b(6);
    b[0] = (uint8_t)(forTick & 0xff); b[1] = (uint8_t)(forTick >> 8);
    std::memcpy(&b[2], &v, 4);
    return b;
}

// RunUdpRollbackConverge: drive two RollbackSessions over real UDP loopback. Non-flaky: after the per-tick
// speculative loop, a bounded drain resends unconfirmed ticks until BOTH peers have confirmed all n.
inline UdpRollbackResult RunUdpRollbackConverge(uint32_t n = 24, uint32_t maxPumps = 200000) {
    UdpRollbackResult res;
    UdpTransport s0, s1;
    if (!s0.Open() || !s1.Open()) return res;
    s0.SetPeer(s1.Port());
    s1.SetPeer(s0.Port());

    std::vector<In6> a, b; MakeNs6Streams(a, b, n);
    // Peer 0: local = A stream, remote = B stream -> canonical (A=local, B=remote).
    RollbackSession<W6, In6> sesA;
    auto stepA = [](W6& w, std::initializer_list<In6> lr, uint32_t) { const In6* p = lr.begin(); StepAB(w, p[0], p[1]); };
    // Peer 1: local = B stream, remote = A stream -> canonical (A=remote, B=local) — the swap.
    RollbackSession<W6, In6> sesB;
    auto stepB = [](W6& w, std::initializer_list<In6> lr, uint32_t) { const In6* p = lr.begin(); StepAB(w, p[1], p[0]); };

    std::vector<std::vector<uint8_t>> in;
    // Per-tick speculative loop: step (predicting missing remote), send local, poll + confirm arrivals.
    for (uint32_t t = 0; t < n; ++t) {
        StepPredicted(sesA, a[t], stepA);
        StepPredicted(sesB, b[t], stepB);
        s0.Send(EncodeTickInput((uint16_t)t, a[t]));   // peer 0 sends its A input
        s1.Send(EncodeTickInput((uint16_t)t, b[t]));   // peer 1 sends its B input
        // Drain whatever has arrived so far and confirm it (late/reordered arrivals -> rollback).
        in.clear(); s0.Poll(in);
        for (auto& dg : in) if (dg.size() >= 6) { uint16_t ft = (uint16_t)(dg[0] | (dg[1] << 8)); In6 v; std::memcpy(&v, &dg[2], 4);
            ConfirmRemote(sesA, ft, v, stepA); }
        in.clear(); s1.Poll(in);
        for (auto& dg : in) if (dg.size() >= 6) { uint16_t ft = (uint16_t)(dg[0] | (dg[1] << 8)); In6 v; std::memcpy(&v, &dg[2], 4);
            ConfirmRemote(sesB, ft, v, stepB); }
    }
    // Reliable drain: resend every tick each pump, confirm arrivals, until BOTH sessions confirmed all n.
    for (uint32_t pump = 0; pump < maxPumps; ++pump) {
        if (sesA.confirmedThrough >= n && sesB.confirmedThrough >= n) break;
        for (uint32_t t = 0; t < n; ++t) {
            if (sesA.confirmedThrough <= t) s1.Send(EncodeTickInput((uint16_t)t, b[t]));  // B->A resend
            if (sesB.confirmedThrough <= t) s0.Send(EncodeTickInput((uint16_t)t, a[t]));  // A->B resend
        }
        in.clear(); s0.Poll(in);
        for (auto& dg : in) if (dg.size() >= 6) { uint16_t ft = (uint16_t)(dg[0] | (dg[1] << 8)); In6 v; std::memcpy(&v, &dg[2], 4);
            ConfirmRemote(sesA, ft, v, stepA); }
        in.clear(); s1.Poll(in);
        for (auto& dg : in) if (dg.size() >= 6) { uint16_t ft = (uint16_t)(dg[0] | (dg[1] << 8)); In6 v; std::memcpy(&v, &dg[2], 4);
            ConfirmRemote(sesB, ft, v, stepB); }
    }

    res.ran         = (sesA.confirmedThrough >= n && sesB.confirmedThrough >= n);
    res.peer0Digest = Digest6(sesA.world);
    res.peer1Digest = Digest6(sesB.world);
    res.converged   = (res.peer0Digest == res.peer1Digest);
    res.matchedPin  = res.converged && (res.peer0Digest == Ns6AuthorityDigest(n));
    res.peer0Rolled = sesA.didRollback;
    res.peer1Rolled = sesB.didRollback;
    res.packetsSent = s0.SentCount() + s1.SentCount();
    res.packetsRecv = s0.RecvCount() + s1.RecvCount();
    return res;
}

}  // namespace udp
}  // namespace net
}  // namespace hf
