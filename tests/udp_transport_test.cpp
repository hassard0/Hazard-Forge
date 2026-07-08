// Unit test for the REAL UDP TRANSPORT (engine/net/udp_transport.h, Slice NW1, flagship #24 NETCODE, the
// BEAT_UE5 "Rollback Kit" post-P0 step). This is the ONE nondeterministic-I/O integration test: it opens
// REAL UDP sockets (Winsock on Windows, BSD sockets elsewhere) on 127.0.0.1 and exchanges REAL datagrams.
//
// 🔵 HONEST BOUNDARY: real sockets are nondeterministic I/O (arrival order/loss vary run-to-run). The gate
// is scoped so the SIM OUTCOME is what's asserted bit-exact: two peers exchanging inputs over real UDP each
// INDEPENDENTLY re-derive the IDENTICAL match/world digest == a pinned value — deterministic rollback over
// a real network, the claim UE5's float architecture structurally cannot make. NON-FLAKY BY CONSTRUCTION:
// a same-process two-socket loopback + a bounded ACK+resend reliability loop (drop-then-resend converges);
// the DIGEST assertions are the pinned, deterministic part. (True two-process run: benchmarks/udp_two_process.md.)
//
// What this pins:
//   (a) BASIC SOCKET       — two UdpTransport endpoints exchange a datagram over 127.0.0.1; received bytes
//                            == sent bytes (real send/recv works).
//   (b) LOOPBACK MATCH     — two peers run the GAME1 duel over real UDP -> both final match digests ==
//                            the pinned GAME1 matchDigest (rollback-netcode-over-real-network converges).
//   (c) RESILIENCE         — with injected reorder/duplicate/drop-then-resend, both peers STILL converge to
//                            the identical pinned digest (the layer handles real-network conditions).
//   (d) TRANSPORT-INVARIANT— the over-UDP digest == the in-process RunDuelMatch().matchDigest (real
//                            transport does not change the deterministic outcome).
//   (e) ROLLBACK OVER UDP  — per-tick predict+rollback over real UDP converges to the pinned NS6 authority.
//   (f) PIN == nw1_report  — the pinned constants the showcase renders match the live socket outcome.

#include "net/udp_transport.h"
#include "net/nw1_report.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::net;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// The pinned outcomes (the deterministic part; sockets are nondeterministic I/O).
static constexpr uint64_t kPinnedMatchDigest    = 0x78123003c3a55a37ull;  // GAME1 best-of-3 (duel_test pin)
static constexpr uint64_t kPinnedRollbackDigest = 0x1aa9738bcc0c7001ull;  // NS6/NS3 authority (session_test pin)

int main() {
    HF_TEST_MAIN_INIT();

    // ---- (a) BASIC SOCKET — real send/recv over 127.0.0.1. -------------------------------------------
    {
        udp::UdpTransport a, b;
        const bool opened = a.Open() && b.Open();
        check(opened, "nw1(a): two UDP endpoints Open() on 127.0.0.1:0");
        if (opened) {
            a.SetPeer(b.Port());
            b.SetPeer(a.Port());
            const uint8_t payload[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x00, 0x99, 0x7F };
            // Non-blocking recv can race the send; a bounded pump makes it reliable without flaking.
            std::vector<std::vector<uint8_t>> in;
            bool matched = false;
            for (int pump = 0; pump < 100000 && !matched; ++pump) {
                a.Send(payload, sizeof payload);
                in.clear();
                b.Poll(in);
                for (auto& dg : in)
                    if (dg.size() == sizeof payload && std::memcmp(dg.data(), payload, sizeof payload) == 0)
                        matched = true;
            }
            check(matched, "nw1(a): received datagram bytes == sent bytes (real UDP send/recv)");
            check(b.RecvCount() >= 1 && a.SentCount() >= 1, "nw1(a): real sendto/recvfrom counters advanced");
        }
    }

    // ---- (b) LOOPBACK MATCH CONVERGES — the GAME1 duel over real UDP. --------------------------------
    {
        const udp::UdpDuelResult r = udp::RunUdpDuelMatch(/*clean*/ udp::FaultConfig{});
        check(r.ran, "nw1(b): the duel-input exchange completed over real UDP");
        check(r.converged, "nw1(b): both peers reconstructed the SAME match digest over real UDP");
        check(r.peer0Digest == kPinnedMatchDigest, "nw1(b): peer 0 over-UDP digest == pinned GAME1 matchDigest");
        check(r.peer1Digest == kPinnedMatchDigest, "nw1(b): peer 1 over-UDP digest == pinned GAME1 matchDigest");
        std::printf("nw1(b): loopback match {peer0:0x%016llx, peer1:0x%016llx, sent:%llu, recv:%llu, pumps:%u}\n",
                    (unsigned long long)r.peer0Digest, (unsigned long long)r.peer1Digest,
                    (unsigned long long)r.packetsSent, (unsigned long long)r.packetsRecv, r.pumps);
    }

    // ---- (c) RESILIENCE — reorder + duplicate + drop-then-resend STILL converges. --------------------
    {
        udp::FaultConfig fc; fc.enabled = true; fc.reorder = true; fc.duplicate = true; fc.dropOnce = true;
        const udp::UdpDuelResult r = udp::RunUdpDuelMatch(fc);
        check(r.ran, "nw1(c): the exchange completed despite injected reorder/dup/drop-then-resend");
        check(r.converged, "nw1(c): both peers STILL converged under injected faults");
        check(r.peer0Digest == kPinnedMatchDigest && r.peer1Digest == kPinnedMatchDigest,
              "nw1(c): both over-UDP digests STILL == the pinned matchDigest under faults");
        check(r.reordered > 0 || r.duplicated > 0 || r.droppedOnce > 0,
              "nw1(c): the fault injector actually reordered/duplicated/dropped datagrams");
        std::printf("nw1(c): resilience {converged:%s, reordered:%llu, duplicated:%llu, droppedOnce:%llu}\n",
                    r.converged ? "true" : "false", (unsigned long long)r.reordered,
                    (unsigned long long)r.duplicated, (unsigned long long)r.droppedOnce);
    }

    // ---- (d) TRANSPORT-INVARIANT — over-UDP digest == the in-process RunDuelMatch(). ------------------
    {
        const uint64_t inProcess = hf::game::duel::RunDuelMatch().matchDigest;
        check(inProcess == kPinnedMatchDigest, "nw1(d): in-process RunDuelMatch digest == pinned (sanity)");
        const udp::UdpDuelResult r = udp::RunUdpDuelMatch(udp::FaultConfig{});
        check(r.peer0Digest == inProcess, "nw1(d): real transport does NOT change the outcome (over-UDP == in-process)");
    }

    // ---- (e) ROLLBACK OVER REAL UDP — per-tick predict+rollback converges to the pinned authority. ----
    {
        const udp::UdpRollbackResult r = udp::RunUdpRollbackConverge(24);
        check(r.ran, "nw1(e): the per-tick rollback session drained over real UDP");
        check(r.converged, "nw1(e): both rollback peers converged to each other over real UDP");
        check(r.matchedPin, "nw1(e): the converged world digest == the pinned NS6/NS3 authority");
        check(r.peer0Digest == kPinnedRollbackDigest && r.peer1Digest == kPinnedRollbackDigest,
              "nw1(e): both peers' rollback digest == the pinned authority (rollback-over-real-network)");
        // Real datagrams arrive late/out-of-order -> a real misprediction+rollback fired over the wire.
        check(r.peer0Rolled || r.peer1Rolled, "nw1(e): a real misprediction+rollback fired over real UDP");
        std::printf("nw1(e): rollback-over-UDP {peer0:0x%016llx, peer1:0x%016llx, rolled:[%s,%s]}\n",
                    (unsigned long long)r.peer0Digest, (unsigned long long)r.peer1Digest,
                    r.peer0Rolled ? "true" : "false", r.peer1Rolled ? "true" : "false");
    }

    // ---- (f) PIN == nw1_report — the showcase's pinned constants match the live socket outcome. -------
    {
        check(nw1::kNw1MatchDigest == kPinnedMatchDigest, "nw1(f): showcase pinned matchDigest == the socket outcome");
        check(nw1::kNw1RollbackDigest == kPinnedRollbackDigest, "nw1(f): showcase pinned rollbackDigest == the authority");
        check(nw1::Nw1RollbackAuthority(24) == kPinnedRollbackDigest, "nw1(f): the report's authority recompute == pin");
        // The showcase renders deterministically (two-run byte-identical) despite the nondeterministic I/O.
        std::vector<uint8_t> img1, img2; nw1::Nw1VizStats s1{}, s2{};
        nw1::RenderNw1UdpViz(img1, s1);
        nw1::RenderNw1UdpViz(img2, s2);
        const bool twoRun = (img1 == img2) && (s1.pixDigest == s2.pixDigest);
        check(twoRun, "nw1(f): the showcase report is two-run byte-identical (deterministic viz)");
        check(s1.converged && s1.rollbackOk && s1.matchDigest == kPinnedMatchDigest,
              "nw1(f): the showcase stats report the pinned converged outcome");
        std::printf("nw1(f): showcase {converged:%s, rollbackOk:%s, matchDigest:0x%016llx, pixDigest:0x%016llx}\n",
                    s1.converged ? "true" : "false", s1.rollbackOk ? "true" : "false",
                    (unsigned long long)s1.matchDigest, (unsigned long long)s1.pixDigest);
    }

    // ---- Report lines. -------------------------------------------------------------------------------
    std::printf("nw1-udp: REAL UDP transport for the deterministic rollback netcode (Winsock/BSD sockets)\n");
    std::printf("nw1-udp: two peers exchange inputs over real 127.0.0.1 datagrams -> identical pinned digest\n");
    std::printf("nw1-udp: match over UDP {digest:0x%016llx} == in-process GAME1 matchDigest (transport-invariant)\n",
                (unsigned long long)kPinnedMatchDigest);
    std::printf("nw1-udp: rollback over UDP {digest:0x%016llx} == NS6 authority (rollback-over-real-network)\n",
                (unsigned long long)kPinnedRollbackDigest);

    if (g_fail == 0) { std::printf("udp_transport_test: ALL CHECKS PASSED\n"); return 0; }
    std::printf("udp_transport_test: %d failures\n", g_fail);
    return 1;
}
