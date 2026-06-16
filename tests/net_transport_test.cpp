// Unit test for the networking transport + client jitter-buffer / interpolation layer
// (engine/net/transport.{h,cpp}, Slice BU). Pure CPU, deterministic, NO sockets / GPU. Builds on the
// BQ snapshot layer (engine/net/snapshot). Asserts:
//   * Channel determinism: the same (config, send stream) yields identical delivered/dropped/reordered
//     sequences across two runs; delivered + dropped == sent; latency respected (a packet sent at T
//     delivers at >= T + latency).
//   * Loss / reorder: dropRate>0 drops some (counted); reorderRate>0 makes delivery order differ from
//     send order at least once over a long stream; dropRate=0 / reorderRate=0 = pure FIFO delay.
//   * Jitter buffer + interpolation: a clean channel -> RenderState(renderTick) == the authority at
//     renderTick within the interp alpha; a single DROPPED snapshot is HIDDEN (the bracketing snapshots
//     interpolate across the gap, within tolerance, NOT a pop); the documented hold-last gap policy
//     holds at the buffer edge.
//   * Convergence / bounds: over the full lossy stream the client's interpolated state stays within the
//     documented tolerance of the authority (no unbounded drift); deterministic.
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "net/transport.h"
#include "net/snapshot.h"

#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// --- A deterministic synthetic authority track: an entity gliding smoothly so interpolation has a
// known ground truth. position = (t*0.1, 0.5 + sin(t*0.05), t*0.02); orientation spins about Y. -----
static net::Snapshot AuthorityAt(int tick) {
    net::Snapshot s;
    s.tick = (uint32_t)tick;
    net::RepEntity e;
    e.id = 0;
    e.flags = net::kFlagPlayer;
    float t = (float)tick;
    e.position = {t * 0.1f, 0.5f + std::sin(t * 0.05f), t * 0.02f};
    float ang = t * 0.03f;
    e.orientation = math::Quat{0.0f, std::sin(ang * 0.5f), 0.0f, std::cos(ang * 0.5f)};
    s.entities.push_back(e);
    return s;
}

static float PosError(const net::Snapshot& a, const net::Snapshot& truth) {
    if (a.entities.empty() || truth.entities.empty()) return 1e9f;
    return math::length(a.entities[0].position - truth.entities[0].position);
}

int main() {
    using net::ChannelConfig;
    using net::SimChannel;
    using net::ClientView;

    const int kStream = 200;  // a long stream so probabilistic loss/reorder actually fires

    // --- 1. Channel determinism ------------------------------------------------------------------
    {
        ChannelConfig cfg;
        cfg.latencyTicks = 3;
        cfg.lossSeed = 12345u;
        cfg.dropRate = 0.15f;
        cfg.reorderRate = 0.1f;

        auto runOnce = [&](std::vector<int>& deliverOrder, uint32_t& del, uint32_t& drp, uint32_t& reo) {
            SimChannel ch(cfg);
            // Send one packet per tick; harvest deliveries each tick; record the SEND tick (carried in
            // the packet) of every delivered packet, in delivery order.
            for (int t = 0; t <= kStream; ++t) {
                net::Snapshot s = AuthorityAt(t);
                ch.Send(t, net::Serialize(s));
                for (const net::Packet& p : ch.Deliver(t)) deliverOrder.push_back(p.sendTick);
            }
            // Drain the tail (latency means the last few are still in flight).
            for (int t = kStream + 1; t <= kStream + cfg.latencyTicks + 8; ++t)
                for (const net::Packet& p : ch.Deliver(t)) deliverOrder.push_back(p.sendTick);
            del = ch.Delivered(); drp = ch.Dropped(); reo = ch.Reordered();
            check(ch.Sent() == (uint32_t)(kStream + 1), "channel: Sent == number of Send calls");
            check(del + drp == ch.Sent(), "channel: delivered + dropped == sent");
            check(ch.Empty(), "channel: queue fully drained after tail");
        };

        std::vector<int> a, b;
        uint32_t da, dra, ra, db, drb, rb;
        runOnce(a, da, dra, ra);
        runOnce(b, db, drb, rb);
        check(a == b, "channel: two runs produce identical delivery order (deterministic)");
        check(da == db && dra == drb && ra == rb, "channel: counters identical across runs");

        // Latency respected: every delivered packet's deliverTick >= sendTick + latency. (Re-run and
        // check the deliverTick directly.)
        {
            SimChannel ch(cfg);
            bool latencyOk = true;
            for (int t = 0; t <= kStream; ++t) {
                ch.Send(t, net::Serialize(AuthorityAt(t)));
                for (const net::Packet& p : ch.Deliver(t + 10000)) {  // flush everything sent so far
                    if (p.deliverTick < p.sendTick + cfg.latencyTicks) latencyOk = false;
                }
            }
            check(latencyOk, "channel: every delivered packet deliverTick >= sendTick + latency");
        }
    }

    // --- 2. Loss / reorder behavior --------------------------------------------------------------
    {
        // (a) dropRate > 0 drops some packets (counted).
        {
            ChannelConfig cfg; cfg.latencyTicks = 2; cfg.lossSeed = 999u; cfg.dropRate = 0.3f;
            SimChannel ch(cfg);
            for (int t = 0; t <= kStream; ++t) ch.Send(t, net::Serialize(AuthorityAt(t)));
            check(ch.Dropped() > 0, "loss: dropRate>0 drops at least one packet");
            check(ch.Dropped() < ch.Sent(), "loss: not everything is dropped");
            check(ch.Delivered() + ch.Dropped() == ch.Sent(), "loss: delivered + dropped == sent");
        }

        // (b) reorderRate > 0 => delivery order differs from send order at least once.
        {
            ChannelConfig cfg; cfg.latencyTicks = 2; cfg.lossSeed = 777u; cfg.reorderRate = 0.25f;
            SimChannel ch(cfg);
            std::vector<int> order;
            for (int t = 0; t <= kStream; ++t) {
                ch.Send(t, net::Serialize(AuthorityAt(t)));
                for (const net::Packet& p : ch.Deliver(t)) order.push_back(p.sendTick);
            }
            for (int t = kStream + 1; t <= kStream + 20; ++t)
                for (const net::Packet& p : ch.Deliver(t)) order.push_back(p.sendTick);
            check(ch.Reordered() > 0, "reorder: reorderRate>0 reorders at least one packet");
            bool outOfOrder = false;
            for (size_t i = 1; i < order.size(); ++i)
                if (order[i] < order[i - 1]) outOfOrder = true;
            check(outOfOrder, "reorder: delivery order differs from send order at least once");
        }

        // (c) dropRate=0 / reorderRate=0 / latency=L => pure FIFO delay.
        {
            const int L = 4;
            ChannelConfig cfg; cfg.latencyTicks = L; cfg.lossSeed = 1u;  // rates default 0
            SimChannel ch(cfg);
            std::vector<int> order;
            for (int t = 0; t <= kStream; ++t) {
                ch.Send(t, net::Serialize(AuthorityAt(t)));
                for (const net::Packet& p : ch.Deliver(t)) {
                    order.push_back(p.sendTick);
                    check(p.deliverTick == p.sendTick + L, "fifo: deliverTick == sendTick + L exactly");
                }
            }
            for (int t = kStream + 1; t <= kStream + L; ++t)
                for (const net::Packet& p : ch.Deliver(t)) order.push_back(p.sendTick);
            check(ch.Dropped() == 0 && ch.Reordered() == 0, "fifo: no drops, no reorders");
            bool fifo = (order.size() == (size_t)(kStream + 1));
            for (size_t i = 0; fifo && i < order.size(); ++i)
                if (order[i] != (int)i) fifo = false;
            check(fifo, "fifo: pure in-order delivery, every packet exactly once");
        }
    }

    // --- 3. Jitter buffer + interpolation --------------------------------------------------------
    {
        // (a) Clean channel: RenderState(renderTick) matches the authority at renderTick within the
        //     interpolation tolerance (the entity glides; lerp is near-exact for a smooth path).
        {
            const int L = 3, interpDelay = 2;
            ChannelConfig cfg; cfg.latencyTicks = L; cfg.lossSeed = 5u;  // lossless, no reorder
            SimChannel ch(cfg);
            ClientView client(interpDelay);
            float maxErr = 0.0f;
            for (int t = 0; t <= 80; ++t) {
                ch.Send(t, net::Serialize(AuthorityAt(t)));
                for (const net::Packet& p : ch.Deliver(t)) client.Receive(p.bytes);
                if (client.HasData() && client.LatestReceivedTick() >= interpDelay + 2) {
                    float rt = client.RenderTick();
                    net::Snapshot rs = client.RenderState(rt);
                    net::Snapshot truth = AuthorityAt((int)std::lround(rt));
                    // rt is integer here (latest - interpDelay) so this is an exact tick comparison.
                    maxErr = std::max(maxErr, PosError(rs, truth));
                }
            }
            check(maxErr < 1e-3f, "interp: clean channel render matches authority at renderTick");
        }

        // (b) A single dropped snapshot is HIDDEN: bracketing snapshots interpolate across the gap; the
        //     interpolated value stays within tolerance of the authority (NOT a visible pop).
        {
            ClientView client(2);
            // Buffer ticks 10..14 but DROP tick 12 (the hole). The client should interpolate tick 12's
            // render across 11<->13, landing near the true authority position (smooth path).
            for (int t = 10; t <= 14; ++t) {
                if (t == 12) continue;  // dropped
                client.ReceiveSnapshot(AuthorityAt(t));
            }
            net::Snapshot atGap = client.RenderState(12.0f);
            net::Snapshot truth12 = AuthorityAt(12);
            float gapErr = PosError(atGap, truth12);
            check(gapErr < 5e-3f, "interp: single dropped snapshot hidden (interp across gap ~ truth)");
            // It must NOT just hold tick 11 (a pop would equal the tick-11 value): the interpolated x
            // should sit between 11 and 13's x, strictly past 11's.
            check(atGap.entities[0].position.x > AuthorityAt(11).entities[0].position.x + 1e-4f,
                  "interp: gap render advanced past the last-good snapshot (no freeze/pop)");
        }

        // (c) Documented gap/edge policy: hold-oldest below the buffer, hold-newest above it (no
        //     extrapolation past the last received snapshot).
        {
            ClientView client(2);
            for (int t = 20; t <= 25; ++t) client.ReceiveSnapshot(AuthorityAt(t));
            net::Snapshot below = client.RenderState(5.0f);    // before the oldest
            net::Snapshot above = client.RenderState(99.0f);   // past the newest
            check(below == AuthorityAt(20), "edge: renderTick below buffer holds the oldest snapshot");
            check(above == AuthorityAt(25), "edge: renderTick above buffer holds the newest (no extrapolation)");
            // Exactly on a buffered tick returns that snapshot.
            net::Snapshot on = client.RenderState(22.0f);
            check(PosError(on, AuthorityAt(22)) < 1e-5f, "edge: renderTick on a buffered tick == that snapshot");
        }
    }

    // --- 4. Convergence / bounds over the full lossy stream --------------------------------------
    {
        const int L = 3, interpDelay = 2;
        ChannelConfig cfg;
        cfg.latencyTicks = L; cfg.lossSeed = 0xBEEFu; cfg.dropRate = 0.15f; cfg.reorderRate = 0.1f;

        auto runStream = [&](float& maxErr) {
            SimChannel ch(cfg);
            ClientView client(interpDelay);
            maxErr = 0.0f;
            for (int t = 0; t <= kStream; ++t) {
                ch.Send(t, net::Serialize(AuthorityAt(t)));
                for (const net::Packet& p : ch.Deliver(t)) client.Receive(p.bytes);
                // Once warmed up, the interpolated render at renderTick must track the authority.
                if (client.HasData() && client.LatestReceivedTick() >= 10) {
                    float rt = client.RenderTick();
                    net::Snapshot rs = client.RenderState(rt);
                    net::Snapshot truth = AuthorityAt((int)std::lround(rt));
                    maxErr = std::max(maxErr, PosError(rs, truth));
                }
            }
        };

        float e1 = 0.0f, e2 = 0.0f;
        runStream(e1);
        runStream(e2);
        check(e1 == e2, "convergence: lossy stream is deterministic (identical max error two runs)");
        // The interpolated client stays within a documented tolerance of the authority despite 15%
        // loss + 10% reorder — the jitter buffer + interpolation hide the imperfect channel, no
        // unbounded drift. (Tolerance is generous vs the ~0.1/tick motion; a hold across a multi-drop
        // hole is the worst case.)
        check(e1 < 0.5f, "convergence: interpolated client stays within tolerance of authority (bounded)");
        std::printf("netsim-bounds: maxInterpError=%.6f (deterministic two-run: %s)\n",
                    e1, (e1 == e2) ? "yes" : "NO");
    }

    if (g_fail == 0) std::printf("net_transport_test: ALL PASS\n");
    else std::printf("net_transport_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
