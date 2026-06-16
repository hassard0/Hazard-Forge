// Slice BU — networking transport + client jitter-buffer / interpolation implementation. See
// transport.h for the contract. Pure CPU above engine/math + engine/net (BQ); ZERO RHI / backend
// symbols. Deterministic: a seeded integer LCG drives drop/reorder/jitter (NO real RNG, NO clock);
// latency is a fixed tick offset; the client interpolation reuses BQ Interpolate. Compiled into BOTH
// hf_core (ASan, unit-tested) and hf_engine (the --netsim-shot showcase).
#include "net/transport.h"

#include <utility>

namespace hf::net {

// --- SimChannel ----------------------------------------------------------------------------------
bool SimChannel::Send(int tick, std::vector<uint8_t> bytes) {
    ++sent_;
    const uint32_t seq = nextSeq_++;

    // Drop draw FIRST so the LCG advances identically whether or not the packet survives (the draw is
    // always consumed). A draw < dropRate => the packet is lost (counted, not queued).
    const float dropDraw = NextUnit();
    if (dropDraw < cfg_.dropRate) {
        ++dropped_;
        // Still consume the reorder + jitter draws so a dropped packet does not shift the LCG phase
        // for subsequent packets — keeps the stream bit-identical regardless of which packets drop.
        (void)NextUnit();  // reorder draw
        (void)NextUnit();  // jitter draw
        return false;
    }

    int deliverTick = tick + cfg_.latencyTicks;

    // Reorder draw: a packet selected for reorder gets an EXTRA seeded delay so it falls behind a
    // later-sent packet (delivery order != send order). The extra is in [2 .. 2+reorderSpread]: the
    // minimum of 2 guarantees this packet (delivered at sendTick+latency+2) arrives strictly AFTER the
    // next in-stream packet (delivered at (sendTick+1)+latency), so the order genuinely changes — not
    // merely a deliverTick tie that the seq tie-break would keep in order.
    const float reorderDraw = NextUnit();
    const float spreadDraw  = NextUnit();
    if (reorderDraw < cfg_.reorderRate) {
        const int span = (cfg_.reorderSpread > 0) ? cfg_.reorderSpread : 0;
        const int extra = 2 + (int)(spreadDraw * (float)(span + 1));
        deliverTick += extra;
        ++reordered_;
    }

    ++delivered_;
    Packet p;
    p.deliverTick = deliverTick;
    p.sendTick    = tick;
    p.seq         = seq;
    p.bytes       = std::move(bytes);
    queue_.emplace(std::make_pair(deliverTick, seq), std::move(p));
    return true;
}

std::vector<Packet> SimChannel::Deliver(int tick) {
    std::vector<Packet> out;
    // The map is ordered by (deliverTick, seq); pop every entry with deliverTick <= tick from the
    // front. Ties (same deliverTick) come out in send-sequence order — deterministic.
    for (auto it = queue_.begin(); it != queue_.end();) {
        if (it->first.first <= tick) {
            out.push_back(std::move(it->second));
            it = queue_.erase(it);
        } else {
            break;  // map is sorted by deliverTick first; nothing further is due
        }
    }
    return out;
}

// --- ClientView ----------------------------------------------------------------------------------
void ClientView::ReceiveSnapshot(const Snapshot& snap) {
    buffer_[(int)snap.tick] = snap;
    if ((int)snap.tick > latestTick_) latestTick_ = (int)snap.tick;
}

void ClientView::Receive(std::span<const uint8_t> bytes) {
    ReceiveSnapshot(Deserialize(bytes));
}

float ClientView::RenderTick() const {
    float rt = (float)latestTick_ - (float)interpDelay_;
    if (rt < 0.0f) rt = 0.0f;
    return rt;
}

Snapshot ClientView::RenderState(float renderTick) const {
    if (buffer_.empty()) return Snapshot{};

    const int oldest = buffer_.begin()->first;
    const int newest = buffer_.rbegin()->first;

    // Edge policy (documented): never extrapolate past the buffer. Clamp-hold at both ends.
    if (renderTick <= (float)oldest) return buffer_.begin()->second;
    if (renderTick >= (float)newest) return buffer_.rbegin()->second;

    // Bracket renderTick: hi = first buffered tick strictly greater than renderTick; lo = the one
    // before it. Because we clamped the ends above, lo and hi both exist and lo < renderTick < hi.
    // A dropped snapshot (a one-tick hole) simply widens the (lo,hi) bracket — Interpolate spans the
    // gap, hiding the drop (no pop) rather than freezing on a stale frame.
    auto hiIt = buffer_.upper_bound((int)renderTick);
    // upper_bound on a truncated int can land ON renderTick's floor; ensure hi.tick > renderTick.
    while (hiIt != buffer_.end() && (float)hiIt->first <= renderTick) ++hiIt;
    if (hiIt == buffer_.end()) return buffer_.rbegin()->second;  // defensive (clamped above)
    auto loIt = hiIt;
    --loIt;

    const int lo = loIt->first;
    const int hi = hiIt->first;
    float alpha = (renderTick - (float)lo) / (float)(hi - lo);
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    return Interpolate(loIt->second, hiIt->second, alpha);
}

}  // namespace hf::net
