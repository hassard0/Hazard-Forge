// Unit test for the networking / replication snapshot layer (engine/net/snapshot.{h,cpp}, Slice BQ).
// Pure CPU, deterministic, NO sockets / GPU. Asserts:
//   * Snapshot round-trip: Deserialize(Serialize(s)) == s (count, ids, transforms, flags).
//   * Delta correctness: DeltaApply(prev, DeltaEncode(prev, curr)) == curr for no-change / moved /
//     added / removed; and DeltaEncode(unchanged) is SMALLER than a full Serialize (compresses).
//   * Replica == authority over a stream: feed authority snapshots (keyframe + deltas) through the
//     Replicator; after each Receive, replica.State() == authority[i] exactly; two runs deterministic.
//   * Interpolation: Interpolate(a,b,0)==a, ==b at 1, midpoint lerps position halfway + orientation
//     normalized.
//   * Roll-game capture: Capture(GameState) yields the expected RepEntities (player + pickups, right
//     ids/positions).
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "net/snapshot.h"

#include "game/roll_game.h"
#include "physics/world.h"
#include "physics/body.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A hand-built snapshot with distinct transforms + flags.
static net::Snapshot MakeSample() {
    net::Snapshot s;
    s.tick = 42;
    s.entities.push_back({0, {0.0f, 0.5f, 0.0f}, math::Quat::Identity(), net::kFlagPlayer});
    s.entities.push_back({1, {2.5f, 0.3f, 0.0f}, math::Quat{0.0f, 0.7071f, 0.0f, 0.7071f}, net::kFlagPickup});
    s.entities.push_back({2, {-1.25f, 0.3f, 4.0f}, math::Quat{0.1f, 0.2f, 0.3f, 0.92736f}, net::kFlagPickup});
    return s;
}

int main() {
    HF_TEST_MAIN_INIT();
    // --- 1. Snapshot round-trip -------------------------------------------------------------------
    {
        net::Snapshot s = MakeSample();
        std::vector<uint8_t> bytes = net::Serialize(s);
        net::Snapshot back = net::Deserialize(bytes);
        check(back == s, "round-trip: Deserialize(Serialize(s)) == s");
        check(back.tick == 42, "round-trip: tick preserved");
        check(back.entities.size() == 3, "round-trip: entity count preserved");
        check(back.entities[1].id == 1 && back.entities[1].flags == net::kFlagPickup,
              "round-trip: id + flags preserved");
        // Raw-IEEE float bytes round-trip exactly (same build).
        check(back.entities[2].position.z == 4.0f && back.entities[2].orientation.w == 0.92736f,
              "round-trip: float fields bit-exact");
        // An empty snapshot round-trips too.
        net::Snapshot empty; empty.tick = 7;
        check(net::Deserialize(net::Serialize(empty)) == empty, "round-trip: empty snapshot");
    }

    // --- 2. Delta correctness + compression -------------------------------------------------------
    {
        net::Snapshot prev = MakeSample();

        // (a) no change.
        {
            net::Snapshot curr = prev;
            std::vector<uint8_t> d = net::DeltaEncode(prev, curr);
            check(net::DeltaApply(prev, d) == curr, "delta: no-change round-trips");
            // Compression: an unchanged delta is strictly smaller than a full serialize.
            check(d.size() < net::Serialize(curr).size(), "delta: unchanged delta < full serialize");
        }
        // (b) one entity moved.
        {
            net::Snapshot curr = prev;
            curr.tick = 43;
            curr.entities[1].position = curr.entities[1].position + math::Vec3{1.0f, 0.0f, -2.0f};
            std::vector<uint8_t> d = net::DeltaEncode(prev, curr);
            check(net::DeltaApply(prev, d) == curr, "delta: moved entity round-trips");
            check(d.size() < net::Serialize(curr).size(), "delta: one-moved delta < full serialize");
        }
        // (c) entity added.
        {
            net::Snapshot curr = prev;
            curr.tick = 44;
            curr.entities.push_back({9, {7.0f, 0.3f, 7.0f}, math::Quat::Identity(), net::kFlagPickup});
            std::vector<uint8_t> d = net::DeltaEncode(prev, curr);
            check(net::DeltaApply(prev, d) == curr, "delta: added entity round-trips");
        }
        // (d) entity removed (a collected pickup disappears).
        {
            net::Snapshot curr = prev;
            curr.tick = 45;
            curr.entities.erase(curr.entities.begin() + 1);  // drop id 1
            std::vector<uint8_t> d = net::DeltaEncode(prev, curr);
            check(net::DeltaApply(prev, d) == curr, "delta: removed entity round-trips");
        }
        // (e) combined: move + add + remove in one delta.
        {
            net::Snapshot curr = prev;
            curr.tick = 46;
            curr.entities[0].position.x += 3.0f;           // move player
            curr.entities.erase(curr.entities.begin() + 2); // remove id 2
            curr.entities.push_back({5, {1, 1, 1}, math::Quat::Identity(), net::kFlagPickup}); // add id 5
            std::vector<uint8_t> d = net::DeltaEncode(prev, curr);
            check(net::DeltaApply(prev, d) == curr, "delta: combined move+add+remove round-trips");
        }
    }

    // --- 3. Replica == authority over a snapshot stream (keyframe + deltas) -----------------------
    // Build a deterministic sequence of authority snapshots, push each through the Replicator
    // (authority Send -> replica Receive), and assert exact match after every Receive. Capture the
    // byte counters; then re-run and assert the whole thing is deterministic.
    auto runStream = [](std::vector<net::Snapshot>& outAuthority,
                        uint64_t& outFull, uint64_t& outDelta, uint32_t& outKf, bool& outAllMatch) {
        net::Replicator rep(/*keyframeInterval=*/4);
        std::vector<net::Snapshot> authority;
        bool allMatch = true;
        net::Snapshot s = MakeSample();
        for (uint32_t t = 0; t < 20; ++t) {
            s.tick = t;
            // Evolve deterministically: drift the player, occasionally drop/add a pickup.
            s.entities[0].position.x = 0.1f * (float)t;
            s.entities[0].position.z = 0.05f * (float)t;
            if (t == 6 && s.entities.size() > 1) s.entities.erase(s.entities.begin() + 1);
            if (t == 12) s.entities.push_back({100 + t, {(float)t, 0.3f, 0.0f},
                                               math::Quat::Identity(), net::kFlagPickup});
            authority.push_back(s);
            std::vector<uint8_t> packet = rep.Send(s);
            rep.Receive(packet);
            if (rep.State() != s) allMatch = false;
            if (!rep.Matches(s)) allMatch = false;
        }
        outAuthority = authority;
        outFull = rep.FullBytes();
        outDelta = rep.DeltaBytes();
        outKf = rep.KeyframesSent();
        outAllMatch = allMatch;
    };

    std::vector<net::Snapshot> auth1, auth2;
    uint64_t full1 = 0, delta1 = 0, full2 = 0, delta2 = 0;
    uint32_t kf1 = 0, kf2 = 0;
    bool match1 = false, match2 = false;
    runStream(auth1, full1, delta1, kf1, match1);
    runStream(auth2, full2, delta2, kf2, match2);

    check(match1, "stream: replica.State() == authority[i] after every Receive");
    check(delta1 < full1, "stream: total delta bytes < total full bytes (savings)");
    check(kf1 >= 1, "stream: at least one keyframe sent");
    check(match2 && full1 == full2 && delta1 == delta2 && kf1 == kf2 && auth1.size() == auth2.size(),
          "stream: two runs are deterministic (identical bytes + match)");

    // --- 4. Interpolation -------------------------------------------------------------------------
    {
        net::Snapshot a; a.tick = 0;
        a.entities.push_back({0, {0, 0, 0}, math::Quat::Identity(), net::kFlagPlayer});
        net::Snapshot b; b.tick = 1;
        // 90deg about Y for the orientation slerp; position 0 -> (4,2,0).
        b.entities.push_back({0, {4, 2, 0}, math::Quat{0.0f, 0.7071068f, 0.0f, 0.7071068f}, net::kFlagPlayer});

        net::Snapshot at0 = net::Interpolate(a, b, 0.0f);
        net::Snapshot at1 = net::Interpolate(a, b, 1.0f);
        check(at0 == a, "interp: alpha 0 == a");
        // tick of the result is b.tick; compare entity transforms for the alpha-1 case.
        check(at1.entities.size() == 1 && at1.entities[0] == b.entities[0],
              "interp: alpha 1 entity == b");

        net::Snapshot mid = net::Interpolate(a, b, 0.5f);
        const math::Vec3& mp = mid.entities[0].position;
        check(std::fabs(mp.x - 2.0f) < 1e-5f && std::fabs(mp.y - 1.0f) < 1e-5f &&
              std::fabs(mp.z - 0.0f) < 1e-5f, "interp: midpoint lerps position halfway");
        const math::Quat& mq = mid.entities[0].orientation;
        float qlen = std::sqrt(mq.x*mq.x + mq.y*mq.y + mq.z*mq.z + mq.w*mq.w);
        check(std::fabs(qlen - 1.0f) < 1e-5f, "interp: midpoint orientation normalized");
        // Clamp: alpha outside [0,1] clamps.
        check(net::Interpolate(a, b, -1.0f) == net::Interpolate(a, b, 0.0f), "interp: alpha < 0 clamps to 0");
        check(net::Interpolate(a, b, 2.0f).entities[0] == net::Interpolate(a, b, 1.0f).entities[0],
              "interp: alpha > 1 clamps to 1");
    }

    // --- 5. Roll-game capture ---------------------------------------------------------------------
    // A known GameState (the fresh roll game: player at start + 3 pickups) captures to the expected
    // RepEntities: player id 0 at the player body position, pickups id 1..3 at their positions.
    {
        physics::World world;
        game::GameState gs = game::MakeRollGame(world);
        net::Snapshot snap = net::Replicator::Capture(/*tick=*/0, gs, world);

        check(snap.tick == 0, "capture: tick set");
        // 1 player + 3 uncollected pickups.
        check(snap.entities.size() == 4, "capture: player + 3 pickups -> 4 entities");
        const net::RepEntity& player = snap.entities[0];
        const physics::RigidBody& body = world.bodies[(size_t)gs.playerBodyIndex];
        check(player.id == 0 && (player.flags & net::kFlagPlayer) != 0, "capture: entity 0 is the player");
        check(player.position.x == body.position.x && player.position.y == body.position.y &&
              player.position.z == body.position.z, "capture: player position == body position");
        // Pickups follow in order, ids 1..3, at their pickup positions, flagged pickup.
        for (size_t i = 0; i < gs.pickups.size(); ++i) {
            const net::RepEntity& e = snap.entities[1 + i];
            check(e.id == (uint32_t)(1 + i), "capture: pickup id is 1-based in order");
            check((e.flags & net::kFlagPickup) != 0, "capture: pickup flagged");
            check(e.position.x == gs.pickups[i].pos.x && e.position.z == gs.pickups[i].pos.z,
                  "capture: pickup position == pickup pos");
        }

        // After collecting a pickup, it drops out of the capture (entity removed).
        gs.pickups[0].collected = true;
        net::Snapshot snap2 = net::Replicator::Capture(/*tick=*/1, gs, world);
        check(snap2.entities.size() == 3, "capture: collected pickup is not replicated");
        // The remaining pickups keep their ORIGINAL ids (stable per pickup index, not re-packed).
        check(snap2.entities[1].id == 2 && snap2.entities[2].id == 3,
              "capture: surviving pickups keep their stable ids");

        // Determinism: capturing the same state twice is identical bytes.
        check(net::Serialize(net::Replicator::Capture(0, gs, world)) ==
              net::Serialize(net::Replicator::Capture(0, gs, world)), "capture: deterministic");
    }

    if (g_fail == 0) { std::printf("replication_test OK\n"); return 0; }
    std::printf("replication_test: %d failures\n", g_fail);
    return 1;
}
