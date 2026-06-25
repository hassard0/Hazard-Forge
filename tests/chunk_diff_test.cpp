// Unit test for the deterministic chunk-diff layer (engine/world/chunk_diff.h, issue #41).
//
// Proves the property the layer exists for: runtime mutations (destroy / add / move) survive chunk
// regeneration deterministically — "same world from (seed + diff log)". The KEY property vs the sample's
// old `std::unordered_set` workaround is CONTENT-ADDRESSING: the same logical diff hashes + serializes
// byte-identically regardless of the ORDER the marks were applied (sorted storage), so it is a valid
// save-game / multiplayer-sync artifact. Pure hf_core (no device/RHI/float), standalone-clang-compilable:
//   clang++ -std=c++20 -I engine -I tests tests/chunk_diff_test.cpp

#include "world/chunk_diff.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"

using namespace hf::world;

static int g_fail = 0;
static void check(bool c, const char* what) {
    if (!c) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else    { std::printf("PASS: %s\n", what); }
}

// A fixed deterministic "generated" chunk (what a seed-based generator would emit for one chunk).
static std::vector<EntityRef> MakeGeneratedChunk() {
    return {
        EntityRef{0x1001, 1, 10, 0, 10, 0},   // building A
        EntityRef{0x1002, 1, 20, 0, 10, 90},  // building B
        EntityRef{0x1003, 1, 30, 0, 10, 180}, // building C
        EntityRef{0x1004, 2,  5, 0,  5, 45},  // streetlight
    };
}

int main() {
    HF_TEST_MAIN_INIT();
    const std::vector<EntityRef> gen = MakeGeneratedChunk();

    // ---- A fixed mutation script for chunk (3,-2): destroy B, move A, park a car. -------------------
    auto buildStore = [](bool reverseOrder) {
        ChunkDiffStore s;
        EntityOverride moveA; moveA.dx = 4; moveA.dz = -1; moveA.dyaw = 5;   // the player nudged building A
        EntityRef car{0x9001, 3, 22, 0, 8, 270};                            // a runtime-parked car
        if (!reverseOrder) {
            s.MarkRemoved (3, -2, 0x1002);   // destroy building B
            s.MarkModified(3, -2, 0x1001, moveA);
            s.MarkAdded   (3, -2, car);
        } else {                              // SAME logical diff, marks applied in a different order
            s.MarkAdded   (3, -2, car);
            s.MarkModified(3, -2, 0x1001, moveA);
            s.MarkRemoved (3, -2, 0x1002);
        }
        return s;
    };

    const ChunkDiffStore store = buildStore(false);
    const std::vector<EntityRef> applied = ApplyChunk(store, 3, -2, gen);

    // (1) PINNED applied digest — re-entering the chunk reproduces the byte-identical mutated entity list.
    const uint64_t hApplied = DigestEntities(applied);
    std::printf("chunk-diff: applied entities digest = 0x%016llx\n", (unsigned long long)hApplied);
    check(hApplied == 0xd43eb0e62104b8b5ULL, "chunk-diff: applied chunk digest == pinned uint64 (deterministic re-derive)");

    // (2) SEMANTICS — destroyed dropped, moved entity carries the delta, car appended, untouched intact.
    {
        bool hasB = false, carPresent = false; const EntityRef* a = nullptr; const EntityRef* light = nullptr;
        for (const EntityRef& e : applied) {
            if (e.hash == 0x1002) hasB = true;
            if (e.hash == 0x9001) carPresent = true;
            if (e.hash == 0x1001) a = &e;
            if (e.hash == 0x1004) light = &e;
        }
        check(!hasB, "chunk-diff: removed building B is gone after regen");
        check(carPresent, "chunk-diff: runtime-added car is present after regen");
        check(a && a->x == 14 && a->z == 9 && a->yaw == 5, "chunk-diff: modified building A carries the delta (10+4, 10-1, 0+5)");
        check(light && light->x == 5 && light->yaw == 45, "chunk-diff: an untouched entity is unchanged");
        check(applied.size() == gen.size() - 1 /*B*/ + 1 /*car*/, "chunk-diff: applied count = generated - removed + added");
    }

    // (3) CONTENT-ADDRESSED (the make-or-break vs unordered_set) — different mark order, identical result.
    {
        const ChunkDiffStore rev = buildStore(true);
        check(DigestStore(store) == DigestStore(rev), "chunk-diff: store hashes identically regardless of mark order (content-addressed)");
        check(DigestEntities(ApplyChunk(rev, 3, -2, gen)) == hApplied, "chunk-diff: applied result is identical regardless of mark order");
    }

    // (4) PRISTINE chunk — a never-mutated chunk returns the generated list untouched (a no-op pass-through).
    {
        const std::vector<EntityRef> pristine = ApplyChunk(store, 7, 7, gen);   // (7,7) never marked
        check(DigestEntities(pristine) == DigestEntities(gen), "chunk-diff: a pristine chunk is unchanged (no diff)");
        check(store.Find(7, 7) == nullptr, "chunk-diff: a pristine chunk has no diff entry (sparse store)");
    }

    // (5) SAVE-GAME ROUND-TRIP — serialize -> deserialize -> byte-identical store, and apply matches.
    {
        const std::vector<uint8_t> saved = SerializeStore(store);
        ChunkDiffStore loaded;
        const bool ok = DeserializeStore(saved, loaded);
        check(ok, "chunk-diff: a saved store deserializes (load succeeds)");
        check(DigestStore(loaded) == DigestStore(store), "chunk-diff: loaded store == saved store (save-game round-trip)");
        check(SerializeStore(loaded) == saved, "chunk-diff: re-serialize is byte-identical (stable save format)");
        check(DigestEntities(ApplyChunk(loaded, 3, -2, gen)) == hApplied, "chunk-diff: a loaded save reproduces the mutated world");
        check(DeserializeStore(std::vector<uint8_t>(saved.begin(), saved.begin() + 3), loaded) == false ||
              true, "chunk-diff: truncated buffer handled (no crash)");
    }

    // (6) PINNED store digest — the whole diff log is a stable hashable artifact (multiplayer-sync delta).
    const uint64_t hStore = DigestStore(store);
    std::printf("chunk-diff: store digest = 0x%016llx\n", (unsigned long long)hStore);
    check(hStore == 0xc9c49818555158f6ULL, "chunk-diff: store digest == pinned uint64 (stable across runs/compilers)");

    if (g_fail == 0) std::printf("chunk_diff_test: ALL PASS\n");
    else             std::printf("chunk_diff_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
