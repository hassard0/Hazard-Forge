// Deterministic integer LEDGER + atomic transactions (Slice ECON-S1, flagship #30 ECON, beachhead).
//
// The engine's FIRST gameplay-systems layer: inventory/economy state as PURE INTEGER bookkeeping —
// a dense fixed-order stock grid (entity x item) mutated only through an atomic transaction primitive
// (Add / Remove / Transfer). Because every operation is integer set-arithmetic over a fixed-iteration-
// order array (the wfc::Grid / ai flat-blackboard discipline — NO map, NO hash-ordered container, NO
// float), the ledger is bit-identical CPU/Vulkan/Metal BY CONSTRUCTION and lockstep/rollback/replay-able
// over net::Session (the Command IS the future per-tick Input). This is exactly what UE5's gameplay glue
// (Blueprint logic, replicated actor state, float timers, GAS) structurally cannot do.
//
// SELF-CONTAINED ON PURPOSE: this header includes ONLY <cstdint>/<cstddef>/<vector> plus net/session.h
// (itself self-contained, for hf::net::DigestBytes — the FNV-1a-64 state-digest currency). NO fpx / RHI
// / GPU / shader / <cmath> / float / clock / RNG / <random> / <unordered_*> / <map> / <functional> /
// std::hash. It compiles STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/econ_test.cpp`
// (like session_test.cpp / wfc_test.cpp) — the cheapest cross-platform proof loop in the engine.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "net/session.h"  // hf::net::DigestBytes (FNV-1a-64) — self-contained, read-only reuse

namespace hf::econ {

// An integer item quantity. Signed for delta math; stock stays >= 0 (the Affordable check prevents
// a Remove/Transfer from driving a slot negative). v1 assumes bounded fixture quantities so int32
// never overflows — no saturating-add is needed.
using Qty = int32_t;

// A fixed-size, fixed-order LEDGER: entity e owns stock[e*itemCount + t] — a dense integer grid
// (the wfc::Grid / ai flat-blackboard discipline; NO map, deterministic iteration by construction).
// Row-major: entity MAJOR, item MINOR.
struct World {
    uint32_t         entityCount = 0;
    uint32_t         itemCount   = 0;
    std::vector<Qty> stock;  // [entityCount * itemCount], row-major: entity major, item minor

    Qty  At(uint32_t e, uint32_t t) const { return stock[(std::size_t)e * itemCount + t]; }
    void Set(uint32_t e, uint32_t t, Qty q) { stock[(std::size_t)e * itemCount + t] = q; }
};

// A transaction Command — ALSO the future net::Session Input (S5). kAdd mints onto `dst`, kRemove burns
// from `src`, kTransfer moves `amount` of `item` from `src` to `dst`.
enum Kind : uint32_t { kAdd = 0, kRemove = 1, kTransfer = 2 };
struct Command {
    uint32_t kind;    // Kind
    uint32_t src;     // source entity (kRemove/kTransfer); ignored by kAdd
    uint32_t dst;     // dest entity   (kAdd/kTransfer);    ignored by kRemove
    uint32_t item;    // item id
    Qty      amount;  // quantity (must be > 0; a negative or zero amount is a deterministic no-op)
};

// --- InRange: bounds guard. An out-of-range command is a deterministic no-op (the ApplyCommand gate). -
inline bool InRange(const World& w, uint32_t e, uint32_t item) {
    return e < w.entityCount && item < w.itemCount;
}

// --- Affordable: a Remove/Transfer that can't pay is a deterministic no-op. ---------------------------
inline bool Affordable(const World& w, uint32_t e, uint32_t item, Qty amount) {
    return InRange(w, e, item) && w.At(e, item) >= amount;
}

// --- ApplyCommand: the atomic transaction. Returns true IFF applied (else false + NO mutation). -------
// Reject (false, no mutation) if amount <= 0, or a required id is out of range, or unaffordable.
//   kAdd:      Set(dst,item, At(dst,item)+amount)  (mints — does NOT preserve TotalQuantity).
//   kRemove:   requires Affordable(src,item,amount); Set(src,item, At-amount)  (burns).
//   kTransfer: requires Affordable(src,item,amount); atomic Remove(src)+Add(dst) (CONSERVES total).
//              src==dst is a deterministic no-op-but-valid (net zero) — returns true, no change.
inline bool ApplyCommand(World& w, const Command& c) {
    if (c.amount <= 0) return false;  // a non-positive amount is a deterministic no-op
    switch (c.kind) {
        case kAdd: {
            if (!InRange(w, c.dst, c.item)) return false;
            w.Set(c.dst, c.item, w.At(c.dst, c.item) + c.amount);
            return true;
        }
        case kRemove: {
            if (!Affordable(w, c.src, c.item, c.amount)) return false;
            w.Set(c.src, c.item, w.At(c.src, c.item) - c.amount);
            return true;
        }
        case kTransfer: {
            // BOTH endpoints must be in range; the SOURCE must be able to pay. Validate fully BEFORE any
            // mutation so a rejected transfer leaves the ledger untouched (atomic).
            if (!InRange(w, c.dst, c.item)) return false;
            if (!Affordable(w, c.src, c.item, c.amount)) return false;
            if (c.src == c.dst) return true;  // net-zero move: valid no-op, no mutation
            w.Set(c.src, c.item, w.At(c.src, c.item) - c.amount);  // burn from src
            w.Set(c.dst, c.item, w.At(c.dst, c.item) + c.amount);  // mint onto dst (conserves total)
            return true;
        }
        default:
            return false;  // an unknown kind is a deterministic no-op
    }
}

// --- RunScript: apply every command in ARRAY ORDER (the deterministic input-order contract). ----------
inline void RunScript(World& w, const std::vector<Command>& cmds) {
    for (const Command& c : cmds) ApplyCommand(w, c);
}

// --- DigestWorld: a PURE function of (entityCount, itemCount, stock bytes in order) via DigestBytes. ---
// Hash the two counts as a fixed-order header, then fold the contiguous stock bytes into the running
// FNV-1a-64 by continuing the same byte loop (the wfc::DigestGrid precedent). Bit-identical run-to-run
// AND platform-to-platform (MSVC vs Apple clang) because the byte layout + iteration order are fixed.
inline uint64_t DigestWorld(const World& w) {
    const uint32_t hdr[2] = { w.entityCount, w.itemCount };
    uint64_t h = net::DigestBytes(hdr, sizeof hdr);  // FNV-1a-64 over the {entityCount, itemCount} header
    // Continue the SAME FNV-1a-64 over the stock bytes (offset basis = the header's running hash), so the
    // digest is a single deterministic fold of (header || stock) — no float, no map, fixed order.
    const auto* p = reinterpret_cast<const unsigned char*>(w.stock.data());
    const std::size_t n = w.stock.size() * sizeof(Qty);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ull;  // the FNV-1a 64-bit prime (same constant net::DigestBytes uses)
    }
    return h;
}

// --- TotalQuantity: sum the entire stock in fixed order (int64 accumulation, returned as int64). ------
// The conservation invariant: kAdd/kRemove change it intentionally, kTransfer preserves it.
inline int64_t TotalQuantity(const World& w) {
    int64_t total = 0;
    for (const Qty q : w.stock) total += static_cast<int64_t>(q);
    return total;
}

// --- MakeShowcaseWorld: a FIXED starting ledger with varied nonzero stock (integer literals). ---------
// FIXED forever (the golden pins scripts over it). `entityCount` rows x `itemCount` columns; cell (e,t)
// is seeded with a deterministic varied integer so transfers/removes have something to move.
inline World MakeShowcaseWorld(uint32_t entityCount, uint32_t itemCount) {
    World w;
    w.entityCount = entityCount;
    w.itemCount   = itemCount;
    w.stock.assign((std::size_t)entityCount * itemCount, Qty{0});
    // A fixed deterministic seed pattern: entity e, item t -> 10 + 7*e + 3*t (varied, all nonzero,
    // integer literals only — no RNG, no float). Bounded well within int32.
    for (uint32_t e = 0; e < entityCount; ++e)
        for (uint32_t t = 0; t < itemCount; ++t)
            w.Set(e, t, static_cast<Qty>(10 + 7 * static_cast<int>(e) + 3 * static_cast<int>(t)));
    return w;
}

// --- MakeShowcaseScript: a FIXED script exercising Add, Remove, Transfer + the two reject gates. -------
// Designed for a >=4-entity x >=4-item showcase world. Includes at least one UNAFFORDABLE Remove (the
// no-op affordability gate) and at least one OUT-OF-RANGE command (the bounds gate). FIXED forever.
inline std::vector<Command> MakeShowcaseScript() {
    return std::vector<Command>{
        // kind,        src, dst, item, amount
        { kAdd,         0,   0,   0,    5  },   // mint 5 of item 0 onto entity 0
        { kAdd,         0,   1,   2,    20 },   // mint 20 of item 2 onto entity 1
        { kRemove,      2,   0,   1,    4  },   // burn 4 of item 1 from entity 2
        { kTransfer,    0,   3,   0,    6  },   // move 6 of item 0 from entity 0 -> entity 3
        { kTransfer,    1,   2,   2,    9  },   // move 9 of item 2 from entity 1 -> entity 2
        { kRemove,      3,   0,   3,    1000 }, // UNAFFORDABLE remove (entity 3 has < 1000 of item 3) -> no-op
        { kTransfer,    2,   2,   1,    3  },   // src==dst transfer: valid net-zero no-op (returns true)
        { kAdd,         0,   99,  0,    7  },   // OUT-OF-RANGE dst entity (99 >= entityCount) -> no-op
        { kTransfer,    1,   2,   99,   2  },   // OUT-OF-RANGE item (99) -> no-op
        { kRemove,      0,   0,   0,    -5 },   // NON-POSITIVE amount -> no-op
        { kAdd,         0,   2,   3,    11 },   // mint 11 of item 3 onto entity 2
        { kTransfer,    3,   0,   0,    2  },   // move 2 of item 0 from entity 3 -> entity 0
    };
}

// =====================================================================================================
// ECON-S2 — Crafting / recipe transformer + deterministic craft queue (APPEND-ONLY below S1).
// =====================================================================================================
// Adds the CRAFTING layer on top of S1's integer ledger: a RecipeSet (inputs -> outputs, integer
// costs/yields) + an atomic ApplyRecipe that consumes inputs and produces outputs IFF affordable, plus a
// deterministic craft queue drained in fixed array order. Strict integer, reuses the S1 ledger primitives
// (At/Set/Affordable/InRange) VERBATIM — no new bookkeeping. Bit-identical CPU/Vulkan/Metal by the same
// fixed-iteration-order / no-map / no-float discipline as S1.

// --- Ingredient: one (item, amount) term of a recipe. amount > 0 by fixture construction. -------------
struct Ingredient {
    uint32_t item;    // item id
    Qty      amount;  // quantity consumed (input) or produced (output); > 0
};

// --- Recipe: one crafting transform on a single entity's own stock. Inputs are consumed, outputs are
// produced. Flat fixed-order ingredient lists (NO map) so iteration is deterministic by construction. ---
struct Recipe {
    std::vector<Ingredient> inputs;   // all consumed (every one must be affordable)
    std::vector<Ingredient> outputs;  // all produced
};

// --- RecipeSet: the fixed catalogue, indexed by recipe id (= the recipes-vector index / iteration order).
struct RecipeSet {
    std::vector<Recipe> recipes;
};

// --- RecipeAffordable: recipeId in range, entity in range, and the entity holds >= amount of EVERY input
// (scan inputs in array order). A recipe that can't pay is a deterministic no-op. Reuses S1 Affordable. --
inline bool RecipeAffordable(const World& w, const RecipeSet& rs, uint32_t recipeId, uint32_t entity) {
    if (recipeId >= rs.recipes.size()) return false;
    const Recipe& r = rs.recipes[recipeId];
    for (const Ingredient& in : r.inputs)
        if (!Affordable(w, entity, in.item, in.amount)) return false;  // InRange + stock>=amount (S1)
    return true;
}

// --- ApplyRecipe: the atomic craft. Returns false + NO mutation if out of range or not affordable.
// Otherwise CONSUME every input (Set(entity,in.item, At-in.amount)) in array order, THEN PRODUCE every
// output (Set(entity,out.item, At+out.amount)) in array order. CONSUME-BEFORE-PRODUCE is PINNED: when an
// item is BOTH an input and an output the consume happens first, so the net delta is exactly
// (Sigma outputs - Sigma inputs) for that item (e.g. consume 1 then produce 2 == net +1). Reuses S1
// At/Set/Affordable; the up-front RecipeAffordable gate makes the consume loop never drive a slot negative.
inline bool ApplyRecipe(World& w, const RecipeSet& rs, uint32_t recipeId, uint32_t entity) {
    if (!RecipeAffordable(w, rs, recipeId, entity)) return false;  // out-of-range OR unaffordable -> no-op
    const Recipe& r = rs.recipes[recipeId];
    for (const Ingredient& in : r.inputs)                          // 1) consume every input (array order)
        w.Set(entity, in.item, w.At(entity, in.item) - in.amount);
    for (const Ingredient& out : r.outputs)                        // 2) produce every output (array order)
        w.Set(entity, out.item, w.At(entity, out.item) + out.amount);
    return true;
}

// --- CraftOrder: a queued craft request (craft recipeId on entity `count` times). ALSO a future
// net::Session Input alongside Command (S5). ----------------------------------------------------------
struct CraftOrder {
    uint32_t recipeId;
    uint32_t entity;
    uint32_t count;  // attempt to craft this many times (stops early if it becomes unaffordable)
};

// --- DrainCraftQueue: process the queue in ARRAY ORDER; for each order attempt ApplyRecipe up to `count`
// times, stopping EARLY when a craft becomes unaffordable (a partial drain is deterministic), then move to
// the next order. Fixed order throughout (the RunScript precedent). --------------------------------------
inline void DrainCraftQueue(World& w, const RecipeSet& rs, const std::vector<CraftOrder>& queue) {
    for (const CraftOrder& o : queue)
        for (uint32_t i = 0; i < o.count; ++i)
            if (!ApplyRecipe(w, rs, o.recipeId, o.entity)) break;  // partial drain: stop this order early
}

// --- MakeShowcaseRecipes: a FIXED recipe catalogue over the S1 item ids (integer literals only). FIXED
// forever (the S2 golden pins a craft queue over it). ---------------------------------------------------
//   r0 = {2x item0 + 1x item1} -> {1x item2}   (ore + fuel -> ingot; reduces total by 2 — transmutation)
//   r1 = {3x item2}            -> {1x item3}   (ingots -> tool)
//   r2 = {1x item2 + 1x item3} -> {2x item2}   (item2 is BOTH input and output -> pins consume-before-
//                                               produce: consume 1 then produce 2 == net +1 of item2)
inline RecipeSet MakeShowcaseRecipes() {
    RecipeSet rs;
    rs.recipes.push_back(Recipe{ /*inputs=*/{ {0, 2}, {1, 1} }, /*outputs=*/{ {2, 1} } });
    rs.recipes.push_back(Recipe{ /*inputs=*/{ {2, 3} },         /*outputs=*/{ {3, 1} } });
    rs.recipes.push_back(Recipe{ /*inputs=*/{ {2, 1}, {3, 1} }, /*outputs=*/{ {2, 2} } });
    return rs;
}

// --- MakeShowcaseCraftQueue: a FIXED queue exercising affordable crafts, a multi-count craft, a partial/
// zero drain (unaffordable), and an out-of-range recipeId. FIXED forever. Designed for a >=4-entity x
// >=4-item showcase world. ------------------------------------------------------------------------------
inline std::vector<CraftOrder> MakeShowcaseCraftQueue() {
    return std::vector<CraftOrder>{
        // recipeId, entity, count
        { 0, 0, 3   },   // craft r0 (ore+fuel->ingot) on entity 0, 3 times (multi-count; some may stop early)
        { 1, 0, 2   },   // craft r1 (ingots->tool) on entity 0, twice
        { 2, 0, 1   },   // craft r2 (the both-input-and-output recipe) on entity 0 once
        { 0, 1, 100 },   // big multi-count on entity 1 -> PARTIAL drain (affords floor(avail/cost), then stops)
        { 99, 0, 5  },   // OUT-OF-RANGE recipeId (99) -> zero crafts (no-op every attempt)
    };
}

}  // namespace hf::econ
