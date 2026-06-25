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

// =====================================================================================================
// ECON-S3 — Resource economy TICK: per-tick production / consumption flow (APPEND-ONLY below S2).
// =====================================================================================================
// The time-stepped core that makes the ledger a LIVING economy: producers add stock each tick, consumers
// drain it, run in a FIXED order with a zero-floor and integer storage caps. This is the gameplay-state
// `IntegrateStep` analog and stays strict INTEGER (the moat) — a determinism property a float-timer UE5
// economy cannot guarantee. Reuses S1's At/Set/InRange VERBATIM; NO new bookkeeping, NO map, NO float.

// --- Flow: one per-(entity,item) production OR consumption term. rate > 0 (units per tick). ------------
struct Flow {
    uint32_t entity;  // entity id
    uint32_t item;    // item id
    Qty      rate;    // units gained (producer) or lost (consumer) per tick; > 0 by fixture construction
};

// --- EconRules: the flat, fixed-order producer/consumer flows + an OPTIONAL dense storage-cap grid. ----
// `cap` is either EMPTY (size 0) meaning "no caps anywhere", OR a dense [entityCount*itemCount] array
// parallel to World::stock (same row-major indexing e*itemCount+t). A cap value <= 0 for a slot means
// that individual slot is UNCAPPED. (NO map — flat arrays keep iteration deterministic by construction.)
struct EconRules {
    std::vector<Flow> producers;  // each tick (array order): entity gains `rate` of `item`, clamped UP to cap
    std::vector<Flow> consumers;  // each tick (array order): entity loses `rate` of `item`, clamped DOWN to 0
    std::vector<Qty>  cap;        // [entityCount*itemCount] storage cap, OR empty = no caps; slot <=0 = uncapped
};

// --- CapAt: the cap for slot (e,t), or the sentinel -1 meaning UNCAPPED. CONVENTION (PINNED): return -1
// when `cap` is empty, when (e,t) is out of range, OR when the stored cap value is <= 0. Otherwise return
// the stored positive cap. Callers treat a return of -1 as "no clamp" (production is unbounded for it). ---
inline Qty CapAt(const EconRules& r, const World& w, uint32_t e, uint32_t t) {
    if (r.cap.empty()) return -1;                  // no cap grid at all -> uncapped
    if (!InRange(w, e, t)) return -1;              // out of range -> uncapped (defensive)
    const std::size_t idx = (std::size_t)e * w.itemCount + t;
    if (idx >= r.cap.size()) return -1;            // cap grid shorter than stock -> uncapped
    const Qty c = r.cap[idx];
    return c > 0 ? c : -1;                         // a non-positive cap value means "this slot is uncapped"
}

// --- EconTick: ONE economy step, in a FIXED two-phase order. ------------------------------------------
// PRODUCTION phase FIRST (all producers, array order): newQ = At(e,item) + rate; if the slot is capped
// (CapAt != -1), clamp UP: newQ = min(newQ, cap); Set. Out-of-range producer = deterministic skip.
// CONSUMPTION phase SECOND, AFTER ALL production (all consumers, array order): newQ = max(0, At-rate);
// Set. Out-of-range consumer = deterministic skip.
// PRODUCTION-BEFORE-CONSUMPTION is PINNED: when an entity BOTH produces and consumes the SAME item in one
// tick, the produced stock is added first (and capped), THEN the consumer drains from the post-production
// value — so a slot that produces P and consumes C nets +(min(At+P,cap) - C) clamped to >= 0, NOT
// At+(P-C). Integer min/max are inline ternaries (no <algorithm>); strict integer, no float.
inline void EconTick(World& w, const EconRules& r) {
    // --- Production phase (clamp UP to the per-slot cap if one exists). ---
    for (const Flow& f : r.producers) {
        if (!InRange(w, f.entity, f.item)) continue;     // out-of-range producer -> deterministic skip
        Qty newQ = w.At(f.entity, f.item) + f.rate;      // gain `rate`
        const Qty cap = CapAt(r, w, f.entity, f.item);
        if (cap != -1 && newQ > cap) newQ = cap;         // clamp UP: production never exceeds storage cap
        w.Set(f.entity, f.item, newQ);
    }
    // --- Consumption phase (AFTER all production; clamp DOWN to zero). ---
    for (const Flow& f : r.consumers) {
        if (!InRange(w, f.entity, f.item)) continue;      // out-of-range consumer -> deterministic skip
        Qty newQ = w.At(f.entity, f.item) - f.rate;       // lose `rate`
        if (newQ < 0) newQ = 0;                           // clamp DOWN: stock never goes negative
        w.Set(f.entity, f.item, newQ);
    }
}

// --- RunEconTicks: step the economy `ticks` times (fixed loop; deterministic of (w, r, ticks) alone —
// the RunLockstep / StepWorldN precedent). ------------------------------------------------------------
inline void RunEconTicks(World& w, const EconRules& r, uint32_t ticks) {
    for (uint32_t i = 0; i < ticks; ++i) EconTick(w, r);
}

// --- MakeShowcaseRules: a FIXED producer/consumer/cap fixture over the S1 4x4 showcase world (integer
// literals only). FIXED forever (the S3 golden pins the post-N-ticks digest over it). Designed so the
// economy reaches STEADY STATE well within 24 ticks. Recall the seed: stock(e,t) = 10 + 7*e + 3*t.
//
//   PRODUCERS (gain per tick, clamped UP to cap):
//     (e0,t0) +5/tick, cap 100   -> PURE producer: starts 10, rises 10,15,...,100, then holds at the cap.
//     (e1,t1) +8/tick, cap 60    -> pure producer: starts 20, rises to 60, holds (a second capped saturator).
//     (e2,t2) +6/tick, cap 90    -> BOTH producer AND consumer of the same (e2,t2) slot (pins phase order).
//   CONSUMERS (lose per tick, clamped DOWN to 0):
//     (e3,t3) -4/tick            -> PURE consumer (no producer): starts 10+21+9=... -> drains to 0, holds.
//     (e2,t2) -2/tick            -> the SAME slot the (e2,t2) producer feeds: net +6-2=+4/tick UP to cap 90
//                                   (production-before-consumption: add 6 [cap 90] THEN subtract 2).
//
// Net per tick for (e2,t2): min(At+6, 90) - 2. Starts At=10+14+6=30 -> 30+6=36-2=34 -> ... climbs by +4
// until it nears 90, where the cap clamps production and the -2 consumer pulls it to a FIXED rest point of
// 88 (90 produced-cap minus 2 consumed). So (e2,t2) ALSO settles (to 88) -> the whole economy is idempotent
// at rest -> the steady-state assertion holds. Every flow saturates/floors before tick 24.
inline EconRules MakeShowcaseRules(const World& w) {
    EconRules r;
    r.producers = std::vector<Flow>{
        // entity, item, rate
        { 0, 0, 5 },   // pure producer -> saturates to cap 100
        { 1, 1, 8 },   // pure producer -> saturates to cap 60
        { 2, 2, 6 },   // BOTH produces and consumes (e2,t2) -> pins production-before-consumption
    };
    r.consumers = std::vector<Flow>{
        // entity, item, rate
        { 3, 3, 4 },   // PURE consumer (no producer feeds it) -> drains to 0
        { 2, 2, 2 },   // consumes the SAME slot (e2,t2) the producer above feeds -> rest point cap-2
    };
    // Dense cap grid parallel to stock; default 0 (= uncapped) for every slot, then set the few real caps.
    r.cap.assign((std::size_t)w.entityCount * w.itemCount, Qty{0});
    if (InRange(w, 0, 0)) r.cap[(std::size_t)0 * w.itemCount + 0] = 100;  // (e0,t0) cap
    if (InRange(w, 1, 1)) r.cap[(std::size_t)1 * w.itemCount + 1] = 60;   // (e1,t1) cap
    if (InRange(w, 2, 2)) r.cap[(std::size_t)2 * w.itemCount + 2] = 90;   // (e2,t2) cap (the both-flows slot)
    return r;
}

// =====================================================================================================
// ECON-S4 — Pricing / market clearing + deterministic rolls (APPEND-ONLY below S3).
// =====================================================================================================
// The MARKET layer: integer supply/demand pricing (integer-ratio elasticity, NO fxmul/fpx — keeps econ.h
// self-contained), deterministic order clearing (trades at the current price, paid in a designated
// currency item, reusing the S1 transfer primitive), and DETERMINISTIC ROLLS (a copied PcgHash — the
// engine.h stays self-contained; pcg.h would pull the fx/particles chain). Pure-CPU INTEGER, the same
// fixed-iteration-order / no-map / no-float discipline as S1-S3 -> bit-identical CPU/Vulkan/Metal AND
// lockstep/replay-able. A deterministic, reproducible market + loot system is exactly what a float-RNG
// UE5 economy cannot lockstep/replay.

// --- EconHash: copied VERBATIM from engine/pcg/pcg.h:42-48 (the canonical PcgHash) -------------------
// pcg.h pulls the fx/particles include chain, which would break this header's self-containment, so we
// COPY the pure-uint32 ops (the wfc::WfcHash precedent) — same constants -> same stream -> bit-exact
// cross-platform. NO RNG, NO clock, NO float; pure uint32 wrapping arithmetic + shifts.
inline uint32_t EconHash(uint32_t seed, uint32_t index) {
    uint32_t h = seed * 2654435761u;               // Knuth multiplicative
    h ^= (index + 0x9E3779B9u + (h << 6) + (h >> 2));
    h += index * 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
inline constexpr uint32_t kRollSalt = 0x45434F4Eu;  // 'ECON' — the roll stream's salt

// --- RollRange: a deterministic integer in [lo, hi] INCLUSIVE from (seed, index). Requires hi >= lo. --
// lo + (EconHash(seed ^ kRollSalt, index) mod (hi - lo + 1)). Pure uint32 (the modulus is unsigned) ->
// bit-exact cross-platform. Used for seeded loot/yield rolls; replay-stable by (seed, index) alone.
inline Qty RollRange(uint32_t seed, uint32_t index, Qty lo, Qty hi) {
    const uint32_t span = static_cast<uint32_t>(hi - lo + 1);  // hi >= lo by contract -> span >= 1
    return lo + static_cast<Qty>(EconHash(seed ^ kRollSalt, index) % span);
}

// --- Market: per-item integer prices + bounds + the designated currency item. -------------------------
// `price` is a dense [itemCount] array (the same fixed-order / no-map discipline as World::stock). The
// `currency` item is money — excluded from its own pricing and rejected as a tradeable good.
struct Market {
    std::vector<Qty> price;            // [itemCount] current unit price per item id
    Qty              minPrice = 1;
    Qty              maxPrice = 1000000;
    uint32_t         currency = 0;     // the item id used as money (excluded from its own pricing)
};

// --- UpdatePrice: the integer-ratio elasticity update, MONOTONIC + clamped. ----------------------------
// delta = demand - supply (>0 excess demand, <0 excess supply); step = (int64)delta * elastNum / elastDen
// with an int64 multiply (avoids overflow) and truncating division toward zero (the C++ `/` default —
// PINNED). newP = clamp(price + step, minP, maxP) via integer ternaries (no <algorithm>). Excess demand
// raises the price, excess supply lowers it, balanced (or a step truncated to 0) holds it (within clamp).
inline Qty UpdatePrice(Qty price, Qty demand, Qty supply, Qty elastNum, Qty elastDen,
                       Qty minP, Qty maxP) {
    const Qty delta = demand - supply;
    const int64_t step = static_cast<int64_t>(delta) * elastNum / elastDen;  // truncating toward zero (PINNED)
    int64_t newP = static_cast<int64_t>(price) + step;
    if (newP < minP) newP = minP;     // clamp DOWN to minPrice (integer ternary, no <algorithm>)
    if (newP > maxP) newP = maxP;     // clamp UP to maxPrice
    return static_cast<Qty>(newP);
}

// --- UpdatePrices: apply UpdatePrice to every item in FIXED id order, SKIPPING the currency item. ------
// `demand`/`supply` are dense [itemCount] arrays parallel to m.price. The currency item's price is never
// repriced (it is the unit of account). Deterministic of (m, demand, supply, elastNum, elastDen) alone.
inline void UpdatePrices(Market& m, const std::vector<Qty>& demand, const std::vector<Qty>& supply,
                         Qty elastNum, Qty elastDen) {
    for (uint32_t i = 0; i < m.price.size(); ++i) {
        if (i == m.currency) continue;                 // the currency item is never repriced
        m.price[i] = UpdatePrice(m.price[i], demand[i], supply[i], elastNum, elastDen,
                                 m.minPrice, m.maxPrice);
    }
}

// --- TradeOrder: a buy/sell at the current price, paid in the currency item. qty > 0 by construction. --
struct TradeOrder {
    uint32_t buyer;   // entity receiving `qty` of `item`, paying currency
    uint32_t seller;  // entity giving up `qty` of `item`, receiving currency
    uint32_t item;    // the good traded (must NOT be the currency item)
    Qty      qty;     // quantity of `item` (> 0)
};

// --- ExecuteTrade: execute a single order at the price IN `m`. Returns true IFF applied (else false +
// NO mutation). Execute IFF: qty > 0, item != currency, buyer/seller/item all in range, the seller holds
// `qty` of `item`, AND the buyer holds `cost = (int64)qty * price[item]` currency (fixtures keep cost
// bounded within Qty). Then atomically transfer `qty` of `item` seller->buyer and `cost` currency
// buyer->seller (reusing the S1 kTransfer primitive, so each leg is itself atomic/validated). An invalid
// or unaffordable order is a deterministic no-op. ------------------------------------------------------
inline bool ExecuteTrade(World& w, const Market& m, const TradeOrder& o) {
    if (o.qty <= 0) return false;                              // non-positive qty -> no-op
    if (o.item == m.currency) return false;                   // a good cannot be the currency item
    if (!InRange(w, o.buyer, o.item)) return false;           // buyer/item in range
    if (!InRange(w, o.seller, o.item)) return false;          // seller/item in range
    if (o.item >= m.price.size()) return false;               // price defined for this item
    if (!InRange(w, o.buyer, m.currency)) return false;       // currency slot in range for both
    if (!InRange(w, o.seller, m.currency)) return false;
    const int64_t cost = static_cast<int64_t>(o.qty) * m.price[o.item];  // total currency price
    // Validate FULLY before any mutation (atomic): seller has the goods, buyer can pay.
    if (!Affordable(w, o.seller, o.item, o.qty)) return false;          // seller has qty of item
    if (cost > w.At(o.buyer, m.currency)) return false;                 // buyer can pay the cost
    if (cost <= 0) return false;                                        // price 0 / overflow guard -> no-op
    const Qty cost32 = static_cast<Qty>(cost);
    // Two S1 transfers (each validated again internally; both endpoints are already in range + affordable).
    ApplyCommand(w, Command{ kTransfer, o.seller, o.buyer, o.item,     o.qty  });  // goods seller->buyer
    ApplyCommand(w, Command{ kTransfer, o.buyer,  o.seller, m.currency, cost32 }); // money buyer->seller
    return true;
}

// --- ClearMarket: execute every order in ARRAY ORDER at the price IN `m` at call time. S4 does NOT
// auto-reprice mid-batch (UpdatePrices is a separate explicit step), keeping the two concerns
// independently pinnable. Invalid/unaffordable orders are deterministic no-ops (the RunScript precedent).
inline void ClearMarket(World& w, const Market& m, const std::vector<TradeOrder>& orders) {
    for (const TradeOrder& o : orders) ExecuteTrade(w, m, o);
}

// --- DigestState: the combined market+ledger golden currency. Combine DigestWorld(w) with a
// net::DigestBytes over m.price + the bounds/currency, in FIXED order. Continues the SAME FNV-1a-64 fold
// (the DigestWorld precedent): seed with DigestWorld, then fold the price bytes, then the bounds/currency
// header. Bit-identical run-to-run AND platform-to-platform (fixed byte layout + iteration order). ------
inline uint64_t DigestState(const World& w, const Market& m) {
    uint64_t h = DigestWorld(w);  // start from the ledger digest (fixed-order FNV-1a-64 fold)
    // Fold the price array bytes (fixed order: item 0..itemCount-1).
    const auto* p = reinterpret_cast<const unsigned char*>(m.price.data());
    const std::size_t n = m.price.size() * sizeof(Qty);
    for (std::size_t i = 0; i < n; ++i) { h ^= static_cast<uint64_t>(p[i]); h *= 1099511628211ull; }
    // Fold the fixed-order {minPrice, maxPrice, currency} trailer.
    const Qty bounds[2] = { m.minPrice, m.maxPrice };
    const auto* bp = reinterpret_cast<const unsigned char*>(bounds);
    for (std::size_t i = 0; i < sizeof bounds; ++i) { h ^= static_cast<uint64_t>(bp[i]); h *= 1099511628211ull; }
    const auto* cp = reinterpret_cast<const unsigned char*>(&m.currency);
    for (std::size_t i = 0; i < sizeof m.currency; ++i) { h ^= static_cast<uint64_t>(cp[i]); h *= 1099511628211ull; }
    return h;
}

// --- MakeShowcaseMarket: a FIXED market over the 4x4 showcase world. FIXED forever (the S4 golden pins
// a clearing+repricing over it). The showcase world is 4 items so currency = item0 (money). Seeds every
// entity with extra item0 (coin) so trades can pay. Recall the seed: stock(e,t) = 10 + 7*e + 3*t -> with
// itemCount 4 every entity starts with 10+7e of item0; we ADD a fixed coin reserve so the affordable
// trades clear and an unaffordable one still has a buyer who simply can't cover that particular cost. ---
// NOTE: this MUTATES `w` is NOT desired — instead the test seeds coin into the world separately. To keep
// MakeShowcaseMarket a pure function of the world's itemCount, it only builds the price/bounds/currency.
inline Market MakeShowcaseMarket(const World& w) {
    Market m;
    m.minPrice = 1;
    m.maxPrice = 1000000;
    m.currency = 0;  // item0 is money (the 4x4 showcase has items 0..3)
    // Fixed integer prices per item (currency item0's price is unused but seeded for a stable digest).
    m.price.assign(w.itemCount, Qty{1});
    if (w.itemCount > 0) m.price[0] = 1;    // currency item — price never used (skipped by UpdatePrices)
    if (w.itemCount > 1) m.price[1] = 5;    // item1 costs 5 coin each
    if (w.itemCount > 2) m.price[2] = 12;   // item2 costs 12 coin each
    if (w.itemCount > 3) m.price[3] = 3;    // item3 costs 3 coin each
    return m;
}

// --- SeedShowcaseCoin: top up every entity's currency (item0) reserve so the showcase trades can pay.
// FIXED forever. Adds a fixed +500 coin to every entity's item0 slot (well within int32). Called by the
// test after MakeShowcaseWorld so buyers can afford the affordable trades (and an unaffordable order
// targets a buyer/cost that still cannot cover). Pure integer, fixed order. ----------------------------
inline void SeedShowcaseCoin(World& w) {
    for (uint32_t e = 0; e < w.entityCount; ++e)
        if (InRange(w, e, 0)) w.Set(e, 0, w.At(e, 0) + 500);  // +500 coin (item0) per entity
}

// --- MakeShowcaseTrades: a FIXED order book exercising affordable trades + the three reject gates. -----
// FIXED forever. Designed for the 4x4 coin-seeded showcase world (currency = item0). Includes affordable
// buy/sells, an UNAFFORDABLE order (cost exceeds the buyer's coin), a CURRENCY-AS-ITEM order (item ==
// currency -> rejected), and an OUT-OF-RANGE order (entity 99). ----------------------------------------
inline std::vector<TradeOrder> MakeShowcaseTrades() {
    return std::vector<TradeOrder>{
        // buyer, seller, item, qty
        { 0, 1, 2, 3 },        // e0 buys 3x item2 from e1 @12 = 36 coin (affordable: e0 has 510 coin)
        { 2, 3, 1, 4 },        // e2 buys 4x item1 from e3 @5  = 20 coin (affordable)
        { 1, 0, 3, 5 },        // e1 buys 5x item3 from e0 @3  = 15 coin (affordable)
        { 3, 2, 1, 2 },        // e3 buys 2x item1 from e2 @5  = 10 coin (affordable; e2 has the item1 it bought)
        { 0, 2, 2, 100000 },   // UNAFFORDABLE: 100000x item2 @12 = 1.2M coin > buyer's reserve -> no-op
        { 0, 1, 0, 5 },        // CURRENCY-AS-ITEM: item == currency (item0) -> rejected no-op
        { 0, 99, 1, 1 },       // OUT-OF-RANGE seller (99 >= entityCount) -> no-op
    };
}

// --- MakeShowcaseDemand / MakeShowcaseSupply: FIXED demand/supply vectors for the pricing test. -------
// FIXED forever. Parallel to m.price ([itemCount]). Item1 excess demand (raises), item2 excess supply
// (lowers), item3 balanced (holds). The currency slot (item0) is skipped by UpdatePrices.
inline std::vector<Qty> MakeShowcaseDemand(const World& w) {
    std::vector<Qty> d(w.itemCount, Qty{0});
    if (w.itemCount > 1) d[1] = 100;   // item1: strong demand
    if (w.itemCount > 2) d[2] = 10;    // item2: weak demand
    if (w.itemCount > 3) d[3] = 50;    // item3: balanced with supply -> holds
    return d;
}
inline std::vector<Qty> MakeShowcaseSupply(const World& w) {
    std::vector<Qty> s(w.itemCount, Qty{0});
    if (w.itemCount > 1) s[1] = 20;    // item1: supply < demand -> price rises
    if (w.itemCount > 2) s[2] = 90;    // item2: supply > demand -> price falls
    if (w.itemCount > 3) s[3] = 50;    // item3: supply == demand -> price holds
    return s;
}

// =====================================================================================================
// ECON-S5 — Quest state machine + lockstep/rollback/desync capstone (APPEND-ONLY below S4). HEADLINE.
// =====================================================================================================
// The flagship's headline: a deterministic integer QUEST / OBJECTIVE state machine advanced by the
// economy state, then the moat proof — the ENTIRE economy + quest state wrapped in net::Session so two
// peers fed only the command stream re-derive a BIT-IDENTICAL economy+quest state (RunLockstep), a
// mispredicted command ROLLS BACK to the bit-identical authority state (RollbackSession), and a
// divergence is LOCATED at the exact tick (DesyncDetector). Pure-CPU INTEGER, reuses S1-S4 verbatim +
// the net::* netcode machinery (read-only). This is the canonical non-deterministic gameplay glue
// (Blueprint logic, replicated actor state, float timers, GAS) that UE5 structurally cannot lockstep,
// deterministically rollback, or bit-exactly replay.

// --- Objective: one integer-condition quest goal. Condition: ledger.At(entity,item) >= threshold. -----
// `prereq` chains objectives: -1 = no prerequisite (starts ACTIVE); else this objective only ACTIVATES
// once objective[prereq] is COMPLETE. Flat fixed index (NO map) -> deterministic iteration by construction.
struct Objective {
    uint32_t entity;     // condition entity
    uint32_t item;       // condition item
    Qty      threshold;  // condition: ledger.At(entity, item) >= threshold
    int32_t  prereq;     // -1 = no prerequisite; else activates once objective[prereq] is complete
};

// --- QuestGraph: the STATIC quest config (NOT snapshotted — config, like RecipeSet/EconRules). --------
struct QuestGraph { std::vector<Objective> objectives; };

// --- ObjStatus: an objective's lifecycle. kLocked -> kActive (prereq met) -> kComplete (condition met).
enum ObjStatus : uint8_t { kLocked = 0, kActive = 1, kComplete = 2 };

// --- QuestState: parallel to QuestGraph::objectives; THIS is the snapshotted mutable state. -----------
struct QuestState { std::vector<uint8_t> status; };  // status[i] == ObjStatus of objective i

// --- MakeQuestState: all objectives kLocked except those with prereq == -1 which start kActive. -------
inline QuestState MakeQuestState(const QuestGraph& g) {
    QuestState q;
    q.status.assign(g.objectives.size(), static_cast<uint8_t>(kLocked));
    for (std::size_t i = 0; i < g.objectives.size(); ++i)
        if (g.objectives[i].prereq == -1) q.status[i] = static_cast<uint8_t>(kActive);
    return q;
}

// --- AdvanceQuests: ONE fixed-order pass over the objectives. A kLocked objective whose prereq is
// kComplete (or has no prereq) becomes kActive; a kActive objective whose integer condition holds
// (ledger.At(entity,item) >= threshold, bounds-checked) becomes kComplete. SINGLE PASS PER CALL: a
// multi-step chain completes across SUCCESSIVE ticks (one link per call), NOT within one call — this
// keeps the advance deterministic + lockstep-aligned (every peer advances exactly one pass per tick).
// Pure integer, no float, no map; the status vector is the only mutated state.
inline void AdvanceQuests(const World& ledger, const QuestGraph& g, QuestState& q) {
    for (std::size_t i = 0; i < g.objectives.size(); ++i) {
        const Objective& o = g.objectives[i];
        if (q.status[i] == static_cast<uint8_t>(kLocked)) {
            // Activate if there is no prereq, or the prereq objective is already complete.
            const bool prereqMet =
                (o.prereq == -1) ||
                (static_cast<std::size_t>(o.prereq) < q.status.size() &&
                 q.status[static_cast<std::size_t>(o.prereq)] == static_cast<uint8_t>(kComplete));
            if (prereqMet) q.status[i] = static_cast<uint8_t>(kActive);
        }
        if (q.status[i] == static_cast<uint8_t>(kActive)) {
            // Complete if the integer ledger condition holds (bounds-checked via InRange).
            if (InRange(ledger, o.entity, o.item) &&
                ledger.At(o.entity, o.item) >= o.threshold)
                q.status[i] = static_cast<uint8_t>(kComplete);
        }
    }
}

// --- EconState: the unified net::Session World — ALL mutable state, value-copy snapshot/restore-able.
// (The QuestGraph/EconRules/RecipeSet config lives OUTSIDE — captured by the step lambda — so this is a
// pure snapshot of mutable state, exactly what RollbackSession copies into its snapshot ring.)
struct EconState {
    World      ledger;
    Market     market;
    QuestState quests;
};

// --- CmdTag / EconCommand: the net::Session Input — a tagged union of the S1-S4 operations. -----------
enum CmdTag : uint32_t { kTxnCmd = 0, kCraftCmd = 1, kTradeCmd = 2, kTickCmd = 3, kNopCmd = 4 };
struct EconCommand {
    uint32_t   tag;         // CmdTag
    Command    txn{};       // kTxnCmd   -> ApplyCommand (S1 Add/Remove/Transfer)
    CraftOrder craft{};     // kCraftCmd -> DrainCraftQueue of one order (S2)
    TradeOrder trade{};     // kTradeCmd -> ExecuteTrade (S4)
    // kTickCmd -> EconTick(ledger, rules);  kNopCmd -> no-op.
};

// --- operator==(EconCommand): LOAD-BEARING for rollback. ConfirmRemote uses `appliedRemote[at] != real`
// to detect a misprediction, so this must compare the tag PLUS every field relevant to that tag (a full
// field equality — a different tag, or any differing field of the active tag, must compare unequal so a
// genuine mispredict fires). Tags compared first; only the active tag's payload is compared (the inactive
// payloads are default-constructed identically, but we compare per-tag to keep equality semantics exact).
inline bool operator==(const EconCommand& a, const EconCommand& b) {
    if (a.tag != b.tag) return false;
    switch (a.tag) {
        case kTxnCmd:
            return a.txn.kind == b.txn.kind && a.txn.src == b.txn.src && a.txn.dst == b.txn.dst &&
                   a.txn.item == b.txn.item && a.txn.amount == b.txn.amount;
        case kCraftCmd:
            return a.craft.recipeId == b.craft.recipeId && a.craft.entity == b.craft.entity &&
                   a.craft.count == b.craft.count;
        case kTradeCmd:
            return a.trade.buyer == b.trade.buyer && a.trade.seller == b.trade.seller &&
                   a.trade.item == b.trade.item && a.trade.qty == b.trade.qty;
        case kTickCmd:
        case kNopCmd:
            return true;  // no payload — equal once the tags match
        default:
            return true;  // unknown tag: tag-equal is enough (both are no-ops)
    }
}
inline bool operator!=(const EconCommand& a, const EconCommand& b) { return !(a == b); }

// --- ApplyEconCommand: apply ONE command to the state, dispatching on tag, using the STATIC config
// (rules/recipes). Reuses S1-S4 VERBATIM — every branch is the existing deterministic integer primitive.
inline void ApplyEconCommand(EconState& s, const EconCommand& c,
                             const EconRules& rules, const RecipeSet& recipes) {
    switch (c.tag) {
        case kTxnCmd:   ApplyCommand(s.ledger, c.txn); break;                          // S1
        case kCraftCmd: DrainCraftQueue(s.ledger, recipes, { c.craft }); break;        // S2 (one order)
        case kTradeCmd: ExecuteTrade(s.ledger, s.market, c.trade); break;              // S4
        case kTickCmd:  EconTick(s.ledger, rules); break;                              // S3
        case kNopCmd:   break;                                                         // no-op
        default:        break;                                                         // unknown -> no-op
    }
}

// --- DigestEconState: the combined snapshot digest = DigestState(ledger, market) folded with the quest
// status bytes, in FIXED order (the DigestState/DigestWorld precedent). Continues the SAME FNV-1a-64:
// seed with DigestState, then fold the quest status bytes. Bit-identical run-to-run AND platform-to-
// platform (fixed byte layout + iteration order; the status is a flat uint8 vector).
inline uint64_t DigestEconState(const EconState& s) {
    uint64_t h = DigestState(s.ledger, s.market);  // ledger + market digest (fixed-order FNV-1a-64)
    const std::uint64_t qh = net::DigestBytes(s.quests.status.data(), s.quests.status.size());
    // Fold the quest digest's 8 bytes into the running hash (fixed order), so the combined digest is a
    // single deterministic fold of (state || quests).
    const auto* qp = reinterpret_cast<const unsigned char*>(&qh);
    for (std::size_t i = 0; i < sizeof qh; ++i) { h ^= static_cast<uint64_t>(qp[i]); h *= 1099511628211ull; }
    return h;
}

// --- MakeShowcaseQuests: a FIXED CHAINED quest graph over the 4x4 showcase world (integer literals).
// FIXED forever (the S5 golden pins the chain's completion). A 3-link chain on entity 0:
//   obj0:                entity0 has >= 1  item2  (craft an ingot — r0)
//   obj1 (prereq obj0):  entity0 has >= 1  item3  (craft a tool  — r1, needs 3x item2)
//   obj2 (prereq obj1):  entity0 has >= 5  item3  (stockpile tools)
// The chain completes across SUCCESSIVE ticks: obj0 activates+completes once item2 accrues, obj1 then
// activates (next tick) and completes once item3 accrues, obj2 then activates and completes at >=5 item3.
inline QuestGraph MakeShowcaseQuests() {
    QuestGraph g;
    g.objectives.push_back(Objective{ /*entity=*/0, /*item=*/2, /*threshold=*/1, /*prereq=*/-1 });
    g.objectives.push_back(Objective{ /*entity=*/0, /*item=*/3, /*threshold=*/1, /*prereq=*/ 0 });
    g.objectives.push_back(Objective{ /*entity=*/0, /*item=*/3, /*threshold=*/5, /*prereq=*/ 1 });
    return g;
}

// --- MakeShowcaseState: bundle the mutable showcase state (ledger + market + quests). The ledger is the
// coin-seeded 4x4 showcase world; the market is the showcase market; quests start at MakeQuestState of
// the showcase graph. FIXED forever. (entityCount/itemCount must match the showcase world: 4x4.) -------
inline EconState MakeShowcaseState() {
    EconState s;
    s.ledger = MakeShowcaseWorld(4, 4);
    SeedShowcaseCoin(s.ledger);                       // +500 coin (item0) per entity so trades can pay
    s.market = MakeShowcaseMarket(s.ledger);
    s.quests = MakeQuestState(MakeShowcaseQuests());
    return s;
}

// --- MakeShowcaseCommandStream: a FIXED mixed command stream (txns / crafts / trades / ticks) that drives
// the economy AND completes the quest chain. FIXED forever (the S5 golden pins the lockstep+rollback over
// it). Entity 0 must accumulate item2 (>=1, then >=3 for the tool craft) then item3 (>=5) for the chain
// to finish. Recall the seed: stock(e,t) = 10 + 7e + 3t, so entity0 starts item0=510 (coin-seeded),
// item1=13, item2=16, item3=19. r0 = {2*item0 + 1*item1} -> {1*item2}; r1 = {3*item2} -> {1*item3}.
// We craft a batch of item2 (obj0), then craft item3 tools (obj1 + obj2 at >=5), interleaving txns/
// trades/ticks so the stream exercises EVERY command tag. The chain completes across the run's ticks.
inline std::vector<EconCommand> MakeShowcaseCommandStream() {
    std::vector<EconCommand> s;
    auto craft = [](uint32_t recipeId, uint32_t entity, uint32_t count) {
        EconCommand c{}; c.tag = kCraftCmd; c.craft = CraftOrder{ recipeId, entity, count }; return c;
    };
    auto txn = [](Command cmd) { EconCommand c{}; c.tag = kTxnCmd;  c.txn   = cmd; return c; };
    auto trade = [](TradeOrder o){ EconCommand c{}; c.tag = kTradeCmd; c.trade = o;   return c; };
    auto tick = []() { EconCommand c{}; c.tag = kTickCmd; return c; };
    auto nop  = []() { EconCommand c{}; c.tag = kNopCmd;  return c; };

    // tick 0:  craft r0 x6 on entity0 (ore+fuel -> ingot): item2 16 -> 22; obj0 (>=1 item2) will complete.
    s.push_back(craft(0, 0, 6));
    // tick 1:  a ledger txn (mint 4 item1 onto entity0) — exercises kTxnCmd; obj0 already complete, obj1
    //          activates this tick (prereq0 complete) but item3=19 already >= 1 so obj1 completes too.
    s.push_back(txn(Command{ kAdd, 0, 0, 1, 4 }));
    // tick 2:  craft r1 x6 on entity0 (3*item2 -> 1*item3): item2 22 -> 4 (6 crafts), item3 19 -> 25;
    //          obj2 (>=5 item3) activates (prereq1 complete) — 25 >= 5 so it completes -> CHAIN DONE.
    s.push_back(craft(1, 0, 6));
    // tick 3:  a trade (e0 buys 3x item2 from e1 @12 = 36 coin) — exercises kTradeCmd.
    s.push_back(trade(TradeOrder{ 0, 1, 2, 3 }));
    // tick 4:  an economy tick — exercises kTickCmd (producers/consumers flow one step).
    s.push_back(tick());
    // tick 5:  a transfer txn (move 2 item0 from e0 -> e3).
    s.push_back(txn(Command{ kTransfer, 0, 3, 0, 2 }));
    // tick 6:  a no-op — exercises kNopCmd (the quest advance still runs, idempotent here).
    s.push_back(nop());
    // tick 7:  a second economy tick (the economy keeps evolving; quests already all complete).
    s.push_back(tick());
    return s;
}

}  // namespace hf::econ
