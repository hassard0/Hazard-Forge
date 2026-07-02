// Slice HR1 — DETERMINISTIC STRAND/HAIR SIMULATION CORE (engine/sim/hair.h): Q16.16 PBD rods —
// strands as 1D constraint chains with bending stiffness, root-pinned, gravity-driven, with
// strand<->strand collision — bit-identical + lockstep-replayable (Track-S S2). Pure CPU (header-only,
// no device, no backend symbols). Namespace hf::sim::hair. The ~80% mold is engine/sim/cloth.h
// (READ-ONLY reuse: ClothParticle/IntegrateParticle/SolveDistanceConstraint/the CL7 pair grid).
//
// What this test PINS:
//   * InitStrands layout (flat strand-major, exact integer multiples, pinCount roots pinned).
//   * BuildHairConstraints counts + hand-enumerated stretch/bend chains (rest2 == 2*restLen).
//   * The exclusion set: same-strand verts <= 2 apart excluded; >= 3 apart + cross-strand collide.
//   * SolveBendConstraint EXACT integer scheme: kBend==kOne == SolveDistanceConstraint BIT-EXACT;
//     kBend==kOne/2 hand-checked fractional projection; kBend==0 exact no-op.
//   * DETERMINISM: two runs byte-identical; net::DigestBytes-over-pos + HairDigest PINNED (must be
//     identical under MSVC and clang — the cross-compiler proof).
//   * THE PHYSICS: (i) a single strand under gravity settles VERTICAL (tip pinned, analytic band);
//     (ii) BENDING: a horizontal-rooted stiff strand droops LESS than a limp one (real numbers, the
//     known PBD bending-as-distance softness reported honestly); (iii) COLLISION: two adjacent strands
//     pushed together separate to >= 2r - slack (the CL7 1-LSB honesty; negative control included).
//   * Identity-at-zero: kBend==0 + collision-off == the hand-composed pure stretch chain (bit-check).
//   * STRETCH integrity: the settled max segment-length error PINNED (the honest PBD residual).
//   * LOCKSTEP: replica == authority BIT-EXACT from inputs alone; rollback corrects a real
//     misprediction to authority BIT-EXACT (negative control: the mispredicted run differed).
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/hair.h"

#include "net/session.h"   // net::DigestBytes — the spec'd pos-digest currency

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"     // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace hair  = hf::sim::hair;
namespace cloth = hf::sim::cloth;
using hair::fx;
using hair::kOne;
using hair::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << kFrac); }

// net::DigestBytes over the packed pos array (x,y,z int32 words per vert, index order) — the spec'd
// determinism pin. Packed into an explicit int32 buffer so the digest is layout-independent.
static uint64_t PosDigest(const std::vector<hair::HairVert>& verts) {
    std::vector<int32_t> pos;
    pos.reserve(verts.size() * 3);
    for (const hair::HairVert& v : verts) {
        pos.push_back(v.pos.x); pos.push_back(v.pos.y); pos.push_back(v.pos.z);
    }
    return net::DigestBytes(pos.data(), pos.size() * sizeof(int32_t));
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= InitStrands: flat strand-major layout, exact multiples, pinned roots ==========
    {
        hair::HairStrands hs; hs.S = 3; hs.M = 5; hs.restLen = kOne / 2;
        const hair::FxVec3 origin{0, FromInt(8), 0};
        const hair::FxVec3 rootStep{kOne, 0, 0};             // roots on a bar along +X, 1.0 apart
        const hair::FxVec3 growDir{0, -kOne, 0};             // strands grow straight DOWN
        std::vector<hair::HairVert> vs = hair::InitStrands(hs, origin, rootStep, growDir, 1);

        check(vs.size() == 15, "InitStrands: size == S*M");
        // Strand 1 vert 3: root (1, 8, 0) + 3*0.5 down -> (1, 6.5, 0). Exact integer multiples.
        const hair::HairVert& v13 = vs[(size_t)hair::VertIndex(hs, 1, 3)];
        check(v13.pos.x == kOne && v13.pos.y == FromInt(8) - (3 * (kOne / 2)) && v13.pos.z == 0,
              "InitStrands: strand 1 vert 3 at root + 3*restLen*growDir (exact)");
        // Vert 0 of each strand pinned; the rest dynamic; prev==pos, vel==0.
        int pinned = 0; bool restOk = true;
        for (int s = 0; s < hs.S; ++s)
            for (int i = 0; i < hs.M; ++i) {
                const hair::HairVert& v = vs[(size_t)hair::VertIndex(hs, s, i)];
                if (v.flags & hair::kFlagPinned) { ++pinned; if (i != 0 || v.invMass != 0) restOk = false; }
                else if (v.invMass != kOne) restOk = false;
                if (std::memcmp(&v.prev, &v.pos, sizeof(hair::FxVec3)) != 0) restOk = false;
                if (v.vel.x != 0 || v.vel.y != 0 || v.vel.z != 0) restOk = false;
            }
        check(pinned == 3 && restOk, "InitStrands: exactly the 3 roots pinned; at rest; dynamic kOne");
        // pinCount=2 pins verts 0 AND 1 (the direction-clamped root).
        std::vector<hair::HairVert> vs2 = hair::InitStrands(hs, origin, rootStep, growDir, 2);
        check((vs2[0].flags & hair::kFlagPinned) && (vs2[1].flags & hair::kFlagPinned) &&
              !(vs2[2].flags & hair::kFlagPinned),
              "InitStrands: pinCount=2 pins verts 0+1 (direction-clamped root)");
    }

    // ================= BuildHairConstraints: hand-enumerated stretch + bend chains ====================
    {
        hair::HairStrands hs; hs.S = 2; hs.M = 4; hs.restLen = kOne / 2;
        hair::HairConstraints hc = hair::BuildHairConstraints(hs);
        // Per strand: M-1 = 3 stretch, M-2 = 2 bend.
        check(hc.stretch.size() == 6, "BuildHairConstraints: S*(M-1) stretch edges");
        check(hc.bend.size() == 4, "BuildHairConstraints: S*(M-2) bend edges");
        // Strand 0: stretch (0,1)(1,2)(2,3); bend (0,2)(1,3). Strand 1 offset by M=4.
        check(hc.stretch[0].i == 0 && hc.stretch[0].j == 1 && hc.stretch[0].restLen == kOne / 2 &&
              hc.stretch[0].kind == hair::kConstraintStructural,
              "BuildHairConstraints: stretch (0,1) restLen kind=structural");
        check(hc.stretch[3].i == 4 && hc.stretch[3].j == 5,
              "BuildHairConstraints: strand 1 stretch offset by M");
        check(hc.bend[0].i == 0 && hc.bend[0].j == 2 && hc.bend[0].restLen == kOne &&
              hc.bend[0].kind == hair::kConstraintBend,
              "BuildHairConstraints: bend (0,2) rest2 == 2*restLen kind=bend");
        check(hc.bend[3].i == 5 && hc.bend[3].j == 7, "BuildHairConstraints: strand 1 bend offset by M");
        // Determinism: a second build is byte-identical.
        hair::HairConstraints hc2 = hair::BuildHairConstraints(hs);
        check(hc.stretch.size() == hc2.stretch.size() && hc.bend.size() == hc2.bend.size() &&
              std::memcmp(hc.stretch.data(), hc2.stretch.data(),
                          hc.stretch.size() * sizeof(hair::Constraint)) == 0 &&
              std::memcmp(hc.bend.data(), hc2.bend.data(),
                          hc.bend.size() * sizeof(hair::Constraint)) == 0,
              "BuildHairConstraints: two builds byte-identical");
        // M=2: stretch only, no bend. M=1: neither (degenerate no-op).
        hair::HairStrands two; two.S = 1; two.M = 2; two.restLen = kOne;
        hair::HairConstraints hcTwo = hair::BuildHairConstraints(two);
        check(hcTwo.stretch.size() == 1 && hcTwo.bend.empty(),
              "BuildHairConstraints: M=2 -> 1 stretch, 0 bend");
        hair::HairStrands one; one.S = 1; one.M = 1; one.restLen = kOne;
        hair::HairConstraints hcOne = hair::BuildHairConstraints(one);
        check(hcOne.stretch.empty() && hcOne.bend.empty(),
              "BuildHairConstraints: M=1 -> no constraints (degenerate)");
    }

    // ================= The collision EXCLUSION set: <=2 apart excluded; >=3 + cross-strand not ========
    {
        hair::HairStrands hs; hs.S = 2; hs.M = 6; hs.restLen = kOne / 2;
        hair::HairConstraints hc = hair::BuildHairConstraints(hs);
        hair::ClothAdjacency excl = hair::BuildHairExclusion((size_t)(hs.S * hs.M), hc);
        // Same strand: (0,1) stretch-connected, (0,2) bend-connected -> excluded; (0,3) NOT.
        check(cloth::IsConstraintConnected(excl, 0, 1), "exclusion: same-strand +/-1 excluded (stretch)");
        check(cloth::IsConstraintConnected(excl, 0, 2), "exclusion: same-strand +/-2 excluded (bend)");
        check(!cloth::IsConstraintConnected(excl, 0, 3), "exclusion: same-strand >=3 apart COLLIDES");
        // Cross-strand: never constraint-connected (vert 0 vs strand 1's vert 6).
        check(!cloth::IsConstraintConnected(excl, 0, 6), "exclusion: cross-strand pairs COLLIDE");
        check(!cloth::IsConstraintConnected(excl, 5, 6),
              "exclusion: strand-boundary neighbours (5,6) are NOT chained across strands");
    }

    // ================= SolveBendConstraint: the exact integer fractional projection ===================
    {
        // kBend == kOne -> BIT-EXACT to cloth::SolveDistanceConstraint on the same input.
        {
            std::vector<hair::HairVert> a(2), b(2);
            a[0].pos = {0, 0, 0}; a[0].prev = a[0].pos; a[0].invMass = kOne; a[0].flags = 0;
            a[1].pos = {FromInt(3), FromInt(1), 0}; a[1].prev = a[1].pos; a[1].invMass = kOne; a[1].flags = 0;
            b = a;
            const hair::Constraint c{0u, 1u, FromInt(2), hair::kConstraintBend};
            hair::SolveBendConstraint(a, c, kOne);
            cloth::SolveDistanceConstraint(b, c);
            check(std::memcmp(a.data(), b.data(), a.size() * sizeof(hair::HairVert)) == 0,
                  "SolveBendConstraint: kBend==kOne == SolveDistanceConstraint BIT-EXACT");
        }
        // kBend == kOne/2 hand-check: pinned i at 0, free j at x=3, rest2=2 -> pen=1, penS=0.5,
        // n=+x, wj=kOne -> j moves to exactly x = 3 - 0.5 = 2.5 (the fractional projection).
        {
            std::vector<hair::HairVert> ps(2);
            ps[0].pos = {0, 0, 0}; ps[0].prev = ps[0].pos; ps[0].invMass = 0; ps[0].flags = hair::kFlagPinned;
            ps[1].pos = {FromInt(3), 0, 0}; ps[1].prev = ps[1].pos; ps[1].invMass = kOne; ps[1].flags = 0;
            const hair::Constraint c{0u, 1u, FromInt(2), hair::kConstraintBend};
            hair::SolveBendConstraint(ps, c, kOne / 2);
            check(ps[0].pos.x == 0 && ps[0].pos.y == 0, "SolveBendConstraint: pinned endpoint unchanged");
            check(ps[1].pos.x == FromInt(2) + kOne / 2 && ps[1].pos.y == 0,
                  "SolveBendConstraint: kBend=0.5 moves the free endpoint HALF the pen (x=2.5 exact)");
        }
        // kBend == 0 -> exact no-op (byte-untouched).
        {
            std::vector<hair::HairVert> ps(2);
            ps[0].pos = {0, 0, 0}; ps[0].prev = ps[0].pos; ps[0].invMass = kOne; ps[0].flags = 0;
            ps[1].pos = {FromInt(3), 0, 0}; ps[1].prev = ps[1].pos; ps[1].invMass = kOne; ps[1].flags = 0;
            std::vector<hair::HairVert> before = ps;
            hair::SolveBendConstraint(ps, hair::Constraint{0u, 1u, FromInt(2), hair::kConstraintBend}, 0);
            check(std::memcmp(ps.data(), before.data(), ps.size() * sizeof(hair::HairVert)) == 0,
                  "SolveBendConstraint: kBend==0 is an EXACT no-op");
        }
        // Out-of-range endpoints: deterministic no-op (bounds-checked).
        {
            std::vector<hair::HairVert> ps(1);
            ps[0].pos = {0, 0, 0}; ps[0].prev = ps[0].pos; ps[0].invMass = kOne; ps[0].flags = 0;
            std::vector<hair::HairVert> before = ps;
            hair::SolveBendConstraint(ps, hair::Constraint{0u, 7u, FromInt(2), hair::kConstraintBend}, kOne);
            check(std::memcmp(ps.data(), before.data(), sizeof(hair::HairVert)) == 0,
                  "SolveBendConstraint: out-of-range endpoint is a no-op");
        }
    }

    // ================= DETERMINISM: two runs byte-identical + PINNED digests ==========================
    // A mixed-stiffness 4-strand set swinging down from horizontal (a dynamic, non-trivial state).
    {
        hair::HairStrands hs; hs.S = 4; hs.M = 10; hs.restLen = kOne / 2;
        hair::HairConstraints hc = hair::BuildHairConstraints(hs);
        hair::ClothAdjacency excl = hair::BuildHairExclusion((size_t)(hs.S * hs.M), hc);
        const std::vector<hair::HairVert> init =
            hair::InitStrands(hs, hair::FxVec3{0, FromInt(8), 0}, hair::FxVec3{0, 0, kOne},
                              hair::FxVec3{kOne, 0, 0}, 1);       // horizontal, roots along Z
        const std::vector<fx> kb{0, kOne / 4, kOne / 2, kOne};
        hair::HairParams p;
        p.gravity = {0, FromInt(-10), 0};
        p.dt = kOne / 60; p.groundY = FromInt(-100); p.iters = 6;
        p.radius = kOne / 8; p.damp = kOne - kOne / 32;           // 0.96875 (exact Q16.16)

        std::vector<hair::HairVert> a = init, b = init;
        hair::StepHairSteps(hs, a, hc, excl, kb, p, 120);
        hair::StepHairSteps(hs, b, hc, excl, kb, p, 120);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(hair::HairVert)) == 0,
              "determinism: two runs byte-identical");

        const uint64_t dp = PosDigest(a);
        const uint64_t dh = hair::HairDigest(a);
        std::printf("HR1 pin: determinism PosDigest = 0x%016llx  HairDigest = 0x%016llx\n",
                    (unsigned long long)dp, (unsigned long long)dh);
        check(dp == 0x2cbfb2a278636035ull, "HR1 pin: net::DigestBytes-over-pos == the pinned value");
        check(dh == 0x8b5783ddcc27b9beull, "HR1 pin: HairDigest == the pinned value");

        // Coherence: every root held (pinned, byte-untouched); every free vert moved.
        bool rootsHeld = true; int moved = 0;
        for (int s = 0; s < hs.S; ++s) {
            const size_t r = (size_t)hair::VertIndex(hs, s, 0);
            if (std::memcmp(&a[r], &init[r], sizeof(hair::HairVert)) != 0) rootsHeld = false;
        }
        for (size_t i = 0; i < a.size(); ++i)
            if (!(a[i].flags & hair::kFlagPinned) &&
                std::memcmp(&a[i].pos, &init[i].pos, sizeof(hair::FxVec3)) != 0) ++moved;
        check(rootsHeld, "coherence: every pinned root byte-untouched");
        check(moved == (int)a.size() - hs.S, "coherence: every free vert moved (the strands fell)");
    }

    // ================= THE PHYSICS (i): a single strand settles VERTICAL under gravity ================
    // One limp strand (kBend=0, no collision) released HORIZONTAL swings down and settles hanging
    // straight below the root: tip ~= root + (0, -(M-1)*restLen, 0). Analytic band + exact pin.
    {
        hair::HairStrands hs; hs.S = 1; hs.M = 8; hs.restLen = kOne / 2;   // length 3.5
        hair::HairConstraints hc = hair::BuildHairConstraints(hs);
        hair::ClothAdjacency excl = hair::BuildHairExclusion((size_t)hs.M, hc);
        std::vector<hair::HairVert> vs =
            hair::InitStrands(hs, hair::FxVec3{0, FromInt(8), 0}, hair::FxVec3{0, 0, 0},
                              hair::FxVec3{kOne, 0, 0}, 1);       // released horizontal (+X)
        const std::vector<fx> kb{0};
        hair::HairParams p;
        p.gravity = {0, FromInt(-10), 0};
        p.dt = kOne / 60; p.groundY = FromInt(-100); p.iters = 8;
        p.radius = 0; p.damp = kOne - kOne / 32;
        hair::StepHairSteps(hs, vs, hc, excl, kb, p, 900);        // 15 s: settle

        const hair::HairVert& tip = vs[(size_t)(hs.M - 1)];
        const fx expectY = FromInt(8) - (fx)(7 * (kOne / 2));     // 8 - 3.5 = 4.5
        std::printf("HR1 physics(i): vertical-hang tip = (%d, %d, %d) LSB (analytic (0, %d, 0)); "
                    "maxStretchErr = %d LSB\n",
                    tip.pos.x, tip.pos.y, tip.pos.z, expectY, hair::MaxStretchError(vs, hc));
        // Band: the settled tip is straight below the root within 1/16 world unit laterally and the
        // chain length within 1/8 unit of (M-1)*restLen (the honest PBD stretch residual).
        check(tip.pos.x > -kOne / 16 && tip.pos.x < kOne / 16, "physics(i): tip.x ~ 0 (vertical)");
        check(tip.pos.z == 0, "physics(i): tip.z == 0 (planar scene stays planar)");
        check(tip.pos.y > expectY - kOne / 8 && tip.pos.y < expectY + kOne / 8,
              "physics(i): tip.y within the analytic band (root.y - (M-1)*restLen)");
        // Exact settled-tip pin (the golden discipline: identical MSVC + clang). HONEST: the settled
        // tip carries a 227-LSB (~0.0035 unit) residual x and sits 402 LSB (~0.006 unit) above the
        // analytic length — the deterministic PBD/truncation residual, not zero.
        check(tip.pos.x == 227 && tip.pos.y == 294510 && tip.pos.z == 0,
              "HR1 pin physics(i): settled tip == the exact pinned integers (227, 294510, 0)");
        // STRETCH integrity (d): the settled max segment error, pinned honestly.
        const fx serr = hair::MaxStretchError(vs, hc);
        check(serr <= kOne / 16, "physics(d): settled max stretch error <= 1/16 world unit");
        check(serr == 105, "HR1 pin physics(d): settled maxStretchErr == 105 LSB (~0.0016 world units)");
    }

    // ================= THE PHYSICS (ii): BENDING — stiff droops LESS from a horizontal root ===========
    // Two direction-clamped strands (verts 0+1 pinned HORIZONTAL, the standard PBD rooting — a 1-pin
    // strand pivots freely and hangs vertical regardless of kBend, the documented bending-as-distance
    // limit): kBend=0 vs kBend=kOne. The stiff tip settles HIGHER. HONEST: the fractional projection
    // saturates (effective stiffness ~ 1-(1-k)^iters and the distance-bend model is soft for long
    // chains under gravity), so even kBend=kOne droops well below horizontal — real numbers pinned.
    {
        hair::HairStrands hs; hs.S = 2; hs.M = 10; hs.restLen = kOne / 2;
        hair::HairConstraints hc = hair::BuildHairConstraints(hs);
        hair::ClothAdjacency excl = hair::BuildHairExclusion((size_t)(hs.S * hs.M), hc);
        std::vector<hair::HairVert> vs =
            hair::InitStrands(hs, hair::FxVec3{0, FromInt(8), 0}, hair::FxVec3{0, 0, FromInt(10)},
                              hair::FxVec3{kOne, 0, 0}, 2);       // horizontal roots, far apart in Z
        const std::vector<fx> kb{0, kOne};                        // strand 0 LIMP, strand 1 STIFF
        hair::HairParams p;
        p.gravity = {0, FromInt(-10), 0};
        p.dt = kOne / 60; p.groundY = FromInt(-100); p.iters = 8;
        p.radius = 0; p.damp = kOne - kOne / 32;
        hair::StepHairSteps(hs, vs, hc, excl, kb, p, 900);

        const fx rootY  = FromInt(8);
        const fx limpY  = vs[(size_t)hair::VertIndex(hs, 0, hs.M - 1)].pos.y;
        const fx stiffY = vs[(size_t)hair::VertIndex(hs, 1, hs.M - 1)].pos.y;
        std::printf("HR1 physics(ii): tipY limp(k=0) = %d, stiff(k=kOne) = %d LSB "
                    "(droop %d vs %d LSB from rootY %d)\n",
                    limpY, stiffY, rootY - limpY, rootY - stiffY, rootY);
        check(stiffY > limpY + kOne / 4,
              "physics(ii): the stiff strand's tip settles HIGHER (droops less) by > 0.25 units");
        // Exact pins (the honest droop numbers — bending-as-distance SATURATES: even kBend==kOne
        // droops 3.67 of the 4.5-unit free length vs the limp 4.01; the stiffness gap is a real but
        // modest 0.33 units, NOT rigidity).
        check(limpY == 261582 && stiffY == 283494,
              "HR1 pin physics(ii): settled tip heights == the exact pinned integers");
    }

    // ================= THE PHYSICS (iii): strand<->strand COLLISION (with/without contrast) ===========
    // Two strands hanging straight down, roots authored only r apart (closer than the 2r thickness):
    // WITHOUT collision the free verts stay ~r apart (penetrating); WITH collision they separate to
    // >= 2r - slack (the CL7 1-LSB fixed-point honesty). Both-pinned root pairs are excluded from the
    // metric (the solver correctly never moves pinned roots — MinFreePairDistance documents this).
    {
        hair::HairStrands hs; hs.S = 2; hs.M = 8; hs.restLen = kOne / 2;
        hair::HairConstraints hc = hair::BuildHairConstraints(hs);
        hair::ClothAdjacency excl = hair::BuildHairExclusion((size_t)(hs.S * hs.M), hc);
        const fx radius = kOne / 4, thickness = kOne / 2;         // r=0.25 -> 2r=0.5
        const std::vector<hair::HairVert> init =
            hair::InitStrands(hs, hair::FxVec3{0, FromInt(8), 0}, hair::FxVec3{kOne / 4, 0, 0},
                              hair::FxVec3{0, -kOne, 0}, 1);      // roots only r=0.25 apart
        const std::vector<fx> kb{0, 0};
        hair::HairParams p;
        p.gravity = {0, FromInt(-10), 0};
        p.dt = kOne / 60; p.groundY = FromInt(-100); p.iters = 6;
        p.damp = kOne - kOne / 32;
        const fx slack = kOne / 16;                               // the honest PBD-residual slack

        // WITHOUT collision (radius 0): the parallel strands stay ~0.25 apart -> penetrating pairs.
        std::vector<hair::HairVert> off = init;
        p.radius = 0;
        hair::StepHairSteps(hs, off, hc, excl, kb, p, 240);
        const int penOff = hair::CountFreePenetrating(off, excl, thickness, 0);
        const fx minOff  = hair::MinFreePairDistance(off, excl);
        std::printf("HR1 physics(iii): WITHOUT collision %d pairs < 2r (min free dist %d LSB)\n",
                    penOff, minOff);
        check(penOff > 0, "physics(iii): WITHOUT collision the strands interpenetrate (control)");
        check(minOff < thickness - slack, "physics(iii): control min distance well below 2r");

        // WITH collision: the free verts separate to >= 2r - slack.
        std::vector<hair::HairVert> on = init;
        p.radius = radius;
        hair::StepHairSteps(hs, on, hc, excl, kb, p, 240);
        const int penRaw   = hair::CountFreePenetrating(on, excl, thickness, 0);
        const int penSlack = hair::CountFreePenetrating(on, excl, thickness, slack);
        const fx minOn     = hair::MinFreePairDistance(on, excl);
        std::printf("HR1 physics(iii): WITH collision raw %d, slack(%d) %d pairs (min free dist %d LSB, "
                    "thickness %d)\n", penRaw, slack, penSlack, minOn, thickness);
        check(penSlack == 0, "physics(iii): WITH collision zero pairs below 2r - slack (separated)");
        check(minOn >= thickness - slack, "physics(iii): min free pair distance >= 2r - slack");
        // Exact pins (the CL7 honesty: settled contacts sit a few LSBs UNDER 2r from the
        // FxNormalize/fxmul truncation — 3 raw pairs at 3 LSBs under thickness, min dist 32765 vs
        // thickness 32768; the slack count is the physical claim).
        check(penRaw == 3 && minOn == 32765,
              "HR1 pin physics(iii): raw pairs == 3 (LSB-under contacts), min free dist == 32765 LSB");
    }

    // ================= Identity-at-zero: kBend=0 + collision-off == the pure stretch chain ============
    // StepHair with all-zero kBend and radius 0 must be BIT-IDENTICAL to the hand-composed pure
    // stretch chain (integrate -> iters x stretch pass -> ground clamp -> velocity re-encode) built
    // from the cloth.h primitives directly — the off-switch contract.
    {
        hair::HairStrands hs; hs.S = 3; hs.M = 9; hs.restLen = kOne / 2;
        hair::HairConstraints hc = hair::BuildHairConstraints(hs);
        hair::ClothAdjacency excl = hair::BuildHairExclusion((size_t)(hs.S * hs.M), hc);
        const std::vector<hair::HairVert> init =
            hair::InitStrands(hs, hair::FxVec3{0, FromInt(6), 0}, hair::FxVec3{0, 0, kOne},
                              hair::FxVec3{kOne, 0, 0}, 1);
        const std::vector<fx> kbZero{0, 0, 0};
        hair::HairParams p;
        p.gravity = {0, FromInt(-10), 0};
        p.dt = kOne / 60; p.groundY = FromInt(-100); p.iters = 5;
        p.radius = 0; p.damp = kOne - kOne / 32;
        const int steps = 90;

        std::vector<hair::HairVert> viaStep = init;
        hair::StepHairSteps(hs, viaStep, hc, excl, kbZero, p, steps);

        // The hand-composed pure stretch chain (the same primitives, no hair.h step function).
        std::vector<hair::HairVert> pure = init;
        for (int s = 0; s < steps; ++s) {
            for (size_t i = 0; i < pure.size(); ++i)
                cloth::IntegrateParticle(pure[i], p.gravity, p.groundY, p.dt);
            for (int it = 0; it < p.iters; ++it)
                for (size_t e = 0; e < hc.stretch.size(); ++e)
                    cloth::SolveDistanceConstraint(pure, hc.stretch[e]);
            cloth::CollidePlane(pure, p.groundY);
            for (size_t i = 0; i < pure.size(); ++i) {
                hair::HairVert& v = pure[i];
                if (v.flags & hair::kFlagPinned) continue;
                v.vel.x = hair::fxmul(hair::fxdiv(v.pos.x - v.prev.x, p.dt), p.damp);
                v.vel.y = hair::fxmul(hair::fxdiv(v.pos.y - v.prev.y, p.dt), p.damp);
                v.vel.z = hair::fxmul(hair::fxdiv(v.pos.z - v.prev.z, p.dt), p.damp);
            }
        }
        check(viaStep.size() == pure.size() &&
              std::memcmp(viaStep.data(), pure.data(), pure.size() * sizeof(hair::HairVert)) == 0,
              "identity-at-zero: kBend=0 + radius=0 StepHair == the pure stretch chain BIT-EXACT");
    }

    // ================= LOCKSTEP: replica == authority; rollback corrects a real divergence ============
    {
        hair::HairStrands hs; hs.S = 3; hs.M = 8; hs.restLen = kOne / 2;
        hair::HairConstraints hc = hair::BuildHairConstraints(hs);
        hair::ClothAdjacency excl = hair::BuildHairExclusion((size_t)(hs.S * hs.M), hc);
        const std::vector<hair::HairVert> init =
            hair::InitStrands(hs, hair::FxVec3{0, FromInt(8), 0}, hair::FxVec3{kOne, 0, 0},
                              hair::FxVec3{0, -kOne, 0}, 1);      // hanging strands, roots 1.0 apart
        const std::vector<fx> kb{0, kOne / 2, kOne};
        hair::HairParams p;
        p.gravity = {0, FromInt(-10), 0};
        p.dt = kOne / 60; p.groundY = FromInt(-100); p.iters = 6;
        p.radius = kOne / 8; p.damp = kOne - kOne / 32;
        const int ticks = 24, mispredictTick = 8;

        // The authoritative ROOT-DRAG stream (the netcode input): two drags on two strands.
        const std::vector<hair::HairCommand> authStream{
            hair::HairCommand{3,  hair::kCmdRootMove, 1u, hair::FxVec3{kOne / 2, 0, 0}},
            hair::HairCommand{8,  hair::kCmdRootMove, 2u, hair::FxVec3{0, 0, kOne / 4}},
            hair::HairCommand{14, hair::kCmdRootMove, 0u, hair::FxVec3{-(kOne / 2), 0, 0}},
        };
        // The MISPREDICTED stream: auth + a WRONG large drag at mispredictTick (a real divergence).
        std::vector<hair::HairCommand> mispredictStream = authStream;
        mispredictStream.push_back(hair::HairCommand{(uint32_t)mispredictTick, hair::kCmdRootMove, 1u,
                                                     hair::FxVec3{FromInt(2), 0, 0}});

        const std::vector<hair::HairVert> authority =
            hair::RunHairLockstep(hs, init, hc, excl, kb, p, authStream, ticks);
        const std::vector<hair::HairVert> replica =
            hair::RunHairLockstep(hs, init, hc, excl, kb, p, authStream, ticks);
        check(authority.size() == replica.size() &&
              std::memcmp(authority.data(), replica.data(),
                          authority.size() * sizeof(hair::HairVert)) == 0,
              "lockstep: replica == authority BIT-EXACT (inputs-only re-sim)");

        const std::vector<hair::HairVert> rolledBack =
            hair::RunHairRollback(hs, init, hc, excl, kb, p, authStream, mispredictStream,
                                  ticks, mispredictTick);
        check(rolledBack.size() == authority.size() &&
              std::memcmp(rolledBack.data(), authority.data(),
                          authority.size() * sizeof(hair::HairVert)) == 0,
              "rollback: corrected to authority BIT-EXACT (positive control)");

        const std::vector<hair::HairVert> mispredicted =
            hair::RunHairLockstep(hs, init, hc, excl, kb, p, mispredictStream, ticks);
        check(mispredicted.size() == authority.size() &&
              std::memcmp(mispredicted.data(), authority.data(),
                          authority.size() * sizeof(hair::HairVert)) != 0,
              "rollback: the mispredicted run DIFFERS from authority (the divergence was real)");

        // The command actually moved the roots (a drag is not a no-op) + OOB strand is a no-op.
        check(authority[(size_t)hair::VertIndex(hs, 1, 0)].pos.x == FromInt(1) + kOne / 2,
              "lockstep: the root drag displaced strand 1's root by exactly the command arg");
        std::vector<hair::HairVert> oob = init;
        hair::ApplyHairCommand(hs, oob, hair::HairCommand{0, hair::kCmdRootMove, 99u,
                                                          hair::FxVec3{kOne, 0, 0}});
        check(std::memcmp(oob.data(), init.data(), init.size() * sizeof(hair::HairVert)) == 0,
              "lockstep: an out-of-range strand command is a deterministic no-op");

        const uint64_t d = hair::HairDigest(authority);
        std::printf("HR1 pin: lockstep authority HairDigest = 0x%016llx\n", (unsigned long long)d);
        check(d == 0xa30438a24459f82full, "HR1 pin: lockstep authority digest == the pinned value");
    }

    if (g_fail == 0) std::printf("hair_test: ALL PASS\n");
    else std::printf("hair_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
