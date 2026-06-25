# Slice SEQ-S2 — Easing-curve LUT + multi-track Sequence (Issue #25)

S1 shipped the Q16.16 scalar keyframe track + integer Step/Linear interpolation (digest
`0xd314f17ebe3d480b`). S2 adds the two things a real timeline needs: **eased interpolation** (smooth
ease-in/out curves, the float `acos`/`sin` that breaks UE5 Sequencer rebuilt as a host-baked INTEGER LUT —
zero runtime transcendentals, the `ik.h` `BuildSinLut` / `FxSinLut` precedent) and a **multi-track
`Sequence`** (a timeline is many channels sampled at one time → a value-bus). Pure-CPU INTEGER, append-only
to `engine/seq/seq.h` — S1's types/functions and its pinned digest stay UNTOUCHED + golden-invariant.

## Append to engine/seq/seq.h (below S1, in hf::seq)

### 1. Easing enum — APPEND, do NOT renumber S1's `Step=0, Linear=1`
```cpp
enum class Easing : uint32_t {
    Step = 0, Linear = 1,           // S1 — frozen
    EaseInOutSine = 2,              // smooth S-curve (the showcase ease)
    EaseInQuad    = 3,              // accelerate from rest
    EaseOutQuad   = 4,              // decelerate to rest
};
```

### 2. Host-baked integer easing LUTs (the `ik.h` mold — zero runtime transcendentals)
A fixed-resolution table over the unit interval, evaluated by the same scaled-index + fractional-lerp
read as `FxSinLut`:
```cpp
constexpr int kEaseBins = 256;                       // table has kEaseBins+1 entries (incl. the t=1 endpoint)
constexpr int kEaseShift = 8;                         // kEaseBins == 1<<kEaseShift
// Each table maps t01 in [0,kOne] -> eased value in [0,kOne]; entry[i] = ease(i/kEaseBins) in Q16.16.
```
- The QUADRATIC tables are EXACT integer (no transcendental): build them at static-init from `fxmul`:
  `EaseInQuad(t)=fxmul(t,t)`; `EaseOutQuad(t)=kOne - fxmul(kOne-t, kOne-t)`. Provide
  `BuildQuadInLut()`/`BuildQuadOutLut()` returning `std::vector<fx>` of size `kEaseBins+1` with
  `entry[i] = ...` for `t = fxdiv((fx)i, (fx)kEaseBins)` ... **BUT** `i/kEaseBins` in Q16.16 is just
  `(fx)((int64)i * kOne / kEaseBins)` — compute it directly, do NOT call fxdiv with a non-Q16.16 numerator.
  (For `i==kEaseBins`, `t==kOne` exactly.)
- The SINE table `EaseInOutSine(t) = (1 - cos(pi*t))/2` needs a transcendental at BAKE time. Two options —
  pick the one that keeps the header self-contained with NO `<cmath>`:
  **PREFERRED:** ship the 257 entries as a COMMITTED `constexpr` integer literal array (generate it once with
  a throwaway program, paste the values). This keeps seq.h `<cmath>`-free (HARD constraint) and is the
  cleanest determinism story (the bytes ARE the spec). Generate via the same `(1-cos(pi*i/256))/2*65536`
  rounded-to-nearest, then PIN a digest of the table so any regeneration is caught.
  (Do NOT add `<cmath>` to seq.h to compute it at runtime — that reintroduces a float dependency on the
  hot header. Bake-and-paste.)
- Accessors return references to function-local `static const std::vector<fx>` (built once, the quad ones
  via the builders, the sine one from the literal array):
  `const std::vector<fx>& SineEaseTable();` / `QuadInTable();` / `QuadOutTable();`

### 3. `Ease(Easing, fx t01) -> fx` — the LUT read (FxSinLut mold)
```cpp
inline fx Ease(Easing e, fx t01) {
    // t01 assumed in [0,kOne]; clamp defensively.
    if (t01 <= 0)   return 0;
    if (t01 >= kOne) return kOne;
    if (e == Easing::Step)   return 0;        // handled in SampleScalar; Ease() of Step returns the lower key weight
    if (e == Easing::Linear) return t01;      // identity
    const std::vector<fx>& tbl = (e==EaseInOutSine) ? SineEaseTable()
                               : (e==EaseInQuad)    ? QuadInTable() : QuadOutTable();
    // scaled = t01 * kEaseBins  (Q16.16 index.fraction); i = scaled>>16 in [0,kEaseBins)
    int64_t scaled = (int64_t)t01 * kEaseBins;          // Q16.16 * int = Q16.16 scaled
    int      i     = (int)(scaled >> 16);
    fx       frac  = (fx)(scaled & 0xFFFF);             // Q16.16 fraction within the bin
    return tbl[i] + fxmul(frac, tbl[i+1] - tbl[i]);     // lerp between adjacent table entries
}
```
(Note: `Ease(Linear,t)==t` and `Ease(*,0)==0`, `Ease(*,kOne)==kOne` EXACTLY — assert these. Endpoints are
exact because the tables pin `entry[0]=0`, `entry[kEaseBins]=kOne`.)

### 4. Wire `Ease` into `SampleScalar` — the ONE-LINE S1 insertion point
S1 left the structure so this is a single change: where S1 returned
`values[k] + fxmul(t01, values[k+1]-values[k])`, now compute `fx w = Ease(tr.easing, t01);` and return
`values[k] + fxmul(w, values[k+1]-values[k])`. **Linear must produce the IDENTICAL result as S1**
(`Ease(Linear,t01)==t01`) — so S1's pinned digest `0xd314f17ebe3d480b` stays UNCHANGED (the showcase track
is `Linear`). VERIFY this: re-run S1's assertion 1, it MUST still equal `0xd314f17ebe3d480b`. (Step still
short-circuits to `values[k]` before reaching `Ease`, exactly as S1.)

### 5. Multi-track `Sequence` — a timeline is N channels sampled at one time
```cpp
struct Sequence { std::vector<ScalarTrack> tracks; };          // channel c = tracks[c]
inline std::vector<fx> SampleSequence(const Sequence& seq, fx t) {   // the value-bus at time t
    std::vector<fx> bus; bus.reserve(seq.tracks.size());
    for (const auto& tr : seq.tracks) bus.push_back(SampleScalar(tr, t));
    return bus;
}
inline std::vector<fx> SampleSequenceSweep(const Sequence& seq, fx dt, uint32_t n) {
    // n ticks; each tick appends the whole bus -> contiguous (n * tracks) fx, byte-stable digest input.
    std::vector<fx> out; out.reserve((size_t)n * seq.tracks.size());
    for (uint32_t i=0;i<n;++i){ fx t=(fx)((int64_t)i*dt); for(const auto& tr:seq.tracks) out.push_back(SampleScalar(tr,t)); }
    return out;
}
```

### 6. Fixtures (FIXED forever — the goldens pin them)
- `Sequence MakeShowcaseSequence()` — 3 channels exercising the new easings: channel 0 = the S1
  `MakeShowcaseTrack()` VERBATIM (Linear — anchors S1 invariance), channel 1 = same `times` with
  `easing=EaseInOutSine` and `values={0, kOne, 0, kOne}`, channel 2 = same `times`,
  `easing=EaseInQuad`, `values={0, 2*kOne, kOne, 0}`. (Keep FIXED.)

## The goldens (PINNED, cross-platform) — append to tests/seq_test.cpp
```
seq-s2: sine-ease-table digest = 0x<...>
seq-s2: quad-in-table digest = 0x<...>   quad-out-table digest = 0x<...>
seq-s2: sequence-bus sweep digest = 0x<...>
PASS seq-s1: ... (all 6 S1 assertions STILL green — S1 digest 0xd314f17ebe3d480b UNCHANGED)
PASS seq-s2: easing-table digests == pinned (the host-baked LUTs are byte-stable cross-platform)
PASS seq-s2: Ease(*, 0) == 0 and Ease(*, kOne) == kOne exactly (every easing fixes the endpoints)
PASS seq-s2: Ease(Linear, t) == t for a sweep of t (the identity easing is a no-op)
PASS seq-s2: EaseInOutSine is symmetric — Ease(s, kOne/2) == kOne/2 (the S-curve midpoint, +-1 LSB)
PASS seq-s2: EaseInOutSine(t) != Linear(t) at t=kOne/4 (the curve actually bends — not a sneaky identity)
PASS seq-s2: SampleSequence at t=0.5s returns the per-channel eased bus (multi-track sampling)
PASS seq-s2: SampleSequenceSweep digest == pinned uint64 (the multi-track timeline is byte-stable)
```
Assertions:
1. **S1 INVARIANT** — re-run S1 assertion 1: `DigestTrack(SampleSweep(MakeShowcaseTrack(), kOne/30, 90))
   == 0xd314f17ebe3d480b` UNCHANGED (Linear path is bit-identical to S1).
2. **TABLE DIGESTS PINNED** — `net::DigestBytes` of each of the 3 easing tables == 3 pinned `uint64`s
   (the host-baked LUTs are byte-stable MSVC + clang).
3. **ENDPOINTS EXACT** — for every `Easing` value, `Ease(e, 0)==0` and `Ease(e, kOne)==kOne`.
4. **LINEAR IDENTITY** — for a sweep of `t01` in `[0,kOne]` (e.g. 65 samples), `Ease(Linear,t01)==t01`.
5. **SINE SYMMETRY** — `Ease(EaseInOutSine, kOne/2) == kOne/2` (exact within ±1 LSB — assert
   `abs(x-kOne/2) <= 1`; the (1-cos(pi/2))/2 = 0.5 midpoint).
6. **CURVE BENDS** — `Ease(EaseInOutSine, kOne/4) != Ease(Linear, kOne/4)` (the S-curve is NOT a hidden
   identity — at t=0.25 the sine ease is below linear).
7. **MULTI-TRACK BUS** — `SampleSequence(MakeShowcaseSequence(), kOne/2)` returns a 3-element bus whose
   channel 0 == `SampleScalar(MakeShowcaseTrack(), kOne/2)` (channel 0 is the S1 track verbatim).
8. **SEQUENCE SWEEP PINNED** — `net::DigestBytes(SampleSequenceSweep(MakeShowcaseSequence(), kOne/30, 90))`
   == a pinned `uint64` (the whole multi-track timeline is byte-stable cross-platform).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/seq/seq.h` + `engine/sim/fpx.h` + `engine/math/math.h` + `engine/net/session.h` +
`tests/seq_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs
`clang++ -std=c++20 -I engine -I tests tests/seq_test.cpp -o /tmp/seq && /tmp/seq`, confirming ALL
assertions PASS with IDENTICAL pinned digests. Local Windows clang is the fast pre-check. NO Metal.

## Constraints (HARD)
- APPEND-ONLY to `engine/seq/seq.h` (add the new `Easing` members, the LUT builders/tables/accessors,
  `Ease`, the one-line `SampleScalar` wire, `Sequence`/`SampleSequence`/`SampleSequenceSweep`,
  `MakeShowcaseSequence` below S1). Do NOT modify S1's `FindSegment`/`SampleSweep`/`DigestTrack`/
  `MakeShowcaseTrack` semantics — the S1 digest MUST stay `0xd314f17ebe3d480b`. The ONLY edit inside an S1
  function is the single `Ease(...)` insertion in `SampleScalar`, which is a NO-OP for Linear/Step.
- Header stays SELF-CONTAINED: only `<cstddef>/<cstdint>/<vector>` + `sim/fpx.h` + `net/session.h`. **NO
  `<cmath>`, NO `<algorithm>`, NO float, NO `<random>`/clock/`std::hash`.** The sine table is a COMMITTED
  integer literal array (bake-and-paste), NOT computed with `<cmath>` at runtime.
- Pure-CPU INTEGER throughout. The LUTs are `fx` (int32 Q16.16); the read is `fxmul`/shift only.
- `tests/seq_test.cpp` stays self-contained; APPEND the S2 assertions + `MakeShowcaseSequence`. Keep ALL 6
  S1 assertions green.
- Branch `fix-seq-s2`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target seq_test'`
  then run `seq_test`, confirm ALL assertions (S1 + S2) PASS, exit 0. ALSO compile standalone with the local
  clang `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests. First run: pick the
  pinned digests from the printed values, pin them, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `seq_test` builds + PASSES on Windows with
  every assertion green (esp. S1 digest UNCHANGED + the table digests + sine symmetry + the sequence sweep),
  and the local clang standalone passes with identical digests. Report: commit hash, full test output
  (printed digests + PASS lines), the pinned `uint64`s, confirmation the S1 digest is unchanged,
  confirmation the header is still self-contained (list `#include`s), how you generated the sine table
  literals (and that NO `<cmath>` is in seq.h), and the local-clang result. Commit message via a temp file +
  `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only + self-containment, runs the Mac/clang standalone for the identical
  digests, ff-merges to master + pushes + deletes the branch + advances to S3 — the event track that
  composes `flow.h`.)
