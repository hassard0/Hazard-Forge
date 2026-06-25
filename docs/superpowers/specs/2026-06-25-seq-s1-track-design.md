# Slice SEQ-S1 — Scalar keyframe track + integer lerp (Issue #25, beachhead)

The beachhead of the DETERMINISTIC CINEMATIC SEQUENCER (issue #25) — the timeline-evaluation RUNTIME a
sequencer editor would drive (the visual editor GUI is OUT OF SCOPE). The moat: UE5 Sequencer is float to
the bone (float keyframe values, float curve interpolation, `FQuat::Slerp` via `acos`/`sin`, float playback
timing — two machines sampling the same sequence at the same time diverge in the low bits). A timeline whose
evaluation is **bit-identical cross-platform + lockstep/replay/scrub-able** is the moat extension UE5 lacks
— the sibling of the just-shipped `flow.h` (deterministic Blueprints).

S1 establishes the substrate: a Q16.16 scalar keyframe track + integer keyframe interpolation (the float
that breaks UE5, rebuilt in fixed point). Pure-CPU INTEGER. The golden is a hard-pinned `net::DigestBytes`
over a sampled sweep, proven identical Windows/MSVC + Mac/clang via a standalone clang compile — NO
render-bake, the cheapest proof.

## NEW file: engine/seq/seq.h (namespace hf::seq)
Header-only and **SELF-CONTAINED**: include ONLY `<cstddef>`, `<cstdint>`, `<vector>`, plus
`#include "sim/fpx.h"` (the Q16.16 toolbox: `fx`/`kOne`/`fxmul`/`fxdiv`) and `#include "net/session.h"` (for
`hf::net::DigestBytes`). NO `<cmath>` / float / clock / RNG / `<random>` / `<unordered_*>` / `<map>` /
`<functional>` / `std::hash` / `<algorithm>` (binary search is a hand-written integer loop; do NOT
`std::lower_bound`). It MUST compile standalone: `clang++ -std=c++20 -I engine -I tests tests/seq_test.cpp`
(like `flow_test.cpp` / `ik` tests — fpx.h + session.h are both standalone-compatible; confirm fpx.h pulls
only math.h + std, which it does). Do NOT include or reuse `engine/anim/animation.h` (the FLOAT keyframe
path — it stays untouched + golden-invariant; seq rebuilds interpolation in Q16.16 from scratch).

### Types (all in hf::seq)
```cpp
using hf::sim::fpx::fx;        // Q16.16 fixed-point scalar (int32)
using hf::sim::fpx::kOne;      // 1.0 in Q16.16 (65536)
using hf::sim::fpx::fxmul;     // (int64)a*b >> 16
using hf::sim::fpx::fxdiv;     // ((int64)a << 16) / b  (already guards/truncates; the engine's fixed divide)

enum class Easing : uint32_t { Step = 0, Linear = 1 };   // S2 APPENDS EaseInOutSine etc. — do NOT renumber

struct ScalarTrack {
    std::vector<fx> times;     // Q16.16 seconds, STRICTLY ASCENDING (the invariant — keys sorted, no dupes)
    std::vector<fx> values;    // Q16.16 keyframe values; values.size() == times.size()
    Easing          easing = Easing::Linear;
};
```

### Functions (pure integer, deterministic)
1. **`std::size_t FindSegment(const ScalarTrack& tr, fx t)`** — binary-search (a hand-written integer loop,
   NO `<algorithm>`) for the segment index `k` such that `times[k] <= t < times[k+1]`. Clamp: if `t <=
   times.front()` → 0; if `t >= times.back()` → `times.size()-1` (the last key index; sampling there holds
   the last value). Empty track is handled by `SampleScalar` (returns 0). Pure int compares.
2. **`fx SampleScalar(const ScalarTrack& tr, fx t)`** — the beachhead sample:
   - empty `times` → return 0 (deterministic).
   - clamp `t` to `[times.front(), times.back()]`.
   - `k = FindSegment(tr, t)`. If `k == times.size()-1` (at/past the last key) OR `tr.easing == Step` →
     return `values[k]` (hold).
   - `den = times[k+1] - times[k]`; `t01 = (den == 0) ? 0 : fxdiv(t - times[k], den)` (a Q16.16 in
     `[0, kOne]`; the coincident-key guard — but the strictly-ascending invariant means `den > 0`).
   - return `values[k] + fxmul(t01, values[k+1] - values[k])` — the load-bearing `a + t·(b-a)` integer lerp.
   (S1 only handles Step + Linear; Linear uses `t01` directly as the eased parameter. S2 inserts
   `Ease(easing, t01)` here. Keep the structure so S2's insertion is a one-line change.)
3. **`std::vector<fx> SampleSweep(const ScalarTrack& tr, fx dt, uint32_t n)`** — sample `tr` at `n` fixed
   ticks `t = (fx)((int64)i * dt)` for `i` in `[0, n)` → a byte-stable `std::vector<fx>` (the digest input;
   use int64 for `i*dt` to avoid overflow, cast to `fx`).
4. **`uint64_t DigestTrack(const std::vector<fx>& sweep)`** → `return hf::net::DigestBytes(sweep.data(),
   sweep.size() * sizeof(fx));` — the pinned-golden currency (contiguous `int32`, byte-stable).

### Fixture (deterministic, FIXED forever)
- `ScalarTrack MakeShowcaseTrack()` — a fixed 4-key track exercising rising, falling, and through-zero
  segments: `times = {0, kOne, 2*kOne, 3*kOne}` (0,1,2,3 seconds), `values = {0, kOne, -kOne/2, 2*kOne}`,
  `easing = Linear`. (Keep it FIXED forever — the golden pins its sweep.)

## The golden (PINNED, cross-platform) — tests/seq_test.cpp
Self-contained test in the `flow_test.cpp` shape (copy the `check()` helper + `HF_TEST_MAIN_INIT()` from
`tests/test_main.h`). Register `hf_add_pure_test(seq_test)` in `tests/CMakeLists.txt` next to
`flow_test`/`econ_test`.
```
seq-s1: showcase sweep digest = 0x<...>
PASS seq-s1: DigestTrack(SampleSweep(showcase, kOne/30, 90)) == pinned uint64 (the cross-platform proof)
PASS seq-s1: re-sampling the same track is bit-identical (deterministic)
PASS seq-s1: nudging one keyframe value changes the digest (keys are load-bearing)
PASS seq-s1: a linear-segment midpoint is exact — SampleScalar at t=0.5s of a 0->kOne key == kOne/2
PASS seq-s1: clamp-low — sampling before the first key holds values.front()
PASS seq-s1: clamp-high — sampling after the last key holds values.back()
```
Assertions:
1. **PINNED DIGEST** — `DigestTrack(SampleSweep(MakeShowcaseTrack(), kOne/30, 90))` (3 seconds at 30 Hz) ==
   a hard-pinned `uint64_t` (run once, read the printed value, pin THAT; identical MSVC + clang — the
   make-or-break cross-platform anchor).
2. **REPLAY-STABLE** — a second `SampleSweep` → identical digest.
3. **LOAD-BEARING** — clone the showcase, `values[1] += 1`, re-sweep → a DIFFERENT digest.
4. **LINEAR MIDPOINT** — on the segment `times[0..1]` (values 0 → kOne), `SampleScalar(tr, kOne/2)` ==
   `kOne/2` (the midpoint of a 0→1 linear ramp; exact, within ±1 LSB — assert `==` since 0.5 is exact in
   Q16.16). (Confirm: at t=0.5s, t01 = fxdiv(kOne/2, kOne) = kOne/2; value = 0 + fxmul(kOne/2, kOne) =
   kOne/2.)
5. **CLAMP LOW** — `SampleScalar(tr, -kOne)` == `tr.values.front()` (holds before the first key).
6. **CLAMP HIGH** — `SampleScalar(tr, 100*kOne)` == `tr.values.back()` (holds after the last key).

## Cross-platform proof (the cheap loop — NO render-bake)
The CONTROLLER `scp`s `engine/seq/seq.h` + `engine/sim/fpx.h` + `engine/math/math.h` (fpx's dep) +
`engine/net/session.h` + `tests/seq_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`)
to the Mac and runs `clang++ -std=c++20 -I engine -I tests tests/seq_test.cpp -o /tmp/seq && /tmp/seq`,
confirming the test PASSES with the IDENTICAL pinned digest. (A local Windows clang at `C:\Program Files\
LLVM\bin\clang++.exe` is the fast pre-check.) NO Metal, NO `tests/golden/*`.

## Constraints (HARD)
- NEW header `engine/seq/seq.h` (new dir `engine/seq/`); compiles STANDALONE under clang with just
  `-I engine -I tests` (self-contained: only `<cstddef>/<cstdint>/<vector>` + `sim/fpx.h` + `net/session.h`).
  Do NOT modify `sim/fpx.h` / `net/session.h` / any existing header (read-only reuse). Do NOT include or
  reuse `anim/animation.h` (the float path). Do NOT add `<algorithm>` (hand-written binary search). Do NOT
  add it to any RHI/GPU target.
- Pure-CPU INTEGER on the bit-exact path: NO float / `<cmath>` / clock / RNG / `<random>` / `<unordered_*>`
  / `<map>` / `<functional>` / `std::hash` / `<algorithm>`. All interpolation is `fxmul`/`fxdiv` integer.
- `tests/seq_test.cpp` is SELF-CONTAINED (copy the scaffolding; do NOT include other tests). Register
  `hf_add_pure_test(seq_test)` in `tests/CMakeLists.txt`. Use `test_main.h` `HF_TEST_MAIN_INIT()`.
- Branch `fix-seq-s1`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target seq_test'`
  Run the test exe, confirm it PRINTS the digest and PASSES. First run: pick the pinned digest from the
  printed value, pin it, rebuild, confirm green. ALSO compile standalone with the local clang and confirm
  the IDENTICAL digest (MSVC==clang).
- COMPLETION CRITERIA — do NOT commit until: the header compiles, `seq_test` builds + PASSES on Windows with
  all assertions green, and the local clang standalone passes with the identical digest. Report: the commit
  hash, the full test output (printed digest + PASS lines), the exact pinned `uint64_t`, confirmation the
  header is self-contained (list its `#include`s), the showcase track you built, and the local-clang result.
  Commit message via a temp file + `git commit -F` (Bash heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits the diff, runs the Mac/clang standalone to confirm the IDENTICAL digest, then
  ff-merges to master + pushes + deletes the branch + advances to S2 — the easing-curve LUT + multi-track
  sequence.)
