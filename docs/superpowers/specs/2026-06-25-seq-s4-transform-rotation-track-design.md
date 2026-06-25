# Slice SEQ-S4 — Transform / rotation track: the Q16.16 rotation crux (Issue #25)

S1–S3 shipped scalar/easing/event tracks. S4 is the HEADLINE crux: a **transform track** — translation +
rotation + scale sampled at a tick into an integer `FxTransform` — where the rotation is the float that
breaks UE5 Sequencer (`FQuat::Slerp` via `acos`/`sin`, float playback timing → two machines diverge in the
low bits) rebuilt as a **deterministic Q16.16 integer nlerp**: zero runtime transcendentals, bit-identical
cross-platform. v1 uses normalized lerp (nlerp), NOT true constant-velocity slerp — documented as the
fidelity tradeoff (LUT-slerp via `ik.h`'s `FxAcosLut` is the later upgrade; nlerp is correct, deterministic,
and standard for cutscenes).

Pure-CPU INTEGER, append-only to `engine/seq/seq.h`. S1–S3 stay UNTOUCHED + golden-invariant (re-assert
their digests). NO new include — S4 reuses `sim/fpx.h`'s `FxVec3`/`FxQuat`/`FxQuatMul`/`FxQuatNormalize` +
`FxAdd`/`FxSub`/`FxScale` (already pulled in). The 6 includes stay exactly as S3 left them.

## Append to engine/seq/seq.h (below S3, in hf::seq)

### 1. Reuse the fpx quaternion/vector substrate (read-only)
```cpp
using hf::sim::fpx::FxVec3;          // {fx x,y,z}
using hf::sim::fpx::FxQuat;          // {fx x,y,z,w}  (w defaults to kOne = identity)
using hf::sim::fpx::FxQuatMul;       // Hamilton product (int64 fxmul terms)
using hf::sim::fpx::FxQuatNormalize; // integer unit-normalize via FxISqrt (NO <cmath>)
```
(`FxVec3` has `FxAdd`/`FxSub`/`FxScale`; `FxQuat` has NO arithmetic helpers — do component-wise inline.)

### 2. `FxTransform` — the integer twin of a TRS pose
```cpp
struct FxTransform {
    FxVec3 t;                          // translation (Q16.16 world units)
    FxQuat r = FxQuat{0,0,0,kOne};     // rotation (unit quaternion, identity default)
    FxVec3 s = FxVec3{kOne,kOne,kOne};  // scale (Q16.16, identity default)
};
```

### 3. `Vec3Track` — three scalar tracks sampled into an FxVec3
```cpp
struct Vec3Track { ScalarTrack x, y, z; };
inline FxVec3 SampleVec3(const Vec3Track& tr, fx t) {
    return FxVec3{ SampleScalar(tr.x,t), SampleScalar(tr.y,t), SampleScalar(tr.z,t) };
}
```
(Reuses S1's `SampleScalar` verbatim — translation/scale are just three eased scalar channels.)

### 4. `RotationTrack` — keyframe quaternions + integer nlerp (THE CRUX)
```cpp
struct RotationTrack {
    std::vector<fx>     times;   // Q16.16 seconds, STRICTLY ASCENDING (sorted, no dupes)
    std::vector<FxQuat> keys;    // unit quaternions; keys.size()==times.size()
};
inline FxQuat SampleRotation(const RotationTrack& tr, fx t);
```
`SampleRotation`:
- empty → return identity `FxQuat{0,0,0,kOne}`. Single key → return `keys[0]`.
- clamp `t` to `[times.front(), times.back()]`; find segment `k` (reuse the S1 `FindSegment` integer
  binary-search pattern over `tr.times` — write a small inline loop if `FindSegment` takes a `ScalarTrack`;
  do NOT add `<algorithm>`). At/past the last key → return `keys.back()`.
- `den = times[k+1]-times[k]`; `t01 = (den==0)?0:fxdiv(t-times[k], den)` (Q16.16 in `[0,kOne]`).
- **SHORTEST-ARC FLIP** — `qa = keys[k]`, `qb = keys[k+1]`. Compute the int64 dot
  `dot = (int64)qa.x*qb.x + (int64)qa.y*qb.y + (int64)qa.z*qb.z + (int64)qa.w*qb.w` (Q32.32). If `dot < 0`,
  negate every component of `qb` (the shortest-arc convention — quaternions double-cover SO(3); without
  this a 0→large-angle interp takes the long way). Pure-int sign test.
- **INTEGER NLERP** — component-wise lerp then normalize:
  `FxQuat m{ qa.x + fxmul(t01, qb.x-qa.x), ... w likewise };` then `return FxQuatNormalize(m);`
  (the `a + t·(b-a)` lerp per component, exactly the scalar lerp, then the proven integer normalize. NO
  transcendental, NO `acos`/`sin` — that is the determinism win. Document: nlerp is not constant angular
  velocity; LUT-slerp via `FxAcosLut` is the future fidelity slice.)

### 5. `TransformTrack` + sampler
```cpp
struct TransformTrack { Vec3Track translation; RotationTrack rotation; Vec3Track scale; };
inline FxTransform SampleTransform(const TransformTrack& tr, fx t) {
    return FxTransform{ SampleVec3(tr.translation,t), SampleRotation(tr.rotation,t), SampleVec3(tr.scale,t) };
}
```

### 6. Hand-LE serialization of a sampled transform sweep (padding-safe digest input)
`FxTransform` is 10 `fx` (3 t + 4 r + 3 s) but the STRUCT may carry padding — NEVER `DigestBytes` a vector
of structs. Serialize each sampled transform field-by-field into a contiguous `fx` buffer (the replay.h /
S1 discipline):
```cpp
inline std::vector<fx> SampleTransformSweep(const TransformTrack& tr, fx dt, uint32_t n) {
    std::vector<fx> out; out.reserve((size_t)n * 10u);
    for (uint32_t i=0;i<n;++i){ fx t=(fx)((int64_t)i*dt); FxTransform x=SampleTransform(tr,t);
        out.push_back(x.t.x); out.push_back(x.t.y); out.push_back(x.t.z);
        out.push_back(x.r.x); out.push_back(x.r.y); out.push_back(x.r.z); out.push_back(x.r.w);
        out.push_back(x.s.x); out.push_back(x.s.y); out.push_back(x.s.z); }
    return out;   // contiguous fx -> DigestTrack() (S1's net::DigestBytes) is padding-safe + byte-stable
}
```

### 7. Fixtures (FIXED forever)
- `RotationTrack MakeShowcaseRotation()` — a fixed rotation arc through ~3 keys exercising the shortest-arc
  flip. Use unit quaternions about the Y axis at, e.g., 0°, 90°, 180° so an interp crosses the half-turn
  (where the double-cover flip matters). Provide the keys as PINNED Q16.16 literal `FxQuat`s (a small
  throwaway computes `cos(θ/2)`/`sin(θ/2)` ONCE → snap to Q16.16 → paste; seq.h stays `<cmath>`-free).
  Verify each pasted key is ≈ unit (FxQuatNormalize-stable). Keep FIXED.
- `TransformTrack MakeShowcaseTransform()` — translation = a Vec3Track reusing the S1 `MakeShowcaseTrack()`
  shape on each axis (or distinct simple ramps), rotation = `MakeShowcaseRotation()`, scale = a gentle
  Vec3Track (e.g. constant `kOne` or a small ramp). Keep FIXED — the golden pins its sweep.

## The goldens (PINNED, cross-platform) — append to tests/seq_test.cpp
```
seq-s4: transform-sweep digest = 0x<...>
PASS seq-s1/s2/s3: ... (ALL prior assertions STILL green — every prior digest UNCHANGED)
PASS seq-s4: SampleTransformSweep digest == pinned uint64 (the transform timeline is byte-stable)
PASS seq-s4: re-sampling the transform track is bit-identical (deterministic)
PASS seq-s4: a sampled rotation stays ~unit — |q| within the documented Q16.16 drift band of kOne
PASS seq-s4: shortest-arc — interpolating across the 180-degree key takes the short path (w stays >= 0 side)
PASS seq-s4: rotation endpoints are exact — SampleRotation at a key time == that key (normalized)
PASS seq-s4: nudging one rotation key changes the transform-sweep digest (keys are load-bearing)
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0xd314f17ebe3d480b`, the three S2 table digests + S2 sequence-sweep
   `0xee44096d40ab3946`, and S3 event-sweep `0x1035f49824b6ac7a`, ALL UNCHANGED.
2. **PINNED TRANSFORM SWEEP** — `DigestTrack(SampleTransformSweep(MakeShowcaseTransform(), kOne/30, 90))`
   == a pinned `uint64` (byte-stable MSVC + clang).
3. **REPLAY-STABLE** — a second sweep → identical digest.
4. **~UNIT ROTATION** — for several sampled `t`, `FxLength`-equivalent of `SampleRotation` is within a
   documented band of `kOne` (nlerp+normalize keeps it ≈unit; assert `abs(len - kOne) <= BAND` with BAND a
   small pinned constant, e.g. a few LSB — measure it, pin it, document it the FPX4 way). Compute the
   quat length as `FxISqrt((int64)x*x + y*y + z*z + w*w)` (Q32.32 → Q16.16).
5. **SHORTEST-ARC** — interpolate at the midpoint of the segment that crosses the 180° key; assert the
   result took the SHORT path (e.g. the interpolated quat's `w` has the expected sign / the angle to both
   endpoints is ≤ 90°). Concretely: build a 2-key rotation `{q(0°), q(180° about Y)}` where a naive lerp
   without the flip would pass through a degenerate/!=shortest result; assert the flipped nlerp midpoint
   equals the expected `q(90°)`-direction (within the drift band). (The point: prove the `dot<0` flip fires
   and changes the result vs. not flipping.)
6. **ENDPOINT EXACT** — `SampleRotation(tr, times[k])` == `FxQuatNormalize(keys[k])` for each key time
   (sampling AT a key returns that key, normalized).
7. **LOAD-BEARING ROTATION KEY** — clone the showcase, perturb one rotation key component, re-sweep → a
   DIFFERENT transform-sweep digest (rotation keys are load-bearing).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/seq/seq.h` + `engine/sim/fpx.h` + `engine/math/math.h` + `engine/flow/flow.h` +
`engine/net/session.h` + `tests/seq_test.cpp` (+ `tests/test_main.h` + `engine/platform/crash_dialogs.h`)
to the Mac and runs `clang++ -std=c++20 -I engine -I tests tests/seq_test.cpp -o /tmp/seq && /tmp/seq`,
confirming ALL assertions PASS with IDENTICAL pinned digests (esp. the transform sweep + the ~unit band).
Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/seq/seq.h` (add `FxTransform`/`Vec3Track`/`SampleVec3`/`RotationTrack`/
  `SampleRotation`/`TransformTrack`/`SampleTransform`/`SampleTransformSweep`/`MakeShowcaseRotation`/
  `MakeShowcaseTransform` below S3). Do NOT modify any S1–S3 type or function semantics. ALL prior digests
  stay pinned.
- NO new include — reuse fpx.h's quaternion/vector API (already included). The 6 includes stay
  `<cstddef>/<cstdint>/<vector>` + `sim/fpx.h` + `net/session.h` + `flow/flow.h`. STILL NO `<cmath>`,
  `<algorithm>`, float, `<random>`, clock, `std::hash`. The showcase quats are COMMITTED Q16.16 literals
  (baked once with a throwaway, generator deleted — seq.h stays `<cmath>`-free).
- Pure-CPU INTEGER. The nlerp is component-wise `fxmul` + `FxQuatNormalize`; the shortest-arc dot is int64.
  Serialization is field-by-field `fx` (NEVER memcpy/DigestBytes a struct vector — padding).
- `tests/seq_test.cpp` stays self-contained; APPEND the S4 assertions + the two fixtures. Keep ALL S1–S3
  assertions green.
- Branch `fix-seq-s4`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target seq_test'`
  then run `seq_test`, confirm ALL assertions (S1–S4) PASS, exit 0. ALSO compile standalone with the local
  clang `C:\Program Files\LLVM\bin\clang++.exe` and confirm IDENTICAL digests. First run: pin the S4 digest
  + measure/pin the ~unit drift BAND from the printed values, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `seq_test` builds + PASSES on Windows with
  every assertion green (esp. prior-invariant digests + the transform sweep + shortest-arc + ~unit band),
  and the local clang standalone passes with identical digests. Report: commit hash, full test output
  (printed digest + PASS lines), the pinned S4 `uint64` + the drift BAND you measured, confirmation ALL
  prior digests are unchanged, confirmation the header is self-contained (list the 6 `#include`s, NO new
  one) and `<cmath>`-free, the showcase quats you baked + how, how you proved the shortest-arc flip fires,
  and the local-clang result. Flag any deviation. Commit message via a temp file + `git commit -F` (Bash
  heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only + self-containment, runs the Mac/clang standalone for the identical
  digests, ff-merges to master + pushes + deletes the branch + advances to S5 — the lockstep/replay/SCRUB
  capstone via net::Session, the moat headline.)
