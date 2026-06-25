# Slice SEQ-S6 — Float render capstone: the cutscene motion-trail money-shot (Issue #25)

S1–S5 built + PROVED the deterministic Q16.16 timeline (scalar/easing/event/transform tracks + lockstep/
scrub via net::Session). S6 is the OPTIONAL money-shot: render the bit-exact transform timeline as a lit 3D
scene — a hero object sampled at successive times along the timeline, drawn as a **ghosted motion trail**
that makes the keyframe interpolation visible (the sequencer's signature visual). The SIM stays bit-exact
integer; the ONLY float crossing is the per-sample `FxTransform → math::Mat4` conversion (render-only, OUT
of the bit-exact path) — the exact FPX6/FR6/JT6/VH6 capstone pattern.

This is the FLOAT visresolve-bar slice (the one float slice of the flagship): the golden is Metal-baked, the
gate is Metal-determinism (two renders BYTE-IDENTICAL) + provenance (every instance IS a bit-exact
`SampleTransform`) + visual parity; the Vulkan-vs-Metal cross-vendor delta is the documented float baseline
(~24–55 mean, NOT zero-diff). NO new shader, NO new RHI — reuse the existing instanced-lit pipeline VERBATIM.

## STEP 0 (DO THIS FIRST, REPORT IT): study the existing dual-harness wiring
Before writing anything, READ how a recent render capstone is wired across BOTH harnesses and report the
exact files/symbols, so seq replicates it by direct analogy:
- The Vulkan `--fract-render-shot` (or `--vehicle-render-shot`/`--fpx-render-shot`) showcase — find which
  file defines it (grep the repo for `"--fract-render-shot"` / `fract-render-shot` OUTSIDE
  `engine/editor/introspect.cpp`; the introspect.cpp entry is just the help text). Note the scene-build +
  draw scaffolding.
- The Metal `--fract-render` showcase in `metal_headless/visual_test.mm` (grep `"--fract-render"`).
- How both build the SAME deterministic scene (the shared sim helper) so the integer state is byte-identical
  cross-backend by construction.
- The instanced-lit draw path they reuse (`lit_instanced.vert` + `lit.frag` + `scene::InstanceTransformLayout`
  + the FrameData camera/light UBO), MATTE (roughness 1.0) to avoid the sky-IBL iridescence trap (the
  GF6/FR6 lesson).
Report these concretely in your final message — they define the seq wiring.

## 1. NEW file: engine/seq/seq_render.h (namespace hf::seq) — the float bridge
Mirrors `engine/econ/econ_render.h` / `engine/wfc/wfc_render.h`: the render bridge lives in a SEPARATE
`_render.h` so `seq.h` stays float-free (`<cmath>`-free, the bit-exact header). `#include "seq/seq.h"` +
`#include "math/math.h"` + `#include "sim/fpx.h"` (for `FxToFloat`) read-only. NO backend symbols,
header-only.
```cpp
// One math::Mat4 from a bit-exact FxTransform — the FPX6 FxBodyTransform pattern for a full TRS pose
// (translate * rotate * scale), the ONE float crossing. Render-only; NO sim mutation.
inline math::Mat4 SeqTransformToMat4(const FxTransform& x) {
    const math::Vec3 t{ fpx::FxToFloat(x.t.x), fpx::FxToFloat(x.t.y), fpx::FxToFloat(x.t.z) };
    const math::Quat q = math::Normalize(math::Quat{
        fpx::FxToFloat(x.r.x), fpx::FxToFloat(x.r.y), fpx::FxToFloat(x.r.z), fpx::FxToFloat(x.r.w) });
    const math::Vec3 s{ fpx::FxToFloat(x.s.x), fpx::FxToFloat(x.s.y), fpx::FxToFloat(x.s.z) };
    return math::FromTRS(t, q, s);
}
// One instance per sampled transform, in sample order. Empty input -> empty output (the empty no-op).
// Pure deterministic host float (no RNG/clock), render-only. Provenance: each Mat4 derives from a bit-exact
// SampleTransform output.
inline std::vector<math::Mat4> SeqToRenderInstances(const std::vector<FxTransform>& samples) {
    std::vector<math::Mat4> out; out.reserve(samples.size());
    for (const FxTransform& x : samples) out.push_back(SeqTransformToMat4(x));
    return out;
}
```
(Confirm the exact `math::Quat`/`math::Vec3`/`Normalize`/`FromTRS` names from `math/math.h` — match them.)

## 2. The shared deterministic cutscene scene (byte-identical cross-backend)
A small shared helper (place it where the fract scene helper lives, OR in `seq_render.h` as a pure-integer
function returning `std::vector<FxTransform>`) that BOTH the Vulkan `--seq-render-shot` and the Metal
`--seq-render` call, so the sampled transforms are byte-identical by construction:
```cpp
// The hero cutscene: a TransformTrack (translation arc + rotation sweep through the S4 shortest-arc keys +
// a gentle scale pulse) sampled at N successive times -> a ghosted MOTION TRAIL of the animated object.
// Pure integer (SampleTransform is bit-exact). FIXED forever.
inline std::vector<FxTransform> MakeCutsceneTrail(uint32_t n);   // e.g. n=24 samples over the timeline
```
Build a fixed `TransformTrack` (reuse `MakeShowcaseTransform()` from S4, or a slightly richer one — a clear
translation sweep so the trail SPREADS across the ground, the S4 rotation keys so the object visibly turns,
a subtle scale). Sample at `t_i = (fx)((int64)i * dt)` for `i in [0,n)` (a fixed `dt`). Return the vector
of `FxTransform`. This is the provenance source.

## 3. The showcases (BOTH — the metal-showcase gate: wire BOTH, the controller greps for the Metal flag)
- **Vulkan `--seq-render-shot`** (in whatever file the other `--*-render-shot` showcases live): build the
  cutscene via `MakeCutsceneTrail`, `SeqToRenderInstances` -> the instance set, draw as lit instanced cubes
  (the cube mesh) MATTE through the existing instanced-lit pipeline over the ground from a fixed 3/4 camera +
  directional light. Print the FOUR proof lines (below).
- **Metal `--seq-render`** (in `metal_headless/visual_test.mm`, next to `--fract-render`): the SAME scene
  (call the SAME `MakeCutsceneTrail` + `SeqToRenderInstances`), the SAME draw, baking to the golden path.
  ⚠️ DO NOT OMIT THIS — the controller WILL grep `visual_test.mm` for `"--seq-render"` before the Mac bake;
  if it's missing the bake builds the wrong default scene (the size-mismatch tell).
- **introspect.cpp**: add a `{"--seq-render-shot", "Deterministic Cinematic Sequencer LIT 3D RENDER CAPSTONE
  …"}` help entry in the established style (the one-float-crossing/provenance/visresolve language).

### The FOUR proofs (fail loudly, exact lines — the established render-capstone contract)
1. `seq-render: {samples:<N>, instances:<N>} from bit-exact timeline` (provenance: instances == samples).
2. `seq-render determinism: two runs BYTE-IDENTICAL` (two `SeqToRenderInstances` calls byte-equal; and the
   Metal two-run render DIFF 0.0000).
3. `seq-render provenance: instances == rebuild` (`SeqToRenderInstances(MakeCutsceneTrail(n))` ==
   a fresh rebuild, byte-for-byte — a pure function of the bit-exact samples).
4. `seq-render empty: base only (no-op)` (empty samples -> empty instances -> the cleared base scene).

## 4. A small CPU-side unit test (the cheap pre-bake proof) — tests/seq_render_test.cpp
A self-contained pure test (register `hf_add_pure_test(seq_render_test)`) that does NOT need Metal:
```
PASS seq-render: SeqToRenderInstances(MakeCutsceneTrail(24)) has 24 instances (provenance count)
PASS seq-render: two SeqToRenderInstances calls are byte-identical (deterministic float bridge)
PASS seq-render: empty samples -> empty instances (the no-op)
PASS seq-render: each instance translation == FxToFloat of the sample's bit-exact translation (provenance)
```
(Hand-check assertion 4: for a sample `x`, `SeqTransformToMat4(x)` 's translation column == `FxToFloat(x.t.*)`.
Read `math::Mat4`'s layout from `math.h` to index the translation. This proves the float bridge faithfully
carries the bit-exact integer state — the provenance the image golden then visually confirms.)

## Cross-platform proof (the render-bake — CONTROLLER does this on the Mac)
The implementer's pre-bake proof is the Windows build of `--seq-render-shot` (Vulkan) running + printing the
four proofs, AND the `seq_render_test` pure test green on MSVC + local clang. The CONTROLLER then on the Mac:
greps `visual_test.mm` for `"--seq-render"`, builds `metal_headless` (`cmake -S metal_headless -B
build-metal -G Ninja && cmake --build build-metal`), runs `--seq-render` TWICE (asserts the two PNGs are
byte-identical = determinism), bakes `tests/golden/metal/seq_render.png`, and records the Vulkan-vs-Metal
cross-vendor mean (the documented float baseline, NOT zero). NO strict pixel-hash golden (float slice).

## Constraints (HARD)
- NEW `engine/seq/seq_render.h` (render-only float bridge; mirrors econ_render.h). Do NOT add float to
  `seq.h` — it stays the 6-include `<cmath>`-free bit-exact header. Do NOT modify S1–S5 (all prior digests
  stay pinned: S1 `0xd314f17ebe3d480b`; S2 sine `0x8f13b44545cc3c97`/quad-in `0x7ebbb0956a7f50a2`/quad-out
  `0x5289c36d8551004a`/seq-sweep `0xee44096d40ab3946`; S3 `0x1035f49824b6ac7a`; S4 `0x59e3f94ce2da437d`;
  S5 lockstep `0x9ec0eb2bfbb40dca`/trace `0x7c63291062cf0ca7`/rollback `0x5963d4a3c0282769`). Re-run
  `seq_test` to confirm they're all still green.
- NO new shader, NO new RHI. Reuse the existing instanced-lit pipeline VERBATIM (the path
  `--fract-render`/`--vehicle-render` use). MATTE (roughness 1.0) — no iridescence.
- The bridge is pure deterministic host float (no RNG, no clock); the SCENE sampling is bit-exact integer
  `SampleTransform`. Provenance: every instance IS a `SampleTransform` output.
- WIRE BOTH showcases (`--seq-render-shot` Vulkan + `--seq-render` Metal). Do NOT omit the Metal one.
- `tests/seq_render_test.cpp` is self-contained; register `hf_add_pure_test(seq_render_test)`. Keep
  `seq_test` (S1–S5) green.
- Branch `fix-seq-s6`, commit there, do NOT merge. Do NOT commit any `tests/golden/*` (the CONTROLLER bakes
  the Metal golden on the Mac — you do NOT commit a Windows-baked PNG).
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target seq_render_test'`
  (and the visual-test target that hosts `--seq-render-shot` — report its name). Run `seq_render_test` +
  the Vulkan `--seq-render-shot`, confirm the four proofs print + the pure test passes. ALSO local clang
  standalone for `seq_render_test` (identical). Do NOT attempt a Metal build on Windows (Mac-only).
- COMPLETION CRITERIA — do NOT commit until: `seq_render.h` compiles, `seq_render_test` builds + PASSES on
  Windows (+ local clang), the Vulkan `--seq-render-shot` runs + prints the four proofs, and `seq_test`
  (S1–S5) is still fully green. Report: the STEP-0 wiring findings (the exact files/symbols for the Vulkan
  showcase + the Metal showcase + the shared scene helper + the instanced-lit draw path), commit hash, the
  `seq_render_test` output, the Vulkan `--seq-render-shot` four-proof output, confirmation the Metal
  `--seq-render` showcase IS wired in visual_test.mm (quote the `strcmp`/dispatch line), confirmation `seq.h`
  is untouched + all S1–S5 digests still green, confirmation NO `tests/golden/*` committed, the local-clang
  result, and any deviation flagged. Commit message via a temp file + `git commit -F` (Bash heredoc). Commit
  to the branch and STOP.
  (The CONTROLLER then: greps visual_test.mm for `--seq-render`, audits the bridge + append-only, runs the
  Mac render bake [build metal_headless, run `--seq-render` twice for determinism, bake the golden, record
  the cross-vendor mean], commits the golden, ff-merges to master + pushes + deletes the branch, then writes
  the ARCHITECTURE.md seq section + comments issue #25 — leaving it OPEN for the GUI editor.)
