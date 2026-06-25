# Slice PROFILE-S4 — Draw-call / GPU-pass inspection (Issue #31)

S1–S3 built the capture event stream, scope tree, and multi-frame timeline. S4 adds the **render-structure
inspection** — the "draw call inspection" half of the issue: a deterministic record of the frame's render
passes and their draw calls (pass order, draw counts, instance counts, pipeline IDs), with its own pinned
structural digest. This is the deterministic *structure* of the render — NOT the GPU timing (which is the
non-golden overlay).

**THE CLANG-PURITY BOUNDARY:** the real render structure lives in `render::RenderGraph` (`render_graph.h`),
which pulls `rhi.h` + `<functional>` + `<string>` → NOT standalone-clang-compilable. So `profile.h` NEVER
`#include`s it. Instead S4 defines a plain-POD `RenderStructInput` (vectors of integers), and the **live
engine (S6) populates it** from `RenderGraph::LastOrder()` + per-pass draw enumeration + MDI `drawCount` — the
exact `replay.h` `serWorld`/`deserWorld` injected-callable boundary. The profiler stays pure; the render
graph stays on the engine side.

Pure-integer, append-only to `engine/profile/profile.h` (below S3; do NOT modify S1–S3 — `0xedc7791443141dfd`
/ `0xb41eb67a1d13443e` / `0xc68ff46e1ab25f37` stay pinned). NO new include. S4 is a STANDALONE structure (it
does NOT mutate S1's `Capture` — it builds its own `RenderStructure`, so S1's `EncodeStructural` is untouched).

## Append to engine/profile/profile.h (below S3, in hf::profile)

### 1. The render-structure records (integers — structure, NOT timing)
```cpp
struct DrawRecord {
    uint32_t passId        = 0;   // index of the owning pass in execution order
    uint32_t pipelineId    = 0;   // the bound pipeline/PSO id (a stable engine-side id, not a pointer)
    uint32_t drawCount     = 0;   // 1 for a normal draw; N for an MDI draw that collapses N objects
    uint32_t instanceCount = 0;   // instances in this draw
    uint32_t indexCount    = 0;   // indices drawn (0 for non-indexed)
};
struct PassRecord {
    uint32_t nameId         = 0;   // the interned pass name id
    uint32_t firstDraw      = 0;   // index of this pass's first DrawRecord in RenderStructure.draws
    uint32_t drawCount      = 0;   // number of draws in this pass
    uint32_t totalInstances = 0;   // sum of instanceCount over this pass's draws
};
struct RenderStructure { std::vector<PassRecord> passes; std::vector<DrawRecord> draws; };
```

### 2. The injected input (a plain POD — the live engine fills this from RenderGraph)
```cpp
// One pass as the live caller describes it: a name id + its draws (in submission order). The caller builds
// `passes` in RenderGraph::LastOrder() execution order. profile.h NEVER #includes render_graph.h — this POD
// is the boundary (the replay.h serWorld pattern).
struct RenderPassInput { uint32_t nameId = 0; std::vector<DrawRecord> draws; };
struct RenderStructInput { std::vector<RenderPassInput> passes; };  // in execution (topo) order
```

### 3. `IngestRenderStructure` — POD → the deterministic RenderStructure
```cpp
inline RenderStructure IngestRenderStructure(const RenderStructInput& in);
```
- For each pass `p` in `in.passes` IN ORDER (passId = its index — execution order is the deterministic key):
  - `firstDraw = draws.size()`, `totalInstances = 0`.
  - For each `d` in `p.draws` (in order): push `DrawRecord{ passId, d.pipelineId, d.drawCount, d.instanceCount,
    d.indexCount }`; `totalInstances += d.instanceCount`.
  - push `PassRecord{ p.nameId, firstDraw, (uint32_t)p.draws.size(), totalInstances }`.
- Return the `RenderStructure`. (Execution order is the canonical order — no sorting needed; passId == index.)

### 4. `DigestRenderStructure` — the pinned structural digest (counts + ids, NO timing)
```cpp
inline uint64_t DigestRenderStructure(const RenderStructure& rs);
```
Hand-LE in passId/draw order: `PutU32(passCount)`, then per pass `PutU32(nameId)`, `PutU32(firstDraw)`,
`PutU32(drawCount)`, `PutU32(totalInstances)`; then `PutU32(drawTotal)`, then per draw `PutU32(passId)`,
`PutU32(pipelineId)`, `PutU32(drawCount)`, `PutU32(instanceCount)`, `PutU32(indexCount)`; then
`net::DigestBytes`. (Integers only — pass order, draw counts, pipeline ids. No nanoseconds anywhere — GPU
timing is the overlay, never in this digest.)

### 5. Fixture (FIXED forever)
- `RenderStructInput MakeShowcaseRenderInput()` — a fixed 3-pass frame (nameIds 10/11/12 representing
  Shadow/Lit/Composite — fixed ints, the test documents the mapping):
  - pass 0 (Shadow, nameId 10): 1 draw `{pipelineId 100, drawCount 1, instanceCount 1, indexCount 36}`.
  - pass 1 (Lit, nameId 11): 1 **MDI** draw `{pipelineId 200, drawCount 64, instanceCount 64, indexCount
    1536}` (the MDI collapses 64 objects — `drawCount=64` is the inspection headline).
  - pass 2 (Composite, nameId 12): 1 draw `{pipelineId 300, drawCount 1, instanceCount 1, indexCount 3}`.
  (Keep FIXED — the golden pins its `DigestRenderStructure`.)

## The golden (PINNED, cross-platform) — append to tests/profile_test.cpp
```
profile-s4: render-structure digest = 0x<...>  (<P> passes, <D> draws)
PASS profile-s1..s3: ... (all prior assertions STILL green — every prior digest UNCHANGED)
PASS profile-s4: IngestRenderStructure(showcase) has 3 passes, 3 draws, in execution order
PASS profile-s4: DigestRenderStructure == pinned uint64 (the render structure is byte-stable cross-platform)
PASS profile-s4: the Lit MDI pass's draw reports drawCount==64 (draw-call inspection — the MDI collapse count)
PASS profile-s4: pass totalInstances aggregates exactly (Lit pass totalInstances == 64)
PASS profile-s4: a changed draw count changes the render-structure digest (counts are load-bearing)
PASS profile-s4: a changed pipelineId changes the digest (pipeline binding is load-bearing)
```
Assertions:
1. **PRIOR INVARIANT** — re-assert S1 `0xedc7791443141dfd`, S2 `0xb41eb67a1d13443e`, S3 `0xc68ff46e1ab25f37`,
   all UNCHANGED.
2. **STRUCTURE** — `IngestRenderStructure(MakeShowcaseRenderInput())` → `passes.size() == 3`, `draws.size()
   == 3`; pass 0/1/2 have `passId` 0/1/2; `passes[1].firstDraw == 1`.
3. **PINNED DIGEST** — `DigestRenderStructure(IngestRenderStructure(MakeShowcaseRenderInput()))` == a
   hard-pinned `uint64_t` (run once, pin THAT; identical MSVC + clang).
4. **MDI INSPECTION** — the Lit pass's draw has `drawCount == 64` (the multi-draw-indirect collapse count —
   the "draw call inspection" datum).
5. **AGGREGATION** — `passes[1].totalInstances == 64` (the Lit pass's instance sum).
6. **COUNT LOAD-BEARING** — change the MDI `drawCount` to 65 → a DIFFERENT digest.
7. **PIPELINE LOAD-BEARING** — change a `pipelineId` → a DIFFERENT digest (pipeline binding is part of the
   render structure).

## Cross-platform proof (the cheap loop — NO render-bake)
Controller `scp`s `engine/profile/profile.h` + `engine/net/session.h` + `tests/profile_test.cpp` (+
`tests/test_main.h` + `engine/platform/crash_dialogs.h`) to the Mac and runs `clang++ -std=c++20 -I engine
-I tests tests/profile_test.cpp -o /tmp/profile && /tmp/profile`, confirming ALL assertions PASS with the
IDENTICAL pinned digests. Local Windows clang is the fast pre-check.

## Constraints (HARD)
- APPEND-ONLY to `engine/profile/profile.h` (add `DrawRecord`/`PassRecord`/`RenderStructure`/`RenderPassInput`/
  `RenderStructInput`/`IngestRenderStructure`/`DigestRenderStructure`/`MakeShowcaseRenderInput` below S3). Do
  NOT modify S1–S3 — all prior digests stay pinned. Do NOT add members to S1's `Capture` (S4 is a standalone
  `RenderStructure`).
- NO new include. Header stays self-contained (4 includes) + standalone-clang-compilable. **Do NOT
  `#include "render/render_graph.h"` or `rhi/rhi.h`** — the `RenderStructInput` POD is the boundary. STILL NO
  `<string>`/`<cmath>`/clock/RNG/`<unordered_*>`/`<map>`/`std::hash`/`<algorithm>`.
- No timing in the render-structure digest — counts + ids only (GPU timing is the S6 overlay).
- `tests/profile_test.cpp` stays self-contained; APPEND the S4 assertions + `MakeShowcaseRenderInput`. Keep
  ALL S1–S3 assertions green.
- Branch `fix-profile-s4`, commit there, do NOT merge. Do NOT commit any `tests/golden/*`.
- Build Windows via the PowerShell tool (single-quote the cmd arg for the (x86) parens):
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target profile_test'`
  (if the `cmd /c '...&&...'` handoff swallows output, run the build via the PowerShell tool natively). Run
  `profile_test`, confirm ALL assertions (S1–S4) PASS, exit 0. ALSO compile standalone with the local clang
  `C:\Program Files\LLVM\bin\clang++.exe` and confirm the IDENTICAL digests. First run: pin the
  render-structure digest, rebuild, confirm green.
- COMPLETION CRITERIA — do NOT commit until the header compiles, `profile_test` builds + PASSES on Windows
  with every assertion green (esp. prior invariants + the pinned render digest + the MDI drawCount==64
  inspection + the load-bearing count/pipeline tests), and the local clang standalone passes with identical
  digests. Report: commit hash, full test output (printed digest + PASS lines), the pinned render-structure
  `uint64`, confirmation S1–S3 digests unchanged, confirmation includes unchanged (4, no render_graph.h/rhi.h),
  and the local-clang result. Flag any deviation. Commit message via a temp file + `git commit -F` (Bash
  heredoc). Commit to the branch and STOP.
  (The CONTROLLER audits append-only, runs the Mac/clang standalone for the identical digests, ff-merges to
  master + pushes + deletes the branch + advances to S5 — the SCRUB: serializable `.capture` + seek-to-frame
  via net::CatchUp, the headline.)
