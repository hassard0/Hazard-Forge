# Slice DSP4 — Declarative node-graph Patch (Issue #26, flagship #23 DETERMINISTIC AUDIO, 4th slice — THE HEADLINE)

The MetaSounds-class headline: compose the DSP1–DSP3 nodes (Osc / Env / Filter) + a new Mix node into a flat,
integer-index-wired **Patch** evaluated per block in fixed topological order — `EvaluateBlock(patch, …)` is a pure
function of the patch + the call sequence, so a whole synth voice is a declarative graph that produces a
byte-identical buffer. Pure-CPU integer, strict **bit-exact buffer-hash golden** (no image, no Mac render-bake).
Builds on DSP1/DSP2/DSP3. Uses the flat-index-graph discipline (the `pcg::PcgGraph` / `ai` decision-tree / `nav`
poly precedent — indices, NOT pointers).

## The addition — `engine/audio/dsp.h` (APPEND-ONLY after the DSP3 block; keep self-contained, no mixer/fpx include)
Add to `hf::audio::dsp` (do NOT modify DSP1/2/3; keep dsp.h compilable STANDALONE — only `<cstdint>`/`<vector>`):
- `enum class NodeType { Osc, Env, Filter, Mix };`
- `struct DspNode {`
    `NodeType type = NodeType::Osc;`
    `int in0 = -1; int in1 = -1;`   // input node indices — MUST be < this node's own index (fixed topo order)
    `OscNode    osc;`               // used when type==Osc (a source — ignores in0/in1)
    `EnvNode    env;`               // used when type==Env    (shapes in0's block)
    `BiquadNode filt;`              // used when type==Filter (filters in0's block)
    `int32_t    gain0 = 32767; int32_t gain1 = 32767;`  // Q15 mix gains (type==Mix: mixes in0,in1)
  `};`
- `struct Patch { std::vector<DspNode> nodes; int outNode = -1; };`  (outNode = the index whose block is the
  patch output; default to the last node if -1.)
- `void EvaluateBlock(Patch& p, int sampleRate, int frames, std::vector<int16_t>& outAppend)`: allocate a per-node
  scratch `std::vector<std::vector<int16_t>> blk(p.nodes.size())`; for each node `i` in ascending index order:
  - `Osc`   → `RenderBlock(p.nodes[i].osc, sampleRate, frames, blk[i])` (a source; ignores inputs).
  - `Env`   → `ApplyEnvBlock(p.nodes[i].env, blk[in0], blk[i])`.
  - `Filter`→ `FilterBlock(p.nodes[i].filt, blk[in0], blk[i])`.
  - `Mix`   → for each frame `f`: `blk[i].push_back(ClampI16(MulQ15(blk[in0][f], gain0) + MulQ15(blk[in1][f], gain1)))`
    (the mixer's int32-accumulate-then-clamp; if an input index is -1 treat that contribution as 0).
  Then append `blk[ (p.outNode < 0 ? (int)p.nodes.size()-1 : p.outNode) ]` to `outAppend`. The node states
  (`osc.phase`, `env.t`, `filt` delay line) are mutated IN `p.nodes` so they carry across `EvaluateBlock` calls —
  evaluating the patch block-by-block advances the whole graph deterministically. Validate inputs defensively
  (an out-of-range / forward `in` index → treat as a silent/zero input, never read out of bounds). NO `<cmath>`,
  NO float.
- A convenience `std::vector<int16_t> RenderPatch(Patch& p, int sampleRate, int totalFrames)` that evaluates
  `totalFrames` in ONE `EvaluateBlock` (the "one big buffer" reference).

## CPU test — extend `tests/dsp_test.cpp` (add a DSP4 section, keep DSP1–3 checks)
Assertions: (1) **single-osc patch == bare osc (make-or-break no-op)** — a `Patch` with one `Osc` node (outNode 0)
renders BYTE-IDENTICAL to `RenderOsc(...)` with the same params (the graph adds no overhead); (2) **chain
equivalence** — a 3-node patch `Osc → Env → Filter` (node1.in0=0, node2.in0=1, outNode=2) renders BYTE-IDENTICAL
to the hand-chained DSP1→DSP2→DSP3 (`RenderOsc` then `ApplyEnvBlock` then `FilterBlock` on fresh nodes) — proving
the graph composition is exactly the manual chain; (3) **block-boundary determinism** — that 3-node patch
evaluated as ONE `N*256` buffer (`RenderPatch`) vs N separate 256-frame `EvaluateBlock` calls on ONE persistent
`Patch` → byte-identical (all node states carry); (4) **mix node** — a `Mix` of two oscillators (different freqs)
is byte-identical to the hand-mixed `ClampI16(MulQ15(a,g0)+MulQ15(b,g1))` per frame, and gain0=gain1=0 → silence;
(5) **pinned hash** — a representative patch (e.g. two-osc mix → filter) digest == a hard-pinned `uint64_t`; a
different patch wiring → a different hash. Print the hashes + `dsp_test: ALL CHECKS PASSED`. Report lines:
```
dsp4-patch: declarative node graph (Osc/Env/Filter/Mix, flat integer-index wiring)
dsp4-patch: single-osc patch == bare RenderOsc BYTE-IDENTICAL (graph overhead transparent)
dsp4-patch: chain Osc->Env->Filter == hand-chained DSP1->DSP2->DSP3 BYTE-IDENTICAL
dsp4-patch: block-boundary determinism: one-buffer == N-blocks {hash:0x<H>}
dsp4-patch: mix of two oscillators pinned {hash:0x<Hm>}
```

## Proof (STRICT bit-exact buffer hash — cross-platform bar)
The golden is the pinned `DigestBuffer` in `dsp_test`; the cross-platform proof = the controller compiles+runs
`dsp_test` on the Mac (clang) → IDENTICAL hashes. Make-or-break: single-osc patch == bare osc, the chain ==
hand-chain, and one-buffer == N-blocks (all node states carry).

## Constraints (HARD)
- APPEND-ONLY to engine/audio/dsp.h (do NOT modify DSP1/2/3) + extend tests/dsp_test.cpp. Keep dsp.h
  self-contained (only `<cstdint>`/`<vector>`, NO mixer/fpx include — the Mac clang single-file compile must keep
  working). Do NOT modify mixer.{h,cpp}, scene.wav, or any other file. NO RHI/shader/GPU. Pure-CPU integer. NO
  `<cmath>`, NO float. Flat node array, integer-index wiring (NOT pointers); inputs reference LOWER indices (fixed
  topological order); bounds-check defensively.
- Branch `fix-issue-26-dsp4`, commit there, do NOT merge. NO golden file (the pinned hash lives in the test).
- Build Windows + run dsp_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86) parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target dsp_test'`
  then run dsp_test.exe (under build\windows-msvc-release).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, dsp_test prints ALL CHECKS PASSED, the single-osc
  patch == bare osc byte-identical, the chain == hand-chain byte-identical, one-buffer == N-blocks holds, the mix
  node matches the hand-mix, the pinned hash matches, AND DSP1–3 checks still pass. Commit message via a temp file
  + `git commit -F` (use the Bash tool heredoc). Commit to the branch and STOP. Report: commit hash, the printed
  hashes, confirmation dsp_test passes (DSP1–4), and confirmation dsp.h stays self-contained. (The CONTROLLER runs
  dsp_test on the Mac to confirm identical hashes, then merges.)
