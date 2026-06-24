# Slice NS1 — Generic deterministic session core (Issue #27, flagship #24 NETCODE, beachhead)

The irreducible primitive: a generic, header-only, transport-agnostic **`Session<World,Input>`** that drives N
ticks of a deterministic `Step` from a per-tick input ring and exposes a **state digest** — generalizing the
copy-pasted FPX5 `RunLockstep` (fpx.h:567) into ONE parameterized engine. Two peers fed the same inputs re-derive
a bit-identical world EVERY tick (the lockstep invariant) and a hard-pinned final digest. Pure-CPU integer,
strict **bit-exact digest golden** (no image, no Mac render-bake — same `session_test` hash on MSVC and clang).
NS1 is the lockstep core only; input-delay (NS2), prediction+rollback (NS3), transport (NS4), desync-detect (NS5),
and the adversarial capstone (NS6) build on it.

## Keep it self-contained (the cheap cross-platform proof)
`session.h` must compile STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/session_test.cpp` on the
Mac (like dsp.h) — so it includes ONLY `<cstdint>`/`<vector>` (and `<cstddef>`), NO fpx.h / mixer / RHI. The
session is templated on `World`/`Input` with the deterministic `Step`/`Digest` supplied by the caller, and the
test drives it with TRIVIAL self-contained integer **toy worlds** (NOT fpx — proving the abstraction is generic
without pulling a heavy dependency; the real-sim drivers are validated in a later, non-standalone path).

## The header — `engine/net/session.h` (NEW, header-only, namespace `hf::net`)
- `inline uint64_t DigestBytes(const void* data, std::size_t n)` — FNV-1a-64 (offset basis
  `1469598103934665603ull`, prime `1099511628211ull` — the engine's FNV, same as `dsp.h::DigestBuffer`), hashing
  `n` raw bytes. The generic state-digest currency (the per-sim harnesses only `memcmp`; a pinned `uint64_t` golden
  is the new contribution).
- A per-tick **input ring**: `template <class Input> struct InputRing { ... };` storing the inputs that apply on
  each tick, queried by tick index. For NS1 a simple growable structure is fine (e.g. `std::vector<std::vector<Input>>`
  indexed by tick, or a fixed-capacity ring keyed by `tick % cap`); expose `void AddInput(uint32_t tick, const
  Input&)` and `const std::vector<Input>& At(uint32_t tick) const` (returns an empty vector for a tick with no
  inputs — deterministic). Inputs at the same tick are kept in **insertion order** (deterministic application
  order — the `SimTick` in-array-order contract).
- The generic session:
  ```
  template <class World, class Input>
  struct Session {
      World    world;          // the deterministic state
      InputRing<Input> ring;   // inputs by tick
      uint32_t tick = 0;       // next tick to step
  };
  ```
  + a templated advance that takes the deterministic policy callables (function pointers / template params /
  small callables — your choice, but NO `<functional>` if it can be avoided, to keep includes minimal):
  - `template <class World, class Input, class StepFn> void Advance(Session<World,Input>& s, StepFn step)`:
    calls `step(s.world, s.ring.At(s.tick), s.tick)` then `++s.tick`. `step` is the deterministic per-tick
    transition (the `SimTick`/`ApplyCommand`+integrate analog), pure-of-its-inputs.
  - `template <class World, class DigestFn> uint64_t CurrentDigest(const Session<World,?>& s, DigestFn digest)`
    OR simpler: let the caller compute `digest(s.world)` directly. Provide whatever is cleanest; the digest is
    `DigestBytes` over the world's deterministic bytes.
  - A convenience `template <...> uint64_t RunLockstep(World init, const InputRing<Input>& ring, uint32_t ticks,
    StepFn step, DigestFn digest)` that runs `ticks` advances from `init` over a COPY of `ring` and returns the
    final digest (the one-shot reference — the generalized `fpx.h:567 RunLockstep`).
  All pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG, NO GPU.

## CPU test — `tests/session_test.cpp` (NEW, register `hf_add_pure_test(session_test)` in tests/CMakeLists.txt)
Define TWO self-contained integer toy worlds + their Step/Digest, to prove the abstraction is generic:
- **ToyA** = `struct { int64_t acc; }`, `Input` = `int32_t`; `StepA(w, inputs, tick)` adds each input (folded with
  the tick, e.g. `w.acc += (int64_t)in * (tick+1)`) so the order/tick matter; `DigestA(w) = DigestBytes(&w.acc,
  sizeof w.acc)`.
- **ToyB** = `struct { std::vector<int32_t> cells; }`, `Input` = `struct { int idx; int32_t delta; }`; `StepB`
  applies each input to `cells[idx % size]`; `DigestB(w) = DigestBytes(w.cells.data(), w.cells.size()*4)`.
Assertions: (1) **2-peer lockstep (make-or-break)** — build a fixed input ring; run TWO independent `Session`s
(peer A, peer B) from the same `init` over the same ring, advancing both in lockstep, and assert their digests
are EQUAL at EVERY tick (not just the end) — the lockstep invariant; (2) **pinned digest** — the converged final
digest of ToyA (and ToyB) == a hard-pinned `uint64_t` (compute on first run, hardcode); (3) **input-order
determinism** — two inputs on the same tick applied in insertion order give a deterministic result; reversing the
insertion order changes the digest (proving order is load-bearing + deterministic); (4) **replay-stable** — two
`RunLockstep` calls over the same ring are bit-identical; (5) **generic** — both ToyA and ToyB drive the SAME
`Session`/`Advance`/`RunLockstep` code (the abstraction is not fitted to one world). Print the digests +
`session_test: ALL CHECKS PASSED`. Report lines:
```
ns1-session: generic lockstep session (2 peers, toy worlds A+B)
ns1-session: 2-peer lockstep — digests equal every tick {ticks:<T>}
ns1-session: converged digest pinned {toyA:0x<Ha>, toyB:0x<Hb>}
ns1-session: input-order determinism {inOrder:0x<H1>, reversed:0x<H2>} H1 != H2
```

## Proof (STRICT bit-exact digest — cross-platform bar)
The golden is the pinned `DigestBytes` in `session_test`; the cross-platform proof = the controller compiles+runs
`session_test` on the Mac (clang, standalone) → IDENTICAL pinned hashes. Make-or-break: the 2-peer per-tick digest
equality + the pinned converged digest.

## Constraints (HARD)
- NEW engine/net/session.h + tests/session_test.cpp + the tests/CMakeLists.txt registration ONLY. Keep session.h
  self-contained (only `<cstdint>`/`<vector>`/`<cstddef>`, NO fpx/mixer/RHI include — the Mac clang single-file
  compile must work). Reuse the FNV constants (the dsp.h values). Do NOT modify any other file. NO RHI/shader/GPU.
  Pure-CPU integer. NO `<cmath>`, NO float, NO clock/RNG.
- Branch `fix-issue-27-ns1`, commit there, do NOT merge. NO golden file (the pinned hash lives in the test).
- Build Windows + run session_test via the PowerShell tool (NOT bash), single-quoting the cmd arg for the (x86)
  parens:
  `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cmake --build C:\Users\ihass\dev\hazard-forge\build\windows-msvc-release --target session_test'`
  then run session_test.exe (under build\windows-msvc-release).
- COMPLETION CRITERIA — do NOT commit until: the build succeeds, session_test prints ALL CHECKS PASSED, the
  2-peer per-tick digest equality holds, the pinned digests match, input-order determinism holds, both toy worlds
  drive the same session code. Commit message via a temp file + `git commit -F` (use the Bash tool heredoc). Commit
  to the branch and STOP. Report: commit hash, the printed digests, confirmation session_test passes, and
  confirmation session.h stays self-contained (compiles standalone). (The CONTROLLER compiles+runs session_test on
  the Mac to confirm identical hashes, then merges.)
