# benchmarks/ — runnable proofs UE5 structurally cannot pass

These are the "offense" `docs/BEAT_UE5_PLAN.md` (Phase 3) calls for: head-to-head, **runnable**
comparisons where UE5 is not behind but *structurally disqualified* — because its float physics core
(FPU evaluation order, fused-multiply-add contraction, vendor math libraries, non-deterministic task
scheduling) cannot produce bit-identical simulation across machines, and cannot be retrofitted without
rearchitecting every shipped project.

Everything here composes the already-shipped, golden-gated determinism substrate **read-only** — no
engine header is modified. The proof is a single pure-CPU binary, `moat_proofs`, plus scripts to run it.

## What this is, and what it is not

- **It is** a directory of runnable proof scripts + a self-contained proof binary that *asserts* pinned
  deterministic outcomes. The gate is "the binary runs and every pinned digest holds." It is also a
  `ctest` target (`hf_moat_proofs`), so `scripts/verify.ps1` runs it alongside the pure-core suite.
- **It is not** a marketing deck. Each proof states exactly what it does and does not show. Two of the
  proofs are honest about their limits (see the per-proof notes): the cross-platform hash's *full*
  Win+Mac+Linux demonstration needs three machines (each leg is independently reproducible), and the
  bandwidth proof is a specific-scene measurement, not a universal ratio.

## Run it

```sh
# POSIX (Linux/macOS) — builds standalone against the header-only cores, runs, propagates exit code
./benchmarks/run_all.sh

# Windows (PowerShell) — same, using clang++; prints a per-proof PASS/FAIL table
pwsh benchmarks/run_all.ps1

# the cross-platform-hash leg for THIS machine (compare the printed hash across OSes)
./benchmarks/cross_platform_hash.sh            # native
./benchmarks/cross_platform_hash.sh --docker   # the Linux/gcc leg in a gcc:14 container

# or via the build tree, as a normal test
ctest --test-dir build/windows-msvc-release -R hf_moat_proofs -V
```

The binary prints one line per proof and **exits 0 iff all six pass**. Every digest is pinned identical
under MSVC and local clang — the same integer core, so bit-for-bit equal.

## The six proofs

Each line prints `PROOF k [name]: PASS digest=0x… (UE5: <one-line structural reason>)`.

### PROOF 1 — cross-platform determinism
**Demonstrates:** the canonical 24-tick gameplay+physics match (`verdict::BuildCanonicalReplay` →
`RunVerdictLockstep`) converges to a pinned final-world hash (`DigestSnapshot = 0x76a37a56d256c401`), and
the two lockstep peers fed *only inputs* re-derive the whole world byte-for-byte.
**Run:** in the binary, or `./benchmarks/cross_platform_hash.sh` to print just this hash.
**Expected:** `PROOF 1 [cross-platform determinism]: PASS digest=0x76a37a56d256c401 ticks=24`.
**Honesty:** this leg proves determinism *on this toolchain*. The claim that the *same* hash appears on
Windows/MSVC, macOS/Apple-clang and Linux/gcc is the three-machine demonstration — each leg is
reproducible, and `docs/DETERMINISM_THREE_PLATFORMS.md` records 135/135 pure-core digests already
matching on Linux/gcc. The integer core (fixed-point, host-baked LUTs, no runtime transcendentals) is
what makes all three agree.
**Why UE5 can't:** float Chaos physics + FPU-order/FMA/vendor-math/task-scheduling divergence — no two
machines produce the same simulation bit-for-bit.

### PROOF 2 — rollback/desync detection
**Demonstrates:** an NS5 lockstep run where one peer is fed an extra input at tick 7; the
`DesyncDetector` (per-tick digest exchange) catches the divergence at the *exact* tick, with the two
diverging digests differing and every earlier tick identical.
**Expected:** `PROOF 2 [rollback/desync detection]: PASS caughtTick=7 digest=0x49aa655446b5c3a2`.
**Why UE5 can't:** with no bit-exact per-tick digest, a desync is invisible — float drift is smoothed
over by interpolation and never localized.

### PROOF 3 — provable anti-cheat
**Demonstrates:** the AC1 authority verifier re-simulates a client's *submitted* input stream and
compares its own per-tick outcome digests against the client's *claimed* ones. An honest client is
VERIFIED (no divergence); a cheater that claims an impossible health bump is REJECTED at the *exact*
tamper tick (6). The verdict is reproducible — a third party re-runs `Verify` and gets the identical
answer. The input stream carries an integrity commitment (`0xd7326bc4bbec56ac`).
**Expected:** `PROOF 3 [provable anti-cheat]: PASS honest=VERIFIED cheaterCaughtTick=6`.
**Boundary:** this proves *simulation integrity* (outcomes cannot be faked because they are
re-derivable). It does not by itself catch *input-level* cheats (e.g. an aimbot choosing
superhuman-but-legal inputs) — that is a separate statistical layer.
**Why UE5 can't:** a server cannot re-derive a client's float physics bit-for-bit, so "provable
anti-cheat by re-simulation" is a category it is disqualified from.

### PROOF 4 — reproducible counterfactual (what-if fork)
**Demonstrates:** the FK1 fork records the canonical match, then rewinds to tick 8 and injects two
different single-input mutations (a physics shove, a gameplay ability) → a 3-node timeline tree. All
three timelines share ticks [0, 8) byte-for-byte, diverge at exactly tick 8, and reach three distinct
pinned whole-timeline digests — and re-running the fork reproduces them bit-for-bit.
**Expected:** `PROOF 4 [reproducible counterfactual]: PASS forkTick=8 digest=0x562f8800b4577f5a
(branchA=0x88faa2ebe6383db9 branchB=0x07fe638fd0bcf831)`.
**Why UE5 can't:** with no reproducible re-derivation of a float sim (and no inverse), a bit-exact
what-if timeline is structurally impossible — the best a float engine can store is interpolated
transforms.

### PROOF 5 — reproducible procedural generation
**Demonstrates:** the PCG pipeline (`Generate` = scatter → mask → transform → prune) produces a
byte-identical instance field from a seed alone; the same seed re-generates the identical field
(`0x5194e133e4e4da0a`), and a different seed produces a different, pinned field
(`0xda657619051660f7`).
**Expected:** `PROOF 5 [reproducible procedural generation]: PASS instances=51 seedA=0x5194e133e4e4da0a
seedB=0xda657619051660f7`.
**Why UE5 can't:** its PCG runtime uses float noise/placement math, so a seeded field is not
bit-reproducible machine-to-machine — two clients cannot generate the same world from the seed alone.

### PROOF 6 — bandwidth (inputs vs state)
**Demonstrates:** for the canonical scene, rollback netcode sends **inputs** (the whole 24-tick command
stream = 48 bytes, once) while a state-sync engine sends a **full world snapshot every tick** (884
bytes/tick × 24 = 21,216 bytes) — a ~442× difference for this scene.
**Expected:** `PROOF 6 [bandwidth inputs<<state]: PASS inputStream=48B stateSync=884B/tick*24=21216B
ratio=442x`.
**Honesty:** this is a *specific-scene* measurement, not a universal ratio — the win scales with world
size and tick rate, and a production state-sync stack would delta-compress. The point is the
*architectural* asymmetry: rollback can send inputs and re-simulate *because* the simulation is
deterministic; a float engine must replicate authoritative state because it cannot.
**Why UE5 can't:** non-deterministic physics forces authoritative state replication; it cannot send
inputs and trust every client to re-simulate the identical result.

## The substrate these proofs exercise (all shipped + golden-gated)

| Proof | Composed core (read-only) |
|---|---|
| 1 | `engine/game/verdict.h` — VD1-VD6 whole-world lockstep + `DigestSnapshot` |
| 2 | `engine/net/session.h` — NS5 `DesyncDetector` / `DigestTrace` |
| 3 | `engine/net/authority_verify.h` — AC1 re-simulation verifier |
| 4 | `engine/replay/fork.h` — FK1 what-if fork (over `replay.h` RP1-6 + NS6 CatchUp) |
| 5 | `engine/pcg/pcg.h` — PCG1-PCG5 deterministic generation |
| 6 | `engine/game/verdict.h` — canonical scene snapshot vs command stream sizing |

Each of those cores has its own pinned-digest test (`verdict_test`, `session_test`,
`authority_verify_test`, `fork_test`, `pcg_test`, …) in the pure-core suite; this binary reuses them as
the *proof surface* rather than re-deriving anything.
