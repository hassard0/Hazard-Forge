# Beat UE5 — The Strategy

*How a one-author deterministic engine beats a thousand-engineer incumbent: not by matching it,
but by winning the categories it structurally cannot enter, and being good-enough everywhere else.
Companion to `GAP_CLOSING_ROADMAP.md` (the parity work) — this document is the offense.*

---

## 1. Where we stand (the honest review)

**Built and verified (as of master `a67e7e3`):**

| Metric | Value |
|---|---|
| Commits | 1,299 |
| Golden-verified render bakes (Metal-baked, cross-vendor) | 327 |
| Test suites (pinned cross-platform digests) | 179 |
| Engine source | ~71k LOC across 277 files, 282 shaders |
| Deterministic-sim flagships | rigid, cloth, fluid, granular, fracture, particles, vehicles, ragdoll, boids, persist/sleep + 3 two-way couplings (rigid↔fluid, rigid↔grain, grain↔fluid) |
| Rendering flagships | Nanite-class virtual geometry (meshlet/cluster-LOD/vis-buffer/SW-raster), Lumen-class DDGI, clipmap VSM, HW ray tracing (both vendors), substrate materials, clustered/froxel lighting, TAA, virtual texturing |
| Tooling flagships | node-graph VM + live editor, sequencer, UMG-class UI, profiler w/ scrub capture, asset compiler + live hot-reload, FBX/OBJ/USD/glTF+Draco import (clean-room) |
| Determinism proof | bit-identical goldens on **3 platforms** (Windows/MSVC, macOS/Apple-clang, Linux/gcc) |
| Netcode | lockstep + rollback proven over *every* sim family |
| GitHub backlog | cleared (1 open, environment-blocked) |

**Shipped since this review (the moat, productized):** ✅ **a real game** — a deterministic rollback-physics
knockout duel (`--game1-duel`, golden `game1_duel`) composing the whole moat stack; ✅ **provable anti-cheat**
by server re-simulation (`--ac1-verify`, golden `ac1_verify`); ✅ **what-if fork replay** — bit-exact
counterfactual timelines (`--fk1-fork`, golden `fk1_fork`); ✅ **real UDP transport** for the rollback netcode
(`--nw1-udp`, golden `nw1_udp`); ✅ **runnable benchmarks** — the six "proofs UE5 loses" as a ctest binary
(`benchmarks/`, `hf_moat_proofs`). Phase 0 (the game), the first Rollback-Kit step (real UDP), and Phase 3 (the
benchmark suite) are now on the board (see §3).

**Honest weaknesses (still open):** the game is a **headless deterministic match simulation** — no live input
window / interactive play loop yet; content proven at fixture scale (~144 instances, 8 lights, hand-authored
LODs), never a Sponza; the interactive editor is 60% built (UI plumbing missing); real UDP ships but the
committed gate is a same-process loopback (the two-process run is a documented script); no TSR upscaling; no
mobile/console; zero external users; one maintainer.

**The strategic read:** the weaknesses are all *productization*; the strengths are all *structural*.
That asymmetry is the plan.

## 2. What "beat" means (and doesn't)

**It cannot mean** out-UE5-ing UE5: their ecosystem, marketplace, content pipeline, platform
reach, and two decades of shipped-title hardening are a war of attrition we lose by definition.
Fighting there is how challenger engines die.

**It means winning categories where UE5 is structurally disqualified** — not behind, *disqualified*,
because following us would require rearchitecting their float core and GUI-first workflow:

1. **Determinism.** Chaos physics, float math, nondeterministic task scheduling — UE5 cannot make
   two machines produce bit-identical simulation, and cannot retrofit it without breaking every
   shipped project. HF has it proven across 14+ sim families on 3 platforms. This decides entire
   genres: rollback fighters, lockstep RTS, competitive esports (replay integrity, anti-cheat by
   re-simulation, provable fairness), server-authoritative physics at a fraction of the bandwidth
   (send inputs, not state).
2. **Agent-native development (AX).** The next decade of game development is increasingly
   agent-driven — and UE5 is the worst possible substrate for it: GUI-first, binary assets
   (blueprints/umaps), hour-long builds, nondeterministic output an agent can't verify. HF is the
   opposite by construction: headless-everything, text-first APIs, deterministic goldens as the
   verification gate. **This engine's own 1,299 commits were built by an autonomous agent loop** —
   the engine is its own existence proof, and the Agent SDK (flagship DX1-DX6, shipped) is the
   beachhead product. UE5 cannot enter this category without becoming a different product.
3. **Verifiability as a feature.** 327 pixel-goldens and 179 pinned digests make the engine
   *CI-native*: every render, every sim tick, provably reproducible. For studios: regression-proof
   content pipelines. For research/serious-games: reproducible science. UE5 has nothing comparable.

**And being good-enough everywhere else** — the `GAP_CLOSING_ROADMAP.md` tiers exist so the moat
verticals are never disqualified by a missing basic (an editor, a real scene, upscaling).

The playbook is classic disruption: dominate beachheads where the incumbent's strengths are
irrelevant, expand along adjacencies, let the incumbent's architecture prevent pursuit.

## 3. The plan

### Phase 0 — Ship a game (the credibility gate) — ✅ FIRST GAME SHIPPED
An engine without a shipped game beats nothing. **Shipped:** GAME1 (`engine/game/duel.h`, golden
`game1_duel`, `--game1-duel`) — a **small, complete, deterministic rollback-physics knockout duel** that is
impossible-in-UE5 by construction: two rigid-body avatars, a shove ability (GAS1 cost/cooldown), a stun tag
(GT1), gameplay cues (GC1), best-of-3 rounds, ring-out wins. It composes the whole moat stack (bit-exact
`fpx` physics + whole-world lockstep/rollback + FK1 what-if fork + AC1 anti-cheat) into one deliverable with a
pinned match digest `0x78123003c3a55a37`. Built agent-first, golden-gated, on the existing fpx/verdict
substrate + the GAS1/GT1/GC1 gameplay framework.
*What's shipped vs the original deliverable:* the game **is** the "this match re-simulates bit-identically"
proof (composed with `benchmarks/`), and it is provably fair, fork-able, and rollback-replayable. *Still to
do:* a **live interactive input window / itch.io-shippable playable binary** — GAME1 is a headless
deterministic match simulation today. **The proof is a game, not a test suite** — and it now exists.

### Phase 1 — Finish the weapon (parity table stakes)
Execute `GAP_CLOSING_ROADMAP.md` Tiers 1–2 (already specced): **ED1-ED6** complete the interactive
editor (60% → 100%, mostly UI plumbing over existing ops); **SC1** the real-Sponza bake (the single
highest-credibility artifact); **SC2-SC6** auto-LOD, many-light, scatter-at-scale; **UP1-UP3**
deterministic TSR. Everything stays golden-verified.
*Exit criterion:* a newcomer downloads, launches an editor, loads Sponza, flies around, edits, saves.

### Phase 2 — Productize the two moats
- **The Rollback Kit** ("GGPO for a whole engine"): NW1-NW4 real UDP transport + dedicated-server
  harness over the existing deterministic substrate. Every sim family becomes rollback-multiplayer
  out of the box — fighters, RTS, sports, racing (vehicles already lockstep-proven). No other engine
  ships this. **NW1 shipped (the first step):** real UDP transport (`engine/net/udp_transport.h`, golden
  `nw1_udp`, `--nw1-udp`) — Winsock/BSD datagrams behind the same transport interface, so two peers converge
  to the identical match digest over real `127.0.0.1` UDP despite injected reorder/duplicate/loss. Real
  sockets are the one documented nondeterministic-I/O exception; the outcome digest stays bit-exact because
  the rollback layer (NS3-6) tolerates arrival order + loss-with-resend. *Committed gate:* a same-process
  two-socket loopback (the true two-process run is `benchmarks/udp_two_process.md`); NW2-NW4 (a
  dedicated-server harness) remain.
- **The Agent SDK** (flagship DX, shipped — now market it as the product): the headless, golden-gated,
  text-first dev loop packaged for AI-agent game development. Docs positioned at agents, not just
  humans: machine-readable capability maps (CAPABILITIES.md exists), deterministic verification
  hooks, one-command proof runs. First-mover in a category with no incumbent.

### Phase 3 — Pick fights we win (reproducible benchmarks as marketing) — ✅ BENCHMARK SUITE SHIPPED
Publish head-to-head, *runnable* comparisons UE5 structurally loses — in this repo's own ethos,
every claim ships with the script that proves it. **Shipped:** `benchmarks/` — a runnable proof binary
(`moat_proofs.cpp`, ctest target `hf_moat_proofs`, run by `scripts/verify.ps1` + `benchmarks/run_all.ps1` /
`run_all.sh`) that asserts **six** pinned deterministic outcomes and exits 0 iff all pass, each stating the
structural reason UE5 is disqualified. It is a runnable proof binary, not a golden image.
1. ✅ **The desync test** *(proof 1 cross-platform determinism + proof 2 desync detection)*: identical inputs,
   two peers — HF is bit-identical (`DigestSnapshot 0x76a37a56d256c401`); `DesyncDetector` catches an injected
   divergence at the exact tick. UE5 Chaos diverges.
2. ✅ **The replay test** *(proof 4 reproducible counterfactual + `DETERMINISM_THREE_PLATFORMS.md`)*: a recorded
   match re-simulates bit-for-bit; the same pure-core digests already reproduce on Windows/Mac/Linux (135/135
   on Linux/gcc). *(Full three-machine same-hash demo needs three machines — each leg reproducible.)*
   Also shipped beyond the original plan: **provable anti-cheat** (proof 3) and **reproducible procgen** (proof 5).
3. ✅ **The bandwidth test** *(proof 6 inputs << state)*: server-authoritative physics — HF sends inputs
   (48 B for the whole 24-tick stream) vs a state-sync engine's full snapshot every tick (884 B/tick × 24) —
   ~442× for this scene. *(A specific-scene measurement, not a universal ratio.)*
4. **The agent trial:** an AI agent implements + verifies a gameplay feature end-to-end, headless —
   wall-clock HF vs UE5. *(Not yet in the benchmark binary — future.)*
5. **The CI test:** cold clone → fully verified build (all goldens green) — minutes vs hours.
   *(Not yet packaged as a benchmark — future.)*

### Phase 4 — Ecosystem seeding (the long game)
Permissive license vs UE5's 5% royalty; the getting-started path (exists) hardened by external
feedback; the Phase-0 game open-sourced as the flagship sample; target the first 10 real users
*in the moat verticals* (fighting-game community, RTS devs, esports infra, agent-tooling builders) —
not generic gamedev. Each vertical win is a case study the next one cites.

## 4. Sequencing & success metrics

| Phase | Runs | Done when |
|---|---|---|
| P0 game | first; everything serves it | a stranger plays it online; a replay re-simulates bit-exact on 3 OSes — 🟡 **first game shipped** (`game1_duel`, deterministic + replayable + provably fair); live-input/online play still to do |
| P1 parity | interleaved with P0 (the game *needs* the editor + real content) | Sponza in a launchable editor at 60fps; TSR shipping |
| P2 kits | after P0 proves the loop | `create-hf-game --rollback` works; Agent SDK has an external agent shipping a feature — 🟡 **real UDP transport shipped** (`nw1_udp`); dedicated-server harness (NW2-4) still to do |
| P3 benchmarks | as each proof becomes runnable | all 5 published with scripts; nobody refutes them — 🟡 **suite shipped** (`benchmarks/`, `hf_moat_proofs`): the desync, replay, and bandwidth proofs run; agent-trial + CI-time proofs still to do |
| P4 ecosystem | continuous after P0 | 10 external users in moat verticals; first external contribution merged |

## 5. What we refuse to do

- Chase mobile/console before an SDK-bearing partner exists (env-blocked; stubs would be lies).
- Chase photoreal-content parity (Megascans is capital, not code).
- Dilute determinism for convenience features — the moat is the strategy; every new system ships
  bit-exact or clearly marked as the float/render exception layer.
- Claim victories without runnable proof. Every "beats UE5" statement in this plan must ship as a
  script in `benchmarks/`.

---

**The one-sentence strategy:** *ship a game UE5 cannot make, wrap the engine that made it into two
products UE5 cannot build (the Rollback Kit and the Agent SDK), prove both with benchmarks UE5
cannot pass, and recruit the verticals that need exactly that.*
