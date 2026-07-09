# Agent-Native Hazard Forge

**Thesis:** Hazard Forge is an engine an AI agent can develop **end-to-end** — discover what it can
do, drive every capability headlessly, and verify its own change against a machine-checkable oracle,
with **no human and no GUI in the loop**. This whole 30-slice autonomous run is the existence proof.
Slice AX2 *productizes* that advantage as data (a machine-readable capability manifest) and as a
runnable measurement (the agent-feature-trial).

This is not a marketing claim about "AI features." It is a **structural** property of how the engine
is built, and it is the axis on which HF is uniquely positioned against UE5.

---

## The four structural reasons an agent can develop HF

1. **A deterministic golden oracle.** Every capability is proven by a **byte-identical golden** — a
   committed image (strict zero-differing-pixel, or a float visual-resolve bar) or a text/JSON digest.
   Equal bytes = PASS, any diff = FAIL. An agent therefore knows whether its change is correct
   **without asking a human** — the digest *is* the correctness signal. The determinism goes deep:
   the simulation moat (rigid/cloth/fluid/grain/fracture/ragdoll/…) is Q16.16 fixed-point and
   **bit-identical across CPU, Vulkan, and Metal**, and lockstep/rollback-replayable
   (see `docs/DETERMINISM_THREE_PLATFORMS.md`).

2. **Headless everything.** Each capability runs behind a headless showcase **flag** in
   `samples/hello_triangle` (e.g. `--fpx-render`, `--pbr`, `--nav-path`) that captures a golden and
   exits — no window, no interaction. Pure-CPU logic ships as `ctest` targets. The verify harness
   (`scripts/verify.ps1`) runs all of it start to finish, unattended.

3. **Introspection + authored-scene loop.** An agent can OBSERVE the whole live engine+scene state
   as JSON (`--introspect` → `editor::DescribeEngine`), negotiate a versioned SDK contract
   (`--agent-api` → `editor::DescribeAgentApi`), selectively query it (`--agent-query`), AUTHOR a
   scene from a declarative spec and canonicalize it (`--author-scene`), hot-reload it
   (`--hot-reload`), and record/replay it deterministically (`--replay-verify`,
   `--determinism-stress`). The mutate-then-verify loop is fully headless and text-golden-gated.

4. **A discovery API.** The **capability manifest** (`--capability-manifest` →
   `agent::BuildCapabilityManifest`, `engine/agent/capability_manifest.h`) enumerates the engine's
   major capability families as machine-readable JSON: for each capability, the showcase **flag**
   that exercises it, the **golden** that verifies it, its **moat property**, and a one-line
   description — plus a `verifyContract` block that states the agent-dev loop itself as data.

## The agent-dev loop (the contract)

```
author  →  build  →  run-flag  →  byte-compare-golden   ==  PASS
```

> To add or verify capability **X**: run its flag **F**, byte-compare its golden **G**.
> Equal bytes == PASS, any diff == FAIL. The digest is the oracle.

This contract is emitted verbatim inside the manifest's `verifyContract` object, so an agent reads
the loop and the capability list from the same document.

## Why UE5 cannot be developed the same way (factual, structural)

These are not quality judgments — they are structural facts about the two engines:

| Property                          | Hazard Forge                                   | Unreal Engine 5                                            |
|-----------------------------------|------------------------------------------------|-----------------------------------------------------------|
| Correctness oracle                | Byte-identical golden (image/text digest)      | None — authoritative sim is **float, non-deterministic**  |
| Simulation reproducibility        | Q16.16 fixed-point, bit-exact CPU/Vulkan/Metal | Chaos physics is float; results vary run-to-run/hardware  |
| Authoring path                    | Headless declarative spec → canonical JSON     | **GUI-bound editor**; no headless authored-scene→verify   |
| Verify harness                    | `scripts/verify.ps1`, fully unattended         | Editor-in-the-loop; no equivalent golden gate             |
| Capability discovery              | Machine-readable manifest (this slice)         | Docs + a GUI; not a machine contract                      |

An agent needs a machine-checkable oracle to close the loop on its own. HF's determinism + headless
design provides one; UE5's float simulation and GUI-bound authoring structurally do not.

## Try it

```bash
# Discover the capability surface (deterministic JSON: flag + golden + moat per capability).
hello_triangle --capability-manifest manifest.json

# Negotiate the versioned Agent-SDK contract.
hello_triangle --agent-api agent_api.json

# Observe the whole live engine+scene state.
hello_triangle --introspect scene.json

# Measure the agent-dev loop mechanics (author → run → byte-compare-golden = PASS).
ctest -R hf_agent_trial --output-on-failure
```

## What is real vs. framing (honest scope)

- The capability manifest is **representative, not exhaustive**: it enumerates the *major* capability
  families (dozens of capabilities across six groups), each with a real flag + a real golden that
  exists in `scripts/verify.ps1`. The `capability_manifest_test` cross-checks that **every cited
  golden name actually appears in `verify.ps1`** — the manifest does not lie about what verifies it.
  It is not a line-for-line mirror of all ~377 goldens.
- The **agent-feature-trial** (`benchmarks/agent_trial.cpp`, `ctest hf_agent_trial`) measures the
  **loop mechanics**, not a live LLM race. It does not spawn a model and does not rebuild the tree.
  It scripts a tiny "feature" over the real frozen scene pipeline and proves the byte-compare oracle
  *rejects* an incomplete feature, *accepts* the finished one, is deterministic, and is discriminating
  — then reports the verify-oracle wall-clock. That is the property that makes HF agent-developable;
  the trial exercises it directly.
- The agent-native advantage is a **real, already-shipped capability** (this run demonstrates it).
  AX2 packages and measures it; it is not a new runtime feature.

## Source

- `engine/agent/capability_manifest.h` — the in-code capability registry + deterministic JSON generator.
- `samples/hello_triangle` `--capability-manifest` — the headless emitter + self-check.
- `tests/capability_manifest_test.cpp` — determinism, well-formedness, and the manifest-doesn't-lie proof.
- `benchmarks/agent_trial.cpp` — the agent-feature-trial (loop-mechanics measurement).
- `tests/golden/agent/capability_manifest.json` — the committed byte-golden.
- `scripts/verify.ps1` — the `--capability-manifest` golden block (backend-agnostic, verified once).
