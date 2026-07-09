// benchmarks/agent_trial.cpp — the AGENT-FEATURE-TRIAL harness (Slice AX2, FLAGSHIP #31, and the
// BEAT_UE5_PLAN Phase-3 #4 deliverable).
//
// A runnable micro-trial that DEMONSTRATES + MEASURES the agent-dev loop THIS ENTIRE 30-slice
// autonomous run embodies: author -> build -> run-a-flag -> byte-compare-a-golden = PASS. It is a
// self-contained, pure-CPU ctest target (hf_agent_trial), so verify.ps1 runs it alongside the pure
// suite.
//
// HONEST SCOPE (read this): this measures the LOOP MECHANICS, not a live LLM race. It does not spawn
// a model and does not rebuild the tree. What it proves is the property that makes HF agent-developable
// and UE5 not: a byte-identical golden is a MACHINE-CHECKABLE ORACLE — an agent knows whether its change
// is correct with NO human and NO GUI. The trial scripts a tiny "feature" over the REAL frozen
// hf_core scene pipeline (scene::CanonicalizeSceneText) and shows the oracle:
//   (1) REJECTS the incomplete feature (RED: the before-spec's canonical bytes differ from the golden),
//   (2) ACCEPTS the finished feature   (GREEN: the after-spec's canonical bytes equal the golden),
//   (3) is DETERMINISTIC (two runs of the finished feature are byte-identical), and
//   (4) is DISCRIMINATING (RED != GREEN — the oracle is meaningful, not vacuous).
// It reports the loop step count + the wall-clock of the run+compare oracle (the "verify latency" an
// agent pays per iteration). Exits 0 IFF the loop behaves correctly.
//
// WHY UE5 CANNOT RUN THIS LOOP (factual, structural): its authoritative simulation is float and
// non-deterministic, so there is no byte-identical golden to compare against; and its authoring is a
// GUI-bound editor with no headless authored-scene->verify path. HF is headless + deterministic by
// construction — the two properties this trial exercises.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

#include "ecs/ecs.h"
#include "scene/scene_io.h"
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// The stand-in resources the scene spec refers to by name (opaque pointers, never dereferenced —
// the scene_io contract). A real agent registers real GPU resources; the loop mechanics are identical.
static scene::SceneResources MakeResources() {
    scene::SceneResources res;
    res.AddMesh("plane",  reinterpret_cast<scene::Mesh*>(0x1000));
    res.AddMesh("sphere", reinterpret_cast<scene::Mesh*>(0x2000));
    res.AddTexture("checker",     reinterpret_cast<rhi::ITexture*>(0x3000));
    res.AddTexture("flat_normal", reinterpret_cast<rhi::ITexture*>(0x4000));
    return res;
}

// The "feature" the trial adds: a second entity (a metallic sphere) atop the ground plane. The
// BEFORE spec omits it (the incomplete change); the AFTER spec includes it (the finished change).
static const char* kSpecBefore =
    "[ { \"mesh\": \"plane\", \"baseColor\": \"checker\", \"metallic\": 0, \"roughness\": 0.9,"
    "    \"position\": [0, -0.5, 0], \"scale\": [8, 1, 8] } ]";

static const char* kSpecAfter =
    "[ { \"mesh\": \"plane\", \"baseColor\": \"checker\", \"metallic\": 0, \"roughness\": 0.9,"
    "    \"position\": [0, -0.5, 0], \"scale\": [8, 1, 8] },"
    "  { \"mesh\": \"sphere\", \"baseColor\": \"checker\", \"normalMap\": \"flat_normal\","
    "    \"metallic\": 1, \"roughness\": 0.2, \"position\": [1.5, 0.5, 0] } ]";

// Canonicalize a spec through the FROZEN hf_core pipeline (LoadScene -> DumpScene). This is the
// "run a headless flag" step of the loop, in-process.
static std::string Canonicalize(const char* spec) {
    ecs::Registry reg;
    scene::SceneResources res = MakeResources();
    return scene::CanonicalizeSceneText(spec, reg, res);
}

int main() {
    HF_TEST_MAIN_INIT();
    std::printf("=== Hazard Forge — agent-feature-trial (measures the agent-dev LOOP mechanics) ===\n");

    // THE GOLDEN: the canonical bytes of the FINISHED feature — the artifact an agent verifies against.
    // (In the real loop this is a committed file; here we generate the target once, deterministically.)
    const std::string golden = Canonicalize(kSpecAfter);
    check(!golden.empty(), "golden target is non-empty");

    int loopSteps = 0;
    const auto t0 = std::chrono::steady_clock::now();

    // STEP 1 (RED): run the loop on the INCOMPLETE feature; the oracle must REJECT it (bytes differ).
    ++loopSteps;
    const std::string beforeOut = Canonicalize(kSpecBefore);
    const bool redRejected = (beforeOut != golden);
    check(redRejected, "RED: oracle rejects the incomplete feature (before-bytes != golden)");

    // STEP 2 (GREEN): run the loop on the FINISHED feature; the oracle must ACCEPT it (bytes equal).
    ++loopSteps;
    const std::string afterOut = Canonicalize(kSpecAfter);
    const bool greenAccepted = (afterOut == golden);
    check(greenAccepted, "GREEN: oracle accepts the finished feature (after-bytes == golden)");

    const auto t1 = std::chrono::steady_clock::now();

    // STEP 3 (determinism): the finished feature canonicalizes byte-identically twice — the oracle is
    // a pure function of the input, so the pass/fail signal is stable across runs + platforms.
    const bool deterministic = (Canonicalize(kSpecAfter) == afterOut);
    check(deterministic, "oracle is deterministic (two runs of the finished feature byte-identical)");

    // STEP 4 (discriminating): RED != GREEN — the oracle is meaningful, not vacuously passing.
    check(beforeOut != afterOut, "oracle is discriminating (RED bytes != GREEN bytes)");

    const auto verifyUs =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // A machine-checkable summary line + the measured loop metrics.
    const bool loopPass = redRejected && greenAccepted && deterministic && (beforeOut != afterOut);
    std::printf("agent-trial: {loopSteps:%d, redRejected:%s, greenAccepted:%s, deterministic:%s, "
                "discriminating:%s}\n",
                loopSteps, redRejected ? "true" : "false", greenAccepted ? "true" : "false",
                deterministic ? "true" : "false", (beforeOut != afterOut) ? "true" : "false");
    std::printf("agent-trial: verifyOracleWallClockUs=%lld goldenBytes=%zu "
                "(byte-compare golden == the machine-checkable pass/fail; no human, no GUI, no LLM in "
                "this measurement)\n",
                (long long)verifyUs, golden.size());
    std::printf("=== %s ===\n",
                (loopPass && g_fail == 0) ? "AGENT-DEV LOOP MECHANICS PROVEN" : "AGENT-TRIAL FAILED");

    return (loopPass && g_fail == 0) ? 0 : 1;
}
