// Unit test for the machine-readable CAPABILITY MANIFEST (engine/agent/capability_manifest.h, Slice
// AX2, FLAGSHIP #31). Build the manifest and assert:
//   * it is DETERMINISTIC (two calls byte-identical),
//   * the JSON is WELL-FORMED (round-trips through the vendored json.h),
//   * the shape is right (schemaVersion==1, engine, verifyContract{loop,oracle,...}, moatProperties,
//     groups[], groupCount/capabilityCount agree with the walked arrays, contentHash present),
//   * every capability entry has a non-empty name/flag/golden/moat/description, and moat is drawn from
//     the closed moatProperties vocabulary,
//   * THE MANIFEST DOESN'T LIE: every cited golden name actually appears in scripts/verify.ps1 (the
//     authoritative golden ledger), verified by reading the file passed as HF_VERIFY_PS1.
//
// Pure C++ (hf_core / header-only), ASan-eligible. No backend symbols.
#include "agent/capability_manifest.h"

#include "json/json.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// --- Minimal typed accessors over json.h (mirrors introspect_test's helpers) ---------------------
static const json_value_s* MemberOf(const json_object_s* obj, const char* key) {
    if (!obj) return nullptr;
    for (const json_object_element_s* el = obj->start; el; el = el->next)
        if (el->name && std::strcmp(el->name->string, key) == 0) return el->value;
    return nullptr;
}
static const json_object_s* AsObject(const json_value_s* v) {
    return (v && v->type == json_type_object) ? static_cast<const json_object_s*>(v->payload) : nullptr;
}
static const json_array_s* AsArray(const json_value_s* v) {
    return (v && v->type == json_type_array) ? static_cast<const json_array_s*>(v->payload) : nullptr;
}
static std::string AsString(const json_value_s* v) {
    if (!v || v->type != json_type_string) return {};
    const json_string_s* s = static_cast<const json_string_s*>(v->payload);
    return std::string(s->string, s->string_size);
}
static long AsInt(const json_value_s* v, long fallback = -1) {
    if (!v || v->type != json_type_number) return fallback;
    return std::strtol(static_cast<const json_number_s*>(v->payload)->number, nullptr, 10);
}

int main() {
    HF_TEST_MAIN_INIT();

    const std::string json = agent::BuildCapabilityManifest();

    // --- Determinism: a second call must be byte-identical. --------------------------------------
    check(json == agent::BuildCapabilityManifest(),
          "BuildCapabilityManifest is deterministic (byte-identical across runs)");

    // --- Parse (well-formed). --------------------------------------------------------------------
    json_parse_result_s err{};
    json_value_s* root = json_parse_ex(json.data(), json.size(), json_parse_flags_default,
                                       nullptr, nullptr, &err);
    check(root != nullptr, "manifest parses as JSON");
    const json_object_s* top = AsObject(root);
    check(top != nullptr, "top-level value is an object");

    check(AsInt(MemberOf(top, "schemaVersion")) == 1, "schemaVersion == 1");
    check(AsString(MemberOf(top, "engine")) == "Hazard Forge", "engine is Hazard Forge");
    check(!AsString(MemberOf(top, "thesis")).empty(), "thesis present");
    check(!AsString(MemberOf(top, "contentHash")).empty(), "contentHash present");

    // --- verifyContract: the machine-readable agent-dev LOOP contract. ---------------------------
    const json_object_s* vc = AsObject(MemberOf(top, "verifyContract"));
    check(vc != nullptr, "verifyContract object present");
    const json_array_s* loop = AsArray(MemberOf(vc, "loop"));
    check(loop != nullptr && loop->length == 4, "verifyContract.loop has 4 steps");
    if (loop && loop->length == 4) {
        const json_array_element_s* s0 = loop->start;
        const json_array_element_s* s1 = s0->next;
        const json_array_element_s* s2 = s1->next;
        const json_array_element_s* s3 = s2->next;
        check(AsString(s0->value) == "author", "loop[0] == author");
        check(AsString(s1->value) == "build", "loop[1] == build");
        check(AsString(s2->value) == "run-flag", "loop[2] == run-flag");
        check(AsString(s3->value) == "byte-compare-golden", "loop[3] == byte-compare-golden");
    }
    check(!AsString(MemberOf(vc, "oracle")).empty(), "verifyContract.oracle present");
    check(!AsString(MemberOf(vc, "moatVsUe5")).empty(), "verifyContract.moatVsUe5 present");
    check(AsString(MemberOf(vc, "harness")) == "scripts/verify.ps1", "verifyContract.harness is verify.ps1");

    // --- The closed moat vocabulary. -------------------------------------------------------------
    std::set<std::string> moatVocab;
    const json_array_s* moats = AsArray(MemberOf(top, "moatProperties"));
    check(moats != nullptr && moats->length >= 4, "moatProperties is a non-trivial list");
    if (moats)
        for (const json_array_element_s* el = moats->start; el; el = el->next)
            moatVocab.insert(AsString(el->value));

    // --- Walk groups + capabilities. -------------------------------------------------------------
    const json_array_s* groups = AsArray(MemberOf(top, "groups"));
    check(groups != nullptr, "groups array present");
    long groupCount = 0, capCount = 0;
    std::vector<std::string> citedGoldens;
    std::set<std::string> capNames;
    if (groups) {
        for (const json_array_element_s* ge = groups->start; ge; ge = ge->next) {
            ++groupCount;
            const json_object_s* g = AsObject(ge->value);
            check(g != nullptr, "group is an object");
            check(!AsString(MemberOf(g, "group")).empty(), "group has a non-empty id");
            check(!AsString(MemberOf(g, "description")).empty(), "group has a description");
            const json_array_s* caps = AsArray(MemberOf(g, "capabilities"));
            check(caps != nullptr && caps->length > 0, "group has a non-empty capabilities list");
            if (!caps) continue;
            for (const json_array_element_s* ce = caps->start; ce; ce = ce->next) {
                ++capCount;
                const json_object_s* c = AsObject(ce->value);
                check(c != nullptr, "capability is an object");
                const std::string name   = AsString(MemberOf(c, "name"));
                const std::string flag    = AsString(MemberOf(c, "flag"));
                const std::string golden  = AsString(MemberOf(c, "golden"));
                const std::string moat    = AsString(MemberOf(c, "moat"));
                const std::string desc    = AsString(MemberOf(c, "description"));
                check(!name.empty(), "capability.name non-empty");
                check(flag.rfind("--", 0) == 0, "capability.flag starts with --");
                check(!golden.empty(), "capability.golden non-empty");
                check(!desc.empty(), "capability.description non-empty");
                check(moatVocab.count(moat) == 1, "capability.moat is in the moatProperties vocabulary");
                check(capNames.insert(name).second, "capability.name is unique");
                citedGoldens.push_back(golden);
            }
        }
    }
    check(AsInt(MemberOf(top, "groupCount")) == groupCount, "groupCount matches walked groups");
    check(AsInt(MemberOf(top, "capabilityCount")) == capCount, "capabilityCount matches walked capabilities");
    check((long)agent::CapabilityCount() == capCount, "CapabilityCount() agrees with the emitted JSON");
    check((long)agent::CapabilityGroupCount() == groupCount, "CapabilityGroupCount() agrees with the emitted JSON");
    check(capCount >= 24, "manifest enumerates a representative (>=24) set of capabilities");

    // --- THE MANIFEST DOESN'T LIE: every cited golden name appears in scripts/verify.ps1. --------
    // HF_VERIFY_PS1 is injected by CMake as the absolute path to the authoritative golden ledger.
#ifdef HF_VERIFY_PS1
    {
        std::ifstream f(HF_VERIFY_PS1, std::ios::binary);
        check(f.good(), "scripts/verify.ps1 is readable (HF_VERIFY_PS1)");
        std::string verify((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        check(!verify.empty(), "scripts/verify.ps1 is non-empty");
        // Cross-check against the header-side list too (belt + suspenders — the JSON must cite exactly
        // what the registry does).
        check(agent::CitedGoldenNames().size() == citedGoldens.size(),
              "CitedGoldenNames() count matches the JSON's cited goldens");
        int missing = 0;
        for (const std::string& gname : citedGoldens) {
            if (verify.find(gname) == std::string::npos) {
                std::printf("FAIL: cited golden '%s' NOT found in scripts/verify.ps1\n", gname.c_str());
                ++missing;
            }
        }
        check(missing == 0, "every cited golden name exists in scripts/verify.ps1 (manifest doesn't lie)");
        std::printf("capability-manifest: %ld capabilities in %ld groups; all cited goldens present in verify.ps1\n",
                    capCount, groupCount);
    }
#else
    std::printf("WARN: HF_VERIFY_PS1 not defined; skipping the manifest-doesn't-lie cross-check\n");
#endif

    if (g_fail == 0) std::printf("capability_manifest_test: ALL CHECKS PASS\n");
    else             std::printf("capability_manifest_test: %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
