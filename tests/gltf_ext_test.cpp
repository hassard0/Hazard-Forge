// Unit test for the glTF unsupported-extension diagnostics (engine/asset/gltf_ext.h, issue #36).
//
// The fixed bug: a glb that REQUIRES KHR_draco_mesh_compression (the three.js Ferrari and >50% of CC0
// glbs) used to load empty geometry and render "0 instances" with NO error. DiagnoseExtensions() is the
// pure decision the loader now uses to FAIL LOUDLY + actionably. This test exercises that logic directly
// with synthetic inputs — no device, no cgltf, no glb file — so it is a fast standalone pure test:
//   clang++ -std=c++20 -I engine -I tests tests/gltf_ext_test.cpp

#include "asset/gltf_ext.h"

#include <cstdio>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::asset;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS: %s\n", what); }
}
static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

int main() {
    HF_TEST_MAIN_INIT();

    // (1) The Ferrari case: REQUIRES Draco -> FATAL, actionable (names the ext, the N/M, the path, issue #36).
    {
        ExtDiagnostic d = DiagnoseExtensions({"KHR_draco_mesh_compression"}, 51, 51, "ferrari.glb");
        check(!d.fatal.empty(), "gltf-ext: required Draco is FATAL (no longer silent)");
        check(contains(d.fatal, "KHR_draco_mesh_compression"), "gltf-ext: fatal names the extension");
        check(contains(d.fatal, "51/51"), "gltf-ext: fatal reports compressed primitive count");
        check(contains(d.fatal, "ferrari.glb"), "gltf-ext: fatal names the file");
        check(contains(d.fatal, "#36"), "gltf-ext: fatal points at the actionable issue");
        check(d.warning.empty(), "gltf-ext: fatal dominates (no separate warning)");
    }

    // (2) A normal Khronos asset (no required compression, no Draco prims) -> both empty -> loads silently.
    {
        ExtDiagnostic d = DiagnoseExtensions({}, 0, 12, "Fox.glb");
        check(d.fatal.empty() && d.warning.empty(), "gltf-ext: a normal file produces NO diagnostic (loads silently)");
    }

    // (3) A harmless REQUIRED material extension must NOT false-alarm.
    {
        ExtDiagnostic d = DiagnoseExtensions({"KHR_materials_unlit"}, 0, 3, "unlit.glb");
        check(d.fatal.empty() && d.warning.empty(), "gltf-ext: a required MATERIAL extension is not a false alarm");
    }

    // (4) Draco USED but not REQUIRED -> non-fatal WARNING (a fallback may render), load proceeds.
    {
        ExtDiagnostic d = DiagnoseExtensions({}, 3, 10, "partial.glb");
        check(d.fatal.empty(), "gltf-ext: Draco used-but-not-required is NOT fatal (fallback may render)");
        check(!d.warning.empty() && contains(d.warning, "3/10"), "gltf-ext: used-Draco emits an actionable warning");
    }

    // (5) EXT_meshopt_compression required -> also FATAL (the modern compression alternative).
    {
        ExtDiagnostic d = DiagnoseExtensions({"EXT_meshopt_compression"}, 0, 8, "meshopt.glb");
        check(!d.fatal.empty() && contains(d.fatal, "EXT_meshopt_compression"), "gltf-ext: required meshopt is FATAL");
    }

    // (6) The predicate is precise.
    {
        check(IsUnsupportedGeometryExt("KHR_draco_mesh_compression"), "gltf-ext: Draco is flagged unsupported");
        check(!IsUnsupportedGeometryExt("KHR_texture_transform"), "gltf-ext: a texture extension is NOT flagged");
    }

    if (g_fail == 0) std::printf("gltf_ext_test: ALL PASS\n");
    else             std::printf("gltf_ext_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
