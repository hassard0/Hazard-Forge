// Hazard Forge — runtime material -> SPIR-V via a dxc SUBPROCESS (Slice AW). See runtime_compile.h.
//
// Pure host/tooling: GenerateHlsl(graph) -> temp .hlsl -> invoke the SAME dxc.exe (+ same flags) the
// build uses -> temp .spv -> read words. Byte-identical to the build-time SPIR-V by construction. No
// backend (vk*/MTL*) symbols. Fails SAFE: nullopt + error on any problem; never throws out.
#include "material/runtime_compile.h"

#include "material/codegen.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace hf::material {
namespace {

namespace fs = std::filesystem;

// True if `p` names an existing regular file (a usable dxc).
bool IsFile(const std::string& p) {
    if (p.empty()) return false;
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

// Read a whole file into a string (best-effort; empty on failure).
std::string SlurpText(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Read a .spv file into 32-bit words (nullopt unless it is a non-empty multiple of 4 bytes).
std::optional<std::vector<uint32_t>> ReadSpirv(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return std::nullopt;
    std::streamsize size = f.tellg();
    if (size <= 0 || (size % 4) != 0) return std::nullopt;
    f.seekg(0);
    std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
    if (!f.read(reinterpret_cast<char*>(words.data()), size)) return std::nullopt;
    return words;
}

}  // namespace

std::string ResolveDxcPath() {
    // 1. The exact dxc CMake resolved for the build (CompileShaders.cmake -> DXC_EXECUTABLE), baked
    //    in as HF_DXC_PATH. This is the SAME compiler the build used -> byte-identical SPIR-V.
#ifdef HF_DXC_PATH
    if (IsFile(HF_DXC_PATH)) return HF_DXC_PATH;
#endif
    // 2. Fall back to the same hint locations CompileShaders.cmake searches (Vulkan SDK, WinGet DXC),
    //    then a bare PATH lookup. Mirrors the build's discovery order so the fallback stays faithful.
    std::vector<std::string> candidates;
    if (const char* vk = std::getenv("VULKAN_SDK")) {
        candidates.push_back(std::string(vk) + "/Bin/dxc.exe");
        candidates.push_back(std::string(vk) + "/Bin/dxc");
    }
    if (const char* lad = std::getenv("LOCALAPPDATA")) {
        candidates.push_back(std::string(lad) +
            "/Microsoft/WinGet/Packages/"
            "Microsoft.DirectX.ShaderCompiler_Microsoft.Winget.Source_8wekyb3d8bbwe/bin/x64/dxc.exe");
    }
    for (const std::string& c : candidates)
        if (IsFile(c)) return c;
    // 3. Last resort: trust PATH ("dxc"). May be a Windows-SDK dxc lacking -spirv; the caller surfaces
    //    the compile error if so.
    return "dxc";
}

std::optional<std::vector<uint32_t>> CompileGraphToSpirv(const Graph& graph, std::string* errorOut) {
    auto fail = [&](const std::string& msg) -> std::optional<std::vector<uint32_t>> {
        if (errorOut) *errorOut = msg;
        return std::nullopt;
    };

    // 1. Codegen the HLSL (reuse the AV codegen). An invalid graph yields a "// ERROR:" marker.
    std::string hlsl = GenerateHlsl(graph);
    if (hlsl.rfind("// ERROR", 0) == 0) return fail(hlsl);

    // 2. Write the .hlsl into shaders/generated/ so its `#include "../pbr_core.hlsli"` resolves the
    //    same way the build-time generated shader's does (one dir up from shaders/generated/). A
    //    unique temp name avoids clobbering the committed mat_showcase.frag.hlsl.
    std::string genDir;
#ifdef HF_SHADER_GEN_DIR
    genDir = HF_SHADER_GEN_DIR;
#endif
    if (genDir.empty()) return fail("runtime_compile: HF_SHADER_GEN_DIR not configured");

    std::error_code ec;
    fs::path dir(genDir);
    if (!fs::is_directory(dir, ec)) return fail("runtime_compile: shader-gen dir missing: " + genDir);

    // A per-process-unique stem keeps concurrent / repeated compiles from racing on the same files.
    static unsigned counter = 0;
    std::string stem = "hf_live_" + std::to_string(
        static_cast<unsigned long long>(std::hash<std::string>{}(hlsl)) ^ (++counter));
    fs::path hlslPath = dir / (stem + ".frag.hlsl");
    fs::path spvPath = dir / (stem + ".frag.hlsl.spv");

    {
        std::ofstream out(hlslPath, std::ios::binary | std::ios::trunc);
        if (!out) return fail("runtime_compile: cannot write temp HLSL: " + hlslPath.string());
        out << hlsl;
    }

    // 3. Invoke the SAME dxc + SAME fragment profile/flags as cmake/CompileShaders.cmake:
    //      dxc -spirv -T ps_6_0 -E main -fspv-target-env=vulkan1.3 <in> -Fo <out>
    //    stderr is captured to a temp log so a compile error is surfaced (not swallowed).
    std::string dxc = ResolveDxcPath();
    fs::path logPath = dir / (stem + ".dxc.log");

    // Build a quoted command. On Windows, std::system runs the string via cmd.exe; wrapping the WHOLE
    // command in an extra pair of quotes lets cmd handle paths-with-spaces in both the exe + args.
    std::string cmd;
    cmd += "\"";
    cmd += "\"" + dxc + "\"";
    cmd += " -spirv -T ps_6_0 -E main -fspv-target-env=vulkan1.3";
    cmd += " \"" + hlslPath.string() + "\"";
    cmd += " -Fo \"" + spvPath.string() + "\"";
    cmd += " 2> \"" + logPath.string() + "\"";
    cmd += "\"";

    int rc = std::system(cmd.c_str());

    std::optional<std::vector<uint32_t>> words;
    std::string err;
    if (rc != 0) {
        err = "dxc exited " + std::to_string(rc) + " (" + dxc + ")\n" + SlurpText(logPath);
    } else {
        words = ReadSpirv(spvPath);
        if (!words) err = "dxc produced no/invalid SPIR-V at " + spvPath.string() + "\n" + SlurpText(logPath);
    }

    // 4. Clean up temp files (best-effort; failure to delete is non-fatal).
    fs::remove(hlslPath, ec);
    fs::remove(spvPath, ec);
    fs::remove(logPath, ec);

    if (!words) return fail(err.empty() ? "runtime_compile: unknown dxc failure" : err);
    return words;
}

}  // namespace hf::material
