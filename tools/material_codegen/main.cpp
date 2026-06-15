// Hazard Forge — material codegen host tool (Slice AV).
//
// Reads a *.mat.json material graph and writes the generated HLSL fragment shader. Wired into CMake
// (samples/hello_triangle + the metal_headless build via the committed output) so building the engine
// regenerates shaders/generated/mat_<name>.frag.hlsl from assets/materials/<name>.mat.json — the JSON
// is the source of truth. The generated HLSL is committed for diff transparency + so the standalone
// Metal build has it without running this tool.
//
//   material_codegen <in.mat.json> <out.frag.hlsl>
//
// Exits non-zero (and writes nothing) on a load/validation error so a broken graph fails the build
// loudly rather than emitting a silently-broken shader.
#include "material/codegen.h"
#include "material/material_loader.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: material_codegen <in.mat.json> <out.frag.hlsl>\n");
        return 2;
    }
    const char* in = argv[1];
    const char* out = argv[2];

    hf::material::LoadResult lr = hf::material::LoadGraphFromFile(in);
    if (!lr) {
        std::fprintf(stderr, "material_codegen: failed to load '%s': %s\n", in, lr.error.c_str());
        return 1;
    }

    std::string hlsl = hf::material::GenerateHlsl(lr.graph);
    if (hlsl.rfind("// ERROR", 0) == 0) {
        std::fprintf(stderr, "material_codegen: codegen error for '%s': %s\n", in, hlsl.c_str());
        return 1;
    }

    std::FILE* fp = std::fopen(out, "wb");
    if (!fp) {
        std::fprintf(stderr, "material_codegen: could not open '%s' for writing\n", out);
        return 1;
    }
    std::fwrite(hlsl.data(), 1, hlsl.size(), fp);
    std::fclose(fp);
    std::printf("material_codegen: %s -> %s (%zu bytes, material '%s')\n",
                in, out, hlsl.size(), lr.name.c_str());
    return 0;
}
