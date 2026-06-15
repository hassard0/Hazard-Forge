// Hazard Forge — live runtime material authoring controller (Slice AW).
//
// PURE HOST LOGIC above the RHI seam (no vk*/MTL*/Backend symbols). The controller watches a
// `.mat.json` material file (via the Slice-AM FileWatcher) and, on a detected change, reloads the
// graph, compiles it to SPIR-V (via an INJECTED compiler so this stays GPU/dxc-free + deterministic
// in unit tests), and — on success — promotes the new SPIR-V to "active" so the render loop can
// rebuild its material pipeline from it. On ANY failure (malformed JSON, invalid graph, compile
// error) it KEEPS the previous active SPIR-V and records the error: the live loop must fail SAFE,
// never crash, never swap to a broken material.
//
// The controller deliberately holds only the compiled SPIR-V WORDS (std::vector<uint32_t>) — it does
// NOT own or create any GPU pipeline. The owning render loop reads TakeIfChanged() and rebuilds the
// pipeline from the words through the EXISTING rhi::CreateShaderModule path (which already accepts
// in-memory SPIR-V), so no new RHI seam is introduced.
#pragma once

#include "material/shader_graph.h"
#include "runtime/hot_reload.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace hf::material {

// Compile a validated graph to SPIR-V words, or return nullopt + an error message. The live loop
// injects either the real dxc-subprocess compiler (runtime_compile.h CompileGraphToSpirv) or, in
// unit tests, a deterministic stub. Returning nullopt MUST leave the previous material active.
using CompileFn =
    std::function<std::optional<std::vector<uint32_t>>(const Graph&, std::string* errorOut)>;

// The outcome of a single reload attempt (returned by Reload / on a watched-file change).
enum class ReloadStatus {
    NoChange,   // the watched file did not change since the last poll (Poll-driven path only).
    Swapped,    // reload + compile succeeded; the active SPIR-V was replaced.
    Kept,       // reload OR compile failed; the previous active SPIR-V was KEPT (fail-safe).
};

class LiveMaterial {
public:
    // `path` is the active .mat.json; `compile` is the injected compiler; `watcher` provides the
    // mtime poll (tests inject a fake stat). The ctor does NOT load anything — call LoadInitial()
    // once to establish the baseline active material before driving the live loop.
    LiveMaterial(std::string path, CompileFn compile,
                 runtime::FileWatcher watcher = runtime::FileWatcher{})
        : path_(std::move(path)), compile_(std::move(compile)), watcher_(std::move(watcher)) {}

    // Load + compile the initial material and arm the file watcher on `path_`. Returns true on
    // success (active SPIR-V is set). On failure returns false and lastError() explains why; the
    // active SPIR-V stays empty (the caller should treat that as fatal for the FIRST load only).
    bool LoadInitial();

    // Force a reload of the CURRENT path (no watcher poll): reload graph -> compile -> swap-or-keep.
    // Used by the hotswap dry-run to drive a deterministic A->B swap. Returns Swapped or Kept.
    ReloadStatus Reload();

    // Switch the active material to a NEW path and reload it (re-baselines the watcher). Returns
    // Swapped on success (active SPIR-V replaced) or Kept on failure (previous material retained).
    ReloadStatus SwitchTo(std::string path);

    // Poll the watcher; if the active .mat.json changed, reload it (swap-or-keep). Returns NoChange
    // when nothing changed, else Swapped/Kept. This is the live-loop entry point.
    ReloadStatus Poll();

    // The currently active compiled SPIR-V (empty until a successful load). The render loop reads
    // this to (re)build the material pipeline.
    const std::vector<uint32_t>& activeSpirv() const { return activeSpirv_; }

    // True exactly once after a successful swap, until cleared by ClearChanged(). Lets the render
    // loop rebuild its pipeline only when the SPIR-V actually changed.
    bool changed() const { return changed_; }
    void ClearChanged() { changed_ = false; }

    // The last error recorded by a failed reload/compile (empty after a success). Surfaced to the
    // live UI / logged so a broken edit is visible without crashing.
    const std::string& lastError() const { return lastError_; }

    const std::string& path() const { return path_; }
    // Number of successful swaps so far (for the dry-run / tests to assert a swap happened).
    int swapCount() const { return swapCount_; }

private:
    // The shared reload core: load `path_` -> validate -> compile -> swap-or-keep. Records lastError_
    // and bumps swapCount_/changed_ on success. Never throws.
    ReloadStatus DoReload();

    std::string           path_;
    CompileFn             compile_;
    runtime::FileWatcher  watcher_;
    std::vector<uint32_t> activeSpirv_;
    std::string           lastError_;
    bool                  changed_ = false;
    int                   swapCount_ = 0;
    bool                  watching_ = false;
};

}  // namespace hf::material
