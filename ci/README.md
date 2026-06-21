# CI

This directory holds the GitHub Actions workflow as a **staged** file:

```
ci/github-actions-ci.yml
```

It is intentionally NOT under `.github/workflows/`.

## Why it lives here (and not under .github/workflows/)

The credential currently used to push this repo **lacks the `workflow` OAuth scope**. GitHub
rejects any push that adds or modifies a file under `.github/workflows/` unless the token carries
that scope, so committing the workflow there would make `git push` fail outright. Keeping the YAML
under `ci/` lets the branch push succeed; the file is ready to be activated whenever a suitably
scoped token is available.

## How to enable CI

1. Obtain a token (PAT or GitHub App installation token) that includes the **`workflow`** scope.
   - Classic PAT: tick the `workflow` scope.
   - Fine-grained PAT / App: grant **Actions: read & write** + **Contents: read & write**.
2. Move the file into place and commit with that token:

   ```sh
   mkdir -p .github/workflows
   git mv ci/github-actions-ci.yml .github/workflows/ci.yml
   git commit -m "ci: enable GitHub Actions workflow"
   git push    # now succeeds because the token has the workflow scope
   ```

3. Push. The `windows-vulkan` and `windows-asan` jobs run on GitHub-hosted `windows-2022` runners.

## Jobs

| Job              | Runner                       | What it does                                                              |
| ---------------- | ---------------------------- | ------------------------------------------------------------------------- |
| `windows-vulkan` | `windows-2022` (hosted)      | conan install (cppstd=17 + Ninja; pulls the Khronos validation layer) -> configure (runs the `material_codegen` build-time tool) -> build -> ctest (105 tests) + the introspection JSON-golden byte match + the material-graph introspection JSON-golden byte match + the audio WAV-golden byte match |
| `windows-asan`   | `windows-2022` (hosted)      | `HF_SANITIZE=address` build of the pure-C++ core + tests -> ctest under ASan (105 tests; the pure ones instrumented, `rhi_smoke` not) |
| `macos-metal`    | `[self-hosted, macos, metal]`| headless Metal build -> render + golden-compare **all 221 goldens** (DIFF 0.0) |

The Windows build runs the **`material_codegen`** build-time tool (it bakes the showcase `*.mat.json`
material graphs into the committed generated HLSL the offline shader pipeline compiles) and links
against the Conan-provided **Khronos validation layer**, which the Vulkan-validation gate loads at
runtime (core + synchronization validation; every showcase must be `VUID`/`SYNC-HAZARD`/`[ERROR]`-free).

The `windows-vulkan` job also byte-matches three **non-image** goldens (all backend-agnostic pure
`hf_core`, so verified only on the Windows side): the engine-state JSON
(`tests/golden/introspect/default_scene.json` vs live `--introspect`), the **material-graph
introspection JSON** (`tests/golden/material/showcase3_graph.json` vs a fresh
`--material-introspect assets/materials/showcase3.mat.json`), and the **audio WAV**
(`tests/golden/audio/scene.wav` vs a fresh `--audio-render`). The audio path is integer/fixed-point
end to end and the introspection dumps are deterministic string builders, so the bytes are identical
across compilers and runs; the checks fail on any drift.

### The Metal job

GitHub-hosted macOS runners cannot reach the LAN bench Mac and do not provide a Metal GPU whose
offscreen output matches the baked goldens, so `macos-metal` is gated to a **self-hosted** runner
labelled `metal` (set one up on the bench Mac and register it with the labels `self-hosted, macos,
metal`). On hosted infrastructure the job is skipped.

The job builds `metal_headless` **once**, then for each of the **221** committed goldens runs
`visual_test <flag> /tmp/hf_<name>.png` and compares it to `tests/golden/metal/<name>.png` at
threshold `0.0`. Every pair must report `DIFF 0.0000`; the job fails if any golden drifts.
`scripts/verify.ps1`'s per-golden table is the authoritative, always-current index of all 221
(golden -> flag) pairs; a representative slice of the rendering subset:

| golden              | flag                          | golden              | flag                  |
| ------------------- | ----------------------------- | ------------------- | --------------------- |
| `scene_shadow`      | *(default)*                   | `gizmo`             | `--gizmo 2`           |
| `skinning`          | `--skinning`                  | `csm`               | `--csm`               |
| `pbr_helmet`        | `--pbr`                       | `spot`              | `--spot`              |
| `instanced`         | `--instanced`                 | `point_shadow`      | `--point-shadow`      |
| `ibl_helmet`        | `--ibl`                       | `clustered`         | `--clustered`         |
| `physics`           | `--physics`                   | `ssr`               | `--ssr`               |
| `transparency`      | `--transparency`              | `ssgi`              | `--ssgi`              |
| `bloom`             | `--bloom`                     | `ssgi_denoise`      | `--ssgi-denoise`      |
| `anim_blend`        | `--blend`                     | `scene_import`      | `--scene`             |
| `capstone`          | `--capstone`                  | `debug_viz`         | `--debug`             |
| `volumetric`        | `--volumetric`                | `ssao`              | `--ssao`              |
| `taa`               | `--taa`                       | `camera_pose`       | `--camera 0.2,-0.1,0,3,10` |
| `gpu_cull`          | `--gpu-cull`                  | `probe`             | `--probe`             |
| `mat_graph`         | `--material`                  | `cull`              | `--cull`              |
| `mat_multi`         | `--material-multi`            | `mt`                | `--mt`                |
| `game`              | `--game`                      | `mat_graph2`        | `--material2`         |
| `hud`               | `--hud`                       | `mat_normal`        | `--material-normal`   |
| `terrain`           | `--terrain`                   | `net`               | `--net`               |
| `terrain_stream`    | `--terrain-stream`            | `game_hud`          | `--game-hud`          |
| `mdi`               | `--mdi`                       | `stream`            | `--stream`            |
| `anim_fsm`          | `--anim-fsm`                  | `decal`             | `--decal`             |
| `poststack`         | `--poststack`                 | `ssgi_temporal`     | `--ssgi-temporal`     |
| `netsim`            | `--netsim`                    | `editor`            | `--editor`            |
| `netpredict`        | `--netpredict`                | `bindless`          | `--bindless`          |
| `editor_edit`       | `--editor-edit`               | `gpudriven`         | `--gpudriven`         |
| `gpucull_draw`      | `--gpucull-draw`              | `vfx`               | `--vfx`               |
| `water`             | `--water`                     | `dof`               | `--dof`               |
| `clouds`            | `--clouds`                    | `cloud_shadows`     | `--cloud-shadows`     |
| `clustered_lights`  | `--clustered-lights`          | `hiz_cull`          | `--hiz-cull`          |
| `motion_blur`       | `--motionblur`                | `oit`               | `--oit`               |
| `pom`               | `--pom`                       | `gtao`              | `--gtao`              |
| `froxel_fog`        | `--froxelfog`                 | `contact_shadows`   | `--contactshadow`     |
| `froxel_lights`     | `--froxellights`              | `auto_exposure`     | `--autoexposure`      |
| `vol_shadows`       | `--volshadows`                | `sss`               | `--sss`               |
| `refl_probe`        | `--reflprobe`                 | `color_grade`       | `--colorgrade`        |
| `capture_probe`     | `--captureprobe`              | `planar_reflection` | `--planar`            |
| `cas`               | `--cas`                       | `probegi`           | `--probegi`           |
| `probe_capture`     | `--probecapture`              | `probe_sh`          | `--probesh`           |

For routine local verification of **both** platforms in one command, use:

```powershell
scripts\verify.ps1
```

That script runs the Windows/Vulkan ctest locally (plus the introspection JSON-golden and audio
WAV-golden byte matches) and drives the bench Mac over SSH to build the headless Metal target once
and run the **same 221-golden loop**, each compared at threshold `0.0` (every one must report
`DIFF 0.0000`). It prints a per-golden table and an overall `VERIFY: PASS/FAIL`.

## Local equivalents

| Pipeline           | Local command                                                            |
| ------------------ | ------------------------------------------------------------------------ |
| Windows build+test | `cmake --preset windows-msvc-debug` + build + `ctest --preset windows-msvc-debug` |
| ASan build+test    | `cmake --preset windows-msvc-asan` + build + `ctest --preset windows-msvc-asan`   |
| Full cross-platform| `scripts\verify.ps1`                                                      |
