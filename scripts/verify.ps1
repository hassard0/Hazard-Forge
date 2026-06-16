<#
.SYNOPSIS
    Full cross-platform verification for Hazard Forge: Windows/Vulkan + Mac/Metal in one command.

.DESCRIPTION
    Runs the complete verification gate and exits non-zero on ANY failure:

      1. Windows / Vulkan
         - conan install (cppstd=17 + Ninja generator), cmake configure, build, ctest.
         - Plus the Slice-AL introspection JSON golden: an EXACT byte match of the live
           --introspect output for the default scene vs tests/golden/introspect/default_scene.json.
           Backend-agnostic (pure hf_core) so it is verified ONLY here -- the Mac is not needed.
         - All steps run inside a VS BuildTools x64 dev shell so cl/ninja resolve.

      2. Mac / Metal (headless, over SSH on the LAN)
         - tar the repo (excluding build dirs + .git + stray PNGs, KEEPING the tracked goldens),
           scp it to the Mac, extract, configure+build the metal_headless target ONCE, then for
           EACH committed Metal golden run visual_test with its showcase flag and compare
           the output to the matching golden with threshold 0.0 (every pair must be DIFF 0.0000).
           A per-golden table is printed; the Mac portion passes only if ALL 41 diff 0.0000.

    Idempotent and re-runnable: build dirs are reused; the Mac staging dir is recreated each run.

.PREREQUISITES
      Windows:
        - Visual Studio 2022 BuildTools with the C++ x64 toolset at:
            C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1
        - conan 2.x and cmake 3.25+ on PATH.
        - A conan profile that can build the deps (cppstd is overridden to 17 by this script).
      Mac (one-time setup, already done on the bench Mac):
        - Passwordless SSH as ianhassard@192.168.4.215 using ~/.ssh/id_ed25519 (LAN-only).
        - ~/mac-remote-rig/{env.sh,compare.sh} present (Metal/MoltenVK env + PNG diff tool).
        - Xcode Command Line Tools + Homebrew glslc/spirv-cross/cmake/ninja/python3.

.NOTES
    Run from anywhere; the script locates the repo root as its own parent's parent.
    Does NOT re-bake any golden. Rendering behavior is never modified.
#>

[CmdletBinding()]
param(
    # Skip a platform if you only want to verify one side (both run by default).
    [switch]$SkipWindows,
    [switch]$SkipMac
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Config ----------------------------------------------------------------------------------------
$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VsDevShell = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1'

$MacUser    = 'ianhassard'
$MacHost    = '192.168.4.215'
$SshKey     = "$env:USERPROFILE\.ssh\id_ed25519"
$MacStage   = '~/hf-verify'                       # remote staging dir (recreated each run)
$TarName    = 'hf-verify.tar.gz'

# The committed Metal goldens, each produced by a distinct visual_test invocation. Name = the
# golden basename under tests/golden/metal/; Flag = the argv passed to visual_test BEFORE the output
# path (empty for the default Slice-F scene). The flags are the REAL ones parsed in
# metal_headless/visual_test.mm main() - confirmed there, not guessed. Every pair must diff 0.0000.
$Goldens = @(
    @{ Name = 'scene_shadow';  Flag = '' }                       # default visual_test <out>
    @{ Name = 'skinning';      Flag = '--skinning' }             # Slice O
    @{ Name = 'pbr_helmet';    Flag = '--pbr' }                  # Slice P
    @{ Name = 'mat_graph';     Flag = '--material' }             # Slice AV (data-driven material graph)
    @{ Name = 'mat_graph2';    Flag = '--material2' }            # Slice AW (second material; build-time codegen)
    @{ Name = 'mat_multi';     Flag = '--material-multi' }       # Slice AZ (three distinct graph materials in one frame)
    @{ Name = 'mat_normal';    Flag = '--material-normal' }      # Slice BE (NormalMap node: tangent-space normal map)
    @{ Name = 'instanced';     Flag = '--instanced' }            # Slice Q
    @{ Name = 'ibl_helmet';    Flag = '--ibl' }                  # Slice R
    @{ Name = 'physics';       Flag = '--physics' }              # Slice S
    @{ Name = 'transparency';  Flag = '--transparency' }         # Slice T
    @{ Name = 'bloom';         Flag = '--bloom' }                # Slice U
    @{ Name = 'scene_import';  Flag = '--scene' }                # Slice V
    @{ Name = 'debug_viz';     Flag = '--debug' }                # Slice W
    @{ Name = 'anim_blend';    Flag = '--blend' }                # Slice X
    @{ Name = 'anim_fsm';      Flag = '--anim-fsm' }             # Slice BL (animation state machine + cross-fade)
    @{ Name = 'ssao';          Flag = '--ssao' }                 # Slice Y
    @{ Name = 'capstone';      Flag = '--capstone' }             # Slice Z
    @{ Name = 'camera_pose';   Flag = '--camera 0.2,-0.1,0,3,10' } # Slice AA (scripted pose)
    @{ Name = 'gizmo';         Flag = '--gizmo 2' }              # Slice AB (select obj 2)
    @{ Name = 'csm';           Flag = '--csm' }                  # Slice AD (cascaded shadows)
    @{ Name = 'spot';          Flag = '--spot' }                 # Slice AE (spot-light shadows)
    @{ Name = 'point_shadow';  Flag = '--point-shadow' }         # Slice AF (omnidirectional point shadows)
    @{ Name = 'clustered';     Flag = '--clustered' }            # Slice AG (clustered / Forward+ lighting)
    @{ Name = 'ssr';           Flag = '--ssr' }                  # Slice AH (screen-space reflections)
    @{ Name = 'ssgi';          Flag = '--ssgi' }                 # Slice BP (screen-space global illumination)
    @{ Name = 'ssgi_denoise';  Flag = '--ssgi-denoise' }         # Slice BR (SSGI bilateral spatial denoise)
    @{ Name = 'ssgi_temporal'; Flag = '--ssgi-temporal' }        # Slice BV (temporal SSGI fixed-N accumulation)
    @{ Name = 'volumetric';    Flag = '--volumetric' }          # Slice AJ (volumetric fog / light shafts)
    @{ Name = 'probe';         Flag = '--probe' }                # Slice AK (reflection + irradiance probes)
    @{ Name = 'taa';           Flag = '--taa' }                  # Slice AP (temporal anti-aliasing)
    @{ Name = 'cull';          Flag = '--cull' }                 # Slice AQ (frustum-culling visualization)
    @{ Name = 'gpu_cull';      Flag = '--gpu-cull' }             # Slice AR (GPU-driven culling + indirect draw)
    @{ Name = 'mdi';           Flag = '--mdi' }                  # Slice BM (GPU multi-draw-indirect; Metal renders the identical scene per-object)
    @{ Name = 'mt';            Flag = '--mt' }                   # Slice AU (multithreaded recording; Metal N=4 parallel encoder)
    @{ Name = 'game';          Flag = '--game' }                # Slice AX (playable roll-a-ball game sample)
    @{ Name = 'net';           Flag = '--net' }                 # Slice BQ (replication; replica reconstructs + renders the scene)
    @{ Name = 'netsim';        Flag = '--netsim' }              # Slice BU (transport sim: lossy/laggy channel + client jitter-buffer interpolation)
    @{ Name = 'netpredict';    Flag = '--netpredict' }          # Slice BY (client prediction + server reconciliation: rewind+replay corrects a server-only misprediction)
    @{ Name = 'hud';           Flag = '--hud' }                 # Slice BA (text / HUD overlay)
    @{ Name = 'game_hud';      Flag = '--game-hud' }            # Slice BA (game scene + live SCORE HUD)
    @{ Name = 'stream';        Flag = '--stream' }             # Slice BD (scene/asset streaming; resident cell subset)
    @{ Name = 'terrain';       Flag = '--terrain' }            # Slice BF (procedural terrain / heightmap)
    @{ Name = 'terrain_stream'; Flag = '--terrain-stream' }    # Slice BJ (terrain streaming + per-tile LOD)
    @{ Name = 'decal';         Flag = '--decal' }             # Slice BH (screen-space projected decals)
    @{ Name = 'poststack';     Flag = '--poststack' }         # Slice BN (data-driven post-process stack)
    @{ Name = 'editor';        Flag = '--editor' }            # Slice BT (docked editor UI: Hierarchy/Inspector/Stats/Viewport)
    @{ Name = 'editor_edit';   Flag = '--editor-edit' }      # Slice BX (editor live-edit: edited scene + scene_io round-trip)
)

$winResult = 'SKIP'
$macResult = 'SKIP'
$script:macGoldenResults = @()   # per-golden [{ Name; Diff; Ok }] filled by Invoke-MacVerify

function Write-Section($t) { Write-Host ""; Write-Host "==== $t ====" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------------------------------
# Windows / Vulkan
# ---------------------------------------------------------------------------------------------------
function Invoke-WindowsVerify {
    Write-Section "WINDOWS / VULKAN"

    if (-not (Test-Path $VsDevShell)) {
        throw "VS BuildTools dev shell not found at: $VsDevShell"
    }

    # Build a single child-shell script: enter the VS dev shell, then run the full pipeline.
    # `$LASTEXITCODE` is checked after each native step so the first failure aborts with non-zero.
    # NOTE: do NOT set `$ErrorActionPreference='Stop' here. Launch-VsDevShell shells out to
    # vswhere and can emit a benign native-command warning to stderr; under 'Stop' that aborts
    # the shell. We gate every step on `$LASTEXITCODE explicitly instead, which is exact.
    $inner = @"
`$ErrorActionPreference = 'Continue'
& '$VsDevShell' -Arch amd64 2>`$null | Out-Null
Set-Location '$RepoRoot'

Write-Host '--- conan install ---'
conan install . -of build/windows-msvc-debug -s build_type=Debug -s compiler.cppstd=17 ``
    -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing
if (`$LASTEXITCODE -ne 0) { exit 11 }

# Conan appends each build's generated preset to CMakeUserPresets.json. If an ASan build was also
# configured (build/windows-msvc-asan), both conan presets are named 'conan-debug' and CMake
# refuses to read presets ('Duplicate preset'). Pin CMakeUserPresets.json to the debug include only
# so the verify run is deterministic regardless of what else has been built. NOTE: written WITHOUT a
# BOM -- conan's preset reader does json.loads() and chokes on a UTF-8 BOM.
`$pinned = @'
{
    "version": 4,
    "vendor": { "conan": {} },
    "include": [ "build/windows-msvc-debug/build/Debug/generators/CMakePresets.json" ]
}
'@
[System.IO.File]::WriteAllText((Join-Path (Get-Location) 'CMakeUserPresets.json'), `$pinned, (New-Object System.Text.UTF8Encoding(`$false)))

Write-Host '--- cmake configure ---'
cmake --preset windows-msvc-debug
if (`$LASTEXITCODE -ne 0) { exit 12 }

Write-Host '--- cmake build ---'
cmake --build --preset windows-msvc-debug
if (`$LASTEXITCODE -ne 0) { exit 13 }

Write-Host '--- ctest ---'
ctest --preset windows-msvc-debug
if (`$LASTEXITCODE -ne 0) { exit 14 }

# --- JSON introspection golden (Slice AL): an EXACT byte-for-byte match of the live --introspect
# output for the default scene against the committed text golden. This is the agent-OBSERVE artifact
# (editor::DescribeEngine). It is backend-AGNOSTIC (pure hf_core, no vk*/Metal symbols), so unlike the
# 26 IMAGE goldens it does NOT need the Mac: the bytes are identical on Vulkan and Metal. We therefore
# verify it once, here, on the Windows/Vulkan build. ---
Write-Host '--- introspection JSON golden ---'
`$introExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$introGolden = 'tests/golden/introspect/default_scene.json'
`$introLive = Join-Path `$env:TEMP 'hf_introspect_live.json'
& `$introExe --introspect `$introLive 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host 'introspect run failed'; exit 15 }
# Compare RAW bytes (the program writes LF-only, no BOM); any difference is a failure.
`$gBytes = [System.IO.File]::ReadAllBytes((Resolve-Path `$introGolden).Path)
`$lBytes = [System.IO.File]::ReadAllBytes(`$introLive)
`$introOk = (`$gBytes.Length -eq `$lBytes.Length)
if (`$introOk) { for (`$bi = 0; `$bi -lt `$gBytes.Length; `$bi++) { if (`$gBytes[`$bi] -ne `$lBytes[`$bi]) { `$introOk = `$false; break } } }
if (-not `$introOk) {
    Write-Host 'introspection JSON golden MISMATCH (tests/golden/introspect/default_scene.json)'
    exit 16
}
Write-Host 'introspection JSON golden: exact match'

# --- Audio mixer WAV golden (Slice BB): an EXACT byte-for-byte match of a fresh --audio-render of the
# fixed deterministic audio scene against the committed tests/golden/audio/scene.wav. The mixer is
# INTEGER / fixed-point end to end (Q15 gains, int32 accumulate, int16 hard-clamp), so the rendered
# WAV is bit-identical run-to-run AND across compilers (MSVC vs Apple clang) -- we verify it once here
# on the Windows build (the gate). Pure CPU (hf_core), no vk*/Metal symbols, like the JSON golden. ---
Write-Host '--- audio mixer WAV golden ---'
`$audExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$audGolden = 'tests/golden/audio/scene.wav'
`$audLive = Join-Path `$env:TEMP 'hf_audio_live.wav'
& `$audExe --audio-render `$audLive 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host 'audio-render run failed'; exit 24 }
`$agBytes = [System.IO.File]::ReadAllBytes((Resolve-Path `$audGolden).Path)
`$alBytes = [System.IO.File]::ReadAllBytes(`$audLive)
`$audOk = (`$agBytes.Length -eq `$alBytes.Length)
if (`$audOk) { for (`$ai = 0; `$ai -lt `$agBytes.Length; `$ai++) { if (`$agBytes[`$ai] -ne `$alBytes[`$ai]) { `$audOk = `$false; break } } }
if (-not `$audOk) {
    Write-Host 'audio WAV golden MISMATCH (tests/golden/audio/scene.wav)'
    exit 25
}
Write-Host 'audio WAV golden: exact match'

# --- Material-graph introspection JSON golden (Slice BI): an EXACT byte-for-byte match of a fresh
# --material-introspect dump of assets/materials/showcase3.mat.json against the committed
# tests/golden/material/showcase3_graph.json. DescribeGraphJson is pure CPU (hf_core, no vk*/Metal
# symbols) and deterministic by construction, so the text is bit-identical run-to-run AND across
# backends/compilers -- we verify it once here on the Windows build (the gate), like the JSON + WAV
# goldens. No Metal round-trip needed. ---
Write-Host '--- material-graph introspection JSON golden ---'
`$matExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$matGolden = 'tests/golden/material/showcase3_graph.json'
`$matSrc = 'assets/materials/showcase3.mat.json'
`$matLive = Join-Path `$env:TEMP 'hf_matintrospect_live.json'
& `$matExe --material-introspect `$matSrc `$matLive 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host 'material-introspect run failed'; exit 26 }
`$mgBytes = [System.IO.File]::ReadAllBytes((Resolve-Path `$matGolden).Path)
`$mlBytes = [System.IO.File]::ReadAllBytes(`$matLive)
`$matOk = (`$mgBytes.Length -eq `$mlBytes.Length)
if (`$matOk) { for (`$mi = 0; `$mi -lt `$mgBytes.Length; `$mi++) { if (`$mgBytes[`$mi] -ne `$mlBytes[`$mi]) { `$matOk = `$false; break } } }
if (-not `$matOk) {
    Write-Host 'material-graph introspection JSON golden MISMATCH (tests/golden/material/showcase3_graph.json)'
    exit 27
}
Write-Host 'material-graph introspection JSON golden: exact match'

# --- Vulkan validation gate (Slice AT): run representative showcases under the Khronos validation
# layer (synchronization + core validation) and FAIL on any real validation error. This is the
# permanent oracle that keeps the engine Vulkan-validation-CLEAN: Slice AS activated the layer and
# Slice AT fixed the two latent core-validation bugs (GPU-particle descriptor invalidation +
# swapchain semaphore reuse), so any regression that re-introduces a hazard surfaces here.
#
# The layer is provided by the conan 'vulkan-validationlayers' package (see conanfile.py); it is NOT
# installed system-wide on this box, so we must point VK_LAYER_PATH at the package's bin dir (the dir
# holding VkLayer_khronos_validation.json) or the layer loads as a no-op and the gate is blind. We
# locate it by globbing the conan2 cache for the layer manifest; if VK_LAYER_PATH is already set in
# the environment we honor that instead.
Write-Host '--- Vulkan validation gate ---'
`$vkExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$layerDir = `$env:VK_LAYER_PATH
if (-not `$layerDir -or -not (Test-Path (Join-Path `$layerDir 'VkLayer_khronos_validation.json'))) {
    `$manifest = Get-ChildItem -Path (Join-Path `$env:USERPROFILE '.conan2\p') -Recurse -Filter 'VkLayer_khronos_validation.json' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (`$manifest) { `$layerDir = `$manifest.Directory.FullName }
}
if (-not `$layerDir -or -not (Test-Path (Join-Path `$layerDir 'VkLayer_khronos_validation.json'))) {
    Write-Host 'Vulkan validation layer not found (conan vulkan-validationlayers missing?) - cannot run the validation gate'
    exit 17
}
Write-Host ('validation layer dir: ' + `$layerDir)
`$env:VK_LAYER_PATH = `$layerDir
`$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
`$env:VK_VALIDATION_FEATURE_ENABLE = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION'
# Representative showcases: --shot (GPU particles + shared-base/varied-normal materials, where the
# UPDATE_AFTER_BIND bug lived) and --csm-shot (cascaded shadow atlas, a heavy multi-pass graph path).
`$vkShots = @(@('--shot'), @('--csm-shot'), @('--mt-shot'), @('--mdi-shot'))
`$vkErrors = 0
foreach (`$shot in `$vkShots) {
    `$shotArgs = `$shot + @((Join-Path `$env:TEMP ('hf_validate_' + (`$shot[0] -replace '-','') + '.png')))
    `$vlog = & `$vkExe @shotArgs 2>&1
    # A REAL validation error is a 'VUID-<name>' token, a 'SYNC-HAZARD-*', an 'UNASSIGNED-*', or any
    # '[ERROR'-tagged line. The benign duplicate-limit notice mentions the bare word 'VUID' (no
    # hyphen) inside a [WARNING] line, so we match the hyphenated token form to avoid a false positive.
    `$bad = `$vlog | Select-String -Pattern 'VUID-|SYNC-HAZARD|UNASSIGNED-|\[ERROR'
    if (`$bad) {
        Write-Host ('validation FAIL on ' + (`$shot -join ' ') + ':')
        `$bad | ForEach-Object { Write-Host ('  ' + `$_) }
        `$vkErrors += `$bad.Count
    } else {
        Write-Host ('validation clean: ' + (`$shot -join ' '))
    }
}
if (`$vkErrors -ne 0) { Write-Host ('Vulkan validation gate FAILED (' + `$vkErrors + ' error line(s))'); exit 18 }
Write-Host 'Vulkan validation gate: CLEAN (zero VUID / SYNC-HAZARD / UNASSIGNED across showcases)'

# --- Multithreaded-recording determinism oracle (Slice AU): the SAME draw-heavy scene recorded with
# 1 worker vs N workers must be BYTE-IDENTICAL. Partition + in-order secondary execution guarantee
# the draw order is independent of worker count; this asserts it on the LIVE Vulkan render. Render
# with --workers 1 and --workers 4, then compare the captured BMPs byte-for-byte. (The unit test
# parallel_record_test pins the partition; this pins the end-to-end render.) ---
Write-Host '--- multithreaded-recording 1-vs-N determinism ---'
`$mtExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$mt1 = Join-Path `$env:TEMP 'hf_mt_w1.bmp'
`$mt4 = Join-Path `$env:TEMP 'hf_mt_w4.bmp'
& `$mtExe --mt-shot `$mt1 --workers 1 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host 'mt-shot --workers 1 failed'; exit 19 }
& `$mtExe --mt-shot `$mt4 --workers 4 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host 'mt-shot --workers 4 failed'; exit 19 }
`$mt1Bytes = [System.IO.File]::ReadAllBytes(`$mt1)
`$mt4Bytes = [System.IO.File]::ReadAllBytes(`$mt4)
`$mtOk = (`$mt1Bytes.Length -eq `$mt4Bytes.Length)
if (`$mtOk) { for (`$mi = 0; `$mi -lt `$mt1Bytes.Length; `$mi++) { if (`$mt1Bytes[`$mi] -ne `$mt4Bytes[`$mi]) { `$mtOk = `$false; break } } }
if (-not `$mtOk) { Write-Host 'multithreaded-recording MISMATCH (--workers 1 != --workers 4)'; exit 20 }
Write-Host 'multithreaded-recording: --workers 1 == --workers 4 (byte-identical render)'

# --- Live runtime material authoring (Slice AW): the runtime==build-time proof. --material-live-shot
# renders the showcase material via the RUNTIME path (in-process codegen -> dxc SUBPROCESS -> SPIR-V
# -> pipeline). Because that subprocess is the SAME dxc + SAME flags the build used for
# mat_showcase.frag.hlsl, the runtime SPIR-V is byte-identical to the build-time SPIR-V, so the live
# image MUST be byte-identical to --material-shot (the committed-HLSL build-time path). Assert the two
# captured BMPs are byte-for-byte equal. Also run the headless live A->B hot-swap dry-run + assert it
# passes (the swap happened, deterministic, no crash). Both run under the validation gate above. ---
Write-Host '--- live material authoring: runtime == build-time ---'
`$lmExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$lmBuild = Join-Path `$env:TEMP 'hf_material_build.bmp'
`$lmLive  = Join-Path `$env:TEMP 'hf_material_live.bmp'
& `$lmExe --material-shot `$lmBuild 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host '--material-shot failed'; exit 21 }
& `$lmExe --material-live-shot `$lmLive 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host '--material-live-shot failed'; exit 21 }
`$bBytes = [System.IO.File]::ReadAllBytes(`$lmBuild)
`$vBytes = [System.IO.File]::ReadAllBytes(`$lmLive)
`$lmOk = (`$bBytes.Length -eq `$vBytes.Length)
if (`$lmOk) { for (`$li = 0; `$li -lt `$bBytes.Length; `$li++) { if (`$bBytes[`$li] -ne `$vBytes[`$li]) { `$lmOk = `$false; break } } }
if (-not `$lmOk) { Write-Host 'runtime != build-time: --material-live-shot != --material-shot (SPIR-V drift?)'; exit 22 }
Write-Host 'live material authoring: --material-live-shot == --material-shot (byte-identical; runtime==build-time)'

Write-Host '--- live material authoring: hot-swap dry-run ---'
`$dry = & `$lmExe --material-hotswap-dry-run 2>&1
`$dry | ForEach-Object { Write-Host `$_ }
`$dryOk = (`$LASTEXITCODE -eq 0) -and (`$dry | Select-String -SimpleMatch 'hotswap-dry-run: PASS')
if (-not `$dryOk) { Write-Host 'hot-swap dry-run FAILED'; exit 23 }
Write-Host 'live material authoring: hot-swap dry-run PASS'

exit 0
"@

    # Run the inner pipeline in a fresh powershell so the dev-shell env is isolated per run.
    # Invoke via a temp -File (not -Command) and merge the child's stderr into stdout so a benign
    # native warning (e.g. vswhere) cannot bubble up as a terminating ErrorRecord under the outer
    # $ErrorActionPreference='Stop'. We rely solely on the child's exit code for pass/fail.
    $innerFile = Join-Path $env:TEMP 'hf-verify-win.ps1'
    Set-Content -LiteralPath $innerFile -Value $inner -Encoding UTF8
    $logFile = Join-Path $env:TEMP 'hf-verify-win.log'

    # Locally relax error handling and route the child's stderr to a file so a benign native
    # warning cannot terminate the outer 'Stop' scope. We trust only the child exit code.
    $savedEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $innerFile > $logFile 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $savedEAP

    if (Test-Path $logFile) { Get-Content $logFile | ForEach-Object { Write-Host $_ } }
    if ($code -ne 0) {
        $script:winResult = 'FAIL'
        Write-Host "Windows verification FAILED (stage exit code $code)" -ForegroundColor Red
        return
    }
    $script:winResult = 'PASS'
    Write-Host "Windows verification PASSED" -ForegroundColor Green
}

# ---------------------------------------------------------------------------------------------------
# Mac / Metal
# ---------------------------------------------------------------------------------------------------
function Invoke-MacVerify {
    Write-Section "MAC / METAL"

    if (-not (Test-Path $SshKey)) { throw "SSH key not found at: $SshKey" }

    $ssh = @('ssh', '-i', $SshKey, '-o', 'StrictHostKeyChecking=accept-new', "$MacUser@$MacHost")
    $scp = @('scp', '-i', $SshKey, '-o', 'StrictHostKeyChecking=accept-new', '-q')

    # 1) tar the repo. Keep the tracked golden; drop build dirs, .git, conan user presets, and the
    #    stray top-level bring-up PNGs (NOT the golden under tests/golden/). Uses git-bash tar via WSL?
    #    No - we use the bsdtar shipped with Windows 10/11 (tar.exe on PATH).
    $tarPath = Join-Path $env:TEMP $TarName
    if (Test-Path $tarPath) { Remove-Item $tarPath -Force }

    Write-Host "--- tar repo (keeping golden) ---"
    Push-Location $RepoRoot
    try {
        # bsdtar: exclude build artifacts + .git + known stray PNGs. The golden under
        # tests/golden/ is intentionally NOT matched by these patterns.
        & tar `
            --exclude='./build' `
            --exclude='./build-metal' `
            --exclude='./.git' `
            --exclude='./CMakeUserPresets.json' `
            --exclude='./hf_editor.png' `
            --exclude='./hf_nmap_crop.png' `
            --exclude='./hf_nmap_cube.png' `
            --exclude='./hf_nmap_cubeface.png' `
            --exclude='./hf_nmap_metal_cube.png' `
            -czf $tarPath .
        if ($LASTEXITCODE -ne 0) { throw "tar failed" }
    } finally { Pop-Location }

    # Sanity: EVERY golden MUST be in the archive or its compare on the Mac can't run.
    $archiveList = (& tar -tzf $tarPath)
    foreach ($g in $Goldens) {
        $needle = "golden/metal/$($g.Name).png"
        if (-not ($archiveList | Select-String -SimpleMatch $needle)) {
            throw "golden '$($g.Name).png' missing from archive - refusing to continue"
        }
    }

    # 2) recreate the remote staging dir + copy + extract (idempotent).
    Write-Host "--- scp + extract on Mac ---"
    & $ssh[0] $ssh[1..($ssh.Count-1)] "rm -rf $MacStage && mkdir -p $MacStage"
    if ($LASTEXITCODE -ne 0) { throw "remote mkdir failed" }
    & $scp[0] $scp[1..($scp.Count-1)] $tarPath "${MacUser}@${MacHost}:$MacStage/"
    if ($LASTEXITCODE -ne 0) { throw "scp failed" }

    # 3) extract + build ONCE + loop ALL goldens. To avoid the login shell being zsh and to dodge
    #    PowerShell here-string backtick-escaping fragility, the per-golden loop is generated as a
    #    standalone bash script, scp'd to the Mac, and run with an explicit `bash`. For each
    #    (flag -> golden) pair it renders visual_test <flag> /tmp/hf_<name>.png and compares to
    #    tests/golden/metal/<name>.png at threshold 0.0, emitting a machine-parseable
    #    "RESULT <name> <diff> <PASS|FAIL>" line we parse on the Windows side. The loop does NOT
    #    abort on a single failure (no `set -e` inside it): a drifted golden must still yield the full
    #    table so a reviewer sees exactly which one(s) changed. Build failures abort hard.
    #    compare.sh prints "DIFF <value>" and exits 0 only when DIFF <= threshold (0.0 here).
    #
    #    The PAIRS data is "<name>|<flag>" lines (flag empty for the default Slice-F scene). LF-only
    #    line endings are required so bash reads them cleanly.
    $pairLines = ($Goldens | ForEach-Object { "$($_.Name)|$($_.Flag)" }) -join "`n"

    # NOTE: this bash body is a single-quoted PS here-string, so NOTHING in it is expanded/escaped by
    # PowerShell. The PAIRS block is injected by string-replacing the @@PAIRS@@ token afterwards.
    $bashBody = @'
#!/usr/bin/env bash
set -e
source ~/mac-remote-rig/env.sh
# Run from the staging dir (this script was scp'd into it).
cd "$(cd "$(dirname "$0")" && pwd)"
tar -xzf "$TARBALL"
cmake -S metal_headless -B build-metal -G Ninja >/dev/null
cmake --build build-metal >/dev/null
set +e
read -r -d '' PAIRS <<'PAIRS_EOF'
@@PAIRS@@
PAIRS_EOF
while IFS='|' read -r name flag; do
    [ -z "$name" ] && continue
    out="/tmp/hf_${name}.png"
    golden="tests/golden/metal/${name}.png"
    if [ -z "$flag" ]; then
        ./build-metal/visual_test "$out" >/dev/null 2>&1
    else
        # flag may be multi-token (e.g. "--camera 0.2,-0.1,0,3,10") -> intentional word-split.
        ./build-metal/visual_test $flag "$out" >/dev/null 2>&1
    fi
    if [ $? -ne 0 ]; then
        echo "RESULT $name RENDER_FAIL FAIL"
        continue
    fi
    diffline=$(~/mac-remote-rig/compare.sh "$golden" "$out" 0.0 2>&1)
    crc=$?
    # compare.sh prints e.g. "DIFF 0.0000 (threshold 0.0)" -> keep ONLY the numeric value (field 2).
    diffval=$(echo "$diffline" | sed -n 's/^DIFF \([^ ]*\).*/\1/p' | head -1)
    [ -z "$diffval" ] && diffval=NO-DIFF
    if [ $crc -eq 0 ]; then
        echo "RESULT $name $diffval PASS"
    else
        echo "RESULT $name $diffval FAIL"
    fi
done <<< "$PAIRS"
'@

    $bashScript = $bashBody.Replace('@@PAIRS@@', $pairLines)
    # Write LF-only, no BOM (bash chokes on CRLF and a UTF-8 BOM).
    $bashScript = $bashScript -replace "`r`n", "`n"
    $localSh = Join-Path $env:TEMP 'hf-verify-mac.sh'
    [System.IO.File]::WriteAllText($localSh, $bashScript, (New-Object System.Text.UTF8Encoding($false)))

    & $scp[0] $scp[1..($scp.Count-1)] $localSh "${MacUser}@${MacHost}:$MacStage/run-goldens.sh"
    if ($LASTEXITCODE -ne 0) { throw "scp of golden-runner script failed" }

    $out = & $ssh[0] $ssh[1..($ssh.Count-1)] "STAGE='$MacStage' TARBALL='$TarName' bash $MacStage/run-goldens.sh" 2>&1
    $code = $LASTEXITCODE
    $out | ForEach-Object { Write-Host $_ }

    # Parse the RESULT lines into the per-golden table.
    $parsed = @{}
    foreach ($line in $out) {
        $s = [string]$line
        if ($s -match '^RESULT\s+(\S+)\s+(\S+)\s+(PASS|FAIL)\s*$') {
            $parsed[$matches[1]] = @{ Diff = $matches[2]; Ok = ($matches[3] -eq 'PASS') }
        }
    }

    $results = @()
    $allOk = $true
    foreach ($g in $Goldens) {
        if ($parsed.ContainsKey($g.Name)) {
            $r = $parsed[$g.Name]
            $results += @{ Name = $g.Name; Diff = $r.Diff; Ok = $r.Ok }
            if (-not $r.Ok) { $allOk = $false }
        } else {
            # No RESULT line emitted for this golden - treat as a failure (build/loop aborted early).
            $results += @{ Name = $g.Name; Diff = 'NO-RESULT'; Ok = $false }
            $allOk = $false
        }
    }
    $script:macGoldenResults = $results

    if ($code -ne 0 -or -not $allOk) {
        $script:macResult = 'FAIL'
        $bad = ($results | Where-Object { -not $_.Ok } | ForEach-Object { $_.Name }) -join ', '
        Write-Host "Mac verification FAILED (remote exit $code; non-0.0000 goldens: $bad)" -ForegroundColor Red
        return
    }
    $script:macResult = 'PASS'
    Write-Host ("Mac verification PASSED (all {0} goldens DIFF 0.0000)" -f $Goldens.Count) -ForegroundColor Green
}

# ---------------------------------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------------------------------
$failed = $false

if (-not $SkipWindows) {
    try { Invoke-WindowsVerify } catch { $script:winResult = 'FAIL'; Write-Host "Windows verify error: $_" -ForegroundColor Red }
}
if (-not $SkipMac) {
    try { Invoke-MacVerify } catch { $script:macResult = 'FAIL'; Write-Host "Mac verify error: $_" -ForegroundColor Red }
}

Write-Section "SUMMARY"
function Show($label, $r) {
    $color = if ($r -eq 'PASS') { 'Green' } elseif ($r -eq 'SKIP') { 'Yellow' } else { 'Red' }
    Write-Host ("  {0,-22} {1}" -f $label, $r) -ForegroundColor $color
}
Show 'Windows / Vulkan (ctest)' $winResult
Show ("Mac / Metal ({0} goldens)" -f $Goldens.Count) $macResult

# Per-golden Metal table (only when the Mac portion ran).
if ($script:macGoldenResults -and $script:macGoldenResults.Count -gt 0) {
    Write-Host ""
    Write-Host "  Metal goldens (threshold 0.0 - every diff must be 0.0000):" -ForegroundColor Cyan
    Write-Host ("    {0,-16} {1,-12} {2}" -f 'golden', 'DIFF', 'result')
    Write-Host ("    {0,-16} {1,-12} {2}" -f '------', '----', '------')
    $passCount = 0
    foreach ($r in $script:macGoldenResults) {
        $tag = if ($r.Ok) { 'PASS' } else { 'FAIL' }
        if ($r.Ok) { $passCount++ }
        $color = if ($r.Ok) { 'Green' } else { 'Red' }
        Write-Host ("    {0,-16} {1,-12} {2}" -f $r.Name, $r.Diff, $tag) -ForegroundColor $color
    }
    Write-Host ("    {0} / {1} goldens at DIFF 0.0000" -f $passCount, $script:macGoldenResults.Count)
}

if ($winResult -eq 'FAIL' -or $macResult -eq 'FAIL') {
    Write-Host ""
    Write-Host "VERIFY: FAIL" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "VERIFY: PASS" -ForegroundColor Green
exit 0
