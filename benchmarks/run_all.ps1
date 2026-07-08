<#
.SYNOPSIS
    Build + run the Hazard Forge moat-proof binary on Windows, print a summary table.

.DESCRIPTION
    Compiles benchmarks/moat_proofs.cpp standalone against the header-only pure cores
    (-I engine -I tests -I third_party — no CMake, no device, no RHI), runs it, and prints a
    per-proof PASS/FAIL table parsed from the binary's output. The binary exits 0 IFF all six
    proofs pass; this script propagates that exit code.

    Uses clang++ (C:\Program Files\LLVM\bin\clang++.exe) by default — a pure header build needs no
    MSVC dev shell. Set $env:HF_CXX to override the compiler. The digests are integer-core
    identical under MSVC and clang, so either toolchain reproduces the pinned goldens.

.NOTES
    Run from anywhere; the script locates the repo root as its own parent's parent.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $repo

$cxx = if ($env:HF_CXX) { $env:HF_CXX } else { 'C:\Program Files\LLVM\bin\clang++.exe' }
if (-not (Test-Path $cxx)) { Write-Error "compiler not found: $cxx (set `$env:HF_CXX)"; exit 2 }

$out = Join-Path $env:TEMP 'hf_moat_proofs.exe'

Write-Host '=== Hazard Forge benchmarks - building moat_proofs ===' -ForegroundColor Cyan
Write-Host "  compiler: $cxx"
& $cxx -std=c++20 -O2 -I engine -I tests -I third_party benchmarks/moat_proofs.cpp -o $out
if ($LASTEXITCODE -ne 0) { Write-Error 'BUILD FAILED'; exit 2 }

Write-Host '=== running ===' -ForegroundColor Cyan
$lines = & $out
$rc = $LASTEXITCODE
$lines | ForEach-Object { Write-Host $_ }

Write-Host ''
Write-Host '=== summary ===' -ForegroundColor Cyan
$fmt = '{0,-42} {1}'
Write-Host ($fmt -f 'PROOF', 'RESULT')
Write-Host ('-' * 52)
foreach ($l in $lines) {
    if ($l -match '^PROOF \d+ \[([^\]]+)\]:\s+(PASS|FAIL)') {
        $name = $Matches[1]; $res = $Matches[2]
        $color = if ($res -eq 'PASS') { 'Green' } else { 'Red' }
        Write-Host ($fmt -f $name, $res) -ForegroundColor $color
    }
}
Write-Host ('-' * 52)
if ($rc -eq 0) { Write-Host 'RESULT: ALL 6 PROOFS PASS (exit 0)' -ForegroundColor Green }
else           { Write-Host "RESULT: FAILURE (exit $rc)" -ForegroundColor Red }

Remove-Item $out -ErrorAction SilentlyContinue
exit $rc
