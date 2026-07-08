#!/usr/bin/env bash
# benchmarks/run_all.sh — build + run the moat-proof binary (POSIX / Linux / macOS).
#
# Compiles benchmarks/moat_proofs.cpp standalone against the header-only pure cores (the same
# -I engine -I tests discipline scripts/linux_determinism_check.sh uses — no CMake, no device,
# no RHI) and runs it. The binary prints one PROOF line per claim and exits 0 IFF all six pass.
#
#   ./benchmarks/run_all.sh            # native (g++ or clang++)
#   CXX=clang++ ./benchmarks/run_all.sh
#
# Exit 0 iff every proof passes (the binary's own exit code is propagated).

set -u
# Run from the repo root: use CWD if it looks like the repo, else derive from this script's path.
[ -d engine ] && [ -d benchmarks ] || cd "$(dirname "$0")/.." || exit 2

CXX="${CXX:-c++}"
OUT="$(mktemp -d)/hf_moat_proofs"

echo "=== Hazard Forge benchmarks — building moat_proofs ($CXX) ==="
if ! "$CXX" -std=c++20 -O2 -I engine -I tests -I third_party benchmarks/moat_proofs.cpp -o "$OUT"; then
  echo "BUILD FAILED" >&2
  exit 2
fi

echo "=== running ==="
"$OUT"
rc=$?

echo
echo "=== summary ==="
if [ "$rc" -eq 0 ]; then
  echo "  RESULT: ALL 6 PROOFS PASS (exit 0)"
else
  echo "  RESULT: FAILURE (exit $rc) — a pinned digest or assertion did not hold"
fi
rm -f "$OUT"
exit "$rc"
