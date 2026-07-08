#!/usr/bin/env bash
# benchmarks/cross_platform_hash.sh — the "same seed+inputs -> same final hash on Win/Mac/Linux" check.
#
# THE CLAIM (docs/BEAT_UE5_PLAN.md Phase 3, proof #2 "the replay test"): a recorded match re-simulated on
# a different OS/toolchain reaches the IDENTICAL final-world hash. This script runs ONE leg of that check —
# it builds + runs the moat-proof binary on THIS machine and prints PROOF 1's final-world DigestSnapshot,
# the number a human/CI compares against the SAME number printed on the other machines.
#
# Each leg is independently reproducible; the FULL three-platform demonstration needs three machines
# (Windows/MSVC, macOS/Apple-clang, Linux/gcc). The determinism substrate is designed so all three print
# the identical hash (see docs/DETERMINISM_THREE_PLATFORMS.md — 135/135 pure-core digests already match on
# Linux/gcc); this script makes running your own leg one command.
#
# Usage:
#   ./benchmarks/cross_platform_hash.sh              # native leg (this machine's g++ / clang++)
#   ./benchmarks/cross_platform_hash.sh --docker     # the Linux/gcc leg in a container (no local toolchain)
#
# The Linux/gcc container leg reuses the Docker approach from scripts/linux_determinism_check.sh.

set -u
[ -d engine ] && [ -d benchmarks ] || cd "$(dirname "$0")/.." || exit 2

# --- the Linux/gcc container leg (no local toolchain needed) -------------------------------------------
if [ "${1:-}" = "--docker" ]; then
  echo "=== cross-platform hash: Linux/gcc container leg ==="
  exec docker run --rm -v "$PWD:/repo:ro" gcc:14 bash -c \
    'cd /repo && g++ -std=c++20 -O2 -I engine -I tests -I third_party benchmarks/moat_proofs.cpp -o /tmp/mp && /tmp/mp | grep "PROOF 1"'
fi

# --- the native leg (this machine's compiler) ---------------------------------------------------------
CXX="${CXX:-c++}"
OUT="$(mktemp -d)/hf_moat_proofs"
echo "=== cross-platform hash: native leg ($CXX, $(uname -s 2>/dev/null || echo unknown)) ==="
if ! "$CXX" -std=c++20 -O2 -I engine -I tests -I third_party benchmarks/moat_proofs.cpp -o "$OUT"; then
  echo "BUILD FAILED" >&2; exit 2
fi

# Extract PROOF 1's final-world digest (the cross-platform hash to compare across machines).
line="$("$OUT" | grep 'PROOF 1')"
hash="$(printf '%s\n' "$line" | sed -n 's/.*digest=\(0x[0-9a-f]*\).*/\1/p')"
rm -f "$OUT"

echo "$line"
echo
echo "  CROSS-PLATFORM FINAL-WORLD HASH: ${hash:-<parse-failed>}"
echo "  -> run this on Windows/MSVC, macOS/Apple-clang, and Linux/gcc; the hash must be IDENTICAL on all three."
echo "     (pinned golden: 0x76a37a56d256c401 — the canonical 24-tick replay)"
