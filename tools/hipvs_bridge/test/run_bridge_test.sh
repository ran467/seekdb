#!/usr/bin/env bash
#
# Offline test for the hipVS/cuVS bridge.
#
# It compiles tools/hipvs_bridge/seekdb_cuvs_bridge.c against the small cuVS
# stand-in in ./stubs and exercises the bridge plumbing (build/search/serialize/
# deserialize plus the OB_CUVS_* parameter overrides). No GPU, no hipVS install
# and no network are required, so it can run on any developer machine.
#
# Usage:
#   tools/hipvs_bridge/test/run_bridge_test.sh [output_binary]
#
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BRIDGE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
OUT=${1:-"$(mktemp -d)/bridge_test"}
CC=${CC:-cc}

mkdir -p "$(dirname "${OUT}")"

"${CC}" \
  -std=c11 \
  -O2 \
  -Wall \
  -I"${SCRIPT_DIR}/stubs" \
  -I"${BRIDGE_DIR}" \
  "${SCRIPT_DIR}/test_bridge.c" \
  "${SCRIPT_DIR}/stubs/cuvs_stub.c" \
  "${BRIDGE_DIR}/seekdb_cuvs_bridge.c" \
  -o "${OUT}" \
  -lm

"${OUT}"
