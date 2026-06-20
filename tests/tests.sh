#!/usr/bin/env bash
# Run the MATH benchmark on Qwen/Qwen3.5-0.8B-Base and save results in the tests folder.
#
# The `math` benchmark expands to all 7 MATH subjects (algebra, geometry, number
# theory, ...) and reports the unweighted-mean exact-match accuracy.
#
# Usage:
#   bash tests/tests.sh
set -euo pipefail

# Directory of this script -> results live in tests/results/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

MODEL="Qwen/Qwen3-0.6B-Base"

mill --output_dir "${RESULTS_DIR}" eval \
    "${MODEL}" \
    mmlu_pro \
    --model_args "dtype=bfloat16"

# Show the aggregated table once the run finishes.
mill --output_dir "${RESULTS_DIR}" collect --tasks math

# mill --output_dir ./results schedule Qwen/Qwen3-0.6B-Base mmlu_pro --n_shots 0
